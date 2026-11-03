"""Hyperparameter sweeps over makemore_improve.py, using AdamW and warmup + cosine down
to 0.1x peak.

Each entry in `rounds` is one experiment, with its own grid and results file. To run a
new experiment, add a new entry and leave the old ones as the record of what was run.

    python makemore_experiment.py --round round2 --workers 4         # the full grid, 1 seed
    python makemore_experiment.py --round round2 --top 5 --seeds 43 44 45  # re-run the best 5

Finished runs are appended to the results file and skipped on the next launch, so an
interrupted sweep can just be restarted. With nothing left to run, it prints the report.
Configs are ranked on dev loss. The val split is never evaluated here: it's held out
for the final config.
"""
import argparse
import contextlib
import io
import itertools
import json
import statistics
import time
from collections import defaultdict
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn

from makemore_improve import (
    create_dataloaders,
    create_model,
    create_optimizer,
    evaluate_model,
    load_data,
    make_names,
    train_model,
)

here = Path(__file__).parent

# held fixed across rounds
batch = 64
n_emb = 24
epochs = 15
warmup_steps = 500
split_seed = 42


@dataclass(frozen=True)
class Config:
    peak_lr: float
    n_hidden: int
    weight_decay: float
    seed: int
    epochs: int = epochs
    dropout: float = 0.0
    activation: str = "tanh"


activations: dict[str, type[nn.Module]] = {"tanh": nn.Tanh, "gelu": nn.GELU}


@dataclass(frozen=True)
class Round:
    peak_lrs: list[float]
    hidden_sizes: list[int]
    dropouts: list[float]
    weight_decays: list[float]
    results: Path
    epochs: int = epochs
    activations: tuple[str, ...] = ("tanh",)
    n_names: int = 0  # names to generate from each trained model


rounds = {
    # 4 x 4 x 1 x 3 = 48 configs
    "round1": Round(
        peak_lrs=[5e-4, 1e-3, 2e-3, 3e-3],
        hidden_sizes=[128, 200, 300, 400],
        dropouts=[0.0],
        weight_decays=[0.0, 0.01, 0.05],
        results=here / "sweep_results.jsonl",
    ),
    # 2 x 3 x 3 x 2 = 36 configs
    "round2": Round(
        peak_lrs=[3e-4, 5e-4],
        hidden_sizes=[400, 600, 800],
        dropouts=[0.0, 0.1, 0.2],
        weight_decays=[0.05, 0.1],
        results=here / "sweep2_results.jsonl",
    ),
    # A single exploratory run: can GELU make the big network work at high dropout?
    "round2-1": Round(
        peak_lrs=[3e-4],
        hidden_sizes=[800],
        dropouts=[0.2],
        weight_decays=[0.1],
        results=here / "sweep2-1_results.jsonl",
        activations=("gelu",),
        n_names=20,
    ),
    # 3 x 1 x 3 x 1 = 9 configs: re-tune lr and dropout for GELU, which overfits harder than tanh
    "round3": Round(
        peak_lrs=[2e-4, 3e-4, 5e-4],
        hidden_sizes=[800],
        dropouts=[0.2, 0.3, 0.4],
        weight_decays=[0.1],
        results=here / "sweep3_results.jsonl",
        activations=("gelu",),
    ),
    # 2 x 1 x 2 x 1 = 4 configs: double the epochs, since high dropout looks short of training
    "round4": Round(
        peak_lrs=[5e-4, 8e-4],
        hidden_sizes=[800],
        dropouts=[0.3, 0.4],
        weight_decays=[0.1],
        results=here / "sweep4_results.jsonl",
        epochs=30,
        activations=("gelu",),
    ),
    # 2 configs: the round 4 winner with more dropout, to find where dropout stops helping
    "round5": Round(
        peak_lrs=[5e-4],
        hidden_sizes=[800],
        dropouts=[0.5, 0.6],
        weight_decays=[0.1],
        results=here / "sweep5_results.jsonl",
        epochs=30,
        activations=("gelu",),
    ),
    # 2 configs: round 5's curves were still falling, so double the epochs for 0.5 against
    # the round 4 winner's 0.4. 0.6 was too far behind to include.
    "round6": Round(
        peak_lrs=[5e-4],
        hidden_sizes=[800],
        dropouts=[0.4, 0.5],
        weight_decays=[0.1],
        results=here / "sweep6_results.jsonl",
        epochs=60,
        activations=("gelu",),
    ),
}


type Record = dict[str, Any]


def linear_weight_decay_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """Decay Linear weights only; LayerNorm, biases and embeddings get none."""
    decay = [m.weight for m in model.modules() if isinstance(m, nn.Linear)]
    decay_ids = {id(p) for p in decay}
    no_decay = [p for p in model.parameters() if id(p) not in decay_ids]
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def weights_filename(cfg: Config) -> str:
    return (
        f"lr{cfg.peak_lr:g}_h{cfg.n_hidden}_d{cfg.dropout:g}_wd{cfg.weight_decay:g}"
        f"_{cfg.activation}_e{cfg.epochs}_s{cfg.seed}.pt"
    )


def run_config(cfg: Config, threads: int, n_names: int = 0, weights_dir: Path | None = None) -> Record:
    """Train one config from scratch. Everything random derives from cfg.seed."""
    torch.set_num_threads(threads)
    names, stoi, itos = load_data(str(here / "names.txt"))
    # Rebuilt per run: the train loader's generator is stateful, so sharing it between
    # runs would make the shuffle order depend on which configs ran before this one.
    train_set, dev_set, _ = create_dataloaders(names, stoi, seed=split_seed, batch=batch)

    # cfg.seed varies only init and shuffle order. The split stays fixed, because changing
    # the dev set moves dev loss by far more than the differences this sweep is measuring.
    cast(torch.Generator, train_set.generator).manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    model = create_model(
        len(stoi),
        n_emb=n_emb,
        n_hidden=cfg.n_hidden,
        dropout_rate=cfg.dropout,
        activation=activations[cfg.activation],
    )
    opt, sched = create_optimizer(
        model,
        "Adam",
        "WarmupCosine",
        opt_kwargs={"lr": cfg.peak_lr},
        sched_args={
            "total_steps": cfg.epochs * len(train_set),
            "lr": cfg.peak_lr,
            "warmup_steps": warmup_steps,
        },
        param_groups=linear_weight_decay_groups(model, cfg.weight_decay),
    )

    # One epoch at a time, so dev loss can be recorded along the way. The curve's train
    # loss has dropout on, so it reads higher than the final train_loss, which has it off.
    curve: list[dict[str, float]] = []
    start = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        with contextlib.redirect_stdout(io.StringIO()):  # train_model prints the starting loss
            epoch_losses = train_model(model, train_set, opt, sched, epochs=1)
        # Iterating a DataLoader draws from the global RNG. Restoring it keeps the dropout
        # masks, and so the whole run, the same as training without these evaluations.
        rng_state = torch.get_rng_state()
        dev_loss = evaluate_model(model, [dev_set])[0]
        torch.set_rng_state(rng_state)
        curve.append({
            "epoch": epoch,
            "train_loss": statistics.mean(epoch_losses),
            "dev_loss": dev_loss,
            "lr": opt.param_groups[0]["lr"],
        })
    seconds = time.perf_counter() - start
    if weights_dir is not None:
        torch.save(model.state_dict(), weights_dir / weights_filename(cfg))

    train_loss, dev_loss = evaluate_model(model, [train_set, dev_set])
    return {
        "config": asdict(cfg),
        "train_loss": train_loss,
        "dev_loss": dev_loss,
        "seconds": seconds,
        "names": make_names(model, itos, n_names),
        "curve": curve,
    }


def load_records(path: Path) -> list[Record]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def group_by_config(records: list[Record]) -> dict[Config, list[Record]]:
    """Group runs of the same config across seeds (the seed in the key is always 0)."""
    groups: dict[Config, list[Record]] = defaultdict(list)
    for r in records:
        groups[replace(Config(**r["config"]), seed=0)].append(r)
    return groups


def mean_dev(rs: list[Record]) -> float:
    devs: list[float] = [r["dev_loss"] for r in rs]
    return statistics.mean(devs)


def rank_configs(records: list[Record]) -> list[Config]:
    groups = group_by_config(records)
    return sorted(groups, key=lambda c: mean_dev(groups[c]))


def planned_configs(args: argparse.Namespace, records: list[Record]) -> list[Config]:
    sweep = rounds[args.round]
    if args.top is not None:
        bases = rank_configs(records)[: args.top]
        if not bases:
            raise SystemExit(f"--top needs existing results in {args.results}")
    else:
        bases = [
            Config(lr, n_hidden, wd, seed=0, epochs=args.epochs or sweep.epochs, dropout=dropout, activation=activation)
            for lr, n_hidden, dropout, wd, activation in itertools.product(
                sweep.peak_lrs, sweep.hidden_sizes, sweep.dropouts, sweep.weight_decays, sweep.activations
            )
        ]
    return [replace(c, seed=s) for c in bases for s in args.seeds]


def print_report(records: list[Record], n: int = 10) -> None:
    if not records:
        return
    groups = group_by_config(records)
    print(f"\n{len(records)} runs of {len(groups)} configs. Best {min(n, len(groups))} by dev loss:")
    print(f"{'peak_lr':>8} {'hidden':>6} {'drop':>5} {'wd':>5} {'seeds':>5} {'dev':>7} {'dev_std':>7} {'train':>7} {'gap':>6}")
    for c in rank_configs(records)[:n]:
        rs = groups[c]
        dev = mean_dev(rs)
        std = statistics.stdev(r["dev_loss"] for r in rs) if len(rs) > 1 else float("nan")
        train = statistics.mean(r["train_loss"] for r in rs)
        print(f"{c.peak_lr:>8.0e} {c.n_hidden:>6} {c.dropout:>5} {c.weight_decay:>5} {len(rs):>5} {dev:>7.4f} {std:>7.4f} {train:>7.4f} {dev - train:>6.3f}")

    print("\nMean dev loss by value, averaged over everything else:")
    for field in ("peak_lr", "n_hidden", "dropout", "weight_decay"):
        by_value: dict[float, list[float]] = defaultdict(list)
        for c, rs in groups.items():  # per config, so re-run configs don't count extra
            by_value[getattr(c, field)].append(mean_dev(rs))
        values = "  ".join(f"{v:g}: {statistics.mean(ls):.4f}" for v, ls in sorted(by_value.items()))
        print(f"  {field:<13} {values}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--round", required=True, choices=rounds, help="which experiment to run")
    parser.add_argument("--workers", type=int, default=4, help="parallel training processes")
    parser.add_argument("--threads", type=int, default=1, help="torch threads per process")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="seeds to run each config with")
    parser.add_argument("--top", type=int, help="run only the N best configs so far, instead of the grid")
    parser.add_argument("--epochs", type=int, help="override the round's epochs")
    parser.add_argument("--results", type=Path, help="override the round's results file")
    parser.add_argument("--save-weights", action="store_true", help="save each trained model to weights/<round>/")
    args = parser.parse_args()
    args.results = args.results or rounds[args.round].results
    weights_dir = here / "weights" / args.round if args.save_weights else None
    if weights_dir is not None:
        weights_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.results)
    done = {Config(**r["config"]) for r in records}
    pending = [c for c in planned_configs(args, records) if c not in done]
    print(f"{len(pending)} runs to do, {len(done)} already in {args.results}")

    failures = 0
    start = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        n_names = rounds[args.round].n_names
        futures: dict[Future[Record], Config] = {
            pool.submit(run_config, c, args.threads, n_names, weights_dir): c for c in pending
        }
        for i, future in enumerate(as_completed(futures), 1):
            cfg = futures[future]
            try:
                record = future.result()
            except Exception as e:
                failures += 1
                print(f"[{i}/{len(pending)}] FAILED {cfg}: {e!r}", flush=True)
                continue
            with args.results.open("a") as f:
                f.write(json.dumps(record) + "\n")
            records.append(record)
            elapsed = (time.perf_counter() - start) / 60
            print(
                f"[{i}/{len(pending)}] lr={cfg.peak_lr:g} hidden={cfg.n_hidden} drop={cfg.dropout:g}"
                f" wd={cfg.weight_decay:g} seed={cfg.seed}"
                f"  dev={record['dev_loss']:.4f} train={record['train_loss']:.4f}"
                f"  {record['seconds']:.0f}s  ({elapsed:.1f}m elapsed)",
                flush=True,
            )
            if record["names"]:
                print("  " + ", ".join(record["names"]), flush=True)

    print_report(records)
    if failures:
        raise SystemExit(f"{failures} runs failed; restart to retry them")


if __name__ == "__main__":
    main()
