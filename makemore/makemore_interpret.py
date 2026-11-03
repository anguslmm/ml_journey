"""Look inside a trained makemore model.

    python makemore_interpret.py embeddings  # do similar characters get similar vectors?
    python makemore_interpret.py importance  # which context positions matter?
    python makemore_interpret.py neurons     # what does each neuron respond to?
    python makemore_interpret.py detectors   # how many neurons detect one letter, or one name chunk?
    python makemore_interpret.py ablation    # how much breaks when neurons are removed?
    python makemore_interpret.py steer       # silence or boost one neuron, compare names
    python makemore_interpret.py calibration # is the model over- or under-confident?
    python makemore_interpret.py probe       # which concepts can be read from each layer?
    python makemore_interpret.py all

Pick a model with --weights, and find charts and reports in plots/interpret/. Everything
here uses train and dev only. Val stays held out.
"""
import argparse
import re
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from makemore_experiment import activations, batch, here, n_emb, split_seed
from makemore_improve import block, create_dataloaders, create_model, load_data, make_names
from makemore_plots import blue, gridline, ink, muted_ink, surface

default_weights = here / "weights" / "round4" / "lr0.0005_h800_d0.4_wd0.1_gelu_e30_s42.pt"
out_dir = here / "plots" / "interpret"

orange = "#eb6834"
aqua = "#1baf7a"
series = [blue, orange, aqua]  # fixed order: layers 1, 2, 3
diverging_cmap = LinearSegmentedColormap.from_list("zscore", ["#104281", blue, "#f0efec", "#e34948", "#a32323"])

vowels = "aeiou"


class Lab:
    """A trained model, plus the train and dev windows to study it with."""

    def __init__(self, weights: Path) -> None:
        match = re.fullmatch(r"lr.+_h(\d+)_d[\d.]+_wd.+_(\w+)_e\d+_s\d+\.pt", weights.name)
        if match is None:
            raise SystemExit(f"Can't read the model settings from the filename: {weights.name}")
        self.names, self.stoi, self.itos = load_data(str(here / "names.txt"))
        train_set, dev_set, _ = create_dataloaders(self.names, self.stoi, seed=split_seed, batch=batch)
        self.Xtr, self.Ytr = cast(data.TensorDataset, train_set.dataset).tensors
        self.Xdev, self.Ydev = cast(data.TensorDataset, dev_set.dataset).tensors

        # Dropout has no weights and is off in eval mode, so its rate doesn't matter here.
        self.model = create_model(
            len(self.stoi),
            n_emb=n_emb,
            n_hidden=int(match[1]),
            activation=activations[match[2]],
        )
        self.model.load_state_dict(torch.load(weights))
        self.model.eval()
        self.n_hidden = int(match[1])
        kinds = tuple(activations.values())
        self.layers = [m for m in self.model if isinstance(m, kinds)]  # layers[0] sees 2 chars, then 4, then 8

    def text(self, chars: torch.Tensor) -> str:
        return "".join(self.itos[int(i)] for i in chars)

    @torch.no_grad()
    def logits(self, X: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.model(x) for x in X.split(4096)])

    def dev_loss(self) -> float:
        return F.cross_entropy(self.logits(self.Xdev), self.Ydev).item()

    @torch.no_grad()
    def activations(self, X: torch.Tensor, layer: int) -> torch.Tensor:
        """A layer's response to the last characters of each window: (windows, neurons)."""
        captured: list[torch.Tensor] = []

        def capture(_: nn.Module, __: object, output: torch.Tensor) -> None:
            captured.append(output[:, -1, :].clone())

        handle = self.layers[layer].register_forward_hook(capture)
        self.logits(X)
        handle.remove()
        return torch.cat(captured)


@contextmanager
def edited(lab: Lab, layer: int, edit: Callable[[torch.Tensor], torch.Tensor]) -> Iterator[None]:
    """Run the model with a layer's outputs changed."""

    def hook(_: nn.Module, __: object, output: torch.Tensor) -> torch.Tensor:
        return edit(output)

    handle = lab.layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def save(fig: Figure, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def style(ax: Axes) -> None:
    ax.grid(axis="y", color=gridline, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0)


def embeddings(lab: Lab) -> None:
    """Project the character vectors to 2D, and list each character's nearest neighbours."""
    vectors = cast(nn.Embedding, lab.model[0]).weight.detach()
    centered = vectors - vectors.mean(0)
    U, S, _ = torch.linalg.svd(centered, full_matrices=False)
    points = U[:, :2] * S[:2]
    shares = (S**2 / (S**2).sum()).tolist()

    fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
    groups = [("Vowels", blue, vowels), ("Consonants", orange, "bcdfghjklmnpqrstvwxyz"), ("End of name", muted_ink, ".")]
    for label, color, chars in groups:
        rows = [lab.stoi[c] for c in chars]
        ax.scatter(points[rows, 0], points[rows, 1], s=70, color=color, edgecolors=surface, linewidths=2, label=label)
    for i in range(len(lab.itos)):
        ax.annotate(lab.itos[i], points[i].tolist(), xytext=(7, 5), textcoords="offset points", color=ink, fontsize=11)
    ax.set_xlabel(f"Principal component 1 ({shares[0]:.0%} of variance)")
    ax.set_ylabel(f"Principal component 2 ({shares[1]:.0%} of variance)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0)
    ax.legend(frameon=False)
    ax.set_title("Character embeddings, projected to 2D")
    save(fig, "embeddings.png")

    similarity = F.cosine_similarity(vectors[:, None], vectors[None, :], dim=-1)
    print("\nNearest characters by cosine similarity:")
    for i in range(len(lab.itos)):
        nearest = similarity[i].topk(4).indices[1:].tolist()
        print(f"  {lab.itos[i]}: {' '.join(lab.itos[j] for j in nearest)}")


def importance(lab: Lab) -> None:
    """Measure how much each context position matters, in two independent ways."""
    X, Y = lab.Xdev, lab.Ydev
    base = F.cross_entropy(lab.logits(X), Y, reduction="none")
    occlusion: list[float] = []
    for position in range(block):
        covered = X.clone()
        covered[:, position] = 0
        increase = F.cross_entropy(lab.logits(covered), Y, reduction="none") - base
        occlusion.append(increase[X[:, position] != 0].mean().item())

    # Gradient x input: how much would the log-probability of the right answer shift?
    embedded = lab.model[0](X).detach().requires_grad_()
    logits = lab.model[1:](embedded)
    F.cross_entropy(logits, Y, reduction="sum").backward()
    grad = cast(torch.Tensor, embedded.grad)
    per_position = (grad * embedded).sum(-1).abs().detach()
    saliency = [per_position[X[:, p] != 0, p].mean().item() for p in range(block)]

    labels = [str(block - p) for p in range(block)]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    left.bar(labels, occlusion, color=blue, width=0.6)
    left.set_ylabel("Increase in loss when covered")
    left.set_title("Occlusion")
    right.bar(labels, saliency, color=blue, width=0.6)
    right.set_ylabel("Mean |gradient x input|")
    right.set_title("Saliency")
    for ax in (left, right):
        ax.set_xlabel("Characters back from the one being predicted")
        ax.invert_xaxis()
        style(ax)
    save(fig, "importance.png")

    print("\nChars back  occlusion  saliency")
    for p in reversed(range(block)):
        print(f"{block - p:>10}  {occlusion[p]:>9.4f}  {saliency[p]:>8.4f}")


def describe(texts: list[str]) -> str:
    """Name the strongest thing a neuron's top inputs have in common."""
    best = (0.0, "")
    for back in range(1, len(texts[0]) + 1):
        char, count = Counter(t[-back] for t in texts).most_common(1)[0]
        best = max(best, (count / len(texts), f"{char!r} {back} back"))
    vowel_share = sum(t[-1] in vowels for t in texts) / len(texts)
    return f"{best[1]} in {best[0]:.0%}, last char a vowel in {vowel_share:.0%}"


def neurons(lab: Lab, top: int = 20) -> None:
    """Report the inputs that most excite the most selective neurons in each layer."""
    lines: list[str] = []
    for layer in range(3):
        length = 2 ** (layer + 1)
        grams = torch.unique(lab.Xtr[:, -length:], dim=0)
        windows = torch.zeros(len(grams), block, dtype=torch.long)
        windows[:, -length:] = grams
        acts = lab.activations(windows, layer)

        selectivity = (acts.topk(10, dim=0).values.mean(0) - acts.mean(0)) / acts.std(0)
        lines.append(f"\nLayer {layer + 1}: sees {length} chars, {len(grams)} distinct inputs in train")
        for neuron in selectivity.topk(top).indices.tolist():
            order = acts[:, neuron].topk(20).indices
            texts = [lab.text(grams[i]) for i in order]
            lines.append(f"  neuron {neuron:>3} (selectivity {selectivity[neuron]:.1f}): {describe(texts)}")
            lines.append("    " + "  ".join(texts[:8]))

    report = "\n".join(lines)
    print(report)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "neurons.txt").write_text(report + "\n")
    plot_pairs(lab)


def plot_pairs(lab: Lab) -> None:
    """Layer 1 only sees two characters, so its neurons can be mapped over every possible pair."""
    order = [0] + [lab.stoi[c] for c in vowels] + [i for i in range(1, len(lab.itos)) if lab.itos[i] not in vowels]
    pairs = torch.cartesian_prod(torch.tensor(order), torch.tensor(order))
    windows = torch.zeros(len(pairs), block, dtype=torch.long)
    windows[:, -2:] = pairs
    acts = lab.activations(windows, 0)
    zscores = (acts - acts.mean(0)) / acts.std(0)
    chosen = zscores.topk(10, dim=0).values.mean(0).topk(6).indices.tolist()

    size = len(order)
    limit = float(zscores[:, chosen].abs().quantile(0.995))
    fig, axes = plt.subplots(2, 3, figsize=(14, 9.5), layout="constrained")
    mesh = None
    ticks = [lab.itos[i] for i in order]
    for ax, neuron in zip(axes.flat, chosen):
        grid = zscores[:, neuron].view(size, size)
        mesh = ax.pcolormesh(grid, cmap=diverging_cmap, vmin=-limit, vmax=limit)
        ax.set_xticks([i + 0.5 for i in range(size)], ticks, fontsize=7)
        ax.set_yticks([i + 0.5 for i in range(size)], ticks, fontsize=7)
        ax.invert_yaxis()
        ax.set_title(f"Neuron {neuron}")
        ax.set_xlabel("Most recent character")
        ax.set_ylabel("Character before it")
        ax.tick_params(length=0)
    assert mesh is not None
    colorbar = fig.colorbar(mesh, ax=axes, label="Activation (standard deviations from the mean)", shrink=0.7)
    colorbar.outline.set_visible(False)
    fig.suptitle("The 6 most selective layer 1 neurons, over all character pairs")
    save(fig, "neuron_pairs.png")


def ablation(lab: Lab, repeats: int = 5) -> None:
    """Remove growing random fractions of each layer, averaged over random masks."""
    fractions = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9]
    generator = torch.Generator().manual_seed(split_seed)
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    print("\nDev loss with a fraction of each layer's neurons removed:")
    print("fraction " + " ".join(f"{f:>7g}" for f in fractions))
    for layer in range(3):
        losses: list[float] = []
        for fraction in fractions:
            trials: list[float] = []
            for _ in range(repeats):
                keep = (torch.rand(lab.n_hidden, generator=generator) >= fraction).float()
                with edited(lab, layer, lambda output: output * keep):
                    trials.append(lab.dev_loss())
            losses.append(sum(trials) / repeats)
        ax.plot(fractions, losses, color=series[layer], linewidth=2, marker="o", markersize=8, label=f"Layer {layer + 1}")
        ax.annotate(f"Layer {layer + 1}", (fractions[-1], losses[-1]), xytext=(8, 0), textcoords="offset points",
                    color=ink, va="center")
        print(f"layer {layer + 1}  " + " ".join(f"{loss:>7.4f}" for loss in losses))
    ax.set_xlabel("Fraction of neurons removed")
    ax.set_ylabel("Dev loss")
    ax.set_title(f"Dev loss as neurons are removed (mean of {repeats} random masks)")
    ax.legend(frameon=False)
    style(ax)
    save(fig, "ablation.png")


def loudest(lab: Lab, layer: int) -> int:
    return int(lab.activations(lab.Xdev, layer).std(0).argmax())


def steer(lab: Lab, layer: int, neuron: int | None, strength: float, n: int = 30) -> None:
    """Generate names from the same random seed with one neuron silenced, then boosted."""
    chosen = loudest(lab, layer) if neuron is None else neuron
    spread = lab.activations(lab.Xdev, layer)[:, chosen].std().item()
    mask = torch.zeros(lab.n_hidden)
    mask[chosen] = 1.0

    def report(label: str) -> None:
        torch.manual_seed(split_seed)
        names = make_names(lab.model, lab.itos, n)
        letters = Counter("".join(names)).most_common(5)
        mean_length = sum(map(len, names)) / len(names)
        print(f"\n{label}: dev loss {lab.dev_loss():.4f}, mean length {mean_length:.1f}, top letters {letters}")
        print("  " + ", ".join(names))

    print(f"\nSteering layer {layer + 1}, neuron {chosen}")
    report("Unchanged")
    with edited(lab, layer, lambda output: output * (1 - mask)):
        report("Silenced")
    with edited(lab, layer, lambda output: output + mask * strength * spread):
        report(f"Boosted by {strength:g} standard deviations")


def calibration(lab: Lab) -> None:
    """Rescale the model's confidence: a best temperature above 1 means it's overconfident."""
    logits = lab.logits(lab.Xdev)
    temperatures = [t / 20 for t in range(14, 41)]
    losses = [F.cross_entropy(logits / t, lab.Ydev).item() for t in temperatures]
    best = min(range(len(losses)), key=losses.__getitem__)

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    ax.plot(temperatures, losses, color=blue, linewidth=2)
    ax.scatter([temperatures[best]], [losses[best]], s=70, color=blue, edgecolors=surface, linewidths=2, zorder=3)
    ax.annotate(f"Best: {temperatures[best]:g} ({losses[best]:.4f})", (temperatures[best], losses[best]),
                xytext=(0, -18), textcoords="offset points", color=ink, ha="center")
    ax.set_xlabel("Temperature (above 1 softens predictions)")
    ax.set_ylabel("Dev loss")
    ax.set_title("Dev loss by prediction temperature")
    style(ax)
    save(fig, "calibration.png")
    print(f"\nDev loss at temperature 1: {losses[temperatures.index(1.0)]:.4f}")
    print(f"Best temperature: {temperatures[best]:g}, dev loss {losses[best]:.4f}")


def probe(lab: Lab, samples: int = 30000, steps: int = 150) -> None:
    """Fit a simple classifier on each layer's activations, to see which concepts it encodes."""
    vowel_ids = torch.tensor([lab.stoi[c] for c in vowels])

    def first_char(X: torch.Tensor) -> torch.Tensor:
        return X.gather(1, (X != 0).float().argmax(1, keepdim=True)).squeeze(1)

    concepts: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
        "last char is a vowel": lambda X: torch.isin(X[:, -1], vowel_ids),
        "second-last char is a vowel": lambda X: torch.isin(X[:, -2], vowel_ids),
        "name starts with a vowel": lambda X: torch.isin(first_char(X), vowel_ids),
        "contains a double letter": lambda X: ((X[:, 1:] == X[:, :-1]) & (X[:, 1:] != 0)).any(1),
        "more than 4 chars so far": lambda X: (X != 0).sum(1) > 4,
    }

    torch.manual_seed(split_seed)
    rows = torch.randperm(len(lab.Xtr))[:samples]
    Xtr = lab.Xtr[rows]
    train_acts = [lab.activations(Xtr, layer) for layer in range(3)]
    dev_acts = [lab.activations(lab.Xdev, layer) for layer in range(3)]

    print(f"\nDev accuracy of a linear classifier on each layer ({samples} training windows):")
    print(f"{'concept':<30} {'guess':>6} {'layer 1':>8} {'layer 2':>8} {'layer 3':>8}")
    for name, concept in concepts.items():
        ytr, ydev = concept(Xtr).float(), concept(lab.Xdev).float()
        guess = max(ydev.mean().item(), 1 - ydev.mean().item())
        scores: list[float] = []
        for layer in range(3):
            mean, std = train_acts[layer].mean(0), train_acts[layer].std(0) + 1e-6
            inputs, dev_inputs = (train_acts[layer] - mean) / std, (dev_acts[layer] - mean) / std
            classifier = nn.Linear(lab.n_hidden, 1)
            optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.01)
            for _ in range(steps):
                optimizer.zero_grad()
                F.binary_cross_entropy_with_logits(classifier(inputs).squeeze(1), ytr).backward()
                optimizer.step()
            with torch.no_grad():
                scores.append(((classifier(dev_inputs).squeeze(1) > 0).float() == ydev).float().mean().item())
        print(f"{name:<30} {guess:>6.1%} " + " ".join(f"{s:>8.1%}" for s in scores))


def detectors(lab: Lab, threshold: float = 0.5) -> None:
    """Count layer 1's single-letter detectors, and layer 3's detectors for chunks of names."""
    size = len(lab.itos)
    pairs = torch.cartesian_prod(torch.arange(size), torch.arange(size))
    windows = torch.zeros(len(pairs), block, dtype=torch.long)
    windows[:, -2:] = pairs
    acts = lab.activations(windows, 0)
    responses = (acts - acts.mean(0)) / acts.std(0)

    # How much of each neuron's behaviour is explained by "this exact letter was here"?
    fits = []
    for position in (0, 1):
        present = F.one_hot(pairs[:, position], size).float()
        present = (present - present.mean(0)) / present.std(0)
        fits.append(((responses.T @ present) / len(pairs)).clamp(min=0) ** 2)
    fit = torch.stack(fits)  # (side, neuron, letter)
    best_by_side = fit.amax(2)
    best, side = best_by_side.max(0)
    letter = fit[side, torch.arange(lab.n_hidden)].argmax(1)

    found = best >= threshold
    counts = torch.zeros(2, size, dtype=torch.long)
    for s, c in zip(side[found].tolist(), letter[found].tolist()):
        counts[s, c] += 1

    order = list(range(1, size)) + [0]
    x = torch.arange(size).float()
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 4.8), layout="constrained")
    left.bar(x - 0.2, counts[1, order], width=0.38, color=blue, label="Most recent character")
    left.bar(x + 0.2, counts[0, order], width=0.38, color=orange, label="Character before it")
    left.set_xticks(x.tolist(), [lab.itos[i] for i in order])
    left.set_ylabel("Neurons")
    left.set_title(f"Layer 1 single-letter detectors ({int(found.sum())} of {lab.n_hidden} neurons)")
    left.legend(frameon=False)
    right.hist(best.tolist(), bins=30, range=(0, 1), color=blue, edgecolor=surface)
    right.axvline(threshold, color=ink, linewidth=1)
    right.set_xlabel("Share of a neuron's variation explained by its best single letter")
    right.set_ylabel("Neurons")
    right.set_title("How letter-like layer 1 neurons are")
    for ax in (left, right):
        style(ax)
    save(fig, "detectors.png")

    print(f"\nLayer 1 neurons where one letter explains at least {threshold:.0%}: {int(found.sum())} of {lab.n_hidden}")
    print(f"  on the most recent character: {int(counts[1].sum())}, on the one before: {int(counts[0].sum())}")
    for cutoff in (0.3, 0.5, 0.7):
        print(f"  at {cutoff:.0%}: {int((best >= cutoff).sum())}")

    grams = torch.unique(lab.Xtr, dim=0)
    layer3 = lab.activations(grams, 2)
    texts = [lab.text(g) for g in grams]
    order_names = torch.randperm(len(lab.names), generator=torch.Generator().manual_seed(split_seed)).tolist()
    train_names = {lab.names[i] for i in order_names[: int(len(lab.names) * 0.8)]}
    stems: list[tuple[float, str, int]] = []
    for neuron in range(lab.n_hidden):
        top = [texts[int(i)] for i in layer3[:, neuron].topk(20).indices]
        chunks = Counter(c for t in top for c in {t[i:i + 4] for i in range(len(t) - 3)} if "." not in c)
        if chunks:
            chunk, count = chunks.most_common(1)[0]
            stems.append((count / 20, chunk, neuron))
    stems.sort(reverse=True)
    strong = [stem for stem in stems if stem[0] >= 0.6]
    print(f"\nLayer 3 neurons with one 4-letter chunk in at least 60% of their top 20 inputs: {len(strong)} of {lab.n_hidden}")
    names_with = {chunk: sum(chunk in n for n in train_names) for _, chunk, _ in stems}
    print("  share  chunk  neuron  train names containing it")
    for share, chunk, neuron in strong[:15]:
        print(f"  {share:>5.0%}  {chunk}  {neuron:>6}  {names_with[chunk]}")

    fig, ax = plt.subplots(figsize=(10, 6.5), layout="constrained")
    ax.scatter([names_with[c] for _, c, _ in stems], [share for share, _, _ in stems], s=45, color=blue,
               edgecolors=surface, linewidths=1)
    labelled: set[str] = set()
    for share, chunk, _ in strong:
        if chunk not in labelled:
            labelled.add(chunk)
            offset = (7, -14) if chunk == "iyah" else (7, 4)  # sits right next to "lian"
            ax.annotate(chunk, (names_with[chunk], share), xytext=offset, textcoords="offset points", color=ink)
    ax.set_xscale("log")
    ax.set_xlabel("Training names containing the neuron's chunk")
    ax.set_ylabel("Share of the neuron's top 20 inputs containing that chunk")
    ax.set_title(f"What layer 3 neurons lock onto ({lab.n_hidden} neurons, each with its strongest 4-letter chunk)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(color=gridline, linewidth=0.8)
    ax.set_axisbelow(True)
    save(fig, "layer3_chunks.png")
    few = sum(names_with[chunk] < 10 for _, chunk, _ in stems)
    print(f"\nNeurons whose strongest chunk is in fewer than 10 names: {few} of {len(stems)}")
    print(f"Lowest name count among chunks in at least 60% of top inputs: {min(names_with[c] for _, c, _ in strong)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tool", choices=["embeddings", "importance", "neurons", "detectors", "ablation", "steer", "calibration", "probe", "all"])
    parser.add_argument("--weights", type=Path, default=default_weights)
    parser.add_argument("--layer", type=int, default=3, choices=[1, 2, 3], help="steer: which layer")
    parser.add_argument("--neuron", type=int, help="steer: which neuron (default: the one with the widest range)")
    parser.add_argument("--strength", type=float, default=4.0, help="steer: boost size, in standard deviations")
    args = parser.parse_args()

    lab = Lab(args.weights)
    print(f"Model: {args.weights.name}, dev loss {lab.dev_loss():.4f}")
    if args.tool in ("embeddings", "all"):
        embeddings(lab)
    if args.tool in ("importance", "all"):
        importance(lab)
    if args.tool in ("neurons", "all"):
        neurons(lab)
    if args.tool in ("detectors", "all"):
        detectors(lab)
    if args.tool in ("ablation", "all"):
        ablation(lab)
    if args.tool in ("steer", "all"):
        steer(lab, args.layer - 1, args.neuron, args.strength)
    if args.tool in ("calibration", "all"):
        calibration(lab)
    if args.tool in ("probe", "all"):
        probe(lab)


if __name__ == "__main__":
    main()
