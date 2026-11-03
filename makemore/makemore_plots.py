"""Plot how the sweep's hyperparameters relate to dev loss.

    python makemore_plots.py --round round2             # save PNGs to plots/round2/ and open rotatable windows
    python makemore_plots.py --round round2 --no-show   # just save the PNGs

Works on a partial results file too, so it can be run while a sweep is still going.
Each config counts once, using its mean dev loss across seeds.
"""
import argparse
import itertools
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from makemore_experiment import Config, Record, group_by_config, here, load_records, mean_dev, rounds

type Scores = dict[Config, float]
type Grid = np.ndarray[tuple[int, int], np.dtype[np.float64]]

labels = {
    "peak_lr": "Peak learning rate",
    "n_hidden": "Hidden size",
    "dropout": "Dropout",
    "weight_decay": "Weight decay",
}

ink = "#0b0b0b"
muted_ink = "#52514e"
gridline = "#e1e0d9"
surface = "#fcfcfb"
blue = "#2a78d6"
light_blue = "#9ec5f4"
# One hue, reversed so that the lowest (best) dev loss is the darkest.
loss_cmap = LinearSegmentedColormap.from_list("loss", ["#0d366b", "#256abf", "#5598e7", "#9ec5f4", "#cde2fb"])

plt.rcParams.update({
    "figure.facecolor": surface,
    "axes.facecolor": surface,
    "savefig.facecolor": surface,
    "text.color": ink,
    "axes.labelcolor": muted_ink,
    "axes.edgecolor": gridline,
    "xtick.color": muted_ink,
    "ytick.color": muted_ink,
    "font.size": 10,
})


def values_of(scores: Scores, field: str) -> list[float]:
    return sorted({getattr(c, field) for c in scores})


def swept_params(scores: Scores) -> list[str]:
    """The parameters that actually vary in these results."""
    return [field for field in labels if len(values_of(scores, field)) > 1]


def positions(field: str, values: list[float]) -> list[float]:
    """Where each value sits on an axis: lr is log scaled, weight decay spans too wide a range for linear."""
    if field == "peak_lr":
        return [float(np.log10(v)) for v in values]
    if field == "weight_decay":
        return [float(i) for i in range(len(values))]
    return values


def set_ticks(
    field: str,
    values: list[float],
    set_ticks_fn: Callable[[list[float]], Any],
    set_labels_fn: Callable[[list[str]], Any],
) -> None:
    set_ticks_fn(positions(field, values))
    set_labels_fn([f"{v:g}" for v in values])


def mean_grid(scores: Scores, x_field: str, y_field: str) -> Grid:
    """Mean dev loss per (y, x) pair, averaged over the other parameters. NaN means no runs yet."""
    xs, ys = values_of(scores, x_field), values_of(scores, y_field)
    grid = np.full((len(ys), len(xs)), np.nan)
    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            cell = [s for c, s in scores.items() if getattr(c, x_field) == x and getattr(c, y_field) == y]
            if cell:
                grid[row, col] = np.mean(cell)
    return grid


def loss_range(grids: list[Grid]) -> tuple[float, float]:
    """Shared colour range, so that a shade means the same loss in every panel."""
    all_cells = np.concatenate([g.ravel() for g in grids])
    return float(np.nanmin(all_cells)), float(np.nanmax(all_cells))


def plot_each_parameter(scores: Scores, subtitle: str) -> Figure:
    """Dev loss against each parameter alone: a dot per config, a line through the means."""
    params = swept_params(scores)
    fig, axes = plt.subplots(
        1, len(params), figsize=(max(4.4 * len(params), 6), 4.5), sharey=True, squeeze=False, layout="constrained"
    )
    for ax, field in zip(axes.flat, params):
        values = values_of(scores, field)
        xs = positions(field, values)
        means = []
        for x, value in zip(xs, values):
            ys = [s for c, s in scores.items() if getattr(c, field) == value]
            ax.scatter([x] * len(ys), ys, s=45, color=light_blue, edgecolors=surface, linewidths=1, zorder=2)
            means.append(np.mean(ys))
        ax.plot(xs, means, color=blue, linewidth=2, marker="o", markersize=8, zorder=3, label="Mean of configs")
        set_ticks(field, values, ax.set_xticks, ax.set_xticklabels)
        ax.set_xlabel(labels[field])
        ax.grid(axis="y", color=gridline, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=0)
    axes.flat[0].set_ylabel("Dev loss")
    axes.flat[0].legend(frameon=False, loc="best")
    fig.suptitle(f"Dev loss by each parameter ({subtitle}; each dot is one config)")
    return fig


def plot_heatmaps(scores: Scores, subtitle: str) -> Figure:
    """Dev loss for each pair of parameters, with the exact value written in each cell."""
    pairs = list(itertools.combinations(swept_params(scores), 2))
    grids = [mean_grid(scores, x, y) for x, y in pairs]
    vmin, vmax = loss_range(grids)
    rows = math.ceil(len(pairs) / 3)
    fig, axes = plt.subplots(rows, 3, figsize=(14, 4.8 * rows), squeeze=False, layout="constrained")
    for ax in axes.flat[len(pairs):]:
        ax.set_visible(False)
    mesh = None
    for ax, (x_field, y_field), grid in zip(axes.flat, pairs, grids):
        xs, ys = values_of(scores, x_field), values_of(scores, y_field)
        mesh = ax.pcolormesh(grid, cmap=loss_cmap, vmin=vmin, vmax=vmax, edgecolors=surface, linewidth=2)
        for row, col in np.ndindex(grid.shape):
            if not np.isnan(grid[row, col]):
                dark_cell = (grid[row, col] - vmin) / max(vmax - vmin, 1e-9) < 0.5
                ax.text(col + 0.5, row + 0.5, f"{grid[row, col]:.4f}", ha="center", va="center",
                        color="#ffffff" if dark_cell else ink)
        ax.set_xticks(np.arange(len(xs)) + 0.5, [f"{v:g}" for v in xs])
        ax.set_yticks(np.arange(len(ys)) + 0.5, [f"{v:g}" for v in ys])
        ax.set_xlabel(labels[x_field])
        ax.set_ylabel(labels[y_field])
        ax.tick_params(length=0)
        ax.spines[:].set_visible(False)
    assert mesh is not None
    colorbar = fig.colorbar(mesh, ax=axes, label="Dev loss (darker is better)", shrink=0.85)
    colorbar.outline.set_visible(False)
    fig.suptitle(f"Mean dev loss for each pair, averaged over the other parameters ({subtitle})")
    return fig


def plot_surfaces(scores: Scores, subtitle: str) -> Figure:
    """The same pairs as 3D surfaces. Dots are the measured means; the mesh just joins them."""
    pairs = list(itertools.combinations(swept_params(scores), 2))
    grids = [mean_grid(scores, x, y) for x, y in pairs]
    vmin, vmax = loss_range(grids)
    rows = math.ceil(len(pairs) / 3)
    fig = plt.figure(figsize=(16, 5.8 * rows), layout="constrained")
    for i, ((x_field, y_field), grid) in enumerate(zip(pairs, grids), 1):
        ax = fig.add_subplot(rows, 3, i, projection="3d")
        xs, ys = values_of(scores, x_field), values_of(scores, y_field)
        X, Y = np.meshgrid(positions(x_field, xs), positions(y_field, ys))
        ax.plot_surface(X, Y, grid, cmap=loss_cmap, vmin=vmin, vmax=vmax, rstride=1, cstride=1,
                        edgecolor=surface, linewidth=0.6, alpha=0.95)
        ax.scatter(X.ravel(), Y.ravel(), grid.ravel(), c=grid.ravel(), cmap=loss_cmap, vmin=vmin, vmax=vmax, s=30,
                   edgecolors=surface, linewidths=0.8, depthshade=False)
        set_ticks(x_field, xs, ax.set_xticks, ax.set_xticklabels)
        set_ticks(y_field, ys, ax.set_yticks, ax.set_yticklabels)
        ax.set_zlim(vmin, vmax)
        ax.set_xlabel(labels[x_field], labelpad=8)
        ax.set_ylabel(labels[y_field], labelpad=8)
        ax.set_zlabel("Dev loss", labelpad=10)
        ax.tick_params(axis="z", pad=5)
        ax.view_init(elev=28, azim=45)  # look from the high lr / large model corner
        ax.set_box_aspect((1, 1, 0.75), zoom=1.15)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color(surface)
            axis._axinfo["grid"]["color"] = gridline
    fig.suptitle(f"Dev loss surfaces, averaged over the other parameters ({subtitle}); drag to rotate")
    return fig


def plot_curves(records: list[Record]) -> Figure:
    """Training curves for up to 4 runs, to show whether they were still improving at the end."""
    runs = sorted(records, key=lambda r: (r["config"]["dropout"], r["config"]["seed"]))[:4]
    fig, (train_ax, dev_ax) = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for run, color in zip(runs, (blue, "#eb6834", "#1baf7a", "#eda100")):
        config = run["config"]
        label = f"Dropout {config['dropout']:g}, seed {config['seed']}"
        epochs = [point["epoch"] for point in run["curve"]]
        dev = [point["dev_loss"] for point in run["curve"]]
        train_ax.plot(epochs, [point["train_loss"] for point in run["curve"]], color=color, linewidth=2, label=label)
        dev_ax.plot(epochs, dev, color=color, linewidth=2, label=label)
        best = min(range(len(dev)), key=dev.__getitem__)
        dev_ax.scatter([epochs[best]], [dev[best]], s=70, color=color, edgecolors=surface, linewidths=2, zorder=3)
        print(f"{label}: best dev {dev[best]:.4f} at epoch {epochs[best]} of {epochs[-1]}, final dev {dev[-1]:.4f}")
    train_ax.set_title("Training loss, dropout on")
    dev_ax.set_title("Dev loss (dot marks the best epoch)")
    for ax in (train_ax, dev_ax):
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(axis="y", color=gridline, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=0)
    dev_ax.legend(frameon=False)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--round", required=True, choices=rounds, help="which experiment to plot")
    parser.add_argument("--results", type=Path, help="override the round's results file")
    parser.add_argument("--out-dir", type=Path, help="default: plots/<round>")
    parser.add_argument("--no-show", action="store_true", help="save the PNGs without opening windows")
    args = parser.parse_args()

    args.results = args.results or rounds[args.round].results
    args.out_dir = args.out_dir or here / "plots" / args.round

    records = load_records(args.results)
    groups = group_by_config(records)
    if not groups:
        raise SystemExit(f"No results in {args.results}")
    scores: Scores = {c: mean_dev(rs) for c, rs in groups.items()}
    swept = swept_params(scores)
    subtitle = f"{len(scores)} configs"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    figures: dict[str, Figure] = {}
    if swept:
        figures["each_parameter"] = plot_each_parameter(scores, subtitle)
    if len(swept) >= 2:
        figures["heatmaps"] = plot_heatmaps(scores, subtitle)
        figures["surfaces"] = plot_surfaces(scores, subtitle)
    curves = [r for r in records if "curve" in r]
    if curves:
        figures["curves"] = plot_curves(curves)
    for name, fig in figures.items():
        path = args.out_dir / f"{name}.png"
        fig.savefig(path, dpi=150)
        print(f"Saved {path}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
