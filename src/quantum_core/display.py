"""
Plotting utilities for the experiment suites.

All figures are sized for a two-column paper (one figure per column) and use
English labels. Experiment parameters are deliberately *not* printed inside
the figures: they belong in the caption and in the saved config.json.

Results passed to these functions may be a dict {key: result} or a list of
result dicts, as returned by run_experiment(). When several seeds are present
for the same configuration, curves show the median over seeds and a shaded
inter-quartile range (IQR).
"""
import os
import logging
from typing import Dict, Iterable, List, Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


logger = logging.getLogger(__name__)

_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
           "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#000000"]
_MARKERS = ["o", "s", "^", "v", "D", "p", "*", "h", "X", "d", "P"]
_RC = {
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7,
}

Results = Union[Dict[str, dict], List[dict]]


def _as_list(results: Results) -> List[dict]:
    return list(results.values()) if isinstance(results, dict) else list(results)


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _curves(results: Iterable[dict], best_so_far: bool) -> np.ndarray:
    """Stack the CVaR traces of several runs, padded to the longest one.

    A run that stopped before the budget keeps its last value (its search is
    over), so the padding does not invent progress.
    """
    traces = []
    for res in results:
        y = np.array([h["cvar_mean_energy"] for h in res["history"]], dtype=float)
        if best_so_far:
            y = np.minimum.accumulate(y)
        traces.append(y)
    length = max(len(t) for t in traces)
    return np.vstack([np.pad(t, (0, length - len(t)), mode="edge") for t in traces])


def _plot_band(ax, data: np.ndarray, color, marker, label) -> None:
    x = np.arange(1, data.shape[1] + 1)
    med = np.median(data, axis=0)
    ax.plot(x, med, color=color, marker=marker, markevery=max(1, len(x) // 8),
            markersize=3.5, linewidth=1.4, label=label)
    if data.shape[0] > 1:
        q1, q3 = np.percentile(data, [25, 75], axis=0)
        ax.fill_between(x, q1, q3, color=color, alpha=0.15, linewidth=0)


def _grouped_panels(results: Results, group_key: str, label_fmt, best_so_far: bool,
                    ylabel: str, output_dir: str, filename: str,
                    titles: Optional[Dict[str, str]] = None) -> str:
    runs = _as_list(results)
    ansatze = [a for a in ("QAOA", "VQE") if any(r["ansatz_type"] == a for r in runs)]
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, len(ansatze), figsize=(3.3 * len(ansatze), 2.6), squeeze=False)
        for ax, ansatz in zip(axes[0], ansatze):
            groups: Dict = {}
            for r in runs:
                if r["ansatz_type"] == ansatz:
                    groups.setdefault(r[group_key], []).append(r)
            for idx, key in enumerate(sorted(groups, key=lambda k: (isinstance(k, str), k))):
                _plot_band(ax, _curves(groups[key], best_so_far),
                           _COLORS[idx % len(_COLORS)], _MARKERS[idx % len(_MARKERS)],
                           label_fmt(key))
            _style(ax, (titles or {}).get(ansatz, ansatz), "Circuit evaluations", ylabel)
        handles, labels = axes[0][0].get_legend_handles_labels()
        ncol = min(len(labels), 7)
        fig.tight_layout(rect=(0, 0.08 * ((len(labels) + ncol - 1) // ncol), 1, 1))
        fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False,
                   bbox_to_anchor=(0.5, 0.0))
        out_path = os.path.join(output_dir, filename)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    logger.info("Figure saved to %s", out_path)
    return out_path


def plot_cost_evolution(results: Results, output_dir: str,
                        filename: str = "cost_evolution.png") -> str:
    """CVaR cost per circuit evaluation, one curve per alpha (median ± IQR over seeds)."""
    return _grouped_panels(
        results, "cvar_alpha", lambda a: rf"$\alpha={a:g}$", best_so_far=False,
        ylabel=r"CVaR$_\alpha$ cost", output_dir=output_dir, filename=filename,
    )


def plot_optimizer_benchmark(results: Results, output_dir: str,
                             filename: str = "optimizer_benchmark.png",
                             titles: Optional[Dict[str, str]] = None) -> str:
    """Best CVaR found so far per circuit evaluation, one curve per optimizer."""
    return _grouped_panels(
        results, "optimizer_name", lambda o: str(o).replace("_PDFO", " (PDFO)"),
        best_so_far=True, ylabel=r"Best CVaR$_\alpha$ so far",
        output_dir=output_dir, filename=filename, titles=titles,
    )


def plot_curves_by(results: Results, key: str, output_dir: str, filename: str) -> str:
    """Generic variant: one curve per distinct value of `key` (e.g. instance size)."""
    return _grouped_panels(
        results, key, lambda v: f"{key}={v}", best_so_far=False,
        ylabel=r"CVaR$_\alpha$ cost", output_dir=output_dir, filename=filename,
    )


def _success_curves(runs: List[dict], thresholds, x_max: float):
    """Fraction of instances whose best success probability reached each threshold."""
    x = np.linspace(0.0, x_max, 400)
    out = {}
    for tau in thresholds:
        reached = []
        for r in runs:
            n = r["num_qubits"]
            sp = np.maximum.accumulate([h["success_prob"] for h in r["history"]])
            t = np.arange(1, len(sp) + 1) / n
            hit = np.nonzero(sp >= tau)[0]
            reached.append(t[hit[0]] if len(hit) else np.inf)
        reached = np.array(reached)
        out[tau] = (x, np.array([(reached <= xi).mean() for xi in x]))
    return out


def plot_success_probability(results: Results, output_dir: str,
                             filename: str = "success_probability.png") -> str:
    """Fraction of instances reaching p_succ >= 1% (blue) and >= 10% (red) versus
    normalised circuit evaluations t/n. Rows: depths. Columns: QAOA, VQE.
    Reproduces the structure of Fig. 4 in Barkoutsos et al. (Quantum 4, 256, 2020).
    """
    runs = _as_list(results)
    all_reps = sorted({r["reps"] for r in runs})
    all_alphas = sorted({r["cvar_alpha"] for r in runs})
    x_max = max(len(r["history"]) / r["num_qubits"] for r in runs)
    shades = np.linspace(0.4, 1.0, max(len(all_alphas), 1))
    palettes = {0.01: matplotlib.colormaps["Blues"], 0.10: matplotlib.colormaps["Reds"]}

    with plt.rc_context(_RC):
        fig, axes = plt.subplots(len(all_reps), 2, figsize=(6.6, 2.3 * len(all_reps)),
                                 sharey=True, squeeze=False)
        for row, reps in enumerate(all_reps):
            for col, ansatz in enumerate(["QAOA", "VQE"]):
                ax = axes[row][col]
                for a_idx, alpha in enumerate(all_alphas):
                    sel = [r for r in runs if r["reps"] == reps and r["ansatz_type"] == ansatz
                           and r["cvar_alpha"] == alpha]
                    if not sel:
                        continue
                    for tau, (xs, ys) in _success_curves(sel, palettes, x_max).items():
                        ax.step(xs, ys, where="post", linewidth=1.3,
                                color=palettes[tau](shades[a_idx]),
                                label=rf"{tau:.0%}, $\alpha={alpha:g}$")
                _style(ax, f"{ansatz}, p = {reps}",
                       "Normalized evaluations t / n" if row == len(all_reps) - 1 else "",
                       "Fraction of instances" if col == 0 else "")
                ax.set_ylim(0, 1.05)
        handles, labels = axes[0][1].get_legend_handles_labels()
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
                   bbox_to_anchor=(0.5, 0.0))
        out_path = os.path.join(output_dir, filename)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    logger.info("Success probability figure saved to %s", out_path)
    return out_path


def plot_distribution_evolution(result: dict, output_dir: str, filename: Optional[str] = None) -> Optional[str]:
    """Outcome distribution at the first and last circuit evaluation.

    Bitstrings are sorted by energy (lowest left). Orange: optimal states;
    green: states inside the CVaR alpha-tail; blue: other states. Labels are
    written in variable order (character i = vertex i).
    """
    history = result.get("history", [])
    if len(history) < 2:
        logger.warning("Not enough evaluations to plot distribution evolution (need >= 2).")
        return None

    n = result["num_qubits"]
    alpha = result["cvar_alpha"]
    total_states = 2 ** n
    optimal_states = set(result.get("optimal_states", frozenset()))

    energies: Dict[int, float] = {}
    for h in (history[0], history[-1]):
        energies.update(h.get("state_energies", {}))
    ordered = sorted(energies, key=lambda s: (energies[s], s))
    ordered += sorted(s for s in range(total_states) if s not in energies)

    def probs(dist):
        tot = sum(dist.values())
        return np.array([dist.get(s, 0) / tot for s in ordered])

    def cvar_states(h):
        dist, en = h["distribution"], h.get("state_energies", {})
        tot = sum(dist.values())
        acc, inside = 0.0, set()
        for s in sorted(en, key=lambda s: en[s]):
            if acc >= alpha:
                break
            inside.add(s)
            acc += dist.get(s, 0) / tot
        return inside

    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4), sharey=True)
        for ax, h, title in zip(axes, (history[0], history[-1]),
                                ("Before optimization", f"After {len(history)} evaluations")):
            tail = cvar_states(h)
            colors = ["#ff7f0e" if s in optimal_states else "#2ca02c" if s in tail else "#1f77b4"
                      for s in ordered]
            ax.bar(np.arange(total_states), probs(h["distribution"]), color=colors, width=0.85)
            ax.set_xticks([])
            _style(ax, title, "Bitstrings, sorted by increasing energy",
                   "Probability" if ax is axes[0] else "")
        axes[1].legend(handles=[
            Patch(color="#ff7f0e", label="Optimal cut"),
            Patch(color="#2ca02c", label=rf"CVaR tail ($\alpha={alpha:g}$)"),
            Patch(color="#1f77b4", label="Other states"),
        ], loc="upper right", framealpha=0.9)
        fig.tight_layout()
        if filename is None:
            filename = f"distribution_{result['ansatz_type']}_alpha{alpha:g}.png"
        out_path = os.path.join(output_dir, filename)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    logger.info("Distribution figure saved to %s", out_path)
    return out_path
