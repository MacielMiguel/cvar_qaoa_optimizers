import os
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def style_plot(ax, title, xlabel, ylabel):
    """Apply corporate styling to matplotlib axes."""
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=True, fancybox=True, framealpha=0.9, loc='best')


def plot_cost_evolution(results_suite1, output_dir):
    """
    Figure 1: Evolution du coût CVaR pour QAOA et VQE avec différents alpha.
    
    Deux subplots: un pour QAOA, un pour VQE.
    Chaque subplot affiche l'évolution du coût pour tous les alphas testés.
    """
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.5, 0.05, 4.5], hspace=0.3)
    
    # Extended distinct colors and markers
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'X', 'd']
    
    # Collect metadata
    num_qubits = None
    p = None
    all_alphas = []
    total_time = 0
    max_iters = 0
    optimizer_name = None
    
    for key, res in results_suite1.items():
        if num_qubits is None:
            num_qubits = res['num_qubits']
        if p is None:
            p = res.get('reps', 1)
        all_alphas.append(res['cvar_alpha'])
        total_time += res['time']
        max_iters = max(max_iters, len(res['history']))
        if optimizer_name is None:
            optimizer_name = res['optimizer_name']
    
    all_alphas = sorted(set(all_alphas))
    
    # Add info panel at top
    ax_info = fig.add_subplot(gs[0, :])
    ax_info.axis('off')
    
    info_text = (
        f"Qubits: {num_qubits}  |  Profondeur (p): {p}  |  Alphas: {len(all_alphas)} "
        f"({min(all_alphas):.2f}–{max(all_alphas):.2f})  |  "
        f"Max Itérations: {max_iters}  |  Optimiseur: {optimizer_name}  |  "
        f"Temps Total: {total_time:.1f}s"
    )
    ax_info.text(0.5, 0.5, info_text, ha='center', va='center', fontsize=12, 
                 fontweight='bold', transform=ax_info.transAxes,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7, pad=1))
    
    # Separate QAOA and VQE
    for plot_idx, ansatz_type in enumerate(['QAOA', 'VQE']):
        ax = fig.add_subplot(gs[2, plot_idx])
        
        # Collect results by alpha for this ansatz type
        results_by_alpha = {}
        for key, res in results_suite1.items():
            if res['ansatz_type'] == ansatz_type:
                alpha = res['cvar_alpha']
                if alpha not in results_by_alpha:
                    results_by_alpha[alpha] = res
        
        # Sort alphas for consistent coloring
        sorted_alphas = sorted(results_by_alpha.keys())
        
        # Plot for each alpha
        for alpha_idx, alpha in enumerate(sorted_alphas):
            res = results_by_alpha[alpha]
            history = res['history']
            y_vals = [step["cvar_mean_energy"] for step in history]
            
            label = fr"$\alpha={alpha:.2f}$ ({res['time']:.1f}s)"
            ax.plot(
                y_vals,
                label=label,
                linewidth=2.5,
                color=colors[alpha_idx % len(colors)],
                marker=markers[alpha_idx % len(markers)],
                markersize=6,
                markevery=max(1, len(y_vals)//15),
                alpha=0.9
            )
        
        style_plot(
            ax,
            fr"{ansatz_type} - Convergence pour différentes valeurs de $\alpha$",
            "Itérations d'optimisation",
            "Coût CVaR attendu"
        )
    
    fig.suptitle("Évolution du coût CVaR-QAOA et CVaR-VQE", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    out_path = os.path.join(output_dir, "cost_evolution.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Cost evolution figure saved to %s", out_path)


def plot_optimizer_benchmark(results_suite2, output_dir):
    """
    Figure 2: Benchmark des optimizers pour QAOA et VQE.
    
    Deux subplots: un pour QAOA, un pour VQE.
    Chaque subplot affiche la convergence pour différents algorithmes d'optimisation.
    """
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.5, 0.05, 4.5], hspace=0.3)
    
    # Extended distinct colors and markers
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'X', 'd']
    
    # Collect metadata
    num_qubits = None
    p = None
    alpha_value = None
    all_optimizers = set()
    total_time = 0
    max_iters = 0
    
    for key, res in results_suite2.items():
        if num_qubits is None:
            num_qubits = res['num_qubits']
        if p is None:
            p = res.get('reps', 1)
        if alpha_value is None:
            alpha_value = res['cvar_alpha']
        all_optimizers.add(res['optimizer_name'])
        total_time += res['time']
        max_iters = max(max_iters, len(res['history']))
    
    all_optimizers = sorted(all_optimizers)
    
    # Add info panel at top
    ax_info = fig.add_subplot(gs[0, :])
    ax_info.axis('off')
    
    info_text = (
        f"Qubits: {num_qubits}  |  Profondeur (p): {p}  |  Alpha: {alpha_value:.2f}  |  "
        f"Optimiseurs: {len(all_optimizers)} ({', '.join(all_optimizers)})  |  "
        f"Max Itérations: {max_iters}  |  Temps Total: {total_time:.1f}s"
    )
    ax_info.text(0.5, 0.5, info_text, ha='center', va='center', fontsize=12,
                 fontweight='bold', transform=ax_info.transAxes,
                 bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.7, pad=1))
    
    # Separate QAOA and VQE
    for plot_idx, ansatz_type in enumerate(['QAOA', 'VQE']):
        ax = fig.add_subplot(gs[2, plot_idx])
        
        # Collect results for this ansatz type
        results_by_optimizer = {}
        for key, res in results_suite2.items():
            if res['ansatz_type'] == ansatz_type:
                optimizer_name = res['optimizer_name']
                results_by_optimizer[optimizer_name] = res
        
        # Sort optimizers for consistent ordering
        sorted_optimizers = sorted(results_by_optimizer.keys())
        
        # Plot for each optimizer
        for opt_idx, opt_name in enumerate(sorted_optimizers):
            res = results_by_optimizer[opt_name]
            history = res['history']
            # Limit to first 50 iterations for better visualization
            y_vals = [step["cvar_mean_energy"] for step in history[:50]]
            
            ax.plot(
                y_vals,
                label=f"{opt_name} ({res['time']:.1f}s)",
                linewidth=2.5,
                color=colors[opt_idx % len(colors)],
                marker=markers[opt_idx % len(markers)],
                markersize=6,
                markevery=max(1, len(y_vals)//15),
                alpha=0.9
            )
        
        alpha_str = f"{alpha_value:.2f}" if alpha_value is not None else "0.5"
        style_plot(
            ax,
            fr"{ansatz_type} - Benchmark d'optimiseurs (α={alpha_str})",
            "Itérations d'optimisation",
            "Coût CVaR attendu"
        )
    
    fig.suptitle("Benchmark des algorithmes d'optimisation", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    out_path = os.path.join(output_dir, "optimizer_benchmark.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Optimizer benchmark figure saved to %s", out_path)


def _compute_success_curves(
    results: list,
    ansatz_type: str,
    reps: int,
    sorted_alphas: list,
    thresholds: list,
    threshold_colors,
    threshold_names: list,
    markers: list,
    ax,
) -> None:
    """Compute and draw success-probability curves on `ax` for one (ansatz, reps) panel."""
    # Filter results for this (ansatz_type, reps)
    results_by_alpha: dict = {}
    for res in results:
        if res['ansatz_type'] == ansatz_type and res['reps'] == reps:
            alpha = res['cvar_alpha']
            results_by_alpha.setdefault(alpha, []).append(res)

    if not results_by_alpha:
        return

    # Shared normalised-iteration grid (each instance uses its own num_qubits)
    all_norm_iters: set = set()
    for alpha in sorted_alphas:
        for res in results_by_alpha.get(alpha, []):
            n = res['num_qubits']
            for k in range(1, len(res['history']) + 1):
                all_norm_iters.add(k / n)
    x_grid = sorted(all_norm_iters)

    for threshold, colors, thresh_name in zip(thresholds, threshold_colors, threshold_names):
        for alpha_idx, alpha in enumerate(sorted_alphas):
            ansatz_results = results_by_alpha.get(alpha, [])
            if not ansatz_results:
                continue

            # Cumulative-best success_prob per instance on the normalised timeline
            instance_curves = []
            for res in ansatz_results:
                n = res['num_qubits']
                best = 0.0
                curve: dict = {}
                for k, step in enumerate(res['history'], start=1):
                    best = max(best, step.get('success_prob', 0.0))
                    curve[k / n] = best
                instance_curves.append(curve)

            # Fraction of instances whose cumulative best >= threshold up to t
            fractions = []
            for t in x_grid:
                count = sum(
                    1 for curve in instance_curves
                    if max((v for nt, v in curve.items() if nt <= t), default=0.0) >= threshold
                )
                fractions.append(count / len(instance_curves))

            ax.plot(
                x_grid,
                fractions,
                label=f"{thresh_name} (α={alpha:.2f})",
                linewidth=1.8,
                color=colors[alpha_idx],
                marker=markers[alpha_idx % len(markers)],
                markersize=5,
                markevery=max(1, len(x_grid) // 15),
                alpha=0.9,
            )


def plot_success_probability(results_suite3: list, output_dir: str) -> None:
    """Plot success-probability curves in a (n_depths × 2) grid.

    Rows correspond to circuit depths (reps); columns to VQE (left) and
    QAOA (right). Reproduces the structure of Figure 4 in Barkoutsos et al.
    (Quantum 4, 256, 2020).

    Blue curves  → 1 % success threshold
    Red  curves  → 10 % success threshold
    Each colour shade corresponds to one CVaR alpha value.
    """
    # ── Collect metadata ────────────────────────────────────────────────
    all_reps   = sorted({res['reps']       for res in results_suite3})
    all_alphas = sorted({res['cvar_alpha'] for res in results_suite3})
    num_qubits = results_suite3[0]['num_qubits'] if results_suite3 else 0
    total_time = sum(res['time'] for res in results_suite3)
    n_rows     = len(all_reps)

    n_alpha = len(all_alphas)
    blue_colors = plt.cm.Blues(np.linspace(0.35, 1.0, max(n_alpha, 1)))
    red_colors  = plt.cm.Reds( np.linspace(0.35, 1.0, max(n_alpha, 1)))
    markers     = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'X', 'd']

    thresholds        = [0.01,       0.10]
    threshold_colors  = [blue_colors, red_colors]
    threshold_names   = ["1%",       "10%"]

    # ── Figure layout: n_rows × 2, VQE left / QAOA right ───────────────
    plt.style.use('ggplot')
    fig, axes = plt.subplots(
        n_rows, 2,
        figsize=(16, 4.5 * n_rows),
        sharey=True,
        sharex=False,
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]   # always 2-D array

    for row_idx, reps in enumerate(all_reps):
        for col_idx, ansatz_type in enumerate(['QAOA', 'VQE']):
            ax = axes[row_idx, col_idx]
            _compute_success_curves(
                results_suite3, ansatz_type, reps,
                all_alphas, thresholds, threshold_colors, threshold_names,
                markers, ax,
            )
            style_plot(
                ax,
                f"{ansatz_type}, depth {reps}",
                "Normalized number of iterations (t / n)",
                "Fraction of Instances",
            )
            ax.set_ylim(0, 1.05)

    # ── Shared legend (top-right panel) ─────────────────────────────────
    handles, labels = axes[0, 1].get_legend_handles_labels()
    if handles:
        axes[0, 1].legend(
            handles, labels,
            fontsize=7, ncol=2,
            loc='lower right',
            framealpha=0.9,
        )
    # Remove auto-legends from all other panels
    for row_idx in range(n_rows):
        for col_idx in range(2):
            if not (row_idx == 0 and col_idx == 1):
                legend = axes[row_idx, col_idx].get_legend()
                if legend:
                    legend.remove()

    fig.suptitle(
        f"Success probability — MaxCut, {num_qubits} qubits  "
        f"(Blue: ≥1%, Red: ≥10%)  |  Total: {total_time:.0f}s",
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()

    out_path = os.path.join(output_dir, "success_probability_report.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Success probability figure saved to %s", out_path)


def plot_distribution_evolution(result: dict, output_dir: str, filename: str | None = None) -> None:
    """Plot the probability distribution at the first and last optimizer iteration.

    Shows side-by-side normalised histograms so you can see how the quantum
    circuit has learned to concentrate probability on lower-energy states.

    Args:
        result:     A single result dict returned by run_experiment().
        output_dir: Directory where the PNG is saved.
        filename:   Output filename (default: distribution_<ansatz>_alpha<alpha>.png).
    """
    history = result.get("history", [])
    if len(history) < 2:
        logger.warning("Not enough iterations to plot distribution evolution (need ≥ 2).")
        return

    num_qubits = result["num_qubits"]
    ansatz_type = result["ansatz_type"]
    cvar_alpha = result["cvar_alpha"]
    shots = result.get("shots", "?")

    first_dist = history[0]["distribution"]
    final_dist = history[-1]["distribution"]
    total_states = 2 ** num_qubits

    def _to_prob_array(dist: dict) -> np.ndarray:
        """Convert {state_int: count} to a probability array over all 2^n states."""
        total = sum(dist.values())
        arr = np.zeros(total_states)
        for state, count in dist.items():
            if state < total_states:
                arr[state] = count / total
        return arr

    def _cvar_states(dist: dict, state_energies: dict, alpha: float) -> set:
        """Return the set of states that fall within the CVaR α-fraction.

        States are sorted by energy (ascending). We accumulate their probability
        mass until we reach alpha, marking each as 'in CVaR'.
        """
        total = sum(dist.values())
        if total == 0 or not state_energies:
            return set()
        probs = {s: c / total for s, c in dist.items()}
        sorted_states = sorted(state_energies, key=lambda s: state_energies[s])
        accumulated = 0.0
        in_cvar = set()
        for s in sorted_states:
            if accumulated >= alpha:
                break
            in_cvar.add(s)
            accumulated += probs.get(s, 0.0)
        return in_cvar

    first_probs = _to_prob_array(first_dist)
    final_probs = _to_prob_array(final_dist)
    optimal_states = result.get("optimal_states", frozenset())
    optimal_labels = [format(s, f"0{num_qubits}b") for s in sorted(optimal_states)]

    first_state_energies = history[0].get("state_energies", {})
    final_state_energies = history[-1].get("state_energies", {})

    # Build a global energy ordering using all sampled energies (union of both iterations).
    # States never sampled in either iteration are placed at the right end (highest energy),
    # sorted by their integer index for determinism.
    combined_energies: dict = {**first_state_energies, **final_state_energies}
    sampled_states_by_energy = sorted(combined_energies, key=lambda s: combined_energies[s])
    unsampled_states = sorted(s for s in range(total_states) if s not in combined_energies)
    ordered_states = sampled_states_by_energy + unsampled_states  # low → high energy

    # x positions are just 0..total_states-1; we re-label them with the ordered bitstrings
    x = np.arange(total_states)
    labels = [format(s, f"0{num_qubits}b") for s in ordered_states]

    # Re-index probability arrays to match the new ordering
    first_probs_ordered = np.array([first_probs[s] for s in ordered_states])
    final_probs_ordered = np.array([final_probs[s] for s in ordered_states])

    first_cvar_states = _cvar_states(first_dist, first_state_energies, cvar_alpha)
    final_cvar_states = _cvar_states(final_dist, final_state_energies, cvar_alpha)

    def _bar_colors(cvar_states: set) -> list:
        """orange = optimal states, green = in CVaR, blue = other."""
        result_colors = []
        for s in ordered_states:
            if s in optimal_states:
                result_colors.append("#ff7f0e")
            elif s in cvar_states:
                result_colors.append("#2ca02c")
            else:
                result_colors.append("#1f77b4")
        return result_colors

    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    fig.suptitle(
        f"Probability Distribution — {ansatz_type}  α={cvar_alpha:.2f}  "
        f"({num_qubits} qubits, {shots} shots)",
        fontsize=14, fontweight="bold",
    )

    for ax, probs, title, cvar_states in zip(
        axes,
        [first_probs_ordered, final_probs_ordered],
        [f"Iteration 1  (before optimisation)", f"Iteration {len(history)}  (after optimisation)"],
        [first_cvar_states, final_cvar_states],
    ):
        bar_colors = _bar_colors(cvar_states)
        ax.bar(x, probs, color=bar_colors, alpha=0.8, width=0.8)
        # Legend proxy artists
        ax.bar([], [], color="#1f77b4", alpha=0.8, label="Other states")
        ax.bar([], [], color="#2ca02c", alpha=0.8, label=f"CVaR α={cvar_alpha:.2f} region")
        if optimal_states:
            opt_legend = ", ".join(optimal_labels[:4])
            if len(optimal_labels) > 4:
                opt_legend += f" (+{len(optimal_labels) - 4} more)"
            ax.bar([], [], color="#ff7f0e", alpha=0.8, label=f"Optimal: {opt_legend}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=max(6, 9 - num_qubits))
        style_plot(ax, title, "Bitstring  (sorted low → high energy)", "Probability")

    fig.tight_layout()

    if filename is None:
        filename = f"distribution_{ansatz_type}_alpha{cvar_alpha:.2f}.png"
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Distribution evolution figure saved to %s", out_path)
