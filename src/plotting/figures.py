"""
Publication-quality figures for Law-as-Code AIF paper.

Designed for two-column academic format (~3.5in column, ~7in full width).
All outputs: 300 DPI PDF, serif font, minimal chart junk.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# ── Professional styling ──
plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.4,
})

PAL = sns.color_palette("colorblind", 8)
C_BASE = PAL[7]   # gray
C_RULE = PAL[1]   # orange
C_AIF  = PAL[0]   # blue
C_EMRG = PAL[3]   # red
C_PRIV = PAL[2]   # green
C_NORM = PAL[0]   # blue
C_KEY  = PAL[4]   # purple
C_EPIS = PAL[5]   # brown
C_RAND = PAL[7]   # gray


def plot_fig1_minimization(results, save_path=None):
    """Fig 1: Biometric exposure and tracking accuracy.

    Two-panel grouped bar chart across 3 conditions × 4 bystander densities.
    """
    densities = sorted(set(r["bystander_density"] for r in results))
    conditions = ["baseline", "rule_based", "aif_lal"]
    colors = [C_BASE, C_RULE, C_AIF]
    labels = ["No LAL", "Rule-based LAL", "AIF-LAL"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 4.5))
    x = np.arange(len(densities))
    width = 0.25

    for i, (cond, label, color) in enumerate(zip(conditions, labels, colors)):
        cond_data = sorted(
            [r for r in results if r["condition"] == cond],
            key=lambda r: r["bystander_density"],
        )
        exp = [r["exposure_ratio_mean"] for r in cond_data]
        exp_e = [r["exposure_ratio_std"] for r in cond_data]
        trk = [r["tracking_acc_mean"] for r in cond_data]
        trk_e = [r["tracking_acc_std"] for r in cond_data]

        ax1.bar(x + i * width, exp, width, yerr=exp_e, label=label,
                color=color, capsize=2, edgecolor="white", linewidth=0.5)
        ax2.bar(x + i * width, trk, width, yerr=trk_e, label=label,
                color=color, capsize=2, edgecolor="white", linewidth=0.5)

    for ax, ylabel, loc in [
        (ax1, "Biometric exposure ratio", "upper left"),
        (ax2, "Tracking accuracy", "lower left"),
    ]:
        ax.set_xlabel("Bystander density")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x + width)
        ax.set_xticklabels(densities)
        ax.set_ylim(0, 1.08)
        ax.legend(loc=loc, framealpha=0.9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"  Fig 1 saved to {save_path}")
    return fig


def plot_fig2_geofence(results, save_path=None):
    """Fig 2: Geofence violations and tracking over time."""
    conditions = ["pid_only", "rule_based", "aif_lal"]
    labels = ["PID-only", "Rule-based", "AIF-LAL"]
    colors = [C_BASE, C_RULE, C_AIF]
    linestyles = ["-", "--", "-"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 3.8), sharex=True)

    T = None
    for cond, label, color, ls in zip(conditions, labels, colors, linestyles):
        data = [r for r in results if r["condition"] == cond][0]
        viol = data["violations_over_time"]
        track = data["tracking_over_time"]
        T = len(viol)
        t = np.arange(T)
        lw = 2.0 if cond == "aif_lal" else 1.2
        ax1.plot(t, viol, ls=ls, color=color, label=label, lw=lw)
        ax2.plot(t, track, ls=ls, color=color, label=label, lw=lw)

    if T:
        for ax in (ax1, ax2):
            ax.axvspan(T * 2 / 3, T, alpha=0.1, color="red", zorder=0)
        ax1.text(T * 5 / 6, 0.82, "Restricted\nairspace",
                 fontsize=6, color="red", ha="center", style="italic", alpha=0.7)

    ax1.set_ylabel("Violation rate")
    ax1.set_ylim(-0.02, 1.05)
    ax1.legend(loc="upper left", framealpha=0.9)

    ax2.set_ylabel("Tracking continuity")
    ax2.set_xlabel("Timestep")
    ax2.set_ylim(-0.02, 1.05)
    ax2.legend(loc="lower left", framealpha=0.9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"  Fig 2 saved to {save_path}")
    return fig


def plot_fig3_emergency(results, save_path=None):
    """Fig 3: Emergency override — key conditions C1, C2, C7.

    Full-width 3×3 grid.
    """
    c1 = [r for r in results if r["condition"] == 1]
    c2 = [r for r in results if r["condition"] == 2]
    c7 = [r for r in results if r["condition"] == 7]
    if not (c1 and c2 and c7):
        return None

    trials = [c1[0], c2[0], c7[0]]
    titles = [
        "C1: Normal urgency,\nprivacy active",
        "C2: Emergency,\nprivacy active",
        "C7: Emergency at $t$=4,\nprivacy active \u2192 susp.",
    ]

    fig, axes = plt.subplots(3, 3, figsize=(7, 5), sharex=True)

    for col, (trial, title) in enumerate(zip(trials, titles)):
        T = len(trial["positions"])
        t = np.arange(T)

        # Row 0: position trajectory
        ax = axes[0, col]
        ax.step(t, trial["positions"], where="mid", color=C_AIF, lw=2)
        ax.axhspan(1.5, 2.5, alpha=0.12, color="red", zorder=0)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(["PAT", "APP", "PRV", "TGT"], fontsize=7)
        ax.set_ylim(-0.3, 3.5)
        ax.set_title(title, fontsize=7.5, pad=4)
        if col == 0:
            ax.set_ylabel("Position")
            ax.text(0.65, 0.58, "privacy\nzone", transform=ax.transAxes,
                    fontsize=5.5, color="red", alpha=0.6, ha="center",
                    va="center", style="italic")

        # Row 1: beliefs
        ax = axes[1, col]
        urg = [b[1] for b in trial["beliefs_urgency"]]
        priv = [b[0] for b in trial["beliefs_privacy"]]
        ax.plot(t, urg, color=C_EMRG, lw=1.5, label="P(emergency)")
        ax.plot(t, priv, color=C_PRIV, lw=1.5, ls="--", label="P(privacy active)")
        ax.set_ylim(-0.05, 1.1)
        if col == 0:
            ax.set_ylabel("Belief")
            ax.legend(loc="center left", fontsize=5.5, framealpha=0.9)

        if trial["condition"] == 7:
            ax.axvline(x=4, color=C_EMRG, ls=":", alpha=0.6, lw=0.8)
            ax.axvline(x=7, color=C_PRIV, ls=":", alpha=0.6, lw=0.8)
            ax.text(4.2, 0.08, "emerg.", fontsize=5, color=C_EMRG, alpha=0.7)
            ax.text(7.2, 0.08, "priv.\u2193", fontsize=5, color=C_PRIV, alpha=0.7)

        # Row 2: policy precision
        ax = axes[2, col]
        ax.fill_between(t, trial["gamma"], alpha=0.2, color=C_AIF)
        ax.plot(t, trial["gamma"], color=C_AIF, lw=1.5)
        ax.set_ylim(-0.05, 1.1)
        ax.set_xlabel("Timestep")
        if col == 0:
            ax.set_ylabel("Policy precision")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"  Fig 3 saved to {save_path}")
    return fig


def plot_fig4_summary(results, save_path=None):
    """Fig 4: Summary metrics across all 7 conditions."""
    conds = [1, 2, 3, 4, 5, 6, 7]
    labels = [f"C{c}" for c in conds]

    is_emergency = [False, True, False, True, False, True, True]
    bar_colors = [C_EMRG if e else C_NORM for e in is_emergency]
    bar_colors[6] = C_KEY

    violation_rates = []
    success_rates = []
    for c in conds:
        ct = [r for r in results if r["condition"] == c]
        if not ct:
            violation_rates.append(0)
            success_rates.append(0)
            continue
        nv = sum(
            1 for trial in ct
            if any(
                ts[0] == 2 and ps == 0
                for ts, ps in zip(trial["true_states"], trial["privacy_schedule"])
            )
        )
        violation_rates.append(nv / len(ct))
        ns = sum(
            1 for trial in ct if any(ts[0] == 3 for ts in trial["true_states"])
        )
        success_rates.append(ns / len(ct))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 4.5))
    x = np.arange(len(conds))

    ax1.bar(x, violation_rates, color=bar_colors, edgecolor="white", lw=0.5)
    ax1.set_ylabel("Privacy violation rate")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 1.1)

    ax2.bar(x, success_rates, color=bar_colors, edgecolor="white", lw=0.5)
    ax2.set_ylabel("Mission success rate")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0, 1.1)

    legend_elements = [
        Patch(facecolor=C_NORM, label="Normal urgency"),
        Patch(facecolor=C_EMRG, label="Emergency"),
        Patch(facecolor=C_KEY, label="C7: key test"),
    ]
    ax2.legend(handles=legend_elements, loc="lower right",
               framealpha=0.9, fontsize=7)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"  Fig 4 saved to {save_path}")
    return fig


def plot_fig5_overhead(results, save_path=None):
    """Fig 5: Computational overhead per decision cycle."""
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    aif_time = results.get("aif_projected_ms", results["aif_mean_ms"])
    comp_labels = ["PID\ncontrol", "AIF\ninference", "CIL\nlogging", "Total\nLAL"]
    times = [results["pid_mean_ms"], aif_time, results["log_mean_ms"]]
    total = sum(times)
    all_times = times + [total]
    colors = [C_BASE, C_AIF, C_PRIV, PAL[4]]

    bars = ax.bar(range(len(comp_labels)), all_times, color=colors,
                  edgecolor="white", lw=0.5, width=0.6)

    ax.axhline(y=100, color="red", ls="--", lw=1.5, alpha=0.6)
    ax.text(3.4, 102, "100 ms budget", fontsize=6, color="red",
            alpha=0.7, ha="right", va="bottom")

    for bar, t in zip(bars, all_times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{t:.1f} ms", ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("Time (ms)")
    ax.set_xticks(range(len(comp_labels)))
    ax.set_xticklabels(comp_labels, fontsize=7)
    ax.set_ylim(0, max(120, total * 1.5))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"  Fig 5 saved to {save_path}")
    return fig


# ── New experiment figures ──

def plot_fig_ablation(ablation_results, save_path=None):
    """EFE Ablation: success rate, violation rate, C_eff tracking error.

    4-panel figure:
    (a) Success rate by EFE config × condition
    (b) Violation rate by EFE config × condition
    (c) C_eff tracking error over time for C7
    (d) Timesteps at APPROACH before advancing
    """
    configs = ["FULL", "PRAGMATIC_ONLY", "EPISTEMIC_ONLY", "RANDOM"]
    conditions = [1, 5, 7]
    colors = [C_AIF, C_RULE, C_EPIS, C_RAND]
    config_labels = ["Full EFE", "Pragmatic", "Epistemic", "Random"]

    fig, axes = plt.subplots(2, 2, figsize=(7, 5))

    # (a) Success rate
    ax = axes[0, 0]
    x = np.arange(len(conditions))
    width = 0.18
    for i, (cfg, label, color) in enumerate(zip(configs, config_labels, colors)):
        rates = []
        ci_low = []
        ci_high = []
        for cond in conditions:
            cr = [r for r in ablation_results
                  if r["config"] == cfg and r["condition"] == cond]
            vals = [r["reached_target"] for r in cr]
            mean = np.mean(vals) if vals else 0
            rates.append(mean)
            ci_low.append(max(0, mean - 1.96 * np.std(vals) / max(np.sqrt(len(vals)), 1)))
            ci_high.append(min(1, mean + 1.96 * np.std(vals) / max(np.sqrt(len(vals)), 1)))
        ax.bar(x + i * width, rates, width, label=label, color=color,
               edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Success rate")
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([f"C{c}" for c in conditions])
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=6, loc="upper left")
    ax.set_title("(a) Mission success", fontsize=8)

    # (b) Violation rate
    ax = axes[0, 1]
    for i, (cfg, label, color) in enumerate(zip(configs, config_labels, colors)):
        rates = []
        for cond in conditions:
            cr = [r for r in ablation_results
                  if r["config"] == cfg and r["condition"] == cond]
            vals = [r["violated_privacy"] for r in cr]
            rates.append(np.mean(vals) if vals else 0)
        ax.bar(x + i * width, rates, width, label=label, color=color,
               edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Violation rate")
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([f"C{c}" for c in conditions])
    ax.set_ylim(0, 1.1)
    ax.set_title("(b) Privacy violations", fontsize=8)

    # (c) C_eff tracking error over time for C7
    ax = axes[1, 0]
    for cfg, label, color in zip(configs, config_labels, colors):
        cr = [r for r in ablation_results
              if r["config"] == cfg and r["condition"] == 7]
        if cr:
            errors = np.array([r["c_eff_tracking_errors"] for r in cr])
            mean_err = errors.mean(axis=0)
            std_err = errors.std(axis=0)
            t = np.arange(len(mean_err))
            ax.plot(t, mean_err, label=label, color=color, lw=1.5)
            ax.fill_between(t, mean_err - std_err, mean_err + std_err,
                            alpha=0.15, color=color)
    ax.set_ylabel("C_eff tracking error")
    ax.set_xlabel("Timestep")
    ax.legend(fontsize=6)
    ax.set_title("(c) Preference convergence (C7)", fontsize=8)

    # (d) Timesteps at APPROACH
    ax = axes[1, 1]
    for i, (cfg, label, color) in enumerate(zip(configs, config_labels, colors)):
        rates = []
        for cond in conditions:
            cr = [r for r in ablation_results
                  if r["config"] == cfg and r["condition"] == cond]
            vals = [r["timesteps_at_approach"] for r in cr]
            rates.append(np.mean(vals) if vals else 0)
        ax.bar(x + i * width, rates, width, label=label, color=color,
               edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Timesteps at APPROACH")
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([f"C{c}" for c in conditions])
    ax.set_title("(d) Information gathering", fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"  Ablation figure saved to {save_path}")
    return fig


def plot_fig_baselines(baseline_results, save_path=None):
    """Baselines comparison: AIF vs HPM_ORACLE vs HPM_NOISY vs BAYES_RULES.

    2-panel grouped bar across all 7 conditions.
    """
    agent_types = ["AIF", "HPM_ORACLE", "HPM_NOISY", "BAYES_RULES"]
    agent_labels = ["AIF", "HPM Oracle", "HPM Noisy", "Bayes+Rules"]
    colors = [C_AIF, C_PRIV, C_RULE, C_EPIS]
    conditions = [1, 2, 3, 4, 5, 6, 7]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5))
    x = np.arange(len(conditions))
    width = 0.18

    for i, (at, label, color) in enumerate(zip(agent_types, agent_labels, colors)):
        success_rates = []
        violation_rates = []
        for cond in conditions:
            cr = [r for r in baseline_results
                  if r["agent_type"] == at and r["condition"] == cond]
            success_rates.append(np.mean([r["reached_target"] for r in cr]) if cr else 0)
            violation_rates.append(np.mean([r["violated_privacy"] for r in cr]) if cr else 0)

        ax1.bar(x + i * width, success_rates, width, label=label, color=color,
                edgecolor="white", linewidth=0.5)
        ax2.bar(x + i * width, violation_rates, width, label=label, color=color,
                edgecolor="white", linewidth=0.5)

    ax1.set_ylabel("Success rate")
    ax1.set_xticks(x + 1.5 * width)
    ax1.set_xticklabels([f"C{c}" for c in conditions])
    ax1.set_ylim(0, 1.1)
    ax1.legend(fontsize=6, loc="lower right")
    ax1.set_title("(a) Mission success by agent and condition", fontsize=8)

    ax2.set_ylabel("Violation rate")
    ax2.set_xticks(x + 1.5 * width)
    ax2.set_xticklabels([f"C{c}" for c in conditions])
    ax2.set_ylim(0, 1.1)
    ax2.set_xlabel("Condition")
    ax2.set_title("(b) Privacy violations by agent and condition", fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"  Baselines figure saved to {save_path}")
    return fig


def plot_fig_sensitivity(sensitivity_results, save_path=None):
    """Sensitivity heatmap: override score across parameter space.

    Heatmap + 1D slices.
    """
    etds = sorted(set(v["emergency_target_drive"] for v in sensitivity_results.values()))
    ncas = sorted(set(v["normal_complaint_aversion"] for v in sensitivity_results.values()))

    # Build heatmap matrix
    score_matrix = np.zeros((len(ncas), len(etds)))
    for v in sensitivity_results.values():
        i = ncas.index(v["normal_complaint_aversion"])
        j = etds.index(v["emergency_target_drive"])
        score_matrix[i, j] = v["override_score"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3), gridspec_kw={"width_ratios": [2, 1]})

    # Heatmap
    im = ax1.imshow(score_matrix, cmap="RdBu_r", vmin=-1, vmax=1,
                     aspect="auto", origin="lower")
    ax1.set_xticks(range(len(etds)))
    ax1.set_xticklabels([f"{e:.0f}" for e in etds])
    ax1.set_yticks(range(len(ncas)))
    ax1.set_yticklabels([f"{n:.0f}" for n in ncas])
    ax1.set_xlabel("Emergency target drive")
    ax1.set_ylabel("Normal complaint aversion")
    ax1.set_title("(a) Override score heatmap", fontsize=8)
    plt.colorbar(im, ax=ax1, label="P(cross|emerg) - P(cross|norm)")

    # 1D slice at NCA=6
    nca_idx = ncas.index(6.0) if 6.0 in ncas else len(ncas) // 2
    ax2.plot(etds, score_matrix[nca_idx, :], "o-", color=C_AIF, lw=2, markersize=5)
    ax2.axhline(y=0, color="gray", ls="--", lw=0.8)
    ax2.set_xlabel("Emergency target drive")
    ax2.set_ylabel("Override score")
    ax2.set_title(f"(b) Slice at NCA={ncas[nca_idx]:.0f}", fontsize=8)
    ax2.set_ylim(-0.2, 1.1)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"  Sensitivity figure saved to {save_path}")
    return fig


def plot_fig_noise(noise_results, save_path=None):
    """Noise robustness: success and violation vs noise level.

    2-panel line plot with 3 agents × 2 conditions.
    """
    noise_levels = sorted([float(k) for k in noise_results.keys()])
    agent_types = ["AIF", "HPM_NOISY", "BAYES_RULES"]
    agent_labels = ["AIF", "HPM Noisy", "Bayes+Rules"]
    colors = [C_AIF, C_RULE, C_EPIS]

    fig, axes = plt.subplots(2, 2, figsize=(7, 5))

    for col, cond in enumerate([1, 7]):
        # Success rate
        ax = axes[0, col]
        for at, label, color in zip(agent_types, agent_labels, colors):
            rates = []
            ci_lo = []
            ci_hi = []
            for noise in noise_levels:
                entry = noise_results[noise][at][cond]
                sr = entry["success_rate"]
                rates.append(sr[0])
                ci_lo.append(sr[1])
                ci_hi.append(sr[2])
            ax.plot(noise_levels, rates, "o-", color=color, label=label, lw=1.5, markersize=4)
            ax.fill_between(noise_levels, ci_lo, ci_hi, alpha=0.1, color=color)
        ax.set_ylabel("Success rate")
        ax.set_title(f"C{cond}: Success vs noise", fontsize=8)
        ax.set_ylim(-0.05, 1.1)
        if col == 0:
            ax.legend(fontsize=6)

        # Violation rate (false positive for C1)
        ax = axes[1, col]
        for at, label, color in zip(agent_types, agent_labels, colors):
            rates = []
            ci_lo = []
            ci_hi = []
            for noise in noise_levels:
                entry = noise_results[noise][at][cond]
                vr = entry["violation_rate"]
                rates.append(vr[0])
                ci_lo.append(vr[1])
                ci_hi.append(vr[2])
            ax.plot(noise_levels, rates, "o-", color=color, label=label, lw=1.5, markersize=4)
            ax.fill_between(noise_levels, ci_lo, ci_hi, alpha=0.1, color=color)
        ax.set_ylabel("Violation rate")
        ax.set_xlabel("Observation noise")
        ax.set_title(f"C{cond}: Violations vs noise", fontsize=8)
        ax.set_ylim(-0.05, 1.1)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"  Noise figure saved to {save_path}")
    return fig


def plot_fig_learning(learning_results, learning_curve, save_path=None):
    """Learning experiment: learning curve + performance table.

    2-panel: (a) learning curve, (b) test performance comparison.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

    # (a) Learning curve
    t = np.arange(len(learning_curve))
    ax1.plot(t, learning_curve, color=C_AIF, lw=1.5)
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("L1(true A, learned A)")
    ax1.set_title("(a) A-matrix learning curve", fontsize=8)

    # (b) Test performance
    configs = ["ORACLE", "LEARNED", "MISSPECIFIED"]
    colors_bar = [C_PRIV, C_AIF, C_EMRG]
    conditions = [1, 2, 3, 4, 5, 6, 7]
    x = np.arange(len(conditions))
    width = 0.25

    for i, (cfg, color) in enumerate(zip(configs, colors_bar)):
        if cfg in learning_results:
            rates = []
            for cond in conditions:
                if cond in learning_results[cfg]:
                    rates.append(learning_results[cfg][cond]["success_rate"][0])
                else:
                    rates.append(0)
            ax2.bar(x + i * width, rates, width, label=cfg, color=color,
                    edgecolor="white", linewidth=0.5)

    ax2.set_ylabel("Success rate")
    ax2.set_xticks(x + width)
    ax2.set_xticklabels([f"C{c}" for c in conditions])
    ax2.set_ylim(0, 1.1)
    ax2.legend(fontsize=6)
    ax2.set_title("(b) Test performance by A-matrix source", fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"  Learning figure saved to {save_path}")
    return fig
