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


def plot_fig1_minimization(results, save_path=None):
    """Fig 1: Biometric exposure and tracking accuracy.

    Two-panel grouped bar chart across 3 conditions × 4 bystander densities.
    Shows AIF-LAL achieves near-zero exposure without tracking loss.
    """
    densities = sorted(set(r["bystander_density"] for r in results))
    conditions = ["baseline", "rule_based", "aif_lal"]
    colors = [C_BASE, C_RULE, C_AIF]
    labels = ["No LAL", "Rule-based LAL", "AIF-LAL"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))
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

    for ax, ylabel, panel, loc in [
        (ax1, "Biometric exposure ratio", "(a)", "upper left"),
        (ax2, "Tracking accuracy", "(b)", "lower left"),
    ]:
        ax.set_xlabel("Bystander density")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x + width)
        ax.set_xticklabels(densities)
        ax.set_ylim(0, 1.08)
        ax.text(0.03, 0.95, panel, transform=ax.transAxes,
                fontweight="bold", va="top", fontsize=10)
        ax.legend(loc=loc, framealpha=0.9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"  Fig 1 saved to {save_path}")
    return fig


def plot_fig2_geofence(results, save_path=None):
    """Fig 2: Geofence violations and tracking over time.

    Two vertically-stacked panels sharing the x-axis. Shaded region marks
    restricted airspace activation. Clearer than dual-axis overlay.
    """
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
    ax1.text(0.03, 0.92, "(a)", transform=ax1.transAxes,
             fontweight="bold", va="top", fontsize=10)

    ax2.set_ylabel("Tracking continuity")
    ax2.set_xlabel("Timestep")
    ax2.set_ylim(-0.02, 1.05)
    ax2.legend(loc="lower left", framealpha=0.9)
    ax2.text(0.03, 0.92, "(b)", transform=ax2.transAxes,
             fontweight="bold", va="top", fontsize=10)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"  Fig 2 saved to {save_path}")
    return fig


def plot_fig3_emergency(results, save_path=None):
    """Fig 3: Emergency override — key conditions C1, C2, C7.

    Full-width 3×3 grid.
      Columns: C1 (normal), C2 (emergency), C7 (emergency onset at t=4).
      Rows: (a) drone position trajectory with privacy-zone shading,
            (b) inferred beliefs (emergency + privacy),
            (c) policy precision.
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

        # Row 1: beliefs (urgency + privacy)
        ax = axes[1, col]
        urg = [b[1] for b in trial["beliefs_urgency"]]
        priv = [b[0] for b in trial["beliefs_privacy"]]
        ax.plot(t, urg, color=C_EMRG, lw=1.5, label="P(emergency)")
        ax.plot(t, priv, color=C_PRIV, lw=1.5, ls="--", label="P(privacy active)")
        ax.set_ylim(-0.05, 1.1)
        if col == 0:
            ax.set_ylabel("Belief")
            ax.legend(loc="center left", fontsize=5.5, framealpha=0.9)

        # Context-switch markers for C7
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
    """Fig 4: Summary metrics across all 7 conditions.

    Two panels: (a) privacy violation rate, (b) mission success rate.
    Bars colored by condition type: blue=normal, red=emergency, purple=C7.
    """
    conds = [1, 2, 3, 4, 5, 6, 7]
    labels = [f"C{c}" for c in conds]

    is_emergency = [False, True, False, True, False, True, True]
    bar_colors = [C_EMRG if e else C_NORM for e in is_emergency]
    bar_colors[6] = C_KEY  # C7 highlighted

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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))
    x = np.arange(len(conds))

    ax1.bar(x, violation_rates, color=bar_colors, edgecolor="white", lw=0.5)
    ax1.set_ylabel("Privacy violation rate")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 1.1)
    ax1.text(0.03, 0.95, "(a)", transform=ax1.transAxes,
             fontweight="bold", va="top", fontsize=10)

    ax2.bar(x, success_rates, color=bar_colors, edgecolor="white", lw=0.5)
    ax2.set_ylabel("Mission success rate")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0, 1.1)
    ax2.text(0.03, 0.95, "(b)", transform=ax2.transAxes,
             fontweight="bold", va="top", fontsize=10)

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
    """Fig 5: Computational overhead per decision cycle.

    Bar chart: PID, AIF inference, CIL logging, and total LAL pipeline.
    Red dashed line marks the 100ms real-time budget.
    """
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
