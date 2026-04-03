"""
Publication-quality figures for Law-as-Code AIF paper.

All figures: 300 DPI, PDF output, matplotlib + seaborn.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid", font_scale=1.1)
COLORS = sns.color_palette("Set2", 8)


def plot_fig1_minimization(results, save_path=None):
    """Fig 1: Grouped bar chart — exposure ratio × tracking accuracy.

    3 conditions × 4 bystander densities.
    """
    densities = sorted(set(r["bystander_density"] for r in results))
    conditions = ["baseline", "rule_based", "aif_lal"]
    cond_labels = ["Baseline\n(No LAL)", "Rule-Based\nLAL", "AIF-LAL"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(densities))
    width = 0.25

    for i, (cond, label) in enumerate(zip(conditions, cond_labels)):
        cond_data = [r for r in results if r["condition"] == cond]
        cond_data.sort(key=lambda r: r["bystander_density"])

        exposure = [r["exposure_ratio_mean"] for r in cond_data]
        exposure_err = [r["exposure_ratio_std"] for r in cond_data]
        tracking = [r["tracking_acc_mean"] for r in cond_data]
        tracking_err = [r["tracking_acc_std"] for r in cond_data]

        ax1.bar(x + i * width, exposure, width, yerr=exposure_err,
                label=label, color=COLORS[i], capsize=3, alpha=0.85)
        ax2.bar(x + i * width, tracking, width, yerr=tracking_err,
                label=label, color=COLORS[i], capsize=3, alpha=0.85)

    ax1.set_xlabel("Bystander Density")
    ax1.set_ylabel("Biometric Exposure Ratio")
    ax1.set_title("(a) Data Exposure")
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(densities)
    ax1.legend(fontsize=9)
    ax1.set_ylim(0, 1.1)

    ax2.set_xlabel("Bystander Density")
    ax2.set_ylabel("Tracking Accuracy")
    ax2.set_title("(b) Target Tracking")
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(densities)
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 1.1)

    fig.suptitle("Experiment 1: GDPR Data Minimization — Exposure vs. Tracking", fontsize=13)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Fig 1 saved to {save_path}")
    return fig


def plot_fig2_geofence(results, save_path=None):
    """Fig 2: Dual-axis time series — violations and tracking over time."""
    fig, ax1 = plt.subplots(figsize=(10, 5))

    conditions = ["pid_only", "rule_based", "aif_lal"]
    cond_labels = ["PID-Only", "Rule-Based", "AIF-LAL"]
    linestyles = ["-", "--", "-"]
    markers = ["o", "s", "D"]

    T = None
    for i, (cond, label) in enumerate(zip(conditions, cond_labels)):
        cond_data = [r for r in results if r["condition"] == cond][0]
        viol = cond_data["violations_over_time"]
        track = cond_data["tracking_over_time"]
        T = len(viol)
        t = np.arange(T)

        ax1.plot(t, viol, linestyle=linestyles[i], marker=markers[i],
                 color=COLORS[i], label=f"{label} (violations)",
                 markevery=3, markersize=5, alpha=0.85)

    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Geofence Violation Rate")
    ax1.set_ylim(-0.05, 1.1)

    # Shaded restricted zone region
    if T:
        ax1.axvspan(T * 2 / 3, T, alpha=0.15, color="red", label="Restricted Zone Active")

    ax2 = ax1.twinx()
    for i, (cond, label) in enumerate(zip(conditions, cond_labels)):
        cond_data = [r for r in results if r["condition"] == cond][0]
        track = cond_data["tracking_over_time"]
        t = np.arange(len(track))
        ax2.plot(t, track, linestyle=":", marker=markers[i],
                 color=COLORS[i + 3], label=f"{label} (tracking)",
                 markevery=3, markersize=4, alpha=0.7)

    ax2.set_ylabel("Tracking Continuity")
    ax2.set_ylim(-0.05, 1.1)

    # Combine legends
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)

    fig.suptitle("Experiment 2: EASA Geofence Compliance", fontsize=13)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Fig 2 saved to {save_path}")
    return fig


def plot_fig3_emergency(results, save_path_prefix=None):
    """Fig 3a-d: Emergency override — state inference, actions, gamma.

    3a: C1 vs C2 (privacy effect under different urgency)
    3b: C3 vs C4 (suspended privacy)
    3c: C5, C6, C7 (switching conditions)
    3d: Summary bar chart across 7 conditions
    """
    figs = {}

    # ── Helper to plot a single condition panel ──
    def plot_condition_panel(ax_row, trial_data, cond_label):
        """Plot 5 subplots for one condition: position, action, obs, belief, gamma."""
        T = len(trial_data["positions"])
        t = np.arange(T)

        # Position
        ax_row[0].step(t, trial_data["positions"], where="mid", color=COLORS[0], linewidth=2)
        ax_row[0].set_ylabel("Position")
        ax_row[0].set_yticks([0, 1, 2, 3])
        ax_row[0].set_yticklabels(["PAT", "APP", "PRV", "TGT"], fontsize=7)
        ax_row[0].set_title(cond_label, fontsize=10)

        # Actions
        ax_row[1].step(t, trial_data["actions"], where="mid", color=COLORS[1], linewidth=2)
        ax_row[1].set_ylabel("Action")
        ax_row[1].set_yticks([0, 1])
        ax_row[1].set_yticklabels(["HOLD", "ADV"], fontsize=7)

        # Urgency belief
        urg = [b[1] for b in trial_data["beliefs_urgency"]]
        ax_row[2].plot(t, urg, color=COLORS[3], linewidth=2, label="P(emergency)")
        ax_row[2].set_ylabel("P(emerg)")
        ax_row[2].set_ylim(-0.05, 1.05)
        ax_row[2].legend(fontsize=7)

        # Privacy belief
        priv = [b[0] for b in trial_data["beliefs_privacy"]]
        ax_row[3].plot(t, priv, color=COLORS[4], linewidth=2, label="P(active)")
        ax_row[3].set_ylabel("P(priv)")
        ax_row[3].set_ylim(-0.05, 1.05)
        ax_row[3].legend(fontsize=7)

        # Gamma (precision)
        ax_row[4].plot(t, trial_data["gamma"], color=COLORS[5], linewidth=2)
        ax_row[4].set_ylabel("Precision")
        ax_row[4].set_xlabel("Timestep")
        ax_row[4].set_ylim(-0.05, 1.05)

    # ── Fig 3a: C1 vs C2 ──
    c1_trials = [r for r in results if r["condition"] == 1]
    c2_trials = [r for r in results if r["condition"] == 2]

    if c1_trials and c2_trials:
        fig3a, axes = plt.subplots(5, 2, figsize=(12, 10), sharex=True)
        plot_condition_panel([axes[i, 0] for i in range(5)], c1_trials[0], "C1: Active + Normal")
        plot_condition_panel([axes[i, 1] for i in range(5)], c2_trials[0], "C2: Active + Emergency")
        fig3a.suptitle("Fig 3a: Privacy Active — Normal vs Emergency", fontsize=13)
        plt.tight_layout()
        if save_path_prefix:
            fig3a.savefig(f"{save_path_prefix}_3a.pdf", dpi=300, bbox_inches="tight")
            print(f"  Fig 3a saved")
        figs["3a"] = fig3a

    # ── Fig 3b: C3 vs C4 ──
    c3_trials = [r for r in results if r["condition"] == 3]
    c4_trials = [r for r in results if r["condition"] == 4]

    if c3_trials and c4_trials:
        fig3b, axes = plt.subplots(5, 2, figsize=(12, 10), sharex=True)
        plot_condition_panel([axes[i, 0] for i in range(5)], c3_trials[0], "C3: Suspended + Normal")
        plot_condition_panel([axes[i, 1] for i in range(5)], c4_trials[0], "C4: Suspended + Emergency")
        fig3b.suptitle("Fig 3b: Privacy Suspended — Normal vs Emergency", fontsize=13)
        plt.tight_layout()
        if save_path_prefix:
            fig3b.savefig(f"{save_path_prefix}_3b.pdf", dpi=300, bbox_inches="tight")
            print(f"  Fig 3b saved")
        figs["3b"] = fig3b

    # ── Fig 3c: C5, C6, C7 ──
    c5_trials = [r for r in results if r["condition"] == 5]
    c6_trials = [r for r in results if r["condition"] == 6]
    c7_trials = [r for r in results if r["condition"] == 7]

    if c5_trials and c6_trials and c7_trials:
        fig3c, axes = plt.subplots(5, 3, figsize=(16, 10), sharex=True)
        plot_condition_panel([axes[i, 0] for i in range(5)], c5_trials[0], "C5: Switch + Normal")
        plot_condition_panel([axes[i, 1] for i in range(5)], c6_trials[0], "C6: Switch + Emergency")
        plot_condition_panel([axes[i, 2] for i in range(5)], c7_trials[0], "C7: Switch + Switch")
        fig3c.suptitle("Fig 3c: Context Switching Conditions", fontsize=13)
        plt.tight_layout()
        if save_path_prefix:
            fig3c.savefig(f"{save_path_prefix}_3c.pdf", dpi=300, bbox_inches="tight")
            print(f"  Fig 3c saved")
        figs["3c"] = fig3c

    # ── Fig 3d: Summary bar chart ──
    fig3d, axes = plt.subplots(1, 3, figsize=(14, 4))
    conds = [1, 2, 3, 4, 5, 6, 7]
    cond_labels = [f"C{c}" for c in conds]

    # Compute per-condition metrics
    violation_rates = []
    success_rates = []
    mean_gammas = []

    for c in conds:
        c_trials = [r for r in results if r["condition"] == c]
        if not c_trials:
            violation_rates.append(0)
            success_rates.append(0)
            mean_gammas.append(0)
            continue

        # Violation: entered PRIVACY_ZONE while privacy ACTIVE
        n_viol = sum(
            1 for trial in c_trials
            if any(ts[0] == 2 and ps == 0
                   for ts, ps in zip(trial["true_states"], trial["privacy_schedule"]))
        )
        violation_rates.append(n_viol / len(c_trials))

        # Success: reached TARGET
        n_succ = sum(
            1 for trial in c_trials
            if any(ts[0] == 3 for ts in trial["true_states"])
        )
        success_rates.append(n_succ / len(c_trials))

        # Mean gamma
        all_gamma = [g for trial in c_trials for g in trial["gamma"]]
        mean_gammas.append(np.mean(all_gamma) if all_gamma else 0)

    x = np.arange(len(conds))

    axes[0].bar(x, violation_rates, color=COLORS[0], alpha=0.85)
    axes[0].set_ylabel("Privacy Violation Rate")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(cond_labels)
    axes[0].set_ylim(0, 1.1)
    axes[0].set_title("(a) Violations")

    axes[1].bar(x, success_rates, color=COLORS[1], alpha=0.85)
    axes[1].set_ylabel("Mission Success Rate")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(cond_labels)
    axes[1].set_ylim(0, 1.1)
    axes[1].set_title("(b) Success")

    axes[2].bar(x, mean_gammas, color=COLORS[2], alpha=0.85)
    axes[2].set_ylabel("Mean Policy Precision")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(cond_labels)
    axes[2].set_ylim(0, 1.1)
    axes[2].set_title("(c) Precision")

    fig3d.suptitle("Fig 3d: Summary — 7 Conditions", fontsize=13)
    plt.tight_layout()
    if save_path_prefix:
        fig3d.savefig(f"{save_path_prefix}_3d.pdf", dpi=300, bbox_inches="tight")
        print(f"  Fig 3d saved")
    figs["3d"] = fig3d

    return figs


def plot_fig4_traceability(results, save_path=None):
    """Fig 4: Stacked bar chart — time budget breakdown.

    Shows projected (scan-compiled) AIF timing for realistic deployment.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Use projected AIF time for the main visualization
    aif_time = results.get("aif_projected_ms", results["aif_mean_ms"])

    components = ["PID Control", "AIF Inference\n(scan-compiled)", "CIL Logging"]
    times = [results["pid_mean_ms"], aif_time, results["log_mean_ms"]]
    colors_bar = [COLORS[0], COLORS[1], COLORS[2]]

    # Individual bars
    x = np.arange(len(components))
    bars = ax.bar(x, times, color=colors_bar, alpha=0.85, width=0.5)

    # 100ms budget line
    ax.axhline(y=100, color="red", linestyle="--", linewidth=2, label="100ms Control Budget")

    # Stacked total bar
    total = sum(times)
    bottom = 0
    for i, (comp, t) in enumerate(zip(components, times)):
        ax.bar([len(components)], [t], bottom=[bottom], color=colors_bar[i], alpha=0.7, width=0.5)
        bottom += t

    ax.set_xticks(list(x) + [len(components)])
    ax.set_xticklabels([c for c in components] + ["Total LAL\nPipeline"])
    ax.set_ylabel("Time (ms)")
    ax.set_title("Experiment 4: LAL Computational Overhead")
    ax.legend(loc="upper right")

    # Annotate total
    pct = total / 100 * 100
    ax.annotate(f"{total:.1f}ms\n({pct:.0f}% of budget)",
                xy=(len(components), total), xytext=(len(components) + 0.3, total + 5),
                fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="gray"))

    # Add value labels on bars
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{t:.2f}ms", ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0, max(120, total * 1.3))
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Fig 4 saved to {save_path}")
    return fig
