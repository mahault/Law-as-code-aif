"""
Statistical utilities for experiment analysis.

Bootstrap confidence intervals, Mann-Whitney U, Cohen's d.
"""

import numpy as np
from scipy import stats as sp_stats


def bootstrap_ci(data, n_boot=10000, ci=0.95, statistic=np.mean, seed=42):
    """Compute bootstrap confidence interval.

    Args:
        data: 1-D array of observations
        n_boot: number of bootstrap resamples
        ci: confidence level (e.g. 0.95 for 95% CI)
        statistic: function to apply to each resample
        seed: random seed

    Returns:
        (point_estimate, ci_low, ci_high)
    """
    data = np.asarray(data)
    rng = np.random.RandomState(seed)
    n = len(data)
    boot_stats = np.array([
        statistic(rng.choice(data, size=n, replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    ci_low = np.percentile(boot_stats, 100 * alpha)
    ci_high = np.percentile(boot_stats, 100 * (1 - alpha))
    return float(statistic(data)), float(ci_low), float(ci_high)


def mann_whitney_u(x, y, alternative="two-sided"):
    """Mann-Whitney U test for two independent samples.

    Args:
        x, y: 1-D arrays
        alternative: 'two-sided', 'less', or 'greater'

    Returns:
        (U_statistic, p_value)
    """
    result = sp_stats.mannwhitneyu(x, y, alternative=alternative)
    return float(result.statistic), float(result.pvalue)


def cohens_d(x, y):
    """Cohen's d effect size for two independent samples.

    Uses pooled standard deviation.

    Args:
        x, y: 1-D arrays

    Returns:
        d: float — Cohen's d
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(
        ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1))
        / (nx + ny - 2)
    )
    if pooled_std < 1e-12:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / pooled_std)
