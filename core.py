"""A small, tested toolkit for designing and analyzing A/B tests on conversion rates.

Design-first: you compute the required sample size from a minimum detectable effect
BEFORE running the test, analyze with a confidence interval and effect size (not just a
p-value), and the module also ships simulators that demonstrate the two classic ways to
fool yourself, peeking (optional stopping) and multiple comparisons.
"""
from dataclasses import dataclass
import numpy as np
from scipy import stats


def sample_size_two_proportions(p1, mde_abs, alpha=0.05, power=0.80, two_sided=True):
    """Required sample size PER ARM to detect an absolute lift `mde_abs` over baseline `p1`.

    Uses the normal approximation. mde_abs is the minimum detectable effect in absolute
    percentage points (e.g. 0.015 for a 12% -> 13.5% lift).
    """
    p2 = p1 + mde_abs
    z_alpha = stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)
    n = (z_alpha + z_beta) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (mde_abs ** 2)
    return int(np.ceil(n))


@dataclass
class ABResult:
    rate_control: float
    rate_treatment: float
    abs_diff: float           # treatment - control, in absolute rate
    rel_lift: float           # abs_diff / control
    z: float
    p_value: float
    ci_low: float             # 95% CI on the absolute difference
    ci_high: float
    n_control: int
    n_treatment: int

    @property
    def significant(self):
        return self.ci_low > 0 or self.ci_high < 0


def two_proportion_test(conv_c, n_c, conv_t, n_t, alpha=0.05):
    """Two-proportion z-test. Pooled SE for the test statistic, unpooled SE for the CI
    on the difference (the standard, and internally consistent, convention)."""
    p_c, p_t = conv_c / n_c, conv_t / n_t
    diff = p_t - p_c
    p_pool = (conv_c + conv_t) / (n_c + n_t)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
    z = diff / se_pool
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    se_ci = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    zc = stats.norm.ppf(1 - alpha / 2)
    return ABResult(
        rate_control=p_c, rate_treatment=p_t, abs_diff=diff,
        rel_lift=(diff / p_c if p_c else float("nan")),
        z=z, p_value=p_value, ci_low=diff - zc * se_ci, ci_high=diff + zc * se_ci,
        n_control=n_c, n_treatment=n_t,
    )


def cohens_h(p1, p2):
    """Effect size for two proportions (arcsine transform). |h| ~ 0.2 small, 0.5 medium, 0.8 large."""
    return 2 * np.arcsin(np.sqrt(p2)) - 2 * np.arcsin(np.sqrt(p1))


def _sig_on_prefix(c, t, n, alpha):
    """Is the two-proportion test significant using the first n observations of each arm?"""
    xc, xt = c[:n].sum(), t[:n].sum()
    pc, pt = xc / n, xt / n
    pp = (xc + xt) / (2 * n)
    se = np.sqrt(pp * (1 - pp) * (2 / n))
    if se == 0:
        return False
    z = (pt - pc) / se
    return 2 * (1 - stats.norm.cdf(abs(z))) < alpha


def simulate_peeking(p, n_total, n_looks, alpha=0.05, n_sims=2000, seed=0):
    """A/A simulation (no true effect). Compare the false-positive rate of a single
    fixed-n test against 'peek and stop at the first significant look'. Returns
    (fp_fixed, fp_peeking)."""
    rng = np.random.default_rng(seed)
    looks = np.linspace(n_total // n_looks, n_total, n_looks).astype(int)
    fp_fixed = fp_peek = 0
    for _ in range(n_sims):
        c = (rng.random(n_total) < p).astype(np.int64)
        t = (rng.random(n_total) < p).astype(np.int64)
        if _sig_on_prefix(c, t, n_total, alpha):
            fp_fixed += 1
        for lp in looks:
            if _sig_on_prefix(c, t, lp, alpha):
                fp_peek += 1
                break
    return fp_fixed / n_sims, fp_peek / n_sims


def familywise_error(k, alpha=0.05):
    """Probability of at least one false positive across k independent tests, no correction."""
    return 1 - (1 - alpha) ** k


def bonferroni_alpha(k, alpha=0.05):
    """Per-test threshold that controls the family-wise error at `alpha` across k tests."""
    return alpha / k
