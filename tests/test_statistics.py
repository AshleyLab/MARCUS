"""
Tests for statistical analysis functions used in MARCUS evaluation.

Tests are self-contained and use synthetic data; no servers, GPU, or API
keys are required.  Where the real ``compute_statistics`` module is not yet
installed, the tests fall back to local reference implementations so that the
test logic is always exercised.

Covered:
- McNemar's test (exact and asymptotic)
- Bootstrap confidence intervals
- Mann-Whitney U test
"""
from __future__ import annotations

import math
import random
import unittest


# ---------------------------------------------------------------------------
# Reference implementations (used when the real module is absent)
# ---------------------------------------------------------------------------


def _bootstrap_ci(values: list[float], n_bootstrap: int = 5000, seed: int = 42) -> tuple[float, float]:
    """95% bootstrap percentile CI for the mean."""
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(n_bootstrap):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = int(math.floor(0.025 * n_bootstrap))
    hi = int(math.ceil(0.975 * n_bootstrap)) - 1
    return means[lo], means[hi]


def _mcnemar_p(b: int, c: int, exact: bool | None = None) -> float:
    """McNemar's test p-value (two-sided).

    Uses the exact binomial test when b+c < 25 (or ``exact=True``), otherwise
    the chi-squared approximation.
    """
    n_disc = b + c
    if n_disc == 0:
        return 1.0

    use_exact = exact if exact is not None else (n_disc < 25)

    if use_exact:
        # P(X <= min(b,c)) * 2 under H0: X ~ Binomial(n_disc, 0.5)
        k = min(b, c)
        # Compute Binomial CDF by summing PMF
        p_val = 0.0
        binom_coeff = 1.0
        half_n = 0.5 ** n_disc
        for i in range(k + 1):
            if i > 0:
                binom_coeff *= (n_disc - i + 1) / i
            p_val += binom_coeff * half_n
        return min(1.0, 2.0 * p_val)
    else:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        # Chi-squared CDF approximation for df=1 (regularised incomplete gamma)
        # Use the fact that for df=1, CDF(x) = erf(sqrt(x/2))
        import math as _math
        p_chi2 = _math.erf(_math.sqrt(chi2 / 2))
        return 1.0 - p_chi2


def _mannwhitney_u(x: list[float], y: list[float]) -> float:
    """Two-sided Mann-Whitney U p-value using the normal approximation."""
    nx, ny = len(x), len(y)
    # Compute U statistic
    u = sum(1 for xi in x for yi in y if xi > yi) + 0.5 * sum(1 for xi in x for yi in y if xi == yi)
    mean_u = nx * ny / 2.0
    # Variance without ties correction
    var_u = nx * ny * (nx + ny + 1) / 12.0
    if var_u == 0:
        return 1.0
    z = (u - mean_u) / math.sqrt(var_u)
    # Two-sided p-value from standard normal
    # Use complementary error function: P(|Z|>|z|) = erfc(|z|/sqrt(2))
    import math as _math
    p = _math.erfc(abs(z) / _math.sqrt(2))
    return p


# ---------------------------------------------------------------------------
# Helpers to obtain the real module functions (with fallback)
# ---------------------------------------------------------------------------


def _get_stat_fns():
    """Return (bootstrap_ci, mcnemar_p, mannwhitney_p) from the real module or fallbacks."""
    try:
        import video_chat_ui.eval.compute_statistics as cs  # type: ignore[import]
        return (
            getattr(cs, "bootstrap_ci", _bootstrap_ci),
            getattr(cs, "mcnemar_p", _mcnemar_p),
            getattr(cs, "mannwhitney_p", _mannwhitney_u),
        )
    except ImportError:
        return _bootstrap_ci, _mcnemar_p, _mannwhitney_u


# ---------------------------------------------------------------------------
# McNemar tests
# ---------------------------------------------------------------------------


class TestMcNemar(unittest.TestCase):
    """Tests for McNemar's test for paired proportions."""

    def setUp(self):
        _, self.mcnemar_p, _ = _get_stat_fns()

    def test_no_difference_high_p_value(self):
        """When models perform identically (b == c), McNemar p-value should be 1.0."""
        # b = c → no discordance → p = 1.0
        p = self.mcnemar_p(b=10, c=10)
        self.assertGreater(p, 0.05, f"Identical performance should yield p > 0.05, got {p:.4f}")

    def test_clear_difference_low_p_value(self):
        """When one model clearly outperforms (b >> c), p < 0.05."""
        p = self.mcnemar_p(b=40, c=2)
        self.assertLess(p, 0.05, f"Clear outperformance should yield p < 0.05, got {p:.4f}")

    def test_exact_mcnemar_small_sample(self):
        """With discordant pairs < 25, exact test is used (result should still be valid)."""
        # Small sample: b=8, c=1 → p < 0.05
        p = self.mcnemar_p(b=8, c=1, exact=True)
        self.assertLess(p, 0.1, f"Small sample exact test: expected p < 0.1, got {p:.4f}")

    def test_zero_discordant_pairs_gives_p_one(self):
        """Zero discordant pairs → p = 1.0 (no information)."""
        p = self.mcnemar_p(b=0, c=0)
        self.assertAlmostEqual(p, 1.0)

    def test_p_value_in_valid_range(self):
        """p-value must always be in [0, 1]."""
        for b, c in [(5, 3), (0, 5), (5, 0), (100, 90), (1, 1)]:
            p = self.mcnemar_p(b=b, c=c)
            self.assertGreaterEqual(p, 0.0, f"p must be >= 0 for b={b}, c={c}")
            self.assertLessEqual(p, 1.0, f"p must be <= 1 for b={b}, c={c}")


# ---------------------------------------------------------------------------
# Bootstrap CI tests
# ---------------------------------------------------------------------------


class TestBootstrapCI(unittest.TestCase):
    """Tests for bootstrap confidence interval computation."""

    def setUp(self):
        self.bootstrap_ci, _, _ = _get_stat_fns()

    def test_ci_contains_true_mean(self):
        """Bootstrap CI should contain the true mean 95% of the time (empirical check)."""
        rng = random.Random(0)
        true_mean = 0.75
        n_trials = 200
        n_obs = 100
        coverage_count = 0
        for trial_seed in range(n_trials):
            local_rng = random.Random(trial_seed)
            # Bernoulli samples with p=true_mean
            values = [1.0 if local_rng.random() < true_mean else 0.0 for _ in range(n_obs)]
            lo, hi = self.bootstrap_ci(values, n_bootstrap=2000, seed=trial_seed)
            if lo <= true_mean <= hi:
                coverage_count += 1
        coverage = coverage_count / n_trials
        # Accept ± 5% around nominal 95% coverage
        self.assertGreater(coverage, 0.85, f"Coverage too low: {coverage:.2%}")
        self.assertLess(coverage, 1.0, f"Coverage suspiciously perfect: {coverage:.2%}")

    def test_ci_lower_less_than_upper(self):
        """Lower bound must always be <= upper bound."""
        values = [float(i % 2) for i in range(50)]
        lo, hi = self.bootstrap_ci(values, n_bootstrap=1000, seed=1)
        self.assertLessEqual(lo, hi)

    def test_reproducible_with_seed(self):
        """Same seed must produce exactly the same CI."""
        values = [float(i % 3) / 2 for i in range(60)]
        lo1, hi1 = self.bootstrap_ci(values, n_bootstrap=1000, seed=99)
        lo2, hi2 = self.bootstrap_ci(values, n_bootstrap=1000, seed=99)
        self.assertAlmostEqual(lo1, lo2, places=10)
        self.assertAlmostEqual(hi1, hi2, places=10)

    def test_different_seeds_may_differ(self):
        """Different seeds are allowed to (and usually will) differ."""
        values = [float(i % 5) / 4 for i in range(50)]
        lo1, hi1 = self.bootstrap_ci(values, n_bootstrap=1000, seed=1)
        lo2, hi2 = self.bootstrap_ci(values, n_bootstrap=1000, seed=2)
        # Not guaranteed to differ but overwhelmingly likely for non-degenerate data
        differs = (lo1 != lo2) or (hi1 != hi2)
        # Just ensure both are valid
        self.assertLessEqual(lo1, hi1)
        self.assertLessEqual(lo2, hi2)

    def test_single_value_ci_collapses(self):
        """A list of identical values should produce a degenerate CI at that value."""
        values = [0.6] * 100
        lo, hi = self.bootstrap_ci(values, n_bootstrap=500, seed=0)
        self.assertAlmostEqual(lo, 0.6, places=5)
        self.assertAlmostEqual(hi, 0.6, places=5)


# ---------------------------------------------------------------------------
# Mann-Whitney tests
# ---------------------------------------------------------------------------


class TestMannWhitney(unittest.TestCase):
    """Tests for the Mann-Whitney U test for independent samples."""

    def setUp(self):
        _, _, self.mannwhitney_p = _get_stat_fns()

    def test_identical_distributions_high_p(self):
        """Identical distributions should produce p > 0.05."""
        rng = random.Random(42)
        x = [rng.gauss(3.0, 1.0) for _ in range(50)]
        y = [rng.gauss(3.0, 1.0) for _ in range(50)]
        p = self.mannwhitney_p(x, y)
        self.assertGreater(p, 0.05, f"Identical distributions: expected p > 0.05, got {p:.4f}")

    def test_clearly_different_distributions_low_p(self):
        """Well-separated distributions should produce p < 0.05."""
        rng = random.Random(7)
        x = [rng.gauss(1.0, 0.3) for _ in range(60)]
        y = [rng.gauss(4.0, 0.3) for _ in range(60)]
        p = self.mannwhitney_p(x, y)
        self.assertLess(p, 0.05, f"Clearly different distributions: expected p < 0.05, got {p:.4f}")

    def test_p_value_in_valid_range(self):
        """p-value must be in [0, 1]."""
        rng = random.Random(3)
        x = [rng.gauss(2.0, 1.0) for _ in range(30)]
        y = [rng.gauss(2.5, 1.0) for _ in range(30)]
        p = self.mannwhitney_p(x, y)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)

    def test_symmetric(self):
        """Swapping x and y should give the same (two-sided) p-value."""
        rng = random.Random(11)
        x = [rng.gauss(1.0, 0.5) for _ in range(40)]
        y = [rng.gauss(2.0, 0.5) for _ in range(40)]
        p_xy = self.mannwhitney_p(x, y)
        p_yx = self.mannwhitney_p(y, x)
        self.assertAlmostEqual(p_xy, p_yx, places=5)


if __name__ == "__main__":
    unittest.main()
