import numpy as np
import pytest
from abtest import (sample_size_two_proportions, two_proportion_test, cohens_h,
                    simulate_peeking, familywise_error, bonferroni_alpha)


def test_sample_size_grows_as_mde_shrinks():
    big = sample_size_two_proportions(0.12, 0.03)
    small = sample_size_two_proportions(0.12, 0.015)
    assert small > big            # a smaller effect needs a larger sample
    assert big > 0


def test_sample_size_reasonable_ballpark():
    # 12% baseline, +1.5pp, 80% power, 5% two-sided -> ~7.7k per arm
    n = sample_size_two_proportions(0.12, 0.015, alpha=0.05, power=0.80)
    assert 7000 < n < 8500


def test_clear_effect_is_significant():
    r = two_proportion_test(1000, 8000, 1300, 8000)   # 12.5% vs 16.25%
    assert r.p_value < 0.001
    assert r.significant
    assert r.ci_low > 0
    assert r.abs_diff == pytest.approx(0.0375, abs=1e-6)


def test_no_effect_is_not_significant():
    r = two_proportion_test(1000, 8000, 1005, 8000)   # essentially identical
    assert r.p_value > 0.05
    assert not r.significant
    assert r.ci_low < 0 < r.ci_high


def test_ci_brackets_point_estimate():
    r = two_proportion_test(1000, 8000, 1200, 8000)
    assert r.ci_low < r.abs_diff < r.ci_high


def test_relative_lift():
    r = two_proportion_test(1000, 10000, 1200, 10000)  # 10% -> 12%
    assert r.rel_lift == pytest.approx(0.20, abs=1e-6)


def test_cohens_h_sign_and_zero():
    assert cohens_h(0.1, 0.2) > 0
    assert cohens_h(0.2, 0.1) < 0
    assert cohens_h(0.15, 0.15) == pytest.approx(0.0, abs=1e-12)


def test_peeking_inflates_false_positive_rate():
    fixed, peek = simulate_peeking(0.12, n_total=4000, n_looks=8, alpha=0.05,
                                   n_sims=800, seed=3)
    assert abs(fixed - 0.05) < 0.03      # fixed-n stays near nominal alpha
    assert peek > fixed + 0.05           # peeking clearly inflates it


def test_familywise_error_and_bonferroni():
    assert familywise_error(1) == pytest.approx(0.05)
    assert familywise_error(10) > 0.35
    assert bonferroni_alpha(10) == pytest.approx(0.005)
    # bonferroni per-test threshold restores family-wise control to ~alpha
    assert familywise_error(10, alpha=bonferroni_alpha(10)) < 0.05
