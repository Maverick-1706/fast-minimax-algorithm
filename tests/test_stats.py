"""Tests for benchmarks.stats: BCa bootstrap, outlier removal, repeat policy."""

import pytest

from benchmarks.stats import (
    bootstrap_ci,
    detect_outliers_iqr,
    remove_outliers,
    should_repeat,
    summarise,
)


# ── Outlier detection ──────────────────────────────────────────────────


class TestOutlierDetection:

    def test_no_outliers_in_uniform_data(self):
        data = [1.0, 1.1, 1.0, 1.1, 1.0, 1.1]
        assert detect_outliers_iqr(data) == []

    def test_detects_extreme_outlier(self):
        data = [1.0, 1.1, 1.0, 1.1, 1.0, 100.0]
        outliers = detect_outliers_iqr(data)
        assert 100.0 in outliers

    def test_too_few_points_returns_empty(self):
        assert detect_outliers_iqr([1.0, 2.0, 3.0]) == []

    def test_custom_k(self):
        data = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0]
        # With k=1.5, 10.0 might be an outlier; with k=10 it won't be
        out_k1 = detect_outliers_iqr(data, k=1.5)
        out_k10 = detect_outliers_iqr(data, k=10.0)
        assert len(out_k1) >= len(out_k10)


# ── Outlier removal ───────────────────────────────────────────────────


class TestOutlierRemoval:

    def test_remove_returns_clean_and_outliers(self):
        data = [1.0, 1.1, 1.0, 1.1, 1.0, 100.0]
        clean, outliers = remove_outliers(data)
        assert 100.0 not in clean
        assert 100.0 in outliers

    def test_no_outliers_no_change(self):
        data = [1.0, 1.1, 1.0, 1.1, 1.0, 1.1]
        clean, outliers = remove_outliers(data)
        assert clean == data
        assert outliers == []

    def test_max_fraction_guard(self):
        # If more than 40% are "outliers", don't remove any
        data = [1.0, 10.0, 20.0, 30.0, 40.0]
        clean, outliers = remove_outliers(data)
        # With such spread data, the guard should trigger
        # and clean should equal data
        assert len(clean) == len(data)


# ── Bootstrap CI ───────────────────────────────────────────────────────


class TestBootstrapCI:

    def test_ci_contains_mean(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        lo, hi = bootstrap_ci(data, seed=0)
        mean = sum(data) / len(data)
        assert lo <= mean <= hi

    def test_ci_narrow_with_large_sample(self):
        import random
        rng = random.Random(42)
        data = [rng.gauss(0.0, 0.01) for _ in range(200)]
        lo, hi = bootstrap_ci(data, seed=0)
        assert hi - lo < 0.01

    def test_single_point_returns_value(self):
        assert bootstrap_ci([5.0]) == (5.0, 5.0)

    def test_empty_returns_zero(self):
        assert bootstrap_ci([]) == (0.0, 0.0)

    def test_bca_method(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        lo, hi = bootstrap_ci(data, method="bca", seed=0)
        mean = sum(data) / len(data)
        assert lo <= mean <= hi

    def test_percentile_method(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        lo, hi = bootstrap_ci(data, method="percentile", seed=0)
        mean = sum(data) / len(data)
        assert lo <= mean <= hi


# ── Summary ────────────────────────────────────────────────────────────


class TestSummarise:

    def test_empty_data(self):
        s = summarise([])
        assert s.n == 0
        assert s.mean == 0.0

    def test_with_outlier_removal(self):
        data = [1.0, 1.1, 1.0, 1.1, 1.0, 100.0]
        s = summarise(data, remove=True)
        assert s.n < len(data)
        assert s.n_outliers > 0
        assert s.n_original == len(data)
        assert s.mean < 10.0  # outlier removed

    def test_without_outlier_removal(self):
        data = [1.0, 1.1, 1.0, 1.1, 1.0, 100.0]
        s = summarise(data, remove=False)
        assert s.n == len(data)
        assert s.n_outliers > 0  # still flagged
        assert s.mean > 10.0  # outlier included

    def test_cv_reported(self):
        data = [1.0, 1.1, 1.0, 1.1, 1.0, 1.1]
        s = summarise(data, remove=False)
        assert 0.0 <= s.cv < 0.1  # low variance data


# ── Repeat policy ─────────────────────────────────────────────────────


class TestShouldRepeat:

    def test_stable_data_no_repeat(self):
        data = [1.0, 1.01, 1.02, 1.0, 1.01]
        d = should_repeat(data)
        assert not d.should_repeat
        assert d.cv < 0.15

    def test_high_cv_triggers_repeat(self):
        data = [1.0, 2.0, 0.5, 3.0, 0.1]
        d = should_repeat(data, cv_threshold=0.1)
        assert d.should_repeat
        assert "CV" in d.reason

    def test_many_outliers_trigger_repeat(self):
        data = [1.0, 1.0, 1.0, 1.0, 100.0, 200.0]
        d = should_repeat(data, outlier_fraction_threshold=0.1)
        assert d.should_repeat
