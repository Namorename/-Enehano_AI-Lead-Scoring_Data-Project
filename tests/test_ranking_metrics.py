"""Tests for ranking_metrics.py — decile_lift, capture_at_k, ranking_report."""
import numpy as np
import pytest

from ranking_metrics import capture_at_k, decile_lift, ranking_report


# ── helpers ───────────────────────────────────────────────────────────────────

def _perfect_score(n: int = 100, pos_rate: float = 0.2):
    """Scores that perfectly rank all positives above all negatives."""
    n_pos = int(n * pos_rate)
    y = np.array([1] * n_pos + [0] * (n - n_pos))
    s = np.array([1.0] * n_pos + [0.0] * (n - n_pos))
    return y, s


def _random_score(n: int = 100, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    s = rng.uniform(0, 1, n)
    return y, s


# ── decile_lift ────────────────────────────────────────────────────────────────

def test_decile_lift_row_count():
    y, s = _random_score(100)
    df = decile_lift(y, s)
    assert len(df) == 10


def test_decile_lift_columns():
    y, s = _random_score(50)
    df = decile_lift(y, s)
    assert set(df.columns) >= {"decile", "n", "precision", "lift", "cumulative_capture"}


def test_decile_lift_total_n():
    y, s = _random_score(100)
    df = decile_lift(y, s)
    assert df["n"].sum() == 100


def test_decile_lift_top_decile_highest_precision_perfect():
    y, s = _perfect_score(100, 0.2)
    df = decile_lift(y, s)
    assert df.iloc[0]["precision"] == pytest.approx(1.0)


def test_decile_lift_cumulative_capture_ends_at_one():
    y, s = _perfect_score()
    df = decile_lift(y, s)
    assert df["cumulative_capture"].iloc[-1] == pytest.approx(1.0)


def test_decile_lift_lift_non_negative():
    y, s = _random_score(200)
    df = decile_lift(y, s)
    assert (df["lift"] >= 0).all()


def test_decile_lift_custom_bands():
    y, s = _random_score(100)
    df = decile_lift(y, s, n_bands=5)
    assert len(df) == 5


# ── capture_at_k ──────────────────────────────────────────────────────────────

def test_capture_at_k_keys():
    y, s = _random_score(100)
    result = capture_at_k(y, s)
    assert "top_5pct" in result
    assert "top_10pct" in result
    assert "top_20pct" in result
    assert "top_30pct" in result


def test_capture_at_k_perfect_scorer():
    y, s = _perfect_score(100, 0.1)
    result = capture_at_k(y, s, k_fractions=(0.10,))
    # Top 10% = all 10 positives
    assert result["top_10pct"]["capture"] == pytest.approx(1.0)
    assert result["top_10pct"]["precision"] == pytest.approx(1.0)


def test_capture_at_k_bounds():
    y, s = _random_score(200)
    result = capture_at_k(y, s)
    for v in result.values():
        assert 0.0 <= v["precision"] <= 1.0
        assert 0.0 <= v["capture"] <= 1.0
        assert v["lift"] >= 0


# ── ranking_report ─────────────────────────────────────────────────────────────

def test_ranking_report_structure_no_baseline():
    y, s = _random_score(100)
    report = ranking_report(y, s)
    assert "ai" in report
    assert "positive_rate" in report
    assert "n" in report
    assert "baseline" not in report


def test_ranking_report_structure_with_baseline():
    y, ai = _perfect_score(100, 0.2)
    _, bl = _random_score(100)
    report = ranking_report(y, ai, bl)
    assert "baseline" in report
    assert "head_to_head" in report


def test_ranking_report_positive_rate():
    y = np.array([1, 0, 1, 0])
    s = np.array([0.9, 0.1, 0.8, 0.2])
    report = ranking_report(y, s)
    assert report["positive_rate"] == pytest.approx(0.5)


def test_ranking_report_head_to_head_uplift_perfect_vs_random():
    rng = np.random.default_rng(42)
    n = 200
    n_pos = 40
    y = np.array([1] * n_pos + [0] * (n - n_pos))
    ai_score = np.array([1.0] * n_pos + [0.0] * (n - n_pos))
    bl_score = rng.uniform(0, 1, n)

    report = ranking_report(y, ai_score, bl_score)
    uplift = report["head_to_head"]["top_20pct"]["capture_uplift"]
    assert uplift > 0


# ── edge cases ─────────────────────────────────────────────────────────────────

def test_ranking_report_raises_on_empty():
    with pytest.raises(ValueError, match="empty"):
        ranking_report(np.array([]), np.array([]))


def test_decile_lift_raises_on_shape_mismatch():
    with pytest.raises(ValueError):
        decile_lift(np.array([1, 0]), np.array([0.5]))