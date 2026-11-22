import math

import pytest

import peakcast


def test_conformal_quantile_uses_the_finite_sample_rank():
    scores = [float(i) for i in range(1, 100)]  # n = 99
    assert peakcast.conformal_quantile(scores, 0.01) == 99.0
    assert peakcast.conformal_quantile(scores, 0.5) == 50.0
    assert math.isinf(peakcast.conformal_quantile(scores[:50], 0.01))


def test_rank_is_exact_at_the_edge_of_certifiability():
    assert math.isinf(peakcast.conformal_quantile([1.0], math.nextafter(0.5, 0.0)))
    assert peakcast.conformal_quantile([1.0], 0.5) == 1.0
    with pytest.raises(ValueError):
        peakcast.conformal_quantile([1.0], 0.0)


def test_lower_risk_gives_a_larger_factor(calibration):
    assert calibration.factor(0.01, 6.0) > calibration.factor(0.2, 6.0)


def test_kinds_have_their_own_groups(calibration):
    assert calibration.factor(0.05, 6.0, kind="s") > calibration.factor(0.05, 6.0, kind="b")
    with pytest.raises(KeyError):
        calibration.factor(0.05, 6.0, kind="x")


def test_uncertifiable_risk_is_an_error(calibration):
    with pytest.raises(ValueError, match="smallest certifiable risk"):
        calibration.factor(0.001, 6.0)
    assert calibration.min_risk() == pytest.approx(1 / 201)


def test_extrapolation_beyond_the_calibrated_range_is_an_error(calibration):
    assert calibration.factor(0.05, calibration.max_ratio) > 1.0
    with pytest.raises(KeyError, match="beyond the calibrated range"):
        calibration.factor(0.05, calibration.max_ratio * 1.01)


def test_from_scores_keeps_the_largest_score_of_a_workload():
    records = [("m1", "b", 3.0, 1.05), ("m1", "b", 3.5, 1.10), ("m2", "b", 3.0, 1.02), ("m3", "b", 99.0, 5.0)]
    cal = peakcast.Calibration.from_scores(records, max_ratio=16.0)
    assert cal.scores["b"] == (1.02, 1.10)  # one score per workload; the record beyond max_ratio is ignored
    covered = peakcast.Calibration.from_scores(records)
    assert covered.max_ratio == 99.0 and len(covered.scores["b"]) == 3  # valid as far as the records reach
    again = peakcast.Calibration.from_dict(cal.to_dict())
    assert again.scores == cal.scores and again.max_ratio == cal.max_ratio


def test_default_calibration_loads():
    cal = peakcast.default_calibration()
    assert cal.max_ratio > 1 and isinstance(cal.scores, dict)
