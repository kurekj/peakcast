import pytest
from conftest import GIB, hinge

import peakcast


def test_expected_extrapolates_the_line(law):
    assert law.expected(256) == pytest.approx(hinge(256), rel=1e-6)


def test_upper_bound_is_above_the_expectation(law):
    forecast = law.peak(256, risk=0.05)
    assert forecast.upper > forecast.expected
    assert forecast.extrapolation == pytest.approx(hinge(256) / hinge(64), rel=1e-6)
    assert "GiB" in str(forecast)


def test_max_batch_is_the_largest_batch_that_fits(law):
    best = law.max_batch(risk=0.05)
    assert law.upper(best, 0.05) <= 24 * GIB < law.upper(best + 1, 0.05)


def test_max_batch_shrinks_with_budget_and_risk(law):
    assert law.max_batch(8 * GIB, 0.05) < law.max_batch(16 * GIB, 0.05)
    assert law.max_batch(16 * GIB, 0.01) <= law.max_batch(16 * GIB, 0.2)


def test_nothing_fits_a_budget_below_the_floor(law):
    assert law.max_batch(1 * GIB, 0.05) == 0


def test_budget_is_required(calibration):
    law = peakcast.MemoryLaw.from_probes([(1, None, 100, 0), (2, None, 200, 0)], calibration=calibration)
    with pytest.raises(ValueError, match="budget"):
        law.max_batch()


def test_uncertifiable_risk_propagates(law):
    with pytest.raises(ValueError):
        law.max_batch(risk=1e-4)


def test_two_axis_law_predicts_longer_sequences(law2):
    def peak(b, n):
        return 0.5 * GIB + b * (2e5 * n + 300.0 * n * n)

    assert law2.expected(16, length=512) == pytest.approx(peak(16, 512), rel=1e-6)
    assert law2.max_batch(length=512) < law2.max_batch(length=128)
    assert law2._kind({"length": 512}) == "s" and law2._kind({"length": 64}) == "i"


def test_unknown_size_argument_is_rejected(law, law2):
    with pytest.raises(TypeError):
        law.expected(4, length=10)
    with pytest.raises(TypeError):
        law2.expected(4, resolution=10)


def test_components_add_up_above_the_floor(law2):
    parts = law2.components(64, length=512)
    floor = parts.pop("floor")
    assert sum(parts.values()) == pytest.approx(law2.expected(64, length=512))
    assert floor > 0


def test_save_and_load_round_trip(law, tmp_path, calibration):
    path = tmp_path / "law.json"
    law.save(str(path))
    again = peakcast.MemoryLaw.load(str(path), calibration)
    assert again.expected(100) == law.expected(100)
    assert again.max_batch(risk=0.05) == law.max_batch(risk=0.05)
    assert "peak = max(" in again.summary()


def test_max_batch_stops_where_the_calibration_ends():
    cal = peakcast.Calibration({"b": tuple([1.1] * 50)}, max_ratio=4.0)
    law = peakcast.MemoryLaw.from_probes([(b, None, int(1e9 * b), 0) for b in (1, 2, 4)], calibration=cal)
    # ratio 4 (batch 16) is the last covered one, however large the budget
    assert law.max_batch(1e15, 0.1) in (15, 16)


def test_a_probed_batch_size_is_known_to_fit(calibration):
    # floor close to the budget: the bound alone would allow nothing
    probes = [(b, None, int(7.5 * GIB + 1e6 * b), 0) for b in (1, 2, 4, 8)]
    law = peakcast.MemoryLaw.from_probes(probes, budget=8 * GIB, calibration=calibration, settled=False)
    assert law.max_batch(risk=0.05) == 8
    assert law.max_batch(budget=7 * GIB, risk=0.05) == 0  # a smaller budget than the probes ran under
