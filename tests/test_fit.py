import numpy as np
import pytest
from conftest import GIB, hinge

import peakcast
from peakcast._fit import nnls


def test_nnls_matches_unconstrained_solution_when_it_is_non_negative():
    rng = np.random.default_rng(0)
    a = rng.random((20, 3))
    x = np.array([0.5, 2.0, 1.0])
    assert np.allclose(nnls(a, a @ x), x, atol=1e-8)


def test_nnls_clamps_negative_coefficients():
    a = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    y = np.array([-1.0, 0.0, 1.0])  # the exact solution has intercept -1
    x = nnls(a, y)
    assert x[0] == 0.0 and x[1] > 0.0


def test_fit_recovers_the_line_behind_a_floor():
    batches = [1, 2, 4, 8, 16, 32, 64]
    fit = peakcast.fit_points(batches, [hinge(b) for b in batches])
    assert fit.predict(512) == pytest.approx(hinge(512), rel=1e-6)
    assert fit.predict(1) == pytest.approx(2.0 * GIB)


def test_fit_uses_the_largest_probes_only():
    batches = [1, 2, 4, 8, 16, 32, 64]
    peaks = [hinge(b) for b in batches]
    assert peaks[0] == peaks[3]  # the first probes are flat
    fit = peakcast.fit_points(batches, peaks)
    assert fit.points_used == 3


def test_two_axis_fit_recovers_linear_and_quadratic_terms():
    def peak(b, n):
        return 5e8 + b * (2e5 * n + 300.0 * n * n)

    pts = [(b, n) for n in (32, 64, 128) for b in (4, 8, 16)]
    fit = peakcast.fit_points([p[0] for p in pts], [peak(*p) for p in pts], [p[1] for p in pts])
    assert fit.predict(64, 512) == pytest.approx(peak(64, 512), rel=1e-6)


def test_single_probe_falls_back_to_the_proportional_bound():
    fit = peakcast.fit_points([4], [4e9])
    assert fit.predict(8) == pytest.approx(8e9)


def test_settled_needs_three_collinear_rising_points():
    assert peakcast.settled([16, 32, 64], [hinge(16), hinge(32), hinge(64)])
    assert not peakcast.settled([1, 2, 4], [hinge(1), hinge(2), hinge(4)])  # flat
    assert not peakcast.settled([8, 16, 32], [hinge(8), hinge(16), hinge(32)])  # leaving the floor
    assert not peakcast.settled([32, 64], [hinge(32), hinge(64)])


def test_mismatched_input_is_rejected():
    with pytest.raises(ValueError):
        peakcast.fit_points([1, 2], [1.0])
