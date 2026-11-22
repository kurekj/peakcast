import pytest

import peakcast

GIB = 1024**3


def hinge(batch, floor=2.0 * GIB, intercept=1.2 * GIB, slope=0.075 * GIB):
    """A convex law: flat floor, then a line."""
    return max(floor, intercept + slope * batch)


@pytest.fixture
def calibration():
    scores = {kind: tuple(1.02 + shift + 0.001 * i for i in range(200))
              for kind, shift in (("b", 0.0), ("i", 0.02), ("s", 0.05))}
    return peakcast.Calibration(scores, max_ratio=16.0)


@pytest.fixture
def law(calibration):
    probes = [(b, None, int(hinge(b)), int(1.1 * hinge(b))) for b in (1, 2, 4, 8, 16, 32, 64)]
    return peakcast.MemoryLaw.from_probes(probes, budget=24 * GIB, calibration=calibration)


@pytest.fixture
def law2(calibration):
    def peak(b, n):
        return max(0.55 * GIB, 0.5 * GIB + b * (2e5 * n + 300.0 * n * n))

    probes = [(b, n, int(peak(b, n)), 0) for n in (32, 64, 128) for b in (8, 16, 32, 64)]
    return peakcast.MemoryLaw.from_probes(probes, axis="length", budget=24 * GIB, calibration=calibration)
