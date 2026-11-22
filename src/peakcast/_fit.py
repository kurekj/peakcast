"""Fitting a memory law to probe measurements.

The peak allocated memory of a training step is the maximum, over the program
points of the step, of the memory live at that point. For fixed tensor shapes
per sample, the memory live at one program point is affine in the batch size
with a non-negative intercept. The peak is therefore a maximum of affine
functions: convex and piecewise linear in the batch size. Two consequences
drive the design of this module:

* at small batch sizes the peak is often flat (the optimizer update, whose
  footprint does not depend on the batch, dominates), so points on that floor
  carry no information about the slope: probing continues until the slope
  has settled and the fit uses the largest probes only;
* a line through the largest probes is a lower bound of the true peak further
  out, so an upper bound has to be added on top of the fit (see
  ``_calibration``).

The fitted law is ``max(floor, sum_k c_k * batch^p_k * size^q_k)`` with
non-negative coefficients found by non-negative least squares on relative
errors.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# (power of batch, power of size) for each candidate term
TERMS_BATCH = ((0, 0), (1, 0))
TERMS_SIZE = ((0, 0), (1, 0), (1, 1), (1, 2))

TAIL_POINTS = 3
FLOOR_TOLERANCE = 0.01
RISE = 0.05
SLOPE_LOW, SLOPE_HIGH = 0.9, 1.15


def nnls(a: np.ndarray, y: np.ndarray, max_iter: int | None = None) -> np.ndarray:
    """Non-negative least squares (Lawson and Hanson active-set method)."""
    a = np.asarray(a, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = a.shape[1]
    max_iter = max_iter or 30 * n
    x = np.zeros(n)
    passive = np.zeros(n, dtype=bool)
    w = a.T @ (y - a @ x)
    tol = 10 * np.finfo(float).eps * np.abs(a).sum() * max(a.shape)
    it = 0
    while (~passive).any() and w[~passive].max() > tol and it < max_iter:
        j = int(np.argmax(np.where(~passive, w, -np.inf)))
        passive[j] = True
        while True:
            it += 1
            s = np.zeros(n)
            s[passive] = np.linalg.lstsq(a[:, passive], y, rcond=None)[0]
            if s[passive].min() > 0 or it >= max_iter:
                break
            neg = passive & (s <= 0)
            alpha = np.min(x[neg] / (x[neg] - s[neg]))
            x = x + alpha * (s - x)
            passive &= x > tol
        x = s
        w = a.T @ (y - a @ x)
    return np.maximum(x, 0.0)


@dataclass(frozen=True)
class Fit:
    """Coefficients of a memory law (bytes)."""

    floor: float
    terms: tuple[tuple[int, int], ...]
    coef: tuple[float, ...]
    size_scale: float
    rel_residual: float
    points_used: int

    def growth(self, batch, size=None) -> np.ndarray:
        b = np.asarray(batch, dtype=np.float64)
        s = np.ones_like(b) if size is None else np.asarray(size, dtype=np.float64) / self.size_scale
        out = np.zeros(np.broadcast(b, s).shape)
        for (pb, ps), c in zip(self.terms, self.coef, strict=False):
            out = out + c * b**pb * s**ps
        return out

    def predict(self, batch, size=None) -> np.ndarray:
        return np.maximum(self.floor, self.growth(batch, size))


def _design(b: np.ndarray, s: np.ndarray, terms) -> np.ndarray:
    return np.stack([b**pb * s**ps for pb, ps in terms], axis=1)


def _tail(b: np.ndarray, s: np.ndarray, tail: int) -> np.ndarray:
    """Indices of the points used for the fit: per size, the largest batches."""
    keep: list[int] = []
    for value in np.unique(s):
        idx = np.flatnonzero(s == value)
        idx = idx[np.argsort(b[idx])]
        keep += list(idx[-tail:])
    return np.array(sorted(keep), dtype=int)


def settled(batch: Sequence[float], peak: Sequence[float]) -> bool:
    """True when the last three probes lie on one line that rises with the batch.

    The slope of a convex law can only grow, so two consecutive secant slopes
    that agree indicate that the probes have left the flat floor and the
    transition after it.
    """
    if len(batch) < 3:
        return False
    b = np.asarray(batch[-3:], dtype=np.float64)
    y = np.asarray(peak[-3:], dtype=np.float64)
    s1 = (y[1] - y[0]) / (b[1] - b[0])
    s2 = (y[2] - y[1]) / (b[2] - b[1])
    if s1 <= 0 or s2 <= 0 or y[2] < (1.0 + RISE) * y[1]:
        return False
    return SLOPE_LOW <= s2 / s1 <= SLOPE_HIGH


def fit_points(
    batch: Sequence[float],
    peak: Sequence[float],
    size: Sequence[float] | None = None,
    tail: int = TAIL_POINTS,
) -> Fit:
    """Fit a memory law to measured peaks.

    ``batch``, ``peak`` and the optional ``size`` are parallel sequences, one
    entry per probe. Without ``size`` the law is ``max(floor, c0 + c1 * batch)``.
    """
    b = np.asarray(batch, dtype=np.float64)
    y = np.asarray(peak, dtype=np.float64)
    if b.size == 0 or b.shape != y.shape:
        raise ValueError("batch and peak must be non-empty and of equal length")
    two_axes = size is not None and len(set(size)) > 1
    scale = float(np.max(size)) if two_axes else 1.0
    s = np.asarray(size, dtype=np.float64) / scale if two_axes else np.ones_like(b)
    terms = TERMS_SIZE if two_axes else TERMS_BATCH
    floor = float(y.min())
    idx = _tail(b, s, tail)
    if two_axes:
        # A size whose probes never left the floor says nothing about growth.
        off_floor = idx[y[idx] > floor * (1.0 + FLOOR_TOLERANCE)]
        if off_floor.size >= len(terms):
            idx = off_floor
    top = int(np.argmax(y))
    if idx.size < 2:
        # Not enough points above the floor to see a slope. Fall back to the
        # proportional law through the largest probe; with non-negative
        # intercepts it is an upper bound of every admissible law.
        coef = [0.0] * len(terms)
        pos = terms.index((1, 2)) if two_axes and s[top] > 0 else terms.index((1, 0))
        pb, ps = terms[pos]
        coef[pos] = y[top] / (b[top] ** pb * s[top] ** ps)
        return Fit(floor, tuple(terms), tuple(coef), scale, float("nan"), int(idx.size))
    a = _design(b[idx], s[idx], terms)
    w = 1.0 / y[idx]
    weighted = a * w[:, None]
    norms = np.linalg.norm(weighted, axis=0)
    norms[norms == 0] = 1.0
    coef = nnls(weighted / norms, np.ones(idx.size)) / norms  # unit columns keep the solver well scaled
    if not np.any(coef[1:] > 0):
        pos = 1
        coef = np.zeros(len(terms))
        coef[pos] = y[top] / (b[top] * (s[top] ** terms[pos][1]))
    res = a @ coef / y[idx] - 1.0
    return Fit(floor, tuple(terms), tuple(float(c) for c in coef), scale,
               float(np.sqrt(np.mean(res**2))), int(idx.size))
