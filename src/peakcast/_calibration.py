"""Conformal calibration of the upper bound.

A fitted law predicts the peak *allocated* memory. What a user needs is a
bound on the memory budget under which training actually runs, which is
larger: the caching allocator fragments its pool and the tail fit is a lower
bound of a convex law. The ratio between the two,

    score = (smallest budget under which the step runs) / (predicted peak),

is measured on a set of calibration workloads. By the split conformal
argument, multiplying a prediction by the ceil((n + 1)(1 - risk))-th smallest
of n calibration scores bounds the required budget of a new workload with
probability at least 1 - risk, provided the new workload is exchangeable with
the calibration workloads.

Scores are kept in groups by the kind of question: ``"b"`` for a batch-only
law, ``"i"`` for a two-axis law asked inside the probed range of its second
axis and ``"s"`` for a two-axis law asked beyond it. Each group holds one
score per calibration workload (its largest). A calibration is valid up to
the extrapolation ratio (predicted peak over largest probed peak) that its
workloads covered; beyond ``max_ratio`` it refuses to answer.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib import resources

DEFAULT_MAX_RATIO = 16.0


@dataclass(frozen=True)
class Calibration:
    """Calibration scores by kind of question, valid up to ``max_ratio``."""

    scores: dict[str, tuple[float, ...]]
    max_ratio: float = DEFAULT_MAX_RATIO
    meta: dict = field(default_factory=dict)

    def group(self, kind: str = "b") -> tuple[float, ...]:
        if kind not in self.scores or not self.scores[kind]:
            raise KeyError(f"the calibration has no scores of kind {kind!r}")
        return self.scores[kind]

    def factor(self, risk: float, ratio: float = 1.0, kind: str = "b") -> float:
        """Multiplier that turns a predicted peak into an upper bound at ``risk``.

        Raises ``KeyError`` if ``ratio`` is beyond the extrapolation the
        calibration covers or the kind is missing, and ``ValueError`` if the
        group is too small to certify ``risk``.
        """
        if ratio > self.max_ratio:
            raise KeyError(f"extrapolation ratio {ratio:.1f} is beyond the calibrated range "
                           f"(up to {self.max_ratio:g}); probe further")
        scores = self.group(kind)
        q = conformal_quantile(scores, risk)
        if not math.isfinite(q):
            n = len(scores)
            raise ValueError(f"risk {risk:g} cannot be certified with {n} calibration workloads "
                             f"of kind {kind!r}; the smallest certifiable risk is {1.0 / (n + 1):.3g}")
        return q

    def min_risk(self, kind: str = "b") -> float:
        """Smallest risk the calibration set can certify for this kind of question."""
        return 1.0 / (len(self.group(kind)) + 1)

    def to_dict(self) -> dict:
        return {"format": 1, "max_ratio": self.max_ratio,
                "scores": {k: list(v) for k, v in self.scores.items()}, "meta": self.meta}

    @classmethod
    def from_dict(cls, data: dict) -> Calibration:
        return cls({k: tuple(sorted(v)) for k, v in data["scores"].items()},
                   float(data.get("max_ratio", DEFAULT_MAX_RATIO)), data.get("meta", {}))

    @classmethod
    def from_scores(cls, records: Sequence[tuple], max_ratio: float | None = None,
                    meta: dict | None = None) -> Calibration:
        """Build a calibration from ``(workload, kind, ratio, score)`` records.

        A workload that contributes several records of one kind is represented
        by its largest score. ``max_ratio`` defaults to the largest
        extrapolation ratio among the records, so the calibration answers only
        within the range its workloads covered; with an explicit value,
        records beyond it are ignored.
        """
        records = list(records)
        if max_ratio is None:
            max_ratio = max((float(r[2]) for r in records), default=DEFAULT_MAX_RATIO)
        groups: dict[str, dict] = {}
        for workload, kind, ratio, score in records:
            if ratio > max_ratio:
                continue
            group = groups.setdefault(kind, {})
            group[workload] = max(float(score), group.get(workload, 0.0))
        return cls({k: tuple(sorted(v.values())) for k, v in groups.items()}, float(max_ratio), meta or {})


def conformal_quantile(scores: Sequence[float], risk: float) -> float:
    """The ceil((n + 1)(1 - risk))-th smallest score; infinite if n is too small."""
    if not 0.0 < risk < 1.0:
        raise ValueError("risk must be in (0, 1)")
    n = len(scores)
    k = (n + 1) - math.floor((n + 1) * risk)  # = ceil((n + 1)(1 - risk)) without rounding 1 - risk
    if k > n:
        return math.inf
    return sorted(scores)[k - 1]


def default_calibration() -> Calibration:
    """The calibration shipped with the package (see ``benchmarks/`` for how it was made)."""
    text = resources.files("peakcast").joinpath("data/calibration.json").read_text(encoding="utf-8")
    return Calibration.from_dict(json.loads(text))
