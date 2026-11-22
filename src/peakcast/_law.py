"""The memory law: the object users query."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from ._calibration import Calibration, default_calibration
from ._fit import Fit, fit_points
from ._probe import Probe, ProbePolicy, _Snapshot, device_budget, measure, run_probes

DEFAULT_RISK = 0.01
BATCH_LIMIT = 1 << 24


@dataclass(frozen=True)
class Forecast:
    """Forecast of the peak memory of one training step (bytes)."""

    expected: float  # predicted peak allocated memory
    upper: float  # budget that suffices with probability at least 1 - risk
    risk: float
    extrapolation: float  # expected peak over the largest probed peak

    def __str__(self) -> str:
        return (f"expected {self.expected / 2**30:.2f} GiB, at most {self.upper / 2**30:.2f} GiB "
                f"(risk {self.risk:g}, extrapolation x{self.extrapolation:.1f})")


class MemoryLaw:
    """Peak GPU memory of a training step as a function of its input sizes.

    Build one with :func:`peakcast.fit` (measures on the GPU) or
    :meth:`from_probes` (from measurements taken elsewhere).
    """

    def __init__(self, probes: Sequence[Probe], *, axis: str | None = None, settled: bool = True,
                 budget: float | None = None, calibration: Calibration | None = None,
                 info: dict | None = None):
        if not probes:
            raise ValueError("at least one probe is required")
        self.probes = tuple(probes)
        self.axis = axis
        self.settled = bool(settled)
        self.budget = budget
        self.info = dict(info or {})
        self._calibration = calibration
        sizes = [p.size for p in self.probes]
        self._two_axes = axis is not None and len({s for s in sizes if s is not None}) > 1
        self._default_size = None if not self._two_axes else max(s for s in sizes)
        self.fit: Fit = fit_points(
            [p.batch for p in self.probes],
            [p.allocated for p in self.probes],
            sizes if self._two_axes else None,
        )
        self._top = max(p.allocated for p in self.probes)

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_probes(cls, probes: Sequence[Probe | tuple], **kwargs) -> MemoryLaw:
        """Build a law from ``Probe`` objects or ``(batch, size, allocated, reserved)`` tuples."""
        return cls([p if isinstance(p, Probe) else Probe(*p) for p in probes], **kwargs)

    @property
    def calibration(self) -> Calibration:
        if self._calibration is None:
            self._calibration = default_calibration()
        return self._calibration

    # -- queries --------------------------------------------------------------

    def _size(self, sizes: dict) -> float | None:
        if not self._two_axes:
            if sizes and (self.axis is None or set(sizes) - {self.axis}):
                raise TypeError(f"unknown size argument(s): {sorted(sizes)}")
            return None
        unknown = set(sizes) - {self.axis}
        if unknown:
            raise TypeError(f"unknown size argument(s): {sorted(unknown)}; the axis is {self.axis!r}")
        return float(sizes.get(self.axis, self._default_size))

    def expected(self, batch: float, **sizes: float) -> float:
        """Predicted peak allocated memory in bytes."""
        return float(self.fit.predict(batch, self._size(sizes)))

    def _kind(self, sizes: dict) -> str:
        # "b": batch-only law; "i": two-axis law inside the probed sizes; "s": beyond them
        size = self._size(sizes)
        if size is None:
            return "b"
        return "s" if size > self._default_size else "i"

    def peak(self, batch: float, risk: float = DEFAULT_RISK, **sizes: float) -> Forecast:
        """Forecast with a calibrated upper bound."""
        expected = self.expected(batch, **sizes)
        ratio = expected / self._top
        factor = self.calibration.factor(risk, ratio, self._kind(sizes))
        return Forecast(expected, expected * factor, risk, ratio)

    def upper(self, batch: float, risk: float = DEFAULT_RISK, **sizes: float) -> float:
        """Budget in bytes that suffices with probability at least ``1 - risk``."""
        return self.peak(batch, risk, **sizes).upper

    def max_batch(self, budget: float | None = None, risk: float = DEFAULT_RISK, **sizes: float) -> int:
        """Largest batch size whose upper bound fits ``budget`` (0 if none does).

        Batch sizes whose extrapolation the calibration does not cover, or
        cannot certify at ``risk``, count as not fitting; a risk that the
        calibration cannot certify at all raises ``ValueError``. A batch
        size that a probe has run successfully under a budget no larger than
        ``budget`` is known to fit, so the result is never smaller than that.
        """
        budget = self._budget(budget)
        return max(self._search(budget, risk, sizes), self._probed(budget, sizes))

    def _probed(self, budget: float, sizes: dict) -> int:
        """Largest batch size that a probe has already run at this size within this budget."""
        if self.budget is None or budget < self.budget:
            return 0
        size = self._size(sizes)
        return max((p.batch for p in self.probes if size is None or p.size == size), default=0)

    def _search(self, budget: float, risk: float, sizes: dict) -> int:
        try:
            if self.upper(1, risk, **sizes) > budget:
                return 0
        except KeyError:  # even batch size 1 extrapolates beyond the calibrated range
            return 0
        lo, hi = 1, 2
        while hi < BATCH_LIMIT and self._fits(hi, budget, risk, sizes):
            lo, hi = hi, hi * 2
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self._fits(mid, budget, risk, sizes):
                lo = mid
            else:
                hi = mid
        return lo

    def _fits(self, batch: int, budget: float, risk: float, sizes: dict) -> bool:
        try:
            return self.upper(batch, risk, **sizes) <= budget
        except (KeyError, ValueError):
            return False  # beyond what the calibration covers or can certify at this risk

    def _budget(self, budget: float | None) -> float:
        if budget is not None:
            return float(budget)
        if self.budget is None:
            raise ValueError("no budget: pass budget= (bytes)")
        return float(self.budget)

    def components(self, batch: float, **sizes: float) -> dict[str, float]:
        """Split the expected peak into the terms of the law (bytes)."""
        size = self._size(sizes)
        s = 1.0 if size is None else size / self.fit.size_scale
        names = {(0, 0): "fixed", (1, 0): "per sample", (1, 1): f"per sample x {self.axis}",
                 (1, 2): f"per sample x {self.axis}^2"}
        pairs = zip(self.fit.terms, self.fit.coef, strict=True)
        out = {names[t]: c * batch ** t[0] * s ** t[1] for t, c in pairs}
        out["floor"] = self.fit.floor
        return out

    # -- reporting and persistence -------------------------------------------

    def summary(self) -> str:
        gib = 2**30
        f = self.fit
        lines = [f"peakcast memory law ({len(self.probes)} probes, "
                 f"{'settled' if self.settled else 'not settled'})"]
        terms = []
        for (pb, ps), c in zip(f.terms, f.coef, strict=False):
            if c <= 0:
                continue
            unit = "" if pb == 0 else " x batch"
            if ps:
                c = c / f.size_scale**ps
                unit += f" x {self.axis}" + ("^2" if ps == 2 else "")
            terms.append(f"{c / 2**20:.4g} MiB{unit}")
        lines.append(f"  peak = max({f.floor / gib:.3f} GiB, " + " + ".join(terms) + ")")
        lines.append(f"  largest probe: batch {max(p.batch for p in self.probes)}, "
                     f"{self._top / gib:.3f} GiB allocated")
        if self.budget:
            lines.append(f"  budget: {self.budget / gib:.2f} GiB")
        return "\n".join(lines)

    __str__ = summary

    def to_dict(self) -> dict:
        return {"format": 1, "axis": self.axis, "settled": self.settled, "budget": self.budget,
                "info": self.info, "probes": [asdict(p) for p in self.probes]}

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=1)

    @classmethod
    def from_dict(cls, data: dict, calibration: Calibration | None = None) -> MemoryLaw:
        return cls([Probe(**p) for p in data["probes"]], axis=data.get("axis"),
                   settled=data.get("settled", True), budget=data.get("budget"),
                   calibration=calibration, info=data.get("info"))

    @classmethod
    def load(cls, path: str, calibration: Calibration | None = None) -> MemoryLaw:
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh), calibration)


def fit(
    step: Callable[[Any], Any],
    make_batch: Callable[..., Any],
    *,
    axis: str | None = None,
    sizes: Sequence[float] | None = None,
    budget: float | None = None,
    device: Any = None,
    restore: Sequence[Any] = (),
    policy: ProbePolicy | None = None,
    steps: int = 2,
    calibration: Calibration | None = None,
) -> MemoryLaw:
    """Measure a training step at small sizes and fit its memory law.

    Parameters
    ----------
    step:
        ``step(batch)`` runs one full training step (forward, backward,
        optimizer update) on the GPU.
    make_batch:
        ``make_batch(batch_size)`` returns a batch on the GPU; with ``axis``
        it is called as ``make_batch(batch_size, size)``.
    axis, sizes:
        Name of a second size axis (for example ``"length"`` or
        ``"resolution"``) and the small values of it to probe, at least three.
    budget:
        Memory budget in bytes. Defaults to what the allocator of this process
        can use on the device now. Probes stay well below it.
    restore:
        Objects with ``state_dict``/``load_state_dict`` (model, optimizer,
        gradient scaler) to put back into their initial state after probing.
    policy:
        A :class:`ProbePolicy`; the default calibration assumes the default.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("peakcast.fit measures on a CUDA device and none is available")
    if (axis is None) != (sizes is None):
        raise ValueError("axis and sizes must be given together")
    sizes = None if sizes is None else list(sizes)
    if sizes is not None and len(set(sizes)) < 3:
        raise ValueError("at least three distinct sizes are needed to fit a second axis")
    budget = float(budget) if budget is not None else float(device_budget(device))
    snapshot = _Snapshot(restore)

    def measure_at(batch: int, size: Any):
        make = (lambda: make_batch(batch)) if size is None else (lambda: make_batch(batch, size))
        return measure(step, make, device, steps)

    try:
        probes, all_settled, oom = run_probes(measure_at, sizes, budget, policy or ProbePolicy())
    finally:
        snapshot.restore()
    if not probes:
        raise RuntimeError("the smallest probe ran out of memory; nothing fits this budget")
    if sizes is not None and len({p.size for p in probes}) < 3:
        raise RuntimeError("fewer than three of the requested sizes could be probed within the budget")
    info = {"device": torch.cuda.get_device_name(device), "torch": torch.__version__,
            "probe_out_of_memory": oom}
    return MemoryLaw(probes, axis=axis, settled=all_settled,
                     budget=budget, calibration=calibration, info=info)
