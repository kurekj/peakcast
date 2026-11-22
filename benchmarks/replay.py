"""Replay peakcast on recorded measurements.

The measurements of measure.py contain the peak memory of every workload at
every power-of-two batch size, so the probing of peakcast can be replayed
exactly, by the code of the library itself, without touching a GPU: the
library asks for a probe and the replay looks the answer up. The ground truth
of truth_caps.py then tells whether a recommended batch size would have
trained under the budget.

This module holds the shared loading and replay functions; make_calibration.py
and evaluate.py use them.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import peakcast
from peakcast._probe import run_probes

GIB = 1024**3
# Small values of the second axis that are probed when a two-axis law is fitted.
PROBE_SIZES = {
    "length": (32, 64, 128),
    "resolution": (128, 160, 192),
    "frames": (4, 8, 16),
    "samples": (16000, 32000, 64000),
}


@dataclass
class Workload:
    model: str
    family: str
    domain: str
    optimizer: str
    amp: bool
    axis_name: str
    axis_default: float
    params: int
    rows: dict  # axis value -> {batch: row}

    @property
    def key(self) -> str:
        return f"{self.model}|{self.optimizer}|{int(self.amp)}"


def _number(text: str):
    value = float(text)
    return int(value) if value == int(value) else value


def load_grid(directory: str) -> dict[str, Workload]:
    """Read grid.csv (see pack.py) into workloads keyed by model, optimizer and precision."""
    out: dict[str, Workload] = {}
    with open(os.path.join(directory, "grid.csv"), newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            amp = r["amp"] == "1"
            key = f"{r['model']}|{r['optimizer']}|{int(amp)}"
            if key not in out:
                out[key] = Workload(r["model"], r["family"], r["domain"], r["optimizer"], amp,
                                    r["axis_name"], _number(r["axis_default"]), int(r["params"]), {})
            row = {"status": r["status"], "seconds": float(r["seconds"] or 0)}
            if r["status"] == "ok":
                row.update(peak_alloc=int(r["peak_alloc"]), peak_reserved=int(r["peak_reserved"]),
                           device_used=int(r["device_used"]))
            out[key].rows.setdefault(_number(r["axis"]), {})[int(r["batch"])] = row
    return out


def load_truth(directory: str) -> dict[tuple, tuple[int, int]]:
    """Read truth.csv: (workload key, size, budget) -> (lo, hi)."""
    out = {}
    with open(os.path.join(directory, "truth.csv"), newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["status"] == "ok":
                key = (f"{r['model']}|{r['optimizer']}|{r['amp']}", _number(r["axis"]), int(r["cap"]))
                out[key] = (int(r["lo"]), int(r["hi"]))
    return out


def replay_law(w: Workload, budget: float, two_axes: bool = False, policy=None, calibration=None,
               truth: dict | None = None, missing: list | None = None):
    """Fit a law by replaying the probes of peakcast on recorded measurements.

    A probe runs out of memory, as it would have under the budget, when its
    batch size is above the largest one that the ground truth found to fit
    (or, without ground truth for that size, when its peak exceeds the
    budget). Probes that were never measured are appended to ``missing``.

    Returns (law, seconds spent probing, a probe ran out of memory), or
    (None, 0, False) if the workload cannot be probed.
    """
    cost = [0.0]

    def measure_at(batch, size):
        value = w.axis_default if size is None else size
        known = (truth or {}).get((w.key, value, int(budget)))
        if known is not None and batch > known[0]:
            return None
        row = w.rows.get(value, {}).get(batch)
        if row is None:
            if missing is not None:
                missing.append((w.model, w.optimizer, int(w.amp), value, batch))
            return None
        if row["status"] != "ok" or (known is None and row["peak_alloc"] > budget):
            return None
        cost[0] += 2 * row["seconds"]
        return row["peak_alloc"], row["peak_reserved"]

    sizes = None
    if two_axes:
        sizes = [s for s in PROBE_SIZES.get(w.axis_name, ()) if s in w.rows]
        if len(sizes) < 3:
            return None, 0.0, False
    if policy is None:
        first = w.rows.get(w.axis_default, {}).get(1)
        # models that cannot train on a single sample are probed from batch size 2
        policy = peakcast.ProbePolicy(min_batch=2 if first and first["status"] == "error" else 1)
    probes, settled, oom = run_probes(measure_at, sizes, budget, policy)
    if not probes or (two_axes and len({p.size for p in probes}) < 3):
        return None, 0.0, oom
    law = peakcast.MemoryLaw(probes, axis=w.axis_name if two_axes else None, settled=settled,
                             budget=budget, calibration=calibration)
    return law, cost[0], oom


@dataclass
class Unit:
    """One (workload, size, budget) case with its ground truth and replayed law."""

    workload: Workload
    size: float
    cap: int
    lo: int
    hi: int
    law: object
    two_axes: bool
    probe_seconds: float
    probe_oom: bool = False

    @property
    def kwargs(self) -> dict:
        return {self.workload.axis_name: self.size} if self.two_axes else {}

    @property
    def kind(self) -> str:
        return self.law._kind(self.kwargs)

    def score(self) -> tuple[float, float]:
        """(extrapolation ratio, score) at the first batch that does not fit."""
        expected = self.law.expected(self.lo + 1, **self.kwargs)
        return expected / self.law._top, self.cap / expected

    def record(self) -> tuple:
        ratio, score = self.score()
        return (self.workload.key, self.kind, ratio, score)


def build_units(grid: dict[str, Workload], truth: dict, two_axes: bool, policy=None,
                missing: list | None = None) -> list[Unit]:
    units = []
    laws: dict[tuple, tuple] = {}
    for (key, size, cap), (lo, hi) in sorted(truth.items()):
        w = grid.get(key)
        if w is None or lo < 1:
            continue
        if not two_axes and size != w.axis_default:
            continue
        if two_axes and (w.optimizer, w.amp) != ("adamw", False):
            continue
        if (key, cap) not in laws:
            laws[(key, cap)] = replay_law(w, cap, two_axes, policy, truth=truth, missing=missing)
        law, seconds, oom = laws[(key, cap)]
        if law is None:
            continue
        units.append(Unit(w, size, cap, lo, hi, law, two_axes, seconds, oom))
    return units


def main_device(results: str) -> str:
    """Name of the device whose measurements define the shipped calibration (results/main.txt)."""
    path = os.path.join(results, "main.txt")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return fh.read().strip()
    return "l40s"
