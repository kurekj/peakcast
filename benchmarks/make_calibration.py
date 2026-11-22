"""Build the calibration that ships with peakcast, and the leave-one-family-out ones.

    python benchmarks/make_calibration.py [DEVICE]

Reads benchmarks/results/DEVICE (default: the main device) (see pack.py), replays the probing of peakcast
for every workload and budget, and writes

  src/peakcast/data/calibration.json    scores of all workloads (the default calibration)
  benchmarks/results/lofo_DEVICE/<family>.json the same without one architecture family,
                                        used by the live comparisons so that the model
                                        under test is never part of its own calibration
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import replay  # noqa: E402

import peakcast  # noqa: E402


def main() -> None:
    first = replay.main_device(os.path.join(HERE, "results"))
    device = sys.argv[1] if len(sys.argv) > 1 else first
    directory = os.path.join(HERE, "results", device)
    grid = replay.load_grid(directory)
    truth = replay.load_truth(directory)
    with open(os.path.join(directory, "meta.json"), encoding="utf-8") as fh:
        info = json.load(fh)
    units = replay.build_units(grid, truth, two_axes=False) + replay.build_units(grid, truth, two_axes=True)
    families = sorted({u.workload.family for u in units})
    meta = {
        "source": "benchmarks/results/" + device,
        "gpu": info.get("gpu"),
        "torch": info.get("torch"),
        "policy": "peakcast.ProbePolicy() defaults",
        "workloads": len({u.workload.key for u in units}),
        "models": len({u.workload.model for u in units}),
        "families": families,
        "score": "budget / predicted peak at the first batch size that does not fit the budget",
    }
    cal = peakcast.Calibration.from_scores([u.record() for u in units], meta=meta)
    if device == first:  # only the main device defines the calibration that ships
        out = os.path.join(HERE, "..", "src", "peakcast", "data", "calibration.json")
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(cal.to_dict(), fh, indent=1)
    lofo = os.path.join(HERE, "results", "lofo_" + device)
    os.makedirs(lofo, exist_ok=True)
    for family in families:
        part = peakcast.Calibration.from_scores(
            [u.record() for u in units if u.workload.family != family],
            meta={**meta, "left_out": family})
        with open(os.path.join(lofo, family + ".json"), "w", encoding="utf-8") as fh:
            json.dump(part.to_dict(), fh)
    for key in sorted(cal.scores):
        s = cal.scores[key]
        print(f"{key}: n={len(s)} median={s[len(s) // 2]:.3f} max={s[-1]:.3f}")


if __name__ == "__main__":
    main()
