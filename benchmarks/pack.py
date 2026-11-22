"""Pack the JSON-lines output of measure.py and truth_caps.py into two CSV files.

    python benchmarks/pack.py RAW_GRID_DIR RAW_TRUTH_DIR benchmarks/results/<device>

The CSV files are what the repository ships and what replay.py reads.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import sys

GRID_COLUMNS = ["model", "family", "domain", "axis_name", "axis_default", "params", "optimizer", "amp",
                "axis", "batch", "status", "peak_alloc", "peak_reserved", "device_used", "seconds"]
TRUTH_COLUMNS = ["model", "optimizer", "amp", "axis", "cap", "status", "lo", "hi", "trials", "seconds"]


def rows(directory: str):
    for path in sorted(glob.glob(os.path.join(directory, "*.jsonl"))):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def write(path: str, columns: list[str], records) -> int:
    count = 0
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(columns)
        for r in records:
            value = [r.get(c, "") for c in columns]
            writer.writerow([int(v) if isinstance(v, bool) else (round(v, 4) if isinstance(v, float) else v)
                             for v in value])
            count += 1
    return count


def main() -> None:
    grid_dir, truth_dir, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    grid = [r for r in rows(grid_dir) if "batch" in r]
    meta = {}
    for r in rows(grid_dir):
        if "gpu" in r:
            meta = {"gpu": r["gpu"], "gpu_total": r["gpu_total"], "torch": r["torch"]}
            break
    n = write(os.path.join(out, "grid.csv"), GRID_COLUMNS, grid)
    m = 0
    if os.path.isdir(truth_dir):
        m = write(os.path.join(out, "truth.csv"), TRUTH_COLUMNS, (r for r in rows(truth_dir) if "cap" in r))
    failures = [{"model": r["model"], "error": r.get("error", r.get("note", ""))} for r in rows(grid_dir)
                if r.get("status") == "build_error" or r.get("note") == "timeout"]
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump({**meta, "grid_rows": n, "truth_rows": m, "failed_models": failures}, fh, indent=1)
    print(f"{out}: {n} grid rows, {m} truth rows, {len(failures)} failed models")


if __name__ == "__main__":
    main()
