"""Find the largest batch size that trains under a given memory budget.

For every model, configuration and budget the script brackets the answer with
the unconstrained measurements of measure.py and then bisects with real
training steps under an allocator limit set by
torch.cuda.set_per_process_memory_fraction. The result is an interval
[lo, hi): batch `lo` was verified to train under the budget and batch `hi`
was verified to run out of memory (hi - lo <= max(1, tol * lo)).

These intervals are the ground truth against which batch-size recommendations
are judged.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure import CONFIGS, is_oom, make_optimizer, run_steps  # noqa: E402
from zoo import get_spec  # noqa: E402

GIB = 1024**3


def trial(spec, model, opt, scaler, amp, b, a, dev, steps) -> str:
    gc.collect()
    torch.cuda.empty_cache()
    status = "ok"
    try:
        run_steps(spec, model, opt, scaler, amp, b, a, dev, steps)
    except Exception as exc:  # noqa: BLE001
        status = "oom" if is_oom(exc) else "error"
    if status != "ok":
        opt.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
    return status


def bracket(rows: list[dict], cap: int):
    """Initial (lo, hi) from unconstrained measurements; None when unknown."""
    ok = sorted((r for r in rows if r["status"] == "ok"), key=lambda r: r["batch"])
    oom = [r["batch"] for r in rows if r["status"] == "oom"]
    lo = max((r["batch"] for r in ok if r["peak_reserved"] <= cap), default=0)
    above = [r["batch"] for r in ok if r["peak_alloc"] > cap] + oom
    above = [b for b in above if b > lo]
    if not above:
        return None
    return lo, min(above)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", required=True, help="directory with the measure.py output")
    ap.add_argument("--caps", default="8,12,16,24,40")
    ap.add_argument("--axis-caps", default="12,24")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--tol", type=float, default=0.03)
    args = ap.parse_args()

    with open(os.path.join(args.grid, args.model + ".jsonl"), encoding="utf-8") as fh:
        grid = [json.loads(line) for line in fh]
    grid = [r for r in grid if "batch" in r]
    dev = torch.device("cuda")
    spec = get_spec(args.model)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.manual_seed(0)
    with open(args.out, "a", encoding="utf-8") as fh:

        def emit(rec: dict) -> None:
            fh.write(json.dumps({"model": spec.name, **rec}) + "\n")
            fh.flush()

        for opt_name, amp in CONFIGS:
            rows_cfg = [r for r in grid if r["optimizer"] == opt_name and r["amp"] == amp]
            if not rows_cfg:
                continue
            torch.cuda.set_per_process_memory_fraction(1.0)
            model = spec.build().to(dev).train()
            opt = make_optimizer(opt_name, model)
            scaler = torch.amp.GradScaler("cuda", enabled=amp)
            for a in sorted({r["axis"] for r in rows_cfg}, key=lambda v: (v != spec.default, v)):
                rows = [r for r in rows_cfg if r["axis"] == a]
                caps = args.caps if a == spec.default else args.axis_caps
                for cap_gib in (float(c) for c in caps.split(",")):
                    cap = int(cap_gib * GIB)
                    if cap > total:
                        continue
                    rec = {"optimizer": opt_name, "amp": amp, "axis": a, "cap": cap}
                    br = bracket(rows, cap)
                    if br is None:
                        emit({**rec, "status": "open"})
                        continue
                    lo, hi = br
                    torch.cuda.set_per_process_memory_fraction(cap / total)
                    t0, trials, lo_tested = time.perf_counter(), 0, False
                    while hi - lo > max(1, int(args.tol * lo)):
                        mid = (lo + hi) // 2
                        status = trial(spec, model, opt, scaler, amp, mid, a, dev, args.steps)
                        trials += 1
                        if status == "ok":
                            lo, lo_tested = mid, True
                        else:
                            hi = mid
                    if lo > 0 and not lo_tested:
                        status = trial(spec, model, opt, scaler, amp, lo, a, dev, args.steps)
                        trials += 1
                        if status != "ok":
                            emit({**rec, "status": "bracket_failed", "lo": lo, "hi": hi})
                            continue
                    emit({**rec, "status": "ok", "lo": lo, "hi": hi, "trials": trials,
                          "seconds": time.perf_counter() - t0})
            del model, opt, scaler
            gc.collect()
            torch.cuda.empty_cache()
        emit({"status": "done"})


if __name__ == "__main__":
    main()
