"""Measure the peak GPU memory of training steps over a grid of input sizes.

One process measures one model: every optimizer and precision configuration,
over a batch grid at the default value of the second axis (until the device
runs out of memory), and over a coarser batch grid at the other values of the
second axis. Results are appended to a JSON-lines file, one line per size.

This is the ground truth for the benchmarks: it deliberately runs every size,
including the ones that exhaust the device, which is exactly what peakcast is
designed to avoid.
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
from zoo import get_spec  # noqa: E402

CONFIGS = (("adamw", False), ("sgd", False), ("adamw", True))
GIB = 1024**3


def batch_grid(fine: bool, limit: int = 16384) -> list[int]:
    out, b = [], 1
    while b <= limit:
        out.append(b)
        if fine and b >= 2:
            out.append(b + b // 2)
        b *= 2
    return [x for x in out if x <= limit]


def is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    text = str(exc).lower()
    return "out of memory" in text or "alloc_failed" in text or "cudnn_status_alloc" in text


def make_optimizer(name: str, model: torch.nn.Module) -> torch.optim.Optimizer:
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=1e-5)
    return torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)


def run_steps(spec, model, opt, scaler, amp, b, a, dev, steps):
    for _ in range(steps):
        batch = spec.make_batch(b, a, dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            loss = spec.loss(model, batch)
        if amp:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        opt.zero_grad(set_to_none=True)
        del batch, loss
    torch.cuda.synchronize()


def measure_one(spec, model, opt, scaler, amp, b, a, dev, steps):
    """Return (status, record) for one size."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    rec = {"batch": b, "axis": a, "base_alloc": torch.cuda.memory_allocated()}
    status, err = "ok", None
    t0 = time.perf_counter()
    try:
        run_steps(spec, model, opt, scaler, amp, b, a, dev, steps)
    except Exception as exc:  # noqa: BLE001
        status = "oom" if is_oom(exc) else "error"
        err = repr(exc)[:300]
    rec["seconds"] = (time.perf_counter() - t0) / steps
    if status != "ok":
        opt.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        rec["error"] = err
    else:
        rec["peak_alloc"] = torch.cuda.max_memory_allocated()
        rec["peak_reserved"] = torch.cuda.max_memory_reserved()
        free, total = torch.cuda.mem_get_info()
        rec["device_used"] = total - free
        rec["reserved_after"] = torch.cuda.memory_reserved()
    rec["status"] = status
    return status, rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--time-cap", type=float, default=6.0)
    ap.add_argument("--grid-cap-gib", type=float, default=24.0)
    ap.add_argument("--configs", default="all")
    ap.add_argument("--no-axis-grid", action="store_true")
    ap.add_argument("--max-batch", type=int, default=16384)
    ap.add_argument("--stop-gib", type=float, default=0.0,
                    help="stop a batch grid once the reserved memory exceeds this (0: run until out of memory)")
    args = ap.parse_args()

    dev = torch.device("cuda")
    spec = get_spec(args.model)
    torch.manual_seed(0)
    meta = {
        "model": spec.name,
        "family": spec.family,
        "domain": spec.domain,
        "axis_name": spec.axis,
        "axis_default": spec.default,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total": torch.cuda.get_device_properties(0).total_memory,
        "torch": torch.__version__,
    }
    configs = CONFIGS if args.configs == "all" else CONFIGS[:1]
    with open(args.out, "a", encoding="utf-8") as fh:

        def emit(rec: dict) -> None:
            fh.write(json.dumps({**meta, **rec}) + "\n")
            fh.flush()

        for opt_name, amp in configs:
            try:
                model = spec.build().to(dev).train()
                opt = make_optimizer(opt_name, model)
            except Exception as exc:  # noqa: BLE001
                emit({"status": "build_error", "error": repr(exc)[:300], "optimizer": opt_name, "amp": amp})
                return
            scaler = torch.amp.GradScaler("cuda", enabled=amp)
            params = sum(p.numel() for p in model.parameters())
            tag = {"optimizer": opt_name, "amp": amp, "params": params}
            axis_values = [spec.default]
            if (opt_name, amp) == CONFIGS[0] and not args.no_axis_grid:
                axis_values += list(spec.grid)
            for a in axis_values:
                fine = a == spec.default
                errors = 0
                for b in batch_grid(fine, args.max_batch):
                    status, rec = measure_one(spec, model, opt, scaler, amp, b, a, dev, args.steps)
                    emit({**tag, **rec})
                    if status == "oom":
                        break
                    if status == "error":
                        errors += 1
                        if errors >= 3 or b >= 8:
                            break
                        continue
                    if rec["seconds"] > args.time_cap:
                        break
                    if args.stop_gib and rec["peak_reserved"] > args.stop_gib * GIB:
                        break
                    if not fine and rec["peak_reserved"] > args.grid_cap_gib * GIB:
                        break
            del model, opt, scaler
            gc.collect()
            torch.cuda.empty_cache()
        emit({"status": "done"})


if __name__ == "__main__":
    main()
