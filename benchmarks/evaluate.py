"""Evaluate peakcast on the recorded measurements and write the numbers of the paper.

    python benchmarks/evaluate.py            # recompute results/summary.json
    python benchmarks/evaluate.py --check    # recompute and compare with the stored summary

Everything here is a replay (see replay.py): no GPU is needed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import replay  # noqa: E402

import peakcast  # noqa: E402

RESULTS = os.path.join(HERE, "results")
GRID_ONLY_CAPS = (8, 12, 16, 24, 40)
RISKS = (0.2, 0.1, 0.05, 0.02, 0.01)
MARGINS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)
GIB = replay.GIB


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def quantiles(values) -> dict:
    v = sorted(values)
    if not v:
        return {"n": 0}
    q = np.percentile(v, [5, 25, 50, 75, 95])
    return {"n": len(v), "p05": float(q[0]), "p25": float(q[1]), "median": float(q[2]), "p75": float(q[3]),
            "p95": float(q[4]), "mean": float(np.mean(v)), "min": float(v[0]), "max": float(v[-1])}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def decision(units, batch_of) -> dict:
    """Judge batch-size recommendations against the ground truth."""
    viol, verified, util, zero, n = 0, 0, [], 0, 0
    for u in units:
        b = batch_of(u)
        if b is None:
            continue
        n += 1
        if b > u.lo:
            viol += 1
            verified += b >= u.hi  # at or above the batch size that was seen to fail
        elif b < 1:
            zero += 1
            util.append(0.0)
        else:
            util.append(b / u.lo)
    lo, hi = wilson(viol, n)
    return {"n": n, "violations": viol, "violations_verified": verified,
            "violation_rate": viol / n if n else None,
            "violation_ci": [lo, hi], "no_answer": zero, "utilization": quantiles(util)}


def lofo_batches(units, risks) -> dict:
    """Leave-one-family-out: batch recommended for each unit at each risk."""
    families = sorted({u.workload.family for u in units})
    out: dict[float, dict[int, int | None]] = {r: {} for r in risks}
    for family in families:
        cal = peakcast.Calibration.from_scores([u.record() for u in units if u.workload.family != family])
        for i, u in enumerate(units):
            if u.workload.family != family:
                continue
            u.law._calibration = cal
            for r in risks:
                try:
                    out[r][i] = u.law.max_batch(u.cap, r, **u.kwargs)
                except (ValueError, KeyError):
                    out[r][i] = None
    return out


def fixed_margin_batch(u, margin: float) -> int:
    lo, hi = 0, 1
    while hi < (1 << 24) and u.law.expected(hi, **u.kwargs) * (1 + margin) <= u.cap:
        lo, hi = hi, hi * 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if u.law.expected(mid, **u.kwargs) * (1 + margin) <= u.cap:
            lo = mid
        else:
            hi = mid
    return lo


def autobatch_rule(u, fraction: float) -> int | None:
    """The rule of Ultralytics AutoBatch, replayed: a line through the reserved memory at
    batch sizes 1, 2, 4, 8 and 16, solved for `fraction` of the budget."""
    rows = u.workload.rows.get(u.size, {})
    pts = [(b, rows[b]["peak_reserved"]) for b in (1, 2, 4, 8, 16)
           if b in rows and rows[b]["status"] == "ok" and b <= u.lo]
    if len(pts) < 2:
        return None
    slope, intercept = np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)
    if slope <= 0:
        return None
    return max(0, int((u.cap * fraction - intercept) / slope))


# ----------------------------------------------------------------------------
# analyses
# ----------------------------------------------------------------------------


def accuracy(units) -> dict:
    """Error of the expected peak at measured batch sizes beyond the largest probe."""
    errs, by_domain, signed = [], defaultdict(list), []
    for u in units:
        rows = u.workload.rows.get(u.size, {})
        top = max(p.batch for p in u.law.probes if (p.size is None or p.size == u.size)) if not u.two_axes else 0
        for b, row in rows.items():
            if row["status"] != "ok" or b <= top or b > u.lo:
                continue
            rel = u.law.expected(b, **u.kwargs) / row["peak_alloc"] - 1.0
            errs.append(abs(rel))
            signed.append(rel)
            by_domain[u.workload.domain].append(abs(rel))
    return {"abs_rel_error": quantiles(errs), "signed_rel_error": quantiles(signed),
            "by_domain": {d: {"n": len(v), "mean": float(np.mean(v)), "median": float(np.median(v))}
                          for d, v in sorted(by_domain.items())}}


def structure(grid) -> dict:
    """Check the two properties the method relies on, on the fine batch grids."""
    sub_ok = sub_n = conv_ok = conv_n = 0
    floor_share = []
    for w in grid.values():
        rows = w.rows.get(w.axis_default, {})
        pts = sorted((b, r["peak_alloc"]) for b, r in rows.items() if r["status"] == "ok")
        if len(pts) < 4:
            continue
        b = np.array([p[0] for p in pts], float)
        y = np.array([p[1] for p in pts], float)
        # sub-homogeneity: memory per sample does not grow with the batch (2 % tolerance)
        per = y / b
        sub_n += len(per) - 1
        sub_ok += int(np.sum(per[1:] <= per[:-1] * 1.02))
        # convexity: secant slopes do not decrease (5 % of the larger slope as tolerance)
        slope = np.diff(y) / np.diff(b)
        conv_n += len(slope) - 1
        conv_ok += int(np.sum(slope[1:] >= slope[:-1] - 0.05 * np.maximum(np.abs(slope[1:]), np.abs(slope[:-1]))))
        flat = int(np.sum(y <= y[0] * 1.03))
        floor_share.append(flat > 2)
    return {"sub_homogeneous_share": sub_ok / sub_n, "convex_share": conv_ok / conv_n,
            "workloads": len(floor_share), "workloads_with_floor": int(sum(floor_share))}


def reserve_ratio(units) -> dict:
    """How much more than the allocated peak a budget must hold (ground truth)."""
    need = []
    for u in units:
        rows = u.workload.rows.get(u.size, {})
        if u.lo in rows and rows[u.lo]["status"] == "ok":
            need.append(u.cap / rows[u.lo]["peak_alloc"])
    return quantiles(need)


def grid_only(grid) -> dict:
    """Accuracy of the law on a device that has measurements but no ground truth.

    Probes are replayed under budgets on the allocated peak, and the law is
    compared with every measurement beyond the largest probe that stays under
    the budget.
    """
    errs, extra, errs_axis, probes, seconds, settled = [], [], [], [], [], []
    for w in grid.values():
        rows = w.rows.get(w.axis_default, {})
        for cap in GRID_ONLY_CAPS:
            law, cost, _ = replay.replay_law(w, cap * GIB)
            if law is None:
                continue
            top = max(p.batch for p in law.probes)
            top_peak = max(p.allocated for p in law.probes)
            beyond = [(b, r["peak_alloc"]) for b, r in rows.items()
                      if r["status"] == "ok" and b > top and r["peak_alloc"] <= cap * GIB]
            if not beyond:
                continue
            probes.append(len(law.probes))
            seconds.append(cost)
            settled.append(law.settled)
            extra.append(max(y for _, y in beyond) / top_peak)
            errs.extend(abs(law.expected(b) / y - 1.0) for b, y in beyond)
        law, _, _ = replay.replay_law(w, GRID_ONLY_CAPS[-1] * GIB, two_axes=True)
        if law is None:
            continue
        inside = {p.size for p in law.probes}
        for size, by_batch in w.rows.items():
            if size <= max(inside):
                continue
            errs_axis.extend(abs(law.expected(b, **{w.axis_name: size}) / r["peak_alloc"] - 1.0)
                             for b, r in by_batch.items()
                             if r["status"] == "ok" and r["peak_alloc"] <= GRID_ONLY_CAPS[-1] * GIB)
    return {"grid_only": True, "workloads": len(grid), "models": len({w.model for w in grid.values()}),
            "families": len({w.family for w in grid.values()}), "budgets_gib": list(GRID_ONLY_CAPS),
            "probes": quantiles(probes), "probe_seconds": quantiles(seconds),
            "settled_share": float(np.mean(settled)), "extrapolation": quantiles(extra),
            "accuracy": {"abs_rel_error": quantiles(errs)}, "accuracy_beyond_axis": quantiles(errs_axis),
            "structure": structure(grid)}


def analyse(name: str, grid, truth, policy=None, extended: bool = True, missing=None) -> dict:
    out: dict = {}
    units = replay.build_units(grid, truth, two_axes=False, policy=policy, missing=missing)
    out["units"] = len(units)
    out["workloads"] = len({u.workload.key for u in units})
    out["models"] = len({u.workload.model for u in units})
    out["families"] = len({u.workload.family for u in units})
    out["settled_share"] = float(np.mean([u.law.settled for u in units]))
    out["probe_oom_share"] = float(np.mean([u.probe_oom for u in units]))
    out["probes"] = quantiles([len(u.law.probes) for u in units])
    out["probe_seconds"] = quantiles([u.probe_seconds for u in units])
    out["extrapolation"] = quantiles([u.score()[0] for u in units])
    out["models_in_grid"] = len({w.model for w in grid.values()})
    out["score"] = quantiles([u.score()[1] for u in units])
    out["accuracy"] = accuracy(units)
    batches = lofo_batches(units, RISKS)
    out["answer_is_largest_probe"] = {
        f"{r:g}": sum(1 for i, u in enumerate(units)
                      if batches[r].get(i) == max(p.batch for p in u.law.probes)) for r in RISKS}
    out["lofo"] = {}
    index = {id(u): i for i, u in enumerate(units)}
    for r in RISKS:
        out["lofo"][f"{r:g}"] = decision(units, lambda u, r=r: batches[r].get(index[id(u)]))
        for cap in sorted({u.cap for u in units}):
            sub = [u for u in units if u.cap == cap]
            out["lofo"][f"{r:g}@{cap // GIB}"] = decision(sub, lambda u, r=r: batches[r].get(index[id(u)]))
    if not extended:
        return out, units
    out["by_domain"] = {}
    for domain in sorted({u.workload.domain for u in units}):
        sub = [u for u in units if u.workload.domain == domain]
        out["by_domain"][domain] = {f"{r:g}": decision(sub, lambda u, r=r: batches[r].get(index[id(u)]))
                                    for r in (0.05, 0.01)}
    out["by_config"] = {}
    for cfg in sorted({(u.workload.optimizer, u.workload.amp) for u in units}):
        sub = [u for u in units if (u.workload.optimizer, u.workload.amp) == cfg]
        out["by_config"][f"{cfg[0]}{'+amp' if cfg[1] else ''}"] = {
            f"{r:g}": decision(sub, lambda u, r=r: batches[r].get(index[id(u)])) for r in (0.05, 0.01)}
    out["margins"] = {f"{m:g}": decision(units, lambda u, m=m: fixed_margin_batch(u, m)) for m in MARGINS}
    out["autobatch_rule"] = {f"{f:g}": decision(units, lambda u, f=f: autobatch_rule(u, f))
                             for f in (0.6, 0.8, 0.9, 1.0)}
    out["halving"] = decision(units, lambda u: 1 << (u.lo.bit_length() - 1))
    out["required_over_allocated"] = reserve_ratio(units)
    out["structure"] = structure(grid)
    return out, units


def two_axes(grid, truth, missing=None) -> dict:
    units = replay.build_units(grid, truth, two_axes=True, missing=missing)
    out: dict = {"units": len(units), "models": len({u.workload.model for u in units})}
    if not units:
        return out
    risks = (0.2, 0.1, 0.05)
    batches = lofo_batches(units, risks)
    index = {id(u): i for i, u in enumerate(units)}
    for kind in ("i", "s"):
        sub = [u for u in units if u.kind == kind]
        out[kind] = {"units": len(sub), "accuracy": accuracy(sub), "score": quantiles([u.score()[1] for u in sub]),
                     "lofo": {f"{r:g}": decision(sub, lambda u, r=r: batches[r].get(index[id(u)])) for r in risks}}
    out["by_axis"] = {}
    for axis in sorted({u.workload.axis_name for u in units}):
        sub = [u for u in units if u.workload.axis_name == axis and u.kind == "s"]
        if sub:
            out["by_axis"][axis] = {"units": len(sub), "accuracy": accuracy(sub)["abs_rel_error"],
                                    "lofo": {f"{r:g}": decision(sub, lambda u, r=r: batches[r].get(index[id(u)]))
                                             for r in risks}}
    return out, units


def transfer(source_units, target_units) -> dict:
    """Calibrate on one device and decide on another."""
    cal = peakcast.Calibration.from_scores([u.record() for u in source_units])
    out = {}
    for r in RISKS:
        def batch(u, r=r):
            u.law._calibration = cal
            try:
                return u.law.max_batch(u.cap, r, **u.kwargs)
            except (ValueError, KeyError):
                return None
        out[f"{r:g}"] = decision(target_units, batch)
        out[f"{r:g}"]["models_out_of_memory"] = sorted(
            {u.workload.model for u in target_units if (batch(u) or 0) > u.lo})
    return out


def ablation(grid, truth) -> dict:
    out = {}
    for fraction in (0.05, 0.1, 0.2, 0.3, 0.4):
        res, _ = analyse("ablation", grid, truth, peakcast.ProbePolicy(fraction=fraction), extended=False)
        out[f"{fraction:g}"] = {k: res[k] for k in ("settled_share", "probe_oom_share", "probes", "probe_seconds",
                                                     "extrapolation")}
        out[f"{fraction:g}"]["abs_rel_error"] = res["accuracy"]["abs_rel_error"]
        out[f"{fraction:g}"]["lofo"] = {r: res["lofo"][r] for r in ("0.05", "0.01")}
    return out


def estimators(grid, truth, results: str) -> dict:
    """Published estimators against the measured peaks (28 models, batch sizes 8, 32, 128)."""
    path = os.path.join(results, "baselines", "results.jsonl")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    out: dict = {}
    bins = [0, 8000, 16000, 24000, 32000, 40000, 50000]
    cases = sorted({(r["model"], r["batch"]) for r in records})
    measured = {}
    for model, batch in cases:
        w = grid.get(f"{model}|adamw|0")
        row = w.rows.get(w.axis_default, {}).get(batch) if w else None
        if row and row["status"] == "ok":
            measured[(model, batch)] = row
    out["cases"] = len(cases)
    out["cases_measured"] = len(measured)
    for tool in ("torchinfo", "xmem"):
        rel_alloc, rel_res, seconds, ok = [], [], [], 0
        for r in records:
            if r["tool"] != tool or r["status"] != "ok":
                continue
            ok += 1
            seconds.append(r["seconds"])
            row = measured.get((r["model"], r["batch"]))
            if row:
                rel_alloc.append(r["estimate_bytes"] / row["peak_alloc"] - 1.0)
                rel_res.append(r["estimate_bytes"] / row["peak_reserved"] - 1.0)
        out[tool] = {"answered": ok, "compared": len(rel_alloc), "seconds": quantiles(seconds),
                     "vs_allocated": quantiles([abs(x) for x in rel_alloc]),
                     "vs_reserved": quantiles([abs(x) for x in rel_res]),
                     "under_allocated_share": float(np.mean([x < 0 for x in rel_alloc])) if rel_alloc else None,
                     "under_reserved_share": float(np.mean([x < 0 for x in rel_res])) if rel_res else None}
    hit = n = answered = 0
    seconds = []
    for r in records:
        if r["tool"] != "gpumemnet" or r["status"] != "ok":
            continue
        answered += 1
        seconds.append(r["seconds"])
        row = measured.get((r["model"], r["batch"]))
        if not row:
            continue
        n += 1
        low = int(r["estimate_bin"].split("-")[0])
        high = bins[bins.index(low) + 1]
        used = row["device_used"] / 2**20
        hit += low <= used < high
    out["gpumemnet"] = {"answered": answered, "compared": n, "correct_bin": hit, "seconds": quantiles(seconds),
                        "correct_bin_share": hit / n if n else None}
    out["llmem"] = {"answered": 0}
    # peakcast on the same cases: laws replayed with a 24 GiB budget
    cap = 24 * GIB
    rel, beyond = [], []
    seconds = []
    for (model, batch), row in measured.items():
        w = grid[f"{model}|adamw|0"]
        law, cost, _ = replay.replay_law(w, cap, truth=truth)
        if law is None:
            continue
        err = abs(law.expected(batch) / row["peak_alloc"] - 1.0)
        rel.append(err)
        seconds.append(cost)
        if batch > max(p.batch for p in law.probes):
            beyond.append(err)
    out["peakcast"] = {"answered": len(rel), "compared": len(rel), "vs_allocated": quantiles(rel),
                       "beyond_probes": quantiles(beyond), "seconds": quantiles(seconds)}
    return out


def same(a, b, tolerance: float = 1e-6) -> bool:
    """Equality of two summaries up to the rounding of floating point numbers."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k], tolerance) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y, tolerance) for x, y in zip(a, b, strict=True))
    if isinstance(a, float) or isinstance(b, float):
        return abs(a - b) <= tolerance * max(1.0, abs(a), abs(b))
    return a == b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--missing", default="", help="write probes that were never measured to this file")
    args = ap.parse_args()
    summary: dict = {}
    missing: list = []
    main_units = None
    first = replay.main_device(args.results)
    for device in sorted(os.listdir(args.results), key=lambda d: (d != first, d)):
        directory = os.path.join(args.results, device)
        if not os.path.exists(os.path.join(directory, "grid.csv")):
            continue
        grid = replay.load_grid(directory)
        if device != first and not os.path.exists(os.path.join(directory, "truth.csv")):
            # a device without ground truth: accuracy of the law only
            with open(os.path.join(directory, "meta.json"), encoding="utf-8") as fh:
                meta = json.load(fh)
            summary[device] = grid_only(grid)
            summary[device]["device"] = {k: meta.get(k) for k in ("gpu", "gpu_total", "torch")}
            summary[device]["failed_models"] = len(meta.get("failed_models", []))
            continue
        truth = replay.load_truth(directory)
        with open(os.path.join(directory, "meta.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        res, units = analyse(device, grid, truth, missing=missing)
        res["device"] = {k: meta.get(k) for k in ("gpu", "gpu_total", "torch")}
        res["failed_models"] = len(meta.get("failed_models", []))
        two = two_axes(grid, truth, missing)
        if isinstance(two, tuple):
            res["two_axes"] = two[0]
        else:
            res["two_axes"] = two
        if device == first:
            res["ablation"] = ablation(grid, truth)
            res["estimators"] = estimators(grid, truth, args.results)
            main_units = units
        elif main_units:
            # a second device with ground truth: the calibration of each device decides on the other
            res["transfer_from_main"] = transfer(main_units, units)
            res["transfer_to_main"] = transfer(units, main_units)
        res["budgets_gib"] = sorted({u.cap // GIB for u in units})
        summary[device] = res
    summary = {"main": first, **summary}
    text = json.dumps(summary, indent=1, sort_keys=True)
    path = os.path.join(args.results, "summary.json")
    if args.missing:
        with open(args.missing, "w", encoding="utf-8") as fh:
            json.dump(sorted(set(missing)), fh)
        print(f"{len(set(missing))} probes were never measured")
    if args.check:
        with open(path, encoding="utf-8") as fh:
            stored = fh.read()
        if not same(json.loads(stored), json.loads(text)):
            print("summary.json does not match the recomputed results")
            sys.exit(1)
        print("summary.json reproduced")
        return
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    for device, res in summary.items():
        if device == "main" or res.get("grid_only"):
            continue
        print(device, "units", res["units"], "workloads", res["workloads"], "settled", round(res["settled_share"], 3))
        for r in RISKS:
            d = res["lofo"][f"{r:g}"]
            if not d["n"]:
                continue
            print(f"  risk {r:g}: violations {d['violations']}/{d['n']} = {d['violation_rate']:.4f}, "
                  f"median utilization {d['utilization'].get('median', float('nan')):.3f}")


if __name__ == "__main__":
    main()
