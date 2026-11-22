"""Variable-length training under a memory budget: length buckets against a fixed batch.

For one text model the script fits a two-axis law (batch and sequence length)
from probes at short lengths, plans one batch size per length bucket with
peakcast.bucket_plan, and trains on a synthetic corpus with log-normally
distributed lengths under an allocator limit. Batches of different shapes are
interleaved in random order, which is the hard case for the caching allocator.

Two baselines use one batch size for the whole corpus, the largest that the
same law allows for the longest sequence: `fixed` pads every batch to the
longest sequence of the corpus, `fixed-sorted` groups sequences of similar
length and pads each batch to its own longest sequence.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure import is_oom, make_optimizer  # noqa: E402
from zoo import get_spec  # noqa: E402

import peakcast  # noqa: E402

GIB = 1024**3


def corpus(n: int, longest: int, seed: int = 0) -> list[int]:
    rng = random.Random(seed)
    return [min(longest, max(8, int(rng.lognormvariate(4.6, 0.8)))) for _ in range(n)]


def train(step, spec, dev, batches, limit: int | None):
    """Run the batches [(batch size, padded length)]; return (seconds, samples, ooms, peak reserved)."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    ooms, samples = 0, 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for count, (b, length) in enumerate(batches):
        if limit is not None and count >= limit:
            break
        try:
            step(spec.make_batch(b, int(length), dev))
            samples += b
        except Exception as exc:  # noqa: BLE001
            if not is_oom(exc):
                raise
            ooms += 1
            gc.collect()
            torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return time.perf_counter() - t0, samples, ooms, torch.cuda.max_memory_reserved()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--caps", default="16")
    ap.add_argument("--risk", type=float, default=0.05)
    ap.add_argument("--calibration", default="")
    ap.add_argument("--samples", type=int, default=20000)
    ap.add_argument("--longest", type=int, default=512)
    ap.add_argument("--buckets", type=int, default=8)
    ap.add_argument("--limit", type=int, default=150, help="batches to run per strategy")
    args = ap.parse_args()

    dev = torch.device("cuda")
    spec = get_spec(args.model)
    total = torch.cuda.get_device_properties(0).total_memory
    calibration = None
    if args.calibration:
        with open(os.path.join(args.calibration, spec.family + ".json"), encoding="utf-8") as fh:
            calibration = peakcast.Calibration.from_dict(json.load(fh))
    lengths = corpus(args.samples, args.longest)
    torch.manual_seed(0)
    model = spec.build().to(dev).train()
    opt = make_optimizer("adamw", model)
    step = peakcast.training_step(model, opt, spec.loss)

    with open(args.out, "a", encoding="utf-8") as fh:
        for cap_gib in (float(c) for c in args.caps.split(",")):
            cap = int(cap_gib * GIB)
            torch.cuda.set_per_process_memory_fraction(cap / total)
            t0 = time.perf_counter()
            law = peakcast.fit(step, lambda b, n: spec.make_batch(b, int(n), dev), axis="length",
                               sizes=(32, 64, 128), budget=cap, restore=[model, opt], calibration=calibration)
            fit_seconds = time.perf_counter() - t0
            plan = peakcast.bucket_plan(law, lengths, buckets=args.buckets, risk=args.risk)
            sampler = peakcast.BucketBatchSampler(lengths, plan, seed=0)
            bucketed = [(len(idx), sampler.limit_of(lengths[idx[0]])) for idx in sampler]
            fixed_batch = law.max_batch(cap, args.risk, length=max(lengths))
            order = list(range(len(lengths)))
            random.Random(0).shuffle(order)
            chunks = [order[i:i + fixed_batch] for i in range(0, len(order), fixed_batch)]
            fixed = [(len(c), max(lengths)) for c in chunks]
            by_length = sorted(range(len(lengths)), key=lambda i: lengths[i])
            chunks = [by_length[i:i + fixed_batch] for i in range(0, len(by_length), fixed_batch)]
            random.Random(0).shuffle(chunks)
            fixed_sorted = [(len(c), max(lengths[i] for i in c)) for c in chunks]
            for name, batches in (("buckets", bucketed), ("fixed", fixed), ("fixed-sorted", fixed_sorted)):
                seconds, samples, ooms, peak = train(step, spec, dev, batches, args.limit)
                rec = {"model": spec.name, "family": spec.family, "cap": cap, "risk": args.risk,
                       "strategy": name, "steps_per_epoch": len(batches), "batches_run": min(args.limit, len(batches)),
                       "seconds": seconds, "samples": samples, "ooms": ooms, "peak_reserved": peak,
                       "fit_seconds": fit_seconds, "probes": len(law.probes), "settled": law.settled,
                       "fixed_batch": fixed_batch,
                       "plan": [[b.max_size, b.batch, b.count] for b in plan],
                       "law": law.summary()}
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
        fh.write(json.dumps({"model": spec.name, "status": "done"}) + "\n")


if __name__ == "__main__":
    main()
