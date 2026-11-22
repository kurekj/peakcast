"""Check the measurement protocol of measure.py.

measure.py measures every batch size of a model in one process, two training
steps per size. This script repeats single sizes in a fresh process and for
many steps, and records the peaks after 2 steps and after `--steps` steps, so
that both simplifications can be quantified:

    python benchmarks/validate_protocol.py --model resnet50 --batch 64 --out protocol.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure import make_optimizer, run_steps  # noqa: E402
from zoo import get_spec  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=30)
    args = ap.parse_args()
    dev = torch.device("cuda")
    spec = get_spec(args.model)
    torch.manual_seed(0)
    model = spec.build().to(dev).train()
    opt = make_optimizer("adamw", model)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    torch.cuda.reset_peak_memory_stats()
    run_steps(spec, model, opt, scaler, False, args.batch, spec.default, dev, 2)
    early = (torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved())
    run_steps(spec, model, opt, scaler, False, args.batch, spec.default, dev, args.steps - 2)
    late = (torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved())
    with open(args.out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"model": spec.name, "batch": args.batch, "steps": args.steps,
                             "alloc_2": early[0], "reserved_2": early[1],
                             "alloc_n": late[0], "reserved_n": late[1]}) + "\n")


if __name__ == "__main__":
    main()
