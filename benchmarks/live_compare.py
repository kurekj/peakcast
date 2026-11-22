"""Live comparison of batch-size selection tools under a memory budget.

For one model and several budgets the script asks each tool for a batch size,
measures how long the tool took, and then checks the answer by training with
that batch size for a number of steps under the same budget. The budget is
enforced with torch.cuda.set_per_process_memory_fraction.

Tools:
  peakcast          peakcast.fit on the live model, then max_batch at the given risks
  lightning-power   Lightning Tuner.scale_batch_size(mode="power")
  lightning-binary  Lightning Tuner.scale_batch_size(mode="binsearch")
  toma              toma.explicit.batch: halve the batch on every out-of-memory error
  torch-max-mem     torch_max_mem.maximize_memory_utilization: the same strategy
  autobatch         ultralytics.utils.autobatch.autobatch (image models, whole device only)

The search-based tools need a starting batch size; they get the same generous
value (--start), which is what a user who does not know the answer would do.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import warnings

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure import is_oom, make_optimizer, run_steps  # noqa: E402
from zoo import get_spec  # noqa: E402

import peakcast  # noqa: E402

GIB = 1024**3


def clean() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def verify(spec, model, opt, b, a, dev, steps) -> str:
    """Train `steps` steps at batch `b`; 'ok', 'oom' or 'error'."""
    if b < 1:
        return "none"
    clean()
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    status = "ok"
    try:
        run_steps(spec, model, opt, scaler, False, b, a, dev, steps)
    except Exception as exc:  # noqa: BLE001
        status = "oom" if is_oom(exc) else "error"
    opt.zero_grad(set_to_none=True)
    clean()
    return status


def tool_peakcast(spec, model, opt, a, dev, cap, risks, calibration):
    step = peakcast.training_step(model, opt, spec.loss)
    try:
        law = peakcast.fit(step, lambda b: spec.make_batch(b, a, dev), budget=cap, restore=[model, opt],
                           calibration=calibration)
    except ValueError:  # batch normalisation over a single value: the model cannot train on one sample
        law = peakcast.fit(step, lambda b: spec.make_batch(b, a, dev), budget=cap, restore=[model, opt],
                           calibration=calibration, policy=peakcast.ProbePolicy(min_batch=2))
    out = {}
    for risk in risks:
        try:
            out[f"peakcast@{risk:g}"] = law.max_batch(cap, risk)
        except ValueError:  # the calibration cannot certify this risk
            continue
    return out, {"probes": len(law.probes), "settled": law.settled,
                 "largest_probe": max(p.batch for p in law.probes)}


def tool_lightning(spec, model, opt_name, a, dev, mode, start):
    import lightning as L
    from lightning.pytorch.tuner import Tuner
    from torch.utils.data import DataLoader

    class Module(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.net = model
            self.batch_size = 2

        def training_step(self, batch, batch_idx):
            return spec.loss(self.net, batch)

        def configure_optimizers(self):
            return make_optimizer(opt_name, self.net)

        def train_dataloader(self):
            return DataLoader(range(10**8), batch_size=self.batch_size,
                              collate_fn=lambda idx: spec.make_batch(len(idx), a, dev))

    module = Module()
    trainer = L.Trainer(accelerator="gpu", devices=1, max_epochs=1, logger=False,
                        enable_checkpointing=False, enable_progress_bar=False,
                        enable_model_summary=False)
    found = Tuner(trainer).scale_batch_size(module, mode=mode, steps_per_trial=3, init_val=2,
                                            max_trials=40)
    del trainer, module
    return int(found or 0)


def tool_toma(spec, model, opt, a, dev, start):
    from toma import explicit

    scaler = torch.amp.GradScaler("cuda", enabled=False)
    found = []

    def run(batchsize):
        try:
            run_steps(spec, model, opt, scaler, False, batchsize, a, dev, 3)
        finally:  # drop gradients of a failed step, as a careful user would
            opt.zero_grad(set_to_none=True)
        found.append(batchsize)

    explicit.batch(run, start)
    opt.zero_grad(set_to_none=True)
    return found[-1] if found else 0


def tool_torch_max_mem(spec, model, opt, a, dev, start):
    from torch_max_mem import maximize_memory_utilization

    scaler = torch.amp.GradScaler("cuda", enabled=False)

    @maximize_memory_utilization(parameter_name="batch_size")
    def run(batch_size: int):
        try:
            run_steps(spec, model, opt, scaler, False, batch_size, a, dev, 3)
        finally:
            opt.zero_grad(set_to_none=True)
        return batch_size

    found = run(batch_size=start)
    if isinstance(found, tuple):
        found = found[0]
    return int(found)


def tool_autobatch(model, a, fraction):
    from ultralytics.utils.autobatch import autobatch

    return int(autobatch(model, imgsz=int(a), fraction=fraction, batch_size=16))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--caps", default="8,24")
    ap.add_argument("--risks", default="0.01,0.05")
    ap.add_argument("--tools", default="peakcast,lightning-power,lightning-binary,toma,torch-max-mem,autobatch")
    ap.add_argument("--calibration", default="", help="directory with <family>.json calibrations")
    ap.add_argument("--verify-steps", type=int, default=20)
    ap.add_argument("--start", type=int, default=4096)
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    dev = torch.device("cuda")
    spec = get_spec(args.model)
    a = spec.default
    total = torch.cuda.get_device_properties(0).total_memory
    calibration = None
    if args.calibration:
        with open(os.path.join(args.calibration, spec.family + ".json"), encoding="utf-8") as fh:
            calibration = peakcast.Calibration.from_dict(json.load(fh))
    risks = [float(r) for r in args.risks.split(",")]
    tools = args.tools.split(",")
    torch.manual_seed(0)
    model = spec.build().to(dev).train()
    opt = make_optimizer("adamw", model)

    with open(args.out, "a", encoding="utf-8") as fh:

        def emit(rec: dict) -> None:
            fh.write(json.dumps({"model": spec.name, "family": spec.family, **rec}) + "\n")
            fh.flush()

        for cap_text in args.caps.split(","):
            whole = cap_text == "full"
            if whole:
                torch.cuda.set_per_process_memory_fraction(1.0)
                free, _ = torch.cuda.mem_get_info()
                cap = int(free + torch.cuda.memory_reserved())
            else:
                cap = int(float(cap_text) * GIB)
                if cap > total:
                    continue
                torch.cuda.set_per_process_memory_fraction(cap / total)
            for tool in tools:
                if tool == "autobatch" and not (whole and spec.domain == "image"):
                    continue
                clean()
                torch.cuda.reset_peak_memory_stats()
                extra: dict = {}
                t0 = time.perf_counter()
                try:
                    if tool == "peakcast":
                        answers, extra = tool_peakcast(spec, model, opt, a, dev, cap, risks, calibration)
                    elif tool == "lightning-power":
                        answers = {tool: tool_lightning(spec, model, "adamw", a, dev, "power", args.start)}
                    elif tool == "lightning-binary":
                        answers = {tool: tool_lightning(spec, model, "adamw", a, dev, "binsearch", args.start)}
                    elif tool == "toma":
                        answers = {tool: tool_toma(spec, model, opt, a, dev, args.start)}
                    elif tool == "torch-max-mem":
                        answers = {tool: tool_torch_max_mem(spec, model, opt, a, dev, args.start)}
                    elif tool == "autobatch":
                        answers = {"autobatch@0.6": tool_autobatch(model, a, 0.6)}
                        model.to(dev).train()
                    else:
                        raise ValueError(tool)
                    error = None
                except Exception as exc:  # noqa: BLE001
                    answers, error = {tool: 0}, repr(exc)[:300]
                seconds = time.perf_counter() - t0
                model.to(dev).train()
                opt.zero_grad(set_to_none=True)
                for name, b in answers.items():
                    clean()
                    torch.cuda.reset_peak_memory_stats()
                    status = verify(spec, model, opt, b, a, dev, args.verify_steps) if error is None else "tool_error"
                    emit({"tool": name, "cap": cap, "cap_label": cap_text, "batch": b, "seconds": seconds,
                          "verify": status, "peak_reserved": torch.cuda.max_memory_reserved(),
                          "peak_alloc": torch.cuda.max_memory_allocated(), "error": error, **extra})
        emit({"status": "done"})


if __name__ == "__main__":
    main()
