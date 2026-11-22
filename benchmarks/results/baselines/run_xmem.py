"""xMem baseline (https://github.com/Stone-ResearchLife/xMem): one process = one workload.

Usage (CPU only):
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 env_xmem/bin/python run_xmem.py --model resnet18 --batch 8

What is xMem's own code and what is ours:
  * profiling: xMem's ModelTrainer + ProfilerPlugin + its unmodified training loops
    (exp/trainer/train_loop.py: conv_train_loop for (x, y) workloads, transformer_train_loop for
    Hugging Face dict workloads), called with the same arguments as FastRunner.train_on_cpu
    (on_cpu=True, iterations=2, zero_grad_mode=0). FastRunner itself is bypassed because it only
    accepts model names from xMem's fixed list / the Hugging Face hub and its own data loaders
    (CIFAR-10 at 86x86, wikitext); we pass the zoo model, three synthetic zoo batches and AdamW.
  * estimation: xMem's XMem.estimate(profiler_file, output_only=True), unmodified
    (Estimator for (x, y) workloads, TrainerEstimator for Hugging Face workloads).
  * TrainerEstimator rebuilds the model from a hub name to list the parameter tensors. Our models
    are randomly initialised from configs, so the config is saved to a local directory (passed as
    the "model name") and utils.huggingface.suggest_automodel_class is replaced by a shim that
    returns the model's own class (the original asks the hub for the pipeline tag).
  * conv_train_loop needs logits with the class dimension at position 1, so two workloads get a
    thin output adapter: fcn_resnet50 (out["out"]) and custom_lstm_lm (logits and labels flattened
    as in zoo._lm_plain_loss).
  * HOME is redirected to ./xmem_home because xMem writes its traces to ~/DL-Estimator.
"""
import argparse
import os
import sys
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["HOME"] = os.path.join(HERE, "xmem_home")
os.makedirs(os.environ["HOME"], exist_ok=True)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.join(HERE, "xMem"))

import torch  # noqa: E402
from common import emit  # noqa: E402
from torch import nn  # noqa: E402
from zoo import get_spec  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--batch", type=int, required=True)
ap.add_argument("--gpu-gb", type=float, default=46068 / 1024)  # NVIDIA L40S as reported by nvidia-smi
ap.add_argument("--out", default=HERE + "/results_xmem.jsonl")
ap.add_argument("--keep-trace", action="store_true")
args = ap.parse_args()
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))


class Adapter(nn.Module):
    def __init__(self, model, fn):
        super().__init__()
        self.model, self.fn = model, fn

    def forward(self, x):
        return self.fn(self.model(x))


def main():
    import utils.huggingface as xh
    from exp.trainer import ModelTrainer
    from exp.trainer.plugins import ProfilerPlugin
    from perf_estimator.config import Config
    from perf_estimator.xmem import XMem
    from utils import search_profiler_file

    name, b = args.model, args.batch
    spec = get_spec(name)
    dev = torch.device("cpu")
    torch.manual_seed(0)
    model = spec.build().train()
    batches = [spec.make_batch(b, spec.default, dev) for _ in range(3)]
    is_hf = isinstance(batches[0], dict)
    conf = Config(save2tmp=False, run_id=f"{name}-b{b}-{uuid.uuid4().hex[:6]}")
    conf.trainer.lr = 1e-5
    adapter = "none"
    if is_hf:
        cfg_dir = conf.base_dir.joinpath("hf_config")
        model.config.save_pretrained(cfg_dir)
        conf.trainer.huggingface_enable = True
        conf.trainer.huggingface_model_name = str(cfg_dir)
        cls = type(model)

        class _Shim:
            from_config = staticmethod(lambda config: cls(config))

        xh.suggest_automodel_class = lambda model_name: _Shim
    else:
        conf.trainer.huggingface_enable = False
        conf.trainer.huggingface_model_name = None
        if name == "fcn_resnet50":
            model, adapter = Adapter(model, lambda o: o["out"]), 'out["out"]'
        elif name == "custom_lstm_lm":
            model = Adapter(model, lambda o: o.reshape(-1, o.shape[-1]))
            batches = [(x, y.reshape(-1)) for x, y in batches]
            adapter = "flatten logits and labels"

    # profiling on CPU, as in xMem's FastRunner.train_on_cpu
    cpu_config = conf.model_copy(deep=True)
    cpu_config.task_id = f"CPU_{uuid.uuid4().hex[:3]}"
    trainer = ModelTrainer(
        model=model,
        data_loader=batches,
        on_cpu=True,
        gpu_id=0,
        iterations=2,
        optimiser=torch.optim.AdamW,
        config=cpu_config,
        plugins=[ProfilerPlugin(config=cpu_config)],
        zero_grad_mode=conf.trainer.zero_out,
    )
    t0 = time.perf_counter()
    trainer.train(is_transformer=is_hf)
    t_prof = time.perf_counter() - t0
    files = search_profiler_file(conf.base_dir)
    if not files:
        raise FileNotFoundError("no profiler trace written")
    trace = str(files[-1])
    trace_mb = os.path.getsize(trace) / 1e6
    del model, trainer, batches

    def estimate(cap):
        t = time.perf_counter()
        res = XMem(batch_size=b, max_gpu_memory_in_gb=cap, config=conf).estimate(
            profiler_file=trace, output_only=True
        )
        return res, time.perf_counter() - t

    res, t_est = estimate(args.gpu_gb)
    oom_at_cap = bool(res["OOM"])
    cap_used = args.gpu_gb
    if oom_at_cap:  # xMem's verdict at the device capacity is OOM: get the unconstrained peak too
        cap_used = 4096
        res, t_est2 = estimate(cap_used)
        t_est += t_est2
    note = (
        f"path={'transformer' if is_hf else 'cnn'} adapter={adapter} "
        f"peak_segment={res['memory']['segment']} peak_tensor={res['memory']['tensor']} "
        f"oom_at_{args.gpu_gb:.2f}GB={oom_at_cap} cap_used_gb={cap_used} oom_at_cap_used={res['OOM']} "
        f"profile_s={t_prof:.1f} estimate_s={t_est:.1f} trace_mb={trace_mb:.0f} "
        f"torch={torch.__version__} threads={torch.get_num_threads()}"
    )
    status = "failed" if res["OOM"] else "ok"
    emit(args.out, "xmem", name, b, None if res["OOM"] else int(res["memory"]["segment"]), None,
         t_prof + t_est, status, note)
    if not args.keep_trace:
        os.remove(trace)


if __name__ == "__main__":
    t_start = time.perf_counter()
    try:
        main()
    except BaseException as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        emit(args.out, "xmem", args.model, args.batch, None, None, time.perf_counter() - t_start,
             "failed", repr(exc)[:400])
        sys.exit(1)
