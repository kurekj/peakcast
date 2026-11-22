"""torchinfo baseline: "Estimated Total Size" of torchinfo.summary for every workload.

Usage: CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=16 python run_torchinfo.py [--out FILE]
The model is built on CPU exactly as in zoo.py. Tuple batches (x, y): only x is passed
(the model's forward takes x). Dict batches (Hugging Face) are passed whole via input_data,
so the forward also computes the loss from the labels.
"""
import argparse
import time

import torch
import torchinfo
from common import BATCHES, HERE, MODELS, done, emit
from zoo import get_spec

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=HERE + "/results_torchinfo.jsonl")
args = ap.parse_args()
torch.set_num_threads(16)
dev = torch.device("cpu")
have = done(args.out, "torchinfo")
for name in MODELS:
    spec = get_spec(name)
    for b in BATCHES:
        if (name, b) in have:
            continue
        try:
            torch.manual_seed(0)
            model = spec.build().to(dev).train()
            batch = spec.make_batch(b, spec.default, dev)
            data = batch if isinstance(batch, dict) else [batch[0]]
            t0 = time.perf_counter()
            st = torchinfo.summary(model, input_data=data, verbose=0)
            sec = time.perf_counter() - t0
            total = st.total_input + st.total_output_bytes + st.total_param_bytes
            note = (f"input={st.total_input} fwd_bwd={st.total_output_bytes} "
                    f"params={st.total_param_bytes} torchinfo={torchinfo.__version__} mode=train")
            emit(args.out, "torchinfo", name, b, int(total), None, sec, "ok", note)
        except Exception as exc:  # noqa: BLE001
            note = "torchinfo forward pass failed: " + repr(exc.__cause__ or exc)[:200]
            cfg = getattr(model, "config", None)
            if cfg is not None and name.startswith("hf_t5") and getattr(cfg, "decoder_start_token_id", None) is None:
                # The zoo builds T5 from a bare T5Config whose decoder_start_token_id is unset, so
                # the forward with labels fails for any tool (spec.loss fails in the same way).
                # For information only: the torchinfo total with the usual T5 value 0.
                try:
                    cfg.decoder_start_token_id = 0
                    st = torchinfo.summary(model, input_data=data, verbose=0)
                    alt = st.total_input + st.total_output_bytes + st.total_param_bytes
                    note += (f"; the zoo workload itself cannot run (spec.loss raises the same error);"
                             f" with config.decoder_start_token_id=0 torchinfo total={int(alt)}")
                except Exception as exc2:  # noqa: BLE001
                    note += "; also failed with decoder_start_token_id=0: " + repr(exc2)[:100]
            emit(args.out, "torchinfo", name, b, None, None, None, "failed", note)
        finally:
            model = batch = None
