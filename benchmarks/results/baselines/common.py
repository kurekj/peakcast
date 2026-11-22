"""Shared helpers for the baseline runs: workload list and result records."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # ~/peakcast_work, for zoo.py

MODELS = """resnet18 resnet50 resnet152 vgg16 densenet121 mobilenet_v2 efficientnet_b0
convnext_tiny regnet_y_1_6gf vit_b_16 swin_t timm_deit_small_patch16_224 timm_mixer_b16_224
hf_bert_base hf_bert_large hf_distilbert hf_roberta_base hf_gpt2_small hf_gpt2_medium
hf_llama_m hf_pythia_160m hf_t5_small hf_t5_base hf_bart_base custom_lstm_lm custom_unet_32
fcn_resnet50 r3d_18""".split()
BATCHES = (8, 32, 128)


def done(path, tool):
    out = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r["tool"] == tool:
                    out.add((r["model"], r["batch"]))
    return out


def emit(path, tool, model, batch, estimate_bytes=None, estimate_bin=None, seconds=None,
         status="ok", note=""):
    rec = {"tool": tool, "model": model, "batch": batch, "estimate_bytes": estimate_bytes,
           "estimate_bin": estimate_bin, "seconds": seconds, "status": status, "note": note}
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
