"""GPUMemNet baseline (https://github.com/itu-rad/GPUMemNet): predicted GPU-memory bin per workload.

Usage (CPU only):
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 env_gpumemnet/bin/python run_gpumemnet.py

The repository has no user-facing estimator ("How to Use GPUMemNet: TODO" in its README), but it
ships trained checkpoints and the test scripts of the paper in Test_Pipeline/. This wrapper calls
the functions of those scripts unchanged (loaded with importlib, working directory Test_Pipeline):

  * CNN workloads: Test_Pipeline/01-Ensemble-mlp4cnn.py
      torchsummary.summary(model, input_size, batch_size) text  (as Test_Pipeline/cnn_models/*.py)
      -> extract_model_info -> prepare_features_for_model -> infer_mlp4cnn
      (checkpoint Ensemble/trained_models/mlp4cnn/mlp4cnn.ckpt)
  * Transformer workloads: Test_Pipeline/03-Ensemble-transformer4transformer.py
      modelsummary hierarchical summary text (as NeuroNetGen_scripts/Transformer/transformer.py)
      -> extract_model_info -> prepare_features_for_model -> LayerSequenceDataset(max_seq_len=314)
      -> infer_transformer4cnn (checkpoint Ensemble/trained_models/transformer4transformer/...ckpt)

Ours: building the zoo model, producing the summary text, appending the "Total params:" line to
the hierarchical summary (modelsummary prints it only after a forward pass; the number is the
value returned by modelsummary.hierarchicalsummary), and mapping the class index to a bin.
Bins (Ensemble/dataloaders/dataloaders4mlp.py): [0, 8000, 16000, 24000, 32000, 40000, 50000] MiB,
left-closed; out-of-memory runs were labelled 42000 MiB, i.e. the last class.
"""
import argparse
import contextlib
import importlib.util
import io
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch  # noqa: E402
from common import BATCHES, MODELS, done, emit  # noqa: E402
from zoo import get_spec  # noqa: E402

CNN = {"resnet18", "resnet50", "resnet152", "vgg16", "densenet121", "mobilenet_v2",
       "efficientnet_b0", "convnext_tiny", "regnet_y_1_6gf", "custom_unet_32", "fcn_resnet50"}
VIT = {"vit_b_16", "timm_deit_small_patch16_224"}  # sequence length = 196 patches + class token
UNSUPPORTED = {
    "swin_t": "hierarchical vision transformer: not a CNN (Conv2d/BatchNorm2d features) and no "
              "single sequence length for the transformer parser",
    "timm_mixer_b16_224": "MLP-Mixer on image patches: the MLP estimator was trained on plain "
                          "fully connected networks with its own generator; no parser for this model",
    "custom_lstm_lm": "recurrent model: the parsers know no LSTM layers (estimators exist only "
                      "for MLP, CNN and Transformer)",
    "r3d_18": "3D CNN: the CNN parser only reads Conv2d/BatchNorm2d lines with 4-D output shapes",
}
EXPLAIN = {
    "densenet121": "torchsummary 1.5.1 (the summary tool of the repository's CNN test scripts) "
                   "crashes on modules that receive a list of tensors (dense blocks)",
    "fcn_resnet50": "torchsummary 1.5.1 crashes on modules that return a dict",
    "vit_b_16": "03-Ensemble-transformer4transformer.py: encode_layer_transformer has no code for "
                "the NonDynamicallyQuantizableLinear entry that the parser emits for "
                "nn.MultiheadAttention",
    "hf_llama_m": "03-Ensemble-transformer4transformer.py: prepare_features_for_model requires a "
                  "LayerNorm count; the model has only RMSNorm layers",
    "hf_bart_base": "03-Ensemble-transformer4transformer.py: prepare_features_for_model requires a "
                    "Dropout count; the model has no nn.Dropout modules (functional dropout)",
}
BINS = [0, 8000, 16000, 24000, 32000, 40000, 50000]

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=HERE + "/results_gpumemnet.jsonl")
args = ap.parse_args()
SUMDIR = os.path.join(HERE, "gpumemnet_summaries")
os.makedirs(SUMDIR, exist_ok=True)
os.chdir(os.path.join(HERE, "GPUMemNet", "Test_Pipeline"))
sys.path.insert(0, os.getcwd())


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


cnn = load("01-Ensemble-mlp4cnn.py", "gmn_cnn")
trf = load("03-Ensemble-transformer4transformer.py", "gmn_trf")
from modelsummary.hierarchicalsummary import hierarchicalsummary  # noqa: E402
from torchsummary import summary as torchsummary_summary  # noqa: E402


def bin_name(k):
    return f"{BINS[k]}-{BINS[k + 1]} MiB"


def run_cnn(name, spec, model, b):
    path = os.path.join(SUMDIR, f"{name}_{b}.model")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        torchsummary_summary(model, (3, spec.default, spec.default), batch_size=b, device="cpu")
    with open(path, "w") as fh:
        fh.write(buf.getvalue())
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        features = cnn.extract_model_info(path, b)
        if features[1] == "None":
            raise RuntimeError("parser failed: " + log.getvalue()[-200:])
        feats = cnn.prepare_features_for_model(features, b)
        tensor = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
        k = cnn.infer_mlp4cnn(tensor[0])
    return k, f"estimator=mlp4cnn features={[round(float(f), 4) for f in feats]}"


def run_trf(name, spec, model, b, seq):
    path = os.path.join(SUMDIR, f"{name}_bs{b}_sl{seq}.txt")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        count = hierarchicalsummary(model)
    with open(path, "w") as fh:
        fh.write(buf.getvalue() + "\n" + "=" * 71 + f"\nTotal params: {count:,}\n")
    with contextlib.redirect_stdout(io.StringIO()):
        features = trf.extract_model_info(path, b, seq)
        _, feats = trf.prepare_features_for_model(features, b)
        n_layers = len(feats[0])
        padded, extra = trf.LayerSequenceDataset(feats, max_seq_len=trf.max_seq_len)
        k = trf.infer_transformer4cnn(padded, extra)
    return k, (f"estimator=transformer4transformer seq_len={seq} parsed_layers={n_layers} "
               f"(truncated to {trf.max_seq_len}) extra_features={[int(v) for v in feats[1:]]}")


have = done(args.out, "gpumemnet")
for name in MODELS:
    spec = get_spec(name)
    for b in BATCHES:
        if (name, b) in have:
            continue
        if name in UNSUPPORTED:
            emit(args.out, "gpumemnet", name, b, None, None, None, "unsupported", UNSUPPORTED[name])
            continue
        t0 = time.perf_counter()
        try:
            torch.manual_seed(0)
            model = spec.build()
            if name in CNN:
                k, note = run_cnn(name, spec, model, b)
            elif name in VIT:
                k, note = run_trf(name, spec, model, b, 197)
                note += " vision transformer: outside the model types the repository tests"
            else:
                k, note = run_trf(name, spec, model, b, spec.default)
            emit(args.out, "gpumemnet", name, b, None, bin_name(k), time.perf_counter() - t0, "ok",
                 f"class={k} " + note)
        except BaseException as exc:  # noqa: BLE001  (the parser calls exit() on some inputs)
            if isinstance(exc, KeyboardInterrupt):
                raise
            emit(args.out, "gpumemnet", name, b, None, None, time.perf_counter() - t0, "failed",
                 f"{type(exc).__name__}: {str(exc)[:200]}. " + EXPLAIN.get(name, ""))
