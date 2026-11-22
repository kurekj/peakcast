Baseline GPU-memory estimators for the peakcast benchmark
==========================================================
Everything here was run on CPU only (CUDA_VISIBLE_DEVICES=""), 2026-11-20.
Workloads: 28 models of ../zoo.py x batch sizes 8, 32, 128, second axis at spec.default,
AdamW, fp32 (list in common.py). Results: results.jsonl (one line per tool, model, batch),
built by merge_results.py from results_<tool>.jsonl.

Tools, commits, licences
------------------------
torchinfo   1.8.0 from PyPI (already in ../env), MIT licence
xMem        https://github.com/Stone-ResearchLife/xMem
            commit 1ee4bd08dda277a6248abee5b6d40e0d6067efaa (2026-07-26), MIT licence
GPUMemNet   https://github.com/itu-rad/GPUMemNet
            commit 1029288675829c55ad027936fe086ce5380736b0 (2026-06-16),
            code Apache-2.0, datasets CC BY 4.0 (LICENSE, NOTICE)
LLMem       https://github.com/taehokim20/LLMem
            commit 3d289b46ea7d969e23634745209e6ad02beacff5 (2025-05-30), MIT licence
The clones (xMem/, GPUMemNet/, LLMem/) are unmodified; the only untracked file is
xMem/examples/xMem-result-convnext-base-batch130.json, written by the xMem example run below.

Environments
------------
../env          (given, not changed): Python 3.12, torch 2.15.0+cu132, torchvision 0.29.2,
                timm 1.0.31, transformers 5.23.0, torchinfo 1.8.0. Used for torchinfo.
env_xmem        ~/miniforge3/bin/conda create -y -p env_xmem python=3.11
                pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
                    --index-url https://download.pytorch.org/whl/cpu     (as in the xMem README)
                pip install -r xMem/requirement.txt                      (transformers 4.52.x)
                pip install timm                                         (needed by the zoo only)
env_gpumemnet   ~/miniforge3/bin/conda create -y -p env_gpumemnet python=3.11
                pip install torch==2.6.0 torchvision==0.21.0 \
                    --index-url https://download.pytorch.org/whl/cpu
                pip install pytorch-lightning torchsummary scikit-learn pandas pyyaml requests \
                    "transformers[torch]~=4.52.0" timm
                pip install modelsummary
Note: xMem and GPUMemNet therefore see the zoo models as built by torch 2.6 / transformers
4.52, the measurements as built by torch 2.15 / transformers 5.23.

Commands (from this directory)
------------------------------
export CUDA_VISIBLE_DEVICES=""
OMP_NUM_THREADS=16 ../env/bin/python run_torchinfo.py           # -> results_torchinfo.jsonl
bash run_xmem_all.sh                                             # -> results_xmem.jsonl
    (5 jobs x 3 threads, 1500 s limit per job; a single job is
     env_xmem/bin/python run_xmem.py --model resnet18 --batch 8; logs in xmem_logs/)
OMP_NUM_THREADS=4 env_gpumemnet/bin/python run_gpumemnet.py      # -> results_gpumemnet.jsonl
../env/bin/python run_llmem_feasibility.py                       # -> results_llmem.jsonl
../env/bin/python merge_results.py                               # -> results.jsonl
Every script skips the workloads that already have a record in its results file.

Sanity checks of the installations:
  cd xMem && HOME=$PWD/../xmem_home PYTHONPATH=$PWD ../env_xmem/bin/python main.py \
      ./examples/convnext-base-batch130.json -b 130 -g 8
      -> "Estimated Peak GPU Memory: 5.22 GB", the value printed in the xMem README
  cd GPUMemNet/Test_Pipeline && ../../env_gpumemnet/bin/python 01-Ensemble-mlp4cnn.py
  cd GPUMemNet/Test_Pipeline && ../../env_gpumemnet/bin/python 03-Ensemble-transformer4transformer.py
      -> both shipped test scripts run on CPU with the shipped checkpoints

What each number is
-------------------
torchinfo   estimate_bytes = total_input + total_output_bytes + total_param_bytes of
            torchinfo.summary(model, input_data=..., verbose=0), i.e. "Estimated Total Size".
            Model on CPU in train mode. (x, y) batches: input_data=[x]. Hugging Face dict
            batches: the whole dict including labels. seconds = time of the summary call.
xMem        estimate_bytes = result["memory"]["segment"] of XMem.estimate(...), the simulated
            peak of reserved (segment) memory. This is what xMem prints as "Estimated Peak GPU
            Memory" and what its experiment interface (exp/baselines/solution) returns as
            estimate_memory. The simulated peak of allocated tensor memory is in the note
            (peak_tensor). seconds = CPU profiling of 3 training iterations + estimation (both
            parts in the note), with 3 torch threads and 5 jobs running side by side.
            Simulated device capacity: 44.99 GB (46068 MiB, NVIDIA L40S). If xMem reports OOM
            at that capacity, the estimate is repeated with a capacity of 4096 GB and the note
            says oom_at_44.99GB=True.
GPUMemNet   estimate_bin = predicted class written as a memory range. The classes are the bins
            [0, 8000, 16000, 24000, 32000, 40000, 50000] MiB of the training code
            (Ensemble/dataloaders); out-of-memory training runs were labelled 42000 MiB there,
            so "40000-50000 MiB" also covers "did not fit the GPU of the authors".
            seconds = building the model + summary + parsing + inference.
LLMem       not run, see run_llmem_feasibility.py.

Wrappers and deviations from the entry points of the tools
----------------------------------------------------------
Details are in the docstrings of run_torchinfo.py, run_xmem.py, run_gpumemnet.py and
run_llmem_feasibility.py. In short:

xMem: its command line (xProfile.py) only accepts 12 torchvision CNN names or Hugging Face hub
  names and trains on CIFAR-10 at 86x86 or on wikitext. run_xmem.py calls the same classes with
  the same arguments as xProfile.py / FastRunner.train_on_cpu, but passes the zoo model, three
  synthetic zoo batches and AdamW (lr 1e-5). The training loops of xMem are used unchanged, so
  the profiled step is the one of xMem (zero_grad at the start of the iteration,
  CrossEntropyLoss on the model output for (x, y) workloads, StepLR scheduler, torch 2.6),
  not run_steps of ../measure.py. (x, y) workloads use the CNN path of xMem (Estimator),
  Hugging Face workloads its transformer path (TrainerEstimator). For the latter the hub
  lookup of the model class is replaced by a shim that builds the class of the zoo model from
  its locally saved config. Output adapters for fcn_resnet50 and custom_lstm_lm. HOME is
  redirected to ./xmem_home. Traces are deleted after the estimate (--keep-trace keeps them).

GPUMemNet: no estimator entry point is shipped (README: "How to Use GPUMemNet: TODO").
  run_gpumemnet.py reuses, unchanged, the functions of Test_Pipeline/01-Ensemble-mlp4cnn.py
  (CNNs, checkpoint mlp4cnn.ckpt) and Test_Pipeline/03-Ensemble-transformer4transformer.py
  (transformers, checkpoint transformer4transformer.ckpt). The summaries fed to the parsers
  are kept in gpumemnet_summaries/. The CNN estimator was trained on generated CNNs without
  residual or dense connections (Ensemble/dataloaders/dataloaders4mlp.py drops them); the
  test scripts of the repository nevertheless apply it to ResNet, MobileNet and EfficientNet,
  as done here. The features are layer counts, parameters, activations and batch size; the
  optimizer and the precision are not inputs.

Known problem of the workload list
----------------------------------
hf_t5_small and hf_t5_base: the zoo builds T5 from a bare T5Config. With transformers 5.23
(../env) and with 4.52 the forward pass with labels raises because decoder_start_token_id is
not set, so spec.loss(model, batch) fails as well. torchinfo and xMem are recorded as failed
for these six workloads; the torchinfo note gives the total obtained with
decoder_start_token_id=0. GPUMemNet only reads the module structure and returns a class.
