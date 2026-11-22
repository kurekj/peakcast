# peakcast

Calibrated forecasts of the peak GPU memory of PyTorch training steps.

`peakcast` answers three questions before a training run starts, from a few
small probe steps on the GPU the run will use:

1. How much memory will a training step need at batch size *b* (and sequence
   length or image resolution *s*)?
2. What is an upper bound on that need that is exceeded with a probability of
   at most a chosen risk, for example 1 %?
3. What is the largest batch size that fits a memory budget at that risk?

It does not search by running out of memory, it is not tied to a model family,
and it returns a function instead of a single number, so the same fit answers
the questions for another budget, another sequence length or a whole dataset
of variable-length samples.

```python
import peakcast

step = peakcast.training_step(model, optimizer, loss)      # loss(model, batch) -> scalar
law = peakcast.fit(step, make_batch, restore=[model, optimizer])

print(law.peak(256, risk=0.01))   # expected 20.88 GiB, at most 28.04 GiB (risk 0.01, extrapolation x3.8)
print(law.max_batch(risk=0.01))   # largest batch size whose upper bound fits the free memory
```

## Installation

```bash
pip install git+https://github.com/kurekj/peakcast@v0.1.0    # needs numpy; PyTorch 2.1 or newer must be installed for fit()
```

From source: `pip install -e ".[dev]"`, then `pytest` and `ruff check .`.
The tests run on a CPU; one extra test runs when a CUDA device is present.

## How it works

**The memory law.** The peak allocated memory of a training step is the
maximum, over the program points of the step, of the memory live at that
point. With fixed shapes per sample, the memory live at one program point is
affine in the batch size with a non-negative intercept (parameters, optimizer
state, framework buffers) and a non-negative slope (activations, gradients of
activations, workspaces). The peak is therefore a maximum of affine functions:
convex and piecewise linear in the batch size. In practice it has a flat
*floor* at small batch sizes, where the optimizer update dominates, followed
by a line. `peakcast` fits `max(floor, c0 + c1 * batch)`, and with a second
size axis `max(floor, c0 + batch * (c1 + c2 * s + c3 * s^2))`, by non-negative
least squares on the largest probes.

**Probing.** `fit` runs real training steps at batch sizes 1, 2, 4, ... and
reads `torch.cuda.max_memory_allocated`. It stops when the last three probes
lie on one rising line and the peak has reached 20 % of the budget. A probe is
skipped when the line through the previous probes puts it above 90 % of the
budget, so probing stays away from the limit; if a probe does run out of
memory, the error is caught and only ends the probing. Model, optimizer and
random number generators are restored afterwards. A model that cannot train on
a single sample (batch normalisation over one value per channel) is probed
from two samples with `policy=peakcast.ProbePolicy(min_batch=2)`.

**The calibrated bound.** A line through the largest probes is a lower bound
of a convex law, and the caching allocator of PyTorch needs more memory than
the allocated peak because of fragmentation. Both effects are covered by one
measured quantity: for a set of calibration workloads, the ratio between the
smallest budget under which a batch size actually trains and the predicted
peak. Split conformal prediction turns the calibration ratios into a
multiplier: with *n* calibration workloads the `ceil((n + 1)(1 - risk))`-th
smallest ratio bounds the ratio of a new workload with probability at least
`1 - risk`, provided the new workload is exchangeable with the calibration
set. The package ships a calibration measured on 456 workloads
(see below); `peakcast.Calibration.from_scores` builds one from your own
measurements.

## API

| Call | Purpose |
|---|---|
| `peakcast.fit(step, make_batch, *, axis=None, sizes=None, budget=None, restore=(), policy=None)` | probe and fit; returns a `MemoryLaw` |
| `law.expected(batch, **size)` | predicted peak allocated memory in bytes |
| `law.peak(batch, risk=0.01, **size)` | `Forecast` with `expected`, `upper`, `risk`, `extrapolation` |
| `law.max_batch(budget=None, risk=0.01, **size)` | largest batch size whose upper bound fits the budget, and never less than a batch size that a probe already ran under it |
| `law.components(batch, **size)` | the terms of the law (fixed, per sample, per sample and size) |
| `law.summary()`, `law.save(path)`, `MemoryLaw.load(path)` | report and persistence (JSON) |
| `peakcast.bucket_plan(law, lengths, buckets=8, risk=0.01)` | batch size per length bucket |
| `peakcast.BucketBatchSampler(lengths, plan)` | `batch_sampler` for a `DataLoader` |
| `peakcast.training_step(model, optimizer, loss, amp=False)` | a standard training step |
| `peakcast.Calibration`, `peakcast.default_calibration()` | conformal calibration data |
| `peakcast.MemoryLaw.from_probes(...)`, `peakcast.fit_points(...)` | fitting from measurements taken elsewhere |

`step(batch)` is any callable that runs one complete training step, so custom
loops (gradient accumulation, several optimizers, mixed precision) work
unchanged. `make_batch(batch_size)` returns a batch on the GPU; with a second
axis it is called as `make_batch(batch_size, size)`.

### A second size axis

```python
law = peakcast.fit(step, make_batch, axis="length", sizes=[32, 64, 128])
law.max_batch(risk=0.05, length=512)        # asks beyond the probed lengths
plan = peakcast.bucket_plan(law, lengths, buckets=8, risk=0.05)
sampler = peakcast.BucketBatchSampler(lengths, plan)
```

`examples/quickstart.py` and `examples/variable_length.py` are complete scripts.

## What the guarantee does and does not say

- The bound is *marginal*: it holds on average over workloads that are
  exchangeable with the calibration workloads, not for each model separately.
  The shipped calibration covers the model families listed in
  `benchmarks/zoo.py`; `benchmarks/evaluate.py` reports what happens when a
  whole family is left out of the calibration.
- The smallest risk that can be certified is `1 / (n + 1)` for a group of *n*
  calibration workloads. `peakcast` raises an error instead of returning a
  bound it cannot certify, and it does not bound forecasts that extrapolate
  further beyond the probes than the calibration workloads did (5.1 times the
  largest probed peak in the shipped calibration).
- The law describes steps whose tensor shapes are fixed by the batch size and
  the declared size axis. Models whose memory depends on the content of the
  data (for example detection models with a variable number of proposals) are
  outside its scope.
- The budget defaults to the memory that the allocator of the process can use
  when `fit` is called. Memory taken later by other processes is not known to
  the law.
- Measurements are specific to the GPU, the PyTorch build and the cuDNN
  settings they were taken with. Refit after changing any of them; it takes
  seconds.

## Benchmarks

`benchmarks/` holds the scripts and `benchmarks/results/` every measurement
behind the numbers below; `python benchmarks/evaluate.py --check` replays the
library on them without a GPU and is part of continuous integration.

| Question | Result |
|---|---|
| Does the law extrapolate? (NVIDIA L40S, 456 workloads of 152 models, budgets of 8 to 40 GiB) | mean error of the predicted peak 0.5 %, 95th percentile 1.9 %, after a median of 7 probes and 1.0 s |
| And along a second axis, beyond the probed sizes? | mean error 1.4 %, median 0.4 % |
| Is the predicted peak enough to choose a batch size? (2,232 cases with ground truth) | no: the largest batch size whose predicted peak fits ran out of memory in 95 % of cases (69 % counting only batch sizes at or above one that was seen to fail) |
| Does the bound hold when a whole architecture family is missing from the calibration? | pooled over all cases: out of memory in 5.7 % at 10 % risk, 2.6 % at 5 %, 0.4 % at 1 % |
| Is that repeated on another system? (RTX PRO 5000 laptop GPU, 24 GB, Windows; 698 cases) | 6.2 %, 3.2 % and 0.7 % |
| Does a calibration carry over to another system? | at 5 % risk, 17 of 698 cases ran out of memory with the L40S calibration on the laptop GPU (2.4 %) and 68 of 2,232 the other way round (3.0 %) |
| How much of the possible batch size is used? | median 86 % at 5 % risk, 79 % at 1 % risk; a straight line through batch sizes 1 to 16 solved for 60 % of the budget reaches 53 % |
| Live, against search tools (laptop GPU with the L40S calibration; 31 models, 8 and 16 GiB) | 1 of 62 recommendations at 1 % risk, 1 at 5 % and 6 at 10 % failed a check of 20 training steps; median time 2.0 s against 6.3 s and 10.1 s for the two modes of the Lightning batch size finder |
| The same on the L40S (32 models, 8, 24 and 40 GiB) | 1 of 96 at 1 % risk, 1 at 5 % and 7 at 10 %; median time 2.2 s against 5.0 s and 13.9 s for Lightning and 2.2 s for toma, which reaches 75 % of the largest verified batch size against 92 % at 5 % risk |

The ground truth is the largest batch size that trains under a memory budget,
found by bisection with real training steps under an allocator limit.

Four findings that a user should know:

- The failures are not spread evenly. At 5 % risk recurrent networks ran
  out of memory in 24 of 42 cases and audio models in 7 of 120, against 28
  of 2,070 for everything else; on the laptop GPU text decoders and
  sequence-to-sequence models also exceeded the risk.
- A bucketed plan makes one decision per bucket and fails more often than a
  single recommendation. At 10 % risk, 18 of 24 plans on the L40S and 12 of
  16 on the laptop GPU ran without a failure; the plans for an LSTM language
  model failed on both, and one plan each for Pythia-160M, BART-base and
  T5-small.
- A fixed margin of 15 % on the same fit gives a similar trade-off. The
  calibration does not beat a well-chosen margin; it chooses the margin from
  data, states the risk that goes with it and refuses to answer where it has
  no data.
- The shipped calibration was measured on an L40S under Linux. Pooled over
  the benchmark it carried over to a laptop GPU under Windows and back, but a
  guarantee for your system needs a calibration measured on it:

```bash
python benchmarks/run_queue.py --out raw --gpus 0                      # peak memory over the batch-size grid
python benchmarks/run_queue.py --out truth --gpus 0 --script truth_caps.py -- --grid raw
python benchmarks/pack.py raw truth benchmarks/results/mydevice        # then: echo mydevice > benchmarks/results/main.txt
python benchmarks/make_calibration.py mydevice                         # writes the calibration and the held-out-family files
```

`peakcast.Calibration.from_scores` builds a calibration from any list of
`(workload, kind, extrapolation ratio, score)` records, and `fit(...,
calibration=...)` uses it.

Compared tools: torchinfo, xMem, GPUMemNet and LLMem (estimators), the batch
size finder of PyTorch Lightning, toma, torch-max-mem and the AutoBatch rule
of Ultralytics YOLO. The scripts that ran them and their raw outputs are in
`benchmarks/results/baselines/` and `benchmarks/live_compare.py`. Four of the
156 models of `benchmarks/zoo.py` could not be measured; they are listed in
`benchmarks/results/l40s/meta.json`.

## Citation

See `CITATION.cff`.

## License

MIT.
