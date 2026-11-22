# Changelog

## 0.1.0 (2026-11-21)

First release.

- `peakcast.fit`: probes a training step at small batch sizes and fits a memory law.
- `MemoryLaw`: expected peak, calibrated upper bound, largest batch under a budget,
  optional second size axis (sequence length, resolution), JSON persistence.
- `Calibration`: split conformal calibration of the upper bound; a default calibration
  measured on 456 workloads ships with the package.
- `bucket_plan` and `BucketBatchSampler`: batch sizes per length bucket for data of variable length.
- Benchmarks with the recorded measurements and the scripts that reproduce every number of the paper.
