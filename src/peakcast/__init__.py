"""peakcast: calibrated forecasts of the peak GPU memory of PyTorch training steps."""

from ._batching import Bucket, BucketBatchSampler, bucket_plan
from ._calibration import Calibration, conformal_quantile, default_calibration
from ._fit import Fit, fit_points, settled
from ._law import DEFAULT_RISK, Forecast, MemoryLaw, fit
from ._probe import Probe, ProbePolicy, device_budget, measure
from ._step import training_step

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_RISK",
    "Bucket",
    "BucketBatchSampler",
    "Calibration",
    "Fit",
    "Forecast",
    "MemoryLaw",
    "Probe",
    "ProbePolicy",
    "bucket_plan",
    "conformal_quantile",
    "default_calibration",
    "device_budget",
    "fit",
    "fit_points",
    "measure",
    "settled",
    "training_step",
    "__version__",
]
