"""Probing: measuring the peak memory of a training step at small sizes."""

from __future__ import annotations

import copy
import gc
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from ._fit import settled


@dataclass(frozen=True)
class Probe:
    """One measurement: sizes in, peak bytes out."""

    batch: int
    size: float | None
    allocated: int
    reserved: int


@dataclass(frozen=True)
class ProbePolicy:
    """Which batch size to probe next, and when to stop.

    Probing doubles the batch size. It stops when the last three probes lie on
    a rising line and the peak has reached ``fraction`` of the budget. A probe
    is skipped when the line through the last two probes puts its peak above
    ``ceiling`` of the budget; if a doubling is skipped, or ran out of memory,
    one probe half-way is tried instead. An out-of-memory error raised by a
    probe is caught and only ends the probing.
    """

    fraction: float = 0.2
    ceiling: float = 0.9
    min_batch: int = 1
    max_probes: int = 20

    def _allowed(self, batch: int, batches: Sequence[int], peaks: Sequence[float], budget: float) -> bool:
        slope = 0.0
        if len(batches) >= 3:  # the first two probes are too noisy to extrapolate from
            slope = max(0.0, (peaks[-1] - peaks[-2]) / (batches[-1] - batches[-2]))
        if slope == 0.0:
            # No growth seen yet (first probes, or still on the floor): try the next
            # batch as long as the last one fitted; a failure is caught by the caller.
            return peaks[-1] <= budget
        return peaks[-1] + slope * (batch - batches[-1]) <= self.ceiling * budget

    def propose(self, batches: Sequence[int], peaks: Sequence[float], budget: float,
                blocked: int | None = None) -> int | None:
        """Next batch size to probe, or ``None`` to stop.

        ``blocked`` is the smallest batch size known not to fit (a probe of it
        ran out of memory).
        """
        if len(batches) >= self.max_probes:
            return None
        if not batches:
            return self.min_batch
        if settled(batches, peaks) and peaks[-1] >= self.fraction * budget:
            return None
        last = batches[-1]
        halfway = last + last // 2
        if len(batches) >= 2 and last < 2 * batches[-2]:
            return None  # the last probe was the half-way one: probing ends
        for candidate in (2 * last, halfway):
            if candidate <= last:
                continue
            if blocked is not None and candidate >= blocked:
                continue
            if self._allowed(candidate, batches, peaks, budget):
                return candidate
        return None


def is_out_of_memory(exc: BaseException) -> bool:
    import torch

    if isinstance(exc, torch.OutOfMemoryError):
        return True
    text = str(exc).lower()
    return isinstance(exc, RuntimeError) and ("out of memory" in text or "alloc_failed" in text)


def device_budget(device: Any = None) -> int:
    """Bytes the allocator of this process can use on ``device`` right now."""
    import torch

    free, _total = torch.cuda.mem_get_info(device)
    return int(free + torch.cuda.memory_reserved(device))


def measure(step: Callable[[Any], Any], make: Callable[[], Any], device: Any = None,
            steps: int = 2) -> tuple[int, int] | None:
    """Peak (allocated, reserved) bytes over ``steps`` calls of ``step(make())``.

    Returns ``None`` if the step ran out of memory. Two steps are the default
    because optimizers allocate their state during the first one.
    """
    import torch

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    failed = False
    try:
        for _ in range(steps):
            batch = make()
            step(batch)
            del batch
        torch.cuda.synchronize(device)
    except Exception as exc:  # noqa: BLE001
        if not is_out_of_memory(exc):
            raise
        failed = True
    if failed:
        gc.collect()
        torch.cuda.empty_cache()
        return None
    return int(torch.cuda.max_memory_allocated(device)), int(torch.cuda.max_memory_reserved(device))


def _to_cpu(value: Any) -> Any:
    """Copy a (nested) state to host memory, so that the snapshot takes no GPU memory."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return type(value)((k, _to_cpu(v)) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return type(value)(_to_cpu(v) for v in value)
    return copy.deepcopy(value)


class _Snapshot:
    """Saves and restores ``state_dict`` of the given objects, training flags and RNG state.

    The copies are kept in host memory. Gradients are not saved: probing ends
    with gradients cleared, as after ``optimizer.zero_grad()``.
    """

    def __init__(self, objects: Iterable[Any]):
        import torch

        self.objects = list(objects)
        self.states = [_to_cpu(o.state_dict()) for o in self.objects]
        self.modes = [
            [(m, m.training) for m in o.modules()] if isinstance(o, torch.nn.Module) else []
            for o in self.objects
        ]
        self.rng = torch.get_rng_state()
        self.cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    def restore(self) -> None:
        import torch

        for obj, state, modes in zip(self.objects, self.states, self.modes, strict=True):
            obj.load_state_dict(state)
            for module, training in modes:
                module.training = training
        torch.set_rng_state(self.rng)
        if self.cuda_rng is not None:
            torch.cuda.set_rng_state_all(self.cuda_rng)
        self.states = []
        gc.collect()
        torch.cuda.empty_cache()


def run_probes(
    measure_at: Callable[[int, Any], tuple[int, int] | None],
    sizes: Sequence[float] | None,
    budget: float,
    policy: ProbePolicy,
) -> tuple[list[Probe], bool, bool]:
    """Probe every size with ``measure_at(batch, size)``.

    Returns (probes, all sizes settled, an out-of-memory error occurred).
    ``measure_at`` returns peak (allocated, reserved) bytes, or ``None`` when
    the step ran out of memory.
    """
    probes: list[Probe] = []
    all_settled, oom = True, False
    for size in [None] if sizes is None else list(sizes):
        batches: list[int] = []
        peaks: list[float] = []
        blocked: int | None = None
        while True:
            b = policy.propose(batches, peaks, budget, blocked)
            if b is None:
                break
            result = measure_at(b, size)
            if result is None:
                oom, blocked = True, b
                if not batches:
                    break
                continue
            probes.append(Probe(b, size, int(result[0]), int(result[1])))
            batches.append(b)
            peaks.append(float(result[0]))
        all_settled = all_settled and settled(batches, peaks)
    return probes, all_settled, oom
