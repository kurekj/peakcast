"""Batching data of variable size under a memory budget.

With sequences of different lengths a fixed batch size has to be chosen for
the longest sequence and wastes memory on all the others. A memory law gives
the largest safe batch for every length, so samples can be grouped into length
buckets, each with its own batch size.
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from ._law import DEFAULT_RISK, MemoryLaw


@dataclass(frozen=True)
class Bucket:
    """Samples with size up to ``max_size`` are batched ``batch`` at a time."""

    max_size: float
    batch: int
    count: int


def bucket_plan(
    law: MemoryLaw,
    lengths: Sequence[float],
    *,
    boundaries: Sequence[float] | None = None,
    buckets: int = 8,
    budget: float | None = None,
    risk: float = DEFAULT_RISK,
) -> list[Bucket]:
    """Choose a batch size for each length bucket.

    ``lengths`` holds the size of every sample along the second axis of the
    law. ``boundaries`` are the upper limits of the buckets; by default they
    are ``buckets`` quantiles of ``lengths``. Every batch is padded to the
    upper limit of its bucket, so that limit is the size the law is asked
    about. Raises ``ValueError`` if a single sample of some bucket does not fit.
    """
    if law.axis is None:
        raise ValueError("the law has no second axis; fit it with axis= and sizes=")
    if len(lengths) == 0:
        raise ValueError("lengths is empty")
    if boundaries is None and buckets < 1:
        raise ValueError("buckets must be at least 1")
    ordered = sorted(lengths)
    if boundaries is None:
        n = len(ordered)
        cuts = [max(0, (n * (i + 1)) // buckets - 1) for i in range(buckets)]
        boundaries = sorted({ordered[min(n - 1, c)] for c in cuts})
    limits = sorted(set(boundaries))
    if not limits or limits[-1] < ordered[-1]:
        limits.append(ordered[-1])
    plan: list[Bucket] = []
    lower = float("-inf")
    for limit in limits:
        count = sum(1 for v in ordered if lower < v <= limit)
        lower = limit
        if count == 0:
            continue
        batch = law.max_batch(budget, risk, **{law.axis: limit})
        if batch < 1:
            raise ValueError(f"a single sample of {law.axis} {limit} does not fit the budget")
        plan.append(Bucket(float(limit), batch, count))
    return plan


class BucketBatchSampler:
    """Yields batches of indices, each from one bucket and of the batch size of that bucket.

    Works as the ``batch_sampler`` of a ``torch.utils.data.DataLoader``; the
    collate function should pad each batch to the upper limit of its bucket
    (:meth:`limit_of` returns it for a sample length).
    """

    def __init__(self, lengths: Sequence[float], plan: Sequence[Bucket], *, shuffle: bool = True,
                 seed: int | None = None, drop_last: bool = False):
        self.plan = sorted(plan, key=lambda b: b.max_size)
        if not self.plan or any(b.batch < 1 for b in self.plan):
            raise ValueError("the plan must have at least one bucket and positive batch sizes")
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._rng = random.Random(seed)
        self._members: list[list[int]] = [[] for _ in self.plan]
        for index, length in enumerate(lengths):
            self._members[self._bucket_of(length)].append(index)

    def _bucket_of(self, length: float) -> int:
        for i, bucket in enumerate(self.plan):
            if length <= bucket.max_size:
                return i
        raise ValueError(f"length {length} exceeds the largest bucket ({self.plan[-1].max_size})")

    def limit_of(self, length: float) -> float:
        return self.plan[self._bucket_of(length)].max_size

    def _batches(self) -> list[list[int]]:
        out: list[list[int]] = []
        for bucket, members in zip(self.plan, self._members, strict=False):
            members = list(members)
            if self.shuffle:
                self._rng.shuffle(members)
            for start in range(0, len(members), bucket.batch):
                chunk = members[start:start + bucket.batch]
                if len(chunk) == bucket.batch or not self.drop_last:
                    out.append(chunk)
        if self.shuffle:
            self._rng.shuffle(out)
        return out

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self._batches())

    def __len__(self) -> int:
        total = 0
        for bucket, members in zip(self.plan, self._members, strict=False):
            full, rest = divmod(len(members), bucket.batch)
            total += full + (1 if rest and not self.drop_last else 0)
        return total
