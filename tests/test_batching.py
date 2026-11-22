import random

import pytest

import peakcast


@pytest.fixture
def lengths():
    rng = random.Random(0)
    return [min(512, max(8, int(rng.lognormvariate(4.5, 0.7)))) for _ in range(2000)]


def test_longer_buckets_get_smaller_batches(law2, lengths):
    plan = peakcast.bucket_plan(law2, lengths, buckets=6)
    assert [b.max_size for b in plan] == sorted(b.max_size for b in plan)
    assert [b.batch for b in plan] == sorted((b.batch for b in plan), reverse=True)
    assert sum(b.count for b in plan) == len(lengths)
    assert plan[-1].max_size == max(lengths)


def test_every_bucket_respects_the_budget(law2, lengths):
    for bucket in peakcast.bucket_plan(law2, lengths, risk=0.05):
        assert law2.upper(bucket.batch, 0.05, length=bucket.max_size) <= law2.budget


def test_explicit_boundaries_are_extended_to_the_longest_sample(law2, lengths):
    plan = peakcast.bucket_plan(law2, lengths, boundaries=[64, 128])
    assert [b.max_size for b in plan][:2] == [64, 128] and plan[-1].max_size == max(lengths)


def test_sampler_yields_every_index_once_within_bucket_limits(law2, lengths):
    plan = peakcast.bucket_plan(law2, lengths, buckets=5)
    sampler = peakcast.BucketBatchSampler(lengths, plan, seed=1)
    batches = list(sampler)
    assert len(batches) == len(sampler)
    assert sorted(i for batch in batches for i in batch) == list(range(len(lengths)))
    by_limit = {b.max_size: b.batch for b in plan}
    for batch in batches:
        limit = sampler.limit_of(max(lengths[i] for i in batch))
        assert len(batch) <= by_limit[limit]
        assert all(sampler.limit_of(lengths[i]) == limit for i in batch)


def test_drop_last_keeps_only_full_batches(law2, lengths):
    plan = peakcast.bucket_plan(law2, lengths, buckets=4)
    sampler = peakcast.BucketBatchSampler(lengths, plan, drop_last=True, shuffle=False)
    sizes = {b.max_size: b.batch for b in plan}
    assert all(len(batch) == sizes[sampler.limit_of(lengths[batch[0]])] for batch in sampler)
    assert len(list(sampler)) == len(sampler)


def test_a_law_without_second_axis_is_rejected(law, lengths):
    with pytest.raises(ValueError):
        peakcast.bucket_plan(law, lengths)


def test_a_sample_that_cannot_fit_is_an_error(law2):
    with pytest.raises(ValueError, match="does not fit"):
        peakcast.bucket_plan(law2, [100000], budget=2 * 1024**3)


def test_invalid_bucket_arguments_are_rejected(law2, lengths):
    with pytest.raises(ValueError):
        peakcast.bucket_plan(law2, lengths, buckets=0)
    with pytest.raises(ValueError):
        peakcast.BucketBatchSampler(lengths, [])
    assert peakcast.bucket_plan(law2, lengths, boundaries=[])[-1].max_size == max(lengths)
