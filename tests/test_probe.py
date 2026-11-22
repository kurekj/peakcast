import pytest
from conftest import GIB, hinge

import peakcast
from peakcast._probe import run_probes


def fake(batch, size):
    return int(hinge(batch)), int(1.2 * hinge(batch))


def test_probing_stops_once_the_slope_has_settled_and_the_fraction_is_reached():
    probes, settled, oom = run_probes(fake, None, 48 * GIB, peakcast.ProbePolicy())
    assert settled and not oom
    assert [p.batch for p in probes] == [1, 2, 4, 8, 16, 32, 64, 128]
    assert probes[-1].allocated >= 0.2 * 48 * GIB


def test_a_probe_predicted_above_the_ceiling_is_skipped():
    budget = 6 * GIB
    probes, _, oom = run_probes(fake, None, budget, peakcast.ProbePolicy())
    assert not oom
    assert all(p.allocated <= budget for p in probes)
    assert probes[-1].batch == 48  # 64 would exceed the ceiling, so the half-way probe is used


def test_an_out_of_memory_probe_is_replaced_by_one_half_way():
    def limited(batch, size):
        return None if batch > 12 else fake(batch, size)

    probes, settled, oom = run_probes(limited, None, 48 * GIB, peakcast.ProbePolicy())
    assert oom and not settled
    assert [p.batch for p in probes] == [1, 2, 4, 8, 12]


def test_nothing_is_probed_after_the_smallest_batch_fails():
    probes, settled, oom = run_probes(lambda b, s: None, None, 48 * GIB, peakcast.ProbePolicy())
    assert probes == [] and oom and not settled


def test_half_way_probes_can_settle_the_slope():
    batches, peaks = [16, 32, 48], [hinge(16), hinge(32), hinge(48)]
    assert peakcast.settled(batches, peaks)


def test_every_size_is_probed():
    probes, _, _ = run_probes(fake, [32, 64, 128], 48 * GIB, peakcast.ProbePolicy())
    assert {p.size for p in probes} == {32, 64, 128}


def test_max_probes_is_respected():
    probes, _, _ = run_probes(fake, None, 1e15, peakcast.ProbePolicy(max_probes=4))
    assert len(probes) == 4


def _cuda():
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


@pytest.mark.gpu
@pytest.mark.skipif(not _cuda(), reason="needs a CUDA device")
def test_fit_on_a_real_model_restores_it_and_predicts_a_larger_batch(calibration):
    import torch

    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 32, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(32, 32, 3, padding=1),
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(32, 10),
    ).cuda()
    optimizer = torch.optim.AdamW(model.parameters())
    before = {k: v.clone() for k, v in model.state_dict().items()}

    def loss(m, batch):
        x, y = batch
        return torch.nn.functional.cross_entropy(m(x), y)

    def make_batch(b):
        return torch.randn(b, 3, 64, 64, device="cuda"), torch.randint(0, 10, (b,), device="cuda")

    step = peakcast.training_step(model, optimizer, loss)
    law = peakcast.fit(step, make_batch, budget=2 * GIB, restore=[model, optimizer], calibration=calibration)
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    target = 4 * max(p.batch for p in law.probes)
    measured = peakcast.measure(step, lambda: make_batch(target))
    assert measured is not None
    assert law.expected(target) == pytest.approx(measured[0], rel=0.1)
    assert law.max_batch(risk=0.05) > 0


def test_probing_ends_after_the_half_way_probe():
    def slow_then_steep(batch, size):
        return (int(1e9 + 1e6 * batch) if batch <= 8 else int(1e9 + 1e6 * batch + 4e7 * (batch - 8))), 0

    probes, _, _ = run_probes(slow_then_steep, None, 1.3e9, peakcast.ProbePolicy(fraction=0.99))
    batches = [p.batch for p in probes]
    assert batches == sorted(batches)
    assert sum(1 for a, b in zip(batches, batches[1:], strict=False) if b < 2 * a) <= 1


def test_no_probe_when_max_probes_is_zero():
    probes, _, _ = run_probes(fake, None, 48 * GIB, peakcast.ProbePolicy(max_probes=0))
    assert probes == []
