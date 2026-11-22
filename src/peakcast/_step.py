"""A ready-made training step for the common case."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def training_step(model: Any, optimizer: Any, loss: Callable[[Any, Any], Any], *,
                  amp: bool = False, scaler: Any = None) -> Callable[[Any], None]:
    """Return ``step(batch)`` that runs forward, backward and the optimizer update.

    ``loss(model, batch)`` returns a scalar tensor. With ``amp=True`` the
    forward pass runs under ``torch.autocast`` and a gradient scaler is used
    (created if not given).
    """
    import torch

    if amp and scaler is None:
        scaler = torch.amp.GradScaler("cuda")

    def step(batch: Any) -> None:
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            value = loss(model, batch)
        if amp:
            scaler.scale(value).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            value.backward()
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return step
