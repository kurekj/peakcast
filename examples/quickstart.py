"""Largest safe batch size for a ResNet-50 training step, without running out of memory.

    pip install git+https://github.com/kurekj/peakcast@v0.1.0 torch torchvision
    python examples/quickstart.py
"""

import torch
import torchvision

import peakcast

model = torchvision.models.resnet50(weights=None).cuda().train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)


def loss(model, batch):
    images, labels = batch
    return torch.nn.functional.cross_entropy(model(images), labels)


def make_batch(batch_size):
    images = torch.randn(batch_size, 3, 224, 224, device="cuda")
    labels = torch.randint(0, 1000, (batch_size,), device="cuda")
    return images, labels


step = peakcast.training_step(model, optimizer, loss)

# Probe a few small batch sizes and fit the memory law. The model and the
# optimizer are put back into their initial state afterwards.
law = peakcast.fit(step, make_batch, restore=[model, optimizer])
print(law.summary())

# What would batch 256 need? `upper` is exceeded with probability at most `risk`.
print(law.peak(256, risk=0.01))

# Largest batch size whose upper bound fits the memory that is free now.
print("largest safe batch size:", law.max_batch(risk=0.01))

# The same question for a smaller card, answered without measuring anything again.
print("on a 12 GiB budget:", law.max_batch(budget=12 * 1024**3, risk=0.01))
