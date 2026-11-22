"""Batch sizes per sequence length for a small Transformer language model.

The law is fitted from probes at lengths 32, 64 and 128 and then asked about
lengths up to 512. Samples are grouped into length buckets and every bucket
gets the largest batch size that fits the budget.

    pip install git+https://github.com/kurekj/peakcast@v0.1.0 torch
    python examples/variable_length.py
"""

import random

import torch
from torch import nn

import peakcast

VOCAB, DIM = 32000, 512


class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, DIM)
        self.position = nn.Embedding(1024, DIM)
        layer = nn.TransformerEncoderLayer(DIM, 8, 4 * DIM, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 6, enable_nested_tensor=False)
        self.head = nn.Linear(DIM, VOCAB)

    def forward(self, tokens):
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        return self.head(self.encoder(self.embed(tokens) + self.position(positions)))


model = LanguageModel().cuda().train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)


def loss(model, tokens):
    logits = model(tokens)
    return nn.functional.cross_entropy(logits.reshape(-1, VOCAB), tokens.reshape(-1))


def make_batch(batch_size, length):
    return torch.randint(0, VOCAB, (batch_size, int(length)), device="cuda")


step = peakcast.training_step(model, optimizer, loss)
budget = 8 * 1024**3
law = peakcast.fit(step, make_batch, axis="length", sizes=[32, 64, 128], budget=budget,
                   restore=[model, optimizer])
print(law.summary())

# A corpus with log-normally distributed lengths, as in most text datasets.
rng = random.Random(0)
lengths = [min(512, max(8, int(rng.lognormvariate(4.6, 0.8)))) for _ in range(20000)]

plan = peakcast.bucket_plan(law, lengths, buckets=8, risk=0.05)
for bucket in plan:
    print(f"length <= {bucket.max_size:4.0f}: batch {bucket.batch:4d}  ({bucket.count} samples)")

sampler = peakcast.BucketBatchSampler(lengths, plan, seed=0)
fixed = law.max_batch(risk=0.05, length=max(lengths))
steps = -(-len(lengths) // fixed)
print(f"steps per epoch: {len(sampler)} with buckets, {steps} with one batch size ({fixed})")

# In a real training loop the sampler is the batch_sampler of the DataLoader and
# the collate function pads each batch to sampler.limit_of(length).
for indices in list(sampler)[:20]:
    limit = sampler.limit_of(lengths[indices[0]])
    step(make_batch(len(indices), limit))
print("20 batches of mixed shapes trained under the budget")
