"""PyTorch #136662: compiled torch.func.grad silently loses an accumulation."""

import torch

from accel_verify import ComparisonConfig, render_report, verify


def objective(x):
    value = torch.zeros_like(x)
    value[:, 0] += x[:, 2].abs()
    value[:, 0] += x[:, 2].abs()
    return value.sum()


gradient = torch.func.grad(objective)
candidate = torch.compile(gradient, backend="inductor", fullgraph=True)
inputs = (torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]]),)
report = verify(
    gradient,
    candidate,
    inputs,
    config=ComparisonConfig(rtol=0, atol=0),
)

print("source: https://github.com/pytorch/pytorch/issues/136662")
print(render_report(report))

