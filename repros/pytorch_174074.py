"""PyTorch #174074: duplicate advanced indices compound an in-place update."""

import torch

from accel_verify import ComparisonConfig, render_report, verify


def program(value):
    value[[0, 0]] += value[[0, 0]]
    return value


inputs = (torch.tensor([[-0.8716, 0.1114]], dtype=torch.float64),)
candidate = torch.compile(program, backend="inductor")
report = verify(
    program,
    candidate,
    inputs,
    config=ComparisonConfig(rtol=0, atol=0),
)

print("source: https://github.com/pytorch/pytorch/issues/174074")
print(render_report(report))

