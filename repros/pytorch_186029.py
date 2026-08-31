"""PyTorch #186029: count_nonzero is wrong after a weighted bincount."""

import torch

from accel_verify import ComparisonConfig, render_report, verify


def program(indices, weights):
    counts = indices.bincount(weights=weights, minlength=6)
    return counts, counts.count_nonzero()


inputs = (
    torch.tensor([0, 2, 2, 3, 1, 0, 3, 3], dtype=torch.int64),
    torch.tensor([1.0, -2.0, 3.0, 0.0, 4.0, -5.0, 6.0, 7.0]),
)
candidate = torch.compile(program, backend="inductor", fullgraph=True)
report = verify(
    program,
    candidate,
    inputs,
    config=ComparisonConfig(rtol=0, atol=0),
)

print("source: https://github.com/pytorch/pytorch/issues/186029")
print(render_report(report))

