"""PyTorch #177821: complex indexing assignment ignored by Inductor."""

import torch

from accel_verify import ComparisonConfig, render_report, verify


def program(x, y):
    x = 2 * x
    combined = torch.cat([x, y], dim=1)
    combined[:, [1, 0]] = combined[:, [0, 1]]
    return combined[:, :2] + x


inputs = (
    torch.arange(4).reshape(2, 2),
    torch.arange(4).reshape(2, 2),
)
candidate = torch.compile(program, backend="inductor")
report = verify(
    program,
    candidate,
    inputs,
    config=ComparisonConfig(rtol=0, atol=0),
)

print("source: https://github.com/pytorch/pytorch/issues/177821")
print(render_report(report))

