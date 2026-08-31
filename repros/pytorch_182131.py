"""PyTorch #182131: float16 casts before an add produce a different result."""

import torch

from accel_verify import ComparisonConfig, render_report, verify


def program(x, condition):
    offset = torch.where(
        condition,
        torch.full_like(x, 0.5),
        torch.full_like(x, -0.5),
    )
    value = offset.to(torch.float16) + x.to(torch.float16)
    return value, value.float().sum()


inputs = (
    torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 10,
    (torch.arange(24) % 5 == 0).reshape(2, 3, 4),
)
candidate = torch.compile(program, backend="inductor", fullgraph=True)
report = verify(
    program,
    candidate,
    inputs,
    config=ComparisonConfig(rtol=0, atol=0),
)

print("source: https://github.com/pytorch/pytorch/issues/182131")
print(render_report(report))

