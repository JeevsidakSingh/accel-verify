"""PyTorch #179561: an intentional BF16 precision cast is elided in fusion."""

import torch

from accel_verify import ComparisonConfig, render_report, verify


def program(x, weight):
    value = torch.matmul(x, weight)
    value = value.to(torch.bfloat16).float()
    value = value * torch.sigmoid(value)
    return value.sum(dim=1)


torch.manual_seed(0)
inputs = (
    torch.randn(4, 32, 32, device="cuda"),
    torch.randn(32, 32, device="cuda"),
)
candidate = torch.compile(program, backend="inductor")
report = verify(
    program,
    candidate,
    inputs,
    config=ComparisonConfig(rtol=0, atol=0),
)

print("source: https://github.com/pytorch/pytorch/issues/179561")
print(render_report(report))
