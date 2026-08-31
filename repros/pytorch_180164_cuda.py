"""PyTorch #180164: slice_scatter has correct output but wrong input gradient."""

import torch

from accel_verify import ComparisonConfig, render_report, verify


def program(x, y):
    out = torch.slice_scatter(x, y, dim=1, start=0, end=6)
    return out.sum(dim=1)


torch.manual_seed(0)
inputs = (
    torch.randn(4, 13, 33, device="cuda", requires_grad=True),
    torch.randn(4, 6, 33, device="cuda", requires_grad=True),
)
upstream = torch.randn(4, 33, device="cuda")
candidate = torch.compile(program, backend="inductor")
report = verify(
    program,
    candidate,
    inputs,
    config=ComparisonConfig(rtol=0, atol=0, check_gradients=True),
    gradient_loss=lambda output: (output * upstream).sum(),
)

print("source: https://github.com/pytorch/pytorch/issues/180164")
print(render_report(report))
