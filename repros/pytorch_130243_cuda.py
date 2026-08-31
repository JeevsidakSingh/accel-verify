"""PyTorch #130243: Inductor stride reordering breaks a user Triton kernel."""

import torch
import triton
import triton.language as tl

from accel_verify import ComparisonConfig, render_report, verify


@triton.jit
def add_kernel(
    in_ptr0,
    in_ptr1,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    y = tl.load(in_ptr1 + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def program(x, other):
    y = x.t().contiguous().t()
    z = y.sin().t()
    out = torch.empty_like(other)
    add_kernel[(z.numel(),)](z, other, out, z.numel(), BLOCK_SIZE=16)
    return out


torch.manual_seed(0)
inputs = (
    torch.randn(2, 2, device="cuda"),
    torch.randn(2, 2, device="cuda"),
)
candidate = torch.compile(program, backend="inductor", fullgraph=True)
report = verify(program, candidate, inputs, config=ComparisonConfig())

print("source: https://github.com/pytorch/pytorch/issues/130243")
print(render_report(report))
