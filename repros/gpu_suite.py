"""Bounded CUDA suite for the Accel-Verify integrated runner."""

import torch
import triton
import triton.language as tl

from accel_verify import ComparisonConfig, VerificationCase


@triton.jit
def add_kernel(in_ptr0, in_ptr1, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    left = tl.load(in_ptr0 + offsets, mask=mask)
    right = tl.load(in_ptr1 + offsets, mask=mask)
    tl.store(out_ptr + offsets, left + right, mask=mask)


def _cast_case(shape, cast_dtype):
    torch.manual_seed(0)
    inputs = (
        torch.randn(shape, device="cuda"),
        torch.randn(shape[-1], shape[-1], device="cuda"),
    )

    def program(value, weight):
        result = torch.matmul(value, weight)
        result = result.to(cast_dtype).float()
        result = result * torch.sigmoid(result)
        return result.sum(dim=1)

    dtype_name = str(cast_dtype).removeprefix("torch.")
    shape_name = "x".join(map(str, shape))
    return VerificationCase(
        name=f"cast-roundtrip-{dtype_name}-{shape_name}",
        reference=program,
        candidate=torch.compile(program, backend="inductor"),
        inputs=inputs,
        config=ComparisonConfig(rtol=0, atol=0),
        metadata={
            "source": "https://github.com/pytorch/pytorch/issues/179561",
            "input_dtype": "float32",
            "cast_dtype": dtype_name,
            "shape": list(shape),
        },
    )


def _gradient_case(batch, size, width, start, end, dtype):
    torch.manual_seed(0)
    inputs = (
        torch.randn(batch, size, width, device="cuda", dtype=dtype, requires_grad=True),
        torch.randn(
            batch,
            end - start,
            width,
            device="cuda",
            dtype=dtype,
            requires_grad=True,
        ),
    )
    upstream = torch.randn(batch, width, device="cuda", dtype=dtype)

    def program(value, replacement):
        scattered = torch.slice_scatter(
            value, replacement, dim=1, start=start, end=end
        )
        return scattered.sum(dim=1)

    dtype_name = str(dtype).removeprefix("torch.")
    return VerificationCase(
        name=f"slice-scatter-gradient-{dtype_name}-{batch}x{size}x{width}-{start}-{end}",
        reference=program,
        candidate=torch.compile(program, backend="inductor"),
        inputs=inputs,
        config=ComparisonConfig(
            rtol=1e-5 if dtype == torch.float32 else 1e-12,
            atol=1e-6 if dtype == torch.float32 else 1e-12,
            check_gradients=True,
        ),
        gradient_loss=lambda output: (output * upstream).sum(),
        metadata={
            "source": "https://github.com/pytorch/pytorch/issues/180164",
            "dtype": dtype_name,
            "shape": [batch, size, width],
            "slice": [start, end],
        },
    )


def _user_triton_case():
    torch.manual_seed(0)
    inputs = (
        torch.randn(2, 2, device="cuda"),
        torch.randn(2, 2, device="cuda"),
    )

    def program(value, other):
        transposed = value.t().contiguous().t()
        intermediate = transposed.sin().t()
        output = torch.empty_like(other)
        add_kernel[(intermediate.numel(),)](
            intermediate,
            other,
            output,
            intermediate.numel(),
            BLOCK_SIZE=16,
        )
        return output

    return VerificationCase(
        name="user-triton-layout-control",
        reference=program,
        candidate=torch.compile(program, backend="inductor", fullgraph=True),
        inputs=inputs,
        metadata={
            "source": "https://github.com/pytorch/pytorch/issues/130243",
            "dtype": "float32",
            "shape": [2, 2],
        },
    )


def build_cases():
    if not torch.cuda.is_available():
        raise RuntimeError("gpu_suite.py requires a CUDA-capable runner")
    return [
        _cast_case((4, 32, 32), torch.bfloat16),
        _cast_case((4, 32, 32), torch.float16),
        _cast_case((3, 257, 129), torch.bfloat16),
        _gradient_case(4, 13, 33, 0, 6, torch.float32),
        _gradient_case(3, 20, 7, 5, 12, torch.float64),
        _user_triton_case(),
    ]
