"""CUDA control case for the ephemeral GitHub self-hosted runner."""

import torch

from accel_verify import ComparisonConfig, VerificationCase


def build_cases():
    if not torch.cuda.is_available():
        raise RuntimeError("gpu_ci_control.py requires a CUDA-capable runner")

    torch.manual_seed(0)
    inputs = (
        torch.randn(32, 64, device="cuda", requires_grad=True),
        torch.randn(64, 16, device="cuda", requires_grad=True),
    )

    def program(value, weight):
        return torch.nn.functional.gelu(value @ weight)

    return [
        VerificationCase(
            name="cuda-compile-control",
            reference=program,
            candidate=torch.compile(program, backend="inductor", fullgraph=True),
            inputs=inputs,
            config=ComparisonConfig(
                rtol=1e-4,
                atol=1e-5,
                check_gradients=True,
            ),
            metadata={
                "purpose": "prove the action passes a valid CUDA compile case",
                "dtype": "float32",
                "shapes": [[32, 64], [64, 16]],
            },
        )
    ]
