import torch

from accel_verify import ComparisonConfig, VerificationCase


def forward(value):
    return torch.sin(value) * value


def build_cases():
    for size in (8, 31):
        reference = forward
        candidate = torch.compile(forward, backend="eager", fullgraph=True)
        yield VerificationCase(
            name=f"cpu-smoke-{size}",
            reference=reference,
            candidate=candidate,
            inputs=(torch.linspace(-2, 2, size, requires_grad=True),),
            config=ComparisonConfig(check_gradients=True),
            metadata={"shape": [size], "dtype": "float32"},
        )
