"""PyTorch #195214: BF16 fusion skips rounding before exact equality."""

import math
import os

import torch

from accel_verify import ComparisonConfig


if not torch.cuda.is_available():
    raise RuntimeError("pytorch_195214_cuda.py requires a CUDA-capable runner")

if os.environ.get("EMULATE_PRECISION_CASTS") == "1":
    torch._inductor.config.emulate_precision_casts = True

torch.manual_seed(0)
values = torch.randn((6, 4, 1, 33), device="cuda", dtype=torch.bfloat16)
scaled = values / math.sqrt(32.0)
maximum = scaled.amax(dim=-1, keepdim=True)
numerator = torch.randn((6, 4, 1, 1), device="cuda", dtype=torch.bfloat16)


def reference(maximum, values, numerator):
    scaled = values / math.sqrt(32.0)
    tie_count = (maximum == scaled).sum(dim=-1, keepdim=True)
    return numerator / tie_count, tie_count


candidate = torch.compile(reference, backend="inductor", fullgraph=True)
inputs = (maximum, values, numerator)
config = ComparisonConfig(rtol=0.0, atol=0.0)
metadata = {
    "source": "https://github.com/pytorch/pytorch/issues/195214",
    "surface": "implicit BF16 rounding before fused exact equality",
    "reported_gpu": "NVIDIA GeForce RTX 4070 Laptop GPU",
}


def main():
    expected, expected_count = reference(*inputs)
    actual, actual_count = candidate(*inputs)
    print(torch.__version__, torch.version.git_version)
    print(
        "eager:",
        expected_count.flatten().tolist(),
        bool(torch.isfinite(expected).all()),
    )
    print(
        "compiled:",
        actual_count.flatten().tolist(),
        bool(torch.isfinite(actual).all()),
    )
    torch.testing.assert_close(actual_count, expected_count, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    main()
