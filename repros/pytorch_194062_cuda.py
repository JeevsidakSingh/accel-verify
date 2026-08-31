"""CUDA extension of PyTorch #194062's symbolic torch.full repro."""

import torch
import torch._dynamo as dynamo

from accel_verify import ComparisonConfig


if not torch.cuda.is_available():
    raise RuntimeError("pytorch_194062_cuda.py requires a CUDA-capable runner")

dynamo.config.capture_scalar_outputs = True


def reference(value):
    return torch.full(
        (2,),
        value.item(),
        dtype=torch.bool,
        device=value.device,
    ).sum()


candidate = torch.compile(reference, backend="inductor", fullgraph=True)
inputs = (torch.tensor(3, device="cuda"),)
config = ComparisonConfig(rtol=0.0, atol=0.0)
metadata = {
    "source": "https://github.com/pytorch/pytorch/issues/194062",
    "surface": "symbolic torch.full dtype cast on the CUDA/Triton backend",
    "extension": "forces the torch.full output onto the input CUDA device",
}


def main():
    expected = reference(*inputs)
    actual = candidate(*inputs)
    print(torch.__version__, torch.version.git_version)
    print("eager:", expected)
    print("compiled:", actual)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    main()
