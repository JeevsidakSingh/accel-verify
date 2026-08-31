"""PyTorch #181695: broadcasting in torch.where produces wrong output."""

import torch
from torch import nn

from accel_verify import ComparisonConfig, render_report, verify


torch.manual_seed(0)
instance_norm = nn.InstanceNorm2d(13)
pool = nn.AdaptiveAvgPool2d((1, 1))


def program(x):
    normalized = instance_norm(x)
    pooled = pool(normalized)
    pooled_twice = pool(pooled)
    selected = torch.where(pooled > 0, pooled, normalized)
    return torch.sub(pooled_twice, selected)


inputs = (torch.randn(15, 13, 13, 13),)
candidate = torch.compile(program, backend="inductor")
report = verify(program, candidate, inputs, config=ComparisonConfig())

print("source: https://github.com/pytorch/pytorch/issues/181695")
print(render_report(report))

