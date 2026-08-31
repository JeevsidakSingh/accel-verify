"""CPU-safe fixture proving that semantic divergence fails the action."""

import torch


reference = lambda value: value
candidate = lambda value: value + 1
inputs = (torch.zeros(4),)
case_name = "expected-semantic-mismatch"
