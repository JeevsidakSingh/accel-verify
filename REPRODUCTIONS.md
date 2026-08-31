# Public issue reproductions

This page records bounded Accel-Verify replays of public compiler-correctness issues. A pass means only that the reported trigger did not diverge in the listed environment; it does not disprove a version-, operating-system-, or hardware-specific report.

## Current CPU replay

Environment: macOS 26.3 arm64, Python 3.11.15, PyTorch 2.13.0, Inductor CPU. Replayed 2026-08-30.

| Public issue | Failure surface | Result |
| --- | --- | --- |
| [PyTorch #174074](https://github.com/pytorch/pytorch/issues/174074) | Duplicate advanced indices compound an in-place update | PASS on this environment |
| [PyTorch #177821](https://github.com/pytorch/pytorch/issues/177821) | Complex indexing assignment is ignored | FAIL: 4/4 output elements differed |
| [PyTorch #181695](https://github.com/pytorch/pytorch/issues/181695) | Broadcast shapes under `torch.where` | FAIL: 15,212/32,955 output elements differed; max abs 4.17569 |
| [PyTorch #182131](https://github.com/pytorch/pytorch/issues/182131) | FP16 cast semantics before an elementwise add | FAIL: 8/24 intermediate elements differed; max abs 0.000976562 |
| [PyTorch #186029](https://github.com/pytorch/pytorch/issues/186029) | `count_nonzero` over `bincount` output | PASS on this environment |

The corresponding executable files are in [`repros/`](repros).

## RTX 3090 stable/nightly replay

Environment: NVIDIA RTX 3090, driver 580.159.03. Replayed on PyTorch 2.8.0+cu128 / Triton 3.4.0 and PyTorch 2.12.0.dev20260408+cu128 / Triton 3.7.0.

| Public issue | Failure surface | Result |
| --- | --- | --- |
| [PyTorch #179561](https://github.com/pytorch/pytorch/issues/179561) | BF16 precision cast silently elided during fusion | FAIL on both builds; max abs 0.1094589 |
| [PyTorch #180164](https://github.com/pytorch/pytorch/issues/180164) | Correct forward result but silently zeroed input gradient | FAIL on both builds; eager gradient sum 140.6300, compiled input gradient entirely zero |
| [PyTorch #130243](https://github.com/pytorch/pytorch/issues/130243) | User Triton kernel and Inductor stride reordering | PASS on both builds, consistent with a fixed historical issue |

These results demonstrate detection after a trigger is known. They do not establish automatic trigger discovery, production incident frequency, or customer demand.
