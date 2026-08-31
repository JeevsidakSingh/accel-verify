# Public issue reproductions

This page records bounded Accel-Verify replays of public compiler-correctness issues. A pass means only that the reported trigger did not diverge in the listed environment; it does not disprove a version-, operating-system-, or hardware-specific report. A detected divergence is also not automatically proof of a compiler bug: the reference contract and numerical conditioning still need review.

## Current CPU replay

Environment: macOS 26.3 arm64, Python 3.11.15, PyTorch 2.13.0, Inductor CPU. Replayed 2026-08-30.

| Public issue | Failure surface | Result |
| --- | --- | --- |
| [PyTorch #174074](https://github.com/pytorch/pytorch/issues/174074) | Duplicate advanced indices compound an in-place update | PASS on this environment |
| [PyTorch #177821](https://github.com/pytorch/pytorch/issues/177821) | Complex indexing assignment is ignored | FAIL: 4/4 output elements differed |
| [PyTorch #181695](https://github.com/pytorch/pytorch/issues/181695) | Sign-sensitive branch on a near-zero pooled reduction | DIVERGENCE: 15,212/32,955 output elements differed; max abs 4.17569. Maintainer analysis classified this as permitted reduction-order sensitivity, not an Inductor bug. |
| [PyTorch #182131](https://github.com/pytorch/pytorch/issues/182131) | FP16 cast semantics before an elementwise add | DIVERGENCE by default: 8/24 intermediate elements differed; max abs 0.000976562. PASS with `TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1`, matching the documented resolution. |
| [PyTorch #186029](https://github.com/pytorch/pytorch/issues/186029) | `count_nonzero` over `bincount` output | PASS on this environment |

The corresponding executable files are in [`repros/`](repros).

## Historical RTX 3090 pre-fix replay

Environment: NVIDIA RTX 3090, driver 580.159.03. Replayed on PyTorch 2.8.0+cu128 / Triton 3.4.0 and PyTorch 2.12.0.dev20260408+cu128 / Triton 3.7.0.

Both tested builds predated the issue-closing fixes: #179561 closed on 2026-04-17 and #180164 closed on 2026-04-14. These runs confirm that Accel-Verify detects the published historical failures; they do **not** show that either failure persists in a current post-fix build.

| Public issue | Failure surface | Result |
| --- | --- | --- |
| [PyTorch #179561](https://github.com/pytorch/pytorch/issues/179561) | BF16 precision cast silently elided during fusion | HISTORICAL FAILURE detected on both pre-fix builds; max abs 0.1094589 |
| [PyTorch #180164](https://github.com/pytorch/pytorch/issues/180164) | Correct forward result but silently zeroed input gradient | HISTORICAL FAILURE detected on both pre-fix builds; eager gradient sum 140.6300, compiled input gradient entirely zero |
| [PyTorch #130243](https://github.com/pytorch/pytorch/issues/130243) | User Triton kernel and Inductor stride reordering | PASS on both builds, consistent with a fixed historical issue |

These results demonstrate detection after a trigger is known. They do not establish current regression status, automatic trigger discovery, production incident frequency, or customer demand.

## Current RTX 3090 stable/nightly replay

Environment: NVIDIA RTX 3090, compute capability 8.6, Linux x86_64. Replayed 2026-08-31 on:

- PyTorch 2.8.0+cu128 / Triton 3.4.0
- PyTorch 2.13.0+cu126 (`cf30153c4c131c8164ee7798e5022d810682e2cb`) / Triton 3.7.1
- PyTorch 2.15.0.dev20260829+cu126 (`332a69317e22b105a867838624a87984e05021e2`) / Triton 3.8.0

| Public issue | Failure surface | Result |
| --- | --- | --- |
| [PyTorch #195214](https://github.com/pytorch/pytorch/issues/195214) | BF16 rounding elided before an exact equality, producing zero tie counts | FAIL on all three builds: 24/24 compiled tie counts were zero and outputs were non-finite. PASS on all three with `emulate_precision_casts=True`. |
| [PyTorch #188890](https://github.com/pytorch/pytorch/issues/188890) | Non-deterministic compiled BF16 gradients | PyTorch 2.13.0: 5/5 fresh processes produced finite relative gradient differences of 0.495629–0.603517. PyTorch 2.8.0 and 5/5 current-nightly processes produced exact repeated gradients. |
| [PyTorch #194062](https://github.com/pytorch/pytorch/issues/194062) | Symbolic `torch.full(..., dtype=bool)` drops its cast | FAIL on all three CUDA builds: eager returned `2`, compiled returned `6`. |

For #188890, one exact 2.13.0 verifier run found matching outputs but 223/896 gradient elements different, with max absolute error 0.239746. The issue's standalone threshold marks only `grad_rel >= 1.0` as failed, so the 49.6–60.4% relative differences still print `failed=False`.

These runs add bounded backend and hardware scope. They do not prove that a result generalizes to GPUs not tested here, and they remain public repros rather than evidence of external adoption.
