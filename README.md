# Accel-Verify

[![CI](https://github.com/JeevsidakSingh/accel-verify/actions/workflows/ci.yml/badge.svg)](https://github.com/JeevsidakSingh/accel-verify/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Accel-Verify catches silent output and input-gradient divergences between a reference PyTorch implementation and an accelerated candidate such as `torch.compile`, Inductor, or a Triton-backed function.

It is an early alpha focused on one job: turn semantic divergence into a reproducible local or CI failure with useful environment evidence.

## Install

```bash
pip install git+https://github.com/JeevsidakSingh/accel-verify.git@v0.1.0
```

Accel-Verify uses the PyTorch installation already selected for your environment. The GitHub Action installs nothing, so it will not silently replace the framework or CUDA stack you intend to verify.

## Create a verification file

```python
# verification.py
import torch

from accel_verify import ComparisonConfig, VerificationCase


def program(value):
    return torch.sin(value) * value


def build_cases():
    for size in (8, 32, 128):
        yield VerificationCase(
            name=f"compiled-control-{size}",
            reference=program,
            candidate=torch.compile(program, backend="inductor"),
            inputs=(torch.randn(size, requires_grad=True),),
            config=ComparisonConfig(check_gradients=True),
            metadata={"shape": [size], "dtype": "float32"},
        )
```

Run it locally:

```bash
accel-verify verify verification.py \
  --json accel-verify-results/report.json \
  --markdown accel-verify-results/report.md
```

Exit codes are part of the interface:

| Code | Meaning |
| ---: | --- |
| `0` | Every verification case passed |
| `1` | Semantic divergence was detected |
| `2` | The suite could not be loaded or executed |

## GitHub Action

Install the framework and candidate-specific dependencies you want to test, then invoke the action at a release or full commit SHA:

```yaml
- uses: JeevsidakSingh/accel-verify@v0.1.0
  id: verify
  with:
    spec: verification.py
    report-directory: accel-verify-results

- uses: actions/upload-artifact@v6
  if: always()
  with:
    name: accel-verify-results
    path: accel-verify-results/
```

The action runs on the caller's selected runner. CUDA verification therefore requires a CUDA-capable runner with the intended PyTorch build already installed.

Action outputs include:

- `status`: `pass` or `fail`
- `exit-code`: verifier process exit code
- `json-report`: generated JSON report path
- `markdown-report`: generated Markdown report path

## Verification contracts

A verification file may expose:

- `reference`, `candidate`, and `inputs` for one simple case
- `CASES`, an iterable of `VerificationCase` objects
- `build_cases()`, returning or yielding `VerificationCase` objects

Reports capture structured output differences, shape or dtype mismatches, input-gradient differences, execution errors, case metadata, and the Python/PyTorch/CUDA/GPU environment.

## Public repros

The [`repros`](repros) directory contains executable adaptations of public PyTorch compiler issues. They are regression examples, not claims that every historical issue still fails on every framework, operating system, or accelerator version.

## Current scope

Supported now:

- Reference versus accelerated callable execution
- Nested output comparison
- Input-gradient comparison
- Configurable `rtol` and `atol`
- Multi-case shape and dtype sweeps
- JSON and Markdown reports
- Deterministic environment capture
- GitHub Actions failure propagation

Not implemented yet:

- Automatic input or trigger generation
- Automatic reproducer minimization
- Hosted hardware provisioning
- Distributed-workload orchestration
- Performance-regression measurement

Numerical differences are not automatically proof of a compiler bug. Users remain responsible for selecting meaningful references and tolerance policies.

## Development

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidance and [SECURITY.md](SECURITY.md) for private vulnerability reporting.

## License

Apache License 2.0.
