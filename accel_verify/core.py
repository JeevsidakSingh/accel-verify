from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import platform
import sys
from typing import Any

import torch


@dataclass(frozen=True)
class ComparisonConfig:
    rtol: float = 1e-5
    atol: float = 1e-8
    equal_nan: bool = True
    check_dtype: bool = True
    check_gradients: bool = False
    seed: int = 0


@dataclass(frozen=True)
class Mismatch:
    path: str
    kind: str
    message: str
    max_abs_error: float | None = None
    max_rel_error: float | None = None


@dataclass(frozen=True)
class VerificationReport:
    output_mismatches: tuple[Mismatch, ...] = ()
    gradient_mismatches: tuple[Mismatch, ...] = ()
    execution_errors: tuple[Mismatch, ...] = ()
    gradients_checked: bool = False
    environment: Mapping[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not (
            self.output_mismatches
            or self.gradient_mismatches
            or self.execution_errors
        )


@dataclass(frozen=True)
class _RunResult:
    output: Any = None
    gradients: tuple[torch.Tensor | None, ...] = ()
    error: Exception | None = None


def capture_environment() -> dict[str, Any]:
    cuda_devices = []
    if torch.cuda.is_available():
        cuda_devices = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": cuda_devices,
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
    }


def _clone_argument(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        clone = value.detach().clone(memory_format=torch.preserve_format)
        clone.requires_grad_(value.requires_grad)
        return clone
    return value


def _tensor_leaves(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, Mapping):
        leaves: list[torch.Tensor] = []
        for key in sorted(value, key=str):
            leaves.extend(_tensor_leaves(value[key]))
        return leaves
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        leaves = []
        for item in value:
            leaves.extend(_tensor_leaves(item))
        return leaves
    return []


def _default_gradient_loss(output: Any) -> torch.Tensor:
    terms = []
    for tensor in _tensor_leaves(output):
        if not (tensor.is_floating_point() or tensor.is_complex()):
            continue
        if not tensor.requires_grad:
            continue
        terms.append(tensor.real.sum() if tensor.is_complex() else tensor.sum())
    if not terms:
        raise ValueError("no differentiable tensor output is available for gradient checking")
    return sum(terms[1:], terms[0])


def _cuda_device_indices(args: tuple[Any, ...]) -> list[int]:
    indices = {
        tensor.device.index or 0
        for tensor in _tensor_leaves(args)
        if tensor.device.type == "cuda"
    }
    return sorted(indices)


def _run(
    function: Callable[..., Any],
    args: tuple[Any, ...],
    config: ComparisonConfig,
    gradient_loss: Callable[[Any], torch.Tensor] | None,
) -> _RunResult:
    cloned_args = tuple(_clone_argument(arg) for arg in args)
    devices = _cuda_device_indices(cloned_args)

    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(config.seed)
            output = function(*cloned_args)
            if not config.check_gradients:
                return _RunResult(output=output)

            grad_inputs = tuple(
                arg
                for arg in cloned_args
                if isinstance(arg, torch.Tensor) and arg.requires_grad
            )
            if not grad_inputs:
                raise ValueError(
                    "gradient checking requires at least one top-level tensor input "
                    "with requires_grad=True"
                )

            loss = (gradient_loss or _default_gradient_loss)(output)
            if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
                raise ValueError("gradient_loss must return a scalar tensor")
            gradients = torch.autograd.grad(loss, grad_inputs, allow_unused=True)
            return _RunResult(output=output, gradients=gradients)
    except Exception as error:  # The report must preserve candidate execution failures.
        return _RunResult(error=error)


def _error_metrics(
    expected: torch.Tensor, actual: torch.Tensor, close: torch.Tensor
) -> tuple[float | None, float | None]:
    mismatched = ~close
    if not bool(mismatched.any()):
        return None, None

    expected_values = expected[mismatched]
    actual_values = actual[mismatched]
    abs_error = (actual_values - expected_values).abs()
    finite_abs = abs_error[torch.isfinite(abs_error)]
    max_abs = float(finite_abs.max().item()) if finite_abs.numel() else float("inf")

    real_dtype = expected.real.dtype if expected.is_complex() else expected.dtype
    tiny = torch.finfo(real_dtype).tiny
    denominator = expected_values.abs().clamp_min(tiny)
    rel_error = abs_error / denominator
    finite_rel = rel_error[torch.isfinite(rel_error)]
    max_rel = float(finite_rel.max().item()) if finite_rel.numel() else float("inf")
    return max_abs, max_rel


def _compare_tensors(
    expected: torch.Tensor,
    actual: torch.Tensor,
    path: str,
    config: ComparisonConfig,
) -> list[Mismatch]:
    if expected.shape != actual.shape:
        return [
            Mismatch(
                path,
                "shape",
                f"expected shape {tuple(expected.shape)}, got {tuple(actual.shape)}",
            )
        ]
    if config.check_dtype and expected.dtype != actual.dtype:
        return [
            Mismatch(
                path,
                "dtype",
                f"expected dtype {expected.dtype}, got {actual.dtype}",
            )
        ]

    if expected.is_floating_point() or expected.is_complex():
        try:
            close = torch.isclose(
                expected,
                actual,
                rtol=config.rtol,
                atol=config.atol,
                equal_nan=config.equal_nan,
            )
        except RuntimeError as error:
            return [Mismatch(path, "comparison", str(error))]
        if bool(close.all()):
            return []
        max_abs, max_rel = _error_metrics(expected, actual, close)
        mismatched = int((~close).sum().item())
        return [
            Mismatch(
                path,
                "value",
                f"{mismatched}/{expected.numel()} elements differ beyond "
                f"rtol={config.rtol:g}, atol={config.atol:g}",
                max_abs,
                max_rel,
            )
        ]

    if torch.equal(expected, actual):
        return []
    mismatched = int((expected != actual).sum().item())
    return [
        Mismatch(
            path,
            "value",
            f"{mismatched}/{expected.numel()} elements differ (exact comparison)",
        )
    ]


def _compare_values(
    expected: Any,
    actual: Any,
    path: str,
    config: ComparisonConfig,
) -> list[Mismatch]:
    if isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor):
        return _compare_tensors(expected, actual, path, config)
    if isinstance(expected, torch.Tensor) or isinstance(actual, torch.Tensor):
        return [
            Mismatch(
                path,
                "type",
                f"expected {type(expected).__name__}, got {type(actual).__name__}",
            )
        ]

    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        mismatches: list[Mismatch] = []
        expected_keys = set(expected)
        actual_keys = set(actual)
        if expected_keys != actual_keys:
            mismatches.append(
                Mismatch(
                    path,
                    "keys",
                    f"expected keys {sorted(expected_keys, key=str)}, "
                    f"got {sorted(actual_keys, key=str)}",
                )
            )
        for key in sorted(expected_keys & actual_keys, key=str):
            mismatches.extend(
                _compare_values(expected[key], actual[key], f"{path}[{key!r}]", config)
            )
        return mismatches

    sequence_types = (list, tuple)
    if isinstance(expected, sequence_types) and isinstance(actual, sequence_types):
        if type(expected) is not type(actual):
            return [
                Mismatch(
                    path,
                    "type",
                    f"expected {type(expected).__name__}, got {type(actual).__name__}",
                )
            ]
        if len(expected) != len(actual):
            return [
                Mismatch(
                    path,
                    "length",
                    f"expected length {len(expected)}, got {len(actual)}",
                )
            ]
        mismatches = []
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            mismatches.extend(
                _compare_values(
                    expected_item, actual_item, f"{path}[{index}]", config
                )
            )
        return mismatches

    if type(expected) is not type(actual):
        return [
            Mismatch(
                path,
                "type",
                f"expected {type(expected).__name__}, got {type(actual).__name__}",
            )
        ]
    if expected == actual:
        return []
    return [Mismatch(path, "value", f"expected {expected!r}, got {actual!r}")]


def verify(
    reference: Callable[..., Any],
    candidate: Callable[..., Any],
    inputs: tuple[Any, ...],
    *,
    config: ComparisonConfig | None = None,
    gradient_loss: Callable[[Any], torch.Tensor] | None = None,
) -> VerificationReport:
    config = config or ComparisonConfig()
    reference_result = _run(reference, inputs, config, gradient_loss)
    candidate_result = _run(candidate, inputs, config, gradient_loss)

    execution_errors = []
    if reference_result.error is not None:
        execution_errors.append(
            Mismatch(
                "reference",
                "execution",
                f"{type(reference_result.error).__name__}: {reference_result.error}",
            )
        )
    if candidate_result.error is not None:
        execution_errors.append(
            Mismatch(
                "candidate",
                "execution",
                f"{type(candidate_result.error).__name__}: {candidate_result.error}",
            )
        )
    if execution_errors:
        return VerificationReport(
            execution_errors=tuple(execution_errors),
            gradients_checked=config.check_gradients,
            environment=capture_environment(),
        )

    output_mismatches = _compare_values(
        reference_result.output, candidate_result.output, "output", config
    )
    gradient_mismatches = []
    if config.check_gradients:
        gradient_mismatches = _compare_values(
            reference_result.gradients,
            candidate_result.gradients,
            "gradients",
            config,
        )

    return VerificationReport(
        output_mismatches=tuple(output_mismatches),
        gradient_mismatches=tuple(gradient_mismatches),
        gradients_checked=config.check_gradients,
        environment=capture_environment(),
    )


def render_report(report: VerificationReport) -> str:
    lines = [f"ACCEL-VERIFY: {'PASS' if report.passed else 'FAIL'}"]
    if report.execution_errors:
        lines.append("outputs: NOT COMPARED")
        lines.append("gradients: NOT COMPARED")
    else:
        lines.append(f"outputs: {'PASS' if not report.output_mismatches else 'FAIL'}")
        if report.gradients_checked:
            lines.append(
                f"gradients: {'PASS' if not report.gradient_mismatches else 'FAIL'}"
            )
        else:
            lines.append("gradients: SKIPPED")

    mismatches = (
        report.execution_errors
        + report.output_mismatches
        + report.gradient_mismatches
    )
    if mismatches:
        lines.append("")
        lines.append("mismatches:")
        for mismatch in mismatches:
            detail = f"- {mismatch.path} [{mismatch.kind}]: {mismatch.message}"
            if mismatch.max_abs_error is not None:
                detail += f"; max_abs={mismatch.max_abs_error:.6g}"
            if mismatch.max_rel_error is not None:
                detail += f"; max_rel={mismatch.max_rel_error:.6g}"
            lines.append(detail)

    lines.append("")
    lines.append("environment:")
    for key, value in report.environment.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)
