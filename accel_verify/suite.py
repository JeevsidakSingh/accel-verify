from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
import math
from typing import Any

import torch

from .core import ComparisonConfig, VerificationReport, render_report, verify


@dataclass(frozen=True)
class VerificationCase:
    name: str
    reference: Callable[..., Any]
    candidate: Callable[..., Any]
    inputs: tuple[Any, ...]
    config: ComparisonConfig = field(default_factory=ComparisonConfig)
    gradient_loss: Callable[[Any], torch.Tensor] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CaseResult:
    name: str
    report: VerificationReport
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SuiteReport:
    cases: tuple[CaseResult, ...]

    @property
    def passed(self) -> bool:
        return all(case.report.passed for case in self.cases)

    @property
    def passed_count(self) -> int:
        return sum(case.report.passed for case in self.cases)

    @property
    def failed_count(self) -> int:
        return len(self.cases) - self.passed_count


def run_suite(cases: Iterable[VerificationCase]) -> SuiteReport:
    case_list = list(cases)
    if not case_list:
        raise ValueError("verification suite must contain at least one case")

    names: set[str] = set()
    results = []
    for case in case_list:
        if not isinstance(case, VerificationCase):
            raise TypeError("every suite item must be a VerificationCase")
        if not case.name.strip():
            raise ValueError("verification case names cannot be empty")
        if case.name in names:
            raise ValueError(f"duplicate verification case name: {case.name}")
        if not isinstance(case.inputs, tuple):
            raise TypeError(f"inputs for {case.name!r} must be a tuple")
        names.add(case.name)
        results.append(
            CaseResult(
                name=case.name,
                report=verify(
                    case.reference,
                    case.candidate,
                    case.inputs,
                    config=case.config,
                    gradient_loss=case.gradient_loss,
                ),
                metadata=dict(case.metadata),
            )
        )
    return SuiteReport(tuple(results))


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def suite_to_dict(suite: SuiteReport) -> dict[str, Any]:
    cases = []
    for case in suite.cases:
        report = case.report
        cases.append(
            {
                "name": case.name,
                "status": "pass" if report.passed else "fail",
                "metadata": _json_safe(case.metadata),
                "outputs": {
                    "status": (
                        "not_compared"
                        if report.execution_errors
                        else "pass" if not report.output_mismatches else "fail"
                    ),
                    "mismatches": [
                        _json_safe(asdict(item)) for item in report.output_mismatches
                    ],
                },
                "gradients": {
                    "status": (
                        "not_compared"
                        if report.execution_errors
                        else "skipped"
                        if not report.gradients_checked
                        else "pass"
                        if not report.gradient_mismatches
                        else "fail"
                    ),
                    "mismatches": [
                        _json_safe(asdict(item)) for item in report.gradient_mismatches
                    ],
                },
                "execution_errors": [
                    _json_safe(asdict(item)) for item in report.execution_errors
                ],
                "environment": _json_safe(report.environment),
            }
        )
    return {
        "schema_version": 1,
        "status": "pass" if suite.passed else "fail",
        "summary": {
            "total": len(suite.cases),
            "passed": suite.passed_count,
            "failed": suite.failed_count,
        },
        "cases": cases,
    }


def _component_status(case: CaseResult) -> tuple[str, str]:
    report = case.report
    if report.execution_errors:
        return "NOT COMPARED", "NOT COMPARED"
    outputs = "PASS" if not report.output_mismatches else "FAIL"
    if not report.gradients_checked:
        gradients = "SKIPPED"
    else:
        gradients = "PASS" if not report.gradient_mismatches else "FAIL"
    return outputs, gradients


def render_suite_markdown(suite: SuiteReport) -> str:
    status = "PASS" if suite.passed else "FAIL"
    lines = [
        "# Accel-Verify Report",
        "",
        f"**Status:** {status}",
        "",
        f"{suite.passed_count} passed, {suite.failed_count} failed, "
        f"{len(suite.cases)} total.",
        "",
        "| Case | Status | Outputs | Gradients |",
        "| --- | --- | --- | --- |",
    ]
    for case in suite.cases:
        outputs, gradients = _component_status(case)
        case_status = "PASS" if case.report.passed else "FAIL"
        lines.append(f"| {case.name} | {case_status} | {outputs} | {gradients} |")

    for case in suite.cases:
        if case.report.passed:
            continue
        lines.extend(["", f"## {case.name}", "", "```text"])
        lines.extend(render_report(case.report).splitlines())
        lines.append("```")
    return "\n".join(lines) + "\n"


def render_suite_text(suite: SuiteReport) -> str:
    status = "PASS" if suite.passed else "FAIL"
    lines = [
        f"ACCEL-VERIFY SUITE: {status}",
        f"cases: {len(suite.cases)} total, {suite.passed_count} passed, "
        f"{suite.failed_count} failed",
    ]
    for case in suite.cases:
        case_status = "PASS" if case.report.passed else "FAIL"
        lines.append(f"- {case.name}: {case_status}")
        if not case.report.passed:
            for line in render_report(case.report).splitlines()[1:]:
                lines.append(f"  {line}")
    return "\n".join(lines)
