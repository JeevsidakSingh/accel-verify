from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any
from uuid import uuid4

from .core import ComparisonConfig
from .suite import (
    VerificationCase,
    render_suite_markdown,
    render_suite_text,
    run_suite,
    suite_to_dict,
)


def _load_module(path: Path) -> ModuleType:
    if not path.is_file():
        raise ValueError(f"verification file does not exist: {path}")
    module_name = f"_accel_verify_spec_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load verification file: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _simple_case(module: ModuleType, path: Path) -> VerificationCase:
    missing = [
        name for name in ("reference", "candidate", "inputs") if not hasattr(module, name)
    ]
    if missing:
        names = ", ".join(missing)
        raise ValueError(
            "verification file must define CASES, build_cases(), or the simple "
            f"reference/candidate/inputs contract; missing: {names}"
        )
    config = getattr(module, "config", ComparisonConfig())
    if not isinstance(config, ComparisonConfig):
        raise TypeError("config must be a ComparisonConfig")
    return VerificationCase(
        name=getattr(module, "case_name", path.stem),
        reference=module.reference,
        candidate=module.candidate,
        inputs=module.inputs,
        config=config,
        gradient_loss=getattr(module, "gradient_loss", None),
        metadata=getattr(module, "metadata", {}),
    )


def load_cases(path: Path) -> list[VerificationCase]:
    module = _load_module(path)
    if hasattr(module, "build_cases"):
        builder = module.build_cases
        if not callable(builder):
            raise TypeError("build_cases must be callable")
        value: Any = builder()
    elif hasattr(module, "CASES"):
        value = module.CASES
    else:
        return [_simple_case(module, path)]
    try:
        return list(value)
    except TypeError as error:
        raise TypeError("CASES/build_cases() must provide an iterable") from error


def _write_report(path: str | None, content: str) -> None:
    if path is None:
        return
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(content, encoding="utf-8")


def _verify(args: argparse.Namespace) -> int:
    try:
        suite = run_suite(load_cases(Path(args.spec).resolve()))
        json_content = json.dumps(suite_to_dict(suite), indent=2) + "\n"
        markdown_content = render_suite_markdown(suite)
        _write_report(args.json_path, json_content)
        _write_report(args.markdown_path, markdown_content)
    except Exception as error:
        print(f"ACCEL-VERIFY: ERROR\n{type(error).__name__}: {error}", file=sys.stderr)
        return 2

    print(render_suite_text(suite))
    return 0 if suite.passed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="accel-verify")
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify_parser = subparsers.add_parser(
        "verify", help="compare eager/reference and accelerated implementations"
    )
    verify_parser.add_argument("spec", help="Python verification file")
    verify_parser.add_argument(
        "--json", dest="json_path", help="write a machine-readable JSON report"
    )
    verify_parser.add_argument(
        "--markdown", dest="markdown_path", help="write a Markdown summary"
    )
    verify_parser.set_defaults(handler=_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
