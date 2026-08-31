from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import textwrap
import unittest

from accel_verify.cli import main


class CliTests(unittest.TestCase):
    def _write_spec(self, directory: Path, content: str) -> Path:
        path = directory / "verification.py"
        path.write_text(textwrap.dedent(content), encoding="utf-8")
        return path

    def test_simple_contract_writes_reports_and_passes(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            spec = self._write_spec(
                directory,
                """
                import torch

                reference = lambda x: x.sin()
                candidate = lambda x: x.sin()
                inputs = (torch.arange(4, dtype=torch.float32),)
                """,
            )
            json_path = directory / "reports" / "report.json"
            markdown_path = directory / "reports" / "report.md"

            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main(
                    [
                        "verify",
                        str(spec),
                        "--json",
                        str(json_path),
                        "--markdown",
                        str(markdown_path),
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertIn("ACCEL-VERIFY SUITE: PASS", output.getvalue())
            self.assertEqual(json.loads(json_path.read_text())["status"], "pass")
            self.assertIn("**Status:** PASS", markdown_path.read_text())

    def test_build_cases_contract_fails_on_mismatch(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            spec = self._write_spec(
                directory,
                """
                import torch
                from accel_verify import VerificationCase

                def build_cases():
                    for offset in (0, 1):
                        yield VerificationCase(
                            name=f"offset-{offset}",
                            reference=lambda x: x,
                            candidate=lambda x, offset=offset: x + offset,
                            inputs=(torch.zeros(2),),
                        )
                """,
            )
            json_path = directory / "report.json"

            with redirect_stdout(io.StringIO()):
                exit_code = main(
                    ["verify", str(spec), "--json", str(json_path)]
                )

            self.assertEqual(exit_code, 1)
            report = json.loads(json_path.read_text())
            self.assertEqual(report["summary"], {"total": 2, "passed": 1, "failed": 1})

    def test_invalid_contract_returns_usage_error(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            spec = self._write_spec(directory, "value = 1")

            error = io.StringIO()
            with redirect_stderr(error):
                exit_code = main(["verify", str(spec)])

            self.assertEqual(exit_code, 2)
            self.assertIn("missing: reference, candidate, inputs", error.getvalue())


if __name__ == "__main__":
    unittest.main()
