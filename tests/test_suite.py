import unittest

import torch

from accel_verify import (
    ComparisonConfig,
    VerificationCase,
    render_suite_markdown,
    run_suite,
    suite_to_dict,
)


class SuiteTests(unittest.TestCase):
    def test_suite_summarizes_pass_and_failure(self):
        suite = run_suite(
            [
                VerificationCase(
                    name="equivalent",
                    reference=lambda x: x.sin(),
                    candidate=lambda x: x.sin(),
                    inputs=(torch.randn(4),),
                ),
                VerificationCase(
                    name="different",
                    reference=lambda x: x,
                    candidate=lambda x: x + 1,
                    inputs=(torch.zeros(2),),
                ),
            ]
        )

        self.assertFalse(suite.passed)
        self.assertEqual(suite.passed_count, 1)
        self.assertEqual(suite.failed_count, 1)
        serialized = suite_to_dict(suite)
        self.assertEqual(serialized["status"], "fail")
        self.assertEqual(serialized["summary"]["total"], 2)
        self.assertEqual(serialized["cases"][1]["outputs"]["status"], "fail")

    def test_markdown_exposes_gradient_only_failure(self):
        value = torch.randn(3, requires_grad=True)

        class WrongGradient(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x.clone()

            @staticmethod
            def backward(ctx, gradient):
                return torch.zeros_like(gradient)

        suite = run_suite(
            [
                VerificationCase(
                    name="gradient",
                    reference=lambda x: x.square(),
                    candidate=lambda x: WrongGradient.apply(x.square()),
                    inputs=(value,),
                    config=ComparisonConfig(check_gradients=True),
                )
            ]
        )

        markdown = render_suite_markdown(suite)
        self.assertIn("| gradient | FAIL | PASS | FAIL |", markdown)
        self.assertIn("gradient", markdown.lower())

    def test_duplicate_case_names_are_rejected(self):
        case = VerificationCase(
            name="duplicate",
            reference=lambda x: x,
            candidate=lambda x: x,
            inputs=(torch.ones(1),),
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            run_suite([case, case])

    def test_empty_suite_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            run_suite([])


if __name__ == "__main__":
    unittest.main()
