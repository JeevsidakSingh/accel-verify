import unittest

import torch

from accel_verify import ComparisonConfig, render_report, verify


class WrongGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        return torch.zeros_like(gradient)


class VerifyTests(unittest.TestCase):
    def test_equivalent_implementations_pass(self):
        value = torch.randn(4, 8, requires_grad=True)
        report = verify(
            lambda x: {"value": torch.sin(x) * x},
            lambda x: {"value": torch.sin(x) * x},
            (value,),
            config=ComparisonConfig(check_gradients=True),
        )
        self.assertTrue(report.passed, render_report(report))

    def test_output_value_mismatch_fails(self):
        value = torch.ones(3)
        report = verify(lambda x: x * 2, lambda x: x * 3, (value,))
        self.assertFalse(report.passed)
        self.assertEqual(report.output_mismatches[0].kind, "value")
        self.assertEqual(report.output_mismatches[0].max_abs_error, 1.0)

    def test_shape_mismatch_fails(self):
        value = torch.ones(2, 3)
        report = verify(lambda x: x.sum(dim=-1), lambda x: x, (value,))
        self.assertEqual(report.output_mismatches[0].kind, "shape")

    def test_gradient_only_mismatch_fails(self):
        value = torch.randn(5, requires_grad=True)

        def reference(x):
            return x.square()

        def candidate(x):
            return WrongGradient.apply(x.square())

        report = verify(
            reference,
            candidate,
            (value,),
            config=ComparisonConfig(check_gradients=True),
        )
        self.assertFalse(report.output_mismatches)
        self.assertTrue(report.gradient_mismatches)

    def test_tolerance_controls_numerical_mismatch(self):
        value = torch.tensor([1.0])
        candidate = lambda x: x + 1e-4
        loose = verify(
            lambda x: x,
            candidate,
            (value,),
            config=ComparisonConfig(rtol=1e-3, atol=0),
        )
        strict = verify(
            lambda x: x,
            candidate,
            (value,),
            config=ComparisonConfig(rtol=1e-6, atol=0),
        )
        self.assertTrue(loose.passed)
        self.assertFalse(strict.passed)

    def test_matched_rng_state(self):
        value = torch.ones(16)
        report = verify(
            lambda x: torch.nn.functional.dropout(x, p=0.5, training=True),
            lambda x: torch.nn.functional.dropout(x, p=0.5, training=True),
            (value,),
        )
        self.assertTrue(report.passed, render_report(report))

    def test_candidate_exception_is_reported(self):
        def candidate(_):
            raise RuntimeError("compiler failed")

        report = verify(lambda x: x, candidate, (torch.ones(1),))
        self.assertFalse(report.passed)
        self.assertIn("compiler failed", report.execution_errors[0].message)
        self.assertIn("outputs: NOT COMPARED", render_report(report))

    def test_report_marks_unrequested_gradients_as_skipped(self):
        report = verify(lambda x: x, lambda x: x, (torch.ones(1),))
        self.assertIn("gradients: SKIPPED", render_report(report))



if __name__ == "__main__":
    unittest.main()
