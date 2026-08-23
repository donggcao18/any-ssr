import importlib.util
import math
import unittest

from expert_selection.lookahead import build_support_schedule
from expert_selection.methods.oia import oia_score
from expert_selection.methods.slu import slu_score


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class LookaheadMathTests(unittest.TestCase):
    def test_oia_formula(self):
        score, diagnostics = oia_score({0: 4.0, 1: 2.0}, {0: 4.0, 1: 3.0}, 1e-12)
        self.assertAlmostEqual(score, math.log(2.0) - math.log(4.0 / 3.0))
        self.assertGreater(diagnostics["source_contraction"], diagnostics["fresh_contraction"])

    def test_slu_formula(self):
        score, diagnostics = slu_score({1: 2.0, 5: 1.0}, {1: 4.0, 5: 2.0}, (1, 5), (0.5, 0.5), 1e-12)
        self.assertAlmostEqual(score, 0.5, places=10)
        self.assertAlmostEqual(diagnostics["utilities"]["1"], 0.5, places=10)

    def test_schedule_cycles_exact_microbatch_count(self):
        schedule = build_support_schedule(3, batch_size=2, optimizer_steps=10, gradient_accumulation=2, seed=9)
        self.assertEqual(len(schedule), 20)
        self.assertTrue(all(batch for batch in schedule))
        self.assertEqual(schedule, build_support_schedule(3, batch_size=2, optimizer_steps=10, gradient_accumulation=2, seed=9))


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in the lightweight local verification runtime")
class GCAGeometryTests(unittest.TestCase):
    def test_capture_and_orthogonal_cases(self):
        import torch
        from expert_selection.methods.gca import captured_energy

        factor_a = torch.tensor([[1.0, 0.0]])
        factor_b = torch.tensor([[1.0], [0.0]])
        captured, total, _ = captured_energy(torch.tensor([[1.0, 0.0], [0.0, 0.0]]), factor_a, factor_b)
        self.assertAlmostEqual(captured / total, 1.0)
        captured, total, _ = captured_energy(torch.tensor([[0.0, 0.0], [0.0, 1.0]]), factor_a, factor_b)
        self.assertAlmostEqual(captured / total, 0.0)

    def test_rank_deficient_factor(self):
        import torch
        from expert_selection.methods.gca import numerical_basis

        basis, info = numerical_basis(torch.tensor([[1.0, 2.0], [2.0, 4.0]]))
        self.assertEqual(info["rank"], 1)
        self.assertEqual(basis.shape[1], 1)

    def test_score_is_invariant_to_equivalent_factor_rescaling(self):
        import torch
        from expert_selection.methods.gca import captured_energy

        generator = torch.Generator().manual_seed(3)
        gradient = torch.randn(5, 4, generator=generator)
        factor_a = torch.randn(2, 4, generator=generator)
        factor_b = torch.randn(5, 2, generator=generator)
        first, total, _ = captured_energy(gradient, factor_a, factor_b)
        second, second_total, _ = captured_energy(gradient, factor_a * 7.0, factor_b / 7.0)
        self.assertAlmostEqual(first / total, second / second_total, places=10)


if __name__ == "__main__":
    unittest.main()
