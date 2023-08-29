from unittest import TestCase

from measure_commuting_operators.one_factorisation import k


class TestOneFactorisation(TestCase):
    def test_size_factorisation(self):
        for (n, exp_num_coves) in zip([2, 3, 4, 5, 6], [1, 3, 3, 5, 5]):
            self.assertEqual(len(k(n)), exp_num_coves)
            self.assertEqual(sum(len(cover) for cover in k(n)), n * (n - 1) // 2)
            for cover in k(n):
                for edge in cover:
                    self.assertIsInstance(edge, tuple)
                    self.assertEqual(len(edge), 2)

    def test_too_many_edges(self):
        self.assertRaises(ValueError, lambda: k(1))
        self.assertRaises(ValueError, lambda: k(7))
