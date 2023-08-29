from unittest import TestCase

from measure_commuting_operators.sorted_insertion import sorted_insertion, single_indices_commute, tuple_indices_commute


class TestSortedInsertion(TestCase):
    def test_sorted_insertion(self):
        pauli_indices = [(0, 0), (1, 0), (3, 0), (3, 3)]
        pauli_coefficients = [5., 3., 2., 4., ]
        grouped_pairs = sorted_insertion(pauli_indices, pauli_coefficients, True)
        self.assertEqual(grouped_pairs[0], [((0, 0), 5.0), ((3, 3), 4.0), ((3, 0), 2.0)])
        self.assertEqual(grouped_pairs[1], [((1, 0), 3.0)])

    def test_sorted_insertion_error(self):
        self.assertRaises(ValueError, lambda: sorted_insertion([], [5.]))

    def test_single_indices_commute(self):
        self.assertTrue(single_indices_commute(3, 3))
        self.assertTrue(single_indices_commute(2, 0))
        self.assertFalse(single_indices_commute(3, 1))

    def test_tuple_indices_commute(self):
        self.assertEqual(tuple_indices_commute((0, 1, 2), (1, 1, 3)), [True, True, False])
