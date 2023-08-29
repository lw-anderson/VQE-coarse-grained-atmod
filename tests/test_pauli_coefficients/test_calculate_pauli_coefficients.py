from unittest import TestCase

import numpy as np

from operators.calculate_pauli_coefficients import calculate_pauli_coefficient_non_interaction, \
    calculate_pauli_coefficient_coupled_pair, calculate_coefficient, calculate_non_zero_terms


class TestCalculatePauliCoefficients(TestCase):
    def test_calculate_pauli_coefficient_non_interaction(self):
        indices, coeffs = calculate_pauli_coefficient_non_interaction(2, "bin")
        np.testing.assert_array_equal(indices, [(0, 0), (0, 3), (3, 0)])
        np.testing.assert_array_equal(coeffs, [4., -2., -1.])

    def test_calculate_pauli_coefficient_coupled_pair(self):
        indices_1, coeffs_1 = calculate_pauli_coefficient_coupled_pair(2, "bin")
        self.assertEqual(indices_1, [(1, 1)])
        self.assertEqual(coeffs_1, [0.5])

        indices_2, coeffs_2 = calculate_pauli_coefficient_coupled_pair(4, encoding="bin")
        # see arXiv:2110.00968 for pauli decomp of four qubit two oscillators
        expected_indices = [(1, 0, 1, 0), (1, 1, 1, 0), (1, 0, 1, 1), (1, 1, 1, 1), (2, 2, 1, 0), (2, 2, 1, 1),
                            (1, 0, 2, 2), (1, 1, 2, 2), (2, 2, 1, 3), (1, 3, 2, 2), (1, 3, 1, 0), (1, 3, 1, 1),
                            (1, 0, 1, 3), (1, 1, 1, 3), (1, 3, 1, 3), (2, 2, 2, 2)]
        np.testing.assert_array_equal(sorted(expected_indices), sorted(indices_2))

    def test_calculate_coefficient(self):
        zz = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.assertEqual(calculate_coefficient(zz, [3, 3]), 1.)
        self.assertEqual(calculate_coefficient(zz, [1, 0]), 0.)

    def test_calculate_non_zero_terms(self):
        zz = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        non_zero_indices, non_zero_coefficients = calculate_non_zero_terms(zz, 2)
        expected_non_zero_indices = [(3, 3)]
        expected_non_zero_coefficients = [1.]
        self.assertEqual(non_zero_indices, expected_non_zero_indices)
        self.assertEqual(non_zero_coefficients, expected_non_zero_coefficients)

        zz_plus_ii = np.array([[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2]])
        non_zero_indices, non_zero_coefficients = calculate_non_zero_terms(zz_plus_ii, 2)
        expected_non_zero_indices = [(0, 0), (3, 3)]
        expected_non_zero_coefficients = [1., 1.]
        self.assertEqual(non_zero_indices, expected_non_zero_indices)
        self.assertEqual(non_zero_coefficients, expected_non_zero_coefficients)
