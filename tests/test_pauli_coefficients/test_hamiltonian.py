from unittest import TestCase

import numpy as np
from numpy.linalg import eig
from scipy.linalg import ishermitian
from scipy.sparse import csr_matrix
from sympy import Matrix

from operators.hamiltonian import get_harmonic_hamiltonian, get_extended_hamiltonian, multi_tensor_product, \
    sparse_multi_tensor_product, binary_to_gray


class TestHamiltonian(TestCase):
    def test_get_harmonic_hamiltonian(self):
        bin_ham = get_harmonic_hamiltonian(3, 6, np.ones((3, 3)), "bin", True)
        self.assertEqual(bin_ham.shape, (2 ** 6, 2 ** 6))
        self.assertTrue(ishermitian(bin_ham))

        gray_ham = get_harmonic_hamiltonian(2, 6, np.ones((2, 2)), "gray", True)
        self.assertEqual(gray_ham.shape, (2 ** 6, 2 ** 6))
        self.assertTrue(ishermitian(gray_ham))

        ham = get_harmonic_hamiltonian(3, 6, np.zeros((3, 3)), "bin", True)
        eigvals, _ = eig(ham)
        np.testing.assert_array_equal(np.unique(eigvals), np.arange(3, 3 * (2 * 3 + 1) + 1, 2))

    def test_get_extended_hamiltonian(self):
        extended_ham = get_extended_hamiltonian(3, 6, np.ones((3, 3,)), 0.1, 0.5, 0.1, "bin", True)
        harmonic_ham = get_harmonic_hamiltonian(3, 6, np.ones((3, 3,)), "bin", True)

        self.assertEqual(extended_ham.shape, (2 ** 6, 2 ** 6))

        extended_eigvals, _ = eig(extended_ham)
        harmonic_eigvals, _ = eig(harmonic_ham)
        self.assertFalse(np.allclose(extended_eigvals, harmonic_eigvals))

    def test_multi_tensor_product(self):
        mat_1 = Matrix([[1, 0], [0, 1]])
        mat_2 = Matrix([[0, 1], [1, 0]])
        prod = multi_tensor_product([mat_1, mat_2])
        expected_prod = Matrix([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
        self.assertEqual(prod, expected_prod)
        multi_tensor_product([mat_1, ] * 5)

    def test_sparse_multi_tensor_product(self):
        mat_1 = csr_matrix([[1, 0], [0, -1]])
        mat_2 = csr_matrix([[0, -1.j], [1.j, 0]])
        prod = sparse_multi_tensor_product([mat_1, mat_2])
        expected_prod = csr_matrix([[0., -1.j, 0., 0.], [1.j, 0., 0., 0., ], [0., 0., 0., 1.j], [0., 0., -1.j, 0.]])
        self.assertTrue((prod == expected_prod).toarray().all())
        sparse_multi_tensor_product([mat_1, ] * 5)

    def test_binary_to_gray(self):
        mat = np.array([[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]])
        gray_mat = np.array([[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 4, 0], [0, 0, 0, 3]])

        np.testing.assert_array_equal(binary_to_gray(mat), gray_mat)
