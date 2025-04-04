import unittest
import numpy as np

import os
import sys
# Get the directory of the current file (tests/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Compute the project root (one level above tests/)
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src import utils


class TestQuantumFunctions(unittest.TestCase):

    def test_truncate_svd(self) -> None:
        # Create a simple 3x3 matrix with known singular values.
        matrix = np.array([[3, 0, 0],
                           [0, 2, 0],
                           [0, 0, 1]], dtype=float)
        # Keeping only the largest singular value should zero-out the smaller ones.
        truncated = utils.truncate_svd(matrix, singular_values_kept=1)
        # Recompute SVD of truncated matrix
        U, S, V = np.linalg.svd(truncated, full_matrices=False)
        # Only the first singular value should be non-zero (up to numerical precision)
        self.assertAlmostEqual(S[0], 3, places=5)
        self.assertTrue(np.allclose(S[1:], 0, atol=1e-5))

    def test_vectorize_density_matrix_flatten(self) -> None:
        # Use a simple single-qubit density matrix |0><0|
        rho = np.array([[1, 1],
                        [2, 0]], dtype=complex)
        vec_flat = utils.vectorize_density_matrix(rho, basis='flatten')
        self.assertTrue(np.array_equal(vec_flat, rho.flatten()))

    def test_vectorize_density_matrix_paulis(self) -> None:
        # Test vectorization in the Pauli basis for a single-qubit density matrix |0><0|
        rho = np.array([[1, 0],
                        [0, 0]], dtype=complex)
        vec_paulis = utils.vectorize_density_matrix(rho, basis='paulis')
        # For |0><0| and Pauli basis [I, X, Y, Z] the expected traces are:
        # trace(|0><0| I) = 1, trace(|0><0| X) = 0, trace(|0><0| Y) = 0, trace(|0><0| Z) = 1.
        expected = np.array([1, 0, 0, 1], dtype=complex)
        self.assertTrue(np.allclose(vec_paulis, expected, atol=1e-5))

    def test_devectorize_density_matrix_flatten(self) -> None:
        # Create a 2x2 density matrix, vectorize with flatten, then devectorize and compare
        rho = np.array([[0.7, 0.3],
                        [0.3, 0.3]], dtype=complex)
        vec = rho.flatten()
        rho_reconstructed = utils.devectorize_density_matrix(vec, basis='flatten')
        self.assertTrue(np.allclose(rho, rho_reconstructed, atol=1e-5))

    def test_ket_to_dm(self) -> None:
        # Test conversion from ket to density matrix
        ket = np.array([1, 1j], dtype=complex)
        dm = utils.ket_to_dm(ket)
        expected = np.array([[1, -1j],
                             [1j, 1]], dtype=complex)
        self.assertTrue(np.allclose(dm, expected, atol=1e-5))

    def test_kets_to_vectorized_states_matrix(self) -> None:
        # Test with two single-qubit states: |0> and |1>
        ket0 = np.array([1, 0], dtype=complex)
        ket1 = np.array([0, 1], dtype=complex)
        # For Pauli basis vectorization, each state produces a vector of length 4.
        vec_states_flatten = utils.kets_to_vectorized_states_matrix([ket0, ket1], basis='flatten')
        vec_states_paulis = utils.kets_to_vectorized_states_matrix([ket0, ket1], basis='paulis')
        # Check the shape of the resulting matrices
        self.assertEqual(vec_states_flatten.shape, (4, 2))
        self.assertTrue(np.allclose(vec_states_flatten, [[1, 0], [0, 0], [0, 0], [0, 1]], atol=1e-5))
        self.assertTrue(np.allclose(vec_states_paulis, [[1, 1], [0, 0], [0, 0], [1, -1]], atol=1e-5))
        

if __name__ == "__main__":
    unittest.main()