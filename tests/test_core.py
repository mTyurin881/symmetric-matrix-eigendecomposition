import math
import unittest

from symmetric_matrix_eigendecomposition import (
    jacobi_eigendecomposition,
    jacobi_eigenvalues,
)


class TestJacobiEigendecomposition(unittest.TestCase):
    def _reconstruct(self, matrix, eigenvalues, eigenvectors):
        n = len(matrix)
        for col in range(n):
            for row in range(n):
                acc = 0.0
                for k in range(n):
                    acc += matrix[row][k] * eigenvectors[k][col]
                expected = eigenvalues[col] * eigenvectors[row][col]
                self.assertAlmostEqual(acc, expected, places=9)

    def test_diagonal_matrix(self):
        # Already diagonal: should converge immediately and return identity eigenvectors.
        m = [[3.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 2.0]]
        vals, vecs = jacobi_eigendecomposition(m)
        self.assertEqual(len(vals), 3)
        self.assertEqual(len(vecs), 3)
        # Descending magnitude: 3, 2, -1
        self.assertAlmostEqual(vals[0], 3.0, places=10)
        self.assertAlmostEqual(vals[1], 2.0, places=10)
        self.assertAlmostEqual(vals[2], -1.0, places=10)
        # Eigenvectors form identity permutation
        for i in range(3):
            for j in range(3):
                if (i, j) in {(0, 0), (1, 2), (2, 1)}:
                    self.assertAlmostEqual(vecs[i][j], 1.0, places=10)
                else:
                    self.assertAlmostEqual(vecs[i][j], 0.0, places=10)

    def test_two_by_two_symmetric(self):
        # Known closed-form eigendecomposition.
        m = [[2.0, 1.0], [1.0, 2.0]]
        vals, vecs = jacobi_eigendecomposition(m)
        self.assertAlmostEqual(vals[0], 3.0, places=10)
        self.assertAlmostEqual(vals[1], 1.0, places=10)
        self._reconstruct(m, vals, vecs)

    def test_three_by_three_standard(self):
        m = [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]]
        vals, vecs = jacobi_eigendecomposition(m)
        self.assertEqual(len(vals), 3)
        self._reconstruct(m, vals, vecs)
        # Verify descending magnitude
        self.assertGreaterEqual(abs(vals[0]), abs(vals[1]))
        self.assertGreaterEqual(abs(vals[1]), abs(vals[2]))

    def test_identity_matrix(self):
        m = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        vals, vecs = jacobi_eigendecomposition(m)
        for v in vals:
            self.assertAlmostEqual(v, 1.0, places=10)
        self._reconstruct(m, vals, vecs)

    def test_zero_matrix(self):
        m = [[0.0, 0.0], [0.0, 0.0]]
        vals, vecs = jacobi_eigendecomposition(m)
        self.assertAlmostEqual(vals[0], 0.0, places=10)
        self.assertAlmostEqual(vals[1], 0.0, places=10)
        self._reconstruct(m, vals, vecs)

    def test_negative_eigenvalues(self):
        m = [[-2.0, 0.0, 0.0], [0.0, -5.0, 0.0], [0.0, 0.0, -1.0]]
        vals, _ = jacobi_eigendecomposition(m)
        # Descending magnitude: -5, -2, -1
        self.assertAlmostEqual(vals[0], -5.0, places=10)
        self.assertAlmostEqual(vals[1], -2.0, places=10)
        self.assertAlmostEqual(vals[2], -1.0, places=10)

    def test_eigenvalues_only(self):
        m = [[2.0, 1.0], [1.0, 2.0]]
        vals = jacobi_eigenvalues(m)
        self.assertEqual(len(vals), 2)
        self.assertAlmostEqual(vals[0], 3.0, places=10)
        self.assertAlmostEqual(vals[1], 1.0, places=10)

    def test_tuple_input_supported(self):
        # Input type is documented as list or tuple of lists/tuples.
        m = ((2.0, 1.0), (1.0, 2.0))
        vals, vecs = jacobi_eigendecomposition(m)
        self.assertAlmostEqual(vals[0], 3.0, places=10)
        self.assertAlmostEqual(vals[1], 1.0, places=10)
        self._reconstruct([[2.0, 1.0], [1.0, 2.0]], vals, vecs)

    def test_raises_on_non_square(self):
        m = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        with self.assertRaises(ValueError):
            jacobi_eigendecomposition(m)

    def test_raises_on_empty(self):
        with self.assertRaises(ValueError):
            jacobi_eigendecomposition([])

    def test_raises_on_asymmetric(self):
        m = [[1.0, 2.0], [3.0, 4.0]]
        with self.assertRaises(ValueError):
            jacobi_eigendecomposition(m)

    def test_raises_on_non_list_input(self):
        with self.assertRaises(TypeError):
            jacobi_eigendecomposition("not a matrix")

    def test_raises_on_bad_row(self):
        m = [[1.0, 2.0], "not a row"]
        with self.assertRaises(TypeError):
            jacobi_eigendecomposition(m)

    def test_raises_on_negative_tol(self):
        m = [[1.0, 0.0], [0.0, 1.0]]
        with self.assertRaises(ValueError):
            jacobi_eigendecomposition(m, tol=-1e-9)

    def test_raises_on_zero_max_sweeps(self):
        m = [[1.0, 0.0], [0.0, 1.0]]
        with self.assertRaises(ValueError):
            jacobi_eigendecomposition(m, max_sweeps=0)

    def test_repeated_eigenvalues(self):
        # A matrix with a repeated eigenvalue; eigenvectors are still valid.
        m = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]
        vals, vecs = jacobi_eigendecomposition(m)
        for v in vals:
            self.assertAlmostEqual(v, 2.0, places=10)
        self._reconstruct(m, vals, vecs)

    def test_larger_matrix_4x4(self):
        m = [
            [5.0, 1.0, 0.0, 0.0],
            [1.0, 4.0, 1.0, 0.0],
            [0.0, 1.0, 3.0, 1.0],
            [0.0, 0.0, 1.0, 2.0],
        ]
        vals, vecs = jacobi_eigendecomposition(m)
        self.assertEqual(len(vals), 4)
        self.assertGreaterEqual(abs(vals[0]), abs(vals[1]))
        self.assertGreaterEqual(abs(vals[1]), abs(vals[2]))
        self.assertGreaterEqual(abs(vals[2]), abs(vals[3]))
        self._reconstruct(m, vals, vecs)

    def test_trace_and_determinant_invariants(self):
        # Trace and determinant are preserved under orthogonal transformations.
        m = [[4.0, 1.0, 2.0], [1.0, 3.0, 0.5], [2.0, 0.5, 6.0]]
        vals, _ = jacobi_eigendecomposition(m)
        trace = sum(m[i][i] for i in range(3))
        det = 4.0 * (3.0 * 6.0 - 0.5 * 0.5) - 1.0 * (1.0 * 6.0 - 0.5 * 2.0) + 2.0 * (1.0 * 0.5 - 3.0 * 2.0)
        self.assertAlmostEqual(sum(vals), trace, places=9)
        prod = 1.0
        for v in vals:
            prod *= v
        self.assertAlmostEqual(prod, det, places=9)

    def test_floating_point_input_precision(self):
        # Inputs with floating point noise should still diagonalize cleanly.
        m = [[1.0 + 1e-15, 2.0 - 1e-15], [2.0 - 1e-15, 1.0 + 1e-15]]
        vals, vecs = jacobi_eigendecomposition(m)
        self._reconstruct(m, vals, vecs)
        # Approx 3 and -1
        self.assertAlmostEqual(vals[0], 3.0, places=6)
        self.assertAlmostEqual(vals[1], -1.0, places=6)


if __name__ == "__main__":
    unittest.main()
