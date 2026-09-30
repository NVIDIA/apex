import itertools
import unittest

import numpy as np

from apex.contrib.sparsity.permutation_search_kernels.permutation_utilities import (
    unstructured_prune,
)


class TestUnstructuredPrune(unittest.TestCase):
    def test_signed_weights_keep_largest_magnitudes(self):
        for dtype in (np.float16, np.float32, np.float64):
            with self.subTest(dtype=dtype):
                matrix = np.array([[-100.0, -1.0, 0.25, 50.0]], dtype=dtype)
                original = matrix.copy()
                expected = np.array([[-100.0, 0.0, 0.0, 50.0]], dtype=dtype)
                actual = unstructured_prune(matrix, 0.5)
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(matrix, original)
                self.assertEqual(actual.dtype, dtype)

    def test_retained_magnitude_matches_exhaustive_reference(self):
        matrix = np.array([[-8.0, 1.0, -6.0, 3.0], [2.0, -7.0, 4.0, -5.0]])
        for sparsity in (0.25, 0.5, 0.75):
            with self.subTest(sparsity=sparsity):
                result = unstructured_prune(matrix, sparsity)
                kept = matrix.size - int(matrix.size * sparsity)
                maximum = max(
                    sum(abs(matrix.flat[i]) for i in indices)
                    for indices in itertools.combinations(range(matrix.size), kept)
                )
                self.assertEqual(np.count_nonzero(result), kept)
                self.assertEqual(np.abs(result).sum(), maximum)
                np.testing.assert_array_equal(result[result != 0], matrix[result != 0])

    def test_pruning_is_invariant_to_weight_signs(self):
        magnitude = np.arange(1.0, 25.0).reshape(4, 6).T
        signs = np.where(np.arange(24).reshape(6, 4) % 3 == 0, -1, 1)
        for sparsity in (0.0, 0.25, 0.5, 1.0):
            with self.subTest(sparsity=sparsity):
                positive = unstructured_prune(magnitude, sparsity)
                signed = unstructured_prune(magnitude * signs, sparsity)
                np.testing.assert_array_equal(signed, positive * signs)
                self.assertEqual(signed.shape, magnitude.shape)

    def test_existing_zeros_and_nonnegative_inputs(self):
        for matrix in (np.array([[0.0, -8.0, 3.0, 0.0]]), np.arange(1.0, 9.0).reshape(2, 4)):
            for sparsity in (0.0, 0.5, 1.0):
                with self.subTest(matrix=matrix.tolist(), sparsity=sparsity):
                    original = matrix.copy()
                    result = unstructured_prune(matrix, sparsity)
                    expected = np.sort(np.abs(matrix), axis=None)[
                        int(matrix.size * sparsity) :
                    ].sum()
                    self.assertEqual(np.abs(result).sum(), expected)
                    np.testing.assert_array_equal(matrix, original)


if __name__ == "__main__":
    unittest.main()
