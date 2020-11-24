import unittest
import numpy as np
import numpy.testing as npt

from vector_quantization.mean_removal_quantization import mean_removal_quantize_from_codebook


class TestMeanRemovalQuantizeFromCodebook(unittest.TestCase):
    def test_quantize(self):
        from vector_quantization.codebooks import random_codebook
        from vector_quantization.vectorize import vectorize, image_from_vectors

        codebook = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [5, 5, 5, 5]])
        image = np.array([[1, 1, 3, 3], [1, 1, 3, 3], [4, 4, 0, 0], [4, 4, 0, 0]])
        vectors = vectorize(image, 2)
        means = np.mean(vectors, axis=1, keepdims=True)
        residual_vectors = vectors - means
        quantized_vectors, _ = mean_removal_quantize_from_codebook(residual_vectors, means, codebook)
        quantized_image = image_from_vectors(quantized_vectors, image)
        expected_quantized_image = np.array([[2, 2, 4, 4], [2, 2, 4, 4], [5, 5, 1, 1], [5, 5, 1, 1]])
        npt.assert_almost_equal(quantized_image, expected_quantized_image)
