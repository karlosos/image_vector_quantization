import unittest
import numpy as np
import numpy.testing as npt

from vector_quantization.quantization import quantize_from_codebook


class TestQuantizeFromCodebook(unittest.TestCase):
    def test_quantize(self):
        from vector_quantization.codebooks import random_codebook
        from vector_quantization.vectorize import vectorize, image_from_vectors

        codebook = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [5, 5, 5, 5]])
        image = np.array([[1, 1, 3, 3], [1, 1, 3, 3], [4, 4, 0, 0], [4, 4, 0, 0]])
        vectors = vectorize(image, 2)
        quantized_vectors = quantize_from_codebook(vectors, codebook)
        quantized_image = image_from_vectors(quantized_vectors, image)
        expected_quantized_image = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [5, 5, 1, 1], [5, 5, 1, 1]])
        npt.assert_almost_equal(quantized_image, expected_quantized_image)
