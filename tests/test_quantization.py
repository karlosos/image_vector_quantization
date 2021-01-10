import unittest
import numpy as np
import numpy.testing as npt

from vector_quantization.quantization import quantize_from_codebook
from vector_quantization.quantization import codes_from_vectors
from vector_quantization.quantization import vectors_from_codes


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

    def test_codes_from_vectors(self):
        from vector_quantization.vectorize import vectorize

        codebook = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [5, 5, 5, 5]])
        image = np.array([[1, 1, 3, 3], [1, 1, 3, 3], [4, 4, 0, 0], [4, 4, 0, 0]])
        vectors = vectorize(image, 2)
        codes = codes_from_vectors(vectors, codebook)
        expected_codes = np.array([0, 1, 2, 0])
        npt.assert_almost_equal(expected_codes, codes)

    def test_vectors_from_codes(self):
        from vector_quantization.vectorize import vectorize, image_from_vectors

        codes = np.array([0, 1, 2, 0])
        codebook = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [5, 5, 5, 5]])

        expected_quantized_image = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [5, 5, 1, 1], [5, 5, 1, 1]])
        expected_vectors = vectorize(expected_quantized_image, 2)

        vectors = vectors_from_codes(codes, codebook)
        npt.assert_almost_equal(expected_vectors, vectors)

        quantized_image = image_from_vectors(vectors, expected_quantized_image)
        expected_quantized_image = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [5, 5, 1, 1], [5, 5, 1, 1]])
        npt.assert_almost_equal(quantized_image, expected_quantized_image)
