import unittest
import numpy as np
import numpy.testing as npt

from vector_quantization.differential_encoding import differential_encoding
from vector_quantization.differential_encoding import differential_decoding
from vector_quantization.differential_encoding import differential_median_encoding
from vector_quantization.differential_encoding import differential_median_decoding
from vector_quantization.differential_encoding import median_adaptive_predictor_encoding
from vector_quantization.differential_encoding import median_adaptive_predictor_decoding


class TestDifferentialEncoding(unittest.TestCase):
    def test_encoding_image(self):
        image = np.array([[1, 2, 3, 5, 4], [2, 3, 4, 3, 2], [8, 9, 0, 2, 1]])
        expected_output = np.array(
            [[1.0, 1.0, 1.0, 2.0, -1.0], [1.0, 1.0, 1.0, -1.0, -1.0], [6.0, 1.0, -9.0, 2.0, -1.0]]
        )
        output = differential_encoding(image)
        npt.assert_almost_equal(output, expected_output)

    def test_decoding_image(self):
        coded = np.array([[1.0, 1.0, 1.0, 2.0, -1.0], [1.0, 1.0, 1.0, -1.0, -1.0], [6.0, 1.0, -9.0, 2.0, -1.0]])
        image_expected = np.array([[1, 2, 3, 5, 4], [2, 3, 4, 3, 2], [8, 9, 0, 2, 1]])
        image_decoded = differential_decoding(coded)
        npt.assert_almost_equal(image_decoded, image_expected)


class TestDifferentialMedianEncoding(unittest.TestCase):
    def test_encoding_image(self):
        image = np.array([[1, 2, 3, 5, 4], [2, 3, 4, 3, 2], [8, 9, 0, 2, 1]])
        expected_output = np.array(
            [
                [1.0, 1.0, 1.0, 2.0, -1.0],
                [1.0, 3 - 5 / 3, 4 - 8 / 3, 3 - 4, 2 - 4],
                [6.0, 9 - 13 / 3, 0 - 16 / 3, 2 - 7 / 3, 1 - 7 / 3],
            ]
        )
        output = differential_median_encoding(image)
        npt.assert_almost_equal(output, expected_output)

    def test_decoding_image(self):
        coded = np.array(
            [
                [1.0, 1.0, 1.0, 2.0, -1.0],
                [1.0, 3 - 5 / 3, 4 - 8 / 3, 3 - 4, 2 - 4],
                [6.0, 9 - 13 / 3, 0 - 16 / 3, 2 - 7 / 3, 1 - 7 / 3],
            ]
        )
        image_expected = np.array([[1, 2, 3, 5, 4], [2, 3, 4, 3, 2], [8, 9, 0, 2, 1]])
        image_decoded = differential_median_decoding(coded)
        npt.assert_almost_equal(image_decoded, image_expected)


class TestMedianAdaptivePredictor(unittest.TestCase):
    def test_encoding_image(self):
        image = np.array([[1, 2, 3, 5, 4], [2, 3, 4, 3, 2], [8, 9, 0, 2, 1]])
        expected_output = np.array(
            [[1.0, 1.0, 1.0, 2.0, -1.0], [1.0, 1.0, 1.0, -2.0, -1.0], [6.0, 1.0, -9.0, 2.0, -1.0]]
        )
        output = median_adaptive_predictor_encoding(image)
        npt.assert_almost_equal(output, expected_output)

    def test_decoding_image(self):
        coded = np.array([[1.0, 1.0, 1.0, 2.0, -1.0], [1.0, 1.0, 1.0, -2.0, -1.0], [6.0, 1.0, -9.0, 2.0, -1.0]])
        image_expected = np.array([[1, 2, 3, 5, 4], [2, 3, 4, 3, 2], [8, 9, 0, 2, 1]])
        image_decoded = median_adaptive_predictor_decoding(coded)
        npt.assert_almost_equal(image_decoded, image_expected)
