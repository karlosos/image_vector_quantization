import unittest
import numpy as np
import numpy.testing as npt

from vector_quantization.differential_encoding import differential_encoding
from vector_quantization.differential_encoding import differential_decoding


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
