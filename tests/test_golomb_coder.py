import unittest
import numpy as np
import numpy.testing as npt

from vector_quantization.golomb_coder import golomb_coder, golomb_decoder


class TestGolombCoder(unittest.TestCase):
    def test_encoding_decoding(self):
        values = [0, 10, 200, 243, 18, 19, 35, 82, 49, 0, 0, 0, 1, 99, 99]
        binary_code, m_values = golomb_coder(values)
        values_decoded = golomb_decoder(binary_code, m_values)
        npt.assert_almost_equal(values_decoded, values)
