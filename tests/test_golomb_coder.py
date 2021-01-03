import unittest
import numpy as np
import numpy.testing as npt

from vector_quantization.golomb_coder import golomb_coder, golomb_decoder


class TestGolombCoder(unittest.TestCase):
    def test_encoding_decoding(self):
        values = np.array([[1, 2, 1], [3, 9, 2]])
        first_element, binary_code, m_values = golomb_coder(values)
        values_decoded = golomb_decoder(
            first_value=first_element, binary_code=binary_code, m_values=m_values, size=values.shape
        )
        npt.assert_almost_equal(values_decoded, values)

    def test_encoding_decoding_v_g_gt_l(self):
        values = np.array([[1, 130, 1, 3], [3, 9, 2, 3]])
        first_element, binary_code, m_values = golomb_coder(values)
        values_decoded = golomb_decoder(
            first_value=first_element, binary_code=binary_code, m_values=m_values, size=values.shape
        )
        npt.assert_almost_equal(values_decoded, values)

    def test_encoding_decoding_2(self):
        values = np.array([[1, 30, 1, 3], [3, 9, 2, 3]])
        first_element, binary_code, m_values = golomb_coder(values)
        values_decoded = golomb_decoder(
            first_value=first_element, binary_code=binary_code, m_values=m_values, size=values.shape
        )
        npt.assert_almost_equal(values_decoded, values)

    def test_random_encoding_decoding(self):
        """
        Encode random (noise) images with size 10x10
        """
        for i in range(10):
            values = np.random.rand(20, 20) * 255
            values = values.astype(int)
            first_element, binary_code, m_values = golomb_coder(values)
            values_decoded = golomb_decoder(
                first_value=first_element, binary_code=binary_code, m_values=m_values, size=values.shape
            )
            npt.assert_almost_equal(values_decoded, values)
