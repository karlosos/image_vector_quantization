import unittest
import numpy as np
import numpy.testing as npt

from vector_quantization.golomb_coder import golomb_coder, golomb_decoder
from vector_quantization.golomb_coder import golomb_compress, golomb_decompress


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

    def test_compression_decompression_with_diff_enc(self):
        """
        Integration with differential encoding
        """
        from vector_quantization.vectorize import vectorize
        from vector_quantization.differential_encoding import (
            median_adaptive_predictor_encoding,
            median_adaptive_predictor_decoding,
        )

        # Generate random image
        img = np.random.rand(24, 24) * 255

        # Performing quantization with removed means (MRVQ)
        window_size = 4
        vectors = vectorize(img, window_size=window_size)
        means = np.mean(vectors, axis=1, keepdims=True)  # mean should be in shape like smaller image.
        height, width = img.shape
        means_reshaped = means.reshape((height // window_size, width // window_size))
        means_reshaped = means_reshaped.astype(int)

        # Differential encoding means from MRVQ
        encoded_means = median_adaptive_predictor_encoding(means_reshaped)
        encoded_means = encoded_means.astype(int)

        # Compress encoded means with Golomb Coder
        bit_code = golomb_compress(encoded_means)

        # Decompress encoded means with Golomb Coder
        decoded_means_diff = golomb_decompress(bit_code)

        # Differential decoding
        decoded_means = median_adaptive_predictor_decoding(decoded_means_diff)
        npt.assert_almost_equal(means_reshaped, decoded_means)
