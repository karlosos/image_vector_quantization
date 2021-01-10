import unittest
import numpy as np
import numpy.testing as npt

from vector_quantization.compression import codebook_compression, codebook_decompression


class TestCompression(unittest.TestCase):
    def test_codebook_compression_decompression(self):
        codebook = np.array([[1, 1, 1, 1], [3, 3, 3, 3], [4, 4, 4, 4]])
        codes = np.array([0, 1, 2, 0, 1, 1])
        image_shape = (4, 6)
        window_size = 2

        bit_code = codebook_compression(image_shape, window_size, codes, codebook)

        image_shape_o, window_size_o, codes_o, codebook_o, ptr = codebook_decompression(bit_code)

        npt.assert_almost_equal(image_shape, image_shape_o)
        npt.assert_almost_equal(window_size, window_size_o)
        npt.assert_almost_equal(codes, codes_o)
        npt.assert_almost_equal(codebook_o, codebook)


if __name__ == "__main__":
    unittest.main()
