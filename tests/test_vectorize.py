import unittest
import numpy as np
import numpy.testing as npt

from vector_quantization.vectorize import vectorize


class TestVectorize(unittest.TestCase):
    def test_window_size_2(self):
        """
        Expected outcome as presented here https://i.imgur.com/wycSwR3.png
        """
        img = np.array([[1, 2, 5, 4], [3, 4, 3, 2], [4, 2, 4, 1], [3, 5, 5, 2]])
        expected_vectors = np.array([[1, 2, 3, 4], [5, 4, 3, 2], [4, 2, 3, 5], [4, 1, 5, 2]])

        npt.assert_almost_equal(vectorize(img, 2), expected_vectors)

    def test_window_size_3(self):
        img = np.arange(1, 19)
        img = img.reshape(3, -1)
        expected_vectors = np.array([[1, 2, 3, 7, 8, 9, 13, 14, 15], [4, 5, 6, 10, 11, 12, 16, 17, 18]])

        npt.assert_almost_equal(vectorize(img, 3), expected_vectors)

    def test_incompatible_window_size(self):
        img = np.arange(1, 19)
        img = img.reshape(3, -1)
        self.assertRaises(ValueError, vectorize, img, 2)


if __name__ == "__main__":
    unittest.main()
