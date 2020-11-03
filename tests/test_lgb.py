import unittest
import numpy as np
import numpy.testing as npt

from vector_quantization.lbg import lbg


class TestLBG(unittest.TestCase):
    def test_lbg_finish_on_error(self):
        vectors = np.array([[1, 1], [2, 2], [1, 1], [3, 3], [4, 4], [1, 1]])
        initial_codebook = np.array([[1, 1], [2, 2]])
        codebook, distortions = lbg(vectors, initial_codebook, iterations=50, error=0.5)
        npt.assert_almost_equal(codebook, np.array([[1, 1], [3, 3]]))
        self.assertEqual(len(distortions), 3)

    def test_lbg_finish_on_iterations(self):
        vectors = np.array([[1, 1], [2, 2], [1, 1], [3, 3], [4, 4], [1, 1]])
        initial_codebook = np.array([[1, 1], [2, 2]])
        codebook, distortions = lbg(vectors, initial_codebook, iterations=50, error=0.0)
        npt.assert_almost_equal(codebook, np.array([[1, 1], [3, 3]]))
        self.assertEqual(len(distortions), 50)
