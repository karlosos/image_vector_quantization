import numpy as np
from scipy.spatial import distance
from scipy.cluster.vq import vq


def lbg(vectors, initial_codebook, iterations, error):
    m = 1
    distortions = [np.inf]
    finished = False
    code_book = initial_codebook
    cb_length = code_book.shape[0]

    while not finished and m < iterations:
        # Assign each vector to cluster
        codes, dists = vq(vectors, code_book)
        # Calculate mean distortion
        distortions.append(np.mean(dists))
        # Find new centroids
        for i in range(cb_length):
            code_book[i] = np.mean(vectors[codes == i])
        # Check condition if gain of distortion is small
        print(distortions[-1])
        if (distortions[m - 1] - distortions[m]) / distortions[m] < error:
            finished = True
        m += 1

    return code_book, distortions


if __name__ == "__main__":
    vectors = np.array([[1, 1], [2, 2], [1, 1], [3, 3], [4, 4], [1, 1]])
    initial_codebook = np.array([[1, 1], [2, 2]])
    codebook, distortion = lbg(vectors, initial_codebook, iterations=50, error=0.0)
    print(codebook)
