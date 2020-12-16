import random
import numpy as np


def random_codebook(vectors, length=512, seed=None):
    if seed is not None:
        random.seed(seed)
    codebook = random.sample(np.unique(vectors, axis=0).tolist(), length)
    # Following line is 2 times faster but creates codebook with possible duplicates
    # codebook = random.sample(vectors.tolist(), length)
    return np.array(codebook)


def pnn(vectors, length=512):
    """
    References:

    https://sci-hub.st/https://doi.org/10.1109/29.35395
    http://cs.uef.fi/sipu/pub/LazyPNN.pdf

    TODO: implementation of PNN

    The PNN starts by constructing an initial codebook in which each training vector is considered as its own code vector.
    Two nearest code vectors are merged at each step of the algorithm and the process is repeated until the desired size of the codebook has been reached.
    The algorithm is straightforward to implement in its basic form.

    Should return codebook.
    """
    return random_codebook(vectors, length)  # TODO: this is placeholder, remove it


if __name__ == "__main__":
    """
    This script will perform LBG with initial codebooks from random initialization and from pnn initialization.
    PNN initialization should get better results.
    """
    from PIL import Image

    from image import load_image
    from lbg import lbg
    from vectorize import vectorize, image_from_vectors
    from quantization import quantize_from_codebook
    from vector_quantization.metrics import PSNR

    img = load_image("balloon.bmp")
    vectors = vectorize(img, window_size=4)

    # Random codebook initialization
    initial_codebook = random_codebook(vectors, length=32)
    codebook, distortions = lbg(vectors, initial_codebook, 50, 0.01)
    img_random = image_from_vectors(quantize_from_codebook(vectors, codebook), img.copy())
    Image.fromarray(img_random).show()
    random_psnr = PSNR(img, img_random)

    # PNN
    initial_codebook = pnn(vectors, length=32)
    codebook, distortions = lbg(vectors, initial_codebook, 50, 0.01)
    img_pnn = image_from_vectors(quantize_from_codebook(vectors, codebook), img.copy())
    Image.fromarray(img_pnn).show()
    pnn_psnr = PSNR(img, img_pnn)

    # TODO: PNN PSNR should be better than random
    print("Random PSNR:", random_psnr)
    print("PNN PSNR:", pnn_psnr)
