import numpy as np
import timeit
from timebudget import timebudget

from vector_quantization.vectorize import vectorize, image_from_vectors


@timebudget
def quantize(image, window_size, codebook_fun, codebook_size, verbose=False):
    quantized_img = image.copy()
    vectors = vectorize(quantized_img, window_size=window_size)
    codebook = codebook_fun(vectors, length=codebook_size)

    quantized_vectors = quantize_from_codebook(vectors, codebook)

    quantized_img = image_from_vectors(quantized_vectors, quantized_img)
    return quantized_img


@timebudget
def quantize_from_codebook(vectors, codebook):
    quantized_vectors = np.zeros_like(vectors)
    for idx, vector in enumerate(vectors):
        quantized_vectors[idx, :] = find_closest(vector, codebook)
    return quantized_vectors


def find_closest(vector, codebook):
    # TODO: optimize
    # Probably it will be faster to calculate distances for all vectors
    dists = np.sum(np.sqrt((codebook - vector) ** 2), axis=1)
    closest_id = np.argmin(dists)
    closest = codebook[closest_id]
    return closest


if __name__ == "__main__":
    from PIL import Image
    from metrics import PSNR
    from image import load_image, save_image
    from codebooks import random_codebook

    img = load_image("balloon.bmp")
    quantized_img = quantize(img, window_size=4, codebook_fun=random_codebook, codebook_size=32)
    Image.fromarray(quantized_img).show()
