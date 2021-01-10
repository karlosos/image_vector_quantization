"""
End to end compression script
"""
import numpy as np
from PIL import Image

from vector_quantization.image import load_image, save_image
from vector_quantization.metrics import PSNR
from vector_quantization.codebooks import random_codebook
from vector_quantization.vectorize import vectorize, image_from_vectors
from vector_quantization.quantization import quantize, quantize_from_codebook, vectors_from_codes
from vector_quantization.lbg import lbg
from vector_quantization.mean_removal_quantization import mean_removal_quantize_from_codebook
from vector_quantization.differential_encoding import (
    median_adaptive_predictor_encoding,
    median_adaptive_predictor_decoding,
)
from vector_quantization.golomb_coder import golomb_compress, golomb_decompress


def compression(img):
    # Vector quantization
    window_size = 4
    vectors = vectorize(img, window_size=window_size)
    means = np.mean(vectors, axis=1, keepdims=True)
    residual_vectors = vectors - means
    initial_codebook = random_codebook(residual_vectors, length=32)
    codebook, distortions = lbg(residual_vectors, initial_codebook, 50, 0.01)
    _, (codes, _) = mean_removal_quantize_from_codebook(residual_vectors, means, codebook)

    height, width = img.shape
    means_bit_code = means_compression((height, width), window_size, means)
    return img.shape, window_size, codes, codebook, means_bit_code


def codebook_compression(image_shape, window_size, codes, codebook):
    pass


def means_compression(image_shape, window_size, means):
    height, width = image_shape

    # Means reshaping for differential encoding
    means_reshaped = means.reshape((height // window_size, width // window_size))
    means_reshaped = means_reshaped.astype(int)

    # Differential encoding means
    encoded_means = median_adaptive_predictor_encoding(means_reshaped)
    encoded_means = encoded_means.astype(int)

    # Compress encoded means with Golomb Coder
    bit_code = golomb_compress(encoded_means)

    return bit_code


def decompression(codebook_data, means_bitcode):
    img_shape, window_size, codes, codebook = codebook_decompression(codebook_data)

    # Recreate residual vectors from codebook
    residual_vectors = vectors_from_codes(codes, codebook)

    # Means decompression (golomb decompression and differential decoding)
    means = means_decompression(means_bitcode, window_size)

    # Recreating vectors from residual vectors and means
    vectors = residual_vectors + means

    img_tmp = np.zeros(img_shape)

    img = image_from_vectors(vectors, img_tmp)
    return img


def codebook_decompression(codebook_bitcode):
    img_shape, window_size, codes, codebook = codebook_bitcode
    return img_shape, window_size, codes, codebook


def means_decompression(means_bitcode, window_size):
    # Decompress encoded means with Golomb Coder
    encoded_means = golomb_decompress(means_bitcode)

    # Differential decoding
    means_reshaped = median_adaptive_predictor_decoding(encoded_means)
    means = means_reshaped.reshape(-1, 1)

    return means


if __name__ == "__main__":
    img = load_image("balloon.bmp")
    img_shape, window_size, codes, codebook, means_bit_code = compression(img)

    codebook_data = (img_shape, window_size, codes, codebook)
    out_img = decompression(codebook_data, means_bit_code)

    print("PSNR:", PSNR(img, out_img))
    Image.fromarray(out_img).show()
