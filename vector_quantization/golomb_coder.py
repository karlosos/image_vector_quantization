import numpy as np
import bitarray


def golomb_coder(image):
    """
    Parameter image is encoded using differentian encoding
    """
    pass


def step(e):
    # p = 0.95
    m = 14
    import math

    # print(m, "=", math.ceil(-np.log10(1 + p) / np.log10(p)))
    k = 4
    l_ = 2

    u_g = int(e / m)
    v_g = e - u_g * m
    # print(u_g, v_g)
    u_g_unary = "0" * u_g + "1"

    # print(k, "=", math.ceil(np.log2(m)))
    # print(l, "=", 2 ** k - 1)

    v_g = int(v_g)
    if v_g < l_:
        v_g_code = f"{v_g:08b}"
        v_g_code = v_g_code[-k + 1 :]
    else:
        v_g_code = f"{v_g+l_:08b}"
        v_g_code = v_g_code[-k:]

    print(f"{e} \t {u_g_unary}:{v_g_code}")
    return u_g_unary, v_g_code, m


def decode(u_g_unary, v_g_code, m):
    # Decode u_g_unary
    u_g = 0
    for bit in u_g_unary:
        if bit == "1":
            break
        u_g += 1

    # Decode v_g_code
    import math

    k = math.ceil(np.log2(m))
    l_ = 2 ** k - m

    # TODO: this is bad. This should be decoding from
    v_g_code_tmp = v_g_code[: k - 1]
    v_g = int(v_g_code_tmp, 2)
    if v_g >= l_:
        g = int(v_g_code[k - 1])
        v_g = 2 * v_g + g - l_

    # print("decode:", u_g, v_g)
    e = u_g * m + v_g
    print("decode:", e)


def main():
    from PIL import Image

    from vector_quantization.metrics import PSNR
    from vector_quantization.image import load_image, save_image
    from vector_quantization.codebooks import random_codebook
    from vector_quantization.vectorize import vectorize, image_from_vectors
    from vector_quantization.lbg import lbg
    from vector_quantization.mean_removal_quantization import mean_removal_quantize
    from vector_quantization.mean_removal_quantization import mean_removal_quantize_from_codebook
    from vector_quantization.differential_encoding import median_adaptive_predictor_encoding

    import matplotlib.pyplot as plt

    img = load_image("lennagrey.bmp")

    # Performing quantization with removed means (MRVQ)
    window_size = 4
    vectors = vectorize(img, window_size=window_size)
    means = np.mean(vectors, axis=1, keepdims=True)  # mean should be in shape like smaller image.
    height, width = img.shape
    means_reshaped = means.reshape((height // window_size, width // window_size))

    # Differential encoding means from MRVQ
    encoded = median_adaptive_predictor_encoding(means_reshaped)
    first_byte = f"{int(encoded[0, 0]):08b}"
    print(first_byte)

    # Transform other values to absolute values (remove sign)
    abs_encoded = np.abs(encoded)
    signs = np.sign(encoded)

    print(abs_encoded, signs)

    # Separating values to u_G and v_G

    # Creating binary object
    # out = bitarray.bitarray(first_byte)
    # print(out)
    # pass


if __name__ == "__main__":
    # main()

    numbers = [10, 30, 1, 9, 12, 42]
    # numbers = range(32)
    for i in numbers:
        u_g_unary, v_g_code, m = step(i)
        decode(u_g_unary, v_g_code, m)
