import numpy as np
import bitarray
import math


def golomb_coder(image):
    """
    Golomb coder of image

    :param img: image to compress. Only unsigned values.
    :return string with binary code and list of m values
    """
    height, width = image.shape
    codes = np.empty(image.shape, dtype=object)

    # rewrite first element
    codes[0, 0] = image[0, 0]
    # calculate first row
    for i in range(1, width):
        e = image[0, i]
        S = image[0, i - 1]
        codes[0, i] = value_coder(e, S)
    # calculate first column
    for i in range(1, height):
        e = image[i, 0]
        S = image[i - 1, 0]
        codes[i, 0] = value_coder(e, S)
    # calculate other values
    for i in range(1, height):
        for j in range(1, width):
            e = image[i, j]
            S = np.mean([image[i, j - 1], image[i - 1, j], image[i - 1, j - 1]])
            codes[i, j] = value_coder(e, S)

    print(codes)
    first_element = image[0, 0]
    binary_code = ""
    m_values = []
    for i in range(1, codes.size):
        binary_code += codes.flat[i][0]
        m_values.append(codes.flat[i][1])

    print(first_element)
    print(binary_code)
    print(m_values)

    return first_element, binary_code, m_values


def value_coder(e, S):
    if S < 2:
        p = 0.5
    else:
        p = (S - 1) / S
    m = math.ceil(-(np.log10(1 + p) / np.log10(p)))
    u_g = int(e / m)
    v_g = int(e - u_g * m)

    # u_g to binary code
    u_g_code = "0" * u_g + "1"

    # v_g to binary code
    k = math.ceil(np.log2(m))
    l_ = 2 ** k - m
    if v_g < l_:
        v_g_code = f"{v_g:08b}"
        v_g_code = v_g_code[-k + 1 :]
    else:
        v_g_code = f"{v_g+l_:08b}"
        v_g_code = v_g_code[-k:]

    code = u_g_code + v_g_code
    return code, m


def golomb_decoder(binary_code, m_values):
    """
    Golomb decoder from binary code and m values

    :param binary_code: binary code coded by golomb coder
    :param m_values: list of m values, associated with each coded value
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

    # print(f"{e} \t {u_g_unary}:{v_g_code}")
    print(u_g, v_g)
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


def decode_string(code, m_values):
    import math

    ptr = 0

    values = []
    num_index = 0

    while True:
        # Decode u_g
        u_g = 0
        while True:
            bit = code[ptr]
            ptr += 1
            if bit == "1":
                break
            u_g += 1
        print("u_g", u_g)

        # Decpde v_g

        m = m_values[num_index]
        k = math.ceil(np.log2(m))
        l_ = 2 ** k - m

        v_g_code_tmp = code[ptr : ptr + k - 1]
        ptr = ptr + k - 1

        v_g = int(v_g_code_tmp, 2)
        if v_g >= l_:
            g = int(code[ptr : ptr + 1])
            v_g = 2 * v_g + g - l_
            ptr = ptr + 1

        print("v_g", v_g)

        # print("decode:", u_g, v_g)
        e = u_g * m + v_g
        values.append(e)
        num_index += 1

        if num_index >= len(m_values):
            break
    print(values)
    return values


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


def test_decoding():
    numbers = [10, 30, 1, 9, 12, 42]
    # numbers = range(32)
    code = ""
    m_values = []
    for i in numbers:
        u_g_unary, v_g_code, m = step(i)
        code += u_g_unary + v_g_code
        m_values.append(m)
        # decode(u_g_unary, v_g_code, m)
    print(code)
    print(m_values)
    print("Decoding:.......")
    decode_string(code, m_values)


if __name__ == "__main__":
    # main()
    # test_decoding()
    img = np.array([[1, 2, 3, 9], [3, 2, 5, 3], [5, 3, 2, 1], [5, 3, 7, 99]])
    print(img)
    golomb_coder(img)
