import numpy as np
from bitarray import bitarray
import math


def golomb_compress(img):
    """
    Compress signed matrix with golomb coder

    :param img: matrix with signed values (9 bit per value)
    :returns: binary code with compressed matrix
    """
    img_abs = np.abs(img)
    signs = np.sign(img)

    first_element, binary_code, m_values = golomb_coder(img_abs)

    size = img_abs.shape
    bit_code = to_binary(first_element, binary_code, signs, m_values, size)
    return bit_code


def golomb_decompress(bit_code):
    """
    Decompress signed matrix from bit code with golomb decoder

    :param bit_code: binary code compressed with golomb compression
    :returns: signed matrix
    """
    first_element, binary_code, signs, m_values, size = from_binary(bit_code)
    values_decoded = golomb_decoder(first_value=first_element, binary_code=binary_code, m_values=m_values, size=size)
    img = values_decoded * signs
    return img


def golomb_coder(image):
    """
    Golomb coder of image

    :param img: image to compress. Only unsigned values.
    :return first_element as number, string with binary code and list of m values
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

    first_element = image[0, 0]
    binary_code = ""
    m_values = []
    for i in range(1, codes.size):
        binary_code += codes.flat[i][0]
        m_values.append(codes.flat[i][1])

    return first_element, binary_code, m_values


def value_coder(e, S):
    """
    Encode single values e with given S
    :param e: error (input value)
    :param S: calculated based on neighbors (mean value of three neighbors)
    :returns: binary code as string and m value
    """
    if S < 2:
        p = 0.5
    else:
        p = (S - 1) / S
    m = math.ceil(-(np.log10(1 + p) / np.log10(p)))
    u_g = int(e / m)

    # u_g to binary code
    u_g_code = "0" * u_g + "1"

    if m != 1:
        v_g = int(e - u_g * m)

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
    else:
        code = u_g_code

    return code, m


def golomb_decoder(first_value, binary_code, m_values, size):
    """
    Golomb decoder from binary code and m values

    :param first_value:
    :param binary_code: binary code coded by golomb coder
    :param m_values: list of m values, associated with each coded value
    :param size: image size as a tuple
    """
    values = decode_string(binary_code, m_values)
    values = [first_value] + values
    decoded_img = np.array(values).reshape(size)
    return decoded_img


def decode_string(code, m_values):
    """
    Decode string with binary code with given m_values to output values (e)

    :param code: binary code of e values as a string
    :param m_values: m values for every element
    :returns: encoded values (e)
    """
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

        # Decode v_g
        m = m_values[num_index]
        if m != 1:
            k = math.ceil(np.log2(m))
            l_ = 2 ** k - m

            # There was a problem when m = 2, because k = 1
            v_g_code_tmp = code[ptr : ptr + k - 1]
            ptr = ptr + k - 1

            if v_g_code_tmp == "":
                v_g = 0
            else:
                v_g = int(v_g_code_tmp, 2)

            if v_g >= l_:
                g = int(code[ptr : ptr + 1])
                v_g = 2 * v_g + g - l_
                ptr = ptr + 1
            e = u_g * m + v_g
        else:
            e = u_g

        values.append(e)
        num_index += 1

        if num_index >= len(m_values):
            break

    return values


def to_binary(first_element, binary_code, signs, m_values, size):
    """
    Code matrix data to binary string.
    This will be stored in file.

    :param first_element: first element of a matrix (image/means)
    :param binary_code: bit code with coded e values
    :param signs: matrix with signs of coded matrix
    :param m_values:
    :param size: size of an coded matrix
    :returns: bit code
    """
    res = bitarray()
    # Creating binary object
    # Store first element
    first_byte = f"{int(first_element):08b}"
    res += bitarray(first_byte)

    # Store size
    size_h = f"{size[0]:08b}"
    res += bitarray(size_h)
    size_w = f"{size[1]:08b}"
    res += bitarray(size_w)

    print(f"Len: {size[0]*size[1]} == {len(m_values)}")

    # Store m_values
    for m in m_values:
        res += bitarray(f"{m:08b}")

    # Store signs
    # 1 -> positive
    # 0 -> negative
    signs[signs >= 0] = 1
    signs[signs == -1] = 0
    signs_code = ""
    for sign in signs.flat:
        signs_code += str(sign)
    res += bitarray(signs_code)

    # Store binary_code
    res += bitarray(binary_code)
    return res


def from_binary(code):
    """
    Decode matrix data from bit string.
    This will be loaded from file.
    :param code: bit code
    :returns: same as inputs for `to_binary`
    """
    # Load first element
    first_element_code = code[0:8]
    first_element = int(first_element_code.to01(), 2)

    # Load size
    size_h_code = code[8:16]
    size_w_code = code[16:24]

    size = [int(size_h_code.to01(), 2), int(size_w_code.to01(), 2)]

    # Load m_values
    m_values_count = size[0] * size[1] - 1
    m_values_code = code[24 : 24 + m_values_count * 8]
    m_values = []
    for i in range(m_values_count):
        m_values.append(int(m_values_code[i * 8 : (i + 1) * 8].to01(), 2))

    # Load signs
    signs_count = size[0] * size[1]
    ptr = 24 + m_values_count * 8
    signs_code = code[ptr : ptr + signs_count]
    signs = []
    for i in range(signs_count):
        signs.append(int(signs_code[i]))
    signs = np.array(signs)
    signs = signs.reshape(size)
    signs[signs == 0] = -1
    ptr = ptr + signs_count

    # Load binary code (u_g, v_g)
    binary_code = code[ptr:].to01()

    return first_element, binary_code, signs, m_values, size


def main():
    from PIL import Image

    from vector_quantization.metrics import PSNR
    from vector_quantization.image import load_image, save_image
    from vector_quantization.codebooks import random_codebook
    from vector_quantization.vectorize import vectorize, image_from_vectors
    from vector_quantization.lbg import lbg
    from vector_quantization.mean_removal_quantization import mean_removal_quantize
    from vector_quantization.mean_removal_quantization import mean_removal_quantize_from_codebook
    from vector_quantization.differential_encoding import (
        median_adaptive_predictor_encoding,
        median_adaptive_predictor_decoding,
    )

    import matplotlib.pyplot as plt

    img = load_image("lennagrey.bmp")

    # Performing quantization with removed means (MRVQ)
    window_size = 4
    vectors = vectorize(img, window_size=window_size)
    means = np.mean(vectors, axis=1, keepdims=True)  # mean should be in shape like smaller image.
    height, width = img.shape
    means_reshaped = means.reshape((height // window_size, width // window_size))

    # Differential encoding means from MRVQ
    encoded_means = median_adaptive_predictor_encoding(means_reshaped)
    encoded_means = encoded_means.astype(int)

    # Compress encoded means with Golomb Coder
    bit_code = golomb_compress(encoded_means)

    # Decompress encoded means with Golomb Coder
    decoded_means = golomb_decompress(bit_code)

    # Differential decoding
    decoded_image = median_adaptive_predictor_decoding(decoded_means)

    plt.imshow(decoded_image)
    plt.show()


if __name__ == "__main__":
    main()
