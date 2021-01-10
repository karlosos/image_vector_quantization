import os
import numpy as np
from PIL import Image
from vector_quantization.image import load_image, save_image
from vector_quantization.metrics import PSNR
from vector_quantization.compression import compression, decompression
import pandas as pd


def size_before_golomb_calc(img, codebook_size=2 ** 10, window_size=4):

    block_size = np.log2(codebook_size)
    dict_size = codebook_size * (window_size ** 2) * 9

    heigh, width = img.shape

    codebook_section_size = dict_size + (heigh * width) / (window_size ** 2) * block_size

    means_section_size = (heigh * width) / (window_size ** 2) * 8

    return codebook_section_size, means_section_size


def main():
    filenames = os.listdir("./img/input/")

    data = {"name": [], "size": [], "size_bmp": [], "size_quant": [], "size_golomb": [], "psnr": []}

    codebook_size = 2 ** 9
    window_size = 4

    for filename in filenames:
        for i in range(0, 10):
            try:
                print(filename)
                img = load_image(filename, "./img/input/")

                bitcode = compression(img, codebook_size=codebook_size, window_size=window_size)
                size_bmp = img.size * 8
                size_golomb = len(bitcode)

                out_img = decompression(bitcode)
                psnr = PSNR(img, out_img)
            except ValueError:
                print(f"Try #{i} failed. Retrying..")
                continue

            data["name"].append(filename)
            data["size"].append(img.shape)
            data["size_bmp"].append(size_bmp)
            data["size_golomb"].append(size_golomb)
            data["size_quant"].append(np.sum(size_before_golomb_calc(img, codebook_size, window_size)))
            data["psnr"].append(psnr)
            break

    df = pd.DataFrame(data)
    print(df)
    df.to_csv(f"./experiments/compression_{codebook_size}_{window_size}.csv")


if __name__ == "__main__":
    main()
