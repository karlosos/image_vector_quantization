import os
import numpy as np
import pandas as pd
import subprocess

from vector_quantization.image import load_image, save_image
from vector_quantization.metrics import PSNR


def main():
    data = {"name": [], "image_size": [], "size_our": [], "size_jpg": [], "psnr_our": [], "psnr_jpg": []}

    codebook_size = 1024
    window_size = 4
    df = pd.read_csv(f"./experiments/compression_{codebook_size}_4.csv")

    for row in df.values:
        size_our = row[5]
        psnr_our = row[6]

        filename = f"./img/input/{row[1]}"
        out = f"./img/output/jpg/{row[1]}.jpg"

        cmd = f"magick convert {filename} -define jpeg:extent={int(size_our/8000)}kb {out}"
        print(cmd)
        subprocess.call(cmd, shell=True)

        size_jpg = os.path.getsize(out) * 8

        img = load_image(filename, path="")
        jpg_img = load_image(out, path="")
        psnr_jpg = PSNR(img, jpg_img)

        data["name"].append(row[1])
        data["size_our"].append(size_our)
        data["size_jpg"].append(size_jpg)
        data["psnr_our"].append(psnr_our)
        data["psnr_jpg"].append(psnr_jpg)
        data["image_size"].append(row[2])

    df = pd.DataFrame(data)
    print(df)
    df.to_csv(f"./experiments/jpg_compression_{codebook_size}_{window_size}.csv")


if __name__ == "__main__":
    main()
