"""
In this experiment I was testing differential encoding for a single image.
"""


from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import entropy
import numpy as np

from vector_quantization.metrics import PSNR
from vector_quantization.image import load_image, save_image
from vector_quantization.differential_encoding import differential_encoding
from vector_quantization.differential_encoding import differential_decoding
from vector_quantization.differential_encoding import differential_median_encoding
from vector_quantization.differential_encoding import differential_median_decoding


def differential_enc():
    img = load_image("camera256.bmp")
    encoded = differential_encoding(img)
    # Debug values
    # There's something weird, as there are no negative numbers in encoded
    print(encoded)
    print(np.min(encoded))
    print(np.max(encoded))

    # Plot encoded image
    plt.imshow(encoded)
    plt.show()

    # Make histograms
    hist_img, bins_img = np.histogram(img, bins=np.arange(256))
    hist_img = hist_img / np.sum(hist_img)
    hist_encoded, bins_encoded = np.histogram(encoded.ravel(), bins=np.arange(-255, 256))
    hist_encoded = hist_encoded / np.sum(hist_encoded)

    # Plot histograms
    plt.plot(bins_img[:-1], hist_img, label="średnie")
    plt.plot(bins_encoded[:-1], hist_encoded, label="kodowanie różnicowe")
    plt.legend()
    plt.show()

    # Calculate entropy
    entropy_means = entropy(hist_img, base=2)
    entropy_encoded = entropy(hist_encoded, base=2)
    print(f"Entropia średnie = {entropy_means}, entropia kodowanie różnicowe = {entropy_encoded}")

    # Decode image
    decoded_image = differential_decoding(encoded)
    plt.imshow(decoded_image)
    plt.show()

    # PSNR
    print("PSNR: ", PSNR(img, decoded_image))


def differential_median():
    img = load_image("camera256.bmp")
    encoded = differential_median_encoding(img)
    # Debug values
    # There's something weird, as there are no negative numbers in encoded
    print(encoded)
    print(np.min(encoded))
    print(np.max(encoded))

    # Plot encoded image
    plt.imshow(encoded)
    plt.show()

    # Make histograms
    hist_img, bins_img = np.histogram(img, bins=np.arange(256))
    hist_img = hist_img / np.sum(hist_img)
    hist_encoded, bins_encoded = np.histogram(encoded.ravel(), bins=np.arange(-255, 256))
    hist_encoded = hist_encoded / np.sum(hist_encoded)

    # Plot histograms
    plt.plot(bins_img[:-1], hist_img, label="średnie")
    plt.plot(bins_encoded[:-1], hist_encoded, label="kodowanie różnicowe")
    plt.legend()
    plt.show()

    # Calculate entropy
    entropy_means = entropy(hist_img, base=2)
    entropy_encoded = entropy(hist_encoded, base=2)
    print(f"Entropia średnie = {entropy_means}, entropia kodowanie różnicowe = {entropy_encoded}")

    # Decode image
    decoded_image = differential_median_decoding(encoded)
    plt.imshow(decoded_image)
    plt.show()

    # PSNR
    print("PSNR: ", PSNR(img, decoded_image))


if __name__ == "__main__":
    # differential_enc()
    differential_median()
