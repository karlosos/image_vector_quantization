import numpy as np


def differential_encoding(image):
    image = image.astype("float64")
    height, width = image.shape
    encoded = np.zeros(image.shape, dtype="float64")
    # rewrite first element
    encoded[0, 0] = image[0, 0]
    # calculate first row
    print(image.shape)
    for i in range(1, width):
        encoded[0, i] = image[0, i] - image[0, i - 1]
    # calculate first column
    for i in range(1, height):
        encoded[i, 0] = image[i, 0] - image[i - 1, 0]
    # calculate other values
    for i in range(1, height):
        for j in range(1, width):
            encoded[i, j] = image[i, j] - image[i, j - 1]
    return encoded


def differential_decoding(codes):
    pass


def differential_median_encoding(image):
    height, width = image.shape
    encoded = np.zeros(image.shape)
    # rewrite first element
    encoded[0, 0] = image[0, 0]
    # calculate first row
    print(image.shape)
    for i in range(1, width):
        encoded[0, i] = image[0, i] - encoded[0, i - 1]
    # calculate first column
    for i in range(1, height):
        encoded[i, 0] = image[i, 0] - encoded[i - 1, 0]
    # calculate other values
    # TODO: median encoding
    return encoded


def differential_median_decoding(codes):
    pass


if __name__ == "__main__":
    from PIL import Image

    from vector_quantization.metrics import PSNR
    from vector_quantization.image import load_image, save_image
    from vector_quantization.codebooks import random_codebook
    from vector_quantization.vectorize import vectorize, image_from_vectors
    from vector_quantization.lbg import lbg
    from vector_quantization.mean_removal_quantization import mean_removal_quantize
    from vector_quantization.mean_removal_quantization import mean_removal_quantize_from_codebook

    import matplotlib.pyplot as plt

    img = load_image("lennagrey.bmp")
    # plt.imshow(img)
    # plt.show()
    window_size = 4
    vectors = vectorize(img, window_size=window_size)
    means = np.mean(vectors, axis=1, keepdims=True)  # mean should be in shape like smaller image.
    height, width = img.shape
    means_reshaped = means.reshape((height // window_size, width // window_size))

    # plt.imshow(means_reshaped)
    # plt.show()

    encoded = differential_encoding(means_reshaped)

    # encoded image
    plt.imshow(encoded)
    plt.show()

    # histograms
    # plt.hist(means, np.arange(255))
    # plt.hist(encoded.ravel(), np.arange(-255, 255))
    # plt.show()

    # histogramy - prawdopodobieństwa
    hist_means, bins_means = np.histogram(means, bins=np.arange(256))
    hist_means = hist_means / np.sum(hist_means)
    hist_encoded, bins_encoded = np.histogram(encoded.ravel(), bins=np.arange(-255, 256))
    hist_encoded = hist_encoded / np.sum(hist_encoded)
    plt.plot(bins_means[:-1], hist_means, label="średnie")
    plt.plot(bins_encoded[:-1], hist_encoded, label="kodowanie różnicowe")
    plt.legend()
    plt.show()

    # entropia
    from scipy.stats import entropy

    entropy_means = entropy(hist_means, base=2)
    entropy_encoded = entropy(hist_encoded, base=2)
    # TODO: kodowanie różnicowe daje większą entropię niż 8!!!!!
    print(f"Entropia średnie = {entropy_means}, entropia kodowanie różnicowe = {entropy_encoded}")

    # Test dla lenny
    print("LENNA")
    img = load_image("lennagrey.bmp")
    encoded = differential_encoding(img)
    hist_img, bins_img = np.histogram(img, bins=np.arange(256))
    hist_img = hist_img / np.sum(hist_img)
    hist_encoded, bins_encoded = np.histogram(encoded.ravel(), bins=np.arange(-255, 256))
    hist_encoded = hist_encoded / np.sum(hist_encoded)
    plt.plot(bins_img[:-1], hist_img, label="średnie")
    plt.plot(bins_encoded[:-1], hist_encoded, label="kodowanie różnicowe")
    plt.legend()
    plt.show()
    entropy_means = entropy(hist_means, base=2)
    entropy_encoded = entropy(hist_encoded, base=2)
    print(f"Entropia średnie = {entropy_means}, entropia kodowanie różnicowe = {entropy_encoded}")
