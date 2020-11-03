from PIL import Image

from vector_quantization.metrics import PSNR
from vector_quantization.image import load_image, save_image
from vector_quantization.codebooks import random_codebook
from vector_quantization.vectorize import vectorize, image_from_vectors
from vector_quantization.quantization import quantize, quantize_from_codebook
from vector_quantization.lbg import lbg

if __name__ == "__main__":
    img = load_image("balloon.bmp")
    vectors = vectorize(img, window_size=4)
    initial_codebook = random_codebook(vectors, length=32)
    codebook, distortions = lbg(vectors, initial_codebook, 50, 0.01)
    quantized_img_lbg = image_from_vectors(quantize_from_codebook(vectors, initial_codebook), img)
    Image.fromarray(quantized_img_lbg).show()

    print(codebook)
    print(distortions)
