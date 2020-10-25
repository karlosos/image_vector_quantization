from PIL import Image
from vector_quantization.metrics import PSNR
from vector_quantization.image import load_image, save_image
from vector_quantization.codebooks import random_codebook
from vector_quantization.quantization import quantize

if __name__ == "__main__":

    img = load_image("balloon.bmp")
    quantized_img = quantize(img, window_size=4, codebook_fun=random_codebook, codebook_size=32)
    Image.fromarray(quantized_img).show()
