from PIL import Image
import numpy as np

def _read_png(png_file: str) -> np.array:
    try:
        image = Image.open(png_file).convert('RGBA')
        png_array = np.array(image, dtype=np.uint8)

        assert png_array.shape == (256, 256, 4), f"PNG shape mismatch: {png_file} - {png_array.shape}"
    except:
        return None

    return png_array

def get_pngs(png_file_list: list) -> np.array:
    images = list()

    for png_file in png_file_list:
        images.append(_read_png(png_file))

    return images
