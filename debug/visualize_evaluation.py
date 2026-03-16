import logging
import os

from tabulate import tabulate
from tqdm import tqdm
import numpy as np
from os.path import basename, join, dirname
from matplotlib import pyplot as plt

from debug.render_level import render_level
from debug.scenario_levels import AllLevels


from PIL import Image, ImageDraw, ImageFont


def add_text_below_image(image, text, font_size=20, target_width=500):
    """Add properly formatted text (including tables) below an image with a black background."""
    # Path to the font file
    font_path = "JetBrainsMono-Regular.ttf"  # Update this to the correct path
    try:
        font = ImageFont.truetype(os.path.join(dirname(__file__), font_path), font_size)
    except IOError:
        raise FileNotFoundError(f"Font file not found at {font_path}. Please provide a valid font file path.")

    draw = ImageDraw.Draw(image)

    # Split text by line manually to preserve table formatting
    text_lines = text.split("\n")


    # Calculate total text height
    text_height = sum(
        draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1]
        for line in text_lines
    )
    padding = 20  # Padding around the text
    total_height = image.height + text_height + padding

    # Create a new image
    new_image = Image.new("RGB", (image.width, total_height), (0, 0, 0))  # Black background
    new_image.paste(image, (0, 0))

    # Draw text on the new image
    draw = ImageDraw.Draw(new_image)
    y_text = image.height + 10
    for line in text_lines:
        # Left-align text
        text_x = 10  # Small padding from the left edge
        draw.text((text_x, y_text), line, fill=(255, 255, 255), font=font)
        y_text += draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1]

    return new_image


def create_image_grid(images, grid_size, output_path=None, cell_size=(500, 500)):
    """Create a grid of images resized to the given cell size."""
    grid_width, grid_height = grid_size
    cell_width, cell_height = cell_size

    # Create a blank grid
    grid_img = Image.new("RGB", (cell_width * grid_width, cell_height * grid_height), (0, 0, 0))

    for idx, img in enumerate(images):
        # Resize image to fit cell size
        img_resized = img.resize(cell_size)
        x = (idx % grid_width) * cell_width
        y = (idx // grid_width) * cell_height
        grid_img.paste(img_resized, (x, y))

    if output_path:
        grid_img.save(output_path)
    return grid_img


if __name__ == '__main__':
    # Initialize logger
    logger = logging.getLogger(basename(__file__))
    logger.setLevel(logging.DEBUG)

    base_path = join(dirname(__file__), '.cache')
    os.makedirs(base_path, exist_ok=True)
    example_path = join(base_path, 'iteration_1')
    os.makedirs(example_path, exist_ok=True)


    processed_images = []

    # levels = AllLevels[]
    levels = AllLevels[:]
    scenario_num = 2

    for idx, level in tqdm(enumerate(levels), desc="Processing Levels", total=len(levels)):
        # Render level as numpy array
        level_img = render_level(level, return_numpy=True, tile_size=28)
        # level_img = cv2.cvtColor(level_img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        level_image_pil = Image.fromarray(level_img)  # Convert numpy to PIL Image

        output_str += table

        # Add text below the image
        level_image_with_text = add_text_below_image(level_image_pil, output_str, font_size=15)

        # make sure the height is determined by the image height
        width = 8
        height = int((level_image_with_text.height / level_image_with_text.width) * width)

        plt.figure(figsize=(width, height))
        plt.tight_layout()
        plt.axis('off')  # Turn off axes
        plt.tight_layout(pad=0)  # Remove padding
        plt.gca().set_position([0, 0, 1, 1])  # Ensure the image fills the figure
        plt.imshow(level_image_with_text)
        plt.show()