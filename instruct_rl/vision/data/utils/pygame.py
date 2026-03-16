import pandas as pd
import pygame
import numpy as np
import os
from os.path import dirname, abspath, basename, join
import json


from instruct_rl.evaluate import get_loss_batch
from instruct_rl.vision.data.render import render_array, render_numpy

# set tile seze/color
tile_size = 32
colors = {1: (200, 200, 200), 2: (50, 50, 50), 3: (200, 50, 50)}
tile_types = [1, 2, 3]


def draw_text(screen, text, position, color=(255, 255, 255), size=16):
    font = pygame.font.Font(None, size)
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, position)

def draw_map(screen, level):
    img = render_array(level, tile_size)

    img = pygame.surfarray.make_surface(img)

    # rotate image 90 degrees
    img = pygame.transform.rotate(img, 90)
    # Flip the image vertically
    img = pygame.transform.flip(img, False, True)

    screen.blit(img, (0, 0))

def load_level(file, resize=False):
    level = np.load(file)

    if resize:
        # padding 16x16 with 2; crop to 16x16
        level = np.pad(level, ((0, 16 - level.shape[0]), (0, 16 - level.shape[1])), mode='constant', constant_values=2)
        level = level[:16, :16]
    return level

def load_level_from_arr(files, index, inst_df):
    file = files[index]
    level = load_level(file, resize=True)
    raw_level = load_level(file, resize=False)

    paths = file.split("\\") if "\\" in file else file.split("/")
    instruction, file_name = paths[-2], paths[-1]

    cond = None

    if inst_df is not None:
        inst_df['instruction'] = inst_df['instruction'].apply(lambda x: x.replace(" ", "_").replace(".", "").lower())
        inst_df = inst_df[inst_df['instruction'] == instruction]
        # get the condition_0 to condition_4

        if len(inst_df) > 0:
            inst_row = inst_df.iloc[0]

            output_str = list()

            if inst_row['condition_0'] != -1:
                output_str.append(f"region: {inst_row['condition_0']}")
            if inst_row['condition_1'] != -1:
                output_str.append(f"path length: {inst_row['condition_1']}")
            if inst_row['condition_2'] != -1:
                output_str.append(f"wall count: {inst_row['condition_2']}")
            if inst_row['condition_3'] != -1:
                output_str.append(f"bat count: {inst_row['condition_3']}")
            if inst_row['condition_4'] != -1:
                output_str.append(f"bat direction: {inst_row['condition_4']}")

            output_str.append(f"SC: {inst_row['sub_condition_1']}")

        cond = ', '.join(output_str)
    instruction = instruction.replace("_", " ")

    return level, raw_level, instruction, file, cond

def get_json_path(file):
    paths = file.split("\\") if "\\" in file else file.split("/")

    paths[-3] = "json"
    paths[-1] = paths[-1].replace(".npy", ".json")
    json_path = os.path.join(*paths)

    # if the file starts with /, add it to the json_path
    if file.startswith("/"):
        json_path = "/" + json_path

    return json_path

def get_png_path(file):
    paths = file.split("\\") if "\\" in file else file.split("/")
    paths[-3] = "png"
    paths[-1] = paths[-1].replace(".npy", ".png")
    json_path = os.path.join(*paths)

    if file.startswith("/"):
        json_path = "/" + json_path

    return json_path

def save_level(level, file):
    # Save the level as a numpy file
    np.save(file, level)

    # Save the level as a JSON file
    json_path = get_json_path(file)
    with open(json_path, "w") as f:
        json.dump(level.tolist(), f)

    # Save the level as a PNG file
    png_path = get_png_path(file)
    render_numpy(file, png_path, 16)

    print(f"File changed: {[basename(file), basename(json_path), basename(png_path)]}")

def delete_level(files, file):
    # Delete the level as a numpy file
    os.remove(file)

    # Delete the level as a JSON file
    json_path = get_json_path(file)
    if os.path.exists(json_path):
        os.remove(json_path)

    # Delete the level as a PNG file
    png_path = get_png_path(file)
    if os.path.exists(png_path):
        os.remove(png_path)

    print(f"File deleted: {[basename(file), basename(json_path), basename(png_path)]}")

    files.remove(file)

    return files


def get_reward_enum(instruction, inst_df):
    """
    Get the reward enum for the given instruction from the inst_df DataFrame.
    """
    if inst_df is not None:
        inst_df['instruction'] = inst_df['instruction'].apply(lambda x: x.replace(" ", "_").replace(".", "").lower())
        inst_row = inst_df[inst_df['instruction'] == instruction]

        if len(inst_row) > 0:
            return inst_row.iloc[0]['reward_enum']

    return None

def convert_feature_to_status(feature):
    status = list()

    n_region, path_length, n_block, n_bat, n_bat_dir0, n_bat_dir1, n_bat_dir2, n_bat_dir3 = feature

    status.append(f"{n_region} regions, {path_length} path length, {n_block} blocks, {n_bat} bats")
    status.append(f"Bats on {n_bat_dir0} west, {n_bat_dir1} north, {n_bat_dir2} east, {n_bat_dir3} south")

    return status
