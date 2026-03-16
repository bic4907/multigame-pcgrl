import argparse
import time
from enum import IntEnum

from openai import OpenAI
from tqdm import tqdm
import os
import pandas as pd
from os.path import join, abspath, dirname
import logging
import json
import re
import numpy as np

from instruct_rl.vision.data.generator.utils import Dungeon3Tiles, RESPONSE_FORMAT, condition2string, SYSTEM_MESSAGE, \
    USER_MESSAGE
from instruct_rl.vision.data.render import render_level

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger('httpx').setLevel(logging.WARNING)


def generate_dataset(config):

    logger.info(f"Generating dataset with model {config}")

    # read the csv file
    df = pd.read_csv(config.csv)
    n_inst = len(df)

    client = OpenAI(
        base_url=config.base_url,
        api_key=config.api_key
    )

    config.n = config.n

    for i, row in df.iterrows():
        for seed in tqdm(range(config.n), desc=f"{row['instruction']} ({i}/{n_inst})"):

            subdir = re.sub(r'\s+', '_', row['instruction'].lower()).replace('.', '')
            # make a subdirectory for each instruction by lowercasing and concat spaces with underscore

            # subdirectory for each instruction by lowercasing and concat spaces with underscore
            json_dir = join(config.json_dir, subdir)
            npy_dir = join(config.npy_dir, subdir)
            png_dir = join(config.png_dir, subdir)
            s0_png_dir = join(config.s0_png_dir, subdir)

            os.makedirs(json_dir, exist_ok=True)
            os.makedirs(npy_dir, exist_ok=True)
            os.makedirs(png_dir, exist_ok=True)
            os.makedirs(s0_png_dir, exist_ok=True)

            # if the rendered level exists in the directory, skip it.
            base_file = join(png_dir, f"{config.model}_level_{i}_s{seed}_0.png")
            if os.path.exists(base_file):
                continue

            rnd_weights = {
                Dungeon3Tiles.EMPTY: 0.4,
                Dungeon3Tiles.WALL: 0.57,
                Dungeon3Tiles.BAT: 0.03,
            }

            tile_values = [int(tile.value) if hasattr(tile, 'value') else int(tile) for tile in rnd_weights.keys()]
            tile_probs = list(rnd_weights.values())
            rnd_level = np.random.choice(tile_values, size=(1, 16, 16), p=tile_probs)

            cond_str = condition2string(row)

            rnd_level_str = ''
            for j, level_data in enumerate(rnd_level):
                rnd_level_str += f"Level {j}:\n"
                rnd_level_str += '\n'.join(' '.join(str(cell) for cell in row) for row in level_data)
                rnd_level_str += '\n\n'

            test_messages = [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": USER_MESSAGE.format(instruction=row['instruction'],
                                                                   rnd_level=rnd_level_str,
                                                                   condition=cond_str,
                                                                   n=1)}
            ]

            try:
                response = client.responses.create(
                    model=config.model,
                    input=test_messages,
                    temperature=config.temperature,
                    top_p=0.95,
                    text=RESPONSE_FORMAT,
                )
                array = np.array(json.loads(response.output_text)["matrix"])

            except Exception as e:
                logger.error(f"Failed to generate level {i} with seed {seed}")
                logger.error(e)
                continue


            # save the level files (separatly, json, npy, and png)
            for j, level_data in enumerate(array):
                # save the response to a file with seed name
                json_file_path = join(json_dir, f"{config.model}_level_{i}_s{seed}_{j}.json")
                with open(json_file_path, 'w') as f:
                    json.dump(level_data.tolist(), f)

                np.save(join(npy_dir, f"{config.model}_level_{i}_s{seed}_{j}.npy"), level_data)
                render_level(json_file_path, join(png_dir, f"{config.model}_level_{i}_s{seed}_{j}.png"), tile_size=32)

            for j, level_data in enumerate(rnd_level):
                # save the response to a file with seed name
                json_file_path = join(s0_png_dir, f"{config.model}_level_{i}_s{seed}_{j}.json")
                with open(json_file_path, 'w') as f:
                    json.dump(level_data.tolist(), f)

                render_level(json_file_path, join(s0_png_dir, f"{config.model}_level_{i}_s{seed}_{j}.png"), tile_size=32)


# Test code
if __name__ == "__main__":

    default_csv = abspath(join(dirname(__file__), "..", "..", ".." , "instruct", "test", "bert", "scn-1_se-whole.csv"))
    env_vars = os.environ

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument('--base_url', type=str, default=None)

    parser.add_argument('--api_key', type=str, default=env_vars.get("OPENAI_API_KEY", None))
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--csv", type=str, default=default_csv)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default='levels')

    config = parser.parse_args()

    # if the levels directory does not exist, create it
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    config.json_dir = join(config.output_dir, f"json")
    config.npy_dir = join(config.output_dir, f"numpy")
    config.png_dir = join(config.output_dir, f"png")
    config.s0_png_dir = join(config.output_dir, f"s0png")


    os.makedirs(config.json_dir, exist_ok=True)
    os.makedirs(config.npy_dir, exist_ok=True)
    os.makedirs(config.png_dir, exist_ok=True)
    os.makedirs(config.s0_png_dir, exist_ok=True)

    generate_dataset(config)


