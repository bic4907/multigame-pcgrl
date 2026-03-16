import argparse
import json
import os

import numpy as np
from os.path import join, abspath
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from instruct_rl.vision.data.render import render_level

LEVEL_DIR = 'levels'

def retrive_batch(batch_id: str, csv: str, out_dir: str):
    df = pd.read_csv(abspath(csv))

    level_dir = join(out_dir, LEVEL_DIR)
    os.makedirs(level_dir, exist_ok=True)

    json_dir = join(level_dir, f"json")
    npy_dir = join(level_dir, f"numpy")
    png_dir = join(level_dir, f"png")
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(npy_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    client = OpenAI()

    batch = client.batches.retrieve(batch_id)
    if batch.output_file_id is None:
        print(f"Batch {batch_id} not found.")
        return

    file = client.files.content(batch.output_file_id)

    for line in file.text.splitlines():
        data = json.loads(line)
        model, inst, seed = data["custom_id"].split("_")
        model, inst, seed = model.split("-")[1], inst.split("-")[1], seed.split("-")[1]

        # get the instruction at inst index
        instruction = df.loc[int(inst), 'instruction']
        instruction_conv = instruction.lower().replace(' ', '_').replace('.', '')

        map_json = data['response']['body']['choices'][0]['message']['content']

        try:
            level_data = np.array(json.loads(map_json)["matrix"])
        except:
            print(f"Failed to parse matrix from {data['custom_id']}")
            continue

        inst_json_dir = join(json_dir, instruction_conv)
        inst_npy_dir = join(npy_dir, instruction_conv)
        inst_png_dir = join(png_dir, instruction_conv)
        os.makedirs(inst_json_dir, exist_ok=True)
        os.makedirs(inst_npy_dir, exist_ok=True)
        os.makedirs(inst_png_dir, exist_ok=True)

        # save the response to a file with seed name
        json_file_path = join(inst_json_dir, f"{model}_level_{inst}_s{seed}.json")
        with open(json_file_path, 'w') as f:
            json_obj = json.loads(data['response']['body']['choices'][0]['message']['content'])
            json.dump(json_obj['matrix'][0], f)


        if level_data.ndim == 3:
            level_data = level_data[0]

        np.save(join(inst_npy_dir, f"{model}_level_{inst}_s{seed}.npy"), level_data)
        render_level(json_file_path, join(inst_png_dir, f"{model}_level_{inst}_s{seed}.png"), tile_size=32)

if __name__ == "__main__":
    load_dotenv()

    # read the batch_id.txt in the /batch directory
    try:
        with open(join("batch", "batch_id.txt"), "r") as f:
            batch_id = f.read().strip()
    except FileNotFoundError:
        print("batch_id.txt not found. Please run the batch submission script first.")
        exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="../../../../instruct/sub_condition/bert/scn-1_se-whole.csv")
    parser.add_argument("--out_dir", type=str, default="batch")

    args = parser.parse_args()

    retrive_batch(batch_id=batch_id, csv=args.csv, out_dir=args.out_dir)
