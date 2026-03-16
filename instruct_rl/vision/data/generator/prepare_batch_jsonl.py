import pandas as pd
import os
import json
import argparse
import re
import numpy as np
from os.path import join
from instruct_rl.vision.data.generator.utils import Dungeon3Tiles, condition2string, SYSTEM_MESSAGE, USER_MESSAGE, \
    RESPONSE_FORMAT


def main(args):
    df = pd.read_csv(args.csv)
    os.makedirs(args.out_dir, exist_ok=True)

    jsonl_path = join(args.out_dir, "batch_input.jsonl")

    model_conv = args.model.replace('-', '').replace('.','').lower()

    with open(jsonl_path, "w") as fout:
        for i, row in df.iterrows():
            for seed in range(args.n):
                weights = {
                    Dungeon3Tiles.EMPTY: 0.4,
                    Dungeon3Tiles.WALL: 0.57,
                    Dungeon3Tiles.BAT: 0.03,
                }
                tile_values = [int(tile.value) for tile in weights]
                tile_probs = list(weights.values())
                rnd_level = np.random.choice(tile_values, size=(1, 16, 16), p=tile_probs)

                rnd_level_str = '\n\n'.join(
                    f"Level {j}:\n" + '\n'.join(' '.join(str(cell) for cell in row) for row in level_data)
                    for j, level_data in enumerate(rnd_level)
                )

                msg = {
                    "custom_id": f"model-{model_conv}_level-{i}_seed-{seed}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": args.model,
                        "messages": [
                            {"role": "system", "content": SYSTEM_MESSAGE},
                            {"role": "user", "content": USER_MESSAGE.format(
                                instruction=row['instruction'],
                                rnd_level=rnd_level_str,
                                condition=condition2string(row),
                                n=1
                            )}
                        ],
                        "temperature": args.temperature,
                        "top_p": 0.95,
                        "response_format": RESPONSE_FORMAT
                    }
                }
                fout.write(json.dumps(msg) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--out_dir", type=str, default="batch")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--model", type=str, default="gpt-4.1")
    args = parser.parse_args()
    main(args)
