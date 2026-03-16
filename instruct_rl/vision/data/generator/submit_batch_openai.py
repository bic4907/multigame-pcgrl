import argparse
from dotenv import load_dotenv
from openai import OpenAI
from os.path import join

from instruct_rl.vision.data.generator.watch_batch import watch_batch


def main(args):
    client = OpenAI()

    with open(args.jsonl_path, "rb") as f:
        file = client.files.create(file=f, purpose="batch")

    print("Uploaded file:", file.id)

    batch = client.batches.create(
        input_file_id=file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"type": "level_generation"},
    )

    print("Batch created with ID:", batch.id)

    with open(join(args.out_dir, "batch_id.txt"), "w") as fout:
        fout.write(batch.id)

    print("Batch ID saved to batch_id.txt")
    watch_batch(batch_id=batch.id, repeat=5, sleep_time=1)

if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="batch")
    parser.add_argument("--jsonl_path", type=str, default='./batch/batch_input.jsonl')
    args = parser.parse_args()
    main(args)