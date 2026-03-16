import argparse
import time
from openai import OpenAI
from dotenv import load_dotenv
from os.path import join

# convert unix timestamp to human-readable format
def format_timestamp(timestamp):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp)) if timestamp else "N/A"


def watch_batch(batch_id: str, repeat: int = -1, sleep_time: int = 60):
    client = OpenAI()

    for i in range(repeat if repeat > 0 else int(2147483647)):  # repeat indefinitely if repeat is -1
        batch = client.batches.retrieve(batch_id)

        print(f"\n🧾 Batch ID: {batch.id}")
        print(f"📦 Status: {batch.status}")
        print(f"🕒 Created: {format_timestamp(batch.created_at)}")
        print(f"⏳ Expires: {format_timestamp(batch.expires_at)}")
        print(f"🟩 Completed: {format_timestamp(batch.completed_at)}")
        print(f"🟨 In Progress: {format_timestamp(batch.in_progress_at)}")
        print(f"🟥 Failed: {format_timestamp(batch.failed_at)}")

        if batch.output_file_id:
            print(f"\n📁 Output File ID: {batch.output_file_id}")
            print("👉 Download it from: https://platform.openai.com/files/" + batch.output_file_id)
        else:
            print("\n📁 Output File is not available yet.")

        if batch.status in "completed failed".split():
            print("\nBatch is completed or failed. Exiting watch.")
            break

        time.sleep(sleep_time)

if __name__ == "__main__":
    load_dotenv()

    # read the batch_id.txt in the /batch directory
    try:
        with open(join("batch", "batch_id.txt"), "r") as f:
            batch_id = f.read().strip()
    except FileNotFoundError:
        print("batch_id.txt not found. Please run the batch submission script first.")
        exit(1)

    watch_batch(batch_id=batch_id, repeat=-1)
