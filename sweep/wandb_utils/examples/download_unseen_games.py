"""Example that downloads table/results as CSV from the sweep_unseen_games project.

Usage:
    python -m sweep.wandb_utils.examples.download_unseen_games

resultwater:
    outputs/sweep_unseen_games/<run_id>/results.csv   -- table data and config columns
    outputs/sweep_unseen_games/<run_id>/results.json  -- original table JSON
    outputs/sweep_unseen_games/<run_id>/config.json    — run config
"""

from sweep.wandb_utils import WandbTableDownloader


def main():
    downloader = WandbTableDownloader(
        output_dir="outputs",          # download result save path
        target_files=["results"],      # W&B table name (media/table/results)
        num_workers=4,                 # Number of parallel workers
    )

    # Download every run in the sweep_unseen_games project
    downloader.download("sweep_unseen_games")


if __name__ == "__main__":
    main()
