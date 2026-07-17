"""sweep_unseen_games text to text in  table/results   CSV  to  downloadtext  text.

Usage:
    python -m sweep.wandb_utils.examples.download_unseen_games

resultwater:
    outputs/sweep_unseen_games/<run_id>/results.csv   — text text data + config text
    outputs/sweep_unseen_games/<run_id>/results.json   — text table JSON
    outputs/sweep_unseen_games/<run_id>/config.json    — run config
"""

from sweep.wandb_utils import WandbTableDownloader


def main():
    downloader = WandbTableDownloader(
        output_dir="outputs",          # download result save path
        target_files=["results"],      # W&B  in  text  text text name (media/table/results)
        num_workers=4,                 # parallel text text
    )

    # sweep_unseen_games text to text of  text run download
    downloader.download("sweep_unseen_games")


if __name__ == "__main__":
    main()

