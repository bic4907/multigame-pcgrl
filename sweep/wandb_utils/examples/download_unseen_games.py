"""sweep_unseen_games 프로젝트에서 table/results 를 CSV 로 다운로드하는 예시.

사용법:
    python -m sweep.wandb_utils.examples.download_unseen_games

결과물:
    outputs/sweep_unseen_games/<run_id>/results.csv   — 테이블 데이터 + config 컬럼
    outputs/sweep_unseen_games/<run_id>/results.json   — 원본 table JSON
    outputs/sweep_unseen_games/<run_id>/config.json    — run config
"""

from sweep.wandb_utils import WandbTableDownloader


def main():
    downloader = WandbTableDownloader(
        output_dir="outputs",          # 다운로드 결과 저장 경로
        target_files=["results"],      # W&B 에서 받을 테이블 이름 (media/table/results)
        num_workers=4,                 # 병렬 워커 수
    )

    # sweep_unseen_games 프로젝트의 모든 run 다운로드
    downloader.download("sweep_unseen_games")


if __name__ == "__main__":
    main()

