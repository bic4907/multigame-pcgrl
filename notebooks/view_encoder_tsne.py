"""
notebooks/view_encoder_tsne.py

fewshot 인코더(unseen game 1개 이상) 체크포인트에서 t-SNE 임베딩을 뽑아
notebooks/ 폴더에 저장한다. encoder/export_tsne.py와 동일한 export 로직을
재사용한다. 체크포인트를 지정하는 방법은 두 가지다.

1) --unseen-games + --unseen-ratio (+ --seen-ratio/--delta-weight)
   체크포인트 이름을 직접 문자열로 조립하지 않고, encoder/utils/path.py의
   get_exp_group()/get_exp_dir() (실제 학습 스크립트가 디렉토리 이름을 정할 때
   쓰는 바로 그 함수)을 그대로 태워서 계산한다. sr=1.0, delta_weight=0.0이면
   이름에서 각각 생략되는 규칙까지 항상 실제와 일치한다.
   → sr/ur/dw 조합으로 이름이 예측 가능한 체크포인트 전용 (예: 기존 5개 fewshot 세트).

2) --ckpt-name
   saves_dir 아래의 체크포인트 폴더명을 그대로 지정한다. 이름 규칙에서 벗어난
   체크포인트(수동으로 옮겨둔 것 등)는 이 방식을 쓴다.

사전 준비 (checkpoint를 saves/ 로 받아두기, 예시):
    for g in pk dg zd dm sk; do
      rsync -avP -e ssh \
        autox:/mnt/nas/mgpcgrl/mgpcgrl_fewshot/clipdec-game-all_unseen-${g}_ur-0.03_exp-def_dw-0p03_0/ \
        saves/clipdec-game-all_unseen-${g}_ur-0.03_exp-def_dw-0p03_0/
    done

Usage:
    # 1) 이름이 예측 가능한 경우 (기존 5개: unseen game 1개, ur=0.03, dw=0.03이 기본값)
    python notebooks/view_encoder_tsne.py --unseen-games zd
    python notebooks/view_encoder_tsne.py --unseen-games zd dm sk --samples-per-game 500
    # ur/dw가 다른 체크포인트라면 명시적으로 오버라이드
    python notebooks/view_encoder_tsne.py --unseen-games pkzddm --unseen-ratio 0.1 --delta-weight 0.0

    # 2) 폴더명을 직접 지정 (이름 규칙에서 벗어난 체크포인트)
    python notebooks/view_encoder_tsne.py --ckpt-name clipdec-game-all_unseen-pkzddm_ur-0.1_exp-def_0
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def make_config_by_ratios(unseen_games: str, saves_dir: str, unseen_ratio: float, seen_ratio: float,
                           delta_weight: float, samples_per_game: int, out_dir: str):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from conf.config import CLIPDecoderTrainConfig  # noqa: F401 — ConfigStore 등록
    from encoder.utils.path import init_config

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        cfg = compose(
            config_name="train_clip_decoder",
            overrides=[
                "game=all",
                f"saves_dir={saves_dir}",
                f"unseen_games={unseen_games}",
                f"unseen_ratio={unseen_ratio}",
                f"seen_ratio={seen_ratio}",
                f"delta_weight={delta_weight}",
                f"+tsne.out_dir={out_dir}",
                f"+tsne.samples_per_game={samples_per_game}",
            ],
        )
    # get_exp_group()/get_exp_dir() 이 여기서 실행되어, ckpt_dir/ckpt_name을
    # 명시하지 않아도 config.exp_dir → saves/<계산된 exp_name>으로 정확히 잡힌다.
    return init_config(cfg)


def make_config_by_name(ckpt_name: str, saves_dir: str, samples_per_game: int, out_dir: str):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from conf.config import CLIPDecoderTrainConfig  # noqa: F401 — ConfigStore 등록
    from encoder.utils.path import init_config

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        cfg = compose(
            config_name="train_clip_decoder",
            overrides=[
                "game=all",
                f"encoder.ckpt_dir={saves_dir}",
                f"encoder.ckpt_name={ckpt_name}",
                f"+tsne.out_dir={out_dir}",
                f"+tsne.samples_per_game={samples_per_game}",
            ],
        )
    return init_config(cfg)


def export_and_copy(label: str, cfg, dst_suffix: str) -> None:
    if cfg.encoder.model is None:
        cfg.encoder.model = "cnnclip"
    if cfg.encoder.model != "cnnclip":
        raise NotImplementedError(
            f"view_encoder_tsne currently supports cnnclip decoder checkpoints, got {cfg.encoder.model}"
        )

    from encoder.visualization.tsne_export import export_tsne_from_config

    print(f"[{label}] checkpoint: {cfg.encoder.ckpt_dir or cfg.exp_dir}")
    outputs = export_tsne_from_config(cfg)

    dst_dir = ROOT / "notebooks" / "tsne_raw"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for key, tag in (("png_2d", "2d"), ("png_3d", "3d")):
        src = Path(outputs[key])
        dst = dst_dir / f"tsne_{dst_suffix}_{tag}.png"
        shutil.copy(src, dst)
        print(f"  saved: {dst}")


def view_by_ratios(unseen_games: str, args: argparse.Namespace) -> None:
    cfg = make_config_by_ratios(
        unseen_games=unseen_games,
        saves_dir=args.saves_dir,
        unseen_ratio=args.unseen_ratio,
        seen_ratio=args.seen_ratio,
        delta_weight=args.delta_weight,
        samples_per_game=args.samples_per_game,
        out_dir=str(ROOT / "notebooks" / "tsne_runs"),
    )
    suffix = f"unseen-{unseen_games}_sr-{args.seen_ratio:g}_ur-{args.unseen_ratio:g}_dw-{args.delta_weight:g}"
    export_and_copy(unseen_games, cfg, suffix)


def view_by_name(ckpt_name: str, args: argparse.Namespace) -> None:
    cfg = make_config_by_name(
        ckpt_name=ckpt_name,
        saves_dir=args.saves_dir,
        samples_per_game=args.samples_per_game,
        out_dir=str(ROOT / "notebooks" / "tsne_runs"),
    )
    export_and_copy(ckpt_name, cfg, ckpt_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="fewshot 인코더 체크포인트 t-SNE 시각화")
    parser.add_argument(
        "--unseen-games", nargs="+", default=None,
        help="볼 unseen game 조합 (예: zd, pkzddm). --unseen-ratio와 함께 이름을 자동 계산한다.",
    )
    parser.add_argument(
        "--ckpt-name", nargs="+", default=None,
        help="saves-dir 아래 체크포인트 폴더명을 직접 지정 (이름 규칙에서 벗어난 체크포인트용).",
    )
    parser.add_argument("--saves-dir", default="saves", help="체크포인트가 위치한 saves 디렉토리")
    parser.add_argument("--unseen-ratio", type=float, default=0.03,
                         help="[--unseen-games 전용] ckpt 이름의 ur- 값 (기본값: 기존 5개 fewshot 세트=0.03)")
    parser.add_argument("--seen-ratio", type=float, default=1.0,
                         help="[--unseen-games 전용] ckpt 이름의 sr- 값 (1.0이면 이름에서 생략됨)")
    parser.add_argument("--delta-weight", type=float, default=0.03,
                         help="[--unseen-games 전용] ckpt 이름의 dw- 값 (기본값: 기존 5개 fewshot 세트=0.03, 0.0이면 이름에서 생략됨)")
    parser.add_argument("--samples-per-game", type=int, default=1000)
    args = parser.parse_args()

    if bool(args.unseen_games) == bool(args.ckpt_name):
        parser.error("--unseen-games 또는 --ckpt-name 중 하나만 지정하세요.")

    if args.unseen_games:
        for g in args.unseen_games:
            view_by_ratios(g, args)
    else:
        for name in args.ckpt_name:
            view_by_name(name, args)


if __name__ == "__main__":
    main()
