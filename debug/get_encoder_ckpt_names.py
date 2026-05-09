"""
실제 get_exp_group / get_exp_name 로직을 그대로 사용해
train_mgpcgrl_encoder_unseen_ratios.yaml 의 모든 (unseen_games, seen_ratio) 조합에 대한
실제 폴더명을 출력한다.
"""

import sys
import os

# 프로젝트 루트를 sys.path에 추가
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from encoder.utils.path import get_exp_group, get_exp_name


# ── 최소 config mock ──────────────────────────────────────────────────────────
class MockEncoderConfig:
    model = "cnnclip"
    state = True


class MockConfig:
    # 고정값
    game = "all"
    exp_name = "def"
    seed = 0
    dir_prefix = "clipdec-"
    text_ratio = 1.0
    state_ratio = 1.0
    unseen_ratio = 0.0   # encoder 학습 시 unseen 게임 데이터 비율 (0.0 = zero-shot)
    encoder = MockEncoderConfig()

    def __init__(self, unseen_games, seen_ratio):
        self.unseen_games = unseen_games
        self.seen_ratio = seen_ratio


# ── sweep 파라미터 ─────────────────────────────────────────────────────────────
SEEN_RATIOS = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]

UNSEEN_GAMES = [
    # 1 unseen game
    "zd", "dm", "sk",
    # 2 unseen games
    "zddm", "zdsk", "dmsk",
    # 3 unseen games
    "pkzddm", "dgzddm", "zddmsk",
    # 4 unseen games
    "pkdgzddm", "pkzddmsk", "dgzddmsk",
]


# ── 출력 ──────────────────────────────────────────────────────────────────────
print("=" * 80)
print("실제 인코더 체크포인트 폴더명")
print("=" * 80)

results = []
for sr in SEEN_RATIOS:
    print(f"\n# ── seen_ratio: {sr} {'(sr 생략)' if sr == 1.0 else f'(sr-{sr})'} {'─' * 50}")
    for ug in UNSEEN_GAMES:
        cfg = MockConfig(unseen_games=ug, seen_ratio=sr)
        name = get_exp_name(cfg)
        print(f'      "{name}",')
        results.append((sr, ug, name))

# YAML values 형식으로도 출력
print("\n" + "=" * 80)
print("YAML encoder.ckpt_name values 형식:")
print("=" * 80)
print("  encoder.ckpt_name:")
print("    values: [")
for sr in SEEN_RATIOS:
    n_label = {1: "1 unseen", 2: "2 unseen", 3: "3 unseen", 4: "4 unseen"}
    print(f"      # ── seen_ratio: {sr} {'(sr 생략)' if sr == 1.0 else '(sr-%s)' % sr} {'─' * 40}")
    for ug in UNSEEN_GAMES:
        cfg = MockConfig(unseen_games=ug, seen_ratio=sr)
        name = get_exp_name(cfg)
        print(f'      "{name}",')
print("    ]")

