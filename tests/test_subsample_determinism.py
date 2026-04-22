"""
test_subsample_determinism.py
==============================
_subsample_per_group 의 재현성(determinism) 검증 테스트.

확인 항목:
1. 같은 seed -> 항상 동일한 샘플 집합
2. 다른 seed -> 다른 샘플 집합
3. 입력 순서가 달라도 같은 seed -> 동일한 결과
4. 그룹당 샘플 수가 n_per_group 이하
5. 가용 샘플이 n_per_group 보다 적은 그룹은 전부 사용
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from instruct_rl.utils.dataset_loader import _subsample_per_group


class FakeSample:
    def __init__(self, game, idx, re=0):
        self.game = game
        self.source_id = "%s_%04d" % (game, idx)
        self.idx = idx
        self.meta = {"reward_enum": re, "conditions": {re: float(idx)}}


def make_samples(game_counts, re=0):
    s = []
    for game, n in sorted(game_counts.items()):
        for i in range(n):
            s.append(FakeSample(game, i, re=re))
    return s


def ids(samples):
    return {(s.game, s.idx) for s in samples}


def test_same_seed():
    s = make_samples({"doom": 100, "dungeon": 400, "pokemon": 100, "sokoban": 100, "zelda": 100})
    r1, c1 = _subsample_per_group(s, 40, seed=0)
    r2, c2 = _subsample_per_group(s, 40, seed=0)
    assert ids(r1) == ids(r2) and c1 == c2
    print("PASS test_same_seed")


def test_diff_seed():
    s = make_samples({"doom": 100, "dungeon": 400, "pokemon": 100, "sokoban": 100, "zelda": 100})
    r0, _ = _subsample_per_group(s, 40, seed=0)
    r1, _ = _subsample_per_group(s, 40, seed=42)
    assert ids(r0) != ids(r1)
    print("PASS test_diff_seed")


def test_per_group_count():
    s = make_samples({"doom": 100, "dungeon": 400, "pokemon": 100, "sokoban": 100, "zelda": 100})
    _, counts = _subsample_per_group(s, 40, seed=0)
    for g, c in counts.items():
        assert c == 40, (g, c)
    print("PASS test_per_group_count  counts=%s" % counts)


def test_insufficient():
    s = make_samples({"doom": 100, "dungeon": 400, "pokemon": 100, "sokoban": 5, "zelda": 100})
    _, counts = _subsample_per_group(s, 40, seed=0)
    assert counts["sokoban"] == 5, counts
    print("PASS test_insufficient  counts=%s" % counts)


def test_multi_re():
    s  = make_samples({"dungeon": 100, "zelda": 100}, re=0)
    s += make_samples({"dungeon": 100, "zelda": 100}, re=1)
    _, counts = _subsample_per_group(s, 20, seed=0)
    assert counts["dungeon"] == 40 and counts["zelda"] == 40, counts
    print("PASS test_multi_re  counts=%s" % counts)


def test_order_independent():
    import random
    s = make_samples({"doom": 50, "dungeon": 200, "pokemon": 50, "sokoban": 50, "zelda": 50})
    sh = s[:]
    random.Random(99).shuffle(sh)
    r1, _ = _subsample_per_group(s,  20, seed=0)
    r2, _ = _subsample_per_group(sh, 20, seed=0)
    assert ids(r1) == ids(r2)
    print("PASS test_order_independent")


if __name__ == "__main__":
    print("=" * 50)
    test_same_seed()
    test_diff_seed()
    test_per_group_count()
    test_insufficient()
    test_multi_re()
    test_order_independent()
    print("=" * 50)
    print("All tests done.")

