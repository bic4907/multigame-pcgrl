"""
debug/inspect_ckpt.py
======================
체크포인트의 파라미터 구조와 shape을 출력하는 유틸.

Usage:
    python debug/inspect_ckpt.py <ckpt_path>
    python debug/inspect_ckpt.py pretrained_encoders/vipcgrl/default/ckpts
    python debug/inspect_ckpt.py pretrained_encoders/vipcgrl/default/ckpts/30
"""
import os
import sys
from pathlib import Path


def print_param_tree(params, prefix='', max_depth=5, current_depth=0):
    """파라미터 트리를 재귀적으로 출력"""
    if current_depth >= max_depth:
        print(f'{prefix}...')
        return

    if isinstance(params, dict):
        for k, v in sorted(params.items()):
            if isinstance(v, dict):
                print(f'{prefix}{k}/')
                print_param_tree(v, prefix + '  ', max_depth, current_depth + 1)
            else:
                # array
                shape = getattr(v, 'shape', '?')
                dtype = getattr(v, 'dtype', '?')
                print(f'{prefix}{k}: shape={shape}, dtype={dtype}')
    else:
        shape = getattr(params, 'shape', '?')
        dtype = getattr(params, 'dtype', '?')
        print(f'{prefix}value: shape={shape}, dtype={dtype}')


def inspect_checkpoint(ckpt_path: str, max_depth: int = 5):
    """체크포인트를 로드하고 구조를 출력"""
    from flax.training import checkpoints

    ckpt_path = os.path.abspath(ckpt_path)

    # ckpts 폴더면 가장 높은 step 선택
    if os.path.isdir(ckpt_path):
        subdirs = [d for d in os.listdir(ckpt_path) if d.isdigit()]
        if subdirs:
            latest = max(subdirs, key=int)
            ckpt_path = os.path.join(ckpt_path, latest)
            print(f"Auto-selected latest step: {latest}")

    print(f"Loading checkpoint from: {ckpt_path}")
    print("=" * 80)

    enc_state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_path, target=None, prefix='')

    if enc_state is None:
        print("ERROR: Failed to load checkpoint (returned None)")
        return

    print(f"Top-level keys: {list(enc_state.keys())}")
    print()

    if 'params' in enc_state:
        print("=== params structure ===")
        print_param_tree(enc_state['params'], max_depth=max_depth)
    else:
        print("=== full state structure ===")
        print_param_tree(enc_state, max_depth=max_depth)

    print()
    print("=" * 80)

    # 총 파라미터 수 계산
    def count_params(params):
        total = 0
        if isinstance(params, dict):
            for v in params.values():
                total += count_params(v)
        else:
            total += getattr(params, 'size', 0)
        return total

    if 'params' in enc_state:
        total = count_params(enc_state['params'])
        print(f"Total parameters: {total:,}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    ckpt_path = sys.argv[1]
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    inspect_checkpoint(ckpt_path, max_depth)

