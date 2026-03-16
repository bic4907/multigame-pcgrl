import os
import logging
from os.path import abspath, dirname, join, basename
from collections import Counter
import warnings
import numpy as np
import concurrent.futures
from tqdm import tqdm

from instruct_rl.evaluation.metrics.base import BaseEvaluator


# ── logger setting ────────────────────────────────────────────────
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))


# ── util funcs ────────────────────────────────────────────────────
def _sliding_windows(level: np.ndarray, k: int):
    h, w = level.shape[:2]
    for i in range(h - k + 1):
        for j in range(w - k + 1):
            values = level[i: i + k, j: j + k].flatten()
            values = [int(v) for v in values]
            yield tuple(values)


def _get_distribution(levels: np.ndarray, window_sizes, epsilon):
    dists = []
    for k in window_sizes:
        counts, total = Counter(), 0
        for lvl in levels:
            for key in _sliding_windows(lvl, k):
                counts[key] += 1
                total += 1
        # smoothing
        smoothed = {k_: (v + epsilon) for k_, v in counts.items()}
        norm = sum(smoothed.values())
        dists.append({k_: v / norm for k_, v in smoothed.items()})
    return dists

def _kl(p: dict, q: dict, eps):
    return sum(pv * np.log(pv / q.get(k, eps)) for k, pv in p.items())

def evaluate_task(task_key, levels, task_ids, gt_patterns, window_sizes, eps):
    if task_key not in gt_patterns:
        warnings.warn(f"[TPKL] Task {task_key} not in GT set")
        return [], []

    idx = np.where(task_ids == int(task_key))[0]
    pred_lvls = levels[idx]
    gt_dists = gt_patterns[task_key]  # list of dicts per window size

    level_scores = []
    for lvl in pred_lvls:
        indiv_dists = _get_distribution([lvl], window_sizes, eps)
        indiv_kl = sum(
            0.5 * _kl(p, q, eps) + 0.5 * _kl(q, p, eps)
            for p, q in zip(indiv_dists, gt_dists)
        )
        level_scores.append(indiv_kl)

    return idx, np.array(level_scores)


# ── main Evaluator class ─────────────────────────────────────────
class TPKLEvaluator(BaseEvaluator):
    def preload(self, window_sizes=(2,3,),
                gt_data_name="human_20250630_213109.legacy.npz",
                epsilon=1e-6):
        self.window_sizes, self.epsilon = window_sizes, epsilon
        self.gt_path = abspath(join(dirname(__file__), "human_data", gt_data_name))

        logger.info(f"[TPKL] Loading GT levels from {self.gt_path}")
        raw = np.load(self.gt_path, allow_pickle=True)

        self.gt_patterns = {
            k: _get_distribution(v, window_sizes, epsilon) for k, v in raw.items()
        }
        logger.info("[TPKL] Finished computing GT distributions")

    def run(self, levels: np.ndarray, task_ids: np.ndarray,
            num_workers: int = 4, show_progress: bool = False):
        assert levels.shape[0] == task_ids.shape[0], f"levels/task_ids mismatch (levels: {levels.shape}, task_ids: {task_ids.shape})"
        scores = np.zeros(levels.shape[0])
        unique_tasks = np.unique(task_ids)

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = [
                ex.submit(
                    evaluate_task, str(t), levels, task_ids,
                    self.gt_patterns, self.window_sizes, self.epsilon
                )
                for t in unique_tasks
            ]

            # tqdm progress bar
            iterator = tqdm(concurrent.futures.as_completed(futures),
                            total=len(futures),
                            desc="TPKL-Div Processed Tasks",
                            disable=not show_progress)
            for fut in iterator:
                idx, task_scores = fut.result()
                if len(idx):
                    scores[idx] = task_scores
        return scores


# ── test code ───────────────────────────────────────────────────
if __name__ == "__main__":
    data_path = abspath(join(dirname(__file__),
                             "human_data", "human_20250630_213109.legacy.npz"))
    data = np.load(data_path, allow_pickle=True)

    levels = data["1"]
    task_ids = list(range(0, 40)) * 100
    task_ids = np.array(task_ids[:len(levels)])  # Ensure task_ids matches levels length

    evaluator = TPKLEvaluator()
    evaluator.preload(gt_data_name="human_20250630_213109.legacy.npz")

    levels = np.clip(levels + np.random.randint(-1, 1, size=levels.shape), 0, 3)

    scores = evaluator.run(levels, task_ids, num_workers=4, show_progress=True)
    print(f"Scores shape: {scores.shape}, mean: {scores.mean():.6f}, std: {scores.std():.6f}")
