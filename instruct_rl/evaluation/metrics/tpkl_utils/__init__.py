"""
tpkl_utils/__init__.py
======================
TPKL utility public API.

text structure
------------
patch.py        — text extract / text internal text
distribution.py — GT distribution compute  (build_gt_distribution)
scoring.py      — JSD text  (compute_jsd_scores)
dataset.py      — GT level load  (load_gt_levels)
task.py         — text text / text (quantize_condition, build_task_key,
                                      group_states_by_task)
"""
from instruct_rl.evaluation.metrics.tpkl_utils.distribution import (  # noqa: F401
    build_gt_distribution,
)
from instruct_rl.evaluation.metrics.tpkl_utils.scoring import (  # noqa: F401
    compute_jsd_scores,
)
from instruct_rl.evaluation.metrics.tpkl_utils.dataset import (  # noqa: F401
    load_gt_levels,
)
from instruct_rl.evaluation.metrics.tpkl_utils.task import (  # noqa: F401
    quantize_condition,
    build_task_key,
    group_states_by_task,
)

