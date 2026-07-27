"""
tpkl_utils/__init__.py
======================
TPKL utility public API.

Submodule structure
------------
patch.py        -- internal patch extraction and hashing helpers
distribution.py — GT distribution compute  (build_gt_distribution)
scoring.py      -- JSD scoring (compute_jsd_scores)
dataset.py      — GT level load  (load_gt_levels)
task.py         -- task keys and grouping (quantize_condition, build_task_key,
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
