"""Transferability analysis package.

Analyze how mixing *source*-game data into MGPCGRL (``train_mgpcgrl``) training
changes *target*-game performance, and relate the observed gain/loss to the
statistical properties of each game's reward-condition distribution.

Modules
-------
config      : constants (paths, game list, reward-enum mapping, feature presence).
data        : load the experiment result table + per-game condition distributions.
distances   : directional source->target distribution distance / overlap features.
correlate   : merge distances with performance deltas and score hypotheses.
plots       : scatter / distribution figures.
run         : orchestrator that produces every table, figure and the report.
"""
