from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "results" / "eval"

DATA = [
    (
        "aaai27_eval_cpcgrl/cpcgrl_game-maze_re-0/eval_re-0",
        "progress,0.832\nvit_score,0.741\ntpkldiv,1.230\ndiversity,0.950",
        "maze,0,0.84,0.73,1.21,0.96\nmaze,0,0.82,0.75,1.24,0.94",
    ),
    (
        "aaai27_eval_cpcgrl/cpcgrl_game-maze_re-1/eval_re-1",
        "progress,0.761\nvit_score,0.689\ntpkldiv,1.102\ndiversity,0.877",
        "maze,1,0.77,0.69,1.10,0.88\nmaze,1,0.75,0.68,1.11,0.87",
    ),
    (
        "aaai27_eval_cpcgrl_all/cpcgrl_game-all_re-0/eval_re-0",
        "progress,0.801\nvit_score,0.712\ntpkldiv,1.180\ndiversity,0.910",
        "maze,0,0.80,0.71,1.18,0.91\ndungeon,0,0.80,0.72,1.17,0.90",
    ),
    (
        "aaai27_eval_cpcgrl_all/cpcgrl_game-all_re-1/eval_re-1",
        "progress,0.745\nvit_score,0.670\ntpkldiv,1.050\ndiversity,0.860",
        "maze,1,0.75,0.68,1.07,0.87\ndungeon,1,0.74,0.65,1.04,0.85",
    ),
]

SUMMARY_HEADER = "metric,mean"
RESULT_HEADER = "game,reward_enum,progress,vit_score,tpkldiv,diversity"

for rel, summary_body, result_body in DATA:
    d = ROOT / rel
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.csv").write_text(SUMMARY_HEADER + "\n" + summary_body + "\n")
    (d / "results.csv").write_text(RESULT_HEADER + "\n" + result_body + "\n")
    print(f"[OK] {d}")

print("Test data generation complete")

