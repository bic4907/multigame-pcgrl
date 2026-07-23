"""Orchestrator for the transferability analysis figure.

Run from the repo root:

    python -m results.transferbility.analysis.run

Produces the similarity boxplot (PNG + PDF) in a timestamped subfolder under
``results/transferbility/output/<YYYYMMDD-HHMMSS>/``.
"""
from __future__ import annotations

from datetime import datetime

from . import config, plots


def main() -> None:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = config.OUTPUT_DIR / stamp
    out.mkdir(parents=True, exist_ok=True)
    fig = plots.plot_similarity_boxplot(out)
    print(f"Analysis complete. Outputs in: {out}")
    print(f"  figure: {fig.name} (+ .pdf)")


if __name__ == "__main__":
    main()
