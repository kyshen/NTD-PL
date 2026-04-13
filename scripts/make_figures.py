from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from viz.pipeline import build_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build NTD-PL paper figures from the new plotting pipeline.")
    parser.add_argument("--scope", choices=("all", "main", "appendix"), default="all")
    parser.add_argument("--figure-id", action="append", dest="figure_ids", help="Build only the selected figure id. Can be repeated.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for figure_id, paths in build_figures(
        scope=args.scope,
        figure_ids=args.figure_ids,
    ):
        print(f"[ok] {figure_id}")
        for path in paths:
            print(f"  -> {path}")


if __name__ == "__main__":
    main()
