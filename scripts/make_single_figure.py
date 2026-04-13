from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from viz.pipeline import build_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a single figure from the NTD-PL plotting pipeline.")
    parser.add_argument("figure_id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = build_figure(args.figure_id)
    print(f"[ok] {args.figure_id}")
    for path in paths:
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
