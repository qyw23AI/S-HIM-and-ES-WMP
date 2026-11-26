#!/usr/bin/env python3
"""Summarize matched A1 runs and compute a per-condition performance/energy Pareto front."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graduation.evaluation.pareto import NoUsableReports, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+", help="Paths or folders containing visual-suite reports")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        summarize(args.reports, args.out)
    except NoUsableReports as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
