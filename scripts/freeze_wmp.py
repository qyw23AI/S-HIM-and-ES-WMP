#!/usr/bin/env python3
"""Store a local WMP teacher and the tensors needed to verify exact inference."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graduation.paths import WMP


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Trusted local training checkpoint")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(WMP))
    from graduation.methods.wmp.freeze import freeze_checkpoint

    iteration = freeze_checkpoint(args.source, args.output)
    print(f"Frozen WMP from iteration {iteration} at {args.output}")


if __name__ == "__main__":
    main()
