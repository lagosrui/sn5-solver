"""Canonical Hone ARC entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from arc_inference_phase import run_inference_phase
from arc_prep_phase import run_prep_phase


def main() -> int:
    parser = argparse.ArgumentParser(description="SN5 Hone ARC solver entrypoint")
    parser.add_argument("--phase", choices=["prep", "inference"], required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.phase == "prep":
        run_prep_phase(args.input, args.output)
    else:
        run_inference_phase(args.input, args.output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"arc_main failed: {exc}", file=sys.stderr)
        raise
