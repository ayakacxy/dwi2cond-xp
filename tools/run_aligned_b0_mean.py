#!/usr/bin/env python3
"""Run the MCFLIRT-compatible b0 alignment and mean stage."""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from dwi2cond_xp.preprocessing import write_aligned_b0_mean


def main() -> int:
    """Run b0 alignment with visible per-volume progress."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dwi", type=Path, required=True)
    parser.add_argument("--bvals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qa", type=Path)
    parser.add_argument("--b0-threshold", type=float, default=50.0)
    parser.add_argument("--max-evaluations", type=int, default=240)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    with tqdm(total=None, unit="volume", desc="Rigid b0 alignment") as bar:
        previous = 0

        def update(done: int, total: int) -> None:
            nonlocal previous
            if bar.total is None:
                bar.total = total
            bar.update(done - previous)
            previous = done

        output = write_aligned_b0_mean(
            args.dwi,
            args.bvals,
            args.output,
            b0_threshold=args.b0_threshold,
            max_evaluations=args.max_evaluations,
            workers=args.workers,
            progress=update,
            qa_file=args.qa,
        )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
