#!/usr/bin/env python3
"""Compute a direct unaligned b0 mean without splitting the 4D DWI."""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from dwi2cond_xp.preprocessing import write_unaligned_b0_mean


def main() -> int:
    """Run the blockwise b0-mean stage with visible z-slice progress."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dwi", type=Path, required=True)
    parser.add_argument("--bvals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qa", type=Path)
    parser.add_argument("--b0-threshold", type=float, default=50.0)
    parser.add_argument("--z-chunk", type=int, default=8)
    args = parser.parse_args()

    with tqdm(total=None, unit="slice", desc="Direct b0 mean") as bar:
        previous = 0

        def update(done: int, total: int) -> None:
            nonlocal previous
            if bar.total is None:
                bar.total = total
            bar.update(done - previous)
            previous = done

        output = write_unaligned_b0_mean(
            args.dwi,
            args.bvals,
            args.output,
            b0_threshold=args.b0_threshold,
            z_chunk=args.z_chunk,
            progress=update,
            qa_file=args.qa,
        )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
