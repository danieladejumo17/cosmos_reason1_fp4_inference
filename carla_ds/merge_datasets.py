#!/usr/bin/env python3
"""Merge dataset_output_* folders into a single dataset_output folder.

Files are copied with re-indexed names so that indices are globally unique
and contiguous across the merged dataset. The offset for each source folder
is derived from the maximum index of all previously merged folders + 1.
"""

import re
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "dataset_output"
SUBFOLDERS = ("front", "left", "right", "rear")
FILE_RE = re.compile(r"^(ANOMALY|NORMAL)_(\d+)(\.jpg)$", re.IGNORECASE)


def numeric_suffix(dirname: str) -> int:
    return int(dirname.rsplit("_", 1)[-1])


def gather_source_dirs() -> list[Path]:
    dirs = sorted(
        (d for d in BASE_DIR.iterdir()
         if d.is_dir() and d.name.startswith("dataset_output_")),
        key=lambda p: numeric_suffix(p.name),
    )
    return dirs


def max_index_in_dir(src: Path) -> int:
    """Return the highest file index found across all subfolders."""
    hi = -1
    for sub in SUBFOLDERS:
        sub_dir = src / sub
        if not sub_dir.is_dir():
            continue
        for f in sub_dir.iterdir():
            m = FILE_RE.match(f.name)
            if m:
                hi = max(hi, int(m.group(2)))
    return hi


def merge_one(src: Path, offset: int) -> int:
    """Copy files from src into OUTPUT_DIR with indices shifted by offset.

    Returns the highest new index written.
    """
    hi_new = -1
    copied = 0

    for sub in SUBFOLDERS:
        src_sub = src / sub
        dst_sub = OUTPUT_DIR / sub
        if not src_sub.is_dir():
            print(f"  WARNING: {src_sub} not found, skipping.")
            continue
        dst_sub.mkdir(parents=True, exist_ok=True)

        for f in sorted(src_sub.iterdir()):
            m = FILE_RE.match(f.name)
            if not m:
                continue
            label, idx_str, ext = m.groups()
            new_idx = int(idx_str) + offset
            new_name = f"{label}_{new_idx:04d}{ext}"
            shutil.copy2(f, dst_sub / new_name)
            hi_new = max(hi_new, new_idx)
            copied += 1

    print(f"  Copied {copied} files (offset={offset}, max new index={hi_new})")
    return hi_new


def main():
    source_dirs = gather_source_dirs()
    if not source_dirs:
        print("No dataset_output_* folders found.")
        return

    print(f"Found {len(source_dirs)} source folders: "
          f"{[d.name for d in source_dirs]}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    offset = 0
    for src in source_dirs:
        print(f"{src.name}  (offset {offset}):")
        hi = merge_one(src, offset)
        offset = hi + 1

    total = sum(
        len(list((OUTPUT_DIR / sub).iterdir()))
        for sub in SUBFOLDERS
        if (OUTPUT_DIR / sub).is_dir()
    )
    print(f"\nDone. Merged {total} total files into {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
