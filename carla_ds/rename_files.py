#!/usr/bin/env python3
"""Rename files in subfolders of a given directory:
  NORMAL_* -> Norm_*
  ANOMALY_* -> Anom_*
"""

import argparse
import os
from pathlib import Path

SUBFOLDERS = ["front", "left", "rear", "right"]

PREFIX_MAP = {
    "NORMAL_": "Norm_",
    "ANOMALY_": "Anom_",
}


def main():
    parser = argparse.ArgumentParser(description="Rename NORMAL_/ANOMALY_ prefixes to Norm_/Anom_")
    parser.add_argument("source", type=str, help="Root directory containing front/, left/, rear/, right/ subfolders")
    args = parser.parse_args()

    base_dir = Path(args.source)
    if not base_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {base_dir}")

    print(f"Source: {base_dir}")
    total_renamed = 0

    for subfolder in SUBFOLDERS:
        folder = base_dir / subfolder
        if not folder.is_dir():
            print(f"Skipping missing folder: {folder}")
            continue

        renamed = 0
        for filename in sorted(os.listdir(folder)):
            for old_prefix, new_prefix in PREFIX_MAP.items():
                if filename.startswith(old_prefix):
                    new_name = new_prefix + filename[len(old_prefix):]
                    os.rename(folder / filename, folder / new_name)
                    renamed += 1
                    break

        print(f"{subfolder}: renamed {renamed} files")
        total_renamed += renamed

    print(f"\nTotal renamed: {total_renamed}")


if __name__ == "__main__":
    main()
