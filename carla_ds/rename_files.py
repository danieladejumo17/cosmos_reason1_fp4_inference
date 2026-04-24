#!/usr/bin/env python3
"""Rename files in camera_clean_individualframe subfolders:
  NORMAL_* -> Norm_*
  ANOMALY_* -> Anom_*
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).parent / "camera_clean_individualframe"
SUBFOLDERS = ["front", "left", "rear", "right"]

print(BASE_DIR)

PREFIX_MAP = {
    "NORMAL_": "Norm_",
    "ANOMALY_": "Anom_",
}

total_renamed = 0

for subfolder in SUBFOLDERS:
    folder = BASE_DIR / subfolder
    if not folder.is_dir():
        print(f"Skipping missing folder: {folder}")
        continue

    renamed = 0
    for filename in sorted(os.listdir(folder)):
        print(f"Renaming {filename}")
        for old_prefix, new_prefix in PREFIX_MAP.items():
            if filename.startswith(old_prefix):
                new_name = new_prefix + filename[len(old_prefix):]
                os.rename(folder / filename, folder / new_name)
                renamed += 1
                break

    print(f"{subfolder}: renamed {renamed} files")
    total_renamed += renamed

print(f"\nTotal renamed: {total_renamed}")
