"""
Count samples in CICIoV2024 dataset (raw per-class CSVs and train/val/test splits).

Usage (from project root):
  python scripts/count_ciciov_samples.py
  python scripts/count_ciciov_samples.py --source split --mode hexadecimal
  python scripts/count_ciciov_samples.py --by-class
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# =====================================================
# =========       Constants and options       =========
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "CICIoV2024"
SPLIT_DIR = PROJECT_ROOT / "data" / "CICIoV2024_split"

MODES = ["binary", "decimal", "hexadecimal"]
SPLITS = ["train", "val", "test"]
LABEL_COL = "specific_class"
BINARY_LABEL_COL = "label"


# =====================================================
# =========           Functions               =========
# =====================================================

def label_column(df: pd.DataFrame) -> str | None:
    if LABEL_COL in df.columns:
        return LABEL_COL
    if BINARY_LABEL_COL in df.columns:
        return BINARY_LABEL_COL
    return None


def read_csv_rows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df


def print_class_counts(df: pd.DataFrame, indent: str = "    ") -> None:
    col = label_column(df)
    if col is None:
        print(f"{indent}(no label column)")
        return
    for cls, cnt in df[col].value_counts().sort_index().items():
        print(f"{indent}{cls:20s}: {cnt:>10,}")


def count_raw_mode(mode: str, by_class: bool) -> int | None:
    mode_dir = RAW_DIR / mode
    if not mode_dir.is_dir():
        print(f"[SKIP] Raw folder not found: {mode_dir}")
        return None

    csv_files = sorted(mode_dir.glob("*.csv"))
    if not csv_files:
        print(f"[SKIP] No CSV files in {mode_dir}")
        return None

    print(f"\n{'=' * 60}")
    print(f"  RAW | mode: {mode}")
    print(f"  {mode_dir}")
    print(f"{'=' * 60}")

    total = 0
    for path in csv_files:
        df = read_csv_rows(path)
        n = len(df)
        total += n
        print(f"  {path.name:40s} {n:>10,} rows")
        if by_class:
            print_class_counts(df, indent="      ")

    print(f"\n  TOTAL (raw, {mode}): {total:,} rows")
    return total


def count_split_mode(mode: str, by_class: bool) -> dict[str, int] | None:
    mode_dir = SPLIT_DIR / mode
    if not mode_dir.is_dir():
        print(f"[SKIP] Split folder not found: {mode_dir}")
        return None

    print(f"\n{'=' * 60}")
    print(f"  SPLIT | mode: {mode}")
    print(f"  {mode_dir}")
    print(f"{'=' * 60}")

    counts: dict[str, int] = {}
    for split in SPLITS:
        path = mode_dir / f"{split}.csv"
        if not path.is_file():
            print(f"  [SKIP] {split}.csv not found")
            continue

        df = read_csv_rows(path)
        n = len(df)
        counts[split] = n
        print(f"\n  {split.upper():5s}: {n:>10,} rows  ({path.name})")
        if by_class:
            print_class_counts(df)

    if not counts:
        return None

    total = sum(counts.values())
    print(f"\n  TOTAL (split, {mode}): {total:,} rows")
    if total > 0 and set(counts) == set(SPLITS):
        train_pct = counts["train"] / total * 100
        val_pct = counts["val"] / total * 100
        test_pct = counts["test"] / total * 100
        print(
            f"  Ratios: train {train_pct:.2f} % | "
            f"val {val_pct:.2f} % | test {test_pct:.2f} %"
        )
    return counts


def modes_to_run(selected: str | None) -> list[str]:
    if selected:
        if selected not in MODES:
            raise ValueError(f"Unknown mode: {selected}. Choose from {MODES}")
        return [selected]
    return MODES


# =====================================================
# =========              Main                 =========
# =====================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count samples in CICIoV2024 raw and/or split datasets."
    )
    parser.add_argument(
        "--source",
        choices=["raw", "split", "both"],
        default="both",
        help="Count raw per-class CSVs, split train/val/test, or both (default: both).",
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        default=None,
        help="Single representation mode (default: all three).",
    )
    parser.add_argument(
        "--by-class",
        action="store_true",
        help="Print per-class counts (column specific_class or label).",
    )
    args = parser.parse_args()

    modes = modes_to_run(args.mode)
    print(f"Project root: {PROJECT_ROOT}")

    raw_totals: dict[str, int] = {}
    split_totals: dict[str, int] = {}

    for mode in modes:
        if args.source in ("raw", "both"):
            n = count_raw_mode(mode, args.by_class)
            if n is not None:
                raw_totals[mode] = n

        if args.source in ("split", "both"):
            counts = count_split_mode(mode, args.by_class)
            if counts is not None:
                split_totals[mode] = sum(counts.values())

    if len(modes) > 1 or args.source == "both":
        print(f"\n{'=' * 60}")
        print("  SUMMARY")
        print(f"{'=' * 60}")
        if raw_totals:
            for mode, n in raw_totals.items():
                print(f"  Raw total ({mode:12s}): {n:>12,}")
        if split_totals:
            for mode, n in split_totals.items():
                print(f"  Split total ({mode:12s}): {n:>12,}")

        if len(raw_totals) > 1:
            values = set(raw_totals.values())
            if len(values) == 1:
                print("\n  Raw counts match across binary / decimal / hexadecimal.")
            else:
                print("\n  [NOTE] Raw counts differ between modes (expected: same rows, different encoding).")

        if raw_totals and split_totals:
            for mode in set(raw_totals) & set(split_totals):
                if raw_totals[mode] != split_totals[mode]:
                    diff = split_totals[mode] - raw_totals[mode]
                    print(
                        f"\n  [NOTE] {mode}: split total ({split_totals[mode]:,}) "
                        f"!= raw total ({raw_totals[mode]:,}), diff {diff:+,}"
                    )


if __name__ == "__main__":
    main()
