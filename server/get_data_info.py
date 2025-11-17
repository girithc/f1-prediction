#!/usr/bin/env python3
"""
Preview the header and first data row of every CSV in a folder (default: ./data).
Usage:
  python preview_data.py [path/to/data]
"""

from __future__ import annotations
import csv
import sys
from pathlib import Path
from typing import Optional, Tuple

def read_header_and_first_row(csv_path: Path) -> Tuple[Optional[list], Optional[list]]:
    """
    Returns (header, first_row) for a CSV file.
    Header is the list of column names (from the first row).
    First_row is the list of values for the first data row (or None if none).
    """
    # Handle possible UTF-8 BOM and varied newlines
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        # Try to sniff the dialect (delimiter, quoting) from a small sample
        sample = f.read(2048)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel  # fall back to comma-delimited

        reader = csv.reader(f, dialect)
        try:
            header = next(reader)
        except StopIteration:
            return None, None  # empty file

        # If the first row doesn’t look like a header (rare), we still treat it as header,
        # since your files appear to be standard CSVs with headers.
        try:
            first_row = next(reader)
        except StopIteration:
            first_row = None

        return header, first_row

def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
    if not root.exists() or not root.is_dir():
        print(f"✗ Data directory not found: {root.resolve()}")
        sys.exit(1)

    csv_files = sorted(p for p in root.glob("*.csv") if p.is_file())
    if not csv_files:
        print(f"No CSV files found in {root.resolve()}")
        sys.exit(0)

    for i, csv_path in enumerate(csv_files, start=1):
        header, first_row = read_header_and_first_row(csv_path)
        print("=" * 80)
        print(f"[{i}] {csv_path.name}")
        if header is None:
            print("  (empty file)")
            continue

        # Show header
        print("  columns:")
        print("   ", ", ".join(header))

        # Show first row (or note if none)
        if first_row is None:
            print("  first row: (no data rows)")
        else:
            print("  first row:")
            # Pair values with their header for readability (handles missing/extra fields)
            max_len = max(len(header), len(first_row))
            cols = [(header[j] if j < len(header) else f"<extra_{j-len(header)+1}>",
                     first_row[j] if j < len(first_row) else "")
                    for j in range(max_len)]
            for col, val in cols:
                print(f"    - {col}: {val}")

    print("=" * 80)

if __name__ == "__main__":
    main()
