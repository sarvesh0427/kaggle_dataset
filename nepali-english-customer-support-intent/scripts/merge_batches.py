from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BATCH_DIR = PROJECT_ROOT / "data" / "raw" / "batches_repaired"
OUTPUT_DIR = PROJECT_ROOT / "data" / "merged"

OUTPUT_FILE = OUTPUT_DIR / "dataset_raw_10000.csv"


# ============================================================
# EXPECTED SCHEMA
# ============================================================

EXPECTED_COLUMNS = [
    "id",
    "text",
    "intent",
    "language",
    "is_code_switched",
    "formality",
    "difficulty",
]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("Nepali-English NLP Dataset - Batch Merger")
    print("=" * 60)

    # --------------------------------------------------------
    # Check input directory
    # --------------------------------------------------------

    if not BATCH_DIR.exists():
        raise FileNotFoundError(
            f"Batch directory not found:\n{BATCH_DIR}"
        )

    # --------------------------------------------------------
    # Find CSV files
    # --------------------------------------------------------

    files = sorted(BATCH_DIR.glob("*.csv"))

    print(f"\nBatch directory:")
    print(BATCH_DIR)

    print(f"\nCSV files found: {len(files)}")

    if not files:
        raise FileNotFoundError("No CSV files found.")

    # --------------------------------------------------------
    # Read each batch
    # --------------------------------------------------------

    dataframes = []

    for i, file in enumerate(files, start=1):

        print(f"\n[{i}/{len(files)}] Reading: {file.name}")

        df = pd.read_csv(file)

        print(f"    Rows: {len(df):,}")
        print(f"    Columns: {list(df.columns)}")

        # Check columns
        if list(df.columns) != EXPECTED_COLUMNS:
            raise ValueError(
                f"\nUnexpected columns in {file.name}\n"
                f"Expected: {EXPECTED_COLUMNS}\n"
                f"Found:    {list(df.columns)}"
            )

        dataframes.append(df)

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("Merging batches...")
    print("=" * 60)

    merged = pd.concat(
        dataframes,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Basic information
    # --------------------------------------------------------

    print(f"\nMerged rows: {len(merged):,}")
    print(f"Merged columns: {len(merged.columns)}")

    # --------------------------------------------------------
    # Convert ID to numeric
    # --------------------------------------------------------

    merged["id"] = pd.to_numeric(
        merged["id"],
        errors="raise"
    ).astype(int)

    # --------------------------------------------------------
    # Check duplicate IDs
    # --------------------------------------------------------

    duplicate_ids = merged.loc[
        merged["id"].duplicated(keep=False),
        "id"
    ].unique()

    if len(duplicate_ids) > 0:
        print("\nWARNING: Duplicate IDs found!")

        print(
            sorted(duplicate_ids.tolist())[:50]
        )

        raise ValueError(
            f"Found {len(duplicate_ids)} duplicate IDs."
        )

    # --------------------------------------------------------
    # Check expected ID range
    # --------------------------------------------------------

    expected_ids = set(range(1, 10001))
    actual_ids = set(merged["id"])

    missing_ids = sorted(
        expected_ids - actual_ids
    )

    extra_ids = sorted(
        actual_ids - expected_ids
    )

    if missing_ids:
        print("\nWARNING: Missing IDs:")
        print(missing_ids[:100])

    if extra_ids:
        print("\nWARNING: IDs outside 1-10000:")
        print(extra_ids[:100])

    # --------------------------------------------------------
    # Sort by ID
    # --------------------------------------------------------

    merged = merged.sort_values(
        "id"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Check row count
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("Validation")
    print("=" * 60)

    print(f"Rows:              {len(merged):,}")
    print(f"Unique IDs:        {merged['id'].nunique():,}")
    print(f"Minimum ID:        {merged['id'].min()}")
    print(f"Maximum ID:        {merged['id'].max()}")
    print(f"Missing IDs:       {len(missing_ids)}")
    print(f"Extra IDs:         {len(extra_ids)}")
    print(f"Duplicate IDs:     {len(duplicate_ids)}")

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    merged.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("MERGE COMPLETE")
    print("=" * 60)

    print(f"\nSaved to:")
    print(OUTPUT_FILE)

    print("\nFirst 5 rows:")
    print(merged.head().to_string(index=False))

    print("\nLast 5 rows:")
    print(merged.tail().to_string(index=False))


if __name__ == "__main__":
    main()