from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "merged"
    / "dataset_raw_10000.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "data"
    / "reports"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "dataset_clean_no_exact_duplicates.csv"
)


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("EXACT DUPLICATE CLEANING")
print("=" * 70)

df = pd.read_csv(
    INPUT_FILE,
    encoding="utf-8"
)

original_rows = len(df)

print(f"\nOriginal rows: {original_rows:,}")


# ============================================================
# NORMALIZE TEXT FOR DUPLICATE DETECTION
# ============================================================

df["_text_normalized"] = (
    df["text"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
)


# ============================================================
# FIND DUPLICATES
# ============================================================

duplicate_mask = (
    df["_text_normalized"]
    .duplicated(keep="first")
)

duplicate_rows = df[duplicate_mask].copy()

print(
    f"Exact duplicate rows to remove: "
    f"{len(duplicate_rows):,}"
)


# ============================================================
# SAVE REMOVED RECORDS
# ============================================================

removed_file = (
    REPORT_DIR
    / "removed_exact_duplicates.csv"
)

duplicate_rows[
    [
        "id",
        "text",
        "intent",
        "language",
        "is_code_switched",
        "formality",
        "difficulty",
    ]
].to_csv(
    removed_file,
    index=False,
    encoding="utf-8"
)


# ============================================================
# REMOVE DUPLICATES
# ============================================================

clean_df = df[
    ~duplicate_mask
].copy()


# ============================================================
# REMOVE INTERNAL COLUMN
# ============================================================

clean_df.drop(
    columns=["_text_normalized"],
    inplace=True
)


# ============================================================
# RESET INDEX
# ============================================================

clean_df.reset_index(
    drop=True,
    inplace=True
)


# ============================================================
# SAVE CLEAN DATASET
# ============================================================

clean_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)


# ============================================================
# SUMMARY
# ============================================================

final_rows = len(clean_df)

print("\n" + "=" * 70)
print("CLEANING SUMMARY")
print("=" * 70)

print(
    f"\nOriginal rows:       {original_rows:,}"
)

print(
    f"Removed duplicates:  {original_rows - final_rows:,}"
)

print(
    f"Final rows:          {final_rows:,}"
)

print(
    f"Rows retained:       "
    f"{final_rows / original_rows * 100:.2f}%"
)

print(
    f"\nClean dataset:"
)

print(
    OUTPUT_FILE
)

print(
    f"\nRemoved records report:"
)

print(
    removed_file
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("POST-CLEANING VALIDATION")
print("=" * 70)

remaining_duplicate_text = (
    clean_df["text"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .duplicated()
    .sum()
)

duplicate_ids = (
    clean_df["id"]
    .duplicated()
    .sum()
)

missing_ids = clean_df["id"].isna().sum()

print(
    f"Remaining duplicate texts: {remaining_duplicate_text}"
)

print(
    f"Duplicate IDs:              {duplicate_ids}"
)

print(
    f"Missing IDs:                {missing_ids}"
)


# ============================================================
# FINAL STATUS
# ============================================================

if (
    remaining_duplicate_text == 0
    and duplicate_ids == 0
    and missing_ids == 0
):

    print(
        "\nSTATUS: PASS"
    )

else:

    print(
        "\nSTATUS: CHECK REQUIRED"
    )


print("\nNo changes were made to the original raw dataset.")
print("Cleaning complete.")