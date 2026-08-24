from pathlib import Path
import pandas as pd
from rapidfuzz.fuzz import ratio


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

REPORT_DIR = (
    PROJECT_ROOT
    / "data"
    / "reports"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

# Similarity >= 95:
# almost certainly duplicate/template variation

HIGH_THRESHOLD = 95

# Similarity 90-94:
# needs manual inspection

MEDIUM_THRESHOLD = 90

# Don't compare extremely short texts.
MIN_WORDS = 4


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("NEAR-DUPLICATE & REPETITION AUDIT")
print("=" * 70)

df = pd.read_csv(
    INPUT_FILE,
    encoding="utf-8"
)

print(f"\nLoaded rows: {len(df):,}")


# ============================================================
# NORMALIZE TEXT
# ============================================================

df["text_normalized"] = (
    df["text"]
    .fillna("")
    .astype(str)
    .str.lower()
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)

df["word_count"] = (
    df["text_normalized"]
    .str.split()
    .str.len()
)


# ============================================================
# REMOVE EXACT DUPLICATES FROM COMPARISON
# ============================================================

unique_df = (
    df[
        df["word_count"] >= MIN_WORDS
    ]
    .drop_duplicates(
        subset=["text_normalized"]
    )
    .reset_index(drop=True)
)

print(
    f"Unique texts being compared: "
    f"{len(unique_df):,}"
)


# ============================================================
# FIND NEAR DUPLICATES
# ============================================================

print("\nSearching for near duplicates...")
print("This may take some time.\n")


high_matches = []
medium_matches = []


for i in range(len(unique_df)):

    text_a = unique_df.loc[
        i,
        "text_normalized"
    ]

    for j in range(i + 1, len(unique_df)):

        text_b = unique_df.loc[
            j,
            "text_normalized"
        ]

        # ----------------------------------------------------
        # Quick length filter
        # ----------------------------------------------------

        len_a = len(text_a)
        len_b = len(text_b)

        if min(len_a, len_b) / max(len_a, len_b) < 0.65:
            continue

        # ----------------------------------------------------
        # Similarity
        # ----------------------------------------------------

        score = ratio(
            text_a,
            text_b
        )

        if score >= HIGH_THRESHOLD:

            high_matches.append({
                "id_1": unique_df.loc[i, "id"],
                "id_2": unique_df.loc[j, "id"],
                "text_1": unique_df.loc[i, "text"],
                "text_2": unique_df.loc[j, "text"],
                "intent_1": unique_df.loc[i, "intent"],
                "intent_2": unique_df.loc[j, "intent"],
                "language_1": unique_df.loc[i, "language"],
                "language_2": unique_df.loc[j, "language"],
                "similarity": round(score, 2),
            })

        elif score >= MEDIUM_THRESHOLD:

            medium_matches.append({
                "id_1": unique_df.loc[i, "id"],
                "id_2": unique_df.loc[j, "id"],
                "text_1": unique_df.loc[i, "text"],
                "text_2": unique_df.loc[j, "text"],
                "intent_1": unique_df.loc[i, "intent"],
                "intent_2": unique_df.loc[j, "intent"],
                "language_1": unique_df.loc[i, "language"],
                "language_2": unique_df.loc[j, "language"],
                "similarity": round(score, 2),
            })


# ============================================================
# SAVE HIGH SIMILARITY
# ============================================================

high_df = pd.DataFrame(
    high_matches
)

high_file = (
    REPORT_DIR
    / "near_duplicates_high_similarity.csv"
)

high_df.to_csv(
    high_file,
    index=False,
    encoding="utf-8"
)


# ============================================================
# SAVE MEDIUM SIMILARITY
# ============================================================

medium_df = pd.DataFrame(
    medium_matches
)

medium_file = (
    REPORT_DIR
    / "near_duplicates_medium_similarity.csv"
)

medium_df.to_csv(
    medium_file,
    index=False,
    encoding="utf-8"
)


# ============================================================
# CROSS-INTENT CONFLICTS
# ============================================================

print("\n" + "=" * 70)
print("CROSS-INTENT NEAR DUPLICATES")
print("=" * 70)

if len(high_df) > 0:

    cross_intent_high = high_df[
        high_df["intent_1"] != high_df["intent_2"]
    ]

    print(
        f"\nHigh-similarity cross-intent pairs: "
        f"{len(cross_intent_high):,}"
    )

    cross_intent_high.to_csv(
        REPORT_DIR
        / "near_duplicate_cross_intent_high.csv",
        index=False,
        encoding="utf-8"
    )

else:

    print("\nNo high-similarity pairs found.")


if len(medium_df) > 0:

    cross_intent_medium = medium_df[
        medium_df["intent_1"] != medium_df["intent_2"]
    ]

    print(
        f"Medium-similarity cross-intent pairs: "
        f"{len(cross_intent_medium):,}"
    )

    cross_intent_medium.to_csv(
        REPORT_DIR
        / "near_duplicate_cross_intent_medium.csv",
        index=False,
        encoding="utf-8"
    )

else:

    print("No medium-similarity pairs found.")


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("AUDIT SUMMARY")
print("=" * 70)

print(
    f"\nHigh similarity (>= {HIGH_THRESHOLD}%): "
    f"{len(high_df):,}"
)

print(
    f"Medium similarity ({MEDIUM_THRESHOLD}-{HIGH_THRESHOLD-1}%): "
    f"{len(medium_df):,}"
)

if len(high_df) > 0:

    print("\nTop high-similarity pairs:")

    print(
        high_df
        .sort_values(
            "similarity",
            ascending=False
        )
        .head(20)
        .to_string(index=False)
    )


print("\nReports saved to:")

print(
    f"  {high_file}"
)

print(
    f"  {medium_file}"
)

print("\nNo dataset records were modified.")

print("\nAudit complete.")