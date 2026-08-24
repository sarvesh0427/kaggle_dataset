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

REPORT_DIR = (
    PROJECT_ROOT
    / "data"
    / "reports"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("DUPLICATE & CONFLICT AUDIT")
print("=" * 70)

df = pd.read_csv(
    INPUT_FILE,
    encoding="utf-8"
)

print(f"\nLoaded: {len(df):,} rows")


# ============================================================
# NORMALIZED TEXT
# ============================================================

df["text_normalized"] = (
    df["text"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
)


# ============================================================
# 1. EXACT DUPLICATE TEXT
# ============================================================

print("\n" + "=" * 70)
print("1. DUPLICATE TEXT ANALYSIS")
print("=" * 70)

text_counts = (
    df["text_normalized"]
    .value_counts()
)

duplicate_texts = text_counts[
    text_counts > 1
]

print(
    f"\nUnique texts: "
    f"{df['text_normalized'].nunique():,}"
)

print(
    f"Texts appearing more than once: "
    f"{len(duplicate_texts):,}"
)

print(
    f"Rows involved in duplicate groups: "
    f"{df['text_normalized'].isin(duplicate_texts.index).sum():,}"
)


# ============================================================
# 2. SAME TEXT + SAME INTENT
# ============================================================

print("\n" + "=" * 70)
print("2. SAME TEXT + SAME INTENT")
print("=" * 70)

same_intent = (
    df[
        df["text_normalized"].isin(
            duplicate_texts.index
        )
    ]
    .groupby("text_normalized")["intent"]
    .nunique()
)

same_intent_groups = same_intent[
    same_intent == 1
]

print(
    f"\nDuplicate groups with SAME intent: "
    f"{len(same_intent_groups):,}"
)


# ============================================================
# 3. SAME TEXT + DIFFERENT INTENT
# ============================================================

print("\n" + "=" * 70)
print("3. SAME TEXT + DIFFERENT INTENT")
print("=" * 70)

conflicting_groups = same_intent[
    same_intent > 1
]

print(
    f"\nPotential label conflicts: "
    f"{len(conflicting_groups):,}"
)

if len(conflicting_groups) > 0:

    conflict_df = (
        df[
            df["text_normalized"].isin(
                conflicting_groups.index
            )
        ]
        .sort_values("text_normalized")
    )

    conflict_output = (
        REPORT_DIR
        / "duplicate_label_conflicts.csv"
    )

    conflict_df[
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
        conflict_output,
        index=False,
        encoding="utf-8"
    )

    print(
        f"Saved conflicts to:\n"
        f"{conflict_output}"
    )


# ============================================================
# 4. DUPLICATE TEXT GROUPS
# ============================================================

print("\n" + "=" * 70)
print("4. DUPLICATE GROUP DETAILS")
print("=" * 70)

duplicate_df = (
    df[
        df["text_normalized"].isin(
            duplicate_texts.index
        )
    ]
    .sort_values("text_normalized")
)

duplicate_output = (
    REPORT_DIR
    / "duplicate_text_groups.csv"
)

duplicate_df[
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
    duplicate_output,
    index=False,
    encoding="utf-8"
)

print(
    f"Saved duplicate groups to:\n"
    f"{duplicate_output}"
)


# ============================================================
# 5. INTENT × LANGUAGE
# ============================================================

print("\n" + "=" * 70)
print("5. INTENT × LANGUAGE")
print("=" * 70)

intent_language = pd.crosstab(
    df["intent"],
    df["language"]
)

print(intent_language)


intent_language.to_csv(
    REPORT_DIR / "intent_language_distribution.csv"
)


# ============================================================
# 6. INTENT × DIFFICULTY
# ============================================================

print("\n" + "=" * 70)
print("6. INTENT × DIFFICULTY")
print("=" * 70)

intent_difficulty = pd.crosstab(
    df["intent"],
    df["difficulty"]
)

print(intent_difficulty)


intent_difficulty.to_csv(
    REPORT_DIR / "intent_difficulty_distribution.csv"
)


# ============================================================
# 7. LANGUAGE × DIFFICULTY
# ============================================================

print("\n" + "=" * 70)
print("7. LANGUAGE × DIFFICULTY")
print("=" * 70)

language_difficulty = pd.crosstab(
    df["language"],
    df["difficulty"]
)

print(language_difficulty)


language_difficulty.to_csv(
    REPORT_DIR / "language_difficulty_distribution.csv"
)


# ============================================================
# 8. FORMALITY × LANGUAGE
# ============================================================

print("\n" + "=" * 70)
print("8. FORMALITY × LANGUAGE")
print("=" * 70)

formality_language = pd.crosstab(
    df["formality"],
    df["language"]
)

print(formality_language)


formality_language.to_csv(
    REPORT_DIR / "formality_language_distribution.csv"
)


# ============================================================
# 9. DUPLICATE EXAMPLES
# ============================================================

print("\n" + "=" * 70)
print("9. SAMPLE DUPLICATES")
print("=" * 70)

if len(duplicate_texts) > 0:

    for text, count in duplicate_texts.head(20).items():

        print(
            f"\n[{count} occurrences]"
        )

        examples = df[
            df["text_normalized"] == text
        ][
            [
                "id",
                "text",
                "intent",
                "language",
                "difficulty",
            ]
        ]

        print(
            examples.to_string(
                index=False
            )
        )


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("AUDIT SUMMARY")
print("=" * 70)

print(
    f"\nTotal records:                  {len(df):,}"
)

print(
    f"Unique normalized texts:       "
    f"{df['text_normalized'].nunique():,}"
)

print(
    f"Duplicate text groups:         "
    f"{len(duplicate_texts):,}"
)

print(
    f"Same-intent duplicate groups:  "
    f"{len(same_intent_groups):,}"
)

print(
    f"Conflicting duplicate groups:  "
    f"{len(conflicting_groups):,}"
)

print("\nReports saved to:")
print(REPORT_DIR)

print("\nNo dataset records were modified.")


# ============================================================
# END
# ============================================================

print("\nAudit complete.")