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


EXPECTED_INTENTS = [
    "order_tracking",
    "order_status",
    "order_cancellation",
    "delayed_delivery",
    "wrong_item",
    "damaged_item",
    "missing_item",
    "return_request",
    "refund_request",
    "refund_status",
    "payment_failed",
    "payment_pending",
    "payment_reversed",
    "duplicate_payment",
    "account_login",
    "password_reset",
    "account_locked",
    "product_information",
    "product_availability",
    "human_agent_request",
]


EXPECTED_LANGUAGES = [
    "english",
    "nepali_devanagari",
    "romanized_nepali",
    "code_switched",
]


EXPECTED_FORMALITY = [
    "formal",
    "neutral",
    "informal",
]


EXPECTED_DIFFICULTY = [
    "easy",
    "medium",
    "hard",
]


# ============================================================
# HELPERS
# ============================================================

def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def show_distribution(series, name):
    print(f"\n{name}:")
    counts = series.value_counts(dropna=False)

    percentages = (
        series.value_counts(
            normalize=True,
            dropna=False
        ) * 100
    )

    for value in counts.index:
        print(
            f"  {str(value):25s}"
            f"{counts[value]:6,d}"
            f"  ({percentages[value]:6.2f}%)"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("Nepali-English NLP Dataset - Quality Profile")
    print("=" * 70)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        encoding="utf-8"
    )

    print_section("BASIC INFORMATION")

    print(f"File:              {INPUT_FILE}")
    print(f"Rows:              {len(df):,}")
    print(f"Columns:           {len(df.columns)}")
    print(f"Memory usage:      {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # --------------------------------------------------------
    # Schema
    # --------------------------------------------------------

    print_section("SCHEMA CHECK")

    print("Expected columns:")
    print(EXPECTED_COLUMNS)

    print("\nActual columns:")
    print(df.columns.tolist())

    missing_columns = [
        c for c in EXPECTED_COLUMNS
        if c not in df.columns
    ]

    extra_columns = [
        c for c in df.columns
        if c not in EXPECTED_COLUMNS
    ]

    print("\nMissing columns:", missing_columns)
    print("Extra columns:", extra_columns)

    # --------------------------------------------------------
    # Data types
    # --------------------------------------------------------

    print_section("DATA TYPES")

    print(df.dtypes)

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print_section("MISSING VALUES")

    missing = df.isna().sum()

    missing_pct = (
        missing / len(df) * 100
    )

    for col in df.columns:
        print(
            f"{col:25s}"
            f"{missing[col]:6,d}"
            f"  ({missing_pct[col]:6.2f}%)"
        )

    # --------------------------------------------------------
    # Empty strings
    # --------------------------------------------------------

    print_section("EMPTY STRINGS")

    for col in df.columns:

        if df[col].dtype == "object":

            empty = (
                df[col]
                .fillna("")
                .astype(str)
                .str.strip()
                .eq("")
                .sum()
            )

            print(
                f"{col:25s}{empty:6,d}"
            )

    # --------------------------------------------------------
    # ID validation
    # --------------------------------------------------------

    print_section("ID VALIDATION")

    df["id"] = pd.to_numeric(
        df["id"],
        errors="coerce"
    )

    print("Missing IDs:", df["id"].isna().sum())
    print("Duplicate IDs:", df["id"].duplicated().sum())
    print("Minimum ID:", df["id"].min())
    print("Maximum ID:", df["id"].max())
    print("Unique IDs:", df["id"].nunique())

    expected_ids = set(range(1, 10001))
    actual_ids = set(df["id"].dropna().astype(int))

    missing_ids = sorted(
        expected_ids - actual_ids
    )

    extra_ids = sorted(
        actual_ids - expected_ids
    )

    print("Missing ID count:", len(missing_ids))
    print("Extra ID count:", len(extra_ids))

    if missing_ids:
        print("First missing IDs:", missing_ids[:30])

    if extra_ids:
        print("Extra IDs:", extra_ids[:30])

    # --------------------------------------------------------
    # Exact duplicate rows
    # --------------------------------------------------------

    print_section("DUPLICATES")

    duplicate_rows = df.duplicated().sum()

    duplicate_text = (
        df["text"]
        .duplicated()
        .sum()
    )

    duplicate_ids = (
        df["id"]
        .duplicated()
        .sum()
    )

    print("Exact duplicate rows:", duplicate_rows)
    print("Duplicate text:", duplicate_text)
    print("Duplicate IDs:", duplicate_ids)

    # --------------------------------------------------------
    # Intent
    # --------------------------------------------------------

    print_section("INTENT DISTRIBUTION")

    show_distribution(
        df["intent"],
        "Intent"
    )

    actual_intents = set(
        df["intent"].dropna()
    )

    missing_intents = sorted(
        set(EXPECTED_INTENTS) - actual_intents
    )

    invalid_intents = sorted(
        actual_intents - set(EXPECTED_INTENTS)
    )

    print("\nMissing intents:", missing_intents)
    print("Invalid intents:", invalid_intents)

    # --------------------------------------------------------
    # Language
    # --------------------------------------------------------

    print_section("LANGUAGE DISTRIBUTION")

    show_distribution(
        df["language"],
        "Language"
    )

    actual_languages = set(
        df["language"].dropna()
    )

    missing_languages = sorted(
        set(EXPECTED_LANGUAGES) - actual_languages
    )

    invalid_languages = sorted(
        actual_languages - set(EXPECTED_LANGUAGES)
    )

    print("\nMissing languages:", missing_languages)
    print("Invalid languages:", invalid_languages)

    # --------------------------------------------------------
    # Code switching
    # --------------------------------------------------------

    print_section("CODE-SWITCHING")

    show_distribution(
        df["is_code_switched"],
        "is_code_switched"
    )

    # Compare language vs code-switching
    print("\nLanguage × Code-Switching:")

    print(
        pd.crosstab(
            df["language"],
            df["is_code_switched"],
            margins=True
        )
    )

    # --------------------------------------------------------
    # Formality
    # --------------------------------------------------------

    print_section("FORMALITY DISTRIBUTION")

    show_distribution(
        df["formality"],
        "Formality"
    )

    # --------------------------------------------------------
    # Difficulty
    # --------------------------------------------------------

    print_section("DIFFICULTY DISTRIBUTION")

    show_distribution(
        df["difficulty"],
        "Difficulty"
    )

    # --------------------------------------------------------
    # Text statistics
    # --------------------------------------------------------

    print_section("TEXT STATISTICS")

    text = (
        df["text"]
        .fillna("")
        .astype(str)
    )

    word_counts = text.str.split().str.len()
    char_counts = text.str.len()

    print(
        f"Minimum words:     {word_counts.min()}"
    )
    print(
        f"Maximum words:     {word_counts.max()}"
    )
    print(
        f"Average words:     {word_counts.mean():.2f}"
    )
    print(
        f"Median words:      {word_counts.median():.2f}"
    )

    print(
        f"\nMinimum characters: {char_counts.min()}"
    )
    print(
        f"Maximum characters: {char_counts.max()}"
    )
    print(
        f"Average characters: {char_counts.mean():.2f}"
    )
    print(
        f"Median characters:  {char_counts.median():.2f}"
    )

    # --------------------------------------------------------
    # Very short texts
    # --------------------------------------------------------

    print_section("VERY SHORT TEXTS")

    short_text = df.loc[
        word_counts <= 2,
        ["id", "text", "intent", "language"]
    ]

    print(
        f"Texts with <= 2 words: {len(short_text):,}"
    )

    if len(short_text) > 0:
        print("\nExamples:")
        print(
            short_text
            .head(20)
            .to_string(index=False)
        )

    # --------------------------------------------------------
    # Long texts
    # --------------------------------------------------------

    print_section("LONG TEXTS")

    long_text = df.loc[
        word_counts >= 50,
        ["id", "text", "intent", "language"]
    ]

    print(
        f"Texts with >= 50 words: {len(long_text):,}"
    )

    # --------------------------------------------------------
    # Sample records
    # --------------------------------------------------------

    print_section("RANDOM SAMPLE")

    print(
        df.sample(
            min(20, len(df)),
            random_state=42
        ).to_string(index=False)
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print_section("PROFILE COMPLETE")

    print(
        "No data has been modified or saved."
    )

    print(
        "This script only profiles the raw merged dataset."
    )


if __name__ == "__main__":
    main()