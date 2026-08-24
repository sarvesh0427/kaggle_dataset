from pathlib import Path
import csv


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "batches"
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "batches_repaired"


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
# REPAIR ONE FILE
# ============================================================

def repair_file(input_file, output_file):

    repaired_rows = []
    malformed_count = 0

    with open(
        input_file,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.reader(f)

        header = next(reader)

        if header != EXPECTED_COLUMNS:
            raise ValueError(
                f"\nInvalid header in {input_file.name}\n"
                f"Expected: {EXPECTED_COLUMNS}\n"
                f"Found:    {header}"
            )

        for line_number, row in enumerate(reader, start=2):

            # ------------------------------------------------
            # Correct CSV row
            # ------------------------------------------------

            if len(row) == 7:

                repaired_rows.append(row)

            # ------------------------------------------------
            # Malformed row
            # ------------------------------------------------

            elif len(row) > 7:

                malformed_count += 1

                # Schema:
                #
                # id
                # text
                # intent
                # language
                # is_code_switched
                # formality
                # difficulty
                #
                # The final 5 fields are known.
                # Everything between id and those 5 fields
                # belongs to the text.

                fixed_row = [
                    row[0],
                    ",".join(row[1:-5]),
                    row[-5],
                    row[-4],
                    row[-3],
                    row[-2],
                    row[-1],
                ]

                repaired_rows.append(fixed_row)

                print(
                    f"  REPAIRED: {input_file.name} "
                    f"| line {line_number} "
                    f"| ID {row[0]}"
                )

            else:

                raise ValueError(
                    f"\nInvalid row in {input_file.name} "
                    f"at line {line_number}\n"
                    f"Expected 7 fields, found {len(row)}\n"
                    f"Row: {row}"
                )

    # --------------------------------------------------------
    # Write repaired file
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.writer(
            f,
            quoting=csv.QUOTE_MINIMAL
        )

        writer.writerow(EXPECTED_COLUMNS)
        writer.writerows(repaired_rows)

    return len(repaired_rows), malformed_count


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("CSV Batch Repair")
    print("=" * 60)

    files = sorted(INPUT_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No CSV files found in:\n{INPUT_DIR}"
        )

    print(f"\nFiles found: {len(files)}")
    print(f"Output directory: {OUTPUT_DIR}\n")

    total_rows = 0
    total_repaired = 0

    for i, input_file in enumerate(files, start=1):

        output_file = OUTPUT_DIR / input_file.name

        print(
            f"[{i}/{len(files)}] "
            f"{input_file.name}"
        )

        rows, repaired = repair_file(
            input_file,
            output_file
        )

        total_rows += rows
        total_repaired += repaired

        print(
            f"    Rows: {rows:,} | "
            f"Repaired: {repaired}"
        )

    print("\n" + "=" * 60)
    print("REPAIR COMPLETE")
    print("=" * 60)

    print(f"\nTotal rows:     {total_rows:,}")
    print(f"Rows repaired:  {total_repaired:,}")

    print(f"\nRepaired files saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()