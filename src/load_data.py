# src/load_data.py

import pandas as pd


DEFAULT_FILEPATH = "data/diabetic_data.csv"
MISSING_VALUE = "?"


def load_data(filepath=DEFAULT_FILEPATH):
    """Load the diabetes readmission dataset into a DataFrame."""
    return pd.read_csv(filepath)


def preview_data(df):
    """Print a quick overview of the dataset."""
    print("=" * 50)
    print("DATASET OVERVIEW")
    print("=" * 50)

    print(f"\nTotal patient encounters: {df.shape[0]:,}")
    print(f"Data fields per patient:  {df.shape[1]}")

    print("\nData fields available:")
    for col in df.columns:
        print(f"  - {col}")

    print(f"\nMissing values marked as '{MISSING_VALUE}':")
    missing_counts = (df == MISSING_VALUE).sum()
    missing_counts = missing_counts[missing_counts > 0]

    if missing_counts.empty:
        print("  No '?' values found.")
    else:
        for col, missing in missing_counts.items():
            pct = missing / len(df) * 100
            print(f"  {col}: {missing:,} missing ({pct:.1f}%)")

    print("\nFirst 3 patient records:")
    print(df.head(3).to_string())

    if "readmitted" in df.columns:
        print("\nReadmission outcomes:")
        print(df["readmitted"].value_counts())

        print("\nMeaning:")
        print("  '<30' = readmitted within 30 days")
        print("  '>30' = readmitted after 30 days")
        print("  'NO'  = not readmitted")


if __name__ == "__main__":
    print("Loading dataset...")
    df = load_data()
    preview_data(df)