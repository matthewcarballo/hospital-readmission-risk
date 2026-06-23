# src/diagnose_model.py
# Load the trained model and review the most important features.

from pathlib import Path

import joblib
import pandas as pd


TARGET_COL = "readmitted_30d"
EXPECTED_SIGNALS = [
    "number_inpatient",
    "age",
    "num_medications",
    "time_in_hospital",
]


def load_model_and_features(
    model_path: Path,
    data_path: Path,
) -> tuple[object, pd.Index]:
    """Load the trained model and feature names."""
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    feature_names = df.drop(columns=[TARGET_COL]).columns
    return model, feature_names


def get_top_features(
    model: object,
    feature_names: pd.Index,
    top_n: int = 15,
) -> pd.DataFrame:
    """Return the most important model features."""
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    )
    return importance_df.sort_values("importance", ascending=False).head(top_n)


def print_diagnostic_report(top_features: pd.DataFrame) -> None:
    """Print a short feature-importance report."""
    print("=" * 55)
    print("TOP FEATURES BY IMPORTANCE")
    print("=" * 55)

    for rank, (_, row) in enumerate(top_features.iterrows(), start=1):
        print(f"  {rank:>2}. {row['feature']:<35} {row['importance']:.4f}")

    found = [feature for feature in EXPECTED_SIGNALS if feature in top_features["feature"].values]
    missing = [feature for feature in EXPECTED_SIGNALS if feature not in top_features["feature"].values]

    print("\nCheck against expected signals:")
    print(f"  Found:   {found if found else 'none'}")
    print(f"  Missing: {missing if missing else 'none'}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / "models" / "random_forest_model.pkl"
    data_path = project_root / "data" / "diabetic_data_clean.csv"

    model, feature_names = load_model_and_features(model_path, data_path)
    top_features = get_top_features(model, feature_names)
    print_diagnostic_report(top_features)