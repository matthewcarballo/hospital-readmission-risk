# src/explain_model.py
# Explain the trained Random Forest model with SHAP.

from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split


TARGET_COL = "readmitted_30d"
RANDOM_STATE = 42
POSITIVE_CLASS_INDEX = 1


def load_model_and_data(
    model_path: Path,
    data_path: Path,
) -> tuple[Any, pd.DataFrame, pd.Series]:
    """Load the trained model and split the cleaned data into X/y."""
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    return model, X, y


def get_test_sample(
    X: pd.DataFrame,
    y: pd.Series,
    sample_size: int = 500,
) -> pd.DataFrame:
    """Recreate the test split and sample rows for SHAP analysis."""
    _, X_test, _, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    sample_size = min(sample_size, len(X_test))
    return X_test.sample(n=sample_size, random_state=RANDOM_STATE)


def compute_shap_values(model: Any, X_sample: pd.DataFrame) -> tuple[Any, Any]:
    """Compute SHAP values for the sampled test rows."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    return explainer, shap_values


def get_positive_class_values(shap_values: Any, feature_count: int) -> np.ndarray:
    """Return SHAP values for the readmitted class."""
    if isinstance(shap_values, list):
        return shap_values[POSITIVE_CLASS_INDEX]

    values = getattr(shap_values, "values", shap_values)
    values = np.asarray(values)

    if values.ndim == 3:
        if values.shape[1] == feature_count:
            return values[:, :, POSITIVE_CLASS_INDEX]

        if values.shape[2] == feature_count:
            return values[POSITIVE_CLASS_INDEX, :, :]

    return values


def get_positive_base_value(explainer: Any) -> float:
    """Return the model's expected value for the readmitted class."""
    expected_value = np.asarray(explainer.expected_value)

    if expected_value.ndim == 0:
        return float(expected_value)

    return float(expected_value[POSITIVE_CLASS_INDEX])


def find_risk_patient(
    model: Any,
    X_sample: pd.DataFrame,
    highest: bool = True,
) -> int:
    """Find the highest- or lowest-risk patient in the sample."""
    probabilities = model.predict_proba(X_sample)[:, POSITIVE_CLASS_INDEX]

    if highest:
        patient_position = int(np.argmax(probabilities))
        label = "Highest"
    else:
        patient_position = int(np.argmin(probabilities))
        label = "Lowest"

    print(f"  {label} predicted risk: {probabilities[patient_position]:.3f}")
    return patient_position


def plot_global_importance(
    shap_values: Any,
    X_sample: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save a SHAP summary plot for the readmitted class."""
    values_to_plot = get_positive_class_values(shap_values, X_sample.shape[1])

    plt.figure(figsize=(10, 8))
    shap.summary_plot(values_to_plot, X_sample, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()

    print(f"Saved global SHAP summary plot to {output_path}")


def explain_single_patient(
    explainer: Any,
    X_sample: pd.DataFrame,
    patient_index: int,
    output_path: Path,
) -> None:
    """Save a waterfall plot for one sampled patient."""
    if patient_index >= len(X_sample):
        raise IndexError(
            f"patient_index={patient_index} is outside the sample size of {len(X_sample)}"
        )

    patient_data = X_sample.iloc[[patient_index]]
    shap_values_patient = explainer.shap_values(patient_data)

    values = get_positive_class_values(
        shap_values_patient,
        patient_data.shape[1],
    )[0]

    base_value = get_positive_base_value(explainer)

    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=values,
            base_values=base_value,
            data=patient_data.iloc[0].values,
            feature_names=patient_data.columns.tolist(),
        ),
        show=False,
        max_display=12,
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()

    print(f"Saved individual patient explanation to {output_path}")


def print_top_drivers(
    shap_values: Any,
    X_sample: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Print the top features by mean absolute SHAP value."""
    values_to_use = get_positive_class_values(shap_values, X_sample.shape[1])

    mean_abs_shap = (
        pd.DataFrame(
            {
                "feature": X_sample.columns,
                "mean_abs_shap": np.abs(values_to_use).mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_n)
    )

    print("=" * 55)
    print("TOP FEATURES BY MEAN ABSOLUTE SHAP VALUE")
    print("=" * 55)

    for rank, (_, row) in enumerate(mean_abs_shap.iterrows(), start=1):
        print(f"  {rank:>2}. {row['feature']:<30} {row['mean_abs_shap']:.4f}")

    return mean_abs_shap


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / "models" / "random_forest_model.pkl"
    data_path = project_root / "data" / "diabetic_data_clean.csv"

    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)

    print("Loading model and data...")
    model, X, y = load_model_and_data(model_path, data_path)

    print("Sampling test rows for SHAP analysis...")
    X_sample = get_test_sample(X, y)
    print(f"  Sample size: {X_sample.shape[0]}")

    print("\nComputing SHAP values...")
    explainer, shap_values = compute_shap_values(model, X_sample)

    print("\nGenerating global feature importance plot...")
    plot_global_importance(
        shap_values,
        X_sample,
        reports_dir / "06_shap_summary.png",
    )

    print("\nFinding the highest-risk patient in the sample...")
    high_risk_index = find_risk_patient(model, X_sample, highest=True)

    print("\nExplaining the highest-risk patient...")
    explain_single_patient(
        explainer,
        X_sample,
        patient_index=high_risk_index,
        output_path=reports_dir / "07_shap_patient_high_risk.png",
    )

    print("\nFinding the lowest-risk patient in the sample...")
    low_risk_index = find_risk_patient(model, X_sample, highest=False)

    print("\nExplaining the lowest-risk patient...")
    explain_single_patient(
        explainer,
        X_sample,
        patient_index=low_risk_index,
        output_path=reports_dir / "08_shap_patient_low_risk.png",
    )

    print()
    print_top_drivers(shap_values, X_sample)