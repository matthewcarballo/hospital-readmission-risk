# src/train_model.py
# Train a Random Forest model to predict 30-day hospital readmission.

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split


TARGET_COL = "readmitted_30d"
RANDOM_STATE = 42


def load_clean_data(filepath: Path) -> pd.DataFrame:
    """Load the cleaned dataset."""
    return pd.read_csv(filepath)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split the dataframe into model features and target labels."""
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def split_train_test(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create stratified training and test sets."""
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def train_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series
) -> RandomForestClassifier:
    """Train the Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Evaluate the model on the held-out test set."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print("=" * 55)
    print("MODEL EVALUATION ON TEST SET")
    print("=" * 55)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Not Readmitted", "Readmitted <30d"],
        )
    )

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("Confusion Matrix:")
    print(f"  True negatives:  {tn:,}")
    print(f"  False positives: {fp:,}")
    print(f"  False negatives: {fn:,}")
    print(f"  True positives:  {tp:,}")

    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC-AUC Score: {auc:.3f}")

    return {
        "confusion_matrix": cm,
        "auc": auc,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
    }


def save_model(model: RandomForestClassifier, filepath: Path) -> None:
    """Save the trained model."""
    filepath.parent.mkdir(exist_ok=True)
    joblib.dump(model, filepath)
    print(f"\nModel saved to {filepath}")


def run_training_pipeline(
    data_path: Path,
    model_output_path: Path,
) -> tuple[RandomForestClassifier, dict]:
    """Run the training pipeline from data loading through model export."""
    print("Loading cleaned dataset...")
    df = load_clean_data(data_path)
    print(f"  Shape: {df.shape}")

    X, y = split_features_target(df)

    print("\nSplitting into train/test sets...")
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    print(f"  Train: {X_train.shape[0]:,} rows ({y_train.mean() * 100:.1f}% positive)")
    print(f"  Test:  {X_test.shape[0]:,} rows ({y_test.mean() * 100:.1f}% positive)")

    print("\nTraining Random Forest with class weights...")
    model = train_random_forest(X_train, y_train)
    print("  Training complete.")

    results = evaluate_model(model, X_test, y_test)
    save_model(model, model_output_path)

    return model, results


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "diabetic_data_clean.csv"
    model_path = project_root / "models" / "random_forest_model.pkl"

    run_training_pipeline(data_path, model_path)