# src/preprocess.py
# Data cleaning and feature engineering for hospital readmission prediction.

import numpy as np
import pandas as pd


COLS_TO_DROP = [
    "weight",
    "payer_code",
    "encounter_id",
    "patient_nbr",
    "readmitted",
]

MEDICATION_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]

AGE_MAP = {
    "[0-10)":   5,
    "[10-20)": 15,
    "[20-30)": 25,
    "[30-40)": 35,
    "[40-50)": 45,
    "[50-60)": 55,
    "[60-70)": 65,
    "[70-80)": 75,
    "[80-90)": 85,
    "[90-100)": 95,
}

SPECIALTY_MAP = {
    "InternalMedicine":          "Internal Medicine",
    "Emergency/Trauma":          "Emergency",
    "Family/GeneralPractice":    "General Practice",
    "Cardiology":                "Cardiology",
    "Surgery-General":           "Surgery",
    "Nephrology":                "Nephrology",
    "Orthopedics":               "Other",
    "Orthopedics-Reconstructive":"Other",
    "Radiologist":               "Other",
}

BINARY_MAPS = {
    "gender":      {"Male": 1, "Female": 0},
    "change":      {"Ch": 1, "No": 0},
    "diabetesMed": {"Yes": 1, "No": 0},
}


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create the binary 30-day readmission target."""
    if "readmitted" not in df.columns:
        raise ValueError("Column 'readmitted' is required to create the target variable.")
    df = df.copy()
    df["readmitted_30d"] = (df["readmitted"] == "<30").astype(int)
    return df


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that are mostly missing, identifiers, or no longer needed."""
    existing_cols = [col for col in COLS_TO_DROP if col in df.columns]
    not_found = [col for col in COLS_TO_DROP if col not in df.columns]
    if not_found:
        print(f"  Note: these columns were not found and skipped: {not_found}")
    return df.drop(columns=existing_cols)


def replace_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Replace dataset-specific missing value markers with NaN."""
    df = df.replace("?", np.nan)
    if "gender" in df.columns:
        df["gender"] = df["gender"].replace("Unknown/Invalid", np.nan)
    return df


def encode_age(df: pd.DataFrame) -> pd.DataFrame:
    """Convert age ranges into numeric midpoint values."""
    if "age" in df.columns:
        df["age"] = df["age"].map(AGE_MAP)
    return df


def simplify_medical_specialty(df: pd.DataFrame) -> pd.DataFrame:
    """Group medical specialties into broader categories."""
    if "medical_specialty" not in df.columns:
        return df

    df["medical_specialty"] = df["medical_specialty"].fillna("Missing")
    df["medical_specialty"] = df["medical_specialty"].map(SPECIALTY_MAP).fillna(df["medical_specialty"])

    return df

def simplify_medications(df: pd.DataFrame) -> pd.DataFrame:
    """Convert medication status columns to binary prescribed/not prescribed flags."""
    for col in MEDICATION_COLS:
        if col in df.columns:
            df[col] = (df[col] != "No").astype(int)
    return df


def map_diagnosis_code(code: str | float | None) -> str:
    """Map an ICD-9 diagnosis code to a broad clinical category."""
    if pd.isna(code):
        return "Missing"
    code = str(code)
    if code.startswith("V") or code.startswith("E"):
        return "Other"
    try:
        code_num = float(code)
    except ValueError:
        return "Other"

    if 250 <= code_num < 251:
        return "Diabetes"
    if 390 <= code_num <= 459 or code_num == 785:
        return "Circulatory"
    if 460 <= code_num <= 519 or code_num == 786:
        return "Respiratory"
    if 520 <= code_num <= 579 or code_num == 787:
        return "Digestive"
    if 580 <= code_num <= 629 or code_num == 788:
        return "Genitourinary"
    if 710 <= code_num <= 739:
        return "Musculoskeletal"
    if 800 <= code_num <= 999:
        return "Injury"
    if 140 <= code_num <= 239:
        return "Neoplasms"
    return "Other"


def simplify_diagnosis_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Group ICD-9 diagnosis codes into broad clinical categories."""
    for col in ["diag_1", "diag_2", "diag_3"]:
        if col in df.columns:
            df[col] = df[col].apply(map_diagnosis_code)
    return df


def encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Encode binary and multi-category text columns for modeling."""
    for col, mapping in BINARY_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    categorical_cols = [
        "race", "medical_specialty", "A1Cresult",
        "max_glu_serum", "diag_1", "diag_2", "diag_3",
    ]
    existing_cats = [col for col in categorical_cols if col in df.columns]
    return pd.get_dummies(df, columns=existing_cats, drop_first=True, dtype=int)


def handle_missing_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Fill remaining numeric missing values with each column's median."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    return df


def validate_preprocessed_data(df: pd.DataFrame) -> bool:
    """
    Verify the preprocessed data is ready for modeling.
    Returns True if all checks pass, raises ValueError on critical failures.
    """
    missing_values = df.isnull().sum().sum()
    text_cols = df.select_dtypes(include="object").columns.tolist()

    if missing_values > 0:
        raise ValueError(f"Preprocessing failed: {missing_values} missing values remain.")
    if text_cols:
        raise ValueError(f"Preprocessing failed: text columns remain: {text_cols}")

    print("  Validation passed: no missing values, all columns numeric.")
    return True


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full preprocessing pipeline."""
    print("Starting preprocessing...")
    print(f"  Input shape: {df.shape}")

    df = df.copy()

    if "readmitted_30d" not in df.columns:
        df = create_target(df)

    df = drop_unnecessary_columns(df)
    print(f"  After dropping columns: {df.shape}")

    df = replace_missing_values(df)
    df = encode_age(df)
    df = simplify_medical_specialty(df)
    df = simplify_medications(df)
    df = simplify_diagnosis_codes(df)
    df = encode_categorical_columns(df)
    df = handle_missing_numeric(df)

    validate_preprocessed_data(df)

    print(f"  Final shape: {df.shape}")
    print(f"  Missing values remaining: {df.isnull().sum().sum()}")
    print("Preprocessing complete!")

    return df


if __name__ == "__main__":
    from load_data import load_data  # fixed: no 'src.' prefix when run directly

    df = load_data()
    df_clean = preprocess(df)

    print(f"\nFinal columns ({len(df_clean.columns)}):")
    for col in sorted(df_clean.columns):
        print(f"  {col}")