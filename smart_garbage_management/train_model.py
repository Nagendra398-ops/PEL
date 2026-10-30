
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(
    BASE_DIR, "data", "smart-bin-sensor-fill-levels.csv"
)
TEST_PATH = os.path.join(BASE_DIR, "data", "test_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "overflow_model.pkl")


def convert_to_bool(series):
    """Safely convert True/False or Yes/No labels to booleans."""
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required = [
        "bin_id",
        "timestamp",
        "bin_type",
        "fill_level_percent",
        "battery_level_percent",
        "overflow_alert",
    ]

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError("Missing dataset columns: " + ", ".join(missing))

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], format="mixed", utc=True, errors="coerce"
    )

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.day_of_week

    df["overflow_alert"] = convert_to_bool(df["overflow_alert"])

    feature_columns = [
        "bin_type",
        "fill_level_percent",
        "battery_level_percent",
        "hour",
        "day_of_week",
    ]

    df = df.dropna(
        subset=feature_columns + ["overflow_alert"]
    ).copy()

    X = df[feature_columns]
    y = df["overflow_alert"]

    if y.nunique() < 2:
        raise ValueError("Training requires both True and False target labels.")

    # Split row indices so original test records can be saved unchanged.
    train_indices, test_indices = train_test_split(
        df.index,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    X_train = X.loc[train_indices]
    X_test = X.loc[test_indices]
    y_train = y.loc[train_indices]
    y_test = y.loc[test_indices]

    preprocessor = ColumnTransformer([
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            ["bin_type"],
        ),
        (
            "numeric",
            "passthrough",
            [
                "fill_level_percent",
                "battery_level_percent",
                "hour",
                "day_of_week",
            ],
        ),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight="balanced",
            ),
        ),
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Total usable records:", len(df))
    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))
    print("Test accuracy:", round(accuracy_score(y_test, predictions), 4))
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(TEST_PATH), exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    # Save only held-out test records, including actual and predicted labels.
    test_results = df.loc[test_indices].copy()
    test_results["predicted_overflow_alert"] = predictions
    test_results["prediction_correct"] = (
        test_results["overflow_alert"]
        == test_results["predicted_overflow_alert"]
    )

    test_results.to_csv(TEST_PATH, index=False)

    print("\nSaved model:", MODEL_PATH)
    print("Saved test records:", TEST_PATH)


if __name__ == "__main__":
    main()
