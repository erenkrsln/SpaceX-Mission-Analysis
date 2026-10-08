
"""Train and evaluate Falcon 9 landing predictions."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "booster_flight",
    "reused",
    "gridfins",
    "legs",
]


def evaluate(name, y_true, y_pred):
    """Print classification metrics."""

    print(f"\n=== {name} ===")

    accuracy = accuracy_score(y_true, y_pred)
    balanced = balanced_accuracy_score(y_true, y_pred)

    print(f"Accuracy: {accuracy:.3f}")
    print(f"Balanced Accuracy: {balanced:.3f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Failed", "Successful"],
            zero_division=0,
        )
    )


def train_model(csv_path="data/falcon9_landings.csv"):
    """Train and evaluate a landing prediction model."""

    df = pd.read_csv(csv_path)

    required = FEATURES + ["landing_success"]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Convert boolean features into numerical values
    for col in ["reused", "gridfins", "legs"]:
        df[col] = (
            df[col]
            .astype("string")
            .str.lower()
            .map({
                "true": 1,
                "false": 0,
                "1": 1,
                "0": 0,
            })
        )

    df["booster_flight"] = pd.to_numeric(
        df["booster_flight"],
        errors="coerce",
    )

    df["landing_success"] = pd.to_numeric(
        df["landing_success"],
        errors="coerce",
    )

    df = df.dropna(subset=["landing_success"]).copy()

    X = df[FEATURES]
    y = df["landing_success"].astype(int)

    counts = y.value_counts()

    if len(counts) != 2 or counts.min() < 2:
        raise ValueError(
            "At least two examples per class are required."
        )

    # Split training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    # Baseline model
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)

    baseline_predictions = baseline.predict(X_test)

    # Logistic regression pipeline
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nFalcon 9 Landing Prediction")
    print("---------------------------")
    print(f"Total landing attempts: {len(df)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    print("\nClass distribution:")
    print(y.value_counts().to_string())

    evaluate("Baseline", y_test, baseline_predictions)
    evaluate("Logistic Regression", y_test, predictions)

    # Create confusion matrix
    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    output = Path("outputs/landings")
    output.mkdir(parents=True, exist_ok=True)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Failed", "Successful"],
    )

    display.plot(cmap="Blues", values_format="d")
    plt.title("Falcon 9 Landing Prediction")
    plt.tight_layout()

    plt.savefig(
        output / "confusion_matrix.png",
        dpi=150,
    )

    plt.close()

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nResults saved to:")
    print(output / "confusion_matrix.png")

    return model


if __name__ == "__main__":
    train_model()
