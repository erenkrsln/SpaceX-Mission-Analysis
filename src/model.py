
"""Train and evaluate a logistic regression model for SpaceX launches."""

import argparse

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_model(csv_path):
    """Train a model to classify successful SpaceX launches."""

    # Load dataset
    df = pd.read_csv(csv_path)
    print("Available columns:", df.columns.tolist())

    required_columns = [
        "flight_number",
        "date_utc",
        "success",
        "reused",
        "attempts",
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Clean target variable
    df["success"] = df["success"].astype("string").str.lower()
    df["success"] = df["success"].map({
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0,
    })

    df = df.dropna(subset=["success"]).copy()
    df["success"] = df["success"].astype(int)

    # Extract launch year
    df["year"] = pd.to_datetime(
        df["date_utc"],
        errors="coerce",
        utc=True,
    ).dt.year

    # Convert reused values to numeric
    df["reused"] = (
        df["reused"]
        .astype("string")
        .str.lower()
        .map({
            "true": 1,
            "false": 0,
            "1": 1,
            "0": 0,
        })
    )

    # Convert numerical features
    for column in ["flight_number", "attempts", "year"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    features = [
        "flight_number",
        "reused",
        "attempts",
        "year",
    ]

    X = df[features]
    y = df["success"]

    # Validate dataset
    class_counts = y.value_counts()

    if len(class_counts) < 2:
        raise ValueError(
            "Dataset needs both successful and failed launches."
        )

    if class_counts.min() < 2 or len(df) < 8:
        raise ValueError(
            "Not enough data for stratified training and testing. "
            "Use a larger dataset with more examples of each class."
        )

    # Preprocessing pipeline
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_transformer, features),
    ])

    # Machine learning pipeline
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    # Train model
    model.fit(X_train, y_train)

    # Evaluate model
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\n=== SpaceX Machine Learning Results ===")
    print(f"Total launches: {len(df)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Accuracy: {accuracy:.3f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SpaceX Launch Success Prediction"
    )

    parser.add_argument(
        "--input",
        default="data/example_launches.csv",
        help="Path to the SpaceX dataset",
    )

    args = parser.parse_args()
    train_model(args.input)
