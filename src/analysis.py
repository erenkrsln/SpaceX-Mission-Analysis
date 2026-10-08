"""SpaceX launch analysis: data preparation, plots and classification."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

FEATURES = ["flight_number", "year", "reused", "attempts"]


def clean_launches(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the expected launch table."""
    required = {"flight_number", "date_utc", "success", "reused", "attempts"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing fields: {', '.join(sorted(missing))}")
    df = frame.copy()
    df["date_utc"] = pd.to_datetime(df["date_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["date_utc", "success", "flight_number"])
    df["success"] = df["success"].astype(int)
    df["reused"] = df["reused"].fillna(False).astype(int)
    df["attempts"] = pd.to_numeric(df["attempts"], errors="coerce").fillna(0).astype(int)
    df["flight_number"] = pd.to_numeric(df["flight_number"], errors="coerce")
    df = df.dropna(subset=["flight_number"])
    df["year"] = df["date_utc"].dt.year
    return df.sort_values("date_utc").reset_index(drop=True)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby("year", as_index=False)
            .agg(launches=("success", "size"), successes=("success", "sum"))
            .assign(success_rate=lambda x: (x.successes / x.launches).round(3)))


def make_plot(summary: pd.DataFrame, target: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(summary["year"].astype(str), summary["success_rate"] * 100)
    ax.set(title="Launch success rate by year", xlabel="Year", ylabel="Success rate (%)", ylim=(0, 105))
    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)


def evaluate_model(df: pd.DataFrame) -> dict:
    """Chronological holdout. This predicts launch success, not booster landing."""
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    if len(df) < 12 or df["success"].nunique() < 2:
        return {"status": "Not enough observations or only one target class"}
    split = max(1, int(len(df) * 0.8))
    train, test = df.iloc[:split], df.iloc[split:]
    if test.empty or train["success"].nunique() < 2:
        return {"status": "Training or test split is too small"}
    result = {"status": "ok", "train_size": len(train), "test_size": len(test)}
    for name, model in [("baseline", DummyClassifier(strategy="most_frequent")),
                        ("logistic_regression", LogisticRegression(max_iter=2000))]:
        model.fit(train[FEATURES], train["success"])
        pred = model.predict(test[FEATURES])
        result[name] = {"accuracy": round(accuracy_score(test["success"], pred), 3),
                        "balanced_accuracy": round(balanced_accuracy_score(test["success"], pred), 3)}
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="SpaceX launch exploration")
    p.add_argument("--input", default="data/example_launches.csv")
    p.add_argument("--out", default="outputs")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = clean_launches(pd.read_csv(args.input))
    summary = summarize(df)
    summary.to_csv(out / "yearly_summary.csv", index=False)
    make_plot(summary, out / "success_rate.png")
    print(summary.to_string(index=False))
    print("Model evaluation:", evaluate_model(df))
    print("Output directory:", out)


if __name__ == "__main__":
    main()
