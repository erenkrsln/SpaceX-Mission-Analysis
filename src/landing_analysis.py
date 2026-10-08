
"""Analyze historical Falcon 9 first-stage landings."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def analyze_landings(csv_path="data/falcon9_landings.csv"):
    df = pd.read_csv(csv_path)

    df["date_utc"] = pd.to_datetime(
        df["date_utc"], utc=True
    )
    df["year"] = df["date_utc"].dt.year

    summary = (
        df.groupby("year")["landing_success"]
        .agg(attempts="count", successful="sum", success_rate="mean")
        .reset_index()
    )

    print("\nFalcon 9 Landing Analysis")
    print(summary.to_string(index=False))

    output = Path("outputs/landings")
    output.mkdir(parents=True, exist_ok=True)

    summary.to_csv(
        output / "yearly_summary.csv",
        index=False,
    )

    plt.figure(figsize=(10, 5))
    plt.plot(
        summary["year"],
        summary["success_rate"] * 100,
        marker="o",
    )
    plt.title("Falcon 9 First-Stage Landing Success Rate")
    plt.xlabel("Year")
    plt.ylabel("Landing Success Rate (%)")
    plt.ylim(0, 105)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output / "landing_success_rate.png", dpi=150)
    plt.close()

    print(f"\nResults saved to: {output}")


if __name__ == "__main__":
    analyze_landings()
