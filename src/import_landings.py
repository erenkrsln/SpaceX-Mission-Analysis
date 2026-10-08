
"""Import historical Falcon 9 first-stage landing attempts."""

import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


SOURCE = (
    "https://raw.githubusercontent.com/nisrulz/"
    "android-spacex-app/main/api-server/"
    "server/data/launches_v5.json"
)

FALCON_9_ID = "5e9d0d95eda69973a809d1ec"

OUTPUT_FILE = Path("data/falcon9_landings.csv")

COLUMNS = [
    "flight_number",
    "date_utc",
    "mission",
    "booster_flight",
    "reused",
    "gridfins",
    "legs",
    "landing_type",
    "landing_success",
]


def import_landings():
    """Download and process historical Falcon 9 landing data."""

    print("Downloading historical SpaceX data...", flush=True)

    request = Request(
        SOURCE,
        headers={
            "User-Agent": "SpaceX-Mission-Analysis/1.0",
            "Accept": "application/json",
        },
    )

    try:
        with urlopen(request, timeout=30) as response:
            launches = json.load(response)

    except HTTPError as error:
        raise RuntimeError(
            f"HTTP error while downloading data: {error.code}"
        ) from error

    except URLError as error:
        raise RuntimeError(
            f"Network error: {error.reason}"
        ) from error

    if not isinstance(launches, list):
        raise ValueError("Expected a list of launch records.")

    print(f"Downloaded {len(launches)} launch records.")

    rows = []

    for launch in launches:
        if not isinstance(launch, dict):
            continue

        if launch.get("upcoming") is True:
            continue

        if launch.get("rocket") != FALCON_9_ID:
            continue

        cores = launch.get("cores") or []

        for core in cores:
            if not isinstance(core, dict):
                continue

            # Include only documented landing attempts
            if core.get("landing_attempt") is not True:
                continue

            landing_success = core.get("landing_success")

            if not isinstance(landing_success, bool):
                continue

            rows.append({
                "flight_number": launch.get("flight_number"),
                "date_utc": launch.get("date_utc"),
                "mission": launch.get("name"),
                "booster_flight": core.get("flight"),
                "reused": core.get("reused"),
                "gridfins": core.get("gridfins"),
                "legs": core.get("legs"),
                "landing_type": core.get("landing_type"),
                "landing_success": int(landing_success),
            })

    df = pd.DataFrame(rows, columns=COLUMNS)

    if df.empty:
        raise ValueError(
            "No usable Falcon 9 landing attempts found. "
            "Check the snapshot structure and rocket ID."
        )

    df["date_utc"] = pd.to_datetime(
        df["date_utc"],
        errors="coerce",
        utc=True,
    )

    df = df.dropna(
        subset=["date_utc", "flight_number"]
    ).copy()

    if df.empty:
        raise ValueError("No valid dated landing records remain.")

    df = df.sort_values(
        ["date_utc", "flight_number"]
    ).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved {len(df)} Falcon 9 landing attempts.")
    print(f"File: {OUTPUT_FILE.resolve()}")

    print("\nLanding outcomes:")
    print(
        df["landing_success"]
        .value_counts()
        .rename(index={1: "Successful", 0: "Failed"})
        .to_string()
    )

    print("\nFirst five records:")
    print(df.head().to_string(index=False))

    return df


if __name__ == "__main__":
    import_landings()
