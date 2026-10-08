"""Fetch real SpaceX launch records from the public SpaceX API (requires internet)."""
import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen
import pandas as pd

API = "https://api.spacexdata.com/v4/launches"
BOOSTERS = "https://api.spacexdata.com/v4/cores"


def download_json(url: str) -> list:
    with urlopen(Request(url, headers={"User-Agent": "SpaceX-Mission-Analysis/1.0"}), timeout=25) as response:
        return json.load(response)


def make_table(launches: list, boosters: list) -> pd.DataFrame:
    cores = {c["id"]: c for c in boosters}
    rows = []
    for launch in launches:
        if launch.get("success") is None:
            continue
        core_entries = launch.get("cores") or []
        first = core_entries[0] if core_entries else {}
        core = cores.get(first.get("core"), {})
        rows.append({"flight_number": launch.get("flight_number"),
                     "date_utc": launch.get("date_utc"),
                     "success": launch.get("success"),
                     "reused": bool(first.get("reused", False)),
                     "attempts": core.get("reuse_count") or 0})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/spacex_launches.csv")
    args = parser.parse_args()
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    make_table(download_json(API), download_json(BOOSTERS)).to_csv(target, index=False)
    print(f"Saved {target}")


if __name__ == "__main__":
    main()
