"""Build the shared candidate/feature tables for every trip.

Usage:  python -m src.features.build_table
Writes: data/processed/candidates_{train,student_test}.parquet
        results/tables/trip_quality.csv
"""
import time

import pandas as pd

from src.config import PROCESSED_DIR, RESULTS_DIR
from src.data.loader import list_trips, load_trip
from src.features.candidates import build_trip_table


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "tables").mkdir(parents=True, exist_ok=True)
    infos = []
    for part in ("train", "student_test"):
        tabs = []
        for tid in list_trips(part):
            t0 = time.time()
            tab, info = build_trip_table(load_trip(part, tid))
            tab["trip_id"] = tab["trip_id"].astype(str)
            tabs.append(tab); infos.append(info)
            print(f"{part}/{tid}: pts={info['n_points']:6d} rows={len(tab):7d} "
                  f"labelled={info['label_frac']:.3f} miss={info['candidate_miss_rate']:.3f} {time.time()-t0:.1f}s", flush=True)
        pd.concat(tabs, ignore_index=True).to_parquet(PROCESSED_DIR / f"candidates_{part}.parquet")
    pd.DataFrame(infos).to_csv(RESULTS_DIR / "tables" / "trip_quality.csv", index=False)


if __name__ == "__main__":
    main()
