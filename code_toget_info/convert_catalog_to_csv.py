"""
Convert nio_cyclones_catalog.json into clean, standardized CSV files:
1. nio_cyclones_summary.csv - One row per cyclone with peak intensity and timeline metadata.
2. nio_cyclones_tracks.csv  - Granular timestep track data (Lat, Lon, Wind, Pres, Category).
3. nio_cyclones_catalog.csv - Full flattened catalog linking storm summary with every track point.
"""

import os
import json
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(CURRENT_DIR, "nio_cyclones_catalog.json")

SUMMARY_CSV = os.path.join(CURRENT_DIR, "nio_cyclones_summary.csv")
TRACKS_CSV  = os.path.join(CURRENT_DIR, "nio_cyclones_tracks.csv")
CATALOG_CSV = os.path.join(CURRENT_DIR, "nio_cyclones_catalog.csv")


def convert_catalog():
    if not os.path.exists(JSON_PATH):
        raise FileNotFoundError(f"File not found: {JSON_PATH}")

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        catalog_data = json.load(f)

    summary_records = []
    track_records = []
    catalog_flattened = []

    for storm in catalog_data:
        # Storm summary
        summary_records.append({
            "sid": storm.get("sid", ""),
            "name": storm.get("name", "UNNAMED"),
            "season": storm.get("season", None),
            "max_wind_kts": storm.get("max_wind_kts", None),
            "min_pres_hpa": storm.get("min_pres_hpa", None),
            "peak_category": storm.get("peak_category", ""),
            "start_time": storm.get("start_time", ""),
            "end_time": storm.get("end_time", ""),
            "duration_hours": storm.get("duration_hours", None),
            "num_timesteps": storm.get("num_timesteps", len(storm.get("track", []))),
        })

        # Track points
        for pt in storm.get("track", []):
            track_entry = {
                "sid": storm.get("sid", ""),
                "name": storm.get("name", "UNNAMED"),
                "season": storm.get("season", None),
                "time": pt.get("time", ""),
                "lat": pt.get("lat", None),
                "lon": pt.get("lon", None),
                "wind_kts": pt.get("wind_kts", None),
                "pres_hpa": pt.get("pres_hpa", None),
                "category": pt.get("category", "")
            }
            track_records.append(track_entry)

            # Full merged row
            catalog_flattened.append({
                "sid": storm.get("sid", ""),
                "name": storm.get("name", "UNNAMED"),
                "season": storm.get("season", None),
                "peak_category": storm.get("peak_category", ""),
                "max_wind_kts": storm.get("max_wind_kts", None),
                "min_pres_hpa": storm.get("min_pres_hpa", None),
                "duration_hours": storm.get("duration_hours", None),
                "timestep": pt.get("time", ""),
                "lat": pt.get("lat", None),
                "lon": pt.get("lon", None),
                "wind_kts": pt.get("wind_kts", None),
                "pres_hpa": pt.get("pres_hpa", None),
                "category": pt.get("category", "")
            })

    # Save summary CSV
    df_summary = pd.DataFrame(summary_records)
    df_summary.to_csv(SUMMARY_CSV, index=False)

    # Save tracks CSV
    df_tracks = pd.DataFrame(track_records)
    df_tracks.to_csv(TRACKS_CSV, index=False)

    # Save full catalog CSV
    df_catalog = pd.DataFrame(catalog_flattened)
    df_catalog.to_csv(CATALOG_CSV, index=False)

    print("=" * 65)
    print("  NIO CYCLONE CATALOG -> CSV EXPORT COMPLETED")
    print("=" * 65)
    print(f"  [1] Storms Summary CSV : {SUMMARY_CSV}")
    print(f"      Rows: {len(df_summary):,} cyclones")
    print(f"  [2] Tracks CSV         : {TRACKS_CSV}")
    print(f"      Rows: {len(df_tracks):,} track points")
    print(f"  [3] Full Catalog CSV   : {CATALOG_CSV}")
    print(f"      Rows: {len(df_catalog):,} records")
    print("=" * 65)


if __name__ == "__main__":
    convert_catalog()
