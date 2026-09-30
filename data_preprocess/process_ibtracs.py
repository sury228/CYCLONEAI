"""
CycloneVision AI - IBTrACS Ground Truth Best-Track Processor
Parses NOAA / IMD Best Track Dataset for North Indian Ocean (NI Basin: Bay of Bengal & Arabian Sea).
"""

import os
import pandas as pd
import numpy as np
import json

# IMD Cyclone Intensity Scale Mapping
IMD_SCALE = {
    (0, 17): ("Low Pressure Area", "LPA", 0),
    (17, 27): ("Depression", "D", 1),
    (28, 33): ("Deep Depression", "DD", 2),
    (34, 47): ("Cyclonic Storm", "CS", 3),
    (48, 63): ("Severe Cyclonic Storm", "SCS", 4),
    (64, 89): ("Very Severe Cyclonic Storm", "VSCS", 5),
    (90, 119): ("Extremely Severe Cyclonic Storm", "ESCS", 6),
    (120, 999): ("Super Cyclonic Storm", "SuCS", 7)
}

def get_imd_category(wind_kts):
    """Classifies wind speed (in knots) according to official IMD categories."""
    if pd.isna(wind_kts) or wind_kts == '':
        return "Unknown", "UNK", -1
    try:
        w = float(wind_kts)
    except ValueError:
        return "Unknown", "UNK", -1
    for (low, high), (cat_name, abbr, code) in IMD_SCALE.items():
        if low <= w <= high:
            return cat_name, abbr, code
    return "Unknown", "UNK", -1


def load_and_clean_ibtracs(csv_path="ibtracs.NI.list.v04r01.csv", min_year=1990):
    """
    Loads and cleans IBTrACS CSV data for the North Indian Ocean basin.
    Filters from min_year onwards (modern satellite era by default).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"IBTrACS file not found at: {csv_path}")

    print(f"[*] Reading IBTrACS North Indian Ocean data from {csv_path}...")
    # Row 0 contains column names, Row 1 contains units
    df = pd.read_csv(csv_path, skiprows=[1], low_memory=False)

    # Convert ISO_TIME to datetime
    df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
    df['SEASON'] = pd.to_numeric(df['SEASON'], errors='coerce')
    
    # Filter by modern satellite era
    df_filtered = df[df['SEASON'] >= min_year].copy()
    print(f"[+] Total records since {min_year}: {len(df_filtered)}")

    # Clean numeric coordinates and wind/pressure columns
    # Fallback order: IMD columns -> USA (JTWC) columns -> WMO columns
    def coalesce_col(row, primary, fallback, fallback2):
        for col in [primary, fallback, fallback2]:
            if col in row and pd.notna(row[col]) and str(row[col]).strip() != '':
                try:
                    return float(row[col])
                except (ValueError, TypeError):
                    pass
        return np.nan

    df_filtered['CLEAN_LAT'] = [coalesce_col(r, 'IMD_LAT', 'USA_LAT', 'LAT') for _, r in df_filtered.iterrows()]
    df_filtered['CLEAN_LON'] = [coalesce_col(r, 'IMD_LON', 'USA_LON', 'LON') for _, r in df_filtered.iterrows()]
    df_filtered['CLEAN_WIND_KTS'] = [coalesce_col(r, 'IMD_WIND', 'USA_WIND', 'WMO_WIND') for _, r in df_filtered.iterrows()]
    df_filtered['CLEAN_PRES_HPA'] = [coalesce_col(r, 'IMD_PRES', 'USA_PRES', 'WMO_PRES') for _, r in df_filtered.iterrows()]

    # Drop rows without valid lat/lon
    df_clean = df_filtered.dropna(subset=['CLEAN_LAT', 'CLEAN_LON']).copy()

    # Assign IMD Category
    categories = [get_imd_category(w) for w in df_clean['CLEAN_WIND_KTS']]
    df_clean['IMD_CATEGORY'] = [c[0] for c in categories]
    df_clean['IMD_CAT_CODE'] = [c[2] for c in categories]

    print(f"[+] Cleaned records with valid spatial coordinates: {len(df_clean)}")
    return df_clean


def extract_named_cyclones(df, min_year=2015):
    """
    Groups tracks by Storm Identifier (SID) / Name and calculates track statistics.
    """
    df_recent = df[df['SEASON'] >= min_year].copy()
    unique_storms = df_recent.groupby('SID')
    
    storm_catalog = []
    print(f"\n[*] Extracting major North Indian Ocean Cyclones (>= {min_year}):")
    
    for sid, group in unique_storms:
        name = group['NAME'].iloc[0]
        season = int(group['SEASON'].iloc[0])
        max_wind = group['CLEAN_WIND_KTS'].max()
        min_pres = group['CLEAN_PRES_HPA'].min()
        start_time = group['ISO_TIME'].min()
        end_time = group['ISO_TIME'].max()
        duration_hours = (end_time - start_time).total_seconds() / 3600.0
        
        # Determine highest intensity reached
        top_cat, _, _ = get_imd_category(max_wind)
        
        track_points = []
        for _, row in group.iterrows():
            track_points.append({
                "time": str(row['ISO_TIME']),
                "lat": float(row['CLEAN_LAT']),
                "lon": float(row['CLEAN_LON']),
                "wind_kts": float(row['CLEAN_WIND_KTS']) if pd.notna(row['CLEAN_WIND_KTS']) else None,
                "pres_hpa": float(row['CLEAN_PRES_HPA']) if pd.notna(row['CLEAN_PRES_HPA']) else None,
                "category": row['IMD_CATEGORY']
            })

        storm_summary = {
            "sid": str(sid),
            "name": str(name),
            "season": season,
            "max_wind_kts": float(max_wind) if pd.notna(max_wind) else None,
            "min_pres_hpa": float(min_pres) if pd.notna(min_pres) else None,
            "peak_category": top_cat,
            "start_time": str(start_time),
            "end_time": str(end_time),
            "duration_hours": duration_hours,
            "num_timesteps": len(group),
            "track": track_points
        }
        storm_catalog.append(storm_summary)

    print(f"[+] Total named/tracked storms cataloged: {len(storm_catalog)}")
    
    # Sort by max wind speed (descending)
    storm_catalog.sort(key=lambda x: x['max_wind_kts'] if x['max_wind_kts'] is not None else 0, reverse=True)
    
    # Save to JSON for GIS Dashboard consumption
    output_json = "nio_cyclones_catalog.json"
    with open(output_json, "w") as f:
        json.dump(storm_catalog, f, indent=2)
    print(f"[+] Saved Geo-catalog to {output_json}")

    return storm_catalog


if __name__ == "__main__":
    df_clean = load_and_clean_ibtracs("ibtracs.NI.list.v04r01.csv", min_year=2000)
    catalog = extract_named_cyclones(df_clean, min_year=2018)
    
    print("\nTop 5 Most Severe Cyclones in Catalog:")
    for s in catalog[:5]:
        print(f"  • Cyclone {s['name']} ({s['season']}): Peak Wind = {s['max_wind_kts']} kts, Min Pres = {s['min_pres_hpa']} hPa, Category = {s['peak_category']}")
