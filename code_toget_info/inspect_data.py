import h5py
import pandas as pd
import numpy as np

print("=" * 60)
print("1. INSPECTING IBTrACS (Ground Truth Best Track Data)")
print("=" * 60)

try:
    df_ibtracs = pd.read_csv('ibtracs.NI.list.v04r01.csv', low_memory=False, nrows=10)
    print("IBTrACS Columns (Total:", len(df_ibtracs.columns), "):")
    print(df_ibtracs.columns.tolist()[:25])
    print("\nSample rows:")
    cols_to_show = [c for c in ['SID', 'NAME', 'ISO_TIME', 'USA_LAT', 'USA_LON', 'USA_WIND', 'USA_PRES', 'IMD_LAT', 'IMD_LON', 'IMD_WIND', 'IMD_PRES', 'NATURE'] if c in df_ibtracs.columns]
    print(df_ibtracs[cols_to_show].head(5))
except Exception as e:
    print(f"Error reading IBTrACS CSV: {e}")

print("\n" + "=" * 60)
print("2. INSPECTING Cyclone_Images.h5")
print("=" * 60)

try:
    with h5py.File('Cyclone_Images.h5', 'r') as h5f:
        print("HDF5 Keys (Top-level groups/datasets):", list(h5f.keys()))
        for k in h5f.keys():
            item = h5f[k]
            if isinstance(item, h5py.Dataset):
                print(f"  Dataset '{k}': shape = {item.shape}, dtype = {item.dtype}")
            elif isinstance(item, h5py.Group):
                print(f"  Group '{k}' with {len(item.keys())} items. Subkeys sample: {list(item.keys())[:10]}")
                # check one item inside group
                sub_k = list(item.keys())[0]
                sub_item = item[sub_k]
                if isinstance(sub_item, h5py.Dataset):
                    print(f"    Sub-dataset '{sub_k}': shape = {sub_item.shape}, dtype = {sub_item.dtype}")
                elif isinstance(sub_item, h5py.Group):
                    print(f"    Sub-group '{sub_k}': keys = {list(sub_item.keys())[:10]}")
except Exception as e:
    print(f"Error reading H5 file: {e}")
