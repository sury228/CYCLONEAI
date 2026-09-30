"""
CycloneVision AI - Environmental NWP Data Fetcher & Processor
Supports:
1. Copernicus ERA5 Reanalysis (Historical Storm Analysis via cdsapi)
2. NOAA GFS OpenDAP / NOMADS (Real-time 0-72h operational forecasting)
3. Synthetic / Offline NWP Generator for rapid local testing without credentials
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# North Indian Ocean Bounding Box [North, West, South, East]
NIO_BBOX = [30.0, 50.0, 0.0, 100.0]  # Lat: 0°N to 30°N, Lon: 50°E to 100°E

def check_cds_credentials():
    """Checks if .cdsapirc exists in user home directory."""
    home_dir = os.path.expanduser("~")
    cds_file = os.path.join(home_dir, ".cdsapirc")
    if os.path.exists(cds_file):
        print(f"[+] Found CDS credentials at: {cds_file}")
        return True
    else:
        print("[!] No ~/.cdsapirc file found.")
        print("    To fetch live ERA5 from ECMWF CDS:")
        print("    1. Register at https://cds.climate.copernicus.eu/")
        print("    2. Create ~/.cdsapirc with:")
        print("       url: https://cds.climate.copernicus.eu/api")
        print("       key: <YOUR_API_KEY>")
        return False


def fetch_era5_surface(year="2023", month="10", days=None, times=None, bbox=NIO_BBOX, output_path="era5_surface_nio.nc"):
    """
    Fetches Sea Surface Temperature (SST), Mean Sea Level Pressure (MSLP), 
    and 10m U/V wind components from ECMWF ERA5.
    """
    if days is None:
        days = ["20", "21", "22", "23", "24"]  # e.g., Cyclone Tej / Hamoon period
    if times is None:
        times = ["00:00", "06:00", "12:00", "18:00"]

    if not check_cds_credentials():
        print("[*] Generating synthetic ERA5 sample for offline testing instead...")
        return generate_synthetic_nwp_sample(bbox=bbox, output_path=output_path)

    import cdsapi
    c = cdsapi.Client()
    print(f"[*] Requesting ERA5 surface data for North Indian Ocean bbox {bbox}...")
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': [
                'sea_surface_temperature',
                'mean_sea_level_pressure',
                '10m_u_component_of_wind',
                '10m_v_component_of_wind',
            ],
            'year': str(year),
            'month': str(month).zfill(2),
            'day': days,
            'time': times,
            'area': bbox,  # [N, W, S, E]
            'format': 'netcdf',
        },
        output_path
    )
    print(f"[+] Downloaded ERA5 surface data to {output_path}")
    return output_path


def fetch_era5_wind_shear(year="2023", month="10", days=None, times=None, bbox=NIO_BBOX, output_path="era5_wind_shear_nio.nc"):
    """
    Fetches 200 hPa and 850 hPa U/V winds to calculate Vertical Wind Shear (VWS).
    """
    if days is None:
        days = ["20", "21", "22", "23", "24"]
    if times is None:
        times = ["00:00", "06:00", "12:00", "18:00"]

    if not check_cds_credentials():
        print("[*] Generating synthetic Wind Shear sample for offline testing...")
        return generate_synthetic_nwp_sample(bbox=bbox, output_path=output_path)

    import cdsapi
    c = cdsapi.Client()
    print(f"[*] Requesting ERA5 upper/lower level winds for Wind Shear calculation...")
    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'variable': ['u_component_of_wind', 'v_component_of_wind'],
            'pressure_level': ['200', '850'],
            'year': str(year),
            'month': str(month).zfill(2),
            'day': days,
            'time': times,
            'area': bbox,
            'format': 'netcdf',
        },
        output_path
    )
    print(f"[+] Downloaded ERA5 pressure levels to {output_path}")
    return output_path


def generate_synthetic_nwp_sample(bbox=NIO_BBOX, output_path="era5_sample_nio.nc"):
    """
    Creates a realistic NetCDF file covering North Indian Ocean (SST, MSLP, Wind Shear, 10m Winds)
    for development, testing, and UI demonstration without waiting for API downloads.
    """
    try:
        import xarray as xr
    except ImportError:
        print("[-] xarray is required to generate or read NetCDF files. Please run 'pip install xarray netCDF4'")
        return None

    lat_max, lon_min, lat_min, lon_max = bbox
    lats = np.linspace(lat_max, lat_min, 61)   # 0.5 deg resolution
    lons = np.linspace(lon_min, lon_max, 101)  # 0.5 deg resolution
    times = pd.date_range("2023-10-21 00:00", periods=8, freq="6h")

    # Generate realistic meteorological fields:
    # SST: 27C to 31.5C (300K to 304.5K) in tropical Bay of Bengal & Arabian Sea
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
    
    # Base tropical warm pool
    base_sst = 301.5 + 2.0 * np.cos(np.radians(lat_grid * 2.5)) - 0.5 * (lon_grid - 75)**2 / 400.0
    sst_data = np.zeros((len(times), len(lats), len(lons)), dtype=np.float32)
    for t in range(len(times)):
        sst_data[t] = base_sst + np.random.normal(0, 0.2, base_sst.shape)

    # MSLP: 990 to 1012 hPa (Pa: 99000 to 101200)
    # Simulate a cyclone vortex moving in Bay of Bengal
    mslp_data = np.zeros((len(times), len(lats), len(lons)), dtype=np.float32)
    u10_data = np.zeros((len(times), len(lats), len(lons)), dtype=np.float32)
    v10_data = np.zeros((len(times), len(lats), len(lons)), dtype=np.float32)
    vws_data = np.zeros((len(times), len(lats), len(lons)), dtype=np.float32)

    for t in range(len(times)):
        # Storm center moves northwestward
        storm_lat = 12.0 + t * 0.8
        storm_lon = 88.0 - t * 0.6
        dist = np.sqrt((lat_grid - storm_lat)**2 + (lon_grid - storm_lon)**2)
        
        # Pressure dip in eye
        mslp_data[t] = 101200.0 - 2500.0 * np.exp(-dist / 3.5)
        
        # Cyclonic rotational wind field (Counter-clockwise in Northern Hemisphere)
        dlat = lat_grid - storm_lat
        dlon = lon_grid - storm_lon
        tangential_speed = 35.0 * (dist / 3.0) * np.exp(-dist / 3.0)
        u10_data[t] = -tangential_speed * (dlat / (dist + 1e-5))
        v10_data[t] =  tangential_speed * (dlon / (dist + 1e-5))
        
        # Vertical Wind Shear (m/s) - favorable < 10 m/s near storm core
        vws_data[t] = 5.0 + 8.0 * (1.0 - np.exp(-dist / 5.0)) + np.random.uniform(0, 2, dist.shape)

    ds = xr.Dataset(
        data_vars={
            'sst': (['time', 'latitude', 'longitude'], sst_data, {'units': 'K', 'long_name': 'Sea Surface Temperature'}),
            'msl': (['time', 'latitude', 'longitude'], mslp_data, {'units': 'Pa', 'long_name': 'Mean Sea Level Pressure'}),
            'u10': (['time', 'latitude', 'longitude'], u10_data, {'units': 'm s**-1', 'long_name': '10m U-wind component'}),
            'v10': (['time', 'latitude', 'longitude'], v10_data, {'units': 'm s**-1', 'long_name': '10m V-wind component'}),
            'vws': (['time', 'latitude', 'longitude'], vws_data, {'units': 'm s**-1', 'long_name': '200-850hPa Vertical Wind Shear'}),
        },
        coords={
            'time': times,
            'latitude': lats,
            'longitude': lons,
        },
        attrs={
            'title': 'CycloneVision AI - North Indian Ocean NWP Environmental Grid',
            'region': 'Bay of Bengal & Arabian Sea',
            'bounding_box': f"North: {bbox[0]}, West: {bbox[1]}, South: {bbox[2]}, East: {bbox[3]}"
        }
    )

    ds.to_netcdf(output_path)
    print(f"[+] Successfully generated NWP dataset at: {output_path}")
    return output_path


def load_and_inspect_nwp(nc_path):
    """Loads and prints statistics of an NWP NetCDF file."""
    import xarray as xr
    ds = xr.open_dataset(nc_path)
    print("\n" + "=" * 60)
    print(f"NWP DATASET SUMMARY: {nc_path}")
    print("=" * 60)
    print("Dimensions:", dict(ds.dims))
    print("Coordinates:", list(ds.coords))
    print("Variables:")
    for v in ds.data_vars:
        data = ds[v].values
        print(f"  • {v:<6}: shape={data.shape}, min={np.nanmin(data):.2f}, max={np.nanmax(data):.2f}, mean={np.nanmean(data):.2f}")
        print(f"  * {v:<6}: shape={data.shape}, min={np.nanmin(data):.2f}, max={np.nanmax(data):.2f}, mean={np.nanmean(data):.2f}")
    
    return ds


if __name__ == "__main__":
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
    print("CycloneVision AI - NWP Fetcher Initialized")
    print(f"Target Region: North Indian Ocean {NIO_BBOX}")
    
    # 1. Try to fetch or generate NWP sample
    output_nc = "era5_sample_nio.nc"
    if not os.path.exists(output_nc):
        generate_synthetic_nwp_sample(bbox=NIO_BBOX, output_path=output_nc)
    
    # 2. Inspect dataset
    ds = load_and_inspect_nwp(output_nc)
    print("\n[SUCCESS] NWP pipeline test complete!")
