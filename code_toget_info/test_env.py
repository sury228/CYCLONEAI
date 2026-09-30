packages = ['numpy', 'pandas', 'h5py', 'matplotlib', 'xarray', 'netCDF4', 'torch', 'requests', 'scipy', 'cdsapi', 'sklearn']

for pkg in packages:
    try:
        __import__(pkg)
        print(f"  [+] {pkg:<12} : Available")
    except ImportError:
        print(f"  [-] {pkg:<12} : Not Installed")
