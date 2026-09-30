# CycloneVision AI

CycloneVision processes satellite imagery and provides cyclone detection and track-forecasting components. This public repository contains source code, documentation, and tests. Satellite datasets, live imagery, credentials, generated outputs, and model checkpoints are intentionally excluded.

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your own MOSDAC credentials locally. Never commit `.env`.

## Model 1 Data

Model 1 expects `cyclone_image/Cyclone_Images.h5` with an `Images` dataset shaped `(N, 128, 128, 4)`. This dataset is not included. Verified image-level labels and storm-disjoint splits are required in `cyclone_image/image_annotations.csv`; see [CYCLONEVISION_PROJECT_EXPLANATION_GUIDE.md](CYCLONEVISION_PROJECT_EXPLANATION_GUIDE.md) for the manifest columns and eyewall-mask format.

The existing local Model 1 checkpoint is not included because its training provenance and metrics are unverified. The trainer refuses to train without a verified annotation manifest. The HDF5 images in the local development workspace do not include the timestamp/location mapping needed to derive that manifest automatically from IBTrACS.

## Tests

```powershell
python -m unittest discover -s tests -v
```
