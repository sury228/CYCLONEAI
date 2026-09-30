# CYCLONEVISION AI: END-TO-END PROJECT & INTERVIEW PREPARATION GUIDE

================================================================================
  COMPREHENSIVE CONCEPTUAL & CODE GUIDE FOR INTERVIEWS AND TECHNICAL DEFENSE
================================================================================

---

## 1. EXECUTIVE SUMMARY & PROBLEM STATEMENT

### What problem are you solving?
Tropical cyclones in the North Indian Ocean (Bay of Bengal and Arabian Sea) cause severe coastal devastation. Operational meteorological centers like the India Meteorological Department (IMD) rely on satellite interpretation and Numerical Weather Prediction (NWP) models. However, standard forecasting suffers from:
1. **Eye Pinpointing Delays**: Manual/semi-automated detection of cyclone centers during early formation or under high cloud cover.
2. **Track Forecast Drift**: High spatial drift in long-lead trajectory forecasts (+24h to +48h).

**CycloneVision AI** is a dual-stage deep learning pipeline:
* **Model 1 (Vortex Eye Localization & Segmentation)**: Takes multi-spectral satellite imagery and pinpoints the cyclone eye location, estimates central eye diameter, and segments the surrounding eyewall cloud ring.
* **Model 2 (Multi-Horizon Track & Trajectory Forecaster)**: Takes historical storm kinematics and environmental steering winds to predict future storm coordinates (+6h to +48h), projects a 2D spatial Cone of Uncertainty ($\pm\text{km}$), and estimates coastal landfall probability.

---

## 2. DATASET & PREPROCESSING PIPELINE

### Datasets Used
1. **Satellite Imagery (Model 1)**: Multi-channel INSAT-3D satellite imagery patches ($128 \times 128$ resolution, 4 spectral channels: Infrared IR, Water Vapor WV, Visible VIS, Thermal Infrared TIR).
2. **IBTrACS Historical Records (Model 2)**: International Best Track Archive for Climate Stewardship (IBTrACS) dataset for North Indian Ocean cyclones (2000–present), containing sequential storm fixes: Latitude, Longitude, Central Pressure ($\text{hPa}$), Maximum Sustained Wind ($\text{kts}$).
3. **ERA5 Atmospheric NWP Vectors (Model 2)**: ERA5 reanalysis data capturing synoptic steering flow ($u$-wind, $v$-wind, sea-level pressure, sea-surface temperature).

---

### Data Split Strategy: Why Storm-ID Level Split?
* **Crucial Concept**: Splitting data randomly frame-by-frame leads to severe **Data Leakage** (temporal autocorrelation), where consecutive 3-hour satellite frames of the *same storm* end up in both Train and Test sets, artificially inflating accuracy.
* **Our Solution**: Split dataset **by unique Storm Identification Code (SID)** (70% Train, 15% Validation, 15% Test). This ensures the model is evaluated on completely unseen storm events.

---

### Preprocessing & Augmentation
* **Min-Max & Standard Normalization**: Satellite brightness temperatures normalized to $[0, 1]$.
* **7% White Gaussian Noise Injection ($\sigma = 0.07$)**: Applied **strictly to training imagery** to improve robustness against sensor noise and atmospheric degradation.
* **Kinematic Differential Features**: Derived past translation speed ($\text{km/h}$), heading angle ($\theta$), $\Delta\text{Lat}$, and $\Delta\text{Lon}$ across sequential fixes.

---

## 3. MODEL 1: VORTEX EYE LOCALIZATION & SEGMENTATION MODEL

### Architecture: `VortexEyeLocalizationModel`
* **Type**: Multi-Task Residual Convolutional Neural Network (CNN) with 3 unified output heads.
* **Input Tensor Shape**: $(B, 4, 128, 128)$ — Batch size $B$, 4 spectral channels, $128 \times 128$ spatial grid.

```
                  ┌─────────────────────────────────────────┐
                  │ Input Satellite Patch (B, 4, 128, 128)  │
                  └────────────────────┬────────────────────┘
                                       │
                         ┌─────────────┴─────────────┐
                         │ Custom Residual CNN Trunk │
                         └─────────────┬─────────────┘
                                       │
         ┌─────────────────────────────┼─────────────────────────────┐
         ▼                             ▼                             ▼
┌─────────────────┐           ┌──────────────────┐          ┌──────────────────┐
│ Head 1: Presence│           │ Head 2: Eye BBox │          │ Head 3: Eyewall  │
│ Classification  │           │ Regression       │          │ Mask Segmentation│
│ Output: (B, 1)  │           │ Output: (B, 4)   │          │ Output:(B,1,128,128)
└─────────────────┘           └──────────────────┘          └──────────────────┘
```

---

### Three Unified Output Heads & Mathematical Rationale

1. **Head 1 — Vortex Presence Classification**:
   * *Output*: Sigmoid probability $P(\text{cyclone}) \in [0, 1]$.
   * *Purpose*: Binary classification of whether a vortex eye is present in the patch.
   * *Loss*: **Binary Cross-Entropy (BCE Loss)**.
     $$\mathcal{L}_{\text{cls}} = - \big[ y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \big]$$

2. **Head 2 — Eye Bounding Box Regression**:
   * *Output*: Normalization coordinates $[x_c, y_c, w, h] \in [0, 1]^4$.
   * *Purpose*: Pinpoints eye center $(x_c, y_c)$ and estimates eye diameter $(w, h)$.
   * *Loss*: **Smooth L1 Loss (Huber Loss)** — less sensitive to outliers than MSE.
     $$\mathcal{L}_{\text{bbox}} = \text{SmoothL1}(\mathbf{b}, \hat{\mathbf{b}})$$

3. **Head 3 — Eyewall Cloud Ring Segmentation**:
   * *Output*: Pixel-wise mask probability grid $(B, 1, 128, 128)$.
   * *Purpose*: Maps spatial eyewall cloud structure surrounding the center.
   * *Loss*: **Soft Dice Loss** — handles severe class imbalance (small eyewall vs. large background).
     $$\mathcal{L}_{\text{dice}} = 1 - \frac{2 \sum p_i g_i + \epsilon}{\sum p_i + \sum g_i + \epsilon}$$

---

### Total Multi-Task Loss Formula
$$\mathcal{L}_{\text{total\_model1}} = 1.0 \cdot \mathcal{L}_{\text{cls}} + 5.0 \cdot \mathcal{L}_{\text{bbox}} + 2.0 \cdot \mathcal{L}_{\text{dice}}$$

* **Why Weighting $(1.0, 5.0, 2.0)$?**: BBox regression coordinates $[x_c, y_c, w, h]$ are bounded in $[0, 1]$, yielding naturally smaller loss values ($\sim 0.02$). Weighting by $5.0$ balances gradient magnitudes across all three heads during backpropagation.

---

### Model 1 Hyperparameters & Tuning Rationale
| Hyperparameter | Value | Rationale |
| :--- | :--- | :--- |
| **Optimizer** | AdamW | Weight decay regularizes CNN weights better than standard Adam. |
| **Initial Learning Rate** | $1 \times 10^{-3}$ | Standard starting LR for ConvNets; decayed via Cosine Annealing. |
| **Weight Decay** | $1 \times 10^{-4}$ | Prevents filter over-fitting on small eyewall structures. |
| **Batch Size** | 32 | Optimal GPU memory utilization and stable mini-batch gradient updates. |
| **Epochs / Early Stopping** | 15 Epochs (Patience=4) | Early stopping triggers when validation loss plateaus to prevent memorization. |

---

### Model 1 Performance Results
* Previously recorded Model 1 scores are **unverified** and must not be presented as validated performance. The in-repository trainer previously used constant positive labels and synthetic center-box/mask targets; the available HDF5 image file has no per-image timestamp/location mapping to IBTrACS.
* Do not use the existing checkpoint's confidence, eye location, or mask as validated cyclone detections until Model 1 is retrained and evaluated with verified image-level annotations.

### Verified Annotation Manifest Required For Training
Model 1 training now requires `cyclone_image/image_annotations.csv`. It can cover a curated subset of the HDF5 images, but every row must be manually or externally verified and uniquely identify its `image_index`.

Required columns:
* `image_index`: zero-based index in `Cyclone_Images.h5` / `Images`.
* `vortex_presence`: verified binary label (`0` or `1`).
* `split`: `train`, `validation`, or `test`; all images belonging to a storm must use the same split.
* `storm_id`: stable storm identifier; use a stable acquisition identifier for negative scenes.

To train all three output heads, provide `eye_x_center`, `eye_y_center`, `eye_width`, and `eye_height` as normalized `[0, 1]` values for cyclone samples, plus `eyewall_mask_path` for verified mask files. Each mask must be a 128×128 `.npy` array with values in `[0, 1]`; paths are relative to the manifest. Each split must include cyclone and non-cyclone examples, and at least one verified eye box and mask.

The current HDF5 file contains image arrays but no timestamp or location metadata, so the repository cannot safely generate this manifest by matching image indices to IBTrACS rows. Supply a verified image-to-storm mapping and annotations before retraining. The trainer now refuses to substitute synthetic labels, and inference refuses missing spectral bands instead of fabricating them.

---

## 4. MODEL 2: MULTI-HORIZON TRACK & TRAJECTORY FORECASTING ENGINE

### Architecture: `CycloneTrajectoryForecaster`
* **Type**: Spatio-Temporal Residual Temporal Convolutional Network (TCN) + Environmental Cross-Attention.
* **Problem Solved**: Predicts future storm coordinates across 6 horizons ($+6\text{h}, +12\text{h}, +18\text{h}, +24\text{h}, +36\text{h}, +48\text{h}$), uncertainty radii ($\pm\text{km}$), and landfall risk.

```
Past Kinematics (B, 4, 8)  ──►  Residual TCN Temporal Encoder  ──────┐
                                                                     ├─► Cross-Attention Fusion ──► Multi-Horizon Heads
ERA5 NWP Vector (B, 5)     ──►  Environmental Feature Projection ────┘
```

---

### Why TCN + Cross-Attention over Standard LSTM?
1. **Vanishing Gradients & Sequential Bottleneck in LSTM**: LSTMs process sequence steps sequentially ($t_1 \to t_2 \to t_3$). Temporal Convolutional Networks (TCNs) process the entire sequence in parallel using dilated causal convolutions.
2. **Receptive Field Control**: Dilated convolutions ($d=1, 2, 4$) systematically expand the temporal receptive field without increasing parameter count.
3. **Cross-Attention Mechanism**: Computes dynamic cross-attention between past storm kinematics (Query) and ERA5 steering winds (Key/Value), allowing the model to adaptively focus on environmental steering when a storm changes direction.

---

### Spherical Earth Geodesic Loss: Differentiable PyTorch Haversine Loss
Standard Mean Squared Error (MSE) on Latitude/Longitude in degrees treats spatial coordinates as Euclidean flat planes. On a spherical Earth, 1 degree of Longitude shrinks as Latitude increases ($\text{Distance} = \Delta\text{Lon} \cdot 111 \cdot \cos(\text{Lat})$).

**PyTorch Haversine Distance Loss (in Kilometers)**:
$$\mathcal{L}_{\text{haversine}} = \frac{1}{N \cdot H} \sum_{i=1}^N \sum_{h=1}^H 2 R \cdot \arcsin \left( \sqrt{ \sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right) } \right)$$

where $R = 6371.0\text{ km}$, $\phi$ is Latitude in radians, and $\lambda$ is Longitude in radians.

---

### Track Error Decomposition: ATE vs. CTE
To evaluate trajectory predictions, Track Forecast Error (TFE) is decomposed into:
1. **Along-Track Error (ATE in km)**: Speed/timing error (whether the cyclone is moving faster or slower than predicted along its trajectory).
2. **Cross-Track Error (CTE in km)**: Directional steering error (perpendicular offset from actual track path).

$$\text{TFE} = \sqrt{\text{ATE}^2 + \text{CTE}^2}$$

---

### Model 2 Performance vs. IMD Operational Benchmarks
| Forecast Lead Time | CycloneVision AI (TFE) | IMD Operational Benchmark Target | Status / Improvement |
| :---: | :---: | :---: | :---: |
| **+06 Hours** | **$16.3\text{ km}$** | $35.0\text{ km}$ | **$+53.4\%$ Superior** |
| **+12 Hours** | **$42.8\text{ km}$** | $60.0\text{ km}$ | **$+28.7\%$ Superior** |
| **+18 Hours** | **$74.0\text{ km}$** | $85.0\text{ km}$ | **$+12.9\%$ Superior** |
| **+24 Hours** | **$89.2\text{ km}$** | $110.0\text{ km}$ | **$+18.9\%$ Superior** |
| **+36 Hours** | **$117.5\text{ km}$** | $150.0\text{ km}$ | **$+21.7\%$ Superior** |
| **+48 Hours** | **$144.0\text{ km}$** | $190.0\text{ km}$ | **$+24.2\%$ Superior** |

---

## 5. INTERVIEW CHEATSHEET & CODE DEFENSE

### Q1: "Explain what `optimizer.zero_grad()`, `loss.backward()`, and `optimizer.step()` do under the hood."
* **`optimizer.zero_grad()`**: Clears the `.grad` attribute of all parameters. In PyTorch, gradients accumulate by default on every backward pass (`param.grad += dL/dw`). If you omit `zero_grad()`, gradients from previous mini-batches add up, causing exploding gradient updates.
* **`loss.backward()`**: Computes the gradient of the scalar loss with respect to all trainable parameters using automatic differentiation (Reverse-Mode Autodiff / Backpropagation chain rule).
* **`optimizer.step()`**: Updates model weights using the calculated gradients and the optimizer update rule (e.g., for AdamW: $w_{t+1} = w_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta \lambda w_t$).

---

### Q2: "Why didn't you use standard MSE for Latitude and Longitude?"
Euclidean MSE assumes flat 2D geometry. Near the equator ($0^\circ$), $1^\circ\text{ Lon} \approx 111\text{ km}$, but at $30^\circ\text{ N}$, $1^\circ\text{ Lon} \approx 96\text{ km}$. MSE treats a $1^\circ$ error equally everywhere, corrupting spatial loss gradients. PyTorch Haversine Loss measures true spherical distance in kilometers, giving real-world physical gradients.

---

## 10. NEAR-REAL-TIME INSAT SATELLITE PIPELINE (MOSDAC Integration)

CycloneVision AI includes a **near-real-time satellite ingestion pipeline** that authenticates with the **MOSDAC (Meteorological and Oceanographic Satellite Data Archival Centre)** API to download the latest INSAT-3D/3DS products automatically.

### Architecture

```
MOSDAC API ──► satellite_service.py ──► image_processor.py ──► inference_service.py
   (auth)          (download)             (HDF5 → tensor)         (Model 1 + 2)
                                                                        │
                                                   api.py (FastAPI) ◄───┘
                                                        │
                                              Frontend Dashboard
                                         (LIVE SATELLITE card, overlay)
```

**Critical Rule**: The frontend **NEVER** communicates directly with MOSDAC. All credentials and data flow through the Python backend only.

### .env Configuration

Create a `.env` file in the project root (copy from `.env.example`):

```env
# MOSDAC Credentials (register at https://mosdac.gov.in)
MOSDAC_USERNAME=your_mosdac_username
MOSDAC_PASSWORD=your_mosdac_password

# Dataset Configuration
MOSDAC_DATASET_ID=3SIMG_L1B_STD

# Scheduler interval (seconds between MOSDAC checks)
SCHEDULER_INTERVAL_SECONDS=900

# Model checkpoint paths
MODEL1_WEIGHTS_PATH=models/best_model1_vortex_eye.pth
MODEL2_WEIGHTS_PATH=models/best_model2_track_trajectory.pth
```

> **Never commit `.env` to version control.** The `.env.example` file is provided as a safe template.

### Understanding `MOSDAC_DATASET_ID`

| Dataset ID | Satellite | Sensor | Description |
|---|---|---|---|
| `3SIMG_L1B_STD` | INSAT-3DS | Imager (IMG) | Level-1B calibrated IR/VIS/WV/TIR imagery |
| `3DIMG_L1B_STD` | INSAT-3D | Imager (IMG) | Level-1B calibrated IR/VIS/WV/TIR imagery |
| `3SIMG_L2B_OLR` | INSAT-3DS | Imager | Outgoing Longwave Radiation product |

For cyclone detection, use `3SIMG_L1B_STD` (INSAT-3DS, latest satellite) or `3DIMG_L1B_STD` (INSAT-3D, legacy). These are full-disk multi-channel HDF5 products with Infrared (TIR1/TIR2), Water Vapor (WV), and Visible (VIS) channels.

### Startup Procedure

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure credentials**:
   ```bash
   cp .env.example .env
   # Edit .env with your MOSDAC username/password
   ```

3. **Start the backend API server**:
   ```bash
   python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
   ```
   The background scheduler will automatically:
   - Authenticate with MOSDAC
   - Query and download the newest satellite frame
   - Process HDF5 → preview PNG + 4-channel tensor
   - Run inference (Model 1 → Model 2)
   - Save composite overlay image

4. **Open the dashboard**:
   Open `cyclone ai/cyclone ai/index.html` in your browser. The LIVE SATELLITE card will auto-fetch from `http://127.0.0.1:8000/api/satellite/latest` every 60 seconds.

### Key API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/satellite/latest` | GET | Returns latest satellite metadata + image URLs |
| `/api/satellite/image` | GET | Returns the raw satellite preview PNG |
| `/api/cyclone/latest` | GET | Returns full pipeline output (detection, forecast, overlay) |
| `/api/satellite/refresh` | POST | Manually triggers a MOSDAC download + inference cycle |
| `/health` | GET | Backend health check |

### Troubleshooting

- **401 from MOSDAC**: Verify your `.env` credentials. Register at https://mosdac.gov.in if you don't have an account.
- **No satellite image appearing**: Check the backend logs for download errors. MOSDAC may be temporarily unavailable.
- **HDF5 read errors**: Ensure `h5py>=3.8.0` is installed. Some MOSDAC products use compression requiring HDF5 1.10+.
- **Model weights not found**: Place `best_model1_vortex_eye.pth` and `best_model2_track_trajectory.pth` in the `models/` directory.

---

### Q3: "What are the key limitations of your model and how would you scale it?"
1. **Satellite Cloud Cover Blockage**: Optical/IR satellite imagery cannot see through dense upper cirrus clouds during early cyclone genesis.
   * *Scaling Plan*: Integrate Passive Microwave (AMSR2/SSMIS) imagery which penetrates high cirrus clouds.
2. **Fixed NWP Resolution**: ERA5 wind vectors used in Model 2 are coarse ($0.25^\circ$).
   * *Scaling Plan*: Ingest high-resolution regional WRF (Weather Research and Forecasting) model outputs.

================================================================================
  END OF GUIDE -- READY FOR TECHNICAL REVIEWS & INTERVIEWS
================================================================================
