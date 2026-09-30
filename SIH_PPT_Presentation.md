# Superseded Draft — Use the Fact-Checked Storyboard

This earlier draft contains proposed capabilities and performance claims that are not all supported by the current implementation or evaluation setup. For the current slide-by-slide presentation copy, speaking cues, PNG compositions, and claim caveats, use [SIH_PPT_Slide_Content.md](SIH_PPT_Slide_Content.md).

# Smart India Hackathon (SIH) Presentation Deck
## CycloneVision AI: Tropical Cyclone Identification, Classification & Track Prediction Engine
**Problem Statement ID:** 26070  
**Ministry / Organization:** Ministry of Earth Sciences (MoES) / India Meteorological Department (IMD)  
**Theme:** Disaster Management (Software Category)  
**Target Basin:** North Indian Ocean (Bay of Bengal & Arabian Sea)

---

### ════════════════════════════════════════════════════════════════════════════
### SLIDE 1: TITLE & TEAM DETAILS
### ════════════════════════════════════════════════════════════════════════════

* **Project Title:** CycloneVision AI — Physics-Informed Multi-Modal Cyclone Analysis System
* **Problem Statement ID:** 26070
* **Nodal Ministry / Organization:** Ministry of Earth Sciences (MoES) / India Meteorological Department (IMD)
* **Theme:** Disaster Management
* **Domain:** Remote Sensing, AI/ML, Physics-Informed Neural Networks, GIS & Emergency Response

---

### ════════════════════════════════════════════════════════════════════════════
### SLIDE 2: PROBLEM STATEMENT & PROPOSED SOLUTION
### ════════════════════════════════════════════════════════════════════════════

#### 🔴 The Problem
1. **Severe Weather Threats in NIO:** Bay of Bengal & Arabian Sea account for some of the most destructive tropical cyclones globally (e.g., Cyclone Amphan, Fani, Tauktae).
2. **Manual Forecaster Overhead:** Traditional Dvorak technique relies on manual pattern matching of satellite IR imagery, leading to subjective estimation and delays.
3. **Rapid Intensification (RI) Surprise:** Sudden storm intensification ($>30\text{ kts}$ in 24 hours) is difficult to forecast with numerical models alone.
4. **Data Heterogeneity:** Combining GEO geostationary satellite imagery (INSAT-3D/3DR) with LEO microwave passes (GPM/Oceansat) and NWP grids (ERA5/GFS) is complex.

#### 🟢 Proposed Solution: CycloneVision AI
An **end-to-end multi-modal deep learning platform** that automates:
* **Vortex Identification & Eye Pinpointing:** Sub-pixel localization of storm center.
* **Automated Dvorak Pattern Classification:** Real-time T-Number ($T1.0-T8.0$), Max Sustained Winds ($MSW$), and Central Pressure ($CP$).
* **72-Hour Spatio-Temporal Track & RI Forecasting:** Deep Learning trajectory prediction with automated Rapid Intensification early warnings.
* **Explainable GIS Dashboard:** Interactive Decision Support System (DSS) with Grad-CAM heatmaps for operational meteorologists and NDMA disaster units.

---

### ════════════════════════════════════════════════════════════════════════════
### SLIDE 3: END-TO-END TECHNICAL ARCHITECTURE
### ════════════════════════════════════════════════════════════════════════════

```
                       [ MULTI-SOURCE DATA INGESTION ]
                                      │
  ┌─────────────────────────┬─────────┴───────────────┬─────────────────────────┐
  │ INSAT-3D/3DR (IR/WV)    │ Oceansat-3/SCATSAT      │ GPM / Microwave Imagers │ ERA5 / GFS Reanalysis │
  │ (15-min Geostationary)  │ (Surface Wind Vectors)  │ (Sub-cloud eyewall)     │ (SST, Wind Shear)     │
  └───────────┬─────────────┴─────────┬───────────────┴───────────┬─────────────┴───────────┬───────────┘
              │                       │                           │                         │
              └───────────────────────┴───────────────┬───────────┴─────────────────────────┘
                                                      ▼
                              [ SPATIO-TEMPORAL PREPROCESSING & FUSION LAYER ]
                                                      │
                                   ├─ Geometric Alignment & Resampling (WGS84)
                                   ├─ Cloud Masking & Temperature Calibration
                                   └─ Automated Storm Crop (512x512 Patch Extraction)
                                                      │
                                                      ▼
                                           [ AI / ML CORE ENGINE ]
                                                      │
         ┌────────────────────────────────────────────┼────────────────────────────────────────────┐
         ▼                                            ▼                                            ▼
┌───────────────────────────┐            ┌───────────────────────────┐            ┌───────────────────────────┐
│ 1. Identification Module  │            │ 2. Classification Module  │            │ 3. Prediction Engine      │
├───────────────────────────┤            ├───────────────────────────┤            ├───────────────────────────┤
│ • YOLOv8-Seg / UNet-3D    │            │ • Swin-Transformer / ResNet│            │ • Phy-ConvLSTM / ST-GNN   │
│ • Detects Vortex Center   │            │ • Pattern Classification  │            │ • 6h - 72h Track Forecast │
│ • Pinpoints Eye (Lat/Lon) │            │ • Auto-Dvorak T-Number    │            │ • Rapid Intensification   │
│                           │            │ • MSW (knots) & hPa       │            │   (RI >30kt/24h Warning)  │
└────────┬──────────────────┘            └────────────┬──────────────┘            └────────────┬──────────────┘
         │                                            │                                        │
         └────────────────────────────────────────────┼────────────────────────────────────────┘
                                                      ▼
                              [ EXPLAINABLE AI (XAI) & METEOROLOGICAL VALIDATION ]
                                                      │
                                   ├─ Grad-CAM / Attention Map Overlays
                                   └─ Automated Digital Dvorak Worksheet Generation
                                                      │
                                                      ▼
                              [ REAL-TIME GIS DASHBOARD & ALERT SYSTEM ]
                                                      │
                                   ├─ Interactive Leaflet/Mapbox Map Layers
                                   ├─ Storm Path & Landfall Radius Heatmap
                                   └─ SMS / WhatsApp / Webhook Alerts to NDMA/IMD
```

---

### ════════════════════════════════════════════════════════════════════════════
### SLIDE 4: KEY TECHNICAL CHALLENGES & NOVEL MITIGATIONS
### ════════════════════════════════════════════════════════════════════════════

| # | Key Challenge | Cause | Innovation / Mitigation |
|---|---------------|-------|-------------------------|
| **1** | **Multi-Source Data Heterogeneity** | GEO geostationary (INSAT 15-min cadence) vs LEO Microwave (2 passes/day). | **Cross-Modal Transformer Fusion Network**: Uses cross-attention to align sparse high-res microwave features onto continuous GEO IR time-series grids. |
| **2** | **Optical Cloud Occlusion** | Upper-level cirrus clouds obscure eyewall structure in IR/VIS channels. | **Microwave Imager Integration**: Uses 37 GHz & 89 GHz passive microwave channels to penetrate cloud tops and measure sub-cloud rain rates. |
| **3** | **Rapid Intensification (RI) Prediction** | Highly non-linear events ($>30\text{ kts}/24\text{h}$) causing catastrophic surprises. | **Physics-Informed Neural Network (PINN)**: Embeds thermodynamic constraints ($SST > 26.5^\circ\text{C}$, Ocean Heat Content, Wind Shear) into loss function $L_{total} = L_{data} + \lambda L_{physics}$. |
| **4** | **Class Imbalance in Severe Storms** | Severe/Super Cyclonic Storms represent $<5\%$ of historical satellite data. | **Conditional Diffusion (DDPM) & Focal Loss**: Synthesizes realistic severe storm frames and uses Focal Loss to weight rare extreme events. |
| **5** | **Forecaster Trust & Explainability** | Black-box neural network outputs cannot be audited by meteorologists. | **Digital Dvorak Worksheet + Grad-CAM XAI**: Overlays attention maps showing temperature gradients between storm eye and cold ring eyewall. |
| **6** | **Operational Low Latency** | Gigabytes of satellite NetCDF4/HDF5 data must be processed in real-time. | **TensorRT Engine & Async Pipeline**: Model quantization (INT8/FP16) on GPU microservices with Kafka, achieving $<30$ seconds inference time. |

---

### ════════════════════════════════════════════════════════════════════════════
### SLIDE 5: FEASIBILITY & RECOMMENDED TECH STACK
### ════════════════════════════════════════════════════════════════════════════

* **Satellite Data Ingestion & GIS Processing:** Python, `xarray`, `netCDF4`, `rasterio`, `GDAL`, SatPy
* **AI/ML Engine:** PyTorch, PyTorch Lightning, Ultralytics YOLOv8, Swin Transformer 3D, PyTorch Geometric (ST-GNN)
* **Explainable AI (XAI):** Captum, Grad-CAM, SHAP
* **Backend Architecture:** FastAPI, Celery, Redis, Apache Kafka (Data Streaming)
* **Interactive Frontend & GIS Dashboard:** React / Next.js, Leaflet.js / Mapbox GL JS, TailwindCSS, Turf.js
* **Database & Spatial Queries:** PostgreSQL + PostGIS

---

### ════════════════════════════════════════════════════════════════════════════
### SLIDE 6: IMPACT, DISASTER RESPONSE & BENEFICIARIES
### ════════════════════════════════════════════════════════════════════════════

1. **MoES / IMD Operational Enhancement:** Reduces manual Dvorak analysis time from 30+ minutes per satellite pass to under 30 seconds.
2. **NDMA / SDMA Early Evacuation:** 72-hour track forecasts and Rapid Intensification alerts provide actionable lead-time for coastal state administrations (Odisha, West Bengal, Andhra Pradesh, Gujarat, Maharashtra).
3. **Saves Lives & Economic Assets:** Precise landfall location and wind-swath heatmaps reduce unnecessary mass evacuations while safeguarding coastal infrastructure, ports, and fishermen.

---

### ════════════════════════════════════════════════════════════════════════════
### SLIDE 7: IMPLEMENTATION ROADMAP & FUTURE SCOPE
### ════════════════════════════════════════════════════════════════════════════

* **Phase 1 (Prototype - Current):** Data ingestion pipeline (`HDF5`, `IBTrACS`, `ERA5`), initial multi-modal fusion, and interactive GIS Dashboard prototype.
* **Phase 2 (Hackathon Finale):** Integration of Swin-Transformer for Dvorak T-number estimation and PINN for 72h track prediction.
* **Phase 3 (Deployment):** Operational integration with IMD INSAT-3D/3DR live data stream, NDMA API webhooks, and multi-lingual public bulletin generator.
