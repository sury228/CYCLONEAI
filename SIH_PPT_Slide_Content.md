# CycloneVision AI — SIH Presentation Storyboard

**Purpose:** Copy-ready content for a Smart India Hackathon pitch deck and a matching set of showcase PNGs.
**Project brief:** SIH Problem Statement ID 26070; Disaster Management; MoES / IMD; North Indian Ocean focus. Confirm the exact problem-statement title and organization against the official SIH listing before submission.
**Presentation format:** 16:9 widescreen. Use one visual idea per slide, short labels, and a small footer: `CycloneVision AI | SIH 2026 | Prototype`. Replace team and institute placeholders on Slide 1.

## Accuracy Rules

- This is a prototype, not an IMD-approved or operational forecasting system.
- The saved Model 1 test report shows 100% presence scores, but the dataset builder marks records as cyclones when `IMD_CAT_CODE >= 0`; the project notes also identify an all-positive test-label issue. Do not present those scores as validated detection accuracy.
- The image split is random by sample index (70/15/15), not by storm identity. Image/track pairing and NWP features also need a scientifically aligned data pipeline before performance claims are defensible.
- Model 1's saved test artifact reports Dice = 0.4124 and normalized bounding-box MAE = 0.0552 (about 27.6 km as recorded in the project CSV). Label these as current internal results, not operational accuracy.
- Model 2's saved evaluation artifact reports track errors from 14.56 km at +6h to 100.26 km at +48h. Show only with a visible `internal evaluation; independent storm-level validation pending` qualifier. Do not claim superiority to IMD from the repository's comparison column.
- Live inference currently uses a fixed geographic reference point, fallback/synthetic channel values when bands are absent, and placeholder or derived environmental inputs. A dashboard field is not proof of a calibrated live forecast.
- Features such as PINNs, diffusion augmentation, Grad-CAM, SMS/WhatsApp alerts, and 72-hour prediction are proposals only. The implemented forecast horizons are +6h to +48h.

---

## Slide 1 — Title

**On-slide copy**

> **CycloneVision AI**  
> Satellite-to-track decision support for North Indian Ocean cyclones  
> **SIH Problem Statement 26070**  
> Disaster Management | MoES / IMD  
> Team: `[Team Name]` · Institute: `[Institute Name]`

**Speaking cue:** We are building a prototype that connects satellite-image analysis and historical storm-track forecasting in one demonstrable workflow.

**PNG composition:** Full-bleed, real INSAT-style cloud image from the project's live preview or an appropriately licensed cyclone satellite image. Place the title over a dark, quiet area; add a thin forecast path and small North Indian Ocean locator map. Do not use fabricated detection marks on a real satellite image.

**Asset sources:** `data/live/preview/latest.png` when a genuine processed frame is available; otherwise use a clearly attributed external image. Team details are intentionally placeholders.

---

## Slide 2 — The Problem

**On-slide copy**

> **Cyclone decisions are time-critical**  
> - The Bay of Bengal and Arabian Sea expose densely populated coasts to destructive storms.  
> - Forecasters must interpret changing satellite structure and storm motion under uncertainty.  
> - Data, detection, track guidance, and response information are often viewed in separate tools.  
> **Need:** faster, auditable decision support — with a forecaster in control.

**Speaking cue:** Our goal is not to replace official forecasting. It is to make useful observations and model guidance easier to assemble and inspect.

**PNG composition:** North Indian Ocean map with Bay of Bengal and Arabian Sea labeled, coastline and broad coastal exposure zones. Use a restrained three-stage callout: `Observe` → `Locate` → `Anticipate`. No unsupported casualty, economic-loss, or forecast-delay statistics.

---

## Slide 3 — Our Proposed System

**On-slide copy**

> **One workflow from satellite frame to track view**  
> 1. Ingest an INSAT product  
> 2. Clean and prepare image channels  
> 3. Estimate vortex presence, eye box, and eyewall mask  
> 4. Forecast track points at +6h to +48h  
> 5. Present the result and uncertainty in a dashboard  
> **Human-reviewed decision support, not an autonomous warning authority**

**Speaking cue:** The project joins two trained model components with ingestion, API, and visualization code; the end-to-end integration is the prototype contribution.

**PNG composition:** Horizontal five-step pipeline with a satellite thumbnail, detection overlay, track map, and dashboard crop. Use solid styling for implemented stages and a dotted border where inputs or validation remain provisional.

---

## Slide 4 — Data Foundation

**On-slide copy**

> **Three data families**  
> **Satellite imagery:** HDF5 INSAT products and a project image dataset with 21,076 samples at 128 × 128 × 4  
> **Best tracks:** IBTrACS North Indian Ocean storm positions and intensity records  
> **Environment:** ERA5 sample fields; additional Model 2 environmental features are currently derived or placeholder inputs  
> **Current split:** 70% train · 15% validation · 15% test by image sample index

**Speaking cue:** We have the data loaders and preprocessing path in place. Correct temporal and storm-level alignment is a priority before making strong model-performance claims.

**PNG composition:** Three labeled data tiles feeding a shared preparation band: `HDF5 → channel cleanup/resizing`, `IBTrACS → storm sequences`, `ERA5 → environmental vector`. Put a small amber note under the split: `Image split is not storm-ID isolated yet`.

**Do not say:** The 21,076 images are 21,076 independently labeled cyclones, or that the image and track records are already rigorously paired.

---

## Slide 5 — Model 1: Vortex and Eye Analysis

**On-slide copy**

> **Multi-task image model**  
> Input: 4-channel, 128 × 128 image patch  
> **Output A:** vortex-presence probability  
> **Output B:** normalized eye bounding box `[x, y, width, height]`  
> **Output C:** pixel-level eyewall mask  
> CNN encoder + U-Net-style decoder · PyTorch

**Speaking cue:** A shared image representation supports classification, eye localization, and mask prediction. Geographic conversion is currently an approximate mapping from a configured patch reference.

**PNG composition:** One large satellite patch with bounding box and center marker, paired with a three-head model diagram and a small mask overlay. Label any illustrated detections as `illustrative` unless the image and output are from the same real inference run.

---

## Slide 6 — Model 2: Track Forecast

**On-slide copy**

> **Multi-horizon trajectory model**  
> Past storm fixes + environmental feature vector  
> Temporal convolution encoder + environmental cross-attention  
> **Forecast horizons:** +6h · +12h · +18h · +24h · +36h · +48h  
> Outputs: coordinates · uncertainty radii · landfall probability

**Speaking cue:** The trajectory model uses storm history and environmental features to generate multiple lead-time outputs. Environmental inputs and uncertainty calibration still need stronger real-data validation.

**PNG composition:** A clean Bay of Bengal map: observed track in blue, current fix as a bright marker, forecast points in white/cyan, and translucent widening uncertainty ellipses. Include a compact +6h-to-+48h time ruler. Mark it `model output — prototype`.

---

## Slide 7 — End-to-End Technical Architecture

**On-slide copy**

> **Implemented integration path**  
> MOSDAC catalog / download → HDF5 image processing → Model 1 → Model 2 → FastAPI → Web dashboard  
> - Background polling is configurable (default: 15 minutes)  
> - New frames trigger processing; duplicate frames are skipped  
> - API exposes satellite status, latest inference, image, and overlay  
> - MOSDAC credentials remain on the backend

**Speaking cue:** The backend owns external credentials and coordinates ingestion, processing, inference, and dashboard delivery. Availability still depends on valid MOSDAC access and model/data readiness.

**PNG composition:** A left-to-right architecture diagram with separate data, model, backend, and user-interface bands. Solid arrows indicate code paths present in the repository; dotted arrows indicate external service dependencies. Avoid showing Kafka, Redis, PostGIS, or SMS integrations as implemented.

---

## Slide 8 — What the Operator Sees

**On-slide copy**

> **A single monitoring view**  
> - Satellite frame status, timestamp, and dataset metadata  
> - Detection confidence and estimated eye location  
> - Eye/eyewall overlay and forecast track  
> - Forecast horizons and uncertainty radii  
> - Backend health and manual refresh status  
> Forecast errors appear only when matching future observations are available.

**Speaking cue:** The dashboard is a framework-free HTML/CSS/JavaScript interface connected to FastAPI. It is a prototype interface and should be demonstrated with a genuine backend response, not its mock data.

**PNG composition:** Use an actual screenshot of the dashboard with real API data. Crop around the satellite card, detection summary, and track table; add small numbered callouts. Hide or clearly label demo/mock values. Do not present placeholder map art as a real map.

**Asset sources:** `cyclone ai/cyclone ai/index.html`, `js/app.js`, `js/api.js`, and `js/mockData.js`.

---

## Slide 9 — Current Evaluation Snapshot

**On-slide copy**

> **What the saved evaluation files report**  
> **Model 1:** bounding-box MAE `0.0552` normalized · approx. `27.6 km` as recorded · eyewall Dice `0.4124`  
> **Model 2:** track-error report `14.56 km (+6h)` to `100.26 km (+48h)`  
> **Interpretation:** internal prototype results only. Independent storm-level validation is pending.  
> Model 1's reported 100% presence scores are excluded because test-label balance/validity is unresolved.

**Speaking cue:** We are showing the artifacts honestly, including what they do not establish. We will not claim these results are operational accuracy or superiority to IMD forecasts.

**PNG composition:** Two side-by-side charts. Left: Model 1's reported Dice and box MAE, with a visible `internal` label. Right: track-error points over six horizons. Put a high-contrast footer: `Do not use for public warnings; validation in progress`.

**Asset sources:** `models/vortex_identification_model_1/model1_test_metrics.csv` and `models/track_and_trajectory_model_2/model2_test_metrics.csv`.

---

## Slide 10 — Engineering Contribution and Differentiation

**On-slide copy**

> **What is distinctive in this prototype**  
> - Image analysis combines presence classification, eye-box regression, and eyewall segmentation  
> - Track model forecasts several lead times and exposes uncertainty outputs  
> - MOSDAC ingestion, HDF5 processing, model inference, and dashboard are connected end to end  
> - API and overlay make intermediate outputs inspectable  
> **Design principle:** useful guidance must remain reviewable by a human forecaster.

**Speaking cue:** The differentiator is the connected workflow and its inspectable outputs, not a claim that every proposed research technique has already been implemented.

**PNG composition:** Show four proof points as a compact workflow strip: actual model diagram, real overlay, API payload snippet, and dashboard screenshot. Use only repository-backed visuals.

---

## Slide 11 — Feasibility, Risks, and Next Validation

**On-slide copy**

> **Prototype is runnable; operational trust requires more work**  
> **In place:** Python/PyTorch models · FastAPI · MOSDAC client · HDF5 preprocessing · browser dashboard  
> **Next validation gates:**  
> 1. Pair imagery with verified storm/time/location labels  
> 2. Split image evaluation by unseen storm IDs  
> 3. Replace derived environmental inputs with time/location-matched observations  
> 4. Re-run baselines, calibration, ablations, and uncertainty checks  
> 5. Test failure states, latency, and domain-expert review

**Speaking cue:** We have an implementation foundation, but data alignment and independent validation are the critical path to responsible deployment.

**PNG composition:** A two-column board: `Working prototype` in green/blue and `Validation gates` in amber. Connect each gate to the specific model stage it improves. Avoid an operational readiness badge.

---

## Slide 12 — Impact and Roadmap

**On-slide copy**

> **From prototype to responsible decision support**  
> **Now:** demonstrate ingestion → inference → dashboard on available sample/live-accessible data  
> **Next:** repair labels and storm-level evaluation; validate forecasts with meteorological reviewers  
> **Then:** assess regional data coverage, reliability, latency, and integration requirements with authorized agencies  
> **Potential beneficiaries:** forecasters · disaster-management teams · coastal administrations  
> **Expected contribution:** faster access to consolidated evidence, not automated evacuation decisions

**Speaking cue:** Any operational pilot would require agency partnership, verified data access, independent evaluation, and human approval workflows.

**PNG composition:** A three-stage horizontal roadmap with gates between stages. At the bottom, show beneficiaries as labeled roles rather than unsupported quantified impact. End with a small callout: `Seeking domain feedback and validation partners`.

---

## PNG Production Checklist

1. Export each slide at 1920 × 1080 (16:9); keep body text readable at presentation size.
2. Use genuine project screenshots and model outputs. If a visual is illustrative, label it directly on the image.
3. Keep the visual system consistent: deep charcoal/navy background, storm-cloud whites, cyan observation marks, amber uncertainty/validation notes, and restrained red for risk.
4. Do not reuse the dashboard's mock values as measured outputs. The UI mock data is in `cyclone ai/cyclone ai/js/mockData.js`.
5. Add source footnotes for every metric, dataset count, and satellite image. Keep caveats on the same slide as the claim.
6. Before presenting, verify the SIH problem-statement wording, team details, MOSDAC access status, and whether the shown screenshots came from the live API or local mock data.
