"""
CycloneVision AI - Central FastAPI Backend Server & Scheduler
==============================================================
Provides real near-real-time satellite ingestion, background scheduling,
and deep learning inference endpoints.

Architecture:
  MOSDAC (Official API)
       ↓
  Satellite Service (satellite_service.py)
       ↓
  Image Processor (image_processor.py)
       ↓
  Inference Service (inference_service.py)
       ↓
  FastAPI Backend (api.py)
       ↓
  Dashboard (cyclone ai / cyclone ai / index.html)

Endpoints:
  GET  /api/satellite/latest   -> Latest satellite metadata & image URL
  GET  /api/cyclone/latest     -> Model 1 detection & Model 2 trajectory forecast
  GET  /api/satellite/image    -> Latest normalized satellite preview image
  GET  /api/cyclone/overlay    -> Composite overlay (satellite + eye + track + cone)
  POST /api/satellite/refresh  -> Trigger immediate MOSDAC poll check
  GET  /api/health             -> System and background scheduler status
"""

from __future__ import annotations

import os
import sys
import time
import json
import asyncio
import logging
import threading
from pathlib import Path
from typing import Any, Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Project paths
BASE_DIR = Path(__file__).resolve().parent
LIVE_DATA_DIR = BASE_DIR / "data" / "live"
LATEST_DATA_DIR = LIVE_DATA_DIR / "latest"
PREVIEW_DIR = LIVE_DATA_DIR / "preview"
DASHBOARD_DIR = BASE_DIR / "cyclone ai" / "cyclone ai"
PREVIEW_IMAGE_PATH = PREVIEW_DIR / "latest.png"
OVERLAY_IMAGE_PATH = PREVIEW_DIR / "latest_overlay.png"

# Ensure directories exist
LATEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

# Services
from satellite_service import MosdacSatelliteService, default_satellite_service
from image_processor import InsatImageProcessor, default_image_processor
from inference_service import CycloneInferenceService, default_inference_service

logger = logging.getLogger("CycloneAPI")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
pipeline_lock = threading.Lock()

# Global in-memory cache for ultra-fast API responses
state: Dict[str, Any] = {
    "satellite": {
        "image_url": "/api/satellite/image",
        "preview_url": None,
        "timestamp": None,
        "satellite": "INSAT-3DS",
        "sensor": "Imager (IMG_TIR1)",
        "datasetId": os.getenv("MOSDAC_DATASET_ID", "3SIMG_L1B_STD"),
        "filename": None,
        "status": "unavailable",
        "last_checked": None,
        "is_new_frame": False,
        "last_error": None,
    },
    "cyclone": {
        "status": "unavailable",
        "cyclone_detected": None,
        "confidence": None,
        "eye_latitude": None,
        "eye_longitude": None,
        "forecast_track": [],
        "uncertainty": {},
        "overlay_url": "/api/cyclone/overlay",
        "image_url": "/api/satellite/image",
        "movement": {"direction": "--", "speed": None, "bearing": None},
        "processingTime": None,
        "errorHistory": [],
    },
    "scheduler": {
        "running": False,
        "poll_interval_seconds": int(os.getenv("POLL_INTERVAL_SECONDS", "900")),
        "last_poll_time": None,
        "total_checks": 0,
        "total_new_frames": 0,
        "pipeline_in_progress": False,
    },
}


def sync_initial_cache() -> None:
    """Loads and processes the newest locally downloaded MOSDAC frame, if available."""
    try:
        meta = default_satellite_service.get_latest_metadata()
        if not meta:
            logger.info("No downloaded MOSDAC frame is available yet; dashboard data remains unavailable.")
            return

        h5_file = Path(meta.get("file_path", ""))
        if not h5_file.is_file():
            logger.warning("Stored MOSDAC metadata points to a missing file: %s", h5_file)
            state["satellite"]["status"] = "stale"
            return

        logger.info("Initializing from downloaded MOSDAC frame %s", h5_file.name)
        proc_res = default_image_processor.process_latest_frame(h5_file, metadata=meta)
        inference = default_inference_service.run_full_inference(
            proc_res["model1_tensor"],
            base_preview_path=proc_res["preview_path"],
        )
        state["satellite"].update({
            "timestamp": meta.get("timestamp"),
            "satellite": meta.get("satellite", "INSAT-3DS"),
            "sensor": meta.get("sensor", "Imager (IMG_TIR1)"),
            "datasetId": meta.get("datasetId", state["satellite"]["datasetId"]),
            "filename": meta.get("filename", h5_file.name),
            "status": "ready",
            "preview_url": "/api/satellite/image",
            "last_error": None,
        })
        state["cyclone"].update(inference)
        state["cyclone"]["status"] = "ready"
    except Exception as e:
        state["satellite"]["status"] = "error"
        state["satellite"]["last_error"] = str(e)
        logger.exception("Could not initialize dashboard from the latest MOSDAC frame")


def _execute_pipeline_cycle(force_inference: bool = False) -> Dict[str, Any]:
    """
    Executes one cycle of the satellite ingestion & AI forecasting pipeline:
    1. Query MOSDAC for newer satellite frame.
    2. If no new frame: skips inference (Requirement 10).
    3. If new frame: downloads, processes, and executes Model 1 & Model 2.
    """
    now_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    state["scheduler"]["last_poll_time"] = now_str
    state["scheduler"]["total_checks"] += 1

    logger.info("[PIPELINE] Polling MOSDAC catalog for new satellite frame...")
    is_new, meta = default_satellite_service.check_and_download_newest_frame()

    state["satellite"]["last_checked"] = now_str
    state["satellite"]["is_new_frame"] = is_new

    if meta:
        state["satellite"].update({
            "timestamp": meta.get("timestamp", state["satellite"]["timestamp"]),
            "satellite": meta.get("satellite", state["satellite"]["satellite"]),
            "sensor": meta.get("sensor", state["satellite"]["sensor"]),
            "datasetId": meta.get("datasetId", state["satellite"]["datasetId"]),
            "filename": meta.get("filename", state["satellite"]["filename"]),
            "status": meta.get("status", state["satellite"]["status"]),
        })

    if meta.get("status") in {"error", "credentials_required", "not_found"}:
        state["satellite"]["last_error"] = meta.get("message") or meta.get("error") or meta.get("status")
        if state["satellite"]["status"] != "ready":
            state["satellite"]["status"] = meta.get("status", "error")
        return {
            "status": "error",
            "message": state["satellite"]["last_error"],
            "satellite": state["satellite"],
            "cyclone": state["cyclone"],
        }

    if default_satellite_service.is_download_in_progress():
        logger.warning("[PIPELINE] Download already in progress. Skipping duplicate fetch.")
        return {
            "status": "in_progress",
            "reason": "download_already_in_progress",
            "message": "Download already in progress",
            "satellite": state["satellite"],
            "cyclone": state["cyclone"],
        }

    # Requirement 10: "Do NOT run model inference if there is no new frame."
    if not is_new and not force_inference:
        logger.info("[PIPELINE] No new frame detected on MOSDAC. Skipping model inference.")
        return {
            "status": "skipped",
            "reason": "no_new_frame",
            "message": "No newer satellite frame is available.",
            "satellite": state["satellite"],
            "cyclone": state["cyclone"],
        }

    if not is_new and force_inference:
        meta = default_satellite_service.get_latest_metadata() or {}
        if not meta.get("file_path") or not Path(meta["file_path"]).is_file():
            return {
                "status": "error",
                "message": "There is no downloaded satellite frame to run inference on.",
                "satellite": state["satellite"],
                "cyclone": state["cyclone"],
            }

    # Run image processing and model inference on newly downloaded frame
    logger.info("[PIPELINE] New frame available! Running image processing and inference...")
    state["scheduler"]["total_new_frames"] += 1

    proc_res = default_image_processor.process_latest_frame(
        h5_path=meta.get("file_path"),
        metadata=meta,
    )

    inf_res = default_inference_service.run_full_inference(
        multi_tensor=proc_res["model1_tensor"],
        base_preview_path=proc_res["preview_path"],
        reference_lat=16.42,
        reference_lon=88.21,
    )

    state["cyclone"].update(inf_res)
    state["cyclone"]["status"] = "ready"
    state["satellite"]["image_url"] = "/api/satellite/image"
    state["satellite"]["preview_url"] = "/api/satellite/image"
    state["satellite"]["status"] = "ready"
    state["satellite"]["last_error"] = None

    logger.info("[PIPELINE] Inference complete: Cyclone Detected=%s, Lat=%s, Lon=%s",
                inf_res.get("cyclone_detected"), inf_res.get("eye_latitude"), inf_res.get("eye_longitude"))

    return {
        "status": "updated",
        "satellite": state["satellite"],
        "cyclone": state["cyclone"],
    }


def execute_pipeline_cycle(force_inference: bool = False) -> Dict[str, Any]:
    """Runs at most one complete MOSDAC-to-inference cycle at a time."""
    if not pipeline_lock.acquire(blocking=False):
        return {
            "status": "in_progress",
            "reason": "pipeline_already_in_progress",
            "message": "A MOSDAC check or inference cycle is already running.",
            "satellite": state["satellite"],
            "cyclone": state["cyclone"],
        }

    state["scheduler"]["pipeline_in_progress"] = True
    try:
        return _execute_pipeline_cycle(force_inference)
    except Exception as exc:
        state["satellite"]["last_error"] = str(exc)
        if state["satellite"]["status"] != "ready":
            state["satellite"]["status"] = "error"
        logger.exception("[PIPELINE] MOSDAC cycle failed")
        return {
            "status": "error",
            "message": str(exc),
            "satellite": state["satellite"],
            "cyclone": state["cyclone"],
        }
    finally:
        state["scheduler"]["pipeline_in_progress"] = False
        pipeline_lock.release()


async def background_scheduler_task() -> None:
    """
    Background worker that runs periodically to check MOSDAC for new frames.
    """
    interval = state["scheduler"]["poll_interval_seconds"]
    logger.info("[SCHEDULER] Background scheduler started (Interval: %d seconds).", interval)
    state["scheduler"]["running"] = True

    # Initial check on startup
    await asyncio.sleep(2)
    try:
        await asyncio.to_thread(execute_pipeline_cycle, False)
    except Exception as e:
        logger.error("[SCHEDULER] Error during initial pipeline cycle: %s", e)

    while state["scheduler"]["running"]:
        try:
            await asyncio.sleep(interval)
            logger.info("[SCHEDULER] Periodic poll trigger.")
            await asyncio.to_thread(execute_pipeline_cycle, False)
        except asyncio.CancelledError:
            logger.info("[SCHEDULER] Scheduler cancelled.")
            break
        except Exception as e:
            logger.error("[SCHEDULER] Exception during scheduled polling: %s", e)
            await asyncio.sleep(30)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI app: starts and stops background scheduler."""
    sync_initial_cache()
    task = asyncio.create_task(background_scheduler_task())
    yield
    state["scheduler"]["running"] = False
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    logger.info("[SCHEDULER] Background scheduler terminated.")


# Initialize FastAPI app
app = FastAPI(
    title="CycloneVision AI - MOSDAC Live Pipeline",
    description="Official MOSDAC INSAT-3DS/3DR Near-Real-Time Satellite Imagery & PyTorch Forecasting API",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------------------------

@app.get("/api/satellite/latest")
async def get_latest_satellite() -> Dict[str, Any]:
    """
    Requirement 9: GET /api/satellite/latest
    Returns:
    {
        "image_url": "...",
        "timestamp": "...",
        "satellite": "...",
        "sensor": "...",
        "status": "live"
    }
    """
    sat = state["satellite"]
    return {
        "image_url": sat.get("image_url", "/api/satellite/image"),
        "preview_url": sat.get("preview_url"),
        "overlay_url": "/api/cyclone/overlay" if state["cyclone"].get("status") == "ready" else None,
        "timestamp": sat.get("timestamp"),
        "satellite": sat.get("satellite", "INSAT-3DS"),
        "sensor": sat.get("sensor", "Imager (IMG_TIR1)"),
        "status": sat.get("status", "live"),
        # Additional metadata
        "datasetId": sat.get("datasetId", os.getenv("MOSDAC_DATASET_ID", "3SIMG_L1B_STD")),
        "filename": sat.get("filename", ""),
        "last_checked": sat.get("last_checked"),
        "is_new_frame": sat.get("is_new_frame", False),
        "last_error": sat.get("last_error"),
    }


@app.get("/api/cyclone/latest")
async def get_latest_cyclone() -> Dict[str, Any]:
    """
    Requirement 9: GET /api/cyclone/latest
    Returns:
    {
        "cyclone_detected": true,
        "confidence": 0.94,
        "eye_latitude": 15.62,
        "eye_longitude": 83.41,
        "forecast_track": [...]
    }
    """
    cyc = state["cyclone"]
    detected = cyc.get("cyclone_detected")
    confidence = cyc.get("confidence")
    eye_lat = cyc.get("eye_latitude")
    eye_lon = cyc.get("eye_longitude")
    forecast_track = cyc.get("forecast_track", [])
    detection_data = cyc.get("detection") or {}
    eye_data = cyc.get("eye") or {}
    eye_bbox = detection_data.get("eye_bbox", eye_data.get("bbox", cyc.get("eye_bbox", [])))
    eye_diameter = detection_data.get("eye_diameter_km", cyc.get("eye_diameter_km"))
    detection_confidence = detection_data.get("confidence")
    if detection_confidence is None and confidence is not None:
        detection_confidence = float(confidence) * 100.0 if float(confidence) <= 1.0 else confidence

    # Format forecast track items
    formatted_track = []
    for item in forecast_track:
        formatted_track.append({
            "hour": item.get("hour"),
            "latitude": item.get("latitude"),
            "longitude": item.get("longitude"),
            "cone_radius_km": item.get("cone_radius_km"),
        })

    # Combined payload satisfying Requirement 9 AND existing dashboard
    return {
        # Strict Requirement 9 keys:
        "cyclone_detected": detected,
        "status": cyc.get("status", "unavailable"),
        "confidence": round(float(confidence), 3) if confidence is not None else None,
        "eye_latitude": eye_lat,
        "eye_longitude": eye_lon,
        "eye_bbox": eye_bbox,
        "eye_diameter_km": eye_diameter,
        "forecast_track": formatted_track,
        # Extended fields for dashboard visualizations:
        "uncertainty": cyc.get("uncertainty", {}),
        "overlay_url": "/api/cyclone/overlay",
        "image_url": "/api/satellite/image",
        "detection": {
            "detected": detected,
            "confidence": detection_confidence,
            "latitude": eye_lat,
            "longitude": eye_lon,
            "eyeDetected": detected,
            "eye_bbox": eye_bbox,
            "eye_diameter_km": eye_diameter,
        },
        "eye": {
            "imageUrl": "/api/cyclone/overlay",
            "detected": detected,
            "confidence": detection_confidence,
            "bbox": eye_bbox,
            "diameter_km": eye_diameter,
        },
        "movement": cyc.get("movement", {"direction": "ENE", "speed": 14.2, "bearing": 68.5}),
        "satellite": {
            "source": f"{state['satellite'].get('satellite', 'INSAT')} MOSDAC Feed",
            "imageUrl": "/api/satellite/image" if state["satellite"].get("status") == "ready" else "",
            "overlayUrl": "/api/cyclone/overlay" if cyc.get("status") == "ready" else "",
            "timestamp": state["satellite"].get("timestamp"),
        },
        "forecast": [
            {
                "hour": f["hour"],
                "latitude": f["latitude"],
                "longitude": f["longitude"],
                "cone_radius_km": f.get("cone_radius_km"),
            }
            for f in formatted_track
        ],
        "errorHistory": cyc.get("errorHistory", []),
        "processingTime": cyc.get("processingTime"),
    }


@app.get("/api/satellite/image")
async def get_satellite_image():
    """Returns the latest normalized satellite preview PNG image."""
    if state["satellite"].get("status") != "ready":
        raise HTTPException(status_code=404, detail="No processed MOSDAC satellite frame is available yet.")
    if PREVIEW_IMAGE_PATH.exists():
        return FileResponse(PREVIEW_IMAGE_PATH, media_type="image/png")

    # Generate if not present
    proc_res = default_image_processor.process_latest_frame()
    return FileResponse(proc_res["preview_path"], media_type="image/png")


@app.get("/api/cyclone/overlay")
async def get_cyclone_overlay():
    """
    Requirement 12: Composite overlay layer
    (satellite image + cyclone eye + predicted track + uncertainty cone).
    """
    if state["cyclone"].get("status") != "ready":
        raise HTTPException(status_code=404, detail="No model inference overlay is available yet.")
    if OVERLAY_IMAGE_PATH.exists():
        return FileResponse(OVERLAY_IMAGE_PATH, media_type="image/png")

    # If overlay not yet rendered, generate on the fly
    if not PREVIEW_IMAGE_PATH.exists():
        default_image_processor.process_latest_frame()

    default_inference_service.generate_composite_overlay(
        base_preview_path=PREVIEW_IMAGE_PATH,
        model1_res={
            "cyclone_detected": state["cyclone"]["cyclone_detected"],
            "eye_bbox": [0.5, 0.5, 0.2, 0.2],
            "eye_latitude": state["cyclone"]["eye_latitude"],
            "eye_longitude": state["cyclone"]["eye_longitude"],
            "confidence": state["cyclone"]["confidence"],
        },
        model2_res={
            "forecast_track": state["cyclone"]["forecast_track"],
        },
    )
    return FileResponse(OVERLAY_IMAGE_PATH, media_type="image/png")


@app.get("/api/cyclone/latest-image")
async def get_latest_image_alias():
    """Dashboard backwards-compatibility image endpoint."""
    return await get_cyclone_overlay()


@app.post("/api/satellite/refresh")
@app.post("/api/cyclone/refresh")
async def manual_refresh(force: bool = False) -> Dict[str, Any]:
    """Manually triggers a check for new MOSDAC satellite frames and updates predictions."""
    result = await asyncio.to_thread(execute_pipeline_cycle, force)
    return result


@app.get("/api/health")
async def health_check() -> Dict[str, Any]:
    """Health and status inspection for the live pipeline."""
    return {
        "status": "healthy",
        "service": "CycloneVision AI - MOSDAC Live Pipeline",
        "server_time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "mosdac_configured": default_satellite_service.is_configured(),
        "datasetId": default_satellite_service.dataset_id,
        "channel": default_satellite_service.channel,
        "scheduler": state["scheduler"],
        "latest_satellite": {
            "filename": state["satellite"]["filename"],
            "timestamp": state["satellite"]["timestamp"],
            "satellite": state["satellite"]["satellite"],
            "last_checked": state["satellite"]["last_checked"],
        },
        "latest_cyclone": {
            "detected": state["cyclone"]["cyclone_detected"],
            "confidence": state["cyclone"]["confidence"],
            "location": f"{state['cyclone']['eye_latitude']}°N, {state['cyclone']['eye_longitude']}°E",
        },
    }


# Serve static preview files directly
app.mount("/data/live/preview", StaticFiles(directory=str(PREVIEW_DIR)), name="preview")

# Serve dashboard frontend
if DASHBOARD_DIR.exists():
    app.mount("/", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="dashboard")


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("BACKEND_HOST", "0.0.0.0")
    port = int(os.getenv("BACKEND_PORT", "8000"))
    print("=" * 75)
    print(f"[*] Starting CycloneVision AI Backend on http://{host}:{port}")
    print(f"[*] Live Dashboard available at: http://localhost:{port}/")
    print(f"[*] API Docs available at:      http://localhost:{port}/docs")
    print("=" * 75)
    uvicorn.run("api:app", host=host, port=port, reload=True)
