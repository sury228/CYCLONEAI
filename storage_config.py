from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_STORAGE_DIR = BASE_DIR / "data" / "live" / "latest"
DEFAULT_PREDICTIONS_DIR = BASE_DIR / "live_outputs"
DEFAULT_LATEST_IMAGE_PATH = BASE_DIR / "data" / "live" / "preview" / "latest.png"


def _resolve_path(env_name: str, default: Path) -> Path:
    value = os.getenv(env_name, "").strip()
    if not value:
        return default
    lowered = value.lower()
    if lowered.startswith(("http://", "https://", "gs://", "s3://")):
        print(f"[storage_config] Ignoring invalid remote URL for {env_name}: {value}. Use a local filesystem path, not a Google Drive share URL.")
        return default
    candidate = Path(value).expanduser()
    is_abs = candidate.is_absolute() or value.startswith("/") or value.startswith("\\")
    if not is_abs:
        candidate = (BASE_DIR / candidate).resolve()
    return candidate


def resolve_raw_storage_dir() -> Path:
    return _resolve_path("RAW_STORAGE_PATH", DEFAULT_RAW_STORAGE_DIR)


def resolve_predictions_dir() -> Path:
    return _resolve_path("PREDICTIONS_STORAGE_PATH", DEFAULT_PREDICTIONS_DIR)


def resolve_latest_image_path() -> Path:
    return _resolve_path("LATEST_IMAGE_PATH", DEFAULT_LATEST_IMAGE_PATH)


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def copy_to_storage_dir(source_path: Path, target_dir: Optional[Path] = None, filename: Optional[str] = None) -> Path:
    source = Path(source_path)
    target_dir = Path(target_dir) if target_dir is not None else resolve_raw_storage_dir()
    ensure_directory(target_dir)
    dest_name = filename or source.name
    destination = target_dir / dest_name
    if source.exists():
        shutil.copy2(str(source), str(destination))
    return destination


def replace_latest_image(source_path: Path, target_path: Optional[Path] = None) -> Path:
    source = Path(source_path)
    destination = Path(target_path) if target_path is not None else resolve_latest_image_path()
    ensure_directory(destination.parent)
    if destination.exists():
        destination.unlink()
    shutil.copy2(str(source), str(destination))
    return destination


def parse_prediction_timestamp(value: Optional[str]) -> datetime:
    if value is None or value == "":
        return datetime.utcnow()
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S UTC", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    return datetime.utcnow()


def persist_prediction_payload(payload: Dict[str, Any]) -> Path:
    timestamp_value = payload.get("timestamp")
    ts = parse_prediction_timestamp(str(timestamp_value) if timestamp_value is not None else None)
    day_dir = ts.strftime("%Y-%m-%d")
    time_name = ts.strftime("%H%M")
    target_dir = ensure_directory(resolve_predictions_dir() / day_dir)
    target_path = target_dir / f"{time_name}.json"
    with open(target_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return target_path
