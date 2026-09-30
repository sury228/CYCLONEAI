"""
CycloneVision AI - Satellite Service (MOSDAC Integration)
==========================================================
Official MOSDAC API client for automated, near-real-time retrieval
of INSAT-3DS / INSAT-3DR satellite data products.

Responsibilities:
- Authenticate with MOSDAC using official API endpoints
- Query the configured INSAT-3DS / INSAT-3DR dataset catalog
- Identify the newest available satellite frame
- Download the newest frame directly to data/live/latest/
- Persist structured metadata: timestamp, satellite, sensor, datasetId, filename
- Prevent duplicate downloads of the same frame
- Comply with official MOSDAC download policies
"""

from __future__ import annotations

import os
import sys
import json
import time
import logging
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from datetime import datetime

import requests

from storage_config import copy_to_storage_dir, ensure_directory, resolve_raw_storage_dir

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("SatelliteService")

# Project paths
BASE_DIR = Path(__file__).resolve().parent
LIVE_DATA_DIR = BASE_DIR / "data" / "live"
LATEST_DATA_DIR = LIVE_DATA_DIR / "latest"
PREVIEW_DIR = LIVE_DATA_DIR / "preview"
LATEST_METADATA_PATH = LATEST_DATA_DIR / "metadata.json"
HISTORY_PATH = LIVE_DATA_DIR / "download_history.json"

# Ensure directories exist
LATEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
RAW_STORAGE_DIR = resolve_raw_storage_dir()
ensure_directory(RAW_STORAGE_DIR)

# Official MOSDAC API Endpoints (as defined in official mdapi client)
MOSDAC_BASE_URL = "https://mosdac.gov.in"
MOSDAC_TOKEN_URL = f"{MOSDAC_BASE_URL}/download_api/gettoken"
MOSDAC_SEARCH_URL = f"{MOSDAC_BASE_URL}/apios/datasets.json"
MOSDAC_DOWNLOAD_URL = f"{MOSDAC_BASE_URL}/download_api/download"
MOSDAC_REFRESH_URL = f"{MOSDAC_BASE_URL}/download_api/refresh-token"
MOSDAC_CHECK_INTERNET_URL = f"{MOSDAC_BASE_URL}/download_api/check-internet"
MOSDAC_LOGOUT_URL = f"{MOSDAC_BASE_URL}/download_api/logout"

# Fallback .env loader
def load_env_variables() -> None:
    """Loads environment variables from .env file if available."""
    try:
        from dotenv import load_dotenv
        env_file = BASE_DIR / ".env"
        if env_file.exists():
            load_dotenv(dotenv_path=env_file, override=False)
            return
    except ImportError:
        pass

    # Fallback parser if python-dotenv is not installed
    env_file = BASE_DIR / ".env"
    if env_file.exists():
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception as e:
            logger.warning("Could not parse .env fallback: %s", e)

# Load environment on import
load_env_variables()


def infer_satellite_name(dataset_id: str, filename: str = "") -> str:
    """Infers the satellite platform from the dataset ID or filename."""
    token = (dataset_id + " " + filename).upper()
    if "3S" in token or "INSAT-3DS" in token:
        return "INSAT-3DS"
    if "3R" in token or "INSAT-3DR" in token:
        return "INSAT-3DR"
    if "3D" in token or "INSAT-3D" in token:
        return "INSAT-3D"
    return "INSAT Satellite"


class MosdacSatelliteService:
    """
    Official MOSDAC Satellite Ingestion Client.
    Manages authentication, catalog polling, deduplication, and secure downloading.
    """

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        dataset_id: Optional[str] = None,
        channel: Optional[str] = None,
    ):
        load_env_variables()
        self.username = (username or os.getenv("MOSDAC_USERNAME", "")).strip()
        self.password = (password or os.getenv("MOSDAC_PASSWORD", "")).strip()
        self.dataset_id = (dataset_id or os.getenv("MOSDAC_DATASET_ID", "3SIMG_L1B_STD")).strip()
        self.channel = (channel or os.getenv("MOSDAC_CHANNEL", "IMG_TIR1")).strip()

        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.token_expiry: float = 0.0

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "CycloneVisionAI-MOSDAC-Client/2.0 (ISRO-MOSDAC-API)"
        })

        self._download_lock = threading.RLock()
        self._download_in_progress = False
        self._history: Dict[str, Any] = self._load_history()

    def _load_history(self) -> Dict[str, Any]:
        """Loads download history tracking to avoid duplicate downloads."""
        if HISTORY_PATH.exists():
            try:
                with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Error reading download history: %s. Re-initializing.", e)
        return {"downloaded_files": {}, "last_record_id": None, "last_identifier": None}

    def _save_history(self) -> None:
        """Saves download history to disk."""
        try:
            with open(HISTORY_PATH, "w", encoding="utf-8") as f:
                json.dump(self._history, f, indent=2)
        except Exception as e:
            logger.error("Failed to persist download history: %s", e)

    def is_configured(self) -> bool:
        """Returns True if valid MOSDAC credentials are set."""
        return bool(self.username and self.password)

    def authenticate(self, force: bool = False) -> str:
        """
        Authenticates with MOSDAC and acquires an access token.
        Raises ValueError if credentials are missing or invalid.
        """
        if not self.is_configured():
            raise ValueError(
                "MOSDAC credentials missing. Please set MOSDAC_USERNAME and MOSDAC_PASSWORD in .env"
            )

        # Reuse existing valid token if not expired
        now = time.time()
        if not force and self.access_token and (now < self.token_expiry - 60):
            return self.access_token

        # If refresh token is available and token expired, attempt refresh
        if self.refresh_token and not force:
            refreshed = self._attempt_token_refresh()
            if refreshed:
                return refreshed

        payload = {
            "username": self.username,
            "password": self.password,
        }

        logger.info("Authenticating with MOSDAC API (%s)...", self.username)
        try:
            resp = self.session.post(MOSDAC_TOKEN_URL, json=payload, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                self.access_token = data.get("access_token")
                self.refresh_token = data.get("refresh_token")
                # Default token lifetime is typically 1 hour (3600s)
                self.token_expiry = now + 3300
                logger.info("[+] MOSDAC Authentication successful.")
                return str(self.access_token)
            elif resp.status_code == 401:
                err = resp.json().get("error", "Unauthorized: Invalid username or password.")
                logger.error("[-] MOSDAC Login Failed (401): %s", err)
                raise PermissionError(f"MOSDAC Authentication Error: {err}")
            elif resp.status_code == 503:
                msg = resp.json().get("message", "Service Unavailable - Server Maintenance")
                logger.warning("[!] MOSDAC Service Maintenance: %s", msg)
                raise ConnectionError(f"MOSDAC Service Unavailable: {msg}")
            else:
                resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error("Network error during MOSDAC authentication: %s", e)
            raise ConnectionError(f"Unable to reach MOSDAC token endpoint: {e}") from e

        raise RuntimeError("Unexpected authentication response from MOSDAC.")

    def _attempt_token_refresh(self) -> Optional[str]:
        """Refreshes access token using refresh_token endpoint."""
        if not self.refresh_token:
            return None
        logger.info("Refreshing MOSDAC access token...")
        try:
            resp = self.session.post(
                MOSDAC_REFRESH_URL,
                json={"refresh_token": self.refresh_token},
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                self.access_token = data.get("access_token")
                if "refresh_token" in data:
                    self.refresh_token = data["refresh_token"]
                self.token_expiry = time.time() + 3300
                logger.info("[+] Token refreshed successfully.")
                return self.access_token
        except Exception as e:
            logger.warning("Token refresh failed: %s. Re-authenticating.", e)
        return None

    def query_latest_frame(self, dataset_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Queries the official MOSDAC catalog API for the newest available satellite frame.
        Does NOT require credentials for public catalog discovery.
        """
        did = dataset_id or self.dataset_id
        logger.info("Querying MOSDAC catalog for newest frame in dataset '%s'...", did)

        params = {
            "datasetId": did,
            "count": "1",  # Return newest record
        }

        try:
            resp = self.session.get(MOSDAC_SEARCH_URL, params=params, timeout=25)
            if resp.status_code == 200:
                data = resp.json()
                total_results = data.get("totalResults", 0)
                entries = data.get("entries", [])
                if not entries:
                    logger.warning("No entries found for dataset '%s' in MOSDAC catalog.", did)
                    return {
                        "status": "not_found",
                        "datasetId": did,
                        "totalResults": total_results,
                    }

                latest_entry = entries[0]
                identifier = latest_entry.get("identifier", "")
                record_id = latest_entry.get("id")
                updated = latest_entry.get("updated", "")
                satellite = infer_satellite_name(did, identifier)

                result = {
                    "status": "available",
                    "datasetId": did,
                    "satellite": satellite,
                    "sensor": f"Imager ({self.channel})",
                    "filename": identifier,
                    "record_id": record_id,
                    "timestamp": updated,
                    "totalResults": total_results,
                    "query_time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
                }
                logger.info(
                    "[+] Newest catalog frame found: %s (ID: %s, Time: %s, Satellite: %s)",
                    identifier,
                    record_id,
                    updated,
                    satellite,
                )
                return result
            else:
                logger.error("MOSDAC catalog query failed with status %d: %s", resp.status_code, resp.text[:200])
                return {
                    "status": "error",
                    "code": resp.status_code,
                    "message": resp.text[:200],
                    "datasetId": did,
                }
        except Exception as e:
            logger.error("Exception querying MOSDAC catalog: %s", e)
            return {"status": "error", "error": str(e), "datasetId": did}

    def is_frame_already_downloaded(self, identifier: str, record_id: Any) -> bool:
        """
        Checks whether the specified satellite frame has already been downloaded.
        """
        target_file = LATEST_DATA_DIR / identifier
        if target_file.exists() and target_file.stat().st_size > 1024:
            return True

        hist = self._history.get("downloaded_files", {})
        if identifier in hist or str(record_id) in hist:
            # Verify file actually exists
            rec = hist.get(identifier) or hist.get(str(record_id))
            if isinstance(rec, dict) and "path" in rec and Path(rec["path"]).exists():
                return True

        return False

    def is_download_in_progress(self) -> bool:
        """Returns True while a MOSDAC file download is currently active."""
        return bool(self._download_in_progress)

    def validate_downloaded_file(self, path: Path) -> bool:
        """Validate that the final file looks like a real HDF5 payload and is not empty."""
        if path is None or not path.exists():
            return False
        try:
            size = path.stat().st_size
        except OSError:
            return False
        if size <= 8:
            return False
        try:
            with open(path, "rb") as fh:
                header = fh.read(8)
        except OSError:
            return False
        if header != b"\x89HDF\r\n\x1a\n":
            logger.warning("Downloaded file %s does not contain a valid HDF5 header.", path)
            return False
        return True

    def cleanup_old_local_raw_files(self, max_age_hours: int = 24) -> None:
        """Removes stale local raw .h5 files to keep the local machine from accumulating old frames."""
        if not LATEST_DATA_DIR.exists():
            return
        cutoff = time.time() - (max_age_hours * 3600)
        for candidate in LATEST_DATA_DIR.iterdir():
            if candidate.name in {"metadata.json", "latest_product.h5", "download_history.json"}:
                continue
            if candidate.suffix.lower() not in {".h5", ".hdf", ".nc"}:
                if candidate.name.endswith(".part"):
                    try:
                        candidate.unlink(missing_ok=True)
                    except OSError:
                        pass
                continue
            try:
                if candidate.stat().st_mtime < cutoff:
                    candidate.unlink(missing_ok=True)
            except OSError:
                continue

    def download_frame(
        self,
        record_id: Any,
        identifier: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Downloads a specific frame by record_id from MOSDAC using authenticated Bearer token.
        Saves the file to a temporary .part file and only renames it to the final HDF5 path
        after the file validates cleanly.
        """
        if not self.is_configured():
            raise ValueError(
                "MOSDAC credentials required for downloading. Please configure MOSDAC_USERNAME and MOSDAC_PASSWORD in .env"
            )

        target_path = LATEST_DATA_DIR / identifier
        temp_path = LATEST_DATA_DIR / f"{identifier}.part"

        with self._download_lock:
            self._download_in_progress = True
            try:
                token = self.authenticate()
                headers = {
                    "Authorization": f"Bearer {token}",
                    "Accept-Encoding": "identity",
                }
                params = {"id": record_id}
                chunk_size = 8 * 1024 * 1024
                resume_offset = 0

                if temp_path.exists():
                    resume_offset = temp_path.stat().st_size
                if resume_offset > 0:
                    logger.info("Resuming %s from byte offset %d.", identifier, resume_offset)
                headers["Range"] = f"bytes={resume_offset}-{resume_offset + chunk_size - 1}"

                logger.info("Starting download of %s (ID: %s) to %s...", identifier, record_id, target_path)

                last_error: Optional[Exception] = None
                for attempt in range(1, 65):
                    resp = None
                    resume_offset = temp_path.stat().st_size if temp_path.exists() else 0
                    range_end = resume_offset + chunk_size - 1
                    headers["Range"] = f"bytes={resume_offset}-{range_end}"
                    try:
                        resp = self.session.get(
                            MOSDAC_DOWNLOAD_URL,
                            headers=headers,
                            params=params,
                            stream=True,
                            timeout=(15, 180),
                        )

                        if resp.status_code == 401:
                            logger.warning("Token expired during download. Re-authenticating...")
                            token = self.authenticate(force=True)
                            headers = {
                                "Authorization": f"Bearer {token}",
                                "Accept-Encoding": "identity",
                            }
                            resume_offset = temp_path.stat().st_size if temp_path.exists() else 0
                            headers["Range"] = f"bytes={resume_offset}-{resume_offset + chunk_size - 1}"
                            resp.close()
                            continue

                        if resp.status_code == 404:
                            raise FileNotFoundError(f"File not released or not found on MOSDAC (ID: {record_id})")

                        if resp.status_code in (200, 206):
                            if resp.status_code == 200 and temp_path.exists() and temp_path.stat().st_size > 0:
                                logger.warning("Server ignored Range request for %s; restarting partial download from zero bytes.", identifier)
                                temp_path.unlink(missing_ok=True)
                                resume_offset = 0
                                headers.pop("Range", None)
                            elif resp.status_code == 206 and temp_path.exists():
                                resume_offset = temp_path.stat().st_size

                            total_size = None
                            expected_chunk_size = None
                            if resp.status_code == 206:
                                content_range = resp.headers.get("Content-Range", "")
                                try:
                                    unit, range_value = content_range.split(" ", 1)
                                    byte_range, complete_length = range_value.split("/", 1)
                                    range_start, range_last = (int(value) for value in byte_range.split("-", 1))
                                    total_size = int(complete_length)
                                except (ValueError, TypeError):
                                    raise IOError(f"MOSDAC returned an invalid Content-Range header: {content_range!r}")
                                if unit.lower() != "bytes" or range_start != resume_offset or range_last >= total_size:
                                    raise IOError(f"MOSDAC returned an unexpected Content-Range header: {content_range!r}")
                                expected_last = min(range_end, total_size - 1)
                                if range_last != expected_last:
                                    raise IOError(f"MOSDAC returned an incomplete byte range: {content_range!r}")
                                expected_chunk_size = range_last - range_start + 1
                            else:
                                content_length = resp.headers.get("Content-Length")
                                if content_length and content_length.isdigit():
                                    total_size = int(content_length)

                            file_mode = "ab" if resume_offset > 0 else "wb"

                            with open(temp_path, file_mode) as fh:
                                for chunk in resp.iter_content(chunk_size=65536):
                                    if not chunk:
                                        continue
                                    fh.write(chunk)

                            if not temp_path.exists() or temp_path.stat().st_size <= 0:
                                raise IOError(f"Downloaded file {temp_path} is empty.")

                            current_size = temp_path.stat().st_size
                            if expected_chunk_size is not None and current_size != resume_offset + expected_chunk_size:
                                raise IOError(f"MOSDAC range ended at byte {current_size}, expected {resume_offset + expected_chunk_size}.")
                            if total_size is not None and current_size < total_size:
                                logger.info("Downloaded %d of %d bytes for %s.", current_size, total_size, identifier)
                                resp.close()
                                continue
                            if total_size is not None and current_size != total_size:
                                raise IOError(f"Downloaded {current_size} bytes for {identifier}; MOSDAC advertised {total_size}.")

                            if not self.validate_downloaded_file(temp_path):
                                if temp_path.stat().st_size <= 8:
                                    raise IOError(f"Downloaded file {temp_path} is too small to be valid.")
                                logger.warning("Temporary file %s is incomplete; keeping it for resume.", temp_path)
                                raise IOError(
                                    f"Downloaded file {temp_path.name} is incomplete or invalid; keeping partial download for resume."
                                )

                            if target_path.exists():
                                target_path.unlink(missing_ok=True)
                            os.replace(temp_path, target_path)
                            logger.info("[+] Download complete: %s (%.2f MB)", target_path.name, target_path.stat().st_size / (1024 * 1024))
                            break

                        resp.raise_for_status()
                    except (
                        requests.exceptions.ChunkedEncodingError,
                        requests.exceptions.ConnectionError,
                        requests.exceptions.Timeout,
                        requests.exceptions.RequestException,
                        OSError,
                        IOError,
                        ValueError,
                    ) as exc:
                        last_error = exc
                        logger.warning("Download request %d failed for %s: %s", attempt, identifier, exc)
                        if resp is not None:
                            resp.close()
                        if attempt >= 64:
                            logger.warning("Keeping partial file %s for resume after final failure.", temp_path)
                            raise
                        delay = min(2 ** (attempt - 1), 16)
                        logger.info("Retrying %s in %d seconds (request %d/64)...", identifier, delay, attempt + 1)
                        time.sleep(delay)
                        continue
                else:
                    if last_error is not None:
                        raise last_error
                    raise RuntimeError(f"Download of {identifier} did not complete successfully.")

                # Save standard reference alias: data/live/latest/latest_product.h5
                standard_ref = LATEST_DATA_DIR / "latest_product.h5"
                try:
                    if standard_ref.exists():
                        standard_ref.unlink(missing_ok=True)
                    import shutil
                    shutil.copyfile(target_path, standard_ref)
                except Exception as e:
                    logger.warning("Could not create latest_product.h5 alias: %s", e)

                # Persist a copy in the configured raw-data storage path and keep local temp cleanup lightweight.
                try:
                    copy_to_storage_dir(target_path, RAW_STORAGE_DIR, identifier)
                except Exception as exc:
                    logger.warning("Could not copy downloaded HDF5 to configured raw storage: %s", exc)

                self.cleanup_old_local_raw_files(max_age_hours=24)

                # Update metadata
                saved_metadata = {
                    "timestamp": (metadata or {}).get("timestamp", datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")),
                    "satellite": (metadata or {}).get("satellite", infer_satellite_name(self.dataset_id, identifier)),
                    "sensor": (metadata or {}).get("sensor", f"Imager ({self.channel})"),
                    "datasetId": self.dataset_id,
                    "filename": identifier,
                    "record_id": record_id,
                    "file_path": str(target_path),
                    "file_size_bytes": target_path.stat().st_size,
                    "downloaded_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
                    "status": "ready",
                    "raw_storage_path": str(RAW_STORAGE_DIR / identifier),
                }

                with open(LATEST_METADATA_PATH, "w", encoding="utf-8") as f:
                    json.dump(saved_metadata, f, indent=2)

                # Update download history
                self._history["downloaded_files"][identifier] = {
                    "record_id": record_id,
                    "path": str(target_path),
                    "timestamp": saved_metadata["timestamp"],
                    "downloaded_at": saved_metadata["downloaded_at"],
                }
                self._history["last_record_id"] = record_id
                self._history["last_identifier"] = identifier
                self._save_history()

                return target_path
            finally:
                self._download_in_progress = False

    def check_and_download_newest_frame(self) -> Tuple[bool, Dict[str, Any]]:
        """
        High-level pipeline method:
        1. Checks MOSDAC catalog for newest frame.
        2. Compares against download history and existing files.
        3. If a NEW frame is detected, downloads it.
        4. If already present, skips downloading.

        Returns:
            (is_new_frame: bool, metadata: Dict[str, Any])
        """
        catalog_info = self.query_latest_frame()

        if catalog_info.get("status") != "available":
            # If MOSDAC query failed or dataset not found, check existing local metadata
            existing = self.get_latest_metadata()
            return False, existing or catalog_info

        identifier = catalog_info.get("filename")
        record_id = catalog_info.get("record_id")

        if not identifier or record_id is None:
            return False, catalog_info

        # Check if already downloaded
        if self.is_frame_already_downloaded(identifier, record_id):
            logger.info("Frame '%s' (ID: %s) is already downloaded. Skipping download.", identifier, record_id)
            existing = self.get_latest_metadata()
            if existing:
                existing["status"] = "ready"
                existing["is_new_frame"] = False
                return False, existing
            return False, catalog_info

        if self.is_download_in_progress():
            logger.warning("Download already in progress for frame '%s' (ID: %s).", identifier, record_id)
            catalog_info["status"] = "in_progress"
            catalog_info["message"] = "Download already in progress"
            catalog_info["is_new_frame"] = True
            return False, catalog_info

        # New frame detected!
        logger.info("[*] New satellite frame detected on MOSDAC: %s", identifier)
        if not self.is_configured():
            logger.warning(
                "[!] New frame available on MOSDAC (%s), but MOSDAC_USERNAME/PASSWORD are not set in .env. Cannot download.",
                identifier,
            )
            catalog_info["status"] = "credentials_required"
            catalog_info["is_new_frame"] = True
            return False, catalog_info

        # Download the new frame
        downloaded_path = self.download_frame(record_id, identifier, metadata=catalog_info)
        latest_meta = self.get_latest_metadata() or catalog_info
        latest_meta["status"] = "ready"
        latest_meta["is_new_frame"] = True
        return True, latest_meta

    def get_latest_metadata(self) -> Optional[Dict[str, Any]]:
        """Reads persisted metadata from data/live/latest/metadata.json."""
        if LATEST_METADATA_PATH.exists():
            try:
                with open(LATEST_METADATA_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Could not read latest metadata: %s", e)
        return None


# Default module-level service instance
default_satellite_service = MosdacSatelliteService()


if __name__ == "__main__":
    print("=" * 70)
    print("   CycloneVision AI - MOSDAC Satellite Service Test")
    print("=" * 70)
    svc = MosdacSatelliteService()
    print(f"[*] Configured Dataset ID : {svc.dataset_id}")
    print(f"[*] Configured Channel    : {svc.channel}")
    print(f"[*] Credentials Configured: {'YES' if svc.is_configured() else 'NO (Set in .env)'}")
    print("-" * 70)

    print("[*] Polling MOSDAC catalog for newest frame...")
    info = svc.query_latest_frame()
    print(json.dumps(info, indent=2))

    if svc.is_configured():
        print("-" * 70)
        print("[*] Checking / downloading newest frame...")
        is_new, meta = svc.check_and_download_newest_frame()
        print(f"[*] Is New Frame: {is_new}")
        print(f"[*] Result Metadata: {json.dumps(meta, indent=2)}")
    else:
        print("\n[INFO] Set MOSDAC_USERNAME and MOSDAC_PASSWORD in .env to enable downloading.")
    print("=" * 70)
