import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from satellite_service import MosdacSatelliteService


class SatelliteServiceDownloadTests(unittest.TestCase):
    def test_validate_downloaded_file_accepts_valid_hdf5_header(self):
        service = MosdacSatelliteService(username="user", password="pass")
        with tempfile.TemporaryDirectory() as tmpdir:
            candidate = Path(tmpdir) / "sample.h5"
            candidate.write_bytes(b"\x89HDF\r\n\x1a\n" + b"x" * 16)
            self.assertTrue(service.validate_downloaded_file(candidate))

    def test_validate_downloaded_file_rejects_empty_file(self):
        service = MosdacSatelliteService(username="user", password="pass")
        with tempfile.TemporaryDirectory() as tmpdir:
            candidate = Path(tmpdir) / "sample.h5"
            candidate.write_bytes(b"")
            self.assertFalse(service.validate_downloaded_file(candidate))

    def test_download_lock_reports_in_progress(self):
        service = MosdacSatelliteService(username="user", password="pass")
        service._download_in_progress = True
        self.assertTrue(service.is_download_in_progress())

    def test_download_frame_uses_bounded_range_and_finalizes_valid_payload(self):
        payload = b"\x89HDF\r\n\x1a\n" + b"x" * 16
        response = Mock()
        response.status_code = 206
        response.headers = {"Content-Range": f"bytes 0-{len(payload) - 1}/{len(payload)}"}
        response.iter_content.return_value = [payload]

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            service = MosdacSatelliteService(username="user", password="pass")
            service.authenticate = Mock(return_value="test-token")
            service.session.get = Mock(return_value=response)

            with patch("satellite_service.LATEST_DATA_DIR", data_dir), patch(
                "satellite_service.LATEST_METADATA_PATH", data_dir / "metadata.json"
            ), patch("satellite_service.HISTORY_PATH", data_dir / "history.json"):
                downloaded = service.download_frame(123, "sample.h5")

            self.assertEqual(downloaded.read_bytes(), payload)
            request_headers = service.session.get.call_args.kwargs["headers"]
            self.assertEqual(request_headers["Range"], "bytes=0-8388607")
            self.assertTrue((data_dir / "latest_product.h5").exists())

    def test_already_downloaded_frame_remains_ready(self):
        service = MosdacSatelliteService(username="user", password="pass")
        service.query_latest_frame = Mock(return_value={
            "status": "available",
            "filename": "sample.h5",
            "record_id": 123,
        })
        service.is_frame_already_downloaded = Mock(return_value=True)
        service.get_latest_metadata = Mock(return_value={"status": "ready", "filename": "sample.h5"})

        is_new, metadata = service.check_and_download_newest_frame()

        self.assertFalse(is_new)
        self.assertEqual(metadata["status"], "ready")


class PipelineStateTests(unittest.TestCase):
    def test_duplicate_pipeline_cycle_does_not_start_another_cycle(self):
        import api

        with patch.object(api, "_execute_pipeline_cycle") as run_cycle:
            with api.pipeline_lock:
                result = api.execute_pipeline_cycle()

        self.assertEqual(result["status"], "in_progress")
        run_cycle.assert_not_called()

    def test_initial_cyclone_endpoint_does_not_report_fake_prediction(self):
        import asyncio
        import api

        result = asyncio.run(api.get_latest_cyclone())

        self.assertEqual(result["status"], "unavailable")
        self.assertIsNone(result["confidence"])
        self.assertIsNone(result["cyclone_detected"])
        self.assertEqual(result["forecast"], [])

    def test_latest_cyclone_endpoint_preserves_prediction_details(self):
        import asyncio
        import api

        prediction = {
            "cyclone_detected": True,
            "confidence": 1.0,
            "eye_latitude": 17.122,
            "eye_longitude": 87.771,
            "forecast_track": [{
                "hour": 6,
                "latitude": 17.377,
                "longitude": 88.093,
                "cone_radius_km": 25.7,
            }],
            "uncertainty": {"radii_km": [[25.69, 25.67]], "landfall_probability": 0.89},
            "detection": {
                "confidence": 100.0,
                "eye_bbox": [0.4, 0.3, 0.1, 0.1],
                "eye_diameter_km": 57.9,
            },
            "eye": {"bbox": [0.4, 0.3, 0.1, 0.1]},
            "movement": {"direction": "NE", "speed": 36.5, "bearing": 51.6},
            "errorHistory": [{"forecast": "+6h", "error": 12.3}],
            "processingTime": "1.2s",
            "status": "ready",
        }
        satellite = {"satellite": "INSAT-3DS", "status": "ready", "timestamp": "2026-09-30T00:00:00Z"}

        with patch.dict(api.state["cyclone"], prediction), patch.dict(api.state["satellite"], satellite):
            result = asyncio.run(api.get_latest_cyclone())

        self.assertEqual(result["detection"]["eye_bbox"], prediction["detection"]["eye_bbox"])
        self.assertEqual(result["detection"]["eye_diameter_km"], 57.9)
        self.assertEqual(result["eye"]["bbox"], prediction["detection"]["eye_bbox"])
        self.assertEqual(result["forecast_track"][0]["cone_radius_km"], 25.7)
        self.assertEqual(result["forecast"][0]["cone_radius_km"], 25.7)
        self.assertEqual(result["uncertainty"], prediction["uncertainty"])
        self.assertEqual(result["errorHistory"], prediction["errorHistory"])
        self.assertEqual(result["processingTime"], "1.2s")

    def test_storage_paths_are_resolved_from_environment(self):
        from storage_config import resolve_raw_storage_dir, resolve_predictions_dir, resolve_latest_image_path

        with patch.dict("os.environ", {
            "RAW_STORAGE_PATH": "/tmp/raw-store",
            "PREDICTIONS_STORAGE_PATH": "/tmp/pred-store",
            "LATEST_IMAGE_PATH": "/tmp/realtime/latest_satellite.png",
        }, clear=False):
            self.assertEqual(os.path.normpath(str(resolve_raw_storage_dir())), os.path.normpath(str(Path("/tmp/raw-store"))))
            self.assertEqual(os.path.normpath(str(resolve_predictions_dir())), os.path.normpath(str(Path("/tmp/pred-store"))))
            self.assertEqual(os.path.normpath(str(resolve_latest_image_path())), os.path.normpath(str(Path("/tmp/realtime/latest_satellite.png"))))

    def test_remote_google_drive_urls_fall_back_to_local_default_paths(self):
        from storage_config import resolve_raw_storage_dir, resolve_predictions_dir, resolve_latest_image_path

        with patch.dict("os.environ", {
            "RAW_STORAGE_PATH": "https://drive.google.com/drive/folders/abc",
            "PREDICTIONS_STORAGE_PATH": "https://drive.google.com/drive/folders/def",
            "LATEST_IMAGE_PATH": "https://drive.google.com/drive/folders/ghi",
        }, clear=False):
            self.assertEqual(str(resolve_raw_storage_dir()), str(Path(__file__).resolve().parents[1] / "data" / "live" / "latest"))
            self.assertEqual(str(resolve_predictions_dir()), str(Path(__file__).resolve().parents[1] / "live_outputs"))
            self.assertEqual(str(resolve_latest_image_path()), str(Path(__file__).resolve().parents[1] / "data" / "live" / "preview" / "latest.png"))

    def test_generate_composite_overlay_handles_large_preview_image(self):
        import numpy as np
        from inference_service import default_inference_service

        with tempfile.TemporaryDirectory() as tmpdir:
            preview_path = Path(tmpdir) / "large_preview.png"
            large_img = np.zeros((1009, 1009, 4), dtype=np.uint8)
            large_img[:, :, 3] = 255
            Image.fromarray(large_img, mode="RGBA").save(preview_path)

            out_path = Path(tmpdir) / "overlay.png"
            saved = default_inference_service.generate_composite_overlay(
                base_preview_path=preview_path,
                model1_res={"cyclone_detected": False},
                save_path=out_path,
            )

            self.assertTrue(saved.exists())
            self.assertGreater(saved.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
