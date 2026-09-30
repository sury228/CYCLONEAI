import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from data_fusion.data_fusion_pipeline import load_image_annotations
from data_preprocess.normalization import normalize_satellite_channel
from data_preprocess.process_cyclone_images import CycloneSatelliteDataset
from image_processor import InsatImageProcessor
from models.vortex_identification_model_1.vortex_identification_model import run_model1_training


class ImageAnnotationManifestTests(unittest.TestCase):
    def test_training_refuses_to_start_without_verified_annotations(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing_manifest = Path(temporary_directory) / "missing.csv"
            with self.assertRaisesRegex(FileNotFoundError, "verified row per curated image"):
                run_model1_training(device="cpu", annotation_csv=missing_manifest)

    def _manifest_rows(self):
        return [
            {"image_index": 0, "vortex_presence": 1, "split": "train", "storm_id": "storm-a",
             "eye_x_center": 0.45, "eye_y_center": 0.52, "eye_width": 0.1, "eye_height": 0.12},
            {"image_index": 1, "vortex_presence": 0, "split": "train", "storm_id": "scene-train"},
            {"image_index": 2, "vortex_presence": 1, "split": "validation", "storm_id": "storm-b"},
            {"image_index": 3, "vortex_presence": 0, "split": "validation", "storm_id": "scene-validation"},
            {"image_index": 4, "vortex_presence": 1, "split": "test", "storm_id": "storm-c"},
            {"image_index": 5, "vortex_presence": 0, "split": "test", "storm_id": "scene-test"},
        ]

    def _load_rows(self, rows):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        manifest_path = Path(temporary_directory.name) / "image_annotations.csv"
        pd.DataFrame(rows).to_csv(manifest_path, index=False)
        return load_image_annotations(manifest_path, sample_count=6)

    def test_manifest_sorts_rows_and_preserves_verified_targets(self):
        manifest = self._load_rows(list(reversed(self._manifest_rows())))

        self.assertEqual(manifest["image_index"].tolist(), list(range(6)))
        self.assertEqual(manifest["vortex_presence"].tolist(), [1, 0, 1, 0, 1, 0])
        self.assertEqual(manifest.loc[0, "eye_x_center"], 0.45)
        self.assertTrue(manifest["eyewall_mask_path"].eq("").all())

    def test_manifest_rejects_storms_crossing_data_splits(self):
        rows = self._manifest_rows()
        rows[2]["storm_id"] = "storm-a"

        with self.assertRaisesRegex(ValueError, "more than one split"):
            self._load_rows(rows)

    def test_manifest_accepts_curated_subset_of_images(self):
        rows = self._manifest_rows()[:4]
        manifest = self._load_rows(rows)

        self.assertEqual(manifest["image_index"].tolist(), [0, 1, 2, 3])

    def test_manifest_rejects_out_of_range_image_index(self):
        rows = self._manifest_rows()
        rows[-1]["image_index"] = 6

        with self.assertRaisesRegex(ValueError, "between 0 and 5"):
            self._load_rows(rows)

    def test_manifest_rejects_partial_eye_boxes(self):
        rows = self._manifest_rows()
        rows[2]["eye_x_center"] = 0.5

        with self.assertRaisesRegex(ValueError, "all four normalized coordinates"):
            self._load_rows(rows)


class SatelliteNormalizationTests(unittest.TestCase):
    def test_training_dataset_uses_shared_per_channel_normalization(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "images.h5"
            source = np.arange(128 * 128 * 4, dtype=np.float32).reshape(128, 128, 4) % 256
            with h5py.File(image_path, "w") as h5_file:
                h5_file.create_dataset("Images", data=source[np.newaxis, ...])

            image_tensor = CycloneSatelliteDataset(str(image_path))[0].numpy()

        expected = np.stack(
            [normalize_satellite_channel(source[:, :, channel]) for channel in range(4)],
            axis=0,
        )
        np.testing.assert_allclose(image_tensor, expected, rtol=0.0, atol=1e-6)

    def test_live_tensor_rejects_missing_spectral_bands(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "single_band.h5"
            with h5py.File(image_path, "w") as h5_file:
                h5_file.create_dataset("IMG_TIR1", data=np.ones((128, 128), dtype=np.float32))

            processor = InsatImageProcessor(output_preview_path=Path(temporary_directory) / "preview.png")
            with self.assertRaisesRegex(KeyError, "IMG_WV"):
                processor.prepare_tensor_for_model1(image_path)


if __name__ == "__main__":
    unittest.main()