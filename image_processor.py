"""
CycloneVision AI - Satellite Image Processor
============================================
Reads raw INSAT-3DS / INSAT-3DR HDF5/HDF meteorological products,
extracts spectral channels (TIR1, WV, MIR, VIS), handles NaNs/Infs,
normalizes dynamic range, and renders dashboard-compatible previews.

Responsibilities:
- Inspect and read downloaded HDF/HDF5 products
- Extract configured INSAT channel (e.g. IMG_TIR1, IMG_WV)
- Robustly sanitize NaN, Inf, and satellite telemetry fill-values
- Normalize pixel values for meteorological visualization
- Generate dashboard-compatible preview image saved to data/live/preview/latest.png
- Format 4-channel multi-spectral tensor (4, 128, 128) for Model 1 inference
"""

from __future__ import annotations

import os
import sys
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # Non-interactive background backend
import matplotlib.pyplot as plt
from data_preprocess.normalization import normalize_satellite_channel

logger = logging.getLogger("ImageProcessor")

BASE_DIR = Path(__file__).resolve().parent
LIVE_DATA_DIR = BASE_DIR / "data" / "live"
LATEST_DATA_DIR = LIVE_DATA_DIR / "latest"
PREVIEW_DIR = LIVE_DATA_DIR / "preview"
DEFAULT_PREVIEW_PATH = PREVIEW_DIR / "latest.png"

# Colormap mapping for INSAT sensors
CHANNEL_COLORMAPS = {
    "IMG_TIR1": "inferno",   # Cloud-top brightness temperature
    "IMG_TIR2": "magma",     # Split-window infrared
    "IMG_WV": "Blues_r",     # Upper-tropospheric moisture
    "IMG_MIR": "viridis",    # Mid-infrared
    "IMG_VIS": "gray",       # Visible reflectance
    "DEFAULT": "inferno",
}


class InsatImageProcessor:
    """
    Processes INSAT-3DS / INSAT-3DR HDF5 satellite imagery.
    """

    def __init__(
        self,
        target_channel: str = "IMG_TIR1",
        output_preview_path: Optional[Union[str, Path]] = None,
    ):
        self.target_channel = target_channel
        self.output_preview_path = Path(output_preview_path or DEFAULT_PREVIEW_PATH)
        self.output_preview_path.parent.mkdir(parents=True, exist_ok=True)

    def find_latest_product_file(self) -> Optional[Path]:
        """Locates the newest downloaded HDF5 product in data/live/latest/."""
        # 1. Check for standard symlink/copy latest_product.h5
        std_path = LATEST_DATA_DIR / "latest_product.h5"
        if std_path.exists() and std_path.stat().st_size > 1024:
            return std_path

        # 2. Look for any .h5 / .hdf / .nc in data/live/latest
        files = sorted(
            [
                p for p in LATEST_DATA_DIR.glob("*")
                if p.is_file() and p.suffix.lower() in {".h5", ".hdf", ".hdf5", ".nc"}
            ],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if files:
            return files[0]

        # 3. Fallback to existing project sample (e.g. Cyclone_Images.h5)
        sample_path = BASE_DIR / "cyclone_image" / "Cyclone_Images.h5"
        if sample_path.exists():
            return sample_path

        return None

    def _locate_dataset(self, h5_obj: h5py.File, channel_name: str) -> Optional[h5py.Dataset]:
        """
        Recursively searches the HDF5 group tree for a dataset matching the channel name.
        Handles variations like IMG_TIR1, IMG_TIR1_TEMP, IMG_TIR1_RAD, Images, etc.
        """
        target_keys = [
            channel_name,
            f"{channel_name}_TEMP",
            f"{channel_name}_RAD",
            f"{channel_name}_RAW",
            channel_name.replace("IMG_", ""),
        ]

        # Direct root search
        for key in target_keys:
            if key in h5_obj:
                ds = h5_obj[key]
                if isinstance(ds, h5py.Dataset):
                    return ds

        # Recursive search across groups
        found_dataset = None

        def visitor(name, node):
            nonlocal found_dataset
            if found_dataset is not None:
                return
            if isinstance(node, h5py.Dataset):
                base_name = name.split("/")[-1]
                for tk in target_keys:
                    if tk.lower() == base_name.lower():
                        found_dataset = node
                        return

        h5_obj.visititems(visitor)
        if found_dataset is not None:
            return found_dataset

        # If Images array exists (such as preprocessed dataset)
        if "Images" in h5_obj and isinstance(h5_obj["Images"], h5py.Dataset):
            return h5_obj["Images"]

        return None

    def read_channel_data(
        self,
        h5_file_path: Union[str, Path],
        channel: Optional[str] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reads raw dataset array from HDF5 file, sanitizes NaNs/Infs,
        and returns clean 2D float32 array along with metadata.
        """
        path = Path(h5_file_path)
        if not path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {path}")

        req_channel = channel or self.target_channel
        channel_index_map = {"IMG_TIR1": 0, "IMG_WV": 1, "IMG_TIR2": 2, "IMG_VIS": 3}

        with h5py.File(path, "r") as f:
            ds = self._locate_dataset(f, req_channel)

            if ds is None:
                raise KeyError(f"Could not locate required channel '{req_channel}' in {path}")

            raw_shape = ds.shape
            # Handle multi-dimensional datasets
            if ds.name.endswith("Images") and ds.ndim == 4:
                # Shape (N, 128, 128, 4) - pick index 0, channel index
                ch_idx = channel_index_map.get(req_channel, 0)
                arr = ds[0, :, :, ch_idx].astype(np.float32)
            elif ds.name.endswith("Images") and ds.ndim == 3 and ds.shape[-1] == 4:
                # Shape (128, 128, 4)
                ch_idx = channel_index_map.get(req_channel, 0)
                arr = ds[:, :, ch_idx].astype(np.float32)
            elif ds.name.endswith("Images"):
                raise ValueError(f"Expected a four-channel Images dataset, got shape {ds.shape} in {path}")
            elif ds.ndim == 3:
                # (Channels, Height, Width) or (Height, Width, Channels)
                if ds.shape[0] in {1, 3, 4, 6}:
                    ch_idx = min(channel_index_map.get(req_channel, 0), ds.shape[0] - 1)
                    arr = ds[ch_idx, :, :].astype(np.float32)
                else:
                    ch_idx = min(channel_index_map.get(req_channel, 0), ds.shape[-1] - 1)
                    arr = ds[:, :, ch_idx].astype(np.float32)
            elif ds.ndim == 2:
                arr = ds[:].astype(np.float32)
            else:
                raise ValueError(f"Unsupported dataset dimensions {ds.shape} for channel {req_channel}")

            meta = {
                "dataset_name": ds.name,
                "original_shape": list(raw_shape),
                "channel": req_channel,
                "file": path.name,
            }

        # Handle NaN and Inf values
        arr = self.clean_nan_inf(arr)

        return arr, meta

    def clean_nan_inf(self, arr: np.ndarray) -> np.ndarray:
        """
        Replaces NaN, +Inf, -Inf and satellite fill-values with valid numerical boundaries.
        """
        # Replace non-finite values with 0
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        # INSAT fill values often use negative sentinel flags like -999.0 or extreme saturation values
        arr = np.where(arr < -500.0, 0.0, arr)
        arr = np.where(arr > 65000.0, 0.0, arr)

        return arr.astype(np.float32)

    def normalize_for_visualization(
        self,
        arr: np.ndarray,
        clip_percentiles: Tuple[float, float] = (1.5, 98.5),
    ) -> np.ndarray:
        """
        Normalizes 2D satellite raster array to [0.0, 1.0] for high-contrast meteorological visualization.
        """
        return normalize_satellite_channel(arr, clip_percentiles)

    def generate_preview_image(
        self,
        norm_arr: np.ndarray,
        channel: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
        title_meta: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Generates and saves a high-quality dashboard-compatible PNG preview.
        """
        out_path = Path(save_path or self.output_preview_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        req_channel = channel or self.target_channel
        cmap_name = CHANNEL_COLORMAPS.get(req_channel, CHANNEL_COLORMAPS["DEFAULT"])

        fig, ax = plt.subplots(figsize=(6, 6), dpi=150, facecolor="#0a1220")
        ax.set_facecolor("#0a1220")

        # Invert IR thermal visualization if desired so cold cyclone eye cloud tops appear vivid
        im = ax.imshow(norm_arr, cmap=cmap_name, origin="upper", aspect="equal")
        ax.axis("off")

        # Subtle clean title overlay if metadata available
        if title_meta:
            sat_title = title_meta.get("satellite", "INSAT")
            ts_str = title_meta.get("timestamp", "")
            ax.set_title(
                f"{sat_title} · {req_channel} · {ts_str}",
                color="#e2e8f0",
                fontsize=9,
                pad=6,
                fontweight="bold",
            )

        plt.subplots_adjust(left=0, right=1, bottom=0, top=0.94 if title_meta else 1)
        plt.savefig(
            out_path,
            dpi=150,
            bbox_inches="tight",
            pad_inches=0.04 if title_meta else 0,
            facecolor=fig.get_facecolor(),
        )
        plt.close(fig)

        # Also write a standard 8-bit RGB PNG directly using PIL for ultra-fast serving
        cmap = plt.get_cmap(cmap_name)
        rgba = (cmap(norm_arr) * 255.0).astype(np.uint8)
        img_pil = Image.fromarray(rgba, mode="RGBA").convert("RGB")
        direct_png = out_path.with_suffix(".png")
        img_pil.save(direct_png, format="PNG", optimize=True)

        logger.info("[+] Generated dashboard satellite preview: %s", out_path)
        return out_path

    def prepare_tensor_for_model1(
        self,
        h5_file_path: Union[str, Path],
        target_size: Tuple[int, int] = (128, 128),
    ) -> np.ndarray:
        """
        Builds a 4-channel multi-spectral numpy tensor (4, 128, 128) tailored for Model 1:
          Ch 0: Infrared (IR / IMG_TIR1)
          Ch 1: Water Vapor (WV / IMG_WV)
          Ch 2: Microwave / IR2 (IMG_TIR2)
          Ch 3: Visible / Reflectance (IMG_VIS)
        """
        path = Path(h5_file_path)
        channels = ["IMG_TIR1", "IMG_WV", "IMG_TIR2", "IMG_VIS"]
        channel_arrays = []

        # Require real spectral bands; synthetic substitutes hide missing input data.
        for ch in channels:
            ch_arr, _ = self.read_channel_data(path, channel=ch)
            ch_norm = self.normalize_for_visualization(ch_arr)
            ch_pil = Image.fromarray((ch_norm * 255.0).astype(np.uint8)).resize(
                target_size, Image.Resampling.BILINEAR
            )
            channel_arrays.append(np.asarray(ch_pil, dtype=np.float32) / 255.0)

        # Stack to (4, 128, 128)
        multi_tensor = np.stack(channel_arrays, axis=0).astype(np.float32)
        return multi_tensor

    def process_latest_frame(
        self,
        h5_path: Optional[Union[str, Path]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Complete processing workflow for the newest downloaded satellite frame:
        1. Reads HDF5
        2. Cleans NaNs/Infs
        3. Normalizes
        4. Saves preview to data/live/preview/latest.png
        5. Returns structured result with normalized tensor
        """
        input_file = Path(h5_path) if h5_path else self.find_latest_product_file()
        if input_file is None:
            raise FileNotFoundError("No satellite HDF5 product available to process.")

        logger.info("Processing satellite product: %s", input_file)
        raw_arr, read_meta = self.read_channel_data(input_file, channel=self.target_channel)
        norm_arr = self.normalize_for_visualization(raw_arr)

        preview_meta = metadata or read_meta
        saved_preview = self.generate_preview_image(
            norm_arr,
            channel=self.target_channel,
            save_path=self.output_preview_path,
            title_meta=preview_meta,
        )

        model1_tensor = self.prepare_tensor_for_model1(input_file)

        return {
            "status": "processed",
            "file": input_file.name,
            "file_path": str(input_file),
            "preview_path": str(saved_preview),
            "channel": self.target_channel,
            "dimensions": list(norm_arr.shape),
            "min_val": float(norm_arr.min()),
            "max_val": float(norm_arr.max()),
            "mean_val": float(norm_arr.mean()),
            "model1_tensor": model1_tensor,
            "metadata": preview_meta,
        }


# Default module-level processor
default_image_processor = InsatImageProcessor()


if __name__ == "__main__":
    print("=" * 70)
    print("   CycloneVision AI - Satellite Image Processor Test")
    print("=" * 70)
    proc = InsatImageProcessor()
    target_file = proc.find_latest_product_file()
    print(f"[*] Target HDF5 Product : {target_file}")
    if target_file and target_file.exists():
        res = proc.process_latest_frame(target_file)
        print(f"[+] Preview Image Saved : {res['preview_path']}")
        print(f"[+] Multi-Spectral Tensor: {res['model1_tensor'].shape}")
        print(f"[+] Image Stats         : min={res['min_val']:.3f}, max={res['max_val']:.3f}, mean={res['mean_val']:.3f}")
    else:
        print("[!] No HDF5 file found on disk.")
    print("=" * 70)
