"""
CycloneVision AI - Satellite Imagery Dataset Loader & Visualizer
Processes 'Cyclone_Images.h5' containing multi-spectral satellite imagery patches (128x128x4).
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

from data_preprocess.normalization import normalize_satellite_channel

class CycloneSatelliteDataset(Dataset):
    """
    PyTorch Dataset for Cyclone Multi-Spectral Satellite Images.
    Channels typically correspond to:
      Ch 0: Infrared (IR1 / 10.8 µm - Cloud-top brightness temperature)
      Ch 1: Water Vapor (WV / 6.7 µm - Upper tropospheric moisture)
      Ch 2: Passive Microwave / IR2 (Eye and precipitation structure)
      Ch 3: Visible / Normalized reflectance
    """
    def __init__(self, h5_path="Cyclone_Images.h5", transform=None, normalize=True):
        if not os.path.exists(h5_path):
            alt_path = os.path.join(os.path.dirname(__file__), "..", "cyclone_image", "Cyclone_Images.h5")
            if os.path.exists(alt_path):
                h5_path = alt_path
        self.h5_path = h5_path
        self.transform = transform
        self.normalize = normalize
        
        # Open file to get length
        with h5py.File(self.h5_path, 'r') as f:
            self.length = len(f['Images'])
            self.shape = f['Images'].shape
            
        self.h5_file = None  # Lazy loading for multi-processing compatibility

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
            
        img = self.h5_file['Images'][idx].astype(np.float32)  # Shape: (128, 128, 4)
        
        # Clean NaNs and Infs from raw satellite telemetry / missing channel scans
        img = np.nan_to_num(img, nan=0.0, posinf=255.0, neginf=0.0)
        
        if img.ndim != 3 or img.shape[-1] != 4:
            raise ValueError(f"Expected an image with four channels, got shape {img.shape}")

        # Normalize each band with the same percentile rule used at inference.
        if self.normalize:
            img = np.stack(
                [normalize_satellite_channel(img[:, :, channel]) for channel in range(4)],
                axis=-1,
            )
            
        # Convert HWC to CHW for PyTorch (4, 128, 128)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        
        if self.transform:
            img_tensor = self.transform(img_tensor)
            
        return img_tensor


def visualize_sample_channels(h5_path="Cyclone_Images.h5", sample_idx=0, save_path="cyclone_multispectral_sample.png"):
    """
    Renders and saves a 4-channel breakdown of a cyclone satellite observation.
    """
    with h5py.File(h5_path, 'r') as f:
        total_samples = len(f['Images'])
        if sample_idx >= total_samples:
            sample_idx = 0
        img = f['Images'][sample_idx]  # Shape: (128, 128, 4)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    channel_names = [
        "Channel 1: Infrared (IR)",
        "Channel 2: Water Vapor (WV)",
        "Channel 3: Microwave / IR2",
        "Channel 4: Visible / Refl"
    ]
    colormaps = ['gist_ncar', 'Blues_r', 'magma', 'gray']

    for ch in range(4):
        im = axes[ch].imshow(img[:, :, ch], cmap=colormaps[ch])
        axes[ch].set_title(channel_names[ch], fontsize=11, fontweight='bold')
        axes[ch].axis('off')
        plt.colorbar(im, ax=axes[ch], fraction=0.046, pad=0.04)

    plt.suptitle(f"CycloneVision AI - Multi-Spectral Satellite Channels (Sample #{sample_idx})", fontsize=13, y=1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[+] Multi-spectral visualization saved to {save_path}")


if __name__ == "__main__":
    print("[*] Initializing CycloneSatelliteDataset...")
    dataset = CycloneSatelliteDataset("Cyclone_Images.h5")
    print(f"[+] Total Satellite Samples: {len(dataset):,}")
    print(f"[+] Tensor Shape per sample: {dataset[0].shape}")

    # Generate sample visualization
    visualize_sample_channels("Cyclone_Images.h5", sample_idx=42)
