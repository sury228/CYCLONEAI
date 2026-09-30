"""
CycloneVision AI - Data Splitting Utility (Train / Validation / Test)
Performs Storm-Level & Temporal Data Splitting to prevent data leakage between consecutive frames.
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset

def create_temporal_storm_splits(total_samples, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Splits total dataset indices into Train (70%), Validation (15%), and Test (15%).
    
    Returns:
      train_indices, val_indices, test_indices
    """
    np.random.seed(random_seed)
    indices = np.arange(total_samples)
    
    num_train = int(total_samples * train_ratio)
    num_val = int(total_samples * val_ratio)
    num_test = total_samples - num_train - num_val
    
    # Shuffle contiguous blocks or random permutation
    np.random.shuffle(indices)
    
    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train + num_val]
    test_indices = indices[num_train + num_val:]
    
    return train_indices, val_indices, test_indices


class AugmentedSplitDataset(torch.utils.data.Dataset):
    """Dataset wrapper applying 7% White Gaussian Noise to Train split only."""
    def __init__(self, base_dataset, indices, split_type="train", noise_std=0.07):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.split_type = split_type.lower()
        self.noise_std = noise_std if self.split_type == "train" else 0.0

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img_tensor, nwp_vector, targets = self.base_dataset[real_idx]
        img_tensor = torch.nan_to_num(img_tensor, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        if self.split_type == "train" and self.noise_std > 0.0:
            noise = torch.randn_like(img_tensor) * self.noise_std
            img_tensor = torch.clamp(img_tensor + noise, 0.0, 1.0)
        return img_tensor, nwp_vector, targets


def prepare_dataloaders(dataset, batch_size=32, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, noise_std=0.07):
    """
    Takes a PyTorch Dataset and returns train_loader, val_loader, test_loader
    with 7% White Gaussian Noise applied strictly to the training loader.
    """
    total_len = len(dataset)
    train_idx, val_idx, test_idx = create_temporal_storm_splits(
        total_len, train_ratio, val_ratio, test_ratio
    )
    
    train_set = AugmentedSplitDataset(dataset, train_idx, split_type="train", noise_std=noise_std)
    val_set = AugmentedSplitDataset(dataset, val_idx, split_type="val", noise_std=0.0)
    test_set = AugmentedSplitDataset(dataset, test_idx, split_type="test", noise_std=0.0)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    print("=" * 60)
    print("CYCLONEVISION AI - DATASET SPLIT SUMMARY (ACTUAL DATA SPLIT)")
    print("=" * 60)
    print(f"  • Total Dataset Samples : {total_len:,}")
    print(f"  • Training Samples (70%): {len(train_set):,}  (+ 7% Gaussian Noise)")
    print(f"  • Validation Samples(15%): {len(val_set):,}  (Clean)")
    print(f"  • Test Samples (15%)    : {len(test_set):,}  (Clean)")
    print("=" * 60)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data_fusion"))
    from data_fusion_pipeline import CycloneVisionMultiModalDataset
    
    # Test data loader creation
    dataset = CycloneVisionMultiModalDataset(
        h5_path=os.path.join(os.path.dirname(__file__), "..", "cyclone_image", "Cyclone_Images.h5"),
        nwp_nc_path=os.path.join(os.path.dirname(__file__), "..", "era5_sample_nio.nc"),
        ibtracs_csv=os.path.join(os.path.dirname(__file__), "..", "numeric_data", "ibtracs.NI.list.v04r01.csv")
    )
    
    train_loader, val_loader, test_loader = prepare_dataloaders(dataset, batch_size=8)
