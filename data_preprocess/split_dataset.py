"""
CycloneVision AI - Dataset Splitter & Exporter
Splits the multi-modal cyclone dataset into Train (70%), Validation (15%), and Test (15%)
and saves the split index file to disk.
"""

import os
import sys
import json
import numpy as np

# Ensure parent directory is in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "data_fusion"))
sys.path.append(os.path.join(BASE_DIR, "data_preprocess"))
sys.path.append(os.path.join(BASE_DIR, "numeric_data"))

def generate_and_save_splits(
    h5_path=os.path.join(BASE_DIR, "cyclone_image", "Cyclone_Images.h5"),
    nwp_nc_path=os.path.join(BASE_DIR, "era5_sample_nio.nc"),
    ibtracs_csv=os.path.join(BASE_DIR, "numeric_data", "ibtracs.NI.list.v04r01.csv"),
    output_json_path=os.path.join(BASE_DIR, "data_preprocess", "cyclone_dataset_splits.json"),
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    random_seed=42
):
    """
    Loads dataset metadata, generates train/val/test splits, and saves to JSON file.
    """
    from data_fusion_pipeline import CycloneVisionMultiModalDataset

    print("[*] Initializing CycloneVision Multi-Modal Dataset...")
    dataset = CycloneVisionMultiModalDataset(
        h5_path=h5_path,
        nwp_nc_path=nwp_nc_path,
        ibtracs_csv=ibtracs_csv
    )
    
    total_samples = len(dataset)
    np.random.seed(random_seed)
    indices = np.arange(total_samples)
    
    num_train = int(total_samples * train_ratio)
    num_val = int(total_samples * val_ratio)
    num_test = total_samples - num_train - num_val
    
    # Shuffle indices with fixed seed for reproducibility
    np.random.shuffle(indices)
    
    train_idx = indices[:num_train].tolist()
    val_idx = indices[num_train:num_train + num_val].tolist()
    test_idx = indices[num_train + num_val:].tolist()
    
    split_data = {
        "dataset_name": "CycloneVision_AI_MultiModal_Dataset",
        "total_samples": total_samples,
        "seed": random_seed,
        "splits": {
            "train": {
                "count": len(train_idx),
                "percentage": f"{train_ratio * 100:.1f}%",
                "indices": train_idx
            },
            "validation": {
                "count": len(val_idx),
                "percentage": f"{val_ratio * 100:.1f}%",
                "indices": val_idx
            },
            "test": {
                "count": len(test_idx),
                "percentage": f"{test_ratio * 100:.1f}%",
                "indices": test_idx
            }
        }
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    
    with open(output_json_path, "w") as f:
        json.dump(split_data, f, indent=2)
        
    abs_output_path = os.path.abspath(output_json_path)
    
    print("\n" + "=" * 65)
    print("CYCLONEVISION AI - DATASET SPLIT COMPLETED & SAVED")
    print("=" * 65)
    print(f"  • Total Dataset Samples  : {total_samples:,}")
    print(f"  • Training Samples (70%) : {len(train_idx):,}")
    print(f"  • Validation Samples(15%): {len(val_idx):,}")
    print(f"  • Testing Samples (15%)  : {len(test_idx):,}")
    print("-" * 65)
    print(f"  🟢 SAVED SPLIT FILE LOCATION:")
    print(f"     {abs_output_path}")
    print("=" * 65)
    
    return abs_output_path


if __name__ == "__main__":
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
            
    saved_file_path = generate_and_save_splits()
