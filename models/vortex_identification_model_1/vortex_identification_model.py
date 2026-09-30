"""
CycloneVision AI - Model 1: Vortex Identification & Eye Localization Engine
Combines Neural Network Architecture, Sub-Pixel Coordinate Conversion,
Training/Evaluation Pipeline, CSV Metrics Exporter, and Performance Visualizer.
Saves all output files (pth, csv, png) directly inside this folder.
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, roc_curve
from torch.utils.data import DataLoader, Subset

# Ensure directory paths are configured correctly
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(MODEL_DIR, "..", ".."))

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "data_fusion"))
sys.path.append(os.path.join(BASE_DIR, "data_preprocess"))
sys.path.append(os.path.join(BASE_DIR, "numeric_data"))
sys.path.append(os.path.join(BASE_DIR, "models"))


# =============================================================================
# 1. NEURAL NETWORK ARCHITECTURE
# =============================================================================

class ConvBlock(nn.Module):
    """Basic Conv-BN-ReLU Block for feature extraction."""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class VortexEyeLocalizationModel(nn.Module):
    """
    Multi-Task Deep Learning Model for Cyclone Vortex Identification & Eye Pinpointing.
    
    Inputs:
      - x: Multi-Spectral Satellite Tensor of shape (Batch, 4, Height, Width)
           Ch 0: Infrared (IR)
           Ch 1: Water Vapor (WV)
           Ch 2: Microwave / IR2
           Ch 3: Visible / Reflectance
           
    Outputs:
      - vortex_presence_logit: Raw Logits (Batch, 1) -> Sigmoid for probability
      - vortex_presence_prob: Probability (Batch, 1)
      - eye_bbox: Normalized Bounding Box (Batch, 4) -> [x_center, y_center, width, height]
      - eyewall_mask_logit: Mask Logits (Batch, 1, Height, Width)
      - eyewall_mask_prob: Mask Probabilities (Batch, 1, Height, Width)
    """
    def __init__(self, in_channels=4, num_features=32):
        super(VortexEyeLocalizationModel, self).__init__()
        
        # Encoder (Feature Extraction Pyramid)
        self.enc1 = ConvBlock(in_channels, num_features)       # (B, 32, H, W)
        self.pool1 = nn.MaxPool2d(2, 2)                         # (B, 32, H/2, W/2)
        
        self.enc2 = ConvBlock(num_features, num_features * 2)   # (B, 64, H/2, W/2)
        self.pool2 = nn.MaxPool2d(2, 2)                         # (B, 64, H/4, W/4)
        
        self.enc3 = ConvBlock(num_features * 2, num_features * 4)# (B, 128, H/4, W/4)
        self.pool3 = nn.MaxPool2d(2, 2)                         # (B, 128, H/8, W/8)
        
        # Bottleneck
        self.bottleneck = ConvBlock(num_features * 4, num_features * 8) # (B, 256, H/8, W/8)
        
        # 1. Classification Head (Vortex Presence Detection)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(num_features * 8, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        
        # 2. Eye Coordinate Bounding Box Head (Regression: [x_c, y_c, w, h])
        self.regressor = nn.Sequential(
            nn.Linear(num_features * 8, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
            nn.Sigmoid()  # Coordinates normalized between 0.0 and 1.0
        )
        
        # 3. Eyewall Segmentation Decoder Head (U-Net Decoder)
        self.up3 = nn.ConvTranspose2d(num_features * 8, num_features * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(num_features * 8, num_features * 4)
        
        self.up2 = nn.ConvTranspose2d(num_features * 4, num_features * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(num_features * 4, num_features * 2)
        
        self.up1 = nn.ConvTranspose2d(num_features * 2, num_features, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(num_features * 2, num_features)
        
        self.mask_head = nn.Conv2d(num_features, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        b = self.bottleneck(p3)
        
        pooled = self.gap(b).view(b.size(0), -1)
        vortex_presence_logit = self.classifier(pooled)
        eye_bbox = self.regressor(pooled)
        
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        eyewall_mask_logit = self.mask_head(d1)
        
        return {
            'vortex_presence_logit': vortex_presence_logit,
            'vortex_presence_prob': torch.sigmoid(vortex_presence_logit),
            'eye_bbox': eye_bbox,
            'eyewall_mask_logit': eyewall_mask_logit,
            'eyewall_mask_prob': torch.sigmoid(eyewall_mask_logit)
        }


# =============================================================================
# 2. GEOGRAPHIC COORDINATE CONVERTER
# =============================================================================

def convert_bbox_to_latlon(bbox_normalized, center_lat, center_lon, patch_km=500):
    """
    Converts normalized eye bounding box [x_c, y_c, w, h] to geographic Lat/Lon pinpoint.
    """
    x_c, y_c, w, h = bbox_normalized
    offset_x_deg = (x_c - 0.5) * (patch_km / 111.0)
    offset_y_deg = (0.5 - y_c) * (patch_km / 111.0)
    
    eye_lat = center_lat + offset_y_deg
    eye_lon = center_lon + offset_x_deg
    eye_diameter_km = w * patch_km
    
    return eye_lat, eye_lon, eye_diameter_km


# =============================================================================
# 3. VISUALIZATION & METRICS PLOTTER
# =============================================================================

def generate_performance_plots(history_df, test_y_true, test_y_prob, metrics_dict, save_path):
    """Generates 4-panel performance dashboard (Loss, Acc, ROC Curve, Metrics Summary)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    
    epochs = history_df['epoch']
    
    # 1. Loss Curves
    axes[0, 0].plot(epochs, history_df['train_loss'], 'b-o', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history_df['val_loss'], 'r-s', label='Val Loss', linewidth=2)
    axes[0, 0].set_title('Model 1: Loss Curves Across Epochs', fontweight='bold', fontsize=12)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Multi-Task Loss')
    axes[0, 0].legend()
    
    # 2. Accuracy Curves
    axes[0, 1].plot(epochs, history_df['train_acc'], 'g-o', label='Train Acc (%)', linewidth=2)
    axes[0, 1].plot(epochs, history_df['val_acc'], 'm-s', label='Val Acc (%)', linewidth=2)
    axes[0, 1].axhline(y=metrics_dict['test_accuracy'], color='orange', linestyle='--', label=f"Test Acc ({metrics_dict['test_accuracy']:.1f}%)")
    axes[0, 1].set_title('Model 1: Accuracy Progression (Train vs Val vs Test)', fontweight='bold', fontsize=12)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    
    # 3. ROC Curve
    if len(np.unique(test_y_true)) > 1:
        fpr, tpr, _ = roc_curve(test_y_true, test_y_prob)
        axes[1, 0].plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC Curve (AUC = {metrics_dict['roc_auc']:.3f})")
        axes[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[1, 0].set_xlim([0.0, 1.0])
        axes[1, 0].set_ylim([0.0, 1.05])
        axes[1, 0].set_xlabel('False Positive Rate')
        axes[1, 0].set_ylabel('True Positive Rate')
        axes[1, 0].set_title('Receiver Operating Characteristic (ROC) Curve', fontweight='bold', fontsize=12)
        axes[1, 0].legend(loc="lower right")
    else:
        axes[1, 0].text(0.5, 0.5, f"ROC-AUC: {metrics_dict['roc_auc']:.3f}\n(High Performance Baseline)", ha='center', va='center', fontsize=12)
        axes[1, 0].set_title('ROC Curve', fontweight='bold')

    # 4. Metrics Summary Bar Chart
    metric_names = ['Train Acc', 'Val Acc', 'Test Acc', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    metric_vals = [
        metrics_dict['train_accuracy'] / 100.0,
        metrics_dict['val_accuracy'] / 100.0,
        metrics_dict['test_accuracy'] / 100.0,
        metrics_dict['precision'],
        metrics_dict['recall'],
        metrics_dict['f1_score'],
        metrics_dict['roc_auc']
    ]
    colors = ['#2ca02c', '#9467bd', '#ff7f0e', '#1f77b4', '#17becf', '#d62728', '#bcbd22']
    bars = axes[1, 1].bar(metric_names, metric_vals, color=colors)
    axes[1, 1].set_ylim([0.0, 1.15])
    axes[1, 1].set_title(f"Model 1 Metrics (Best Epoch: #{metrics_dict['best_epoch']})", fontweight='bold', fontsize=12)
    axes[1, 1].set_ylabel('Score [0 - 1.0]')
    for bar in bars:
        height = bar.get_height()
        axes[1, 1].annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[+] Performance Plots saved directly to: {save_path}")


def get_device():
    """
    Automatically detects and selects the best available compute device:
    1. CUDA (NVIDIA GPU)
    2. DirectML (AMD GPU on Windows, e.g. AMD Radeon RX 6500M)
    3. CPU (AMD Ryzen Processor fallback)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        # pyrefly: ignore [missing-import]
        import torch_directml
        return torch_directml.device()
    except Exception:
        return torch.device("cpu")


# =============================================================================
# 4. TRAINING & EVALUATION PIPELINE (OUTPUT SAVED HERE IN LOCAL FOLDER)
# =============================================================================

def run_model1_training(num_epochs=1, batch_size=32, lr=1e-3, device=None, annotation_csv=None):
    if device is None:
        device = get_device()
        
    print("=" * 75)
    print("CYCLONEVISION AI - MODEL 1 (VORTEX IDENTIFICATION) TRAINING & METRICS")
    print(f"Device: {str(device).upper()} | Epochs: {num_epochs} | Batch Size: {batch_size}")
    print(f"Output Directory: {MODEL_DIR}")
    print("=" * 75)
    
    from data_fusion.data_fusion_pipeline import CycloneVisionMultiModalDataset
    
    if annotation_csv is None:
        annotation_csv = os.path.join(BASE_DIR, "cyclone_image", "image_annotations.csv")
    if not os.path.exists(annotation_csv):
        raise FileNotFoundError(
            "Model 1 training requires cyclone_image/image_annotations.csv with one verified row per curated image. "
            "Required columns: image_index,vortex_presence,split,storm_id. Optional eye bbox columns: "
            "eye_x_center,eye_y_center,eye_width,eye_height (normalized 0..1); optional "
            "eyewall_mask_path points to a 128x128 .npy mask relative to the CSV."
        )
    
    print("[*] Loading Multi-Modal Dataset...")
    full_dataset = CycloneVisionMultiModalDataset(
        h5_path=os.path.join(BASE_DIR, "cyclone_image", "Cyclone_Images.h5"),
        nwp_nc_path=os.path.join(BASE_DIR, "era5_sample_nio.nc"),
        ibtracs_csv=os.path.join(BASE_DIR, "numeric_data", "ibtracs.NI.list.v04r01.csv"),
        image_annotations_csv=annotation_csv,
    )

    annotations = full_dataset.image_annotations
    split_indices = {
        split_name: np.flatnonzero(annotations["split"].to_numpy() == split_name).tolist()
        for split_name in ["train", "validation", "test"]
    }
    for split_name, indices in split_indices.items():
        split_annotations = annotations.iloc[indices]
        labels = split_annotations["vortex_presence"].to_numpy()
        if not indices or np.unique(labels).size != 2:
            raise ValueError(f"The {split_name} split must contain both cyclone and non-cyclone samples")
        positive_annotations = split_annotations[split_annotations["vortex_presence"] == 1]
        bbox_values = positive_annotations[["eye_x_center", "eye_y_center", "eye_width", "eye_height"]].to_numpy(dtype=np.float32)
        if not np.isfinite(bbox_values).all():
            raise ValueError(f"Every positive image in the {split_name} split needs a verified eye bounding box")
        if not positive_annotations["eyewall_mask_path"].astype(str).str.strip().ne("").all():
            raise ValueError(f"Every positive image in the {split_name} split needs a verified eyewall mask")

    train_indices = split_indices["train"]
    val_indices = split_indices["validation"]
    test_indices = split_indices["test"]
    
    train_set = Subset(full_dataset, train_indices)
    val_set = Subset(full_dataset, val_indices)
    test_set = Subset(full_dataset, test_indices)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    print(f"[+] Data Loaders Ready: Train={len(train_set):,} | Val={len(val_set):,} | Test={len(test_set):,}")
    
    model = VortexEyeLocalizationModel(in_channels=4).to(device)
    
    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_bbox = nn.SmoothL1Loss()
    criterion_mask = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    def compute_multitask_loss(outputs, targets):
        target_presence = targets["vortex_presence"].to(device)
        target_bbox = targets["eye_bbox"].to(device)
        target_mask = targets["eyewall_mask"].to(device)
        bbox_valid = targets["bbox_valid"].to(device).bool().reshape(-1)
        mask_valid = targets["mask_valid"].to(device).bool().reshape(-1)

        loss_cls = criterion_cls(outputs["vortex_presence_logit"], target_presence)
        loss_bbox = (
            criterion_bbox(outputs["eye_bbox"][bbox_valid], target_bbox[bbox_valid])
            if bbox_valid.any()
            else outputs["eye_bbox"].sum() * 0.0
        )
        loss_mask = (
            criterion_mask(outputs["eyewall_mask_logit"][mask_valid], target_mask[mask_valid])
            if mask_valid.any()
            else outputs["eyewall_mask_logit"].sum() * 0.0
        )
        return loss_cls + 2.0 * loss_bbox + loss_mask, target_presence
    
    best_val_loss = float("inf")
    best_epoch = 1
    
    # SAVE OUTPUTS DIRECTLY HERE IN LOCAL MODEL FOLDER
    model_save_path = os.path.join(MODEL_DIR, "best_model1_vortex_eye.pth")
    csv_save_path = os.path.join(MODEL_DIR, "model1_training_metrics.csv")
    plots_save_path = os.path.join(MODEL_DIR, "model1_performance_plots.png")
    
    epoch_history = []
    start_time = time.time()
    
    # Training Loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct_presence = 0
        total_samples = 0
        
        for batch_idx, (img_tensor, nwp_vector, targets) in enumerate(train_loader):
            img_tensor = img_tensor.to(device)
            batch_size_curr = img_tensor.size(0)
            
            optimizer.zero_grad()
            outputs = model(img_tensor)

            total_loss, target_presence = compute_multitask_loss(outputs, targets)
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item() * batch_size_curr
            preds = (outputs['vortex_presence_prob'] >= 0.5).float()
            correct_presence += (preds == target_presence).sum().item()
            total_samples += batch_size_curr
            
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"  Epoch [{epoch}/{num_epochs}] | Batch [{batch_idx+1}/{len(train_loader)}] | Loss: {total_loss.item():.4f}")
                
        train_loss = running_loss / total_samples
        train_acc = (correct_presence / total_samples) * 100.0
        
        # Validation Pass
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_samples = 0
        val_y_true = []
        val_y_pred = []
        with torch.no_grad():
            for img_tensor, _, targets in val_loader:
                img_tensor = img_tensor.to(device)
                b_sz = img_tensor.size(0)
                outputs = model(img_tensor)
                t_loss, target_presence = compute_multitask_loss(outputs, targets)
                
                val_loss += t_loss.item() * b_sz
                val_preds = (outputs['vortex_presence_prob'] >= 0.5).float()
                val_correct += (val_preds == target_presence).sum().item()
                val_samples += b_sz
                val_y_true.extend(target_presence.detach().cpu().numpy().flatten().tolist())
                val_y_pred.extend(val_preds.detach().cpu().numpy().flatten().tolist())
                
        val_loss /= val_samples
        val_acc = (val_correct / val_samples) * 100.0
        val_f1 = float(f1_score(val_y_true, val_y_pred, zero_division=0))
        val_precision = float(precision_score(val_y_true, val_y_pred, zero_division=0))
        
        print(f"==> Epoch {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}%")
        
        epoch_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'val_prec': val_precision
        })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), model_save_path)
            print(f"  [+] Saved Best Model Checkpoint to: {model_save_path} (Best Epoch: #{best_epoch})")
            
    elapsed = time.time() - start_time
    print(f"\n[+] Model 1 Training Completed in {elapsed:.1f} seconds!")
    
    # 5. Final Evaluation on Unseen Test Data
    print("\n" + "=" * 75)
    print(f"FINAL TEST EVALUATION ON UNSEEN DATA ({len(test_set):,} SAMPLES)")
    print("=" * 75)
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
    model.eval()
    
    all_y_true = []
    all_y_pred = []
    all_y_prob = []
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for img_tensor, _, targets in test_loader:
            img_tensor = img_tensor.to(device)
            b_sz = img_tensor.size(0)
            target_presence = targets["vortex_presence"].to(device)
            outputs = model(img_tensor)
            
            probs = outputs['vortex_presence_prob'].cpu().numpy().flatten()
            preds = (probs >= 0.5).astype(int)
            targets = target_presence.cpu().numpy().flatten().astype(int)
            
            all_y_prob.extend(probs)
            all_y_pred.extend(preds)
            all_y_true.extend(targets)
            
            test_correct += (preds == targets).sum()
            test_total += b_sz
            
    test_acc = (test_correct / test_total) * 100.0
    
    roc_auc = float(roc_auc_score(all_y_true, all_y_prob))
    f1 = float(f1_score(all_y_true, all_y_pred, zero_division=0))
    precision = float(precision_score(all_y_true, all_y_pred, zero_division=0))
    recall = float(recall_score(all_y_true, all_y_pred, zero_division=0))

    final_metrics = {
        'best_epoch': best_epoch,
        'train_accuracy': epoch_history[-1]['train_acc'],
        'val_accuracy': epoch_history[-1]['val_acc'],
        'test_accuracy': test_acc,
        'roc_auc': roc_auc,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'best_val_loss': best_val_loss
    }
    
    # Save Epoch History CSV directly inside this folder
    history_df = pd.DataFrame(epoch_history)
    history_df.to_csv(csv_save_path, index=False)
    print(f"[+] CSV Exported directly to: {csv_save_path}")
    
    # Generate & Save Performance Visualizations directly inside this folder
    generate_performance_plots(history_df, all_y_true, all_y_prob, final_metrics, plots_save_path)
    
    print("\n" + "=" * 75)
    print("CYCLONEVISION AI - MODEL 1 FINAL PERFORMANCE METRICS")
    print("=" * 75)
    print(f"  🟢 Best Epoch          : #{best_epoch}")
    print(f"  🟢 Training Accuracy   : {final_metrics['train_accuracy']:.2f}%")
    print(f"  🟢 Validation Accuracy : {final_metrics['val_accuracy']:.2f}%")
    print(f"  🟢 Test Accuracy       : {final_metrics['test_accuracy']:.2f}%")
    print(f"  🔵 ROC-AUC Score       : {final_metrics['roc_auc']:.4f}")
    print(f"  🔵 F1-Score            : {final_metrics['f1_score']:.4f}")
    print(f"  🔵 Precision           : {final_metrics['precision']:.4f}")
    print(f"  🔵 Recall              : {final_metrics['recall']:.4f}")
    print("-" * 75)
    print(f"  📁 SAVED MODEL FILE    : {model_save_path}")
    print(f"  📁 SAVED CSV METRICS   : {csv_save_path}")
    print(f"  🖼️ SAVED PLOTS IMAGE   : {plots_save_path}")
    print("=" * 75)
    
    return final_metrics


if __name__ == "__main__":
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
    run_model1_training(num_epochs=1, batch_size=32)
