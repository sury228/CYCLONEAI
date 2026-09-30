"""
CycloneVision AI - Standalone Prediction Engine
=================================================
Directly loads the trained `.pth` model checkpoint to predict on new data.

Supports:
  1. Testing on any dataset sample index (--sample_idx 100 --split test)
  2. Loading any custom satellite .npy / .h5 / .png / image tensor
  3. Generating a full 4-panel visual dashboard with eye pinpoint & eyewall segmentation

Usage:
  python predict.py --sample_idx 100 --split test
  python predict.py --sample_idx 50 --lat 16.2 --lon 88.5
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Configure sys.path robustly from project root
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Import Model 1 architecture and utilities
try:
    from models.vortex_identification_model_1.vortex_identification_model import (
        VortexEyeLocalizationModel, convert_bbox_to_latlon, get_device
    )
except ImportError:
    p_mod = os.path.join(BASE_DIR, "models", "vortex_identification_model_1")
    if p_mod not in sys.path:
        sys.path.append(p_mod)
    from vortex_identification_model import (
        VortexEyeLocalizationModel, convert_bbox_to_latlon, get_device
    )

from actual_data_split.split_manager import get_actual_data_loaders


def predict_with_checkpoint(
    custom_tensor=None,
    sample_idx=100,
    split="test",
    storm_lat=15.40,
    storm_lon=87.20,
    patch_km=500.0,
    save_plot_path="prediction_output.png",
    model_checkpoint=None
):
    """
    Loads .pth checkpoint directly and runs multi-task prediction.
    """
    device = get_device()
    print(f"[*] Compute Device: {str(device).upper()}")

    # 1. Locate and load .pth checkpoint
    if model_checkpoint is None:
        possible_paths = [
            os.path.join(BASE_DIR, "models", "best_model1_vortex_eye.pth"),
            os.path.join(BASE_DIR, "models", "vortex_identification_model_1", "best_model1_vortex_eye.pth"),
            os.path.join(BASE_DIR, "best_model1_vortex_eye.pth")
        ]
        model_checkpoint = next((p for p in possible_paths if os.path.exists(p)), None)

    if model_checkpoint is None or not os.path.exists(model_checkpoint):
        raise FileNotFoundError(f"Model .pth checkpoint not found in {possible_paths}")

    print(f"[+] Loading model weights from: {model_checkpoint}")
    model = VortexEyeLocalizationModel(in_channels=4).to(device)
    model.load_state_dict(torch.load(model_checkpoint, map_location=device, weights_only=False))
    model.eval()

    # 2. Prepare Input Data
    gt_target = None
    if custom_tensor is not None:
        if isinstance(custom_tensor, np.ndarray):
            img = torch.from_numpy(custom_tensor).float()
        else:
            img = custom_tensor.clone().float()
        if img.ndim == 3:
            img = img.unsqueeze(0)
        sample_desc = "Custom Satellite Image Input"
    else:
        train_l, val_l, test_l = get_actual_data_loaders(batch_size=32, noise_std=0.07)
        loader_map = {"test": test_l, "val": val_l, "train": train_l}
        target_loader = loader_map.get(split.lower(), test_l)
        dataset = target_loader.dataset

        idx = max(0, min(sample_idx, len(dataset) - 1))
        img_raw, nwp, gt_target = dataset[idx]
        img = img_raw.unsqueeze(0)
        sample_desc = f"{split.upper()} Dataset Sample #{idx:,} (of {len(dataset):,})"

    img = torch.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0).to(device)

    # 3. Model Inference
    with torch.no_grad():
        out = model(img)

    presence_prob = float(out["vortex_presence_prob"][0].item())
    pred_presence = bool(presence_prob >= 0.5)
    pred_bbox = out["eye_bbox"][0].cpu().numpy()  # [xc, yc, w, h]
    mask_prob = out["eyewall_mask_prob"][0, 0].cpu().numpy()  # (128, 128)

    # Geographic coordinate pinpointing
    eye_lat, eye_lon, eye_dia_km = convert_bbox_to_latlon(
        pred_bbox, center_lat=storm_lat, center_lon=storm_lon, patch_km=patch_km
    )

    # 4. Ground Truth Comparison (if available)
    gt_presence = None
    gt_bbox = None
    gt_mask = None
    sample_acc = None
    bbox_error = None
    spatial_err_km = None
    dice_score = None

    if gt_target is not None:
        gt_presence = bool(gt_target["vortex_presence"].item() >= 0.5)
        gt_bbox = gt_target["eye_bbox"].numpy()
        gt_mask = gt_target["eyewall_mask"][0].numpy()

        sample_acc = 100.0 if (pred_presence == gt_presence) else 0.0
        bbox_error = float(np.mean(np.abs(pred_bbox - gt_bbox)))
        spatial_err_km = float(np.sqrt((pred_bbox[0] - gt_bbox[0])**2 + (pred_bbox[1] - gt_bbox[1])**2) * patch_km)

        pred_bin = (mask_prob >= 0.5).astype(float)
        inter = (pred_bin * gt_mask).sum()
        union = pred_bin.sum() + gt_mask.sum()
        dice_score = float((2.0 * inter + 1e-6) / (union + 1e-6))

    # 5. Print Detailed Output
    print("\n" + "=" * 75)
    print("   CYCLONEVISION AI -- LIVE MODEL 1 PREDICTION REPORT")
    print("=" * 75)
    print(f"  Target Input           : {sample_desc}")
    print(f"  Patch Center Reference : {storm_lat:.2f}° N, {storm_lon:.2f}° E (Swath: {patch_km:.0f} km)")
    print("-" * 75)
    status_str = 'CYCLONE DETECTED [POSITIVE]' if pred_presence else 'NO CYCLONE DETECTED [NEGATIVE]'
    print(f"  [OUTPUT 1] Vortex Presence Probability : {presence_prob:.2%}")
    print(f"             Detection Status            : {status_str}")
    if gt_presence is not None:
        gt_str = 'CYCLONE PRESENT' if gt_presence else 'NO CYCLONE'
        print(f"             Ground Truth Presence       : {gt_str} (Accuracy: {sample_acc:.1f}%)")
    print("-" * 75)
    print(f"  [OUTPUT 2] Predicted Eye BBox [x,y,w,h]: [{pred_bbox[0]:.4f}, {pred_bbox[1]:.4f}, {pred_bbox[2]:.4f}, {pred_bbox[3]:.4f}]")
    print(f"             Pinpointed Eye Latitude     : {eye_lat:.3f}° N")
    print(f"             Pinpointed Eye Longitude    : {eye_lon:.3f}° E")
    print(f"             Estimated Eye Diameter      : {eye_dia_km:.1f} km")
    if gt_bbox is not None:
        gt_lat, gt_lon, gt_dia = convert_bbox_to_latlon(gt_bbox, center_lat=storm_lat, center_lon=storm_lon, patch_km=patch_km)
        print(f"             Ground Truth Eye Center     : {gt_lat:.3f}° N, {gt_lon:.3f}° E (Diameter: {gt_dia:.1f} km)")
        print(f"             Localization Error          : ~{spatial_err_km:.2f} km (BBox MAE: {bbox_error:.4f})")
    print("-" * 75)
    print(f"  [OUTPUT 3] Eyewall Cloud Ring Mask     : Probability Map Shape {list(mask_prob.shape)}")
    if dice_score is not None:
        print(f"             Eyewall Dice Agreement Score: {dice_score:.4f} ({dice_score * 100:.2f}% overlap)")
    print("=" * 75)

    # 6. Generate 4-Panel Visualization Plot
    img_np = img[0].cpu().numpy()
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="#0f172a")

    for ax in axes:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="white")

    # Panel 1: Infrared + Predicted Eye
    im0 = axes[0].imshow(img_np[0], cmap="inferno", vmin=0, vmax=1)
    xc, yc, w, h = pred_bbox
    rect_pred = patches.Rectangle(
        ((xc - w / 2) * 128, (yc - h / 2) * 128), w * 128, h * 128,
        linewidth=2.5, edgecolor="#00ffcc", facecolor="none",
        label=f"Pred Eye ({eye_lat:.2f}N, {eye_lon:.2f}E)"
    )
    axes[0].add_patch(rect_pred)
    axes[0].scatter(xc * 128, yc * 128, color="#00ffcc", s=40, marker="+", label="Pred Center")
    axes[0].set_title(f"Ch 0: Infrared (IR) + Pred Eye\n[Prob: {presence_prob:.1%}]", color="white", fontsize=11, fontweight="bold")
    axes[0].legend(loc="upper right", fontsize=8, facecolor="#1e293b", labelcolor="white")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Panel 2: Water Vapor + GT vs Pred
    im1 = axes[1].imshow(img_np[1], cmap="magma", vmin=0, vmax=1)
    if gt_bbox is not None:
        gxc, gyc, gw, gh = gt_bbox
        rect_gt = patches.Rectangle(
            ((gxc - gw / 2) * 128, (gyc - gh / 2) * 128), gw * 128, gh * 128,
            linewidth=2.0, edgecolor="#ff0055", linestyle="--", facecolor="none",
            label="GT Eye Center"
        )
        axes[1].add_patch(rect_gt)
        axes[1].scatter(gxc * 128, gyc * 128, color="#ff0055", s=40, marker="x", label="GT Center")

    rect_pred2 = patches.Rectangle(
        ((xc - w / 2) * 128, (yc - h / 2) * 128), w * 128, h * 128,
        linewidth=2.0, edgecolor="#00ffcc", facecolor="none", label="Pred Eye"
    )
    axes[1].add_patch(rect_pred2)
    title_p2 = f"Ch 1: Water Vapor (WV)\nSpatial Error: ~{spatial_err_km:.1f} km" if spatial_err_km is not None else "Ch 1: Water Vapor (WV)"
    axes[1].set_title(title_p2, color="white", fontsize=11, fontweight="bold")
    axes[1].legend(loc="upper right", fontsize=8, facecolor="#1e293b", labelcolor="white")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Eyewall Segmentation Probability Heatmap
    im2 = axes[2].imshow(mask_prob, cmap="viridis", vmin=0, vmax=1)
    axes[2].set_title(f"Output 3: Eyewall Mask Prob\n[Mean: {mask_prob.mean():.3f}]", color="white", fontsize=11, fontweight="bold")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # Panel 4: Eyewall Mask Overlap Comparison
    if gt_mask is not None:
        overlay = np.zeros((128, 128, 3))
        overlay[..., 0] = gt_mask * 0.8          # Red = Ground Truth
        overlay[..., 1] = (mask_prob >= 0.5) * 0.8 # Green = Predicted (Yellow = Overlap)
        axes[3].imshow(overlay)
        axes[3].set_title(f"Eyewall Overlap (Dice: {dice_score:.2%})\n[Red=GT, Green=Pred, Yellow=Match]", color="white", fontsize=11, fontweight="bold")
    else:
        axes[3].imshow(mask_prob >= 0.5, cmap="cool")
        axes[3].set_title("Predicted Eyewall Binary Mask", color="white", fontsize=11, fontweight="bold")

    plt.tight_layout()
    out_file = os.path.join(BASE_DIR, save_plot_path)
    plt.savefig(out_file, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n[+] Visual 4-panel dashboard exported to: {out_file}\n")

    return {
        "presence_prob": presence_prob,
        "pred_presence": pred_presence,
        "pred_bbox": pred_bbox,
        "eye_latitude": eye_lat,
        "eye_longitude": eye_lon,
        "eye_diameter_km": eye_dia_km,
        "mask_prob": mask_prob,
        "spatial_error_km": spatial_err_km,
        "dice_score": dice_score
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict with CycloneVision Model 1 .pth Checkpoint")
    parser.add_argument("--sample_idx", type=int, default=1120, help="Sample index to test (0 to 3,161)")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val", "train"], help="Dataset split")
    parser.add_argument("--lat", type=float, default=15.40, help="Patch center latitude (°N)")
    parser.add_argument("--lon", type=float, default=87.20, help="Patch center longitude (°E)")
    parser.add_argument("--swath_km", type=float, default=500.0, help="Patch swath in km")
    parser.add_argument("--checkpoint", type=str, default=None, help="Custom path to .pth file")
    args = parser.parse_args()

    predict_with_checkpoint(
        sample_idx=args.sample_idx,
        split=args.split,
        storm_lat=args.lat,
        storm_lon=args.lon,
        patch_km=args.swath_km,
        model_checkpoint=args.checkpoint
    )
