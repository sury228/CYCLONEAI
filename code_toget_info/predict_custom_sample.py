"""
CycloneVision AI - Standalone Interactive Prediction & Precision Inspector
===========================================================================
Allows you to provide custom parameters (sample index, coordinates, or custom tensor)
and performs live multi-task inference using VortexEyeLocalizationModel.
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Path configuration
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
for sub in ["", "data_fusion", "data_preprocess", "numeric_data", "models", "actual_data_split"]:
    p = os.path.join(BASE_DIR, sub) if sub else BASE_DIR
    if p not in sys.path:
        sys.path.append(p)

from vortex_identification_model import VortexEyeLocalizationModel, convert_bbox_to_latlon, get_device
from actual_data_split.split_manager import get_actual_data_loaders


def run_interactive_prediction(
    sample_idx=100,
    split="test",
    storm_lat=15.4,
    storm_lon=87.2,
    patch_km=500.0,
    save_plot_path="custom_prediction_result.png"
):
    device = get_device()
    print(f"[*] Compute Device: {str(device).upper()}")

    # 1. Load Model
    model = VortexEyeLocalizationModel(in_channels=4).to(device)
    model_path = os.path.join(BASE_DIR, "models", "best_model1_vortex_eye.pth")
    if not os.path.exists(model_path):
        print(f"[!] Model checkpoint not found at: {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()

    # 2. Load Data
    _, val_loader, test_loader = get_actual_data_loaders(batch_size=32, noise_std=0.07)
    loader_map = {"test": test_loader, "val": val_loader}
    target_loader = loader_map.get(split.lower(), test_loader)
    dataset = target_loader.dataset

    idx = max(0, min(sample_idx, len(dataset) - 1))
    img_raw, nwp, gt_target = dataset[idx]
    img = img_raw.unsqueeze(0).to(device)
    img = torch.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    # 3. Predict
    with torch.no_grad():
        out = model(img)

    presence_prob = float(out["vortex_presence_prob"][0].item())
    pred_presence = bool(presence_prob >= 0.5)
    pred_bbox = out["eye_bbox"][0].cpu().numpy()
    mask_prob = out["eyewall_mask_prob"][0, 0].cpu().numpy()

    eye_lat, eye_lon, eye_dia_km = convert_bbox_to_latlon(
        pred_bbox, center_lat=storm_lat, center_lon=storm_lon, patch_km=patch_km
    )

    # Ground Truth Comparison
    gt_presence = bool(gt_target["vortex_presence"].item() >= 0.5)
    gt_bbox = gt_target["eye_bbox"].numpy()
    gt_mask = gt_target["eyewall_mask"][0].numpy()

    sample_acc = 100.0 if (pred_presence == gt_presence) else 0.0
    bbox_mae = float(np.mean(np.abs(pred_bbox - gt_bbox)))
    spatial_err_km = float(np.sqrt((pred_bbox[0] - gt_bbox[0])**2 + (pred_bbox[1] - gt_bbox[1])**2) * patch_km)

    pred_bin = (mask_prob >= 0.5).astype(float)
    inter = (pred_bin * gt_mask).sum()
    union = pred_bin.sum() + gt_mask.sum()
    dice_score = float((2.0 * inter + 1e-6) / (union + 1e-6))

    # Print Report
    print("=" * 75)
    print("   CYCLONEVISION AI -- LIVE PREDICTION & PRECISION REPORT")
    print("=" * 75)
    print(f"  Target Sample          : {split.upper()} Dataset Sample #{idx:,} (of {len(dataset):,})")
    print(f"  Patch Reference Center : {storm_lat:.2f} deg N, {storm_lon:.2f} deg E (Swath: {patch_km:.0f} km)")
    print("-" * 75)
    print(f"  [OUTPUT 1] Vortex Presence Probability : {presence_prob:.2%}")
    print(f"             Detection Status            : {'CYCLONE DETECTED [POSITIVE]' if pred_presence else 'NO CYCLONE [NEGATIVE]'}")
    print(f"             Ground Truth Status         : {'CYCLONE PRESENT' if gt_presence else 'NO CYCLONE'} (Sample Acc: {sample_acc:.1f}%)")
    print("-" * 75)
    print(f"  [OUTPUT 2] Predicted Eye BBox [x,y,w,h]: [{pred_bbox[0]:.4f}, {pred_bbox[1]:.4f}, {pred_bbox[2]:.4f}, {pred_bbox[3]:.4f}]")
    print(f"             Pinpointed Eye Latitude     : {eye_lat:.3f} deg N")
    print(f"             Pinpointed Eye Longitude    : {eye_lon:.3f} deg E")
    print(f"             Estimated Eye Diameter      : {eye_dia_km:.1f} km")
    gt_lat, gt_lon, gt_dia = convert_bbox_to_latlon(gt_bbox, center_lat=storm_lat, center_lon=storm_lon, patch_km=patch_km)
    print(f"             Ground Truth Eye Center     : {gt_lat:.3f} deg N, {gt_lon:.3f} deg E (Dia: {gt_dia:.1f} km)")
    print(f"             Localization Error          : ~{spatial_err_km:.2f} km (BBox MAE: {bbox_mae:.4f})")
    print("-" * 75)
    print(f"  [OUTPUT 3] Eyewall Cloud Ring Mask     : Probability Map Shape {list(mask_prob.shape)}")
    print(f"             Eyewall Dice Agreement Score: {dice_score:.4f} ({dice_score * 100:.2f}% pixel overlap)")
    print("=" * 75)

    # Plot
    img_np = img[0].cpu().numpy()
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="#0f172a")

    for ax in axes:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="white")

    # Panel 1
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

    # Panel 2
    im1 = axes[1].imshow(img_np[1], cmap="magma", vmin=0, vmax=1)
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
    axes[1].set_title(f"Ch 1: Water Vapor (WV)\nSpatial Error: ~{spatial_err_km:.1f} km", color="white", fontsize=11, fontweight="bold")
    axes[1].legend(loc="upper right", fontsize=8, facecolor="#1e293b", labelcolor="white")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3
    im2 = axes[2].imshow(mask_prob, cmap="viridis", vmin=0, vmax=1)
    axes[2].set_title(f"Output 3: Eyewall Mask Prob\n[Mean: {mask_prob.mean():.3f}]", color="white", fontsize=11, fontweight="bold")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # Panel 4
    overlay = np.zeros((128, 128, 3))
    overlay[..., 0] = gt_mask * 0.8
    overlay[..., 1] = (mask_prob >= 0.5) * 0.8
    axes[3].imshow(overlay)
    axes[3].set_title(f"Eyewall Overlap (Dice: {dice_score:.2%})\n[Red=GT, Green=Pred, Yellow=Match]", color="white", fontsize=11, fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(BASE_DIR, save_plot_path)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n[+] Visual 4-panel dashboard saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Prediction Inspector for CycloneVision Model 1")
    parser.add_argument("--sample_idx", type=int, default=100, help="Index of sample to test (0 to 3161)")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val", "train"], help="Dataset split")
    parser.add_argument("--lat", type=float, default=15.4, help="Patch center latitude (deg N)")
    parser.add_argument("--lon", type=float, default=87.2, help="Patch center longitude (deg E)")
    parser.add_argument("--swath_km", type=float, default=500.0, help="Patch width/height in km")
    args = parser.parse_args()

    run_interactive_prediction(
        sample_idx=args.sample_idx,
        split=args.split,
        storm_lat=args.lat,
        storm_lon=args.lon,
        patch_km=args.swath_km
    )
