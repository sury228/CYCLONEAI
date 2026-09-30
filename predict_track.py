"""
CycloneVision AI - Standalone Track & Trajectory Prediction Engine (Model 2)
=============================================================================
Directly predicts multi-horizon cyclone tracks (+6h to +48h) and uncertainty cones
from either:
  1. Any unseen dataset sample index (--sample_idx 25 --split test)
  2. Custom user-supplied past storm kinematic fixes (--lats 12.0 12.8 13.5 14.3 --lons 85.0 85.4 85.9 86.4)

Outputs:
  - Table of predicted Future Coordinates (Latitude, Longitude) at each horizon
  - Forecast Cone of Uncertainty radii (+-km)
  - Estimated forward translation speed & heading
  - Landfall probability within 48h
  - High-resolution Synoptic Trajectory Map saved to `track_prediction_output.png`

Usage:
  python predict_track.py --sample_idx 25 --split test
  python predict_track.py --lats 13.5 14.2 14.9 15.6 --lons 86.5 86.2 85.8 85.3
"""

import os
import sys
import argparse
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

# Configure sys.path robustly from project root
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Import Model 2 components
try:
    from models.track_and_trajectory_model_2.track_and_trajectory_model import (
        CycloneTrajectoryForecaster, CycloneTrajectorySequenceDataset,
        haversine_distance_km, decompose_track_errors, get_device
    )
except ImportError:
    p_m2 = os.path.join(BASE_DIR, "models", "track_and_trajectory_model_2")
    if p_m2 not in sys.path:
        sys.path.append(p_m2)
    from track_and_trajectory_model import (
        CycloneTrajectoryForecaster, CycloneTrajectorySequenceDataset,
        haversine_distance_km, decompose_track_errors, get_device
    )


HORIZON_NAMES = ["+6h", "+12h", "+18h", "+24h", "+36h", "+48h"]


def build_kinematics_from_fixes(lats, lons, pressures=None, winds=None):
    """
    Builds (1, 4, 8) kinematic input tensor from 4 sequential storm fixes (t-9h, t-6h, t-3h, t0).
    """
    assert len(lats) == 4 and len(lons) == 4, "Must provide exactly 4 sequential past coordinate fixes."
    lats = np.array(lats, dtype=np.float32)
    lons = np.array(lons, dtype=np.float32)

    if pressures is None:
        pressures = np.array([1000.0, 995.0, 990.0, 982.0], dtype=np.float32)
    else:
        pressures = np.array(pressures, dtype=np.float32)

    if winds is None:
        winds = np.array([35.0, 45.0, 55.0, 65.0], dtype=np.float32)
    else:
        winds = np.array(winds, dtype=np.float32)

    dlat = np.diff(lats, prepend=lats[0])
    dlon = np.diff(lons, prepend=lons[0])
    speed = np.sqrt((dlat * 111.0)**2 + (dlon * 111.0 * np.cos(np.radians(lats)))**2) / 6.0
    heading = np.arctan2(dlon, dlat)

    kin = np.stack([lats, lons, dlat, dlon, speed, heading, pressures, winds], axis=-1)
    return torch.tensor(kin, dtype=torch.float32).unsqueeze(0)


def predict_trajectory_sample(
    sample_idx=25,
    split="test",
    custom_lats=None,
    custom_lons=None,
    custom_pressures=None,
    custom_winds=None,
    custom_nwp=None,
    model_checkpoint=None,
    save_plot_path=None
):
    if save_plot_path is None:
        save_plot_path = os.path.join(BASE_DIR, "models", "track_and_trajectory_model_2", "track_prediction_output.png")
    device = get_device()
    print(f"[*] Compute Device: {str(device).upper()}")

    # 1. Locate and load Model 2 checkpoint
    if model_checkpoint is None:
        possible_paths = [
            os.path.join(BASE_DIR, "models", "track_and_trajectory_model_2", "best_model2_track_trajectory.pth"),
            os.path.join(BASE_DIR, "best_model2_track_trajectory.pth"),
            os.path.join(BASE_DIR, "models", "best_model2_track_trajectory.pth")
        ]
        model_checkpoint = next((p for p in possible_paths if os.path.exists(p)), None)

    model = CycloneTrajectoryForecaster(
        kinematic_dim=8,
        nwp_dim=5,
        visual_dim=64,
        hidden_dim=128,
        num_horizons=6
    ).to(device)

    if model_checkpoint and os.path.exists(model_checkpoint):
        print(f"[+] Loading trained Model 2 weights from: {model_checkpoint}")
        model.load_state_dict(torch.load(model_checkpoint, map_location=device, weights_only=False))
    else:
        print("[!] Note: Model 2 checkpoint not found on disk yet. Running with architectural initialization.")

    model.eval()

    # 2. Prepare Input Data
    true_future = None
    if custom_lats is not None and custom_lons is not None:
        past_kin = build_kinematics_from_fixes(custom_lats, custom_lons, custom_pressures, custom_winds)
        mean_lat = float(np.mean(custom_lats))
        if custom_nwp is not None:
            nwp_vec = torch.tensor(custom_nwp, dtype=torch.float32).unsqueeze(0)
        else:
            nwp_vec = torch.tensor([[-3.5 + 0.1 * mean_lat, 4.2 - 0.08 * mean_lat, 985.0, 29.2, 12.0]], dtype=torch.float32)
        sample_title = f"Custom Storm Track Fixes [Current Eye: {custom_lats[-1]:.2f}°N, {custom_lons[-1]:.2f}°E]"
    else:
        ibtracs_csv = os.path.join(BASE_DIR, "numeric_data", "ibtracs.NI.list.v04r01.csv")
        dataset = CycloneTrajectorySequenceDataset(ibtracs_csv=ibtracs_csv, split_type=split)
        idx = max(0, min(sample_idx, len(dataset) - 1))
        past_kin_item, nwp_item, true_future_item, landfall_gt = dataset[idx]

        past_kin = past_kin_item.unsqueeze(0)
        nwp_vec = nwp_item.unsqueeze(0)
        true_future = true_future_item.numpy()
        sample_title = f"{split.upper()} Dataset Track Sample #{idx:,} (of {len(dataset):,})"

    past_kin = past_kin.to(device)
    nwp_vec = nwp_vec.to(device)

    # 3. Model Inference
    with torch.no_grad():
        out = model(past_kin, nwp_vec)

    pred_future = out["future_coords"][0].cpu().numpy()          # (6, 2) [Lat, Lon]
    radii_km = out["uncertainty_radii"][0].cpu().numpy()            # (6, 2) [sigma_lat_km, sigma_lon_km]
    landfall_prob = float(out["landfall_prob"][0].item())
    past_coords = past_kin[0, :, :2].cpu().numpy()

    # 4. Print Formatted Meteorologist Forecast Table
    print("\n" + "=" * 80)
    print("   CYCLONEVISION AI -- LIVE MULTI-HORIZON TRACK FORECAST REPORT (MODEL 2)")
    print("=" * 80)
    print(f"  Target Cyclone Track  : {sample_title}")
    print(f"  Current Synoptic Fix  : {past_coords[-1, 0]:.3f}° N, {past_coords[-1, 1]:.3f}° E (t = 0h)")
    print(f"  48h Coastal Landfall  : {landfall_prob:.1%} probability ({'HIGH RISK' if landfall_prob >= 0.5 else 'LOW/MODERATE RISK'})")
    print("-" * 80)
    print(f"  {'Horizon':<8} | {'Pred Lat':<10} | {'Pred Lon':<10} | {'Cone Radius':<12} | {'True Lat':<10} | {'True Lon':<10} | {'Error (km)'}")
    print("-" * 80)

    total_err = 0.0
    for h_idx, h_name in enumerate(HORIZON_NAMES):
        p_lat = pred_future[h_idx, 0]
        p_lon = pred_future[h_idx, 1]
        mean_r = float(np.mean(radii_km[h_idx]))

        if true_future is not None:
            t_lat = true_future[h_idx, 0]
            t_lon = true_future[h_idx, 1]
            dist = haversine_distance_km(p_lat, p_lon, t_lat, t_lon)
            total_err += dist
            print(f"  {h_name:<8} | {p_lat:>7.3f}° N | {p_lon:>7.3f}° E | ±{mean_r:>5.1f} km     | {t_lat:>7.3f}° N | {t_lon:>7.3f}° E | {dist:>6.1f} km")
        else:
            print(f"  {h_name:<8} | {p_lat:>7.3f}° N | {p_lon:>7.3f}° E | ±{mean_r:>5.1f} km     | {'--':<10} | {'--':<10} | {'--'}")

    print("-" * 80)
    if true_future is not None:
        mean_tfe = total_err / len(HORIZON_NAMES)
        print(f"  Mean Track Forecast Error (TFE) across +6h to +48h: {mean_tfe:.2f} km")
    print("=" * 80)

    # 5. Generate Synoptic Trajectory Map
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#0f172a")
    ax.set_facecolor("#1e293b")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#475569")

    ax.set_xlim(75.0, 95.0)
    ax.set_ylim(8.0, 24.0)
    ax.set_xlabel("Longitude (°E)", color="white", fontsize=11, fontweight="bold")
    ax.set_ylabel("Latitude (°N)", color="white", fontsize=11, fontweight="bold")
    ax.set_title(f"CycloneVision AI -- 48-Hour Track Forecast & Cone of Uncertainty\n{sample_title}",
                 color="white", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.3, color="gray")

    # Coastline reference
    coast_lons = [80.2, 80.3, 81.0, 82.2, 83.3, 84.8, 86.8, 88.0, 89.5, 91.5]
    coast_lats = [13.0, 14.5, 15.8, 16.9, 17.7, 19.3, 20.8, 21.6, 22.0, 22.2]
    ax.plot(coast_lons, coast_lats, color="#64748b", linewidth=2.5, linestyle="-", label="East India / Odisha Coastline")

    # Past fixes
    ax.plot(past_coords[:, 1], past_coords[:, 0], color="#38bdf8", marker="o", linewidth=2.5,
            markersize=6, label="Past Track (t-9h -> t0)")
    ax.scatter(past_coords[-1, 1], past_coords[-1, 0], color="#facc15", s=120, marker="*",
               zorder=5, label="Current Eye Fix (t=0h)")

    # Ground Truth
    if true_future is not None:
        ax.plot(true_future[:, 1], true_future[:, 0], color="#4ade80", marker="s",
                linewidth=2.0, linestyle="--", label="Ground Truth Track")

    # Predicted track
    full_pred_lons = np.concatenate([[past_coords[-1, 1]], pred_future[:, 1]])
    full_pred_lats = np.concatenate([[past_coords[-1, 0]], pred_future[:, 0]])
    ax.plot(full_pred_lons, full_pred_lats, color="#f43f5e", marker="^",
            linewidth=2.5, label="Predicted Trajectory (+6h -> +48h)")

    # Uncertainty Cone
    for i, (plat, plon) in enumerate(pred_future):
        sigma_lat_deg = radii_km[i, 0] / 111.0
        sigma_lon_deg = radii_km[i, 1] / (111.0 * max(0.1, np.cos(np.radians(plat))))
        ell = patches.Ellipse(
            (plon, plat),
            width=sigma_lon_deg * 2.0,
            height=sigma_lat_deg * 2.0,
            angle=0,
            color="#f43f5e",
            alpha=0.15 + 0.03 * i
        )
        ax.add_patch(ell)
        ax.text(plon + 0.2, plat - 0.2, HORIZON_NAMES[i], color="#fca5a5", fontsize=8, fontweight="bold")

    ax.legend(loc="lower right", facecolor="#1e293b", edgecolor="#475569", labelcolor="white", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_plot_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    print(f"\n[+] Synoptic track forecast plot saved to: {os.path.abspath(save_plot_path)}")
    return {
        "pred_future": pred_future,
        "radii_km": radii_km,
        "landfall_prob": landfall_prob,
        "past_coords": past_coords
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CycloneVision Model 2 Track & Trajectory Predictor")
    parser.add_argument("--sample_idx", type=int, default=25, help="Dataset track sample index")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val", "train"], help="Dataset split")
    parser.add_argument("--lats", type=float, nargs=4, default=None, help="4 past latitudes (e.g. 13.0 13.8 14.5 15.2)")
    parser.add_argument("--lons", type=float, nargs=4, default=None, help="4 past longitudes (e.g. 87.0 86.6 86.1 85.5)")
    parser.add_argument("--pressures", type=float, nargs=4, default=None, help="4 past central pressures (hPa)")
    parser.add_argument("--winds", type=float, nargs=4, default=None, help="4 past sustained wind speeds (kts)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pth checkpoint")
    args = parser.parse_args()

    predict_trajectory_sample(
        sample_idx=args.sample_idx,
        split=args.split,
        custom_lats=args.lats,
        custom_lons=args.lons,
        custom_pressures=args.pressures,
        custom_winds=args.winds,
        model_checkpoint=args.checkpoint
    )
