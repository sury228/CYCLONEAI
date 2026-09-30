"""
CycloneVision AI - Add Model 2 (Track & Trajectory Forecasting) to cyclone_pipleline.ipynb
Appends Steps 7 to 12 for Model 2 matching the structure and beauty of Model 1.
"""

import os
import json

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NOTEBOOK_PATH = os.path.join(BASE_DIR, "cyclone_pipleline.ipynb")

with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Ensure no old Model 2 / Step 7+ remnants exist
cleaned_cells = []
for cell in nb["cells"]:
    src = "".join(cell.get("source", [])) if isinstance(cell.get("source", []), list) else str(cell.get("source", ""))
    if "Model 2" in src or "CycloneTrajectoryForecaster" in src or "Step 7" in src or "Step 8" in src or "Step 9" in src or "Step 10" in src or "Step 11" in src or "Step 12" in src:
        continue
    cleaned_cells.append(cell)

# Define new cells for Model 2
new_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "# Part 2: Model 2 -- Cyclone Track & Trajectory Forecasting Engine\n",
            "**Forecast Horizons:** +6h, +12h, +18h, +24h, +36h, +48h Lead Times  \n",
            "**Coordinate Output:** Latitude (°N) & Longitude (°E) + 2D Gaussian Spatial Dispersion Radii (±km)  \n",
            "**Steering Integration:** Deep Multi-Modal Cross-Attention with ERA5 Environmental NWP Fields (850–200 hPa Deep-Layer Wind Shear, MSLP, SST)\n",
            "\n",
            "### Architecture Flow:\n",
            "```\n",
            "Past Cyclone States (B, 4 timesteps, 6 kinematic vars) [Lat, Lon, MSW, CP, U_trans, V_trans]\n",
            "    │\n",
            "    ├── Dilated Temporal ConvNet (TCN) [Residual 1D Convolutions with Dilation Rates d=1,2,4]\n",
            "    │        │\n",
            "    │        └── Spatio-Temporal Storm Dynamics Representation (128-d)\n",
            "    │\n",
            "Environmental NWP Grid (B, 5 variables) [SST, MSLP, Deep Shear, U10, V10]\n",
            "    │\n",
            "    ├── NWP Projection MLP (64-d)\n",
            "    │\n",
            "    └── Environmental Cross-Attention (Query = Storm Dynamics, Key/Value = Steering Grid)\n",
            "             │\n",
            "             ├── Cumulative Coordinate Decoder ──> Multi-Horizon Track [Lat(t), Lon(t)]\n",
            "             │     (Haversine Great-Circle Geodesic Loss in km)\n",
            "             └── Uncertainty Head ───────────────> Forecast Cone Radii [R_6h, ..., R_48h] (km)\n",
            "```"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 7: Model 2 Architecture Definition -- CycloneTrajectoryForecaster\n",
            "\n",
            "Loads the dedicated `CycloneTrajectoryForecaster` with Dilated TCN, Environmental Cross-Attention, and Cumulative Multi-Horizon Coordinate Decoder."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import importlib\n",
            "import models.track_and_trajectory_model_2.track_and_trajectory_model as model2_module\n",
            "importlib.reload(model2_module)\n",
            "\n",
            "from models.track_and_trajectory_model_2.track_and_trajectory_model import (\n",
            "    CycloneTrajectoryForecaster,\n",
            "    haversine_distance_km,\n",
            "    torch_haversine_loss,\n",
            "    decompose_track_errors,\n",
            "    TrajectoryLoss,\n",
            "    CycloneTrajectoryDataset,\n",
            "    get_device\n",
            ")\n",
            "\n",
            "device = get_device()\n",
            "print(f\"[*] Model 2 Compute Device : {str(device).upper()}\")\n",
            "\n",
            "# Instantiate Model 2\n",
            "model2 = CycloneTrajectoryForecaster(\n",
            "    kinematic_in_dim=6,\n",
            "    nwp_in_dim=5,\n",
            "    hidden_dim=128,\n",
            "    num_horizons=6\n",
            ").to(device)\n",
            "\n",
            "total_params2 = sum(p.numel() for p in model2.parameters())\n",
            "print(f\"[+] Model 2 Architecture   : CycloneTrajectoryForecaster (Dilated TCN + Cross-Attention)\")\n",
            "print(f\"[+] Total Parameters     : {total_params2:,}\")\n",
            "\n",
            "# Forward verification with dummy tensors\n",
            "dummy_kin = torch.randn(2, 4, 6).to(device)   # (B, 4 past steps @ 3h intervals, 6 features)\n",
            "dummy_nwp = torch.randn(2, 5).to(device)      # (B, 5 NWP environmental steering features)\n",
            "\n",
            "with torch.no_grad():\n",
            "    out2 = model2(dummy_kin, dummy_nwp)\n",
            "\n",
            "print(f\"[+] Output Head 1 - Predicted Future Coordinates : {list(out2['predicted_coords'].shape)} (B x 6 horizons x [Lat, Lon])\")\n",
            "print(f\"[+] Output Head 2 - Uncertainty Cone Radii (km) : {list(out2['uncertainty_radii'].shape)} (B x 6 horizons)\")\n",
            "print(\"[SUCCESS] Model 2 Architecture verified OK!\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 8: Model 2 Sequence Dataset & Data Loaders\n",
            "\n",
            "Extracts sequential cyclone trajectories from the IBTrACS North Indian Ocean catalog and matches them with ERA5 environmental grids:\n",
            "- **History Window**: 12 hours (4 synoptic fixes at 3-hour intervals: $t_{-9h}, t_{-6h}, t_{-3h}, t_0$)\n",
            "- **Forecast Horizon**: 48 hours (6 lead-time milestones: $+6h, +12h, +18h, +24h, +36h, +48h$)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from torch.utils.data import DataLoader\n",
            "\n",
            "BATCH_SIZE_M2 = 32\n",
            "\n",
            "train_ds_m2 = CycloneTrajectoryDataset(split=\"train\", seq_len=4, num_horizons=6)\n",
            "val_ds_m2   = CycloneTrajectoryDataset(split=\"val\",   seq_len=4, num_horizons=6)\n",
            "test_ds_m2  = CycloneTrajectoryDataset(split=\"test\",  seq_len=4, num_horizons=6)\n",
            "\n",
            "train_loader_m2 = DataLoader(train_ds_m2, batch_size=BATCH_SIZE_M2, shuffle=True)\n",
            "val_loader_m2   = DataLoader(val_ds_m2,   batch_size=BATCH_SIZE_M2, shuffle=False)\n",
            "test_loader_m2  = DataLoader(test_ds_m2,  batch_size=BATCH_SIZE_M2, shuffle=False)\n",
            "\n",
            "print(f\"[+] Model 2 Dataset Partitions (IBTrACS NIO Sequential Tracks):\")\n",
            "print(f\"    Train Trajectory Sequences : {len(train_ds_m2):,}\")\n",
            "print(f\"    Val Trajectory Sequences   : {len(val_ds_m2):,}\")\n",
            "print(f\"    Test Trajectory Sequences  : {len(test_ds_m2):,}\")\n",
            "\n",
            "p_kin, p_nwp, f_tgt, f_disp = next(iter(train_loader_m2))\n",
            "print(\"\\n[+] Batch Input/Target Tensor Profiles:\")\n",
            "print(f\"    Past Kinematics (History) : {list(p_kin.shape)}  [Lat, Lon, MSW, CP, U_spd, V_spd]\")\n",
            "print(f\"    Environmental NWP Grids   : {list(p_nwp.shape)}  [SST, MSLP, Shear, U10, V10]\")\n",
            "print(f\"    Target Future Coordinates : {list(f_tgt.shape)}  [Lat, Lon] at +6h,+12h,+18h,+24h,+36h,+48h\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 9: Model 2 Training & Haversine Great-Circle Loss\n",
            "\n",
            "Trains the trajectory model using:\n",
            "1. **Great-Circle Haversine Loss**: Directly penalizes geographic distance error in physical kilometers on spherical Earth ($R=6,371\\text{ km}$).\n",
            "2. **Cumulative Displacement Loss**: Penalizes incremental step vectors to prevent path drift.\n",
            "3. **Uncertainty Radius Calibration Loss**: Trains the Gaussian cone spread head to provide calibrated $\\pm\\text{km}$ error bounds."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "criterion_traj = TrajectoryLoss(alpha_disp=0.3, beta_uncert=0.1)\n",
            "optimizer_m2   = optim.AdamW(model2.parameters(), lr=1e-3, weight_decay=1e-4)\n",
            "scheduler_m2   = optim.lr_scheduler.CosineAnnealingLR(optimizer_m2, T_max=15, eta_min=1e-5)\n",
            "\n",
            "EPOCHS_M2 = 15\n",
            "save_path_m2 = os.path.join(BASE_DIR, \"models\", \"track_and_trajectory_model_2\", \"best_model2_track_trajectory.pth\")\n",
            "os.makedirs(os.path.dirname(save_path_m2), exist_ok=True)\n",
            "\n",
            "best_val_dist = float(\"inf\")\n",
            "history_m2 = []\n",
            "\n",
            "print(f\"[*] Starting Model 2 Training ({EPOCHS_M2} Epochs, Cosine Annealing LR)...\")\n",
            "print(f\"    Save Destination: {save_path_m2}\")\n",
            "print(\"=\" * 80)\n",
            "\n",
            "for ep in range(1, EPOCHS_M2 + 1):\n",
            "    # ---- TRAIN ----\n",
            "    model2.train()\n",
            "    tr_loss = tr_dist = tr_samples = 0\n",
            "\n",
            "    for kin, nwp, tgt_coords, tgt_disp in train_loader_m2:\n",
            "        kin        = kin.to(device)\n",
            "        nwp        = nwp.to(device)\n",
            "        tgt_coords = tgt_coords.to(device)\n",
            "        tgt_disp   = tgt_disp.to(device)\n",
            "        bsz        = kin.size(0)\n",
            "\n",
            "        optimizer_m2.zero_grad()\n",
            "        out = model2(kin, nwp)\n",
            "        loss, dist_km = criterion_traj(out, tgt_coords, tgt_disp)\n",
            "\n",
            "        loss.backward()\n",
            "        torch.nn.utils.clip_grad_norm_(model2.parameters(), max_norm=5.0)\n",
            "        optimizer_m2.step()\n",
            "\n",
            "        tr_loss    += loss.item() * bsz\n",
            "        tr_dist    += dist_km * bsz\n",
            "        tr_samples += bsz\n",
            "\n",
            "    scheduler_m2.step()\n",
            "    avg_tr_loss = tr_loss / max(tr_samples, 1)\n",
            "    avg_tr_dist = tr_dist / max(tr_samples, 1)\n",
            "\n",
            "    # ---- VALIDATION ----\n",
            "    model2.eval()\n",
            "    val_loss = val_dist = val_samples = 0\n",
            "    with torch.no_grad():\n",
            "        for kin, nwp, tgt_coords, tgt_disp in val_loader_m2:\n",
            "            kin        = kin.to(device)\n",
            "            nwp        = nwp.to(device)\n",
            "            tgt_coords = tgt_coords.to(device)\n",
            "            tgt_disp   = tgt_disp.to(device)\n",
            "            bsz        = kin.size(0)\n",
            "\n",
            "            out = model2(kin, nwp)\n",
            "            loss, dist_km = criterion_traj(out, tgt_coords, tgt_disp)\n",
            "\n",
            "            val_loss    += loss.item() * bsz\n",
            "            val_dist    += dist_km * bsz\n",
            "            val_samples += bsz\n",
            "\n",
            "    avg_val_loss = val_loss / max(val_samples, 1)\n",
            "    avg_val_dist = val_dist / max(val_samples, 1)\n",
            "\n",
            "    history_m2.append({\n",
            "        \"epoch\": ep,\n",
            "        \"train_loss\": avg_tr_loss,\n",
            "        \"train_dist_km\": avg_tr_dist,\n",
            "        \"val_loss\": avg_val_loss,\n",
            "        \"val_dist_km\": avg_val_dist,\n",
            "        \"lr\": optimizer_m2.param_groups[0]['lr']\n",
            "    })\n",
            "\n",
            "    saved_str = \"\"\n",
            "    if avg_val_dist < best_val_dist:\n",
            "        best_val_dist = avg_val_dist\n",
            "        torch.save(model2.state_dict(), save_path_m2)\n",
            "        saved_str = f\" [BEST SAVED -> {avg_val_dist:.2f} km]\"\n",
            "\n",
            "    print(f\"  Epoch [{ep:02d}/{EPOCHS_M2}] | Train Dist: {avg_tr_dist:.2f} km | Val Dist: {avg_val_dist:.2f} km | LR: {optimizer_m2.param_groups[0]['lr']:.6f}{saved_str}\")\n",
            "\n",
            "print(\"=\" * 80)\n",
            "print(f\"[+] Model 2 Training Complete! Best Validation Mean Track Error: {best_val_dist:.2f} km\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 10: Model 2 Multi-Horizon Track Evaluation & IMD Benchmark Comparison\n",
            "\n",
            "Evaluates Model 2 on unseen test tracks across all 6 forecast horizons (+6h to +48h), computing:\n",
            "- **Track Forecast Error (TFE in km)**: Total Great-Circle position discrepancy.\n",
            "- **Along-Track Error (ATE in km)**: Longitudinal timing/translation speed error along trajectory.\n",
            "- **Cross-Track Error (CTE in km)**: Lateral steering/curvature deviation perpendicular to trajectory.\n",
            "- **IMD Official Benchmark Comparison**: IMD 5-year average operational error targets for the North Indian Ocean."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from models.track_and_trajectory_model_2.track_and_trajectory_model import evaluate_trajectory_model\n",
            "\n",
            "metrics_df_m2, sample_vis_data = evaluate_trajectory_model(\n",
            "    model=model2,\n",
            "    test_loader=test_loader_m2,\n",
            "    device=device,\n",
            "    save_dir=os.path.join(BASE_DIR, \"models\", \"track_and_trajectory_model_2\")\n",
            ")\n",
            "\n",
            "print(\"\\n[+] Model 2 Test Evaluation Results:\")\n",
            "display(metrics_df_m2)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 11: Synoptic Forecast Map & 48h Trajectory Cone of Uncertainty\n",
            "\n",
            "Visualizes the multi-horizon trajectory forecast on a meteorological map of the North Indian Ocean (Bay of Bengal & Arabian Sea):\n",
            "- **Observed Past Track**: Blue line with synoptic eye fixes ($t_{-9h} \\rightarrow t_0$)\n",
            "- **Predicted Trajectory**: Red dashed trajectory vector with confidence waypoints\n",
            "- **Ground Truth Track**: Green trajectory for comparison\n",
            "- **2D Uncertainty Cone**: Shaded probability dispersion envelope showing growing positional spread over lead time"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from models.track_and_trajectory_model_2.track_and_trajectory_model import plot_meteorological_track_forecast\n",
            "\n",
            "fig_m2 = plot_meteorological_track_forecast(\n",
            "    vis_data=sample_vis_data,\n",
            "    save_path=os.path.join(BASE_DIR, \"models\", \"track_and_trajectory_model_2\", \"model2_synoptic_track_forecast.png\")\n",
            ")\n",
            "plt.show()"
        ]
    }
]

nb["cells"] = cleaned_cells + new_cells

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"[+] cyclone_pipleline.ipynb successfully updated!")
print(f"[+] Total notebook cells: {len(nb['cells'])} (Model 1: Steps 1-6 | Model 2: Steps 7-11)")
