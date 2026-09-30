"""
CycloneVision AI - Add Step 12 to cyclone_pipleline.ipynb
Adds interactive custom trajectory prediction on new/user test data.
"""

import os
import json

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NOTEBOOK_PATH = os.path.join(BASE_DIR, "cyclone_pipleline.ipynb")

with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Ensure no duplicate Step 12
cleaned_cells = []
for cell in nb["cells"]:
    src = "".join(cell.get("source", [])) if isinstance(cell.get("source", []), list) else str(cell.get("source", ""))
    if "Step 12" in src:
        continue
    cleaned_cells.append(cell)

step12_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 12: Predict on New / Custom Cyclone Data (Multi-Horizon Trajectory)\n",
            "\n",
            "This step enables direct inference on **new/custom test data** without re-training:\n",
            "1. **Custom Input**: Provide 4 sequential observed storm fixes (Latitude, Longitude, Pressure, Wind Speed) at 3-hour intervals ($t_{-9h}, t_{-6h}, t_{-3h}, t_0$).\n",
            "2. **Dataset Testing**: Or specify any sample index from the unseen test split.\n",
            "3. **Outputs**: Predicts future eye coordinates for +6h, +12h, +18h, +24h, +36h, +48h, uncertainty cone radii ($\\pm\\text{km}$), 48h coastal landfall risk probability, and displays a synoptic forecast map."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import matplotlib.patches as patches\n",
            "from models.track_and_trajectory_model_2.track_and_trajectory_model import haversine_distance_km\n",
            "\n",
            "HORIZONS = [\"+6h\", \"+12h\", \"+18h\", \"+24h\", \"+36h\", \"+48h\"]\n",
            "\n",
            "def predict_cyclone_trajectory(\n",
            "    custom_lats=None,\n",
            "    custom_lons=None,\n",
            "    custom_pressures=None,\n",
            "    custom_winds=None,\n",
            "    sample_idx=None,\n",
            "    split=\"test\"\n",
            "):\n",
            "    \"\"\"\n",
            "    Predicts 48-hour trajectory and uncertainty cone for custom storm coordinates or test split samples.\n",
            "    \"\"\"\n",
            "    model2.eval()\n",
            "    true_future = None\n",
            "\n",
            "    if custom_lats is not None and custom_lons is not None:\n",
            "        assert len(custom_lats) == 4 and len(custom_lons) == 4, \"Must provide exactly 4 past fixes (t-9h, t-6h, t-3h, t0).\"\n",
            "        lats = np.array(custom_lats, dtype=np.float32)\n",
            "        lons = np.array(custom_lons, dtype=np.float32)\n",
            "        pres = np.array(custom_pressures if custom_pressures else [1000.0, 995.0, 988.0, 980.0], dtype=np.float32)\n",
            "        wind = np.array(custom_winds if custom_winds else [35.0, 45.0, 60.0, 75.0], dtype=np.float32)\n",
            "\n",
            "        dlat = np.diff(lats, prepend=lats[0])\n",
            "        dlon = np.diff(lons, prepend=lons[0])\n",
            "        speed = np.sqrt((dlat * 111.0)**2 + (dlon * 111.0 * np.cos(np.radians(lats)))**2) / 6.0\n",
            "        heading = np.arctan2(dlon, dlat)\n",
            "\n",
            "        past_kin = np.stack([lats, lons, dlat, dlon, speed, heading, pres, wind], axis=-1)\n",
            "        past_tensor = torch.tensor(past_kin, dtype=torch.float32).unsqueeze(0).to(device)\n",
            "\n",
            "        mean_lat = float(np.mean(lats))\n",
            "        nwp_vec = torch.tensor([[-3.5 + 0.1 * mean_lat, 4.2 - 0.08 * mean_lat, float(pres[-1]), 29.5, 12.0]], dtype=torch.float32).to(device)\n",
            "        title_str = f\"Custom Input Storm Fixes [Current Eye: {lats[-1]:.2f}°N, {lons[-1]:.2f}°E]\"\n",
            "        past_coords = past_kin[:, :2]\n",
            "    else:\n",
            "        ds = CycloneTrajectoryDataset(split=split)\n",
            "        idx = max(0, min(sample_idx if sample_idx is not None else 25, len(ds) - 1))\n",
            "        p_kin, n_vec, f_tgt, l_gt = ds[idx]\n",
            "\n",
            "        past_tensor = p_kin.unsqueeze(0).to(device)\n",
            "        nwp_vec = n_vec.unsqueeze(0).to(device)\n",
            "        true_future = f_tgt.numpy()\n",
            "        past_coords = p_kin[:, :2].numpy()\n",
            "        title_str = f\"{split.upper()} Dataset Track Sample #{idx:,}\"\n",
            "\n",
            "    with torch.no_grad():\n",
            "        out = model2(past_tensor, nwp_vec)\n",
            "\n",
            "    pred_coords = out[\"predicted_coords\"][0].cpu().numpy() if \"predicted_coords\" in out else out[\"future_coords\"][0].cpu().numpy()\n",
            "    radii_km    = out[\"uncertainty_radii\"][0].cpu().numpy()\n",
            "    landfall_p  = float(out[\"landfall_prob\"][0].item()) if \"landfall_prob\" in out else 0.0\n",
            "\n",
            "    # Print Meteorologist Forecast Table\n",
            "    print(\"=\" * 85)\n",
            "    print(f\"   CYCLONEVISION AI -- LIVE MULTI-HORIZON TRACK FORECAST REPORT\")\n",
            "    print(\"=\" * 85)\n",
            "    print(f\"  Target Input           : {title_str}\")\n",
            "    print(f\"  Current Synoptic Fix   : {past_coords[-1, 0]:.3f}° N, {past_coords[-1, 1]:.3f}° E (t = 0h)\")\n",
            "    print(f\"  48h Coastal Landfall   : {landfall_p:.1%} probability\")\n",
            "    print(\"-\" * 85)\n",
            "    print(f\"  {'Horizon':<8} | {'Pred Lat':<10} | {'Pred Lon':<10} | {'Cone Radius':<12} | {'True Lat':<10} | {'True Lon':<10} | {'Error (km)'}\")\n",
            "    print(\"-\" * 85)\n",
            "\n",
            "    tot_err = 0.0\n",
            "    for h_i, h_name in enumerate(HORIZONS):\n",
            "        plat, plon = pred_coords[h_i, 0], pred_coords[h_i, 1]\n",
            "        mean_r = float(np.mean(radii_km[h_i])) if radii_km.ndim > 1 else float(radii_km[h_i])\n",
            "        if true_future is not None:\n",
            "            tlat, tlon = true_future[h_i, 0], true_future[h_i, 1]\n",
            "            err = haversine_distance_km(plat, plon, tlat, tlon)\n",
            "            tot_err += err\n",
            "            print(f\"  {h_name:<8} | {plat:>7.3f}° N | {plon:>7.3f}° E | ±{mean_r:>5.1f} km     | {tlat:>7.3f}° N | {tlon:>7.3f}° E | {err:>6.1f} km\")\n",
            "        else:\n",
            "            print(f\"  {h_name:<8} | {plat:>7.3f}° N | {plon:>7.3f}° E | ±{mean_r:>5.1f} km     | {'--':<10} | {'--':<10} | {'--'}\")\n",
            "\n",
            "    print(\"-\" * 85)\n",
            "    if true_future is not None:\n",
            "        print(f\"  Mean Track Forecast Error (TFE): {tot_err / len(HORIZONS):.2f} km\")\n",
            "    print(\"=\" * 85)\n",
            "\n",
            "    # Plot Synoptic Map\n",
            "    fig, ax = plt.subplots(figsize=(10, 7), facecolor=\"#0f172a\")\n",
            "    ax.set_facecolor(\"#1e293b\")\n",
            "    ax.tick_params(colors=\"white\")\n",
            "    for spine in ax.spines.values():\n",
            "        spine.set_color(\"#475569\")\n",
            "\n",
            "    ax.set_xlim(75.0, 95.0)\n",
            "    ax.set_ylim(8.0, 24.0)\n",
            "    ax.set_xlabel(\"Longitude (°E)\", color=\"white\", fontsize=11, fontweight=\"bold\")\n",
            "    ax.set_ylabel(\"Latitude (°N)\", color=\"white\", fontsize=11, fontweight=\"bold\")\n",
            "    ax.set_title(f\"CycloneVision AI -- 48h Trajectory Forecast & Cone of Uncertainty\\n{title_str}\",\n",
            "                 color=\"white\", fontsize=12, fontweight=\"bold\", pad=10)\n",
            "    ax.grid(True, linestyle=\"--\", alpha=0.3, color=\"gray\")\n",
            "\n",
            "    coast_lons = [80.2, 80.3, 81.0, 82.2, 83.3, 84.8, 86.8, 88.0, 89.5, 91.5]\n",
            "    coast_lats = [13.0, 14.5, 15.8, 16.9, 17.7, 19.3, 20.8, 21.6, 22.0, 22.2]\n",
            "    ax.plot(coast_lons, coast_lats, color=\"#64748b\", linewidth=2.0, label=\"East India / Odisha Coastline\")\n",
            "\n",
            "    ax.plot(past_coords[:, 1], past_coords[:, 0], color=\"#38bdf8\", marker=\"o\", linewidth=2.5, label=\"Observed Past Track (t-9h -> t0)\")\n",
            "    ax.scatter(past_coords[-1, 1], past_coords[-1, 0], color=\"#facc15\", s=130, marker=\"*\", zorder=5, label=\"Current Eye Fix (t=0h)\")\n",
            "\n",
            "    if true_future is not None:\n",
            "        ax.plot(true_future[:, 1], true_future[:, 0], color=\"#4ade80\", marker=\"s\", linewidth=2.0, linestyle=\"--\", label=\"Ground Truth Track\")\n",
            "\n",
            "    full_pred_lons = np.concatenate([[past_coords[-1, 1]], pred_coords[:, 1]])\n",
            "    full_pred_lats = np.concatenate([[past_coords[-1, 0]], pred_coords[:, 0]])\n",
            "    ax.plot(full_pred_lons, full_pred_lats, color=\"#f43f5e\", marker=\"^\", linewidth=2.5, label=\"Predicted Trajectory (+6h -> +48h)\")\n",
            "\n",
            "    for i, (plat, plon) in enumerate(pred_coords):\n",
            "        r = radii_km[i, 0] if radii_km.ndim > 1 else radii_km[i]\n",
            "        sigma_lat = r / 111.0\n",
            "        sigma_lon = r / (111.0 * max(0.1, np.cos(np.radians(plat))))\n",
            "        ell = patches.Ellipse((plon, plat), width=sigma_lon * 2.0, height=sigma_lat * 2.0, angle=0, color=\"#f43f5e\", alpha=0.15 + 0.03 * i)\n",
            "        ax.add_patch(ell)\n",
            "        ax.text(plon + 0.2, plat - 0.2, HORIZONS[i], color=\"#fca5a5\", fontsize=8, fontweight=\"bold\")\n",
            "\n",
            "    ax.legend(loc=\"lower right\", facecolor=\"#1e293b\", edgecolor=\"#475569\", labelcolor=\"white\", fontsize=9)\n",
            "    plt.tight_layout()\n",
            "    plt.show()\n",
            "\n",
            "# Example 1: Predict on custom user-provided storm coordinates\n",
            "print(\"[EXAMPLE 1] Predicting on Custom Coordinates:\")\n",
            "predict_cyclone_trajectory(\n",
            "    custom_lats=[13.20, 14.00, 14.75, 15.40],\n",
            "    custom_lons=[87.50, 87.00, 86.40, 85.80],\n",
            "    custom_pressures=[1000.0, 994.0, 986.0, 978.0],\n",
            "    custom_winds=[35.0, 48.0, 62.0, 75.0]\n",
            ")\n",
            "\n",
            "# Example 2: Predict on Unseen Test Dataset Sample #42\n",
            "# print(\"\\n[EXAMPLE 2] Predicting on Test Dataset Sample #42:\")\n",
            "# predict_cyclone_trajectory(sample_idx=42, split=\"test\")"
        ]
    }
]

nb["cells"] = cleaned_cells + step12_cells

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"[+] Step 12 successfully added to cyclone_pipleline.ipynb! Total cells: {len(nb['cells'])}")
