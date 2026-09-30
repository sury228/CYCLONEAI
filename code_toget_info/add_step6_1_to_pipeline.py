"""
CycloneVision AI - Add Step 6.1 to cyclone_pipleline.ipynb
Adds interactive prediction for Model 1 (Vortex Presence, Eye Pinpoint, Eyewall Mask) right after Step 6.
"""

import os
import json

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NOTEBOOK_PATH = os.path.join(BASE_DIR, "cyclone_pipleline.ipynb")

with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Ensure no duplicate Step 6.1
cleaned_cells = []
for cell in nb["cells"]:
    src = "".join(cell.get("source", [])) if isinstance(cell.get("source", []), list) else str(cell.get("source", ""))
    if "Step 6.1" in src or "predict_model1_sample" in src:
        continue
    cleaned_cells.append(cell)

step6_1_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 6.1: Predict on New / Custom Satellite Data (Vortex Presence, Eye Pinpointing & Eyewall Mask)\n",
            "\n",
            "Directly applies Model 1 to predict on **new satellite images** or any test dataset sample:\n",
            "1. **Input Options**:\n",
            "   - **Dataset Sample Index**: Specify `--sample_idx` and `--split` to evaluate any unseen sample.\n",
            "   - **Custom Satellite Tensor**: Pass any 4-channel satellite imagery array `(4, 128, 128)` `[IR, WV, MW, VIS]`.\n",
            "2. **Outputs**:\n",
            "   - **Output 1**: Cyclone vortex presence detection confidence (%).\n",
            "   - **Output 2**: Pinpointed Eye geographic coordinates (Latitude, Longitude in °N, °E) & estimated eye diameter (km).\n",
            "   - **Output 3**: High-resolution eyewall cloud ring segmentation mask & Dice agreement score.\n",
            "3. **Visual Dashboard**: Displays a 4-channel meteorological visualizer with pinpointed center and eyewall contour."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import matplotlib.pyplot as plt\n",
            "import matplotlib.patches as patches\n",
            "from vortex_identification_model import convert_bbox_to_latlon\n",
            "\n",
            "def predict_model1_sample(\n",
            "    sample_idx=100,\n",
            "    split=\"test\",\n",
            "    storm_lat=15.40,\n",
            "    storm_lon=87.20,\n",
            "    patch_km=500.0,\n",
            "    custom_tensor=None\n",
            "):\n",
            "    \"\"\"\n",
            "    Runs Model 1 inference and generates a comprehensive 4-panel visual dashboard.\n",
            "    \"\"\"\n",
            "    model.eval()\n",
            "    gt_target = None\n",
            "\n",
            "    if custom_tensor is not None:\n",
            "        if isinstance(custom_tensor, np.ndarray):\n",
            "            img = torch.from_numpy(custom_tensor).float()\n",
            "        else:\n",
            "            img = custom_tensor.clone().float()\n",
            "        if img.ndim == 3:\n",
            "            img = img.unsqueeze(0)\n",
            "        sample_desc = \"Custom Satellite Image Input\"\n",
            "    else:\n",
            "        loader_map = {\"test\": test_loader, \"val\": val_loader, \"train\": train_loader}\n",
            "        target_loader = loader_map.get(split.lower(), test_loader)\n",
            "        dataset = target_loader.dataset\n",
            "\n",
            "        idx = max(0, min(sample_idx, len(dataset) - 1))\n",
            "        img_raw, nwp_raw, gt_target = dataset[idx]\n",
            "        img = img_raw.unsqueeze(0)\n",
            "        sample_desc = f\"{split.upper()} Dataset Sample #{idx:,} (of {len(dataset):,})\"\n",
            "\n",
            "    img = torch.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0).to(device)\n",
            "\n",
            "    with torch.no_grad():\n",
            "        out = model(img)\n",
            "\n",
            "    presence_prob = float(out[\"vortex_presence_prob\"][0].item())\n",
            "    pred_presence = bool(presence_prob >= 0.5)\n",
            "    pred_bbox = out[\"eye_bbox\"][0].cpu().numpy()\n",
            "    mask_prob = out[\"eyewall_mask_prob\"][0, 0].cpu().numpy()\n",
            "\n",
            "    eye_lat, eye_lon, eye_dia_km = convert_bbox_to_latlon(\n",
            "        pred_bbox, center_lat=storm_lat, center_lon=storm_lon, patch_km=patch_km\n",
            "    )\n",
            "\n",
            "    # Ground truth comparison\n",
            "    gt_presence = None\n",
            "    gt_bbox = None\n",
            "    spatial_err_km = None\n",
            "    dice_score = None\n",
            "\n",
            "    if gt_target is not None:\n",
            "        gt_presence = bool(gt_target[\"vortex_presence\"].item() >= 0.5)\n",
            "        gt_bbox = gt_target[\"eye_bbox\"].numpy()\n",
            "        gt_mask = gt_target[\"eyewall_mask\"][0].numpy()\n",
            "        spatial_err_km = float(np.sqrt((pred_bbox[0] - gt_bbox[0])**2 + (pred_bbox[1] - gt_bbox[1])**2) * patch_km)\n",
            "\n",
            "        pred_bin = (mask_prob >= 0.5).astype(float)\n",
            "        inter = (pred_bin * gt_mask).sum()\n",
            "        union = pred_bin.sum() + gt_mask.sum()\n",
            "        dice_score = float((2.0 * inter + 1e-6) / (union + 1e-6))\n",
            "\n",
            "    # Print Report\n",
            "    print(\"=\" * 75)\n",
            "    print(\"   CYCLONEVISION AI -- MODEL 1 PREDICTION REPORT\")\n",
            "    print(\"=\" * 75)\n",
            "    print(f\"  Target Input           : {sample_desc}\")\n",
            "    print(f\"  Patch Center Reference : {storm_lat:.2f}° N, {storm_lon:.2f}° E (Swath: {patch_km:.0f} km)\")\n",
            "    print(\"-\" * 75)\n",
            "    status_str = 'CYCLONE DETECTED [POSITIVE]' if pred_presence else 'NO CYCLONE DETECTED [NEGATIVE]'\n",
            "    print(f\"  [OUTPUT 1] Vortex Presence Probability : {presence_prob:.2%}\")\n",
            "    print(f\"             Detection Status            : {status_str}\")\n",
            "    print(\"-\" * 75)\n",
            "    print(f\"  [OUTPUT 2] Predicted Eye BBox [x,y,w,h]: [{pred_bbox[0]:.4f}, {pred_bbox[1]:.4f}, {pred_bbox[2]:.4f}, {pred_bbox[3]:.4f}]\")\n",
            "    print(f\"             Pinpointed Eye Latitude     : {eye_lat:.3f}° N\")\n",
            "    print(f\"             Pinpointed Eye Longitude    : {eye_lon:.3f}° E\")\n",
            "    print(f\"             Estimated Eye Diameter      : {eye_dia_km:.1f} km\")\n",
            "    if gt_bbox is not None:\n",
            "        gt_lat, gt_lon, gt_dia = convert_bbox_to_latlon(gt_bbox, center_lat=storm_lat, center_lon=storm_lon, patch_km=patch_km)\n",
            "        print(f\"             Ground Truth Eye Center     : {gt_lat:.3f}° N, {gt_lon:.3f}° E (Diameter: {gt_dia:.1f} km)\")\n",
            "        print(f\"             Localization Error          : ~{spatial_err_km:.2f} km\")\n",
            "    print(\"-\" * 75)\n",
            "    print(f\"  [OUTPUT 3] Eyewall Cloud Ring Mask     : Probability Map Shape {list(mask_prob.shape)}\")\n",
            "    if dice_score is not None:\n",
            "        print(f\"             Eyewall Dice Agreement Score: {dice_score:.4f} ({dice_score * 100:.2f}% overlap)\")\n",
            "    print(\"=\" * 75)\n",
            "\n",
            "    # 4-Panel Visual Dashboard\n",
            "    img_np = img[0].cpu().numpy()\n",
            "    fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor=\"#0f172a\")\n",
            "    for ax in axes:\n",
            "        ax.set_facecolor(\"#1e293b\")\n",
            "        ax.tick_params(colors=\"white\")\n",
            "\n",
            "    # Panel 1: IR + Pred Eye\n",
            "    im0 = axes[0].imshow(img_np[0], cmap=\"inferno\", vmin=0, vmax=1)\n",
            "    xc, yc, w, h = pred_bbox\n",
            "    rect_pred = patches.Rectangle(\n",
            "        ((xc - w / 2) * 128, (yc - h / 2) * 128), w * 128, h * 128,\n",
            "        linewidth=2.5, edgecolor=\"#00ffcc\", facecolor=\"none\",\n",
            "        label=f\"Pred Eye ({eye_lat:.2f}N, {eye_lon:.2f}E)\"\n",
            "    )\n",
            "    axes[0].add_patch(rect_pred)\n",
            "    axes[0].scatter(xc * 128, yc * 128, color=\"#00ffcc\", s=40, marker=\"+\", label=\"Pred Center\")\n",
            "    axes[0].set_title(f\"Ch 0: Infrared (IR) + Pred Eye\\n[Prob: {presence_prob:.1%}]\", color=\"white\", fontsize=11, fontweight=\"bold\")\n",
            "    axes[0].legend(loc=\"upper right\", fontsize=8, facecolor=\"#1e293b\", labelcolor=\"white\")\n",
            "    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)\n",
            "\n",
            "    # Panel 2: Water Vapor + GT vs Pred\n",
            "    im1 = axes[1].imshow(img_np[1], cmap=\"magma\", vmin=0, vmax=1)\n",
            "    if gt_bbox is not None:\n",
            "        gxc, gyc, gw, gh = gt_bbox\n",
            "        rect_gt = patches.Rectangle(\n",
            "            ((gxc - gw / 2) * 128, (gyc - gh / 2) * 128), gw * 128, gh * 128,\n",
            "            linewidth=2.5, edgecolor=\"#ff0055\", facecolor=\"none\", linestyle=\"--\", label=\"Ground Truth Eye\"\n",
            "        )\n",
            "        axes[1].add_patch(rect_gt)\n",
            "        axes[1].scatter(gxc * 128, gyc * 128, color=\"#ff0055\", s=40, marker=\"x\", label=\"GT Center\")\n",
            "    rect_p2 = patches.Rectangle(\n",
            "        ((xc - w / 2) * 128, (yc - h / 2) * 128), w * 128, h * 128,\n",
            "        linewidth=2.0, edgecolor=\"#00ffcc\", facecolor=\"none\", label=\"Pred Eye\"\n",
            "    )\n",
            "    axes[1].add_patch(rect_p2)\n",
            "    axes[1].set_title(\"Ch 1: Water Vapor (WV)\\n[GT vs Pred Comparison]\", color=\"white\", fontsize=11, fontweight=\"bold\")\n",
            "    axes[1].legend(loc=\"upper right\", fontsize=8, facecolor=\"#1e293b\", labelcolor=\"white\")\n",
            "    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)\n",
            "\n",
            "    # Panel 3: Microwave + Eyewall Mask\n",
            "    im2 = axes[2].imshow(img_np[2], cmap=\"viridis\", vmin=0, vmax=1)\n",
            "    axes[2].contour(mask_prob, levels=[0.5], colors=[\"#ff0055\"], linewidths=2.5)\n",
            "    dice_label = f\"\\n[Dice Overlap: {dice_score:.1%}]\" if dice_score is not None else \"\"\n",
            "    axes[2].set_title(f\"Ch 2: Microwave (MW) + Eyewall Mask{dice_label}\", color=\"white\", fontsize=11, fontweight=\"bold\")\n",
            "    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)\n",
            "\n",
            "    # Panel 4: Visible Channel + Center Pinpoint\n",
            "    im3 = axes[3].imshow(img_np[3], cmap=\"cividis\", vmin=0, vmax=1)\n",
            "    axes[3].scatter(xc * 128, yc * 128, color=\"#00ffcc\", s=80, marker=\"*\", label=f\"Eye ({eye_lat:.2f}°N, {eye_lon:.2f}°E)\")\n",
            "    axes[3].set_title(f\"Ch 3: Visible (VIS) Channel\\n[Pinpoint: {eye_lat:.2f}°N, {eye_lon:.2f}°E]\", color=\"white\", fontsize=11, fontweight=\"bold\")\n",
            "    axes[3].legend(loc=\"upper right\", fontsize=8, facecolor=\"#1e293b\", labelcolor=\"white\")\n",
            "    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)\n",
            "\n",
            "    plt.tight_layout()\n",
            "    plt.show()\n",
            "\n",
            "# Run prediction on Test Split Sample #100\n",
            "predict_model1_sample(sample_idx=100, split=\"test\")"
        ]
    }
]

# Find index where Part 2 begins (or where Step 7 is)
insert_idx = len(cleaned_cells)
for i, cell in enumerate(cleaned_cells):
    src = "".join(cell.get("source", [])) if isinstance(cell.get("source", []), list) else str(cell.get("source", ""))
    if "Part 2" in src or "Step 7" in src:
        insert_idx = i
        break

# Insert Step 6.1 right before Part 2
final_cells = cleaned_cells[:insert_idx] + step6_1_cells + cleaned_cells[insert_idx:]

nb["cells"] = final_cells

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"[+] Step 6.1 successfully inserted at position {insert_idx}!")
print(f"[+] Total notebook cells: {len(nb['cells'])}")
