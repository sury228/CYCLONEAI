"""
Removes Step 7 from cyclone_pipleline.ipynb to keep the notebook ending at Step 6 Dashboard.
"""

import os
import json

notebook_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cyclone_pipleline.ipynb"))

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Keep only cells that do NOT contain Step 7
cleaned_cells = []
for cell in nb["cells"]:
    src = cell.get("source", [])
    src_text = "".join(src) if isinstance(src, list) else str(src)
    if "Step 7" in src_text or "predict_cyclone_sample" in src_text:
        continue
    cleaned_cells.append(cell)

nb["cells"] = cleaned_cells

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"[+] Step 7 successfully removed. Notebook has {len(cleaned_cells)} cells ending at Step 6.")
