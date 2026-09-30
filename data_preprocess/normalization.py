import numpy as np


def normalize_satellite_channel(arr, clip_percentiles=(1.5, 98.5)):
    """Normalize one satellite channel to [0, 1] using the live preview rule."""
    arr = np.nan_to_num(
        np.asarray(arr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    valid_pixels = arr[arr > 0.0] if np.any(arr > 0.0) else arr

    if valid_pixels.size > 10:
        vmin = float(np.percentile(valid_pixels, clip_percentiles[0]))
        vmax = float(np.percentile(valid_pixels, clip_percentiles[1]))
    else:
        vmin, vmax = float(arr.min()), float(arr.max())

    if vmax <= vmin:
        return np.zeros_like(arr, dtype=np.float32)

    return np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0).astype(np.float32)