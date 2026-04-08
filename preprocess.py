"""
Downsample China population density TIFs from ~1km to ~5km resolution.
Output: compressed GeoTIFFs in ./data_5km/
"""

import os
import glob
import time
import numpy as np
import rasterio
from rasterio.enums import Resampling

INPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(INPUT_DIR, "data_5km")
SCALE_FACTOR = 5  # 1km -> 5km

os.makedirs(OUTPUT_DIR, exist_ok=True)

tif_files = sorted(glob.glob(os.path.join(INPUT_DIR, "China_*_1km_UNadj.tif")))
print(f"Found {len(tif_files)} TIF files to process\n")

for tif_path in tif_files:
    fname = os.path.basename(tif_path)
    year = fname.split("_")[1]
    out_name = f"China_{year}_5km.tif"
    out_path = os.path.join(OUTPUT_DIR, out_name)

    print(f"Processing {year}...", end=" ", flush=True)
    t0 = time.time()

    with rasterio.open(tif_path) as src:
        # Calculate new dimensions
        new_height = src.height // SCALE_FACTOR
        new_width = src.width // SCALE_FACTOR

        # Read with downsampling (average resampling preserves population density meaning)
        data = src.read(
            1,
            out_shape=(new_height, new_width),
            resampling=Resampling.average,
        )

        # Replace nodata with NaN for cleaner handling, then back to nodata
        nodata = src.nodata
        mask = data <= -3e+38
        data[mask] = np.nan

        # Build output transform
        transform = src.transform * src.transform.scale(
            src.width / new_width,
            src.height / new_height,
        )

        profile = src.profile.copy()
        profile.update(
            width=new_width,
            height=new_height,
            transform=transform,
            compress="deflate",
            predictor=2,
            dtype="float32",
            nodata=nodata,
        )

        # Write NaN back to nodata value before saving
        data[np.isnan(data)] = nodata

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data, 1)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    elapsed = time.time() - t0
    print(f"{new_width}x{new_height}, {size_mb:.1f}MB, {elapsed:.1f}s")

print(f"\nDone! Output in: {OUTPUT_DIR}")
total_mb = sum(
    os.path.getsize(os.path.join(OUTPUT_DIR, f)) / (1024 * 1024)
    for f in os.listdir(OUTPUT_DIR)
    if f.endswith(".tif")
)
print(f"Total size: {total_mb:.1f}MB")
