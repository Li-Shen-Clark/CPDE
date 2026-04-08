"""
Compute per-province zonal statistics from 5km population density TIFs.
Uses latitude-corrected pixel areas instead of a fixed 25 km².
Also generates national-level raster summary (not derived from provinces).
Output: data_5km/province_stats.json, data_5km/national_stats.json
"""

import os
import json
import glob
import time
import numpy as np
import rasterio
import geopandas as gpd
from rasterstats import zonal_stats
from rasterio.features import geometry_mask

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data_5km")
GEOJSON_PATH = os.path.join(BASE_DIR, "china_provinces.geojson")
PROV_OUTPUT = os.path.join(DATA_DIR, "province_stats.json")
NATL_OUTPUT = os.path.join(DATA_DIR, "national_stats.json")
EXCLUDE_HMT = {"香港特别行政区", "澳门特别行政区", "台湾省"}

# ---- Load province boundaries ----
gdf = gpd.read_file(GEOJSON_PATH)
gdf = gdf[gdf["name"] != ""].copy()
gdf = gdf[~gdf["adcode"].astype(str).str.contains("_")].copy()
gdf_mainland = gdf[~gdf["name"].isin(EXCLUDE_HMT)].copy()
print(f"Loaded {len(gdf)} provinces")

tif_files = sorted(glob.glob(os.path.join(DATA_DIR, "China_*_5km.tif")))
print(f"Found {len(tif_files)} TIF files")


# ---- Build latitude-corrected pixel area grid (once) ----
# All TIFs share the same grid, so compute once from the first file.
with rasterio.open(tif_files[0]) as src:
    height, width = src.height, src.width
    res_deg = src.res[0]  # ~0.0417 degrees
    lat_top = src.bounds.top
    base_transform = src.transform

# pixel_area[row] = area in km² for that latitude band
pixel_areas = np.empty(height, dtype=np.float64)
for row in range(height):
    lat = lat_top - (row + 0.5) * res_deg
    lat_rad = np.radians(lat)
    # 1° longitude ≈ 111.32 * cos(lat) km; 1° latitude ≈ 111.32 km
    pixel_areas[row] = (res_deg * 111.32) * (res_deg * 111.32 * np.cos(lat_rad))

# Broadcast to 2D for element-wise multiply: area_grid[row, col]
area_grid = np.broadcast_to(pixel_areas[:, np.newaxis], (height, width))
print(f"Pixel area range: {pixel_areas.min():.2f} – {pixel_areas.max():.2f} km² "
      f"(mean {pixel_areas.mean():.2f})")

mainland_mask = geometry_mask(
    list(gdf_mainland.geometry),
    transform=base_transform,
    invert=True,
    out_shape=(height, width),
)
print(f"Mainland mask coverage: {int(mainland_mask.sum()):,} pixels")


def load_density(tif_path):
    """Load a TIF and return float64 array with NaN for nodata."""
    with rasterio.open(tif_path) as src:
        data = src.read(1).astype(np.float64)
    data[data < -1e30] = np.nan
    data[data < 0] = np.nan
    return data


def summarize_scope(data, scope_mask=None):
    """Summarize a density grid for a given spatial mask."""
    valid_mask = ~np.isnan(data)
    if scope_mask is not None:
        valid_mask &= scope_mask

    if not np.any(valid_mask):
        return {
            "est_pop_wan": 0.0,
            "mean_density": 0.0,
            "median_density": 0.0,
            "max_density": 0.0,
            "weighted_mean_density": 0.0,
            "high_density_pixels_1000": 0,
            "total_valid_pixels": 0,
            "inhabited_pixels": 0,
        }

    valid = data[valid_mask]
    est_pop = float(np.nansum(data[valid_mask] * area_grid[valid_mask]))
    inhabited_mask = valid_mask & (data > 0)
    area_sum = float(np.sum(area_grid[valid_mask]))
    weighted_mean = float(est_pop / area_sum) if area_sum > 0 else 0.0

    return {
        "est_pop_wan": round(est_pop / 10000, 2),
        "mean_density": round(float(np.mean(valid)), 2),
        "median_density": round(float(np.median(valid)), 2),
        "max_density": round(float(np.max(valid)), 2),
        "weighted_mean_density": round(weighted_mean, 2),
        "high_density_pixels_1000": int(np.sum(data[inhabited_mask] > 1000)),
        "total_valid_pixels": int(np.sum(valid_mask)),
        "inhabited_pixels": int(np.sum(inhabited_mask)),
    }


# ===================================================================
# Part 1: Province-level stats
# ===================================================================
print("\n=== Province stats ===")
all_prov_stats = {}

for tif_path in tif_files:
    fname = os.path.basename(tif_path)
    year = fname.split("_")[1]
    print(f"Processing {year}...", end=" ", flush=True)
    t0 = time.time()

    data = load_density(tif_path)

    # rasterstats for density stats (mean, median, etc.)
    # Use all_touched=False to avoid double-counting border pixels
    stats = zonal_stats(
        gdf,
        tif_path,
        stats=["mean", "median", "max", "min", "count", "std"],
        nodata=-3.4028230607370965e+38,
        all_touched=False,
    )

    # For population estimate, we need sum(density * pixel_area) per province.
    # rasterstats doesn't support per-pixel area weighting, so we use
    # the population grid = density * area_grid, then zonal_stats sum on that.
    pop_grid = data * area_grid
    # Write a temporary in-memory raster is overkill; instead compute from
    # the already-computed stats indices. But simplest correct approach:
    # use rasterstats with the pop_grid as an array + affine from the source.
    with rasterio.open(tif_path) as src:
        affine = src.transform
        nodata_val = src.nodata

    pop_grid_masked = np.where(np.isnan(pop_grid), nodata_val, pop_grid).astype(np.float32)
    pop_stats = zonal_stats(
        gdf,
        pop_grid_masked,
        stats=["sum"],
        nodata=nodata_val,
        all_touched=False,
        affine=affine,
    )

    year_data = []
    for i, row in gdf.iterrows():
        idx = gdf.index.get_loc(i)
        s = stats[idx]
        ps = pop_stats[idx]
        if s["count"] is None or s["count"] == 0:
            continue
        est_pop = ps["sum"] if ps["sum"] else 0
        year_data.append({
            "adcode": str(row["adcode"]),
            "name": row["name"],
            "mean": round(s["mean"], 2) if s["mean"] else 0,
            "median": round(s["median"], 2) if s["median"] else 0,
            "max": round(s["max"], 2) if s["max"] else 0,
            "min": round(s["min"], 4) if s["min"] else 0,
            "std": round(s["std"], 2) if s["std"] else 0,
            "pixel_count": s["count"],
            "est_pop_wan": round(est_pop / 10000, 2),  # 万人
        })

    all_prov_stats[year] = year_data
    elapsed = time.time() - t0
    print(f"{len(year_data)} provinces, {elapsed:.1f}s")

with open(PROV_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(all_prov_stats, f, ensure_ascii=False, indent=2)
print(f"\nProvince stats → {PROV_OUTPUT} ({os.path.getsize(PROV_OUTPUT)/1024:.0f}KB)")


# ===================================================================
# Part 2: National-level raster summary (direct from raster)
# ===================================================================
print("\n=== National raster summary ===")
national_stats = {}

for tif_path in tif_files:
    fname = os.path.basename(tif_path)
    year = fname.split("_")[1]

    data = load_density(tif_path)
    national_stats[year] = {
        "all": summarize_scope(data),
        "mainland": summarize_scope(data, mainland_mask),
    }
    print(
        f"  {year}: all={national_stats[year]['all']['est_pop_wan']/10000:.2f}亿, "
        f"mainland={national_stats[year]['mainland']['est_pop_wan']/10000:.2f}亿"
    )

with open(NATL_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(national_stats, f, ensure_ascii=False, indent=2)
print(f"\nNational stats → {NATL_OUTPUT} ({os.path.getsize(NATL_OUTPUT)/1024:.0f}KB)")

# ---- Sanity check ----
print("\n=== Sanity check ===")
for y in ["2002", "2010", "2020"]:
    prov_total = sum(p["est_pop_wan"] for p in all_prov_stats[y])
    natl_total = national_stats[y]["all"]["est_pop_wan"]
    mainland_total = national_stats[y]["mainland"]["est_pop_wan"]
    print(
        f"  {y}: 省级汇总={prov_total/10000:.2f}亿, "
        f"全国栅格={natl_total/10000:.2f}亿, 大陆栅格={mainland_total/10000:.2f}亿"
    )
