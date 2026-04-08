"""
中国人口密度时空探索器 (2002-2020)
Spatiotemporal Explorer: Time + Map + Compare + Drill-down

Fixes applied:
  1. Latitude-corrected pixel area (not fixed 25 km²)
  2. National summary from raster (not province rollup)
  3. Adaptive diff colorscale via p99
  4. Mainland-only toggle
  5. Independent year controls per tab
  6. Map-click pixel drill-down
"""

import os
import json
import time as _time
import numpy as np
import pandas as pd
import rasterio
import streamlit as st
import folium
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib import cm
from matplotlib.colors import LogNorm, Normalize
from PIL import Image, ImageDraw
from scipy import stats as sp_stats
from streamlit_folium import st_folium

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="中国人口密度时空探索器",
    page_icon="🌏",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data_5km")
GEOJSON_PATH = os.path.join(BASE_DIR, "china_provinces.geojson")
PROVINCE_STATS_PATH = os.path.join(DATA_DIR, "province_stats.json")
NATIONAL_STATS_PATH = os.path.join(DATA_DIR, "national_stats.json")
YEARS = list(range(2002, 2021))
VMIN, VMAX = 1, 10000

EXCLUDE_HMT = {"香港特别行政区", "澳门特别行政区", "台湾省"}
WORLDPOP_COLLECTION_URL = "https://hub.worldpop.org/Global1_2000-2020"
WORLDPOP_PROJECT_URL = "https://hub.worldpop.org/project/list"
WORLDPOP_CHINA_2002_URL = "https://hub.worldpop.org/geodata/summary?id=44816"
WORLDPOP_CHINA_2020_URL = "https://hub.worldpop.org/geodata/summary?id=44834"
WORLDPOP_DOI_URL = "https://doi.org/10.5258/SOTON/WP00675"
CC_BY_4_URL = "https://creativecommons.org/licenses/by/4.0/"

# ---------------------------------------------------------------------------
# Region classification (东/中/西/东北)
# ---------------------------------------------------------------------------
REGION_MAP = {
    "北京市": "东部", "天津市": "东部", "河北省": "东部", "上海市": "东部",
    "江苏省": "东部", "浙江省": "东部", "福建省": "东部", "山东省": "东部",
    "广东省": "东部", "海南省": "东部",
    "山西省": "中部", "安徽省": "中部", "江西省": "中部", "河南省": "中部",
    "湖北省": "中部", "湖南省": "中部",
    "内蒙古自治区": "西部", "广西壮族自治区": "西部", "重庆市": "西部",
    "四川省": "西部", "贵州省": "西部", "云南省": "西部", "西藏自治区": "西部",
    "陕西省": "西部", "甘肃省": "西部", "青海省": "西部", "宁夏回族自治区": "西部",
    "新疆维吾尔自治区": "西部",
    "辽宁省": "东北", "吉林省": "东北", "黑龙江省": "东北",
}
REGION_COLORS = {"东部": "#E31A1C", "中部": "#FF7F00", "西部": "#33A02C", "东北": "#1F78B4"}


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------
@st.cache_data
def load_year(year):
    path = os.path.join(DATA_DIR, f"China_{year}_5km.tif")
    with rasterio.open(path) as src:
        data = src.read(1)
        bounds = [[src.bounds.bottom, src.bounds.left],
                  [src.bounds.top, src.bounds.right]]
        res_deg = src.res[0]
        lat_top = src.bounds.top
        height = src.height
    data = data.astype(np.float64)
    data[data < -1e30] = np.nan
    data[data < 0] = np.nan
    return data, bounds, res_deg, lat_top, height


@st.cache_data
def load_province_stats():
    with open(PROVINCE_STATS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# [Fix 2] National summary directly from raster, with scope support
@st.cache_data
def load_national_stats(scope_key):
    with open(NATIONAL_STATS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    records = []
    for y in YEARS:
        year_payload = raw[str(y)]
        d = year_payload.get(scope_key, year_payload)
        records.append({
            "年份": y,
            "估算总人口(万)": d["est_pop_wan"],
            "加权平均密度": d["weighted_mean_density"],
            "平均密度": d["mean_density"],
            "高密度像素数": d["high_density_pixels_1000"],
            "有效像元数": d.get("total_valid_pixels"),
            "有人口像元数": d.get("inhabited_pixels"),
        })
    return pd.DataFrame(records)


@st.cache_data
def load_geojson():
    with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
        gj = json.load(f)
    gj["features"] = [
        feat for feat in gj["features"]
        if feat["properties"].get("name", "") != ""
        and "_" not in str(feat["properties"].get("adcode", ""))
    ]
    return gj


@st.cache_data
def build_province_df(province_stats):
    rows = []
    for y in YEARS:
        for p in province_stats[str(y)]:
            rows.append({"年份": y, **p})
    return pd.DataFrame(rows)


@st.cache_data
def compute_rank_history(province_stats, exclude_names):
    rows = []
    for y in YEARS:
        provs = [p for p in province_stats[str(y)] if p["name"] not in exclude_names]
        provs.sort(key=lambda x: x["mean"], reverse=True)
        for rank, p in enumerate(provs, 1):
            rows.append({"年份": y, "省份": p["name"], "排名": rank, "平均密度": p["mean"]})
    return pd.DataFrame(rows)


# --- Image renderers ---
@st.cache_data
def raster_to_image(data):
    norm = LogNorm(vmin=VMIN, vmax=VMAX, clip=True)
    colormap = cm.get_cmap("YlOrRd")
    mask = np.isnan(data) | (data <= 0)
    normalized = np.where(mask, 0, norm(data))
    rgba = colormap(normalized)
    rgba[mask, 3] = 0
    return (rgba * 255).astype(np.uint8)


# [Fix 3] Adaptive diff colorscale using p99
@st.cache_data
def diff_to_image_adaptive(diff):
    """Red-blue diverging colormap with p99-based symmetric bounds."""
    colormap = cm.get_cmap("RdBu_r")
    valid = diff[~np.isnan(diff)]
    if len(valid) == 0:
        return np.zeros((*diff.shape, 4), dtype=np.uint8)
    p99 = float(np.percentile(np.abs(valid), 99))
    p99 = max(p99, 1.0)
    clipped_count = int(np.sum(np.abs(valid) > p99))
    norm = Normalize(vmin=-p99, vmax=p99, clip=True)
    mask = np.isnan(diff)
    normalized = np.where(mask, 0.5, norm(diff))
    rgba = colormap(normalized)
    rgba[mask, 3] = 0
    return (rgba * 255).astype(np.uint8), p99, clipped_count, len(valid)


@st.cache_data
def build_boundary_segments(geojson_obj):
    """Flatten province boundaries into drawable lon/lat line segments."""
    segments = []
    for feat in geojson_obj["features"]:
        geom = feat.get("geometry") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates", [])
        if gtype == "Polygon":
            polygons = [coords]
        elif gtype == "MultiPolygon":
            polygons = coords
        else:
            continue
        for polygon in polygons:
            for ring in polygon:
                segments.append([(float(lon), float(lat)) for lon, lat in ring])
    return segments


@st.cache_data
def render_animation_frame_image(year, opacity):
    """Render a smooth, non-iframe animation frame to reduce white flashes."""
    data, bounds, _, _, _ = load_year(year)
    rgba = raster_to_image(data)

    h, w = rgba.shape[:2]
    min_lat, min_lon = bounds[0]
    max_lat, max_lon = bounds[1]

    background = np.full((h, w, 3), 246, dtype=np.float32)
    alpha = (rgba[..., 3:4].astype(np.float32) / 255.0) * float(opacity)
    rgb = rgba[..., :3].astype(np.float32)
    blended = (background * (1.0 - alpha) + rgb * alpha).astype(np.uint8)

    canvas = Image.fromarray(blended, mode="RGB")
    draw = ImageDraw.Draw(canvas)
    lon_span = max(max_lon - min_lon, 1e-9)
    lat_span = max(max_lat - min_lat, 1e-9)

    for ring in boundary_segments:
        pts = []
        for lon, lat in ring:
            x = (lon - min_lon) / lon_span * (w - 1)
            y = (max_lat - lat) / lat_span * (h - 1)
            pts.append((x, y))
        if len(pts) >= 2:
            draw.line(pts, fill=(88, 88, 88), width=1)

    return np.asarray(canvas)


@st.cache_data
def render_diff_frame_image(year_a, year_b, opacity):
    """Render a static diff frame for summary use."""
    data_a, bounds, _, _, _ = load_year(year_a)
    data_b, _, _, _, _ = load_year(year_b)
    diff = data_b - data_a
    rgba, p99, clipped_n, total_n = diff_to_image_adaptive(diff)

    h, w = rgba.shape[:2]
    background = np.full((h, w, 3), 246, dtype=np.float32)
    alpha = (rgba[..., 3:4].astype(np.float32) / 255.0) * float(opacity)
    rgb = rgba[..., :3].astype(np.float32)
    blended = (background * (1.0 - alpha) + rgb * alpha).astype(np.uint8)

    canvas = Image.fromarray(blended, mode="RGB")
    draw = ImageDraw.Draw(canvas)
    min_lat, min_lon = bounds[0]
    max_lat, max_lon = bounds[1]
    lon_span = max(max_lon - min_lon, 1e-9)
    lat_span = max(max_lat - min_lat, 1e-9)
    for ring in boundary_segments:
        pts = []
        for lon, lat in ring:
            x = (lon - min_lon) / lon_span * (w - 1)
            y = (max_lat - lat) / lat_span * (h - 1)
            pts.append((x, y))
        if len(pts) >= 2:
            draw.line(pts, fill=(88, 88, 88), width=1)

    valid_diff = diff[~np.isnan(diff)]
    return np.asarray(canvas), p99, clipped_n, total_n, float(np.nanmean(valid_diff))


@st.cache_data
def precompute_cumulative_diff_stats(base_year):
    """Precompute cumulative change stats relative to a base year."""
    base_layer = raster_cube[YEARS.index(base_year)]
    stats = {}
    for i, y in enumerate(YEARS):
        diff = raster_cube[i] - base_layer
        valid = diff[~np.isnan(diff)]
        if len(valid) == 0:
            stats[y] = {
                "mean_change": 0.0,
                "increase_share": 0.0,
                "decrease_share": 0.0,
                "max_increase": 0.0,
            }
            continue
        stats[y] = {
            "mean_change": float(np.nanmean(valid)),
            "increase_share": float(np.sum(valid > 0) / len(valid) * 100),
            "decrease_share": float(np.sum(valid < 0) / len(valid) * 100),
            "max_increase": float(np.nanmax(valid)),
        }
    return stats


@st.cache_data
def build_growth_df(exclude_names):
    rows = []
    prov_2002 = {p["name"]: p for p in province_stats["2002"] if p["name"] not in exclude_names}
    prov_2020 = {p["name"]: p for p in province_stats["2020"] if p["name"] not in exclude_names}
    for name, p2002 in prov_2002.items():
        p2020 = prov_2020.get(name)
        if not p2020:
            continue
        abs_change = p2020["mean"] - p2002["mean"]
        growth_rate = (abs_change / p2002["mean"] * 100) if p2002["mean"] > 0 else 0
        rows.append({
            "省份": name,
            "2002年密度": p2002["mean"],
            "2020年密度": p2020["mean"],
            "增长率%": round(growth_rate, 1),
            "绝对增长": round(abs_change, 1),
            "2020人口(万)": p2020["est_pop_wan"],
        })
    return pd.DataFrame(rows).sort_values("绝对增长", ascending=False).reset_index(drop=True)


@st.cache_data
def build_growth_animation_df(exclude_names):
    rows = []
    baseline = {p["name"]: p for p in province_stats[str(YEARS[0])] if p["name"] not in exclude_names}
    for y in YEARS:
        current = {p["name"]: p for p in province_stats[str(y)] if p["name"] not in exclude_names}
        for name, base_row in baseline.items():
            cur_row = current.get(name)
            if not cur_row:
                continue
            abs_change = cur_row["mean"] - base_row["mean"]
            growth_rate = (abs_change / base_row["mean"] * 100) if base_row["mean"] > 0 else 0
            rows.append({
                "年份": y,
                "省份": name,
                "2002年密度": base_row["mean"],
                "当年密度": cur_row["mean"],
                "增长率%": round(growth_rate, 1),
                "绝对增长": round(abs_change, 1),
                "当年人口(万)": cur_row["est_pop_wan"],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# New: Spatial economics indicators
# ---------------------------------------------------------------------------
@st.cache_data
def compute_pixel_area_grid(res_deg, lat_top, height, width):
    """Latitude-corrected pixel area grid (km²) matching the raster cube."""
    pixel_areas = np.empty(height, dtype=np.float64)
    for row in range(height):
        lat = lat_top - (row + 0.5) * res_deg
        pixel_areas[row] = (res_deg * 111.32) * (res_deg * 111.32 * np.cos(np.radians(lat)))
    return np.broadcast_to(pixel_areas[:, np.newaxis], (height, width)).copy()


@st.cache_data
def compute_spatial_gini_series(_cube, _area_grid):
    """Spatial Gini coefficient of population (density×area) per year."""
    records = []
    for i, y in enumerate(YEARS):
        layer = _cube[i]
        valid = ~np.isnan(layer) & (layer >= 0)
        pop = layer[valid] * _area_grid[valid]
        pop = np.sort(pop)
        n = len(pop)
        if n == 0:
            records.append({"年份": y, "空间基尼系数": 0.0})
            continue
        cumsum = np.cumsum(pop)
        total = cumsum[-1]
        if total <= 0:
            records.append({"年份": y, "空间基尼系数": 0.0})
            continue
        gini = 1 - 2 * np.sum(cumsum) / (n * total) + 1 / n
        records.append({"年份": y, "空间基尼系数": round(float(gini), 4)})
    return pd.DataFrame(records)


@st.cache_data
def compute_concentration_series(_cube, _area_grid, top_pct=10):
    """Share of total population living in top-X% densest pixels, per year."""
    records = []
    for i, y in enumerate(YEARS):
        layer = _cube[i]
        valid = ~np.isnan(layer) & (layer >= 0)
        pop = layer[valid] * _area_grid[valid]
        total_pop = float(np.sum(pop))
        if total_pop <= 0:
            records.append({"年份": y, f"Top{top_pct}%人口占比": 0.0})
            continue
        sorted_pop = np.sort(pop)[::-1]
        k = max(1, int(len(sorted_pop) * top_pct / 100))
        top_share = float(np.sum(sorted_pop[:k]) / total_pop * 100)
        records.append({"年份": y, f"Top{top_pct}%人口占比": round(top_share, 2)})
    return pd.DataFrame(records)


@st.cache_data
def compute_population_centroid(_cube, _area_grid, _transform):
    """Population-weighted centroid (lat, lon) per year."""
    h, w = _cube.shape[1], _cube.shape[2]
    rows_idx, cols_idx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    # pixel center coordinates
    lons = _transform.c + (cols_idx + 0.5) * _transform.a
    lats = _transform.f + (rows_idx + 0.5) * _transform.e
    records = []
    for i, y in enumerate(YEARS):
        layer = _cube[i]
        valid = ~np.isnan(layer) & (layer > 0)
        pop = layer[valid] * _area_grid[valid]
        total = float(np.sum(pop))
        if total <= 0:
            records.append({"年份": y, "重心纬度": 0.0, "重心经度": 0.0})
            continue
        clat = float(np.sum(pop * lats[valid]) / total)
        clon = float(np.sum(pop * lons[valid]) / total)
        records.append({"年份": y, "重心纬度": round(clat, 4), "重心经度": round(clon, 4)})
    return pd.DataFrame(records)


@st.cache_data
def compute_sigma_convergence(province_stats, exclude_names):
    """Cross-sectional std dev and CV of provincial mean density, per year."""
    records = []
    for y in YEARS:
        provs = [p for p in province_stats[str(y)] if p["name"] not in exclude_names]
        vals = np.array([p["mean"] for p in provs])
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals))
        cv = std_val / mean_val if mean_val > 0 else 0
        records.append({"年份": y, "省际标准差": round(std_val, 2), "变异系数CV": round(cv, 4), "省际均值": round(mean_val, 2)})
    return pd.DataFrame(records)


@st.cache_data
def compute_beta_convergence(province_stats, exclude_names):
    """OLS: annual growth rate ~ ln(initial density). Returns slope, R², p-value, and scatter data."""
    prov_2002 = {p["name"]: p for p in province_stats["2002"] if p["name"] not in exclude_names}
    prov_2020 = {p["name"]: p for p in province_stats["2020"] if p["name"] not in exclude_names}
    n_years = YEARS[-1] - YEARS[0]
    rows = []
    for name, p0 in prov_2002.items():
        p1 = prov_2020.get(name)
        if not p1 or p0["mean"] <= 0:
            continue
        annual_growth = ((p1["mean"] / p0["mean"]) ** (1.0 / n_years) - 1) * 100
        rows.append({
            "省份": name,
            "ln初始密度": float(np.log(p0["mean"])),
            "年均增长率%": round(annual_growth, 3),
            "区域": REGION_MAP.get(name, "其他"),
        })
    df = pd.DataFrame(rows)
    if len(df) < 3:
        return df, 0.0, 0.0, 1.0
    slope, intercept, r_value, p_value, std_err = sp_stats.linregress(df["ln初始密度"], df["年均增长率%"])
    return df, round(float(slope), 4), round(float(r_value ** 2), 4), float(p_value)


@st.cache_data
def compute_regional_shares(province_stats, exclude_names):
    """Population share by region (东/中/西/东北) per year."""
    records = []
    for y in YEARS:
        provs = [p for p in province_stats[str(y)] if p["name"] not in exclude_names]
        region_pop = {}
        for p in provs:
            r = REGION_MAP.get(p["name"], "其他")
            region_pop[r] = region_pop.get(r, 0) + p["est_pop_wan"]
        total = sum(region_pop.values())
        for r, pop in region_pop.items():
            records.append({
                "年份": y,
                "区域": r,
                "人口(万)": round(pop, 0),
                "人口占比%": round(pop / total * 100, 2) if total > 0 else 0,
            })
    return pd.DataFrame(records)


# [Fix 6] Pixel-level timeseries — stacked cube for instant lookup
@st.cache_data
def load_raster_cube():
    """Load all years into a single (n_years, H, W) array + geo metadata."""
    layers = []
    for y in YEARS:
        path = os.path.join(DATA_DIR, f"China_{y}_5km.tif")
        with rasterio.open(path) as src:
            d = src.read(1).astype(np.float32)
            if not layers:
                transform = src.transform
                h, w = src.height, src.width
        d[d < -1e30] = np.nan
        d[d < 0] = np.nan
        layers.append(d)
    cube = np.stack(layers, axis=0)  # (19, H, W)
    return cube, transform, h, w


@st.cache_data
def get_pixel_timeseries(_cube, _transform, h, w, lat, lng):
    """Extract density values at a lat/lng from the pre-loaded cube."""
    col_idx = int((lng - _transform.c) / _transform.a)
    row_idx = int((lat - _transform.f) / _transform.e)
    if 0 <= row_idx < h and 0 <= col_idx < w:
        vals = _cube[:, row_idx, col_idx]
        return [{"年份": y, "密度": float(v) if not np.isnan(v) else 0.0}
                for y, v in zip(YEARS, vals)]
    return [{"年份": y, "密度": 0.0} for y in YEARS]


# ---------------------------------------------------------------------------
# Load everything
# ---------------------------------------------------------------------------
province_stats = load_province_stats()
geojson_data = load_geojson()
df_prov = build_province_df(province_stats)
raster_cube, cube_transform, cube_h, cube_w = load_raster_cube()
boundary_segments = build_boundary_segments(geojson_data)
ALL_PROVINCE_NAMES = sorted(df_prov["name"].unique())

# Pre-compute pixel area grid for economics indicators
_, _, _res_deg, _lat_top, _ = load_year(YEARS[0])
area_grid = compute_pixel_area_grid(_res_deg, _lat_top, cube_h, cube_w)


# ---------------------------------------------------------------------------
# Helper: filter provinces by scope
# ---------------------------------------------------------------------------
def filter_provs(provs, mainland_only):
    if mainland_only:
        return [p for p in provs if p["name"] not in EXCLUDE_HMT]
    return provs


def filter_names(names, mainland_only):
    if mainland_only:
        return [n for n in names if n not in EXCLUDE_HMT]
    return names


# ---------------------------------------------------------------------------
# Sidebar — global controls
# ---------------------------------------------------------------------------
st.sidebar.markdown("## 🌏 时空探索器")
st.sidebar.caption("中国人口密度 2002-2020")
st.sidebar.markdown("---")

# [Fix 4] Mainland-only toggle
mainland_only = st.sidebar.toggle("仅中国大陆 (不含港澳台)", value=True)

opacity = st.sidebar.slider("🔍 图层透明度", 0.3, 1.0, 0.7, 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("##### 省份钻取")
available_names = filter_names(ALL_PROVINCE_NAMES, mainland_only)
drill_province = st.sidebar.selectbox(
    "选择省份",
    ["(全国总览)"] + available_names,
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small>数据: WorldPop UN-adj 1km<br>"
    "分辨率: ~5km 降采样<br>"
    f"人口: 纬度修正像元面积<br>"
    f"范围: {'中国大陆' if mainland_only else '全国含港澳台'}</small>",
    unsafe_allow_html=True,
)
with st.sidebar.expander("数据来源与许可"):
    st.markdown(
        f"- 官方来源: [WorldPop Global1 2000-2020]({WORLDPOP_COLLECTION_URL})\n"
        f"- 数据类型: [Population Density / UN-adjusted 1km]({WORLDPOP_PROJECT_URL})\n"
        f"- 中国页面示例: [2002]({WORLDPOP_CHINA_2002_URL}) / [2020]({WORLDPOP_CHINA_2020_URL})\n"
        f"- DOI: [{WORLDPOP_DOI_URL}]({WORLDPOP_DOI_URL})\n"
        f"- 许可: [CC BY 4.0]({CC_BY_4_URL})"
    )

# Compute rank once based on scope
exclude_for_rank = EXCLUDE_HMT if mainland_only else set()
scope_key = "mainland" if mainland_only else "all"
df_national = load_national_stats(scope_key)

# ===================================================================
# MAIN CONTENT — economic analysis layout
# ===================================================================
scope_label_readable = "中国大陆" if mainland_only else "全国含港澳台"
rank_df = compute_rank_history(province_stats, exclude_names=frozenset(exclude_for_rank))
df_growth = build_growth_df(frozenset(exclude_for_rank))
focus_default_idx = available_names.index(drill_province) if drill_province in available_names else 0

tab_summary, tab_national, tab_spatial, tab_hetero, tab_research, tab_methods = st.tabs([
    "📌 摘要",
    "📈 全国演变",
    "🗺️ 空间格局",
    "🏛️ 省际异质性",
    "📖 研究课题",
    "📚 数据与方法",
])


# ===================================================================
# TAB 1: Summary — Spatial Economics Narrative
# ===================================================================
with tab_summary:
    st.header("📌 研究摘要：空间集聚、市场潜力与区域收敛")
    st.caption(f"从空间经济学视角解读 {YEARS[0]}-{YEARS[-1]} 年人口密度栅格数据。当前范围：{scope_label_readable}。")

    # --- Compute economics indicators ---
    nat_start = df_national[df_national["年份"] == YEARS[0]].iloc[0]
    nat_end = df_national[df_national["年份"] == YEARS[-1]].iloc[0]
    df_gini = compute_spatial_gini_series(raster_cube, area_grid)
    df_conc10 = compute_concentration_series(raster_cube, area_grid, top_pct=10)
    df_centroid = compute_population_centroid(raster_cube, area_grid, cube_transform)
    beta_df, beta_slope, beta_r2, beta_p = compute_beta_convergence(province_stats, frozenset(exclude_for_rank))

    gini_start = df_gini[df_gini["年份"] == YEARS[0]]["空间基尼系数"].values[0]
    gini_end = df_gini[df_gini["年份"] == YEARS[-1]]["空间基尼系数"].values[0]
    conc_start = df_conc10[df_conc10["年份"] == YEARS[0]][f"Top10%人口占比"].values[0]
    conc_end = df_conc10[df_conc10["年份"] == YEARS[-1]][f"Top10%人口占比"].values[0]
    centroid_start = df_centroid[df_centroid["年份"] == YEARS[0]].iloc[0]
    centroid_end = df_centroid[df_centroid["年份"] == YEARS[-1]].iloc[0]

    # approx distance shift in km
    dlat_km = (centroid_end["重心纬度"] - centroid_start["重心纬度"]) * 111.32
    dlon_km = (centroid_end["重心经度"] - centroid_start["重心经度"]) * 111.32 * np.cos(np.radians(centroid_end["重心纬度"]))
    shift_km = np.sqrt(dlat_km ** 2 + dlon_km ** 2)
    shift_dir = ""
    if dlat_km > 0 and dlon_km > 0:
        shift_dir = "东北"
    elif dlat_km > 0 and dlon_km <= 0:
        shift_dir = "西北"
    elif dlat_km <= 0 and dlon_km > 0:
        shift_dir = "东南"
    else:
        shift_dir = "西南"

    convergence_word = "收敛（追赶效应）" if beta_slope < 0 else "发散（强者恒强）"
    convergence_sig = "显著" if beta_p < 0.05 else "不显著"

    # --- Key findings: spatial economics framing ---
    findings = [
        f"**集聚加速**：空间基尼系数由 {gini_start:.4f} {'升至' if gini_end > gini_start else '降至'} {gini_end:.4f}，"
        f"Top 10% 最密像元的人口份额由 {conc_start:.1f}% {'升至' if conc_end > conc_start else '降至'} {conc_end:.1f}%，"
        f"{'表明集聚力（agglomeration forces）在研究期内持续主导' if gini_end > gini_start else '表明分散力有所增强'}。",

        f"**市场潜力上升**：加权平均密度（market potential 代理指标）由 {nat_start['加权平均密度']:.1f} 增至 "
        f"{nat_end['加权平均密度']:.1f} 人/km²（+{(nat_end['加权平均密度'] - nat_start['加权平均密度'])/nat_start['加权平均密度']*100:.1f}%），"
        f"意味着「平均居民」周围的本地市场规模在扩大。",

        f"**核心区扩展**：高密度像元（>1000 人/km²）由 {int(nat_start['高密度像素数']):,} 增至 {int(nat_end['高密度像素数']):,}，"
        f"核心区在空间上持续扩张，对应 NEG 框架中核心-外围格局的演化。",

        f"**区域{convergence_word}**：β-收敛检验斜率 = {beta_slope:.4f}（R² = {beta_r2:.3f}，{convergence_sig}，p = {beta_p:.3f}），"
        f"{'初始密度低的省份增速更快，呈追赶态势' if beta_slope < 0 else '初始密度高的省份增速更快，空间极化在加剧'}。",

        f"**人口重心漂移**：{YEARS[0]}-{YEARS[-1]} 年重心向{shift_dir}方向移动约 {shift_km:.1f} km，"
        f"反映劳动力流动和 market integration 的宏观方向。",
    ]
    for idx, text in enumerate(findings, 1):
        st.markdown(f"{idx}. {text}")

    # --- KPI row ---
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("空间基尼系数", f"{gini_end:.4f}",
              delta=f"{gini_end - gini_start:+.4f}")
    k2.metric("Top10%人口占比", f"{conc_end:.1f}%",
              delta=f"{conc_end - conc_start:+.1f}%")
    k3.metric("加权平均密度", f"{nat_end['加权平均密度']:.1f}",
              delta=f"{nat_end['加权平均密度'] - nat_start['加权平均密度']:+.1f}")
    k4.metric("β-收敛斜率", f"{beta_slope:.4f}",
              delta="收敛" if beta_slope < 0 else "发散", delta_color="normal" if beta_slope < 0 else "inverse")
    k5.metric("重心漂移", f"{shift_km:.1f} km", delta=f"向{shift_dir}")

    # --- Row 2: Gini trend + diff map ---
    col_trend, col_map = st.columns([3, 2])
    with col_trend:
        fig_summary = make_subplots(specs=[[{"secondary_y": True}]])
        fig_summary.add_trace(go.Scatter(
            x=df_gini["年份"], y=df_gini["空间基尼系数"],
            name="空间基尼系数", mode="lines+markers", line=dict(color="#E31A1C", width=2.5),
        ), secondary_y=False)
        fig_summary.add_trace(go.Scatter(
            x=df_conc10["年份"], y=df_conc10["Top10%人口占比"],
            name="Top10%人口占比(%)", mode="lines+markers", line=dict(color="#3366CC", width=2.5),
        ), secondary_y=True)
        fig_summary.update_layout(
            title="空间集聚动态：基尼系数与人口集中度",
            height=360, margin=dict(t=40, b=20, l=20, r=20),
            hovermode="x unified",
            legend=dict(orientation="h", y=1.1),
        )
        fig_summary.update_yaxes(title_text="基尼系数", secondary_y=False)
        fig_summary.update_yaxes(title_text="Top10% 占比 (%)", secondary_y=True)
        st.plotly_chart(fig_summary, use_container_width=True)

    with col_map:
        diff_preview, diff_p99, _, _, diff_mean = render_diff_frame_image(YEARS[0], YEARS[-1], opacity)
        st.image(diff_preview, use_container_width=True, caption=f"{YEARS[0]}→{YEARS[-1]} 空间再配置地图")
        st.caption(f"红色 = 人口流入区，蓝色 = 流出区。P99 对称色标 ±{diff_p99:.0f} 人/km²，像元平均变化 {diff_mean:+.1f}。")

    # --- Row 3: β-convergence scatter + top movers ---
    col_beta, col_move = st.columns([3, 2])
    with col_beta:
        fig_beta = go.Figure()
        for region in ["东部", "中部", "西部", "东北"]:
            sub = beta_df[beta_df["区域"] == region]
            fig_beta.add_trace(go.Scatter(
                x=sub["ln初始密度"], y=sub["年均增长率%"],
                mode="markers+text", name=region,
                marker=dict(size=10, color=REGION_COLORS.get(region, "#999")),
                text=sub["省份"].str.replace("省|市|自治区|壮族|维吾尔|回族", "", regex=True),
                textposition="top center", textfont_size=8,
            ))
        # OLS line
        x_range = np.linspace(float(beta_df["ln初始密度"].min()) - 0.3, float(beta_df["ln初始密度"].max()) + 0.3, 50)
        sl, intc, _, _, _ = sp_stats.linregress(beta_df["ln初始密度"], beta_df["年均增长率%"])
        fig_beta.add_trace(go.Scatter(
            x=x_range, y=intc + sl * x_range,
            mode="lines", name=f"OLS (β={sl:.4f})",
            line=dict(color="grey", dash="dash", width=2),
        ))
        fig_beta.update_layout(
            title=f"β-收敛检验：ln(初始密度) vs 年均增长率 (R²={beta_r2:.3f})",
            height=400, margin=dict(t=45, b=20, l=20, r=20),
            xaxis_title=f"ln({YEARS[0]}年平均密度)",
            yaxis_title="年均增长率 (%)",
            legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(fig_beta, use_container_width=True)

    with col_move:
        st.markdown("**人口集聚最快的省份**")
        movers = df_growth.nlargest(8, "绝对增长")
        for _, row in movers.iterrows():
            region_tag = REGION_MAP.get(row["省份"], "")
            st.markdown(f"- **{row['省份']}** ({region_tag}) — +{row['绝对增长']:.0f} 人/km²，增长率 {row['增长率%']:+.1f}%")
        st.markdown("---")
        st.markdown("**人口流失最快的省份**")
        losers = df_growth.nsmallest(5, "绝对增长")
        for _, row in losers.iterrows():
            region_tag = REGION_MAP.get(row["省份"], "")
            st.markdown(f"- **{row['省份']}** ({region_tag}) — {row['绝对增长']:+.0f} 人/km²，增长率 {row['增长率%']:+.1f}%")


# ===================================================================
# TAB 2: National evolution — Agglomeration Dynamics
# ===================================================================
with tab_national:
    st.header("📈 集聚动态与市场规模演化")
    st.caption(f"追踪 {YEARS[0]}-{YEARS[-1]} 年集聚力的时间演变。上半部分为空间再配置动画，下半部分为集聚指标时间序列。")

    # --- Animation section (preserved) ---
    ctrl_cols = st.columns([1, 1, 1, 4])
    with ctrl_cols[0]:
        play_btn = st.button("▶️ 播放累计变化", key="national_anim_play")
    with ctrl_cols[1]:
        anim_speed = st.selectbox("速度", [0.3, 0.5, 1.0, 1.5], index=1, format_func=lambda x: f"{x}s/帧", key="national_anim_speed")
    with ctrl_cols[2]:
        manual_year = st.select_slider("年份", options=YEARS, value=2010, key="national_anim_manual")

    cumulative_stats = precompute_cumulative_diff_stats(YEARS[0])
    col_amap, col_astats = st.columns([5, 3])
    map_placeholder = col_amap.empty()
    stats_placeholder = col_astats.empty()

    def render_national_frame(yr):
        frame_img, p99_val, clipped_n, total_n, _ = render_diff_frame_image(YEARS[0], yr, opacity)
        s = cumulative_stats[yr]
        map_placeholder.image(frame_img, caption=f"{yr} 年累计变化（相对 {YEARS[0]} 年）", use_container_width=True)
        with stats_placeholder.container():
            progress = (yr - YEARS[0]) / (YEARS[-1] - YEARS[0])
            st.progress(progress, text=f"**{yr}** / {YEARS[-1]}")
            nat_row_yr = df_national[df_national["年份"] == yr].iloc[0]
            c1, c2 = st.columns(2)
            c1.metric("累计平均变化", f"{s['mean_change']:+.1f} 人/km²")
            c2.metric("最大累计增幅", f"{s['max_increase']:+.0f}")
            c3, c4 = st.columns(2)
            c3.metric("增长区域占比", f"{s['increase_share']:.1f}%")
            c4.metric("下降区域占比", f"{s['decrease_share']:.1f}%")
            st.caption(
                f"当前总人口 {nat_row_yr['估算总人口(万)']/10000:.2f} 亿，"
                f"加权平均密度 {nat_row_yr['加权平均密度']:.1f} 人/km²；"
                f"差值色标约为 ±{p99_val:.0f} 人/km²，裁剪 {clipped_n/total_n*100:.1f}% 像素。"
            )
            yr_provs = sorted(filter_provs(province_stats[str(yr)], mainland_only), key=lambda x: x["mean"], reverse=True)
            st.markdown(f"**{yr} 年高密度 Top 5 省份**")
            for i, p in enumerate(yr_provs[:5], 1):
                st.markdown(f"{i}. **{p['name']}** — {p['mean']:.0f} 人/km²")

    if play_btn:
        for yr in YEARS:
            render_national_frame(yr)
            _time.sleep(anim_speed)
    else:
        render_national_frame(manual_year)

    # --- Agglomeration indicators panel ---
    st.markdown("---")
    st.subheader("集聚指标时间序列")

    col_gini, col_conc = st.columns(2)
    with col_gini:
        fig_gini = go.Figure()
        fig_gini.add_trace(go.Scatter(
            x=df_gini["年份"], y=df_gini["空间基尼系数"],
            mode="lines+markers", line=dict(color="#E31A1C", width=2.5),
            fill="tozeroy", fillcolor="rgba(227,26,28,0.08)",
        ))
        fig_gini.update_layout(
            title="空间基尼系数（人口分布不平等度）",
            height=320, margin=dict(t=40, b=20, l=20, r=20),
            xaxis_title="年份", yaxis_title="基尼系数",
            hovermode="x unified",
        )
        st.plotly_chart(fig_gini, use_container_width=True)
        st.caption("基尼系数越高，人口在空间上越集中。趋势上升 = 集聚力主导。")

    with col_conc:
        fig_conc = go.Figure()
        fig_conc.add_trace(go.Scatter(
            x=df_conc10["年份"], y=df_conc10["Top10%人口占比"],
            mode="lines+markers", line=dict(color="#3366CC", width=2.5),
            fill="tozeroy", fillcolor="rgba(51,102,204,0.08)",
        ))
        fig_conc.update_layout(
            title="Top 10% 最密像元的人口集中度",
            height=320, margin=dict(t=40, b=20, l=20, r=20),
            xaxis_title="年份", yaxis_title="占总人口比例 (%)",
            hovermode="x unified",
        )
        st.plotly_chart(fig_conc, use_container_width=True)
        st.caption("该比例上升意味着更多人口进入少数高密度区域——市场规模效应的直接体现。")

    col_mp, col_sigma = st.columns(2)
    with col_mp:
        fig_mp = make_subplots(specs=[[{"secondary_y": True}]])
        fig_mp.add_trace(go.Scatter(
            x=df_national["年份"], y=df_national["加权平均密度"],
            name="加权平均密度 (Market Potential 代理)", mode="lines+markers",
            line=dict(color="#E31A1C", width=2.5),
        ), secondary_y=False)
        fig_mp.add_trace(go.Scatter(
            x=df_national["年份"], y=df_national["高密度像素数"],
            name="高密度像元数 (核心区规模)", mode="lines+markers",
            line=dict(color="#FF7F00", width=2),
        ), secondary_y=True)
        fig_mp.update_layout(
            title="市场潜力代理指标与核心区规模",
            height=320, margin=dict(t=40, b=20, l=20, r=20),
            hovermode="x unified", legend=dict(orientation="h", y=1.12),
        )
        fig_mp.update_yaxes(title_text="人/km²", secondary_y=False)
        fig_mp.update_yaxes(title_text="像元数", secondary_y=True)
        st.plotly_chart(fig_mp, use_container_width=True)

    with col_sigma:
        df_sigma = compute_sigma_convergence(province_stats, frozenset(exclude_for_rank))
        fig_sigma = go.Figure()
        fig_sigma.add_trace(go.Scatter(
            x=df_sigma["年份"], y=df_sigma["变异系数CV"],
            mode="lines+markers", line=dict(color="#33A02C", width=2.5),
            fill="tozeroy", fillcolor="rgba(51,160,44,0.08)",
        ))
        fig_sigma.update_layout(
            title="σ-收敛：省际密度变异系数 (CV)",
            height=320, margin=dict(t=40, b=20, l=20, r=20),
            xaxis_title="年份", yaxis_title="变异系数",
            hovermode="x unified",
        )
        st.plotly_chart(fig_sigma, use_container_width=True)
        st.caption("CV 下降 = σ-收敛（省际差异缩小）；CV 上升 = σ-发散（差异扩大）。")


# ===================================================================
# TAB 3: Spatial patterns — Core-Periphery & Spatial Reallocation
# ===================================================================
with tab_spatial:
    st.header("🗺️ 核心-外围格局与空间再配置")
    spatial_single, spatial_compare, spatial_centroid = st.tabs(["单年格局", "两期对比", "人口重心与区域份额"])

    with spatial_single:
        year_sp = st.slider("📅 年份", YEARS[0], YEARS[-1], 2010, 1, key="spatial_single_year")
        col_map, col_side = st.columns([5, 3])

        with col_map:
            data_sp, bounds_sp, _, _, _ = load_year(year_sp)
            img_sp = raster_to_image(data_sp)
            m = folium.Map(location=[35.0, 105.0], zoom_start=4, tiles="CartoDB positron")
            folium.raster_layers.ImageOverlay(image=img_sp, bounds=bounds_sp, opacity=opacity, name="栅格密度").add_to(m)
            folium.GeoJson(
                geojson_data,
                style_function=lambda x: {"fillColor": "transparent", "color": "#333", "weight": 0.8, "fillOpacity": 0},
            ).add_to(m)
            map_result = st_folium(m, height=520, use_container_width=True, key="spatial_single_map")

            if map_result and map_result.get("last_clicked"):
                lat_c = map_result["last_clicked"]["lat"]
                lng_c = map_result["last_clicked"]["lng"]
                ts = get_pixel_timeseries(raster_cube, cube_transform, cube_h, cube_w, lat_c, lng_c)
                fig_click = go.Figure(go.Scatter(
                    x=[t["年份"] for t in ts], y=[t["密度"] for t in ts],
                    mode="lines+markers", line=dict(color="#E31A1C", width=2),
                    fill="tozeroy", fillcolor="rgba(227,26,28,0.1)",
                ))
                fig_click.update_layout(
                    title=f"点击位置 ({lat_c:.3f}, {lng_c:.3f}) 的历年密度",
                    height=220, margin=dict(t=40, b=20, l=20, r=20),
                    xaxis_title="年份", yaxis_title="人/km²",
                )
                st.plotly_chart(fig_click, use_container_width=True)

        with col_side:
            year_provs = sorted(filter_provs(province_stats[str(year_sp)], mainland_only), key=lambda x: x["mean"], reverse=True)
            fig_rank = go.Figure(go.Bar(
                y=[p["name"] for p in year_provs[:15]][::-1],
                x=[p["mean"] for p in year_provs[:15]][::-1],
                orientation="h",
                marker=dict(color=[p["mean"] for p in year_provs[:15]][::-1], colorscale="YlOrRd"),
                text=[f"{p['mean']:.0f}" for p in year_provs[:15]][::-1],
                textposition="outside",
            ))
            fig_rank.update_layout(height=320, margin=dict(t=20, b=10, l=10, r=50), title="省级密度排名 Top 15", xaxis_title="人/km²")
            st.plotly_chart(fig_rank, use_container_width=True)

            valid_sp = data_sp[~np.isnan(data_sp) & (data_sp > 0)]
            fig_hist = px.histogram(
                x=valid_sp[valid_sp <= np.percentile(valid_sp, 99.5)],
                nbins=60,
                color_discrete_sequence=["#4E79A7"],
                title="像元密度分布（截取 P99.5）",
            )
            fig_hist.update_layout(height=220, margin=dict(t=35, b=20, l=20, r=20), xaxis_title="人/km²", yaxis_title="像元数")
            st.plotly_chart(fig_hist, use_container_width=True)

    with spatial_compare:
        col_y1, col_y2 = st.columns(2)
        with col_y1:
            year_a = st.selectbox("起始年份", YEARS, index=0, key="spatial_cmp_a")
        with col_y2:
            year_b = st.selectbox("结束年份", YEARS, index=len(YEARS) - 1, key="spatial_cmp_b")

        if year_a != year_b:
            data_a, bounds, _, _, _ = load_year(year_a)
            data_b, _, _, _, _ = load_year(year_b)
            diff = data_b - data_a
            valid_diff = diff[~np.isnan(diff)]
            _, p99_val, clipped_n, total_n = diff_to_image_adaptive(diff)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("增长区域", f"{np.sum(valid_diff > 0) / len(valid_diff) * 100:.1f}%")
            c2.metric("下降区域", f"{np.sum(valid_diff < 0) / len(valid_diff) * 100:.1f}%")
            c3.metric("平均变化", f"{np.nanmean(valid_diff):+.1f} 人/km²")
            c4.metric("P99色标", f"±{p99_val:.0f}")

            col_ma, col_diff, col_mb = st.columns(3)
            with col_ma:
                st.image(render_animation_frame_image(year_a, opacity), use_container_width=True, caption=f"{year_a} 年")
            with col_diff:
                st.image(render_diff_frame_image(year_a, year_b, opacity)[0], use_container_width=True, caption=f"{year_a}→{year_b} 差值图")
                st.caption(f"超出裁剪: {clipped_n:,} 像素 ({clipped_n/total_n*100:.1f}%)")
            with col_mb:
                st.image(render_animation_frame_image(year_b, opacity), use_container_width=True, caption=f"{year_b} 年")

            st.subheader("省级变化分析")
            col_bar, col_hist = st.columns([3, 2])
            prov_a = {p["name"]: p for p in filter_provs(province_stats[str(year_a)], mainland_only)}
            prov_b = {p["name"]: p for p in filter_provs(province_stats[str(year_b)], mainland_only)}
            prov_diff_list = []
            for name in prov_a:
                if name in prov_b:
                    change = prov_b[name]["mean"] - prov_a[name]["mean"]
                    pct = (change / prov_a[name]["mean"] * 100) if prov_a[name]["mean"] > 0 else 0
                    pop_change = prov_b[name]["est_pop_wan"] - prov_a[name]["est_pop_wan"]
                    prov_diff_list.append({"省份": name, "密度变化": round(change, 1), "变化率%": round(pct, 1), "人口变化(万)": round(pop_change, 0)})
            prov_diff_list.sort(key=lambda x: x["密度变化"], reverse=True)

            with col_bar:
                fig_prov_diff = go.Figure(go.Bar(
                    y=[p["省份"] for p in prov_diff_list],
                    x=[p["密度变化"] for p in prov_diff_list],
                    orientation="h",
                    marker_color=["#B2182B" if p["密度变化"] > 0 else "#2166AC" for p in prov_diff_list],
                    text=[f"{p['密度变化']:+.0f}" for p in prov_diff_list],
                    textposition="outside",
                ))
                fig_prov_diff.add_vline(x=0, line_color="grey")
                fig_prov_diff.update_layout(
                    title="各省平均密度变化",
                    height=max(400, len(prov_diff_list) * 22),
                    margin=dict(t=40, b=10, l=10, r=50),
                    xaxis_title="人/km²",
                )
                st.plotly_chart(fig_prov_diff, use_container_width=True)

            with col_hist:
                clipped_hist = valid_diff[(valid_diff > -p99_val * 1.5) & (valid_diff < p99_val * 1.5)]
                fig_hist = px.histogram(x=clipped_hist, nbins=100, title="像元变化分布", color_discrete_sequence=["#3366CC"])
                fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
                fig_hist.update_layout(height=300, margin=dict(t=35, b=20, l=20, r=20), xaxis_title="人/km²", yaxis_title="像元数")
                st.plotly_chart(fig_hist, use_container_width=True)
                st.dataframe(pd.DataFrame(prov_diff_list).head(8), use_container_width=True, hide_index=True)
        else:
            st.warning("请选择两个不同年份。")

    with spatial_centroid:
        st.subheader("人口重心轨迹")
        st.caption("人口重心 = Σ(人口×坐标) / Σ人口。重心漂移方向反映劳动力流动和 market integration 的宏观趋势。")

        col_centroid_map, col_centroid_info = st.columns([3, 2])
        with col_centroid_map:
            fig_centroid = go.Figure()
            fig_centroid.add_trace(go.Scattergeo(
                lon=df_centroid["重心经度"], lat=df_centroid["重心纬度"],
                mode="lines+markers+text",
                marker=dict(
                    size=df_centroid["年份"].apply(lambda y: 6 + (y - YEARS[0]) * 0.8),
                    color=df_centroid["年份"], colorscale="YlOrRd",
                    colorbar=dict(title="年份"),
                ),
                text=df_centroid["年份"].astype(str),
                textposition="top right", textfont_size=8,
                line=dict(color="#333", width=1.5),
                hovertemplate="年份: %{text}<br>经度: %{lon:.4f}<br>纬度: %{lat:.4f}<extra></extra>",
            ))
            fig_centroid.update_geos(
                resolution=50,
                showcountries=True, countrycolor="grey",
                showsubunits=True,
                lonaxis_range=[100, 125], lataxis_range=[25, 42],
                projection_type="mercator",
            )
            fig_centroid.update_layout(
                title=f"人口重心轨迹 ({YEARS[0]}-{YEARS[-1]})",
                height=450, margin=dict(t=40, b=10, l=10, r=10),
                geo=dict(bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_centroid, use_container_width=True)

        with col_centroid_info:
            st.markdown(f"**起点** ({YEARS[0]}): ({centroid_start['重心纬度']:.4f}°N, {centroid_start['重心经度']:.4f}°E)")
            st.markdown(f"**终点** ({YEARS[-1]}): ({centroid_end['重心纬度']:.4f}°N, {centroid_end['重心经度']:.4f}°E)")
            st.markdown(f"**漂移方向**: 向{shift_dir}")
            st.markdown(f"**漂移距离**: {shift_km:.1f} km")
            st.markdown(f"- 纬度方向: {abs(dlat_km):.1f} km ({'北' if dlat_km > 0 else '南'})")
            st.markdown(f"- 经度方向: {abs(dlon_km):.1f} km ({'东' if dlon_km > 0 else '西'})")

            st.markdown("---")
            st.markdown("**经济含义**")
            st.markdown(
                "人口重心向某方向漂移，表明该方向的劳动力吸引力（就业机会、公共服务、市场规模）更强。"
                "在 market integration 框架下，重心漂移方向与贸易成本下降最快的方向一致。"
            )

            # Centroid time series
            fig_latlon = make_subplots(specs=[[{"secondary_y": True}]])
            fig_latlon.add_trace(go.Scatter(
                x=df_centroid["年份"], y=df_centroid["重心纬度"],
                name="纬度", mode="lines+markers", line=dict(color="#E31A1C", width=2),
            ), secondary_y=False)
            fig_latlon.add_trace(go.Scatter(
                x=df_centroid["年份"], y=df_centroid["重心经度"],
                name="经度", mode="lines+markers", line=dict(color="#3366CC", width=2),
            ), secondary_y=True)
            fig_latlon.update_layout(
                title="重心坐标时间序列", height=250,
                margin=dict(t=35, b=20, l=20, r=20),
                hovermode="x unified", legend=dict(orientation="h", y=1.15),
            )
            fig_latlon.update_yaxes(title_text="°N", secondary_y=False)
            fig_latlon.update_yaxes(title_text="°E", secondary_y=True)
            st.plotly_chart(fig_latlon, use_container_width=True)

        st.markdown("---")
        st.subheader("四大区域人口份额演变")
        st.caption("东部/中部/西部/东北的人口份额变化，反映区域间 market access 的相对变化。")

        df_regions = compute_regional_shares(province_stats, frozenset(exclude_for_rank))
        col_area, col_table = st.columns([3, 2])
        with col_area:
            fig_region = go.Figure()
            for region in ["东部", "中部", "西部", "东北"]:
                sub = df_regions[df_regions["区域"] == region]
                fig_region.add_trace(go.Scatter(
                    x=sub["年份"], y=sub["人口占比%"],
                    name=region, mode="lines+markers",
                    line=dict(color=REGION_COLORS.get(region, "#999"), width=2.5),
                    stackgroup="one" if False else None,
                ))
            fig_region.update_layout(
                title="四大区域人口份额",
                height=380, margin=dict(t=40, b=20, l=20, r=20),
                xaxis_title="年份", yaxis_title="占总人口比例 (%)",
                hovermode="x unified",
                legend=dict(orientation="h", y=1.1),
            )
            st.plotly_chart(fig_region, use_container_width=True)

        with col_table:
            pivot = df_regions.pivot_table(index="年份", columns="区域", values="人口占比%")
            pivot = pivot.reindex(columns=["东部", "中部", "西部", "东北"])
            # Show start and end
            st.markdown("**区域份额对比**")
            summary_rows = []
            for region in ["东部", "中部", "西部", "东北"]:
                r_start = pivot.loc[YEARS[0], region]
                r_end = pivot.loc[YEARS[-1], region]
                summary_rows.append({
                    "区域": region,
                    f"{YEARS[0]}占比%": f"{r_start:.2f}",
                    f"{YEARS[-1]}占比%": f"{r_end:.2f}",
                    "变化": f"{r_end - r_start:+.2f}",
                })
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("**解读**")
            east_change = float(pivot.loc[YEARS[-1], "东部"]) - float(pivot.loc[YEARS[0], "东部"])
            if east_change > 0:
                st.markdown(f"东部份额上升 {east_change:+.2f} 个百分点，核心区对劳动力的吸引持续增强，与 NEG 预测的核心-外围集聚一致。")
            else:
                st.markdown(f"东部份额下降 {east_change:+.2f} 个百分点，中西部人口回流或就地城镇化趋势增强。")


# ===================================================================
# TAB 4: Provincial heterogeneity — Regional Convergence
# ===================================================================
with tab_hetero:
    st.header("🏛️ 区域收敛与增长分异")
    st.caption("核心问题：初始密度低的省份是否增长更快（收敛）？空间经济格局在趋同还是极化？")

    col_focus, col_peer, col_year = st.columns([2, 2, 1.3])
    with col_focus:
        focus_province = st.selectbox("焦点省份", available_names, index=focus_default_idx, key="hetero_focus")
    with col_peer:
        peer_candidates = [n for n in available_names if n != focus_province]
        peer_default = peer_candidates.index("江苏省") if "江苏省" in peer_candidates else 0
        compare_province = st.selectbox("对照省份", peer_candidates, index=peer_default, key="hetero_peer")
    with col_year:
        year_dr = st.slider("焦点年份", YEARS[0], YEARS[-1], 2010, 1, key="hetero_year")

    scatter_mode = st.radio(
        "增长轨迹视图",
        ["终点散点图", "按年份动态演变"],
        horizontal=True,
        key="hetero_scatter_mode",
    )

    if scatter_mode == "终点散点图":
        scatter_df = df_growth.copy()
        scatter_df["区域"] = scatter_df["省份"].map(REGION_MAP).fillna("其他")
        label_set = set(scatter_df.nlargest(5, "2020人口(万)")["省份"])
        label_set.update(scatter_df.nlargest(3, "增长率%")["省份"])
        label_set.update(scatter_df.nsmallest(3, "增长率%")["省份"])
        label_set.add(focus_province)
        scatter_df["标签"] = scatter_df["省份"].str.replace("省|市|自治区|壮族|维吾尔|回族", "", regex=True).where(scatter_df["省份"].isin(label_set), "")

        fig_traj = go.Figure()
        for region in ["东部", "中部", "西部", "东北"]:
            sub = scatter_df[scatter_df["区域"] == region]
            fig_traj.add_trace(go.Scatter(
                x=sub["2002年密度"], y=sub["增长率%"],
                mode="markers+text", name=region,
                marker=dict(
                    size=sub["2020人口(万)"].apply(lambda x: max(8, min(50, x / 300))),
                    color=REGION_COLORS.get(region, "#999"),
                    opacity=0.8,
                ),
                text=sub["标签"], textposition="top center", textfont_size=8,
                hovertemplate="%{customdata[0]}<br>2002密度: %{x:.0f}<br>增长率: %{y:.1f}%<br>2020人口: %{customdata[1]:.0f}万<extra></extra>",
                customdata=list(zip(sub["省份"], sub["2020人口(万)"])),
            ))

        # Highlight focus province
        focus_row = scatter_df[scatter_df["省份"] == focus_province].iloc[0]
        fig_traj.add_trace(go.Scatter(
            x=[focus_row["2002年密度"]], y=[focus_row["增长率%"]],
            mode="markers",
            marker=dict(size=22, color="rgba(0,0,0,0)", line=dict(color="black", width=2.5)),
            name="焦点省份", showlegend=False, hoverinfo="skip",
        ))

        # OLS regression line (β-convergence visual)
        x_log = np.log(scatter_df["2002年密度"].values)
        y_vals = scatter_df["增长率%"].values
        sl_h, intc_h, _, _, _ = sp_stats.linregress(x_log, y_vals)
        x_fit = np.linspace(scatter_df["2002年密度"].min() * 0.8, scatter_df["2002年密度"].max() * 1.2, 50)
        y_fit = intc_h + sl_h * np.log(x_fit)
        fig_traj.add_trace(go.Scatter(
            x=x_fit, y=y_fit, mode="lines",
            name=f"OLS趋势 (β={sl_h:.3f})",
            line=dict(color="grey", dash="dash", width=2),
        ))

        fig_traj.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5)
        fig_traj.update_layout(
            title=f"β-收敛散点图：初始密度 vs 增长率（按区域着色）",
            height=470, margin=dict(t=45, b=20, l=20, r=20),
            xaxis_title="2002年平均密度 (人/km², 对数坐标)", yaxis_title="2002→2020 增长率 (%)",
            xaxis_type="log",
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig_traj, use_container_width=True)
        st.caption(f"OLS 斜率 β = {sl_h:.4f}：{'负斜率表示低密度省份增速更快（收敛/追赶）' if sl_h < 0 else '正斜率表示高密度省份增速更快（发散/极化）'}。区域着色揭示地理分组的收敛模式。")
    else:
        scatter_anim_df = build_growth_animation_df(frozenset(exclude_for_rank))
        x_min = float(scatter_anim_df["2002年密度"].min())
        x_max = float(scatter_anim_df["2002年密度"].max())
        y_min = float(scatter_anim_df["增长率%"].min())
        y_max = float(scatter_anim_df["增长率%"].max())
        color_abs = max(abs(float(scatter_anim_df["绝对增长"].min())), abs(float(scatter_anim_df["绝对增长"].max())))

        fig_traj_anim = px.scatter(
            scatter_anim_df,
            x="2002年密度",
            y="增长率%",
            animation_frame="年份",
            animation_group="省份",
            size="当年人口(万)",
            size_max=45,
            color="绝对增长",
            color_continuous_scale="RdYlGn",
            range_x=[x_min * 0.85, x_max * 1.15],
            range_y=[y_min - 5, y_max + 5],
            hover_name="省份",
            hover_data={"2002年密度": ":.0f", "当年密度": ":.0f", "增长率%": ":.1f", "当年人口(万)": ":.0f"},
        )
        fig_traj_anim.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5)
        fig_traj_anim.add_vline(x=float(df_growth["2002年密度"].median()), line_dash="dot", line_color="#999", opacity=0.3)
        fig_traj_anim.update_layout(
            title=f"增长轨迹：按年份动态演变（相对 {YEARS[0]} 年）",
            height=500, margin=dict(t=45, b=20, l=20, r=20),
            xaxis_title="2002年平均密度 (人/km²)",
            yaxis_title=f"相对 {YEARS[0]} 年增长率 (%)",
            xaxis_type="log",
            coloraxis_colorbar_title="绝对增长",
            legend_title_text="",
        )
        fig_traj_anim.update_coloraxes(cmin=-color_abs, cmax=color_abs)
        if fig_traj_anim.layout.updatemenus:
            fig_traj_anim.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 700
            fig_traj_anim.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 350
        st.plotly_chart(fig_traj_anim, use_container_width=True)
        st.caption("动画模式更适合观察轨迹与路径依赖；终点散点图仍然更适合做静态比较与标注。")

    matrix_metric = st.radio("热力矩阵指标", ["平均密度", "相对2002年增长率%", "逐年变化"], horizontal=True, key="hetero_matrix_metric")
    provs_2020 = sorted(filter_provs(province_stats["2020"], mainland_only), key=lambda x: x["mean"], reverse=True)
    prov_names_ordered = [p["name"] for p in provs_2020]
    z_matrix = []
    for pname in prov_names_ordered:
        row = []
        base_val = prev_val = None
        for y in YEARS:
            val = next((p["mean"] for p in province_stats[str(y)] if p["name"] == pname), 0)
            if base_val is None:
                base_val = val
            if matrix_metric == "平均密度":
                row.append(round(val, 1))
            elif matrix_metric == "相对2002年增长率%":
                row.append(round((val - base_val) / base_val * 100, 1) if base_val > 0 else 0)
            else:
                row.append(round(val - prev_val, 1) if prev_val is not None else 0)
            prev_val = val
        z_matrix.append(row)

    if matrix_metric == "平均密度":
        colorscale, zmin, zmax, bar_title = "YlOrRd", 0, 1200, "人/km²"
    elif matrix_metric == "相对2002年增长率%":
        zmax = max(abs(v) for row in z_matrix for v in row)
        colorscale, zmin, zmax, bar_title = "RdYlGn", -zmax * 0.3, zmax, "%"
    else:
        zmax = max(abs(v) for row in z_matrix for v in row if abs(v) < 500)
        colorscale, zmin, zmax, bar_title = "RdBu_r", -zmax, zmax, "人/km²"

    fig_hm = go.Figure(go.Heatmap(
        z=z_matrix, x=YEARS, y=prov_names_ordered, colorscale=colorscale, zmin=zmin, zmax=zmax,
        colorbar_title=bar_title, hovertemplate="省份: %{y}<br>年份: %{x}<br>值: %{z}<extra></extra>",
    ))
    fig_hm.update_layout(height=max(500, len(prov_names_ordered) * 20), margin=dict(t=20, b=20, l=10, r=10), xaxis=dict(dtick=1), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_hm, use_container_width=True)

    # --- σ-convergence detail ---
    st.markdown("---")
    st.subheader("σ-收敛检验：省际离散度的时间趋势")
    col_sigma_detail, col_sigma_text = st.columns([3, 2])
    with col_sigma_detail:
        df_sigma_h = compute_sigma_convergence(province_stats, frozenset(exclude_for_rank))
        fig_sigma_h = make_subplots(specs=[[{"secondary_y": True}]])
        fig_sigma_h.add_trace(go.Scatter(
            x=df_sigma_h["年份"], y=df_sigma_h["省际标准差"],
            name="省际标准差", mode="lines+markers", line=dict(color="#E31A1C", width=2.5),
        ), secondary_y=False)
        fig_sigma_h.add_trace(go.Scatter(
            x=df_sigma_h["年份"], y=df_sigma_h["变异系数CV"],
            name="变异系数 CV", mode="lines+markers", line=dict(color="#3366CC", width=2.5),
        ), secondary_y=True)
        fig_sigma_h.update_layout(
            title="σ-收敛：省际密度标准差与变异系数",
            height=320, margin=dict(t=40, b=20, l=20, r=20),
            hovermode="x unified", legend=dict(orientation="h", y=1.12),
        )
        fig_sigma_h.update_yaxes(title_text="标准差 (人/km²)", secondary_y=False)
        fig_sigma_h.update_yaxes(title_text="CV", secondary_y=True)
        st.plotly_chart(fig_sigma_h, use_container_width=True)
    with col_sigma_text:
        cv_start = df_sigma_h[df_sigma_h["年份"] == YEARS[0]]["变异系数CV"].values[0]
        cv_end = df_sigma_h[df_sigma_h["年份"] == YEARS[-1]]["变异系数CV"].values[0]
        std_start = df_sigma_h[df_sigma_h["年份"] == YEARS[0]]["省际标准差"].values[0]
        std_end = df_sigma_h[df_sigma_h["年份"] == YEARS[-1]]["省际标准差"].values[0]
        st.markdown("**σ-收敛判断**")
        st.markdown(f"- 变异系数 CV: {cv_start:.4f} → {cv_end:.4f} ({'↓ 收敛' if cv_end < cv_start else '↑ 发散'})")
        st.markdown(f"- 标准差: {std_start:.2f} → {std_end:.2f} ({'↓ 缩小' if std_end < std_start else '↑ 扩大'})")
        st.markdown("---")
        st.markdown("**经济含义**")
        if cv_end < cv_start:
            st.markdown("CV 下降表明省际人口密度差异在缩小（σ-收敛），可能反映劳动力流动降低了区域间的密度差距，与 market integration 促进要素趋同一致。")
        else:
            st.markdown("CV 上升表明省际人口密度差异在扩大（σ-发散），集聚力持续将人口从低密度省份吸引至高密度省份，空间极化在加剧。")

    st.markdown("---")
    st.subheader(f"{focus_province} · 深度剖面")

    prov_series = df_prov[df_prov["name"] == focus_province].sort_values("年份")
    prov_current = prov_series[prov_series["年份"] == year_dr].iloc[0]
    first_rank = rank_df[(rank_df["省份"] == focus_province) & (rank_df["年份"] == YEARS[0])]["排名"].values[0]
    cur_rank = rank_df[(rank_df["省份"] == focus_province) & (rank_df["年份"] == year_dr)]["排名"].values[0]
    total_growth = prov_series.iloc[-1]["mean"] - prov_series.iloc[0]["mean"]
    growth_rate = (total_growth / prov_series.iloc[0]["mean"] * 100) if prov_series.iloc[0]["mean"] > 0 else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(f"{year_dr}年平均密度", f"{prov_current['mean']:.1f} 人/km²")
    k2.metric("估算人口", f"{prov_current['est_pop_wan']/10000:.2f} 亿")
    k3.metric("全国排名", f"第 {cur_rank} 名", delta=f"{first_rank - cur_rank:+d} (vs {YEARS[0]})")
    k4.metric("总增长", f"{total_growth:+.1f} 人/km²")
    k5.metric("增长率", f"{growth_rate:+.1f}%")

    col_ts, col_rank = st.columns([3, 2])
    with col_ts:
        fig_detail = make_subplots(specs=[[{"secondary_y": True}]])
        fig_detail.add_trace(go.Scatter(
            x=prov_series["年份"], y=prov_series["mean"],
            name="平均密度", mode="lines+markers",
            line=dict(color="#E31A1C", width=2.5),
            fill="tozeroy", fillcolor="rgba(227,26,28,0.1)",
        ), secondary_y=False)
        fig_detail.add_trace(go.Scatter(
            x=prov_series["年份"], y=prov_series["est_pop_wan"] / 10000,
            name="估算人口(亿)", mode="lines+markers",
            line=dict(color="#3366CC", width=2),
        ), secondary_y=True)
        fig_detail.add_vline(x=year_dr, line_dash="dash", line_color="grey")
        fig_detail.update_layout(title=f"{focus_province} 历年趋势", height=320, margin=dict(t=40, b=20, l=20, r=20), hovermode="x unified", legend=dict(orientation="h", y=1.1))
        fig_detail.update_yaxes(title_text="人/km²", secondary_y=False)
        fig_detail.update_yaxes(title_text="亿", secondary_y=True)
        st.plotly_chart(fig_detail, use_container_width=True)

    with col_rank:
        prov_rank_data = rank_df[rank_df["省份"] == focus_province]
        fig_bump = go.Figure(go.Scatter(
            x=prov_rank_data["年份"], y=prov_rank_data["排名"],
            mode="lines+markers+text",
            text=[f"#{r}" for r in prov_rank_data["排名"]],
            textposition="top center",
            line=dict(color="#E31A1C", width=3), marker=dict(size=8),
        ))
        fig_bump.update_layout(title=f"{focus_province} 排名变化", height=320, margin=dict(t=40, b=20, l=20, r=20), yaxis=dict(autorange="reversed", dtick=1, title="排名"))
        st.plotly_chart(fig_bump, use_container_width=True)

    col_stats, col_compare = st.columns([2, 3])
    with col_stats:
        st.dataframe(
            pd.DataFrame({
                "指标": ["平均密度", "中位数密度", "最大密度", "标准差", "估算人口(万)", "栅格像素数"],
                "值": [
                    f"{prov_current['mean']:.1f}",
                    f"{prov_current['median']:.1f}",
                    f"{prov_current['max']:.0f}",
                    f"{prov_current['std']:.1f}",
                    f"{prov_current['est_pop_wan']:,.0f}",
                    f"{prov_current['pixel_count']:,}",
                ],
            }),
            use_container_width=True,
            hide_index=True,
        )
        yoy_rows = []
        for i in range(1, len(prov_series)):
            prev = prov_series.iloc[i - 1]
            cur = prov_series.iloc[i]
            change = cur["mean"] - prev["mean"]
            pct = (change / prev["mean"] * 100) if prev["mean"] > 0 else 0
            yoy_rows.append({"年份": f"{int(prev['年份'])}→{int(cur['年份'])}", "变化": f"{change:+.1f}", "变化率": f"{pct:+.1f}%"})
        st.dataframe(pd.DataFrame(yoy_rows), use_container_width=True, height=250, hide_index=True)

    with col_compare:
        s1 = df_prov[df_prov["name"] == focus_province].sort_values("年份")
        s2 = df_prov[df_prov["name"] == compare_province].sort_values("年份")
        fig_cross = make_subplots(rows=1, cols=3, subplot_titles=["平均密度", "估算人口(万)", "最大密度"])
        for idx, metric in enumerate(["mean", "est_pop_wan", "max"], 1):
            fig_cross.add_trace(go.Scatter(x=s1["年份"], y=s1[metric], name=focus_province, mode="lines+markers", line=dict(color="#E31A1C", width=2), showlegend=(idx == 1)), row=1, col=idx)
            fig_cross.add_trace(go.Scatter(x=s2["年份"], y=s2[metric], name=compare_province, mode="lines+markers", line=dict(color="#3366CC", width=2), showlegend=(idx == 1)), row=1, col=idx)
        fig_cross.update_layout(height=330, margin=dict(t=40, b=20, l=20, r=20), hovermode="x unified", legend=dict(orientation="h", y=1.1), title=f"{focus_province} vs {compare_province}")
        st.plotly_chart(fig_cross, use_container_width=True)


# ===================================================================
# TAB 5: Research topics — Trade, Spatial, Market Integration focus
# ===================================================================
with tab_research:
    st.header("📖 研究课题：Trade, Spatial Economics & Market Integration")
    st.caption("本数据集（WorldPop UN-adjusted, 2002-2020）适用于以下研究方向，按与贸易/空间/市场一体化的相关性排序。")

    with st.expander("1️⃣ 市场一体化与贸易成本的空间代理", expanded=True):
        st.markdown("""
**核心问题**：中国国内市场一体化程度如何演变？人口再配置能否作为贸易成本下降的显示性证据？

在 NEG 框架中，贸易成本下降会促使劳动力向市场规模更大的核心区集中。人口密度的空间动态
因此可作为 market integration 的代理指标：集聚加速暗示要素流动壁垒在降低。

**适用方法**
- 人口重心漂移方向与距离 → 劳动力流动的宏观方向代理
- 加权平均密度（market potential 代理）时间序列 → 本地市场规模演化
- 东/中/西人口份额变化 → 区域间 market access 相对变化

**本平台已提供的证据**
- **📌 摘要页**：人口重心漂移方向与距离、市场潜力代理指标趋势
- **🗺️ 空间格局 → 人口重心与区域份额**：四大区域人口份额演变图

**参考文献**
- Redding & Venables (2004) *Economic geography and international inequality*
- Zheng & Kahn (2013) *China's bullet trains facilitate market integration*
- Hanson (2005) *Market potential, increasing returns, and geographic concentration*
""")

    with st.expander("2️⃣ 空间不平等与区域收敛"):
        st.markdown("""
**核心问题**：中国省际/区域间人口密度在收敛还是发散？这对 regional economic outcomes 意味着什么？

人口密度收敛/发散直接反映区域经济机会的趋同或分化。σ-收敛和 β-收敛检验
是 regional economics 的经典分析范式。

**适用方法**
- β-收敛：ln(初始密度) vs 年均增长率的 OLS 回归
- σ-收敛：省际密度变异系数（CV）的时间趋势
- 空间基尼系数：像元级人口分布不平等度
- Theil 指数分解：总不平等 = 省际 + 省内

**本平台已提供的证据**
- **📌 摘要页**：β-收敛散点图（按东/中/西/东北着色）、空间基尼系数趋势
- **📈 全国演变**：σ-收敛（CV 时间序列）、空间基尼系数面板
- **🏛️ 区域收敛**：σ-收敛详图、β-收敛散点图、省际热力矩阵

**参考文献**
- Barro & Sala-i-Martin (1992) *Convergence*
- Lessmann & Seidel (2017) *Regional inequality, convergence, and its determinants*
- Fan & Sun (2008) *Regional inequality in China, 1978-2006*
""")

    with st.expander("3️⃣ 基础设施与政策的因果推断"):
        st.markdown("""
**核心问题**：高铁开通、经济特区设立等政策冲击是否显著改变了周边人口密度和 regional outcomes？

人口密度栅格提供了精细的空间连续数据，非常适合做空间断点回归（Spatial RDD）
或双重差分（DID）分析，评估交通基础设施和贸易政策对区域经济的因果效应。

**适用方法**
- **空间 DID**：以高铁站点/自贸区为处理组，匹配对照组，比较政策前后密度变化
- **空间 RDD**：以政策边界（如开发区边界）为断点，检验边界两侧密度差异
- **合成控制法**：为特定城市构建反事实对照

**本平台已提供的证据**
- **🗺️ 空间格局 → 两期对比**：可自由选择政策前后年份，查看空间再配置地图
- **🏛️ 区域收敛 → 焦点省份剖面**：特定省份的密度时间序列，可识别政策拐点

**参考文献**
- Zheng & Kahn (2013) *China's bullet trains facilitate market integration*
- Qin (2017) *"No county left behind?" The distributional impact of high-speed rail upgrades in China*
- Faber (2014) *Trade integration, market size, and industrialization*
""")

    with st.expander("4️⃣ 人口迁移推断与劳动力流动"):
        st.markdown("""
**核心问题**：哪些地区在持续"吸人"，哪些在"失血"？净迁移格局如何反映区域比较优势？

相邻年份的密度差分（Δρ）扣除自然增长率后，残差可近似视为净迁移的空间代理变量。
迁移方向揭示劳动力对区域间市场规模和就业机会差异的"用脚投票"。

**适用方法**
- 差分法：Δρ(x,y) = ρ_t(x,y) - ρ_{t-1}(x,y)
- 扣除自然增长：利用省级出生率/死亡率数据估算自然增量，残差 ≈ 净迁移
- 空间聚类：对 Δρ 做 Local Moran's I，识别显著聚集区

**本平台已提供的证据**
- **📈 全国演变**：累计变化动画——红色 = 人口流入区，蓝色 = 流出区
- **🗺️ 空间格局 → 两期对比**：差值地图 + 省级变化柱状图

**参考文献**
- Bakker et al. (2021) *Estimating net migration at high spatial resolution*
- Tombe & Zhu (2019) *Trade, migration, and productivity: A quantitative analysis of China*
""")

    with st.expander("5️⃣ 城镇化结构与城市体系"):
        st.markdown("""
**核心问题**：中国城市体系的空间结构如何演变？是否服从 Zipf 定律？城镇化模式是"摊大饼"还是"多中心"？

栅格数据允许跳过行政区划限制，用密度阈值直接构建"功能城市区"（Functional Urban Area），
检验城市规模分布并追踪城市边界扩张。

**适用方法**
- 密度阈值 + 连通域分析 → 功能城市区 → Zipf 系数估计
- 城市蔓延指数：斑块面积增速 vs 人口增速
- 多中心性检验：次中心密度峰值的数量与强度变化

**本平台已提供的证据**
- **📈 全国演变**：高密度像元数（核心区规模）时间趋势
- **🗺️ 空间格局 → 单年格局**：像元密度分布直方图

**参考文献**
- Gabaix (1999) *Zipf's Law for Cities: An Explanation*
- Angel et al. (2011) *The dimensions of global urban expansion*
- Au & Henderson (2006) *Are Chinese cities too small?*
""")

    with st.expander("6️⃣ 多源数据叠加与耦合分析"):
        st.markdown("""
**核心问题**：人口密度与经济活动、夜间灯光、贸易流量之间存在怎样的空间耦合关系？

将人口密度栅格与其他空间数据集叠加，可以揭示人口-经济-贸易的协同演化模式，
为 market access 和 economic geography 的实证研究提供多维度支撑。

**适用方法**
- 栅格叠加：人口密度 × 夜间灯光（DMSP/VIIRS）→ 残差分析
- Market access 构建：人口密度 + 交通网络 → Harris (1954) market potential
- 地理加权回归（GWR）：探索贸易-人口关系的空间异质性

**可叠加的数据源**
- 夜间灯光：DMSP-OLS / NPP-VIIRS → 经济活动代理
- GDP 栅格：Kummu et al. (2018) → 空间经济产出
- 交通网络：OpenStreetMap 路网 → 贸易成本度量
- 土地利用：GlobeLand30 / CLCD → 城市用地扩张

**参考文献**
- Henderson et al. (2012) *Measuring economic growth from outer space*
- Donaldson & Hornbeck (2016) *Railroads and American economic growth*
""")

    st.markdown("---")
    st.info("以上课题均可基于本平台展示的数据展开初步分析。各课题标注了本平台已提供的相关证据页面，可直接跳转查看。如需原始 1km 栅格，请访问 [WorldPop 官方数据页](%s)。" % WORLDPOP_COLLECTION_URL)


# ===================================================================
# TAB 6: Data and methods
# ===================================================================
with tab_methods:
    st.header("📚 数据与方法")
    st.caption("把数据口径、处理流程和解释边界写清楚，是经济分析页面最重要的可信度部分。")

    col_data, col_method = st.columns(2)
    with col_data:
        st.subheader("数据来源")
        st.markdown(
            f"1. 官方来源：[{WORLDPOP_COLLECTION_URL}]({WORLDPOP_COLLECTION_URL})\n"
            f"2. 数据类型：WorldPop `UN-adjusted 1km Population Density`\n"
            f"3. 官方中国示例页：[2002]({WORLDPOP_CHINA_2002_URL})、[2020]({WORLDPOP_CHINA_2020_URL})\n"
            f"4. DOI：[{WORLDPOP_DOI_URL}]({WORLDPOP_DOI_URL})\n"
            f"5. 许可：[{CC_BY_4_URL}]({CC_BY_4_URL})"
        )
        st.subheader("样本范围")
        st.markdown(
            f"1. 当前网页使用年份：{YEARS[0]}-{YEARS[-1]}\n"
            "2. 指标单位：人口密度，单位为人/km²\n"
            "3. 当前展示数据经过 1km → 5km 平均降采样\n"
            f"4. 范围切换：当前可在“{scope_label_readable}”与全国口径之间切换"
        )

    with col_method:
        st.subheader("处理流程")
        st.markdown(
            "1. 使用平均重采样把原始 1km 栅格降到约 5km，以降低前端负载。\n"
            "2. 全国人口与全国均值直接由栅格计算，不通过省级统计反推。\n"
            "3. 省级人口估算采用纬度修正后的像元面积，而不是固定 25 km²。\n"
            "4. 变化图使用 P99 对称裁剪，避免少数极端像元压缩整体色阶。\n"
            "5. 空间经济学指标（基尼系数、收敛检验、人口重心）直接由像元级栅格实时计算。"
        )
        st.subheader("解释边界")
        st.markdown(
            "1. 本网页反映的是人口密度栅格估算，不等于官方行政统计口径。\n"
            "2. 5km 降采样会平滑超高密度城市核心区。\n"
            "3. UN-adjusted 表示全国总量与联合国人口估计保持一致。\n"
            "4. 栅格密度差分作为迁移代理变量存在局限：无法区分自然增长与机械增长。\n"
            "5. β/σ-收敛检验基于省级均值，省内异质性未被捕捉。\n"
            "6. 页面不提供原始数据下载，仅提供分析性浏览。"
        )

    st.subheader("变量口径")
    st.dataframe(
        pd.DataFrame([
            {"变量": "估算总人口", "说明": "按栅格密度 × 纬度修正像元面积汇总得到的总人口估算"},
            {"变量": "加权平均密度", "说明": "按有效像元面积加权后的全国平均密度，可作为 market potential 代理指标"},
            {"变量": "高密度像元数", "说明": "人口密度 > 1000 人/km² 的像元数量，代表核心区规模"},
            {"变量": "空间基尼系数", "说明": "基于像元人口（密度×面积）的洛伦兹曲线计算，衡量人口空间分布不平等度"},
            {"变量": "Top10%集中度", "说明": "最密 10% 像元的人口占总人口比例，衡量集聚程度"},
            {"变量": "人口重心", "说明": "Σ(人口×坐标) / Σ人口，漂移方向反映 market integration 趋势"},
            {"变量": "β-收敛斜率", "说明": "ln(初始密度) vs 年均增长率的 OLS 斜率；负值 = 收敛（追赶），正值 = 发散"},
            {"变量": "σ-收敛 (CV)", "说明": "省际密度变异系数（标准差/均值）；下降 = 省际差异缩小"},
            {"变量": "区域人口份额", "说明": "东/中/西/东北四大区域的人口占比，基于省级估算人口汇总"},
            {"变量": "省级平均密度", "说明": "省域内像元密度的算术平均"},
            {"变量": "增长率", "说明": "相对基期（2002 年）的百分比变化"},
        ]),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("建议引用")
    st.code(
        "WorldPop and CIESIN (2018). Global High Resolution Population Denominators Project. "
        "DOI: 10.5258/SOTON/WP00675",
        language="text",
    )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
scope_label = "仅大陆" if mainland_only else "含港澳台"
st.markdown(
    f"<small>"
    f"🌏 中国人口密度时空探索器 ({scope_label}) | "
    f"数据: WorldPop UN-adjusted 1km → 5km降采样 | "
    f"人口估算: 纬度修正像元面积 | 仅供趋势参考<br>"
    f"官方来源: <a href='{WORLDPOP_COLLECTION_URL}' target='_blank'>WorldPop Global1 2000-2020</a> | "
    f"DOI: <a href='{WORLDPOP_DOI_URL}' target='_blank'>10.5258/SOTON/WP00675</a> | "
    f"许可: <a href='{CC_BY_4_URL}' target='_blank'>CC BY 4.0</a> | "
    f"中国示例: <a href='{WORLDPOP_CHINA_2002_URL}' target='_blank'>2002</a> / "
    f"<a href='{WORLDPOP_CHINA_2020_URL}' target='_blank'>2020</a>"
    f"</small>",
    unsafe_allow_html=True,
)
