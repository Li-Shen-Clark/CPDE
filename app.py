"""
中国人口密度时空探索器 (2002-2020)
China Population Density Spatiotemporal Explorer
Spatiotemporal Explorer: Time + Map + Compare + Drill-down

Fixes applied:
  1. Latitude-corrected pixel area (not fixed 25 km²)
  2. National summary from raster (not province rollup)
  3. Adaptive diff colorscale via p99
  4. Mainland-only toggle
  5. Independent year controls per tab
  6. Map-click pixel drill-down
  7. Bilingual Chinese/English UI
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
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* Academic warm-tone base */
html, body, [class*="css"] {
    font-family: 'Crimson Pro', 'Georgia', 'Noto Serif SC', serif;
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 1rem;
    max-width: 1200px;
    background-color: #faf9f6;
}
/* Sidebar */
section[data-testid="stSidebar"] {
    background: #f0efe9;
    border-right: 1px solid #ddd8d0;
}
/* Nav buttons in sidebar */
section[data-testid="stSidebar"] button[kind="secondary"] {
    background: #faf9f6 !important;
    border: 1px solid #d5d0c8 !important;
    border-radius: 6px !important;
    padding: 9px 14px !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    color: #3d3d3d !important;
    transition: background 0.15s ease !important;
    box-shadow: none !important;
    margin-bottom: 2px !important;
}
section[data-testid="stSidebar"] button[kind="secondary"]:hover {
    background: #eae8e1 !important;
    border-color: #b8b2a6 !important;
    box-shadow: none !important;
    color: #1e3a5f !important;
}
/* KPI metric cards — quiet, academic */
[data-testid="stMetric"] {
    background: #f5f4f0;
    padding: 12px 16px;
    border-radius: 6px;
    border: 1px solid #ddd8d0;
}
[data-testid="stMetric"] label {
    font-size: 0.78rem;
    color: #6b6560;
    letter-spacing: 0.02em;
}
/* Headers — weight + spacing, no underline */
h1 {
    font-weight: 700;
    color: #1e3a5f;
    border: none !important;
    padding-bottom: 0;
    margin-bottom: 0.3rem;
}
h2 {
    font-weight: 600;
    color: #2d3748;
    border: none !important;
    padding-bottom: 0;
    margin-bottom: 0.2rem;
}
h3 {
    font-weight: 600;
    color: #3d3d3d;
}
/* Hide Streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
/* Expander */
.streamlit-expanderHeader {
    font-weight: 600;
    font-size: 1rem;
    color: #2d3748;
}
/* Captions */
.stCaption {
    font-size: 0.82rem;
    color: #8a8480;
}
/* Divider */
hr {
    border-color: #ddd8d0;
}
/* Data tables */
[data-testid="stDataFrame"] {
    border-radius: 6px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

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
REGION_MAP_EN = {
    "东部": "East",
    "中部": "Central",
    "西部": "West",
    "东北": "Northeast",
}
REGION_COLORS = {"东部": "#1e3a5f", "中部": "#c0392b", "西部": "#2d6a4f", "东北": "#64748b"}

# Modern Plotly layout defaults
PLOTLY_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Georgia, 'Noto Serif SC', serif", size=12, color="#3d3d3d"),
    plot_bgcolor="#faf9f6",
    paper_bgcolor="#faf9f6",
    margin=dict(t=40, b=20, l=20, r=20),
    hoverlabel=dict(bgcolor="#faf9f6", font_size=12, bordercolor="#ddd8d0"),
    colorway=["#1e3a5f", "#c0392b", "#2d6a4f", "#b8860b", "#64748b", "#8b5e3c", "#5b7065", "#a0522d"],
)


# ---------------------------------------------------------------------------
# Bilingual helper
# ---------------------------------------------------------------------------
# lang is set in the sidebar section below; declared here so T() can reference it.
# We use a module-level sentinel; the real value is set after st.sidebar.radio().

_LANG_STRINGS = {
    # Sidebar
    "sidebar_title_zh": "时空探索器",
    "sidebar_title_en": "Spatiotemporal Explorer",
    "sidebar_subtitle_zh": "中国人口密度 2002-2020",
    "sidebar_subtitle_en": "China Population Density 2002-2020",
    "mainland_toggle_zh": "仅中国大陆 (不含港澳台)",
    "mainland_toggle_en": "Mainland China only (excl. HK/MO/TW)",
    "opacity_slider_zh": "图层透明度",
    "opacity_slider_en": "Layer Opacity",
    "province_drilldown_zh": "##### 省份钻取",
    "province_drilldown_en": "##### Province Drill-down",
    "select_province_zh": "选择省份",
    "select_province_en": "Select Province",
    "all_national_zh": "(全国总览)",
    "all_national_en": "(National Overview)",
    "sidebar_data_info_zh": (
        "<small>数据: WorldPop UN-adj 1km<br>"
        "分辨率: ~5km 降采样<br>"
        "人口: 纬度修正像元面积<br>"
        "范围: {scope}</small>"
    ),
    "sidebar_data_info_en": (
        "<small>Data: WorldPop UN-adj 1km<br>"
        "Resolution: ~5km downsampled<br>"
        "Population: lat-corrected pixel area<br>"
        "Scope: {scope}</small>"
    ),
    "data_source_expander_zh": "数据来源与许可",
    "data_source_expander_en": "Data Sources & License",
    "data_source_md_zh": (
        "- 官方来源: [WorldPop Global1 2000-2020]({col})\n"
        "- 数据类型: [Population Density / UN-adjusted 1km]({proj})\n"
        "- 中国页面示例: [2002]({c2002}) / [2020]({c2020})\n"
        "- DOI: [{doi}]({doi})\n"
        "- 许可: [CC BY 4.0]({cc})"
    ),
    "data_source_md_en": (
        "- Official source: [WorldPop Global1 2000-2020]({col})\n"
        "- Data type: [Population Density / UN-adjusted 1km]({proj})\n"
        "- China example pages: [2002]({c2002}) / [2020]({c2020})\n"
        "- DOI: [{doi}]({doi})\n"
        "- License: [CC BY 4.0]({cc})"
    ),
    # Scope labels
    "scope_mainland_zh": "中国大陆",
    "scope_mainland_en": "Mainland China",
    "scope_all_zh": "全国含港澳台",
    "scope_all_en": "All incl. HK/MO/TW",
    "scope_short_mainland_zh": "仅大陆",
    "scope_short_mainland_en": "Mainland only",
    "scope_short_all_zh": "含港澳台",
    "scope_short_all_en": "Incl. HK/MO/TW",
    # Tab names
    "tab_summary_zh": "摘要",
    "tab_summary_en": "Summary",
    "tab_national_zh": "全国演变",
    "tab_national_en": "National Dynamics",
    "tab_spatial_zh": "空间格局",
    "tab_spatial_en": "Spatial Patterns",
    "tab_hetero_zh": "省际异质性",
    "tab_hetero_en": "Provincial Heterogeneity",
    "tab_research_zh": "研究课题",
    "tab_research_en": "Research Topics",
    "tab_methods_zh": "数据与方法",
    "tab_methods_en": "Data & Methods",
    # Page title
    "page_title_zh": "中国人口密度时空探索器",
    "page_title_en": "China Population Density Explorer",
    # ---- Tab 1: Summary ----
    "summary_header_zh": "研究摘要：空间集聚、市场潜力与区域收敛",
    "summary_header_en": "Summary: Spatial Agglomeration, Market Potential & Regional Convergence",
    "summary_caption_zh": "从空间经济学视角解读 {y0}-{y1} 年人口密度栅格数据。当前范围：{scope}。",
    "summary_caption_en": "Interpreting population density raster data {y0}-{y1} from a spatial economics perspective. Current scope: {scope}.",
    # shift directions
    "shift_ne_zh": "东北", "shift_ne_en": "Northeast",
    "shift_nw_zh": "西北", "shift_nw_en": "Northwest",
    "shift_se_zh": "东南", "shift_se_en": "Southeast",
    "shift_sw_zh": "西南", "shift_sw_en": "Southwest",
    "shift_n_zh": "北", "shift_n_en": "North",
    "shift_s_zh": "南", "shift_s_en": "South",
    "shift_e_zh": "东", "shift_e_en": "East",
    "shift_w_zh": "西", "shift_w_en": "West",
    # convergence words
    "convergence_zh": "收敛（追赶效应）",
    "convergence_en": "Convergence (catch-up effect)",
    "divergence_zh": "发散（强者恒强）",
    "divergence_en": "Divergence (winner-takes-all)",
    "sig_zh": "显著", "sig_en": "significant",
    "insig_zh": "不显著", "insig_en": "not significant",
    # findings
    "finding1_rise_zh": "升至", "finding1_rise_en": "rose to",
    "finding1_fall_zh": "降至", "finding1_fall_en": "fell to",
    "finding1_agg_zh": "表明集聚力（agglomeration forces）在研究期内持续主导",
    "finding1_agg_en": "indicating agglomeration forces dominated throughout the study period",
    "finding1_disp_zh": "表明分散力有所增强",
    "finding1_disp_en": "indicating dispersion forces have strengthened",
    # KPI labels
    "kpi_gini_zh": "空间基尼系数",
    "kpi_gini_en": "Spatial Gini Coeff.",
    "kpi_conc_zh": "Top10%人口占比",
    "kpi_conc_en": "Top10% Pop. Share",
    "kpi_wt_density_zh": "加权平均密度",
    "kpi_wt_density_en": "Weighted Avg. Density",
    "kpi_beta_zh": "β-收敛斜率",
    "kpi_beta_en": "β-Conv. Slope",
    "kpi_shift_zh": "重心漂移",
    "kpi_shift_en": "Centroid Shift",
    "convergence_label_zh": "收敛",
    "convergence_label_en": "Converging",
    "divergence_label_zh": "发散",
    "divergence_label_en": "Diverging",
    # Summary charts
    "summary_chart_title_zh": "集聚力在 19 年间持续主导空间格局",
    "summary_chart_title_en": "Agglomeration Forces Dominated Spatial Patterns Over 19 Years",
    "gini_series_label_zh": "空间基尼系数",
    "gini_series_label_en": "Spatial Gini Coeff.",
    "conc_series_label_zh": "Top10%人口占比(%)",
    "conc_series_label_en": "Top10% Pop. Share (%)",
    "gini_yaxis_zh": "基尼系数",
    "gini_yaxis_en": "Gini Coeff.",
    "conc_yaxis_zh": "Top10% 占比 (%)",
    "conc_yaxis_en": "Top10% Share (%)",
    "diff_map_caption_zh": "{y0}→{y1} 空间再配置地图",
    "diff_map_caption_en": "{y0}→{y1} Spatial Reallocation Map",
    "diff_map_note_zh": "红色 = 人口流入区，蓝色 = 流出区。P99 对称色标 ±{p99:.0f} 人/km²，像元平均变化 {mean:+.1f}。",
    "diff_map_note_en": "Red = population inflow, Blue = outflow. P99 symmetric colorscale ±{p99:.0f} ppl/km², pixel avg change {mean:+.1f}.",
    # Beta convergence chart
    "beta_chart_title_zh": "β-收敛检验：ln(初始密度) vs 年均增长率 (R²={r2:.3f})",
    "beta_chart_title_en": "β-Convergence: ln(Initial Density) vs Annual Growth Rate (R²={r2:.3f})",
    "beta_xaxis_zh": "ln({y0}年平均密度)",
    "beta_xaxis_en": "ln({y0} avg. density)",
    "beta_yaxis_zh": "年均增长率 (%)",
    "beta_yaxis_en": "Annual Growth Rate (%)",
    "ols_label_zh": "OLS (β={beta:.4f})",
    "ols_label_en": "OLS (β={beta:.4f})",
    # Top movers
    "top_movers_zh": "**人口集聚最快的省份**",
    "top_movers_en": "**Fastest Population Gaining Provinces**",
    "top_losers_zh": "**人口流失最快的省份**",
    "top_losers_en": "**Fastest Population Losing Provinces**",
    # ---- Tab 2: National ----
    "national_header_zh": "集聚动态与市场规模演化",
    "national_header_en": "Agglomeration Dynamics & Market Scale Evolution",
    "national_caption_zh": "追踪 {y0}-{y1} 年集聚力的时间演变。上半部分为空间再配置动画，下半部分为集聚指标时间序列。",
    "national_caption_en": "Tracking the temporal evolution of agglomeration forces {y0}-{y1}. Top: spatial reallocation animation. Bottom: agglomeration indicator time series.",
    "play_btn_zh": "播放累计变化",
    "play_btn_en": "Play Cumulative Change",
    "speed_label_zh": "速度",
    "speed_label_en": "Speed",
    "speed_fmt_zh": "{x}s/帧",
    "speed_fmt_en": "{x}s/frame",
    "year_slider_zh": "年份",
    "year_slider_en": "Year",
    "anim_frame_caption_zh": "{yr} 年累计变化（相对 {y0} 年）",
    "anim_frame_caption_en": "{yr} cumulative change (relative to {y0})",
    "anim_mean_change_zh": "累计平均变化",
    "anim_mean_change_en": "Cumul. Avg. Change",
    "anim_max_increase_zh": "最大累计增幅",
    "anim_max_increase_en": "Max Cumul. Increase",
    "anim_increase_share_zh": "增长区域占比",
    "anim_increase_share_en": "Growth Area Share",
    "anim_decrease_share_zh": "下降区域占比",
    "anim_decrease_share_en": "Decline Area Share",
    "anim_stats_caption_zh": "当前总人口 {pop:.2f} 亿，加权平均密度 {wt:.1f} 人/km²；差值色标约为 ±{p99:.0f} 人/km²，裁剪 {clip:.1f}% 像素。",
    "anim_stats_caption_en": "Current total pop. {pop:.2f}B, weighted avg. density {wt:.1f} ppl/km²; diff colorscale ≈±{p99:.0f} ppl/km², clipping {clip:.1f}% pixels.",
    "anim_top5_zh": "**{yr} 年高密度 Top 5 省份**",
    "anim_top5_en": "**{yr} Top 5 High-Density Provinces**",
    "agg_indicators_zh": "集聚指标时间序列",
    "agg_indicators_en": "Agglomeration Indicator Time Series",
    # Gini chart
    "gini_chart_title_zh": "人口持续向少数区域集中",
    "gini_chart_title_en": "Population Continues Concentrating in Fewer Areas",
    "year_label_zh": "年份",
    "year_label_en": "Year",
    "gini_caption_zh": "基尼系数越高，人口在空间上越集中。趋势上升 = 集聚力主导。",
    "gini_caption_en": "Higher Gini = more spatially concentrated population. Rising trend = agglomeration dominates.",
    # Concentration chart
    "conc_chart_title_zh": "核心区吸纳了越来越多的人口",
    "conc_chart_title_en": "Core Areas Absorb an Increasing Share of Population",
    "conc_yaxis2_zh": "占总人口比例 (%)",
    "conc_yaxis2_en": "Share of Total Population (%)",
    "conc_caption_zh": "该比例上升意味着更多人口进入少数高密度区域——市场规模效应的直接体现。",
    "conc_caption_en": "Rising share means more population entering high-density areas — direct evidence of market size effects.",
    # Market potential chart
    "mp_chart_title_zh": "本地市场规模在持续扩大",
    "mp_chart_title_en": "Local Market Scale Continues to Expand",
    "mp_trace1_zh": "加权平均密度 (Market Potential 代理)",
    "mp_trace1_en": "Weighted Avg. Density (Market Potential Proxy)",
    "mp_trace2_zh": "高密度像元数 (核心区规模)",
    "mp_trace2_en": "High-Density Pixels (Core Area Scale)",
    "mp_yaxis1_zh": "人/km²",
    "mp_yaxis1_en": "ppl/km²",
    "mp_yaxis2_zh": "像元数",
    "mp_yaxis2_en": "Pixel Count",
    # Sigma convergence chart
    "sigma_chart_title_zh": "省际差异是在缩小还是扩大？",
    "sigma_chart_title_en": "Are Interprovincial Gaps Narrowing or Widening?",
    "sigma_yaxis_zh": "变异系数",
    "sigma_yaxis_en": "Coeff. of Variation",
    "sigma_caption_zh": "CV 下降 = σ-收敛（省际差异缩小）；CV 上升 = σ-发散（差异扩大）。",
    "sigma_caption_en": "CV declining = σ-convergence (narrowing interprovincial gap); CV rising = σ-divergence (widening gap).",
    # ---- Tab 3: Spatial ----
    "spatial_header_zh": "核心-外围格局与空间再配置",
    "spatial_header_en": "Core-Periphery Pattern & Spatial Reallocation",
    "spatial_sub_single_zh": "单年格局",
    "spatial_sub_single_en": "Single Year",
    "spatial_sub_compare_zh": "两期对比",
    "spatial_sub_compare_en": "Two-Period Comparison",
    "spatial_sub_centroid_zh": "人口重心与区域份额",
    "spatial_sub_centroid_en": "Population Centroid & Regional Shares",
    "year_slider2_zh": "年份",
    "year_slider2_en": "Year",
    "click_chart_title_zh": "点击位置 ({lat:.3f}, {lng:.3f}) 的历年密度",
    "click_chart_title_en": "Density Time Series at Clicked Location ({lat:.3f}, {lng:.3f})",
    "click_xaxis_zh": "年份",
    "click_xaxis_en": "Year",
    "density_unit_zh": "人/km²",
    "density_unit_en": "ppl/km²",
    "top15_title_zh": "省级密度排名 Top 15",
    "top15_title_en": "Top 15 Provinces by Density",
    "pixel_hist_title_zh": "像元密度分布（截取 P99.5）",
    "pixel_hist_title_en": "Pixel Density Distribution (P99.5 truncated)",
    "pixel_count_zh": "像元数",
    "pixel_count_en": "Pixel Count",
    "compare_start_zh": "起始年份",
    "compare_start_en": "Start Year",
    "compare_end_zh": "结束年份",
    "compare_end_en": "End Year",
    "compare_growth_zh": "增长区域",
    "compare_growth_en": "Growth Area",
    "compare_decline_zh": "下降区域",
    "compare_decline_en": "Decline Area",
    "compare_avg_change_zh": "平均变化",
    "compare_avg_change_en": "Avg. Change",
    "compare_p99_zh": "P99色标",
    "compare_p99_en": "P99 Colorscale",
    "compare_clip_note_zh": "超出裁剪: {n:,} 像素 ({pct:.1f}%)",
    "compare_clip_note_en": "Clipped: {n:,} pixels ({pct:.1f}%)",
    "prov_change_header_zh": "省级变化分析",
    "prov_change_header_en": "Provincial Change Analysis",
    "prov_diff_title_zh": "各省平均密度变化",
    "prov_diff_title_en": "Provincial Avg. Density Change",
    "pixel_change_hist_zh": "像元变化分布",
    "pixel_change_hist_en": "Pixel Change Distribution",
    "diff_warning_zh": "请选择两个不同年份。",
    "diff_warning_en": "Please select two different years.",
    # Centroid subtab
    "centroid_header_zh": "人口重心轨迹",
    "centroid_header_en": "Population Centroid Trajectory",
    "centroid_caption_zh": "人口重心 = Σ(人口×坐标) / Σ人口。重心漂移方向反映劳动力流动和 market integration 的宏观趋势。",
    "centroid_caption_en": "Population centroid = Σ(pop×coord)/Σpop. Drift direction reflects macro trends in labor mobility and market integration.",
    "centroid_hover_zh": "年份: %{text}<br>经度: %{lon:.4f}<br>纬度: %{lat:.4f}<extra></extra>",
    "centroid_hover_en": "Year: %{text}<br>Lon: %{lon:.4f}<br>Lat: %{lat:.4f}<extra></extra>",
    "centroid_chart_title_zh": "人口重心轨迹 ({y0}-{y1})",
    "centroid_chart_title_en": "Population Centroid Trajectory ({y0}-{y1})",
    "centroid_start_zh": "**起点** ({y0}): ({lat:.4f}°N, {lon:.4f}°E)",
    "centroid_start_en": "**Start** ({y0}): ({lat:.4f}°N, {lon:.4f}°E)",
    "centroid_end_zh": "**终点** ({y1}): ({lat:.4f}°N, {lon:.4f}°E)",
    "centroid_end_en": "**End** ({y1}): ({lat:.4f}°N, {lon:.4f}°E)",
    "centroid_dir_zh": "**漂移方向**: 向{dir}",
    "centroid_dir_en": "**Drift Direction**: toward {dir}",
    "centroid_dist_zh": "**漂移距离**: {km:.1f} km",
    "centroid_dist_en": "**Drift Distance**: {km:.1f} km",
    "centroid_lat_dir_zh": "- 纬度方向: {km:.1f} km ({dir})",
    "centroid_lat_dir_en": "- Latitudinal: {km:.1f} km ({dir})",
    "centroid_lon_dir_zh": "- 经度方向: {km:.1f} km ({dir})",
    "centroid_lon_dir_en": "- Longitudinal: {km:.1f} km ({dir})",
    "centroid_meaning_zh": "**经济含义**",
    "centroid_meaning_en": "**Economic Interpretation**",
    "centroid_meaning_text_zh": (
        "人口重心向某方向漂移，表明该方向的劳动力吸引力（就业机会、公共服务、市场规模）更强。"
        "在 market integration 框架下，重心漂移方向与贸易成本下降最快的方向一致。"
    ),
    "centroid_meaning_text_en": (
        "Centroid drift toward a direction indicates stronger labor attraction (employment, public services, market size) in that direction. "
        "Under a market integration framework, drift direction aligns with where trade costs fell fastest."
    ),
    "latlon_chart_title_zh": "重心坐标时间序列",
    "latlon_chart_title_en": "Centroid Coordinate Time Series",
    "lat_label_zh": "纬度",
    "lat_label_en": "Latitude",
    "lon_label_zh": "经度",
    "lon_label_en": "Longitude",
    "deg_n_zh": "°N", "deg_n_en": "°N",
    "deg_e_zh": "°E", "deg_e_en": "°E",
    "region_shares_header_zh": "四大区域人口份额演变",
    "region_shares_header_en": "Four-Region Population Share Trends",
    "region_shares_caption_zh": "东部/中部/西部/东北的人口份额变化，反映区域间 market access 的相对变化。",
    "region_shares_caption_en": "Population share changes across East/Central/West/Northeast, reflecting shifts in relative regional market access.",
    "region_shares_chart_title_zh": "四大区域人口份额",
    "region_shares_chart_title_en": "Four-Region Population Shares",
    "region_shares_yaxis_zh": "占总人口比例 (%)",
    "region_shares_yaxis_en": "Share of Total Population (%)",
    "region_shares_table_title_zh": "**区域份额对比**",
    "region_shares_table_title_en": "**Regional Share Comparison**",
    "region_col_zh": "区域",
    "region_col_en": "Region",
    "change_col_zh": "变化",
    "change_col_en": "Change",
    "region_interp_zh": "**解读**",
    "region_interp_en": "**Interpretation**",
    "east_rise_zh": "东部份额上升 {d:+.2f} 个百分点，核心区对劳动力的吸引持续增强，与 NEG 预测的核心-外围集聚一致。",
    "east_rise_en": "East's share rose {d:+.2f} pp, indicating continued labor attraction to the core, consistent with NEG core-periphery predictions.",
    "east_fall_zh": "东部份额下降 {d:+.2f} 个百分点，中西部人口回流或就地城镇化趋势增强。",
    "east_fall_en": "East's share fell {d:+.2f} pp, suggesting population return migration to inland regions or in-situ urbanization.",
    # ---- Tab 4: Heterogeneity ----
    "hetero_header_zh": "区域收敛与增长分异",
    "hetero_header_en": "Regional Convergence & Growth Divergence",
    "hetero_caption_zh": "核心问题：初始密度低的省份是否增长更快（收敛）？空间经济格局在趋同还是极化？",
    "hetero_caption_en": "Key question: Do provinces with lower initial density grow faster (convergence)? Is the spatial economic pattern converging or polarizing?",
    "focus_province_zh": "焦点省份",
    "focus_province_en": "Focus Province",
    "compare_province_zh": "对照省份",
    "compare_province_en": "Comparison Province",
    "focus_year_zh": "焦点年份",
    "focus_year_en": "Focus Year",
    "scatter_mode_zh": "增长轨迹视图",
    "scatter_mode_en": "Growth Trajectory View",
    "scatter_endpoint_zh": "终点散点图",
    "scatter_endpoint_en": "Endpoint Scatter",
    "scatter_animation_zh": "按年份动态演变",
    "scatter_animation_en": "Dynamic by Year",
    "beta_scatter_title_zh": "β-收敛散点图：初始密度 vs 增长率（按区域着色）",
    "beta_scatter_title_en": "β-Convergence Scatter: Initial Density vs Growth Rate (by Region)",
    "beta_scatter_xaxis_zh": "2002年平均密度 (人/km², 对数坐标)",
    "beta_scatter_xaxis_en": "2002 Avg. Density (ppl/km², log scale)",
    "beta_scatter_yaxis_zh": "2002→2020 增长率 (%)",
    "beta_scatter_yaxis_en": "2002→2020 Growth Rate (%)",
    "focus_province_legend_zh": "焦点省份",
    "focus_province_legend_en": "Focus Province",
    "ols_trend_zh": "OLS趋势 (β={beta:.3f})",
    "ols_trend_en": "OLS Trend (β={beta:.3f})",
    "beta_caption_neg_zh": "负斜率表示低密度省份增速更快（收敛/追赶）",
    "beta_caption_neg_en": "Negative slope: lower-density provinces grow faster (convergence/catch-up)",
    "beta_caption_pos_zh": "正斜率表示高密度省份增速更快（发散/极化）",
    "beta_caption_pos_en": "Positive slope: higher-density provinces grow faster (divergence/polarization)",
    "beta_caption_full_zh": "OLS 斜率 β = {beta:.4f}：{direction}。区域着色揭示地理分组的收敛模式。",
    "beta_caption_full_en": "OLS slope β = {beta:.4f}: {direction}. Regional coloring reveals geographic grouping convergence patterns.",
    "anim_title_zh": "增长轨迹：按年份动态演变（相对 {y0} 年）",
    "anim_title_en": "Growth Trajectories: Dynamic by Year (relative to {y0})",
    "anim_xaxis_zh": "2002年平均密度 (人/km²)",
    "anim_xaxis_en": "2002 Avg. Density (ppl/km²)",
    "anim_yaxis_zh": "相对 {y0} 年增长率 (%)",
    "anim_yaxis_en": "Growth Rate relative to {y0} (%)",
    "abs_growth_label_zh": "绝对增长",
    "abs_growth_label_en": "Absolute Growth",
    "anim_caption_zh": "动画模式更适合观察轨迹与路径依赖；终点散点图仍然更适合做静态比较与标注。",
    "anim_caption_en": "Animation mode is better for observing trajectories and path dependence; endpoint scatter is better for static comparison and annotation.",
    "heatmap_metric_zh": "热力矩阵指标",
    "heatmap_metric_en": "Heatmap Metric",
    "heatmap_opt_mean_zh": "平均密度",
    "heatmap_opt_mean_en": "Avg. Density",
    "heatmap_opt_growth_zh": "相对2002年增长率%",
    "heatmap_opt_growth_en": "Growth Rate vs 2002 %",
    "heatmap_opt_yoy_zh": "逐年变化",
    "heatmap_opt_yoy_en": "Year-on-Year Change",
    "heatmap_bar_density_zh": "人/km²",
    "heatmap_bar_density_en": "ppl/km²",
    "heatmap_bar_pct_zh": "%",
    "heatmap_bar_pct_en": "%",
    "sigma_detail_header_zh": "σ-收敛检验：省际离散度的时间趋势",
    "sigma_detail_header_en": "σ-Convergence Test: Temporal Trend of Interprovincial Dispersion",
    "sigma_detail_trace1_zh": "省际标准差",
    "sigma_detail_trace1_en": "Interprovincial Std. Dev.",
    "sigma_detail_trace2_zh": "变异系数 CV",
    "sigma_detail_trace2_en": "Coeff. of Variation CV",
    "sigma_detail_title_zh": "σ-收敛：省际密度标准差与变异系数",
    "sigma_detail_title_en": "σ-Convergence: Std. Dev. & CV of Interprovincial Density",
    "sigma_yaxis1_zh": "标准差 (人/km²)",
    "sigma_yaxis1_en": "Std. Dev. (ppl/km²)",
    "cv_yaxis_zh": "CV",
    "cv_yaxis_en": "CV",
    "sigma_judgment_zh": "**σ-收敛判断**",
    "sigma_judgment_en": "**σ-Convergence Assessment**",
    "cv_converge_zh": "↓ 收敛",
    "cv_converge_en": "↓ Converging",
    "cv_diverge_zh": "↑ 发散",
    "cv_diverge_en": "↑ Diverging",
    "std_shrink_zh": "↓ 缩小",
    "std_shrink_en": "↓ Shrinking",
    "std_expand_zh": "↑ 扩大",
    "std_expand_en": "↑ Expanding",
    "sigma_meaning_zh": "**经济含义**",
    "sigma_meaning_en": "**Economic Interpretation**",
    "sigma_conv_text_zh": "CV 下降表明省际人口密度差异在缩小（σ-收敛），可能反映劳动力流动降低了区域间的密度差距，与 market integration 促进要素趋同一致。",
    "sigma_conv_text_en": "Declining CV indicates narrowing interprovincial density differences (σ-convergence), likely reflecting labor mobility reducing regional density gaps, consistent with market integration promoting factor equalization.",
    "sigma_div_text_zh": "CV 上升表明省际人口密度差异在扩大（σ-发散），集聚力持续将人口从低密度省份吸引至高密度省份，空间极化在加剧。",
    "sigma_div_text_en": "Rising CV indicates widening interprovincial density differences (σ-divergence), with agglomeration forces continuously drawing population from low- to high-density provinces, intensifying spatial polarization.",
    "focus_profile_zh": "{prov} · 深度剖面",
    "focus_profile_en": "{prov} · In-Depth Profile",
    "kpi_yr_density_zh": "{yr}年平均密度",
    "kpi_yr_density_en": "{yr} Avg. Density",
    "kpi_pop_est_zh": "估算人口",
    "kpi_pop_est_en": "Est. Population",
    "kpi_national_rank_zh": "全国排名",
    "kpi_national_rank_en": "National Rank",
    "kpi_rank_val_zh": "第 {rank} 名",
    "kpi_rank_val_en": "Rank #{rank}",
    "kpi_total_growth_zh": "总增长",
    "kpi_total_growth_en": "Total Growth",
    "kpi_growth_rate_zh": "增长率",
    "kpi_growth_rate_en": "Growth Rate",
    "province_trend_title_zh": "{prov} 历年趋势",
    "province_trend_title_en": "{prov} Historical Trend",
    "avg_density_trace_zh": "平均密度",
    "avg_density_trace_en": "Avg. Density",
    "pop_est_trace_zh": "估算人口(亿)",
    "pop_est_trace_en": "Est. Pop. (100M)",
    "hundred_million_zh": "亿",
    "hundred_million_en": "100M",
    "rank_chart_title_zh": "{prov} 排名变化",
    "rank_chart_title_en": "{prov} Rank Change",
    "rank_yaxis_zh": "排名",
    "rank_yaxis_en": "Rank",
    "stats_table_metrics_zh": ["平均密度", "中位数密度", "最大密度", "标准差", "估算人口(万)", "栅格像素数"],
    "stats_table_metrics_en": ["Avg. Density", "Median Density", "Max Density", "Std. Dev.", "Est. Pop. (10k)", "Raster Pixels"],
    "stats_value_col_zh": "值",
    "stats_value_col_en": "Value",
    "stats_metric_col_zh": "指标",
    "stats_metric_col_en": "Metric",
    "yoy_year_col_zh": "年份",
    "yoy_year_col_en": "Period",
    "yoy_change_col_zh": "变化",
    "yoy_change_col_en": "Change",
    "yoy_pct_col_zh": "变化率",
    "yoy_pct_col_en": "Chg. Rate",
    "compare_subplots_zh": ["平均密度", "估算人口(万)", "最大密度"],
    "compare_subplots_en": ["Avg. Density", "Est. Pop. (10k)", "Max Density"],
    # ---- Tab 5: Research ----
    "research_header_zh": "研究课题：Trade, Spatial Economics & Market Integration",
    "research_header_en": "Research Topics: Trade, Spatial Economics & Market Integration",
    "research_caption_zh": "本数据集（WorldPop UN-adjusted, 2002-2020）适用于以下研究方向，按与贸易/空间/市场一体化的相关性排序。",
    "research_caption_en": "This dataset (WorldPop UN-adjusted, 2002-2020) is suitable for the following research directions, ranked by relevance to trade/spatial/market integration research.",
    "topic1_title_zh": "01 — 市场一体化与贸易成本的空间代理",
    "topic1_title_en": "01 — Market Integration & Spatial Proxy for Trade Costs",
    "topic1_body_zh": """
**核心问题**：中国国内市场一体化程度如何演变？人口再配置能否作为贸易成本下降的显示性证据？

在 NEG 框架中，贸易成本下降会促使劳动力向市场规模更大的核心区集中。人口密度的空间动态
因此可作为 market integration 的代理指标：集聚加速暗示要素流动壁垒在降低。

**适用方法**
- 人口重心漂移方向与距离 → 劳动力流动的宏观方向代理
- 加权平均密度（market potential 代理）时间序列 → 本地市场规模演化
- 东/中/西人口份额变化 → 区域间 market access 相对变化

**本平台已提供的证据**
- **摘要页**：人口重心漂移方向与距离、市场潜力代理指标趋势
- **空间格局 → 人口重心与区域份额**：四大区域人口份额演变图

**参考文献**
- Redding & Venables (2004) *Economic geography and international inequality*
- Zheng & Kahn (2013) *China's bullet trains facilitate market integration*
- Hanson (2005) *Market potential, increasing returns, and geographic concentration*
""",
    "topic1_body_en": """
**Core question**: How has the degree of China's domestic market integration evolved? Can population reallocation serve as revealed evidence of declining trade costs?

In the NEG framework, lower trade costs encourage labor to concentrate in core areas with larger market sizes. Spatial dynamics of population density thus serve as a proxy for market integration: accelerating agglomeration implies falling barriers to factor mobility.

**Applicable methods**
- Population centroid drift direction & distance → macro-directional proxy for labor mobility
- Weighted average density (market potential proxy) time series → evolution of local market scale
- East/Central/West population share changes → relative changes in interregional market access

**Evidence already provided on this platform**
- **Summary**: population centroid drift direction & distance, market potential proxy trends
- **Spatial Patterns → Population Centroid & Regional Shares**: four-region population share evolution chart

**References**
- Redding & Venables (2004) *Economic geography and international inequality*
- Zheng & Kahn (2013) *China's bullet trains facilitate market integration*
- Hanson (2005) *Market potential, increasing returns, and geographic concentration*
""",
    "topic2_title_zh": "02 — 空间不平等与区域收敛",
    "topic2_title_en": "02 — Spatial Inequality & Regional Convergence",
    "topic2_body_zh": """
**核心问题**：中国省际/区域间人口密度在收敛还是发散？这对 regional economic outcomes 意味着什么？

人口密度收敛/发散直接反映区域经济机会的趋同或分化。σ-收敛和 β-收敛检验
是 regional economics 的经典分析范式。

**适用方法**
- β-收敛：ln(初始密度) vs 年均增长率的 OLS 回归
- σ-收敛：省际密度变异系数（CV）的时间趋势
- 空间基尼系数：像元级人口分布不平等度
- Theil 指数分解：总不平等 = 省际 + 省内

**本平台已提供的证据**
- **摘要页**：β-收敛散点图（按东/中/西/东北着色）、空间基尼系数趋势
- **全国演变**：σ-收敛（CV 时间序列）、空间基尼系数面板
- **区域收敛**：σ-收敛详图、β-收敛散点图、省际热力矩阵

**参考文献**
- Barro & Sala-i-Martin (1992) *Convergence*
- Lessmann & Seidel (2017) *Regional inequality, convergence, and its determinants*
- Fan & Sun (2008) *Regional inequality in China, 1978-2006*
""",
    "topic2_body_en": """
**Core question**: Is interprovincial/interregional population density in China converging or diverging? What does this imply for regional economic outcomes?

Convergence/divergence of population density directly reflects the convergence or divergence of regional economic opportunities. σ-convergence and β-convergence tests are classic analytical paradigms in regional economics.

**Applicable methods**
- β-convergence: OLS regression of ln(initial density) vs annual growth rate
- σ-convergence: time trend of interprovincial density coefficient of variation (CV)
- Spatial Gini coefficient: pixel-level population distribution inequality
- Theil index decomposition: total inequality = interprovincial + intraprovincial

**Evidence already provided on this platform**
- **Summary**: β-convergence scatter (colored by East/Central/West/Northeast), spatial Gini trend
- **National Dynamics**: σ-convergence (CV time series), spatial Gini panel
- **Provincial Heterogeneity**: σ-convergence detail, β-convergence scatter, interprovincial heatmap

**References**
- Barro & Sala-i-Martin (1992) *Convergence*
- Lessmann & Seidel (2017) *Regional inequality, convergence, and its determinants*
- Fan & Sun (2008) *Regional inequality in China, 1978-2006*
""",
    "topic3_title_zh": "03 — 基础设施与政策的因果推断",
    "topic3_title_en": "03 — Causal Inference for Infrastructure & Policy",
    "topic3_body_zh": """
**核心问题**：高铁开通、经济特区设立等政策冲击是否显著改变了周边人口密度和 regional outcomes？

人口密度栅格提供了精细的空间连续数据，非常适合做空间断点回归（Spatial RDD）
或双重差分（DID）分析，评估交通基础设施和贸易政策对区域经济的因果效应。

**适用方法**
- **空间 DID**：以高铁站点/自贸区为处理组，匹配对照组，比较政策前后密度变化
- **空间 RDD**：以政策边界（如开发区边界）为断点，检验边界两侧密度差异
- **合成控制法**：为特定城市构建反事实对照

**本平台已提供的证据**
- **空间格局 → 两期对比**：可自由选择政策前后年份，查看空间再配置地图
- **区域收敛 → 焦点省份剖面**：特定省份的密度时间序列，可识别政策拐点

**参考文献**
- Zheng & Kahn (2013) *China's bullet trains facilitate market integration*
- Qin (2017) *"No county left behind?" The distributional impact of high-speed rail upgrades in China*
- Faber (2014) *Trade integration, market size, and industrialization*
""",
    "topic3_body_en": """
**Core question**: Did policy shocks such as high-speed rail openings and special economic zone establishment significantly change nearby population density and regional outcomes?

Population density rasters provide fine-grained spatially continuous data, highly suitable for Spatial Regression Discontinuity Design (Spatial RDD) or Difference-in-Differences (DID) analysis to evaluate causal effects of transportation infrastructure and trade policy on regional economies.

**Applicable methods**
- **Spatial DID**: Use HSR stations/FTZs as treatment group, match control group, compare density changes before/after policy
- **Spatial RDD**: Use policy boundaries (e.g., development zone boundaries) as cutoffs to test density differences on both sides
- **Synthetic control method**: Construct counterfactual comparisons for specific cities

**Evidence already provided on this platform**
- **Spatial Patterns → Two-Period Comparison**: freely select pre/post-policy years to view spatial reallocation maps
- **Provincial Heterogeneity → Focus Province Profile**: density time series for specific provinces to identify policy inflection points

**References**
- Zheng & Kahn (2013) *China's bullet trains facilitate market integration*
- Qin (2017) *"No county left behind?" The distributional impact of high-speed rail upgrades in China*
- Faber (2014) *Trade integration, market size, and industrialization*
""",
    "topic4_title_zh": "04 — 人口迁移推断与劳动力流动",
    "topic4_title_en": "04 — Migration Inference & Labor Mobility",
    "topic4_body_zh": """
**核心问题**：哪些地区在持续"吸人"，哪些在"失血"？净迁移格局如何反映区域比较优势？

相邻年份的密度差分（Δρ）扣除自然增长率后，残差可近似视为净迁移的空间代理变量。
迁移方向揭示劳动力对区域间市场规模和就业机会差异的"用脚投票"。

**适用方法**
- 差分法：Δρ(x,y) = ρ_t(x,y) - ρ_{t-1}(x,y)
- 扣除自然增长：利用省级出生率/死亡率数据估算自然增量，残差 ≈ 净迁移
- 空间聚类：对 Δρ 做 Local Moran's I，识别显著聚集区

**本平台已提供的证据**
- **全国演变**：累计变化动画——红色 = 人口流入区，蓝色 = 流出区
- **空间格局 → 两期对比**：差值地图 + 省级变化柱状图

**参考文献**
- Bakker et al. (2021) *Estimating net migration at high spatial resolution*
- Tombe & Zhu (2019) *Trade, migration, and productivity: A quantitative analysis of China*
""",
    "topic4_body_en": """
**Core question**: Which regions are continuously "attracting" people, and which are "bleeding" population? How does the net migration pattern reflect regional comparative advantage?

The density difference (Δρ) between adjacent years, after removing natural growth, can approximately serve as a spatial proxy for net migration. Migration direction reveals labor's "voting with feet" in response to differences in regional market size and employment opportunities.

**Applicable methods**
- Differencing: Δρ(x,y) = ρ_t(x,y) - ρ_{t-1}(x,y)
- Remove natural growth: Use provincial birth/death rate data to estimate natural increment; residual ≈ net migration
- Spatial clustering: Apply Local Moran's I to Δρ to identify significant clusters

**Evidence already provided on this platform**
- **National Dynamics**: cumulative change animation — red = population inflow, blue = outflow
- **Spatial Patterns → Two-Period Comparison**: difference map + provincial change bar chart

**References**
- Bakker et al. (2021) *Estimating net migration at high spatial resolution*
- Tombe & Zhu (2019) *Trade, migration, and productivity: A quantitative analysis of China*
""",
    "topic5_title_zh": "05 — 城镇化结构与城市体系",
    "topic5_title_en": "05 — Urbanization Structure & Urban System",
    "topic5_body_zh": """
**核心问题**：中国城市体系的空间结构如何演变？是否服从 Zipf 定律？城镇化模式是"摊大饼"还是"多中心"？

栅格数据允许跳过行政区划限制，用密度阈值直接构建"功能城市区"（Functional Urban Area），
检验城市规模分布并追踪城市边界扩张。

**适用方法**
- 密度阈值 + 连通域分析 → 功能城市区 → Zipf 系数估计
- 城市蔓延指数：斑块面积增速 vs 人口增速
- 多中心性检验：次中心密度峰值的数量与强度变化

**本平台已提供的证据**
- **全国演变**：高密度像元数（核心区规模）时间趋势
- **空间格局 → 单年格局**：像元密度分布直方图

**参考文献**
- Gabaix (1999) *Zipf's Law for Cities: An Explanation*
- Angel et al. (2011) *The dimensions of global urban expansion*
- Au & Henderson (2006) *Are Chinese cities too small?*
""",
    "topic5_body_en": """
**Core question**: How has China's urban system spatial structure evolved? Does it follow Zipf's Law? Is the urbanization pattern "urban sprawl" or "polycentric"?

Raster data allows bypassing administrative boundary constraints, using density thresholds to directly construct Functional Urban Areas (FUAs), test urban size distributions, and track urban boundary expansion.

**Applicable methods**
- Density threshold + connected component analysis → Functional Urban Areas → Zipf coefficient estimation
- Urban sprawl index: patch area growth rate vs population growth rate
- Polycentricity test: number and intensity changes of secondary center density peaks

**Evidence already provided on this platform**
- **National Dynamics**: high-density pixel count (core area scale) time trend
- **Spatial Patterns → Single Year**: pixel density distribution histogram

**References**
- Gabaix (1999) *Zipf's Law for Cities: An Explanation*
- Angel et al. (2011) *The dimensions of global urban expansion*
- Au & Henderson (2006) *Are Chinese cities too small?*
""",
    "topic6_title_zh": "06 — 多源数据叠加与耦合分析",
    "topic6_title_en": "06 — Multi-Source Data Overlay & Coupling Analysis",
    "topic6_body_zh": """
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
""",
    "topic6_body_en": """
**Core question**: What spatial coupling relationships exist between population density, economic activity, nighttime lights, and trade flows?

Overlaying population density rasters with other spatial datasets can reveal co-evolutionary patterns of population-economy-trade, providing multi-dimensional support for empirical research in market access and economic geography.

**Applicable methods**
- Raster overlay: population density × nighttime lights (DMSP/VIIRS) → residual analysis
- Market access construction: population density + transport network → Harris (1954) market potential
- Geographically Weighted Regression (GWR): explore spatial heterogeneity of trade-population relationships

**Overlayable data sources**
- Nighttime lights: DMSP-OLS / NPP-VIIRS → economic activity proxy
- GDP raster: Kummu et al. (2018) → spatial economic output
- Transport network: OpenStreetMap road network → trade cost measurement
- Land use: GlobeLand30 / CLCD → urban land expansion

**References**
- Henderson et al. (2012) *Measuring economic growth from outer space*
- Donaldson & Hornbeck (2016) *Railroads and American economic growth*
""",
    "research_footer_zh": "以上课题均可基于本平台展示的数据展开初步分析。各课题标注了本平台已提供的相关证据页面，可直接跳转查看。如需原始 1km 栅格，请访问 [WorldPop 官方数据页](%s)。",
    "research_footer_en": "All above topics can be preliminarily analyzed using data shown on this platform. Each topic notes relevant evidence pages already provided. For the original 1km rasters, visit the [WorldPop official data page](%s).",
    # ---- Tab 6: Methods ----
    "methods_header_zh": "数据与方法",
    "methods_header_en": "Data & Methods",
    "methods_caption_zh": "把数据口径、处理流程和解释边界写清楚，是经济分析页面最重要的可信度部分。",
    "methods_caption_en": "Clearly documenting data scope, processing workflow, and interpretation boundaries is the most important credibility element of any economic analysis page.",
    "methods_data_sources_zh": "数据来源",
    "methods_data_sources_en": "Data Sources",
    "methods_data_md_zh": (
        "1. 官方来源：[{col}]({col})\n"
        "2. 数据类型：WorldPop `UN-adjusted 1km Population Density`\n"
        "3. 官方中国示例页：[2002]({c2002})、[2020]({c2020})\n"
        "4. DOI：[{doi}]({doi})\n"
        "5. 许可：[{cc}]({cc})"
    ),
    "methods_data_md_en": (
        "1. Official source: [{col}]({col})\n"
        "2. Data type: WorldPop `UN-adjusted 1km Population Density`\n"
        "3. China example pages: [2002]({c2002}), [2020]({c2020})\n"
        "4. DOI: [{doi}]({doi})\n"
        "5. License: [{cc}]({cc})"
    ),
    "methods_sample_scope_zh": "样本范围",
    "methods_sample_scope_en": "Sample Scope",
    "methods_sample_md_zh": (
        "1. 当前网页使用年份：{y0}-{y1}\n"
        "2. 指标单位：人口密度，单位为人/km²\n"
        "3. 当前展示数据经过 1km → 5km 平均降采样\n"
        "4. 范围切换：当前可在\"{scope}\"与全国口径之间切换"
    ),
    "methods_sample_md_en": (
        "1. Years used on this page: {y0}-{y1}\n"
        "2. Indicator unit: population density in ppl/km²\n"
        "3. Data displayed after 1km → 5km average downsampling\n"
        "4. Scope toggle: currently switchable between \"{scope}\" and national scope"
    ),
    "methods_processing_zh": "处理流程",
    "methods_processing_en": "Processing Workflow",
    "methods_processing_md_zh": (
        "1. 使用平均重采样把原始 1km 栅格降到约 5km，以降低前端负载。\n"
        "2. 全国人口与全国均值直接由栅格计算，不通过省级统计反推。\n"
        "3. 省级人口估算采用纬度修正后的像元面积，而不是固定 25 km²。\n"
        "4. 变化图使用 P99 对称裁剪，避免少数极端像元压缩整体色阶。\n"
        "5. 空间经济学指标（基尼系数、收敛检验、人口重心）直接由像元级栅格实时计算。"
    ),
    "methods_processing_md_en": (
        "1. Original 1km raster downsampled to ~5km using average resampling to reduce frontend load.\n"
        "2. National population and national mean computed directly from raster, not back-calculated from provincial statistics.\n"
        "3. Provincial population estimates use latitude-corrected pixel area rather than a fixed 25 km².\n"
        "4. Change maps use P99 symmetric clipping to prevent a few extreme pixels from compressing the overall color scale.\n"
        "5. Spatial economics indicators (Gini coefficient, convergence tests, population centroid) computed in real time directly from pixel-level rasters."
    ),
    "methods_limits_zh": "解释边界",
    "methods_limits_en": "Interpretation Boundaries",
    "methods_limits_md_zh": (
        "1. 本网页反映的是人口密度栅格估算，不等于官方行政统计口径。\n"
        "2. 5km 降采样会平滑超高密度城市核心区。\n"
        "3. UN-adjusted 表示全国总量与联合国人口估计保持一致。\n"
        "4. 栅格密度差分作为迁移代理变量存在局限：无法区分自然增长与机械增长。\n"
        "5. β/σ-收敛检验基于省级均值，省内异质性未被捕捉。\n"
        "6. 页面不提供原始数据下载，仅提供分析性浏览。"
    ),
    "methods_limits_md_en": (
        "1. This page reflects population density raster estimates, not official administrative statistics.\n"
        "2. 5km downsampling smooths ultra-high-density urban cores.\n"
        "3. UN-adjusted means national totals are consistent with UN population estimates.\n"
        "4. Raster density differencing as a migration proxy has limitations: cannot distinguish natural from mechanical growth.\n"
        "5. β/σ-convergence tests are based on provincial means; within-province heterogeneity is not captured.\n"
        "6. This page does not provide raw data download, only analytical browsing."
    ),
    "methods_variables_zh": "变量口径",
    "methods_variables_en": "Variable Definitions",
    "methods_var_names_zh": ["估算总人口", "加权平均密度", "高密度像元数", "空间基尼系数", "Top10%集中度", "人口重心", "β-收敛斜率", "σ-收敛 (CV)", "区域人口份额", "省级平均密度", "增长率"],
    "methods_var_names_en": ["Est. Total Pop.", "Weighted Avg. Density", "High-Density Pixels", "Spatial Gini Coeff.", "Top10% Concentration", "Population Centroid", "β-Conv. Slope", "σ-Conv. (CV)", "Regional Pop. Share", "Provincial Avg. Density", "Growth Rate"],
    "methods_var_descs_zh": [
        "按栅格密度 × 纬度修正像元面积汇总得到的总人口估算",
        "按有效像元面积加权后的全国平均密度，可作为 market potential 代理指标",
        "人口密度 > 1000 人/km² 的像元数量，代表核心区规模",
        "基于像元人口（密度×面积）的洛伦兹曲线计算，衡量人口空间分布不平等度",
        "最密 10% 像元的人口占总人口比例，衡量集聚程度",
        "Σ(人口×坐标) / Σ人口，漂移方向反映 market integration 趋势",
        "ln(初始密度) vs 年均增长率的 OLS 斜率；负值 = 收敛（追赶），正值 = 发散",
        "省际密度变异系数（标准差/均值）；下降 = 省际差异缩小",
        "东/中/西/东北四大区域的人口占比，基于省级估算人口汇总",
        "省域内像元密度的算术平均",
        "相对基期（2002 年）的百分比变化",
    ],
    "methods_var_descs_en": [
        "Total population estimate aggregated from raster density × latitude-corrected pixel area",
        "National average density weighted by valid pixel area; serves as market potential proxy",
        "Number of pixels with population density > 1000 ppl/km², representing core area scale",
        "Computed from Lorenz curve of pixel-level population (density×area), measuring spatial inequality of population distribution",
        "Share of total population living in the densest 10% of pixels, measuring agglomeration degree",
        "Σ(pop×coord)/Σpop; drift direction reflects market integration trends",
        "OLS slope of ln(initial density) vs annual growth rate; negative = convergence (catch-up), positive = divergence",
        "Interprovincial density coefficient of variation (std/mean); declining = narrowing interprovincial gap",
        "Population shares of the four major regions (East/Central/West/Northeast), aggregated from provincial estimates",
        "Arithmetic mean of pixel densities within a province",
        "Percentage change relative to the base year (2002)",
    ],
    "methods_var_col_zh": "变量",
    "methods_var_col_en": "Variable",
    "methods_desc_col_zh": "说明",
    "methods_desc_col_en": "Description",
    "methods_citation_zh": "建议引用",
    "methods_citation_en": "Suggested Citation",
    # Author info
    "author_info_zh": "**Li Shen** · Spatial Economics & Data Science",
    "author_info_en": "**Li Shen** · Spatial Economics & Data Science",
    # Footer
    "footer_zh": (
        "<small>"
        "中国人口密度时空探索器 ({scope}) | "
        "Li Shen | Spatial Economics &amp; Data Science | "
        "数据: WorldPop UN-adjusted 1km → 5km降采样 | "
        "人口估算: 纬度修正像元面积 | 仅供趋势参考<br>"
        "官方来源: <a href='{col}' target='_blank'>WorldPop Global1 2000-2020</a> | "
        "DOI: <a href='{doi}' target='_blank'>10.5258/SOTON/WP00675</a> | "
        "许可: <a href='{cc}' target='_blank'>CC BY 4.0</a> | "
        "中国示例: <a href='{c2002}' target='_blank'>2002</a> / "
        "<a href='{c2020}' target='_blank'>2020</a>"
        "</small>"
    ),
    "footer_en": (
        "<small>"
        "China Population Density Spatiotemporal Explorer ({scope}) | "
        "Li Shen | Spatial Economics &amp; Data Science | "
        "Data: WorldPop UN-adjusted 1km → 5km downsampled | "
        "Pop. est.: lat-corrected pixel area | For trend reference only<br>"
        "Official source: <a href='{col}' target='_blank'>WorldPop Global1 2000-2020</a> | "
        "DOI: <a href='{doi}' target='_blank'>10.5258/SOTON/WP00675</a> | "
        "License: <a href='{cc}' target='_blank'>CC BY 4.0</a> | "
        "China examples: <a href='{c2002}' target='_blank'>2002</a> / "
        "<a href='{c2020}' target='_blank'>2020</a>"
        "</small>"
    ),
}


def T(key_zh, key_en=None, **kwargs):
    """Return Chinese or English string based on `lang`.
    - T("key") → looks up _LANG_STRINGS["key_zh"] or ["key_en"]
    - T("key_zh", "key_en", var=val) → format with kwargs
    """
    global lang
    if key_en is None:
        # key_zh is a base key, append _zh or _en suffix
        key = key_zh + ("_zh" if lang == "zh" else "_en")
        text = _LANG_STRINGS.get(key, key_zh)
    else:
        # key_zh and key_en are explicit full keys
        key = key_zh if lang == "zh" else key_en
        text = _LANG_STRINGS.get(key, key)
    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, IndexError):
            pass
    return text


def region_name(zh_name):
    """Return region name in current language."""
    if lang == "en":
        return REGION_MAP_EN.get(zh_name, zh_name)
    return zh_name


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
    colormap = cm.get_cmap("inferno")
    mask = np.isnan(data) | (data <= 0)
    normalized = np.where(mask, 0, norm(data))
    rgba = colormap(normalized)
    rgba[mask, 3] = 0
    return (rgba * 255).astype(np.uint8)


# [Fix 3] Adaptive diff colorscale using p99
@st.cache_data
def diff_to_image_adaptive(diff):
    """Coolwarm diverging colormap with p99-based symmetric bounds."""
    colormap = cm.get_cmap("coolwarm")
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

    background = np.full((h, w, 3), 250, dtype=np.float32)
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
            draw.line(pts, fill=(160, 165, 185), width=1)

    return np.asarray(canvas)


@st.cache_data
def render_diff_frame_image(year_a, year_b, opacity):
    """Render a static diff frame for summary use."""
    data_a, bounds, _, _, _ = load_year(year_a)
    data_b, _, _, _, _ = load_year(year_b)
    diff = data_b - data_a
    rgba, p99, clipped_n, total_n = diff_to_image_adaptive(diff)

    h, w = rgba.shape[:2]
    background = np.full((h, w, 3), 250, dtype=np.float32)
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
            draw.line(pts, fill=(160, 165, 185), width=1)

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

# Language toggle MUST come first so T() works for all subsequent labels
lang = st.sidebar.radio(
    "Language / 语言",
    options=["English", "中文"],
    horizontal=True,
    index=0,
    key="lang_select",
)
lang = "en" if lang == "English" else "zh"

st.sidebar.markdown(f"## {T('sidebar_title')}")
st.sidebar.caption(T("sidebar_subtitle"))

st.sidebar.divider()

# --- Navigation (highest priority) ---
_nav_labels = [
    T("tab_summary"),
    T("tab_national"),
    T("tab_spatial"),
    T("tab_hetero"),
    T("tab_research"),
    T("tab_methods"),
]
if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = _nav_labels[0]
if st.session_state["nav_page"] not in _nav_labels:
    st.session_state["nav_page"] = _nav_labels[0]

st.sidebar.markdown(
    f"<p style='font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;"
    f"color:#8a8480;font-weight:600;margin:0 0 6px 0;'>"
    f"{'导航' if lang == 'zh' else 'NAVIGATION'}</p>",
    unsafe_allow_html=True,
)
for _nl in _nav_labels:
    _is_active = (st.session_state["nav_page"] == _nl)
    if _is_active:
        st.sidebar.markdown(
            f"<div style='background:#1e3a5f;color:#faf9f6;"
            f"padding:10px 14px;border-radius:6px;font-weight:600;font-size:0.9rem;"
            f"margin-bottom:6px;'>{_nl}</div>",
            unsafe_allow_html=True,
        )
    else:
        if st.sidebar.button(_nl, key=f"nav_{_nl}", use_container_width=True):
            st.session_state["nav_page"] = _nl
            st.rerun()
_page = st.session_state["nav_page"]

st.sidebar.divider()

# --- Data scope (affects all pages) ---
mainland_only = st.sidebar.toggle(T("mainland_toggle"), value=True)

# --- Contextual controls (only shown on relevant pages) ---
_pages_with_map = {_nav_labels[1], _nav_labels[2]}  # National, Spatial
_page_with_drilldown = _nav_labels[3]  # Heterogeneity

if _page in _pages_with_map:
    opacity = st.sidebar.slider(T("opacity_slider"), 0.3, 1.0, 0.7, 0.05)
else:
    opacity = 0.7

available_names = filter_names(ALL_PROVINCE_NAMES, mainland_only)
if _page == _page_with_drilldown:
    st.sidebar.markdown(T("province_drilldown"))
    drill_province = st.sidebar.selectbox(
        T("select_province"),
        [T("all_national")] + available_names,
        index=0,
    )
else:
    drill_province = T("all_national")

st.sidebar.divider()

# --- Meta info ---
st.sidebar.markdown(T("author_info"))
st.sidebar.markdown("[GitHub](https://github.com/Li-Shen-Clark/CPDE)")
scope_label_readable = T("scope_mainland") if mainland_only else T("scope_all")
st.sidebar.markdown(
    T("sidebar_data_info", scope=scope_label_readable),
    unsafe_allow_html=True,
)
with st.sidebar.expander(T("data_source_expander")):
    st.markdown(
        T("data_source_md",
          col=WORLDPOP_COLLECTION_URL,
          proj=WORLDPOP_PROJECT_URL,
          c2002=WORLDPOP_CHINA_2002_URL,
          c2020=WORLDPOP_CHINA_2020_URL,
          doi=WORLDPOP_DOI_URL,
          cc=CC_BY_4_URL)
    )

# Compute rank once based on scope
exclude_for_rank = EXCLUDE_HMT if mainland_only else set()
scope_key = "mainland" if mainland_only else "all"
df_national = load_national_stats(scope_key)

# ===================================================================
# MAIN CONTENT
# ===================================================================
rank_df = compute_rank_history(province_stats, exclude_names=frozenset(exclude_for_rank))
df_growth = build_growth_df(frozenset(exclude_for_rank))
focus_default_idx = available_names.index(drill_province) if drill_province in available_names else 0

# Pre-compute economics indicators used across multiple pages
nat_start = df_national[df_national["年份"] == YEARS[0]].iloc[0]
nat_end = df_national[df_national["年份"] == YEARS[-1]].iloc[0]
df_gini = compute_spatial_gini_series(raster_cube, area_grid)
df_conc10 = compute_concentration_series(raster_cube, area_grid, top_pct=10)
df_centroid = compute_population_centroid(raster_cube, area_grid, cube_transform)
beta_df, beta_slope, beta_r2, beta_p = compute_beta_convergence(province_stats, frozenset(exclude_for_rank))


# ===================================================================
# TAB 1: Summary — Spatial Economics Narrative
# ===================================================================
# Derived indicators used across Summary + Spatial/Centroid pages
gini_start = df_gini[df_gini["年份"] == YEARS[0]]["空间基尼系数"].values[0]
gini_end = df_gini[df_gini["年份"] == YEARS[-1]]["空间基尼系数"].values[0]
conc_start = df_conc10[df_conc10["年份"] == YEARS[0]][f"Top10%人口占比"].values[0]
conc_end = df_conc10[df_conc10["年份"] == YEARS[-1]][f"Top10%人口占比"].values[0]
centroid_start = df_centroid[df_centroid["年份"] == YEARS[0]].iloc[0]
centroid_end = df_centroid[df_centroid["年份"] == YEARS[-1]].iloc[0]

dlat_km = (centroid_end["重心纬度"] - centroid_start["重心纬度"]) * 111.32
dlon_km = (centroid_end["重心经度"] - centroid_start["重心经度"]) * 111.32 * np.cos(np.radians(centroid_end["重心纬度"]))
shift_km = np.sqrt(dlat_km ** 2 + dlon_km ** 2)

if dlat_km > 0 and dlon_km > 0:
    shift_dir_zh = "东北"
    shift_dir_en = "Northeast"
elif dlat_km > 0 and dlon_km <= 0:
    shift_dir_zh = "西北"
    shift_dir_en = "Northwest"
elif dlat_km <= 0 and dlon_km > 0:
    shift_dir_zh = "东南"
    shift_dir_en = "Southeast"
else:
    shift_dir_zh = "西南"
    shift_dir_en = "Southwest"
shift_dir = shift_dir_zh if lang == "zh" else shift_dir_en

convergence_word = T("convergence") if beta_slope < 0 else T("divergence")
convergence_sig = T("sig") if beta_p < 0.05 else T("insig")


# ===================================================================
# TAB 1: Summary — Spatial Economics Narrative
# ===================================================================
if _page == _nav_labels[0]:
    st.header(T("summary_header"))
    st.caption(T("summary_caption", y0=YEARS[0], y1=YEARS[-1], scope=scope_label_readable))

    # --- Thesis + Key findings ---
    rise_fall_gini = T("finding1_rise") if gini_end > gini_start else T("finding1_fall")
    rise_fall_conc = T("finding1_rise") if conc_end > conc_start else T("finding1_fall")
    agg_text = T("finding1_agg") if gini_end > gini_start else T("finding1_disp")

    # Thesis statement
    if lang == "zh":
        st.markdown(
            f"> **核心发现**：{YEARS[0]}-{YEARS[-1]} 年间，中国人口在空间上持续向少数高密度区域集聚，"
            f"加权平均密度上升 {(nat_end['加权平均密度'] - nat_start['加权平均密度'])/nat_start['加权平均密度']*100:.0f}%，"
            f"人口重心向{shift_dir}偏移 {shift_km:.0f} km。"
        )
    else:
        st.markdown(
            f"> **Key finding**: From {YEARS[0]} to {YEARS[-1]}, China's population continued concentrating in a few high-density areas. "
            f"Weighted average density rose {(nat_end['加权平均密度'] - nat_start['加权平均密度'])/nat_start['加权平均密度']*100:.0f}%, "
            f"and the population centroid shifted {shift_km:.0f} km toward the {shift_dir}."
        )

    if lang == "zh":
        findings = [
            f"**发生了什么** — 空间基尼系数由 {gini_start:.4f} {rise_fall_gini} {gini_end:.4f}，"
            f"Top 10% 最密像元的人口份额由 {conc_start:.1f}% {rise_fall_conc} {conc_end:.1f}%，"
            f"{agg_text}。",

            f"**规模效应** — 加权平均密度由 {nat_start['加权平均密度']:.1f} 增至 "
            f"{nat_end['加权平均密度']:.1f} 人/km²，高密度像元由 {int(nat_start['高密度像素数']):,} 增至 {int(nat_end['高密度像素数']):,}，"
            f"核心区在持续扩张。",

            f"**区域分化** — β-收敛斜率 = {beta_slope:.4f}（R² = {beta_r2:.3f}，p = {beta_p:.3f}），"
            f"{'初始密度低的省份增速更快，呈追赶态势' if beta_slope < 0 else '初始密度高的省份增速更快，空间极化在加剧'}。",

            f"**劳动力流向** — 人口重心向{shift_dir}偏移 {shift_km:.1f} km，"
            f"反映 market integration 的宏观方向。",
        ]
    else:
        findings = [
            f"**What happened** — Spatial Gini rose from {gini_start:.4f} to {gini_end:.4f}; "
            f"Top 10% pixel population share moved from {conc_start:.1f}% to {conc_end:.1f}%, "
            f"{agg_text}.",

            f"**Scale effects** — Weighted avg. density increased from {nat_start['加权平均密度']:.1f} to "
            f"{nat_end['加权平均密度']:.1f} ppl/km²; high-density pixels grew from {int(nat_start['高密度像素数']):,} to {int(nat_end['高密度像素数']):,}, "
            f"the core kept expanding.",

            f"**Regional divergence** — β-convergence slope = {beta_slope:.4f} (R² = {beta_r2:.3f}, p = {beta_p:.3f}); "
            f"{'lower-density provinces grew faster, showing catch-up' if beta_slope < 0 else 'higher-density provinces grew faster, intensifying polarization'}.",

            f"**Labor flow** — Population centroid shifted {shift_km:.1f} km toward the {shift_dir}, "
            f"reflecting the macro direction of market integration.",
        ]

    for idx, text in enumerate(findings, 1):
        st.markdown(f"{idx}. {text}")

    # --- KPI row ---
    st.divider()
    k1, k2, k3 = st.columns(3, gap="medium")
    k1.metric(T("kpi_gini"), f"{gini_end:.4f}", delta=f"{gini_end - gini_start:+.4f}")
    k2.metric(T("kpi_conc"), f"{conc_end:.1f}%", delta=f"{conc_end - conc_start:+.1f}%")
    k3.metric(T("kpi_wt_density"), f"{nat_end['加权平均密度']:.1f}",
              delta=f"{nat_end['加权平均密度'] - nat_start['加权平均密度']:+.1f}")
    k4, k5, _ = st.columns(3, gap="medium")
    k4.metric(T("kpi_beta"), f"{beta_slope:.4f}",
              delta=T("convergence_label") if beta_slope < 0 else T("divergence_label"),
              delta_color="normal" if beta_slope < 0 else "inverse")
    k5.metric(T("kpi_shift"), f"{shift_km:.1f} km",
              delta=f"→ {shift_dir}")

    # --- Row 2: Gini trend + diff map ---
    st.divider()
    col_trend, col_map = st.columns([3, 2])
    with col_trend:
        fig_summary = make_subplots(specs=[[{"secondary_y": True}]])
        fig_summary.add_trace(go.Scatter(
            x=df_gini["年份"], y=df_gini["空间基尼系数"],
            name=T("gini_series_label"), mode="lines+markers",
            line=dict(color="#1e3a5f", width=2.5),
        ), secondary_y=False)
        fig_summary.add_trace(go.Scatter(
            x=df_conc10["年份"], y=df_conc10["Top10%人口占比"],
            name=T("conc_series_label"), mode="lines+markers",
            line=dict(color="#c0392b", width=2.5),
        ), secondary_y=True)
        fig_summary.update_layout(
            **PLOTLY_LAYOUT,
            title=T("summary_chart_title"),
            height=380,
            hovermode="x unified",
            legend=dict(orientation="h", y=1.1),
        )
        fig_summary.update_yaxes(title_text=T("gini_yaxis"), secondary_y=False)
        fig_summary.update_yaxes(title_text=T("conc_yaxis"), secondary_y=True)
        st.plotly_chart(fig_summary, use_container_width=True)

    with col_map:
        diff_preview, diff_p99, _, _, diff_mean = render_diff_frame_image(YEARS[0], YEARS[-1], opacity)
        st.image(diff_preview, use_container_width=True,
                 caption=T("diff_map_caption", y0=YEARS[0], y1=YEARS[-1]))
        st.caption(T("diff_map_note", p99=diff_p99, mean=diff_mean))

    # --- Row 3: β-convergence scatter + top movers ---
    st.divider()
    col_beta, col_move = st.columns([3, 2])
    with col_beta:
        fig_beta = go.Figure()
        for region in ["东部", "中部", "西部", "东北"]:
            sub = beta_df[beta_df["区域"] == region]
            fig_beta.add_trace(go.Scatter(
                x=sub["ln初始密度"], y=sub["年均增长率%"],
                mode="markers+text", name=region_name(region),
                marker=dict(size=10, color=REGION_COLORS.get(region, "#999")),
                text=sub["省份"].str.replace("省|市|自治区|壮族|维吾尔|回族", "", regex=True),
                textposition="top center", textfont_size=8,
            ))
        # OLS line
        x_range = np.linspace(float(beta_df["ln初始密度"].min()) - 0.3,
                              float(beta_df["ln初始密度"].max()) + 0.3, 50)
        sl, intc, _, _, _ = sp_stats.linregress(beta_df["ln初始密度"], beta_df["年均增长率%"])
        fig_beta.add_trace(go.Scatter(
            x=x_range, y=intc + sl * x_range,
            mode="lines", name=T("ols_label", beta=sl),
            line=dict(color="grey", dash="dash", width=2),
        ))
        fig_beta.update_layout(
            **PLOTLY_LAYOUT,
            title=T("beta_chart_title", r2=beta_r2),
            height=380,
            xaxis_title=T("beta_xaxis", y0=YEARS[0]),
            yaxis_title=T("beta_yaxis"),
            legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(fig_beta, use_container_width=True)

    with col_move:
        st.markdown(T("top_movers"))
        movers = df_growth.nlargest(8, "绝对增长")
        for _, row in movers.iterrows():
            region_tag = region_name(REGION_MAP.get(row["省份"], ""))
            if lang == "zh":
                st.markdown(f"- **{row['省份']}** ({region_tag}) — +{row['绝对增长']:.0f} 人/km²，增长率 {row['增长率%']:+.1f}%")
            else:
                st.markdown(f"- **{row['省份']}** ({region_tag}) — +{row['绝对增长']:.0f} ppl/km², growth {row['增长率%']:+.1f}%")
        st.divider()
        st.markdown(T("top_losers"))
        losers = df_growth.nsmallest(5, "绝对增长")
        for _, row in losers.iterrows():
            region_tag = region_name(REGION_MAP.get(row["省份"], ""))
            if lang == "zh":
                st.markdown(f"- **{row['省份']}** ({region_tag}) — {row['绝对增长']:+.0f} 人/km²，增长率 {row['增长率%']:+.1f}%")
            else:
                st.markdown(f"- **{row['省份']}** ({region_tag}) — {row['绝对增长']:+.0f} ppl/km², growth {row['增长率%']:+.1f}%")


# ===================================================================
# TAB 2: National evolution — Agglomeration Dynamics
# ===================================================================
if _page == _nav_labels[1]:
    st.header(T("national_header"))
    st.caption(T("national_caption", y0=YEARS[0], y1=YEARS[-1]))

    # --- Animation section (preserved) ---
    ctrl_cols = st.columns([1.5, 1.5, 2])
    with ctrl_cols[0]:
        play_btn = st.button(T("play_btn"), key="national_anim_play")
    with ctrl_cols[1]:
        _speed_fmt = (lambda x: f"{x}s/帧") if lang == "zh" else (lambda x: f"{x}s/frame")
        anim_speed = st.selectbox(T("speed_label"), [0.3, 0.5, 1.0, 1.5], index=1,
                                  format_func=_speed_fmt, key="national_anim_speed")
    with ctrl_cols[2]:
        manual_year = st.select_slider(T("year_slider"), options=YEARS, value=2010,
                                       key="national_anim_manual")

    cumulative_stats = precompute_cumulative_diff_stats(YEARS[0])
    col_amap, col_astats = st.columns([5, 3])
    map_placeholder = col_amap.empty()
    stats_placeholder = col_astats.empty()

    def render_national_frame(yr):
        frame_img, p99_val, clipped_n, total_n, _ = render_diff_frame_image(YEARS[0], yr, opacity)
        s = cumulative_stats[yr]
        map_placeholder.image(frame_img,
                              caption=T("anim_frame_caption", yr=yr, y0=YEARS[0]),
                              use_container_width=True)
        with stats_placeholder.container():
            progress = (yr - YEARS[0]) / (YEARS[-1] - YEARS[0])
            st.progress(progress, text=f"**{yr}** / {YEARS[-1]}")
            nat_row_yr = df_national[df_national["年份"] == yr].iloc[0]
            c1, c2 = st.columns(2)
            c1.metric(T("anim_mean_change"), f"{s['mean_change']:+.1f} {'人/km²' if lang == 'zh' else 'ppl/km²'}")
            c2.metric(T("anim_max_increase"), f"{s['max_increase']:+.0f}")
            c3, c4 = st.columns(2)
            c3.metric(T("anim_increase_share"), f"{s['increase_share']:.1f}%")
            c4.metric(T("anim_decrease_share"), f"{s['decrease_share']:.1f}%")
            st.caption(T("anim_stats_caption",
                         pop=nat_row_yr['估算总人口(万)'] / 10000,
                         wt=nat_row_yr['加权平均密度'],
                         p99=p99_val,
                         clip=clipped_n / total_n * 100))
            yr_provs = sorted(filter_provs(province_stats[str(yr)], mainland_only),
                              key=lambda x: x["mean"], reverse=True)
            st.markdown(T("anim_top5", yr=yr))
            for i, p in enumerate(yr_provs[:5], 1):
                st.markdown(f"{i}. **{p['name']}** — {p['mean']:.0f} {'人/km²' if lang == 'zh' else 'ppl/km²'}")

    if play_btn:
        for yr in YEARS:
            render_national_frame(yr)
            _time.sleep(anim_speed)
    else:
        render_national_frame(manual_year)

    # --- Agglomeration indicators panel ---
    st.divider()
    st.subheader(T("agg_indicators"))

    col_gini, col_conc = st.columns(2)
    with col_gini:
        fig_gini = go.Figure()
        fig_gini.add_trace(go.Scatter(
            x=df_gini["年份"], y=df_gini["空间基尼系数"],
            mode="lines+markers", line=dict(color="#1e3a5f", width=2.5),
            fill="tozeroy", fillcolor="rgba(30,58,95,0.08)",
        ))
        fig_gini.update_layout(
            **PLOTLY_LAYOUT,
            title=T("gini_chart_title"),
            height=380,
            xaxis_title=T("year_label"), yaxis_title=T("gini_yaxis"),
            hovermode="x unified",
        )
        st.plotly_chart(fig_gini, use_container_width=True)
        st.caption(T("gini_caption"))

    with col_conc:
        fig_conc = go.Figure()
        fig_conc.add_trace(go.Scatter(
            x=df_conc10["年份"], y=df_conc10["Top10%人口占比"],
            mode="lines+markers", line=dict(color="#c0392b", width=2.5),
            fill="tozeroy", fillcolor="rgba(192,57,43,0.08)",
        ))
        fig_conc.update_layout(
            **PLOTLY_LAYOUT,
            title=T("conc_chart_title"),
            height=380,
            xaxis_title=T("year_label"), yaxis_title=T("conc_yaxis2"),
            hovermode="x unified",
        )
        st.plotly_chart(fig_conc, use_container_width=True)
        st.caption(T("conc_caption"))

    col_mp, col_sigma = st.columns(2)
    with col_mp:
        fig_mp = make_subplots(specs=[[{"secondary_y": True}]])
        fig_mp.add_trace(go.Scatter(
            x=df_national["年份"], y=df_national["加权平均密度"],
            name=T("mp_trace1"), mode="lines+markers",
            line=dict(color="#1e3a5f", width=2.5),
        ), secondary_y=False)
        fig_mp.add_trace(go.Scatter(
            x=df_national["年份"], y=df_national["高密度像素数"],
            name=T("mp_trace2"), mode="lines+markers",
            line=dict(color="#b8860b", width=2),
        ), secondary_y=True)
        fig_mp.update_layout(
            **PLOTLY_LAYOUT,
            title=T("mp_chart_title"),
            height=380,
            hovermode="x unified", legend=dict(orientation="h", y=1.12),
        )
        fig_mp.update_yaxes(title_text=T("mp_yaxis1"), secondary_y=False)
        fig_mp.update_yaxes(title_text=T("mp_yaxis2"), secondary_y=True)
        st.plotly_chart(fig_mp, use_container_width=True)

    with col_sigma:
        df_sigma = compute_sigma_convergence(province_stats, frozenset(exclude_for_rank))
        fig_sigma = go.Figure()
        fig_sigma.add_trace(go.Scatter(
            x=df_sigma["年份"], y=df_sigma["变异系数CV"],
            mode="lines+markers", line=dict(color="#2d6a4f", width=2.5),
            fill="tozeroy", fillcolor="rgba(45,106,79,0.08)",
        ))
        fig_sigma.update_layout(
            **PLOTLY_LAYOUT,
            title=T("sigma_chart_title"),
            height=380,
            xaxis_title=T("year_label"), yaxis_title=T("sigma_yaxis"),
            hovermode="x unified",
        )
        st.plotly_chart(fig_sigma, use_container_width=True)
        st.caption(T("sigma_caption"))


# ===================================================================
# TAB 3: Spatial patterns — Core-Periphery & Spatial Reallocation
# ===================================================================
if _page == _nav_labels[2]:
    st.header(T("spatial_header"))
    spatial_single, spatial_compare, spatial_centroid = st.tabs([
        T("spatial_sub_single"),
        T("spatial_sub_compare"),
        T("spatial_sub_centroid"),
    ])

    with spatial_single:
        year_sp = st.slider(T("year_slider2"), YEARS[0], YEARS[-1], 2010, 1, key="spatial_single_year")
        col_map, col_side = st.columns([2, 1])

        with col_map:
            data_sp, bounds_sp, _, _, _ = load_year(year_sp)
            img_sp = raster_to_image(data_sp)
            m = folium.Map(location=[35.0, 105.0], zoom_start=4, tiles="CartoDB positron")
            folium.raster_layers.ImageOverlay(image=img_sp, bounds=bounds_sp, opacity=opacity,
                                              name="栅格密度").add_to(m)
            folium.GeoJson(
                geojson_data,
                style_function=lambda x: {"fillColor": "transparent", "color": "#333",
                                          "weight": 0.8, "fillOpacity": 0},
            ).add_to(m)
            map_result = st_folium(m, height=520, use_container_width=True, key="spatial_single_map")

            if map_result and map_result.get("last_clicked"):
                lat_c = map_result["last_clicked"]["lat"]
                lng_c = map_result["last_clicked"]["lng"]
                ts = get_pixel_timeseries(raster_cube, cube_transform, cube_h, cube_w, lat_c, lng_c)
                fig_click = go.Figure(go.Scatter(
                    x=[t["年份"] for t in ts], y=[t["密度"] for t in ts],
                    mode="lines+markers", line=dict(color="#1e3a5f", width=2),
                    fill="tozeroy", fillcolor="rgba(30,58,95,0.1)",
                ))
                fig_click.update_layout(
                    **PLOTLY_LAYOUT,
                    title=T("click_chart_title", lat=lat_c, lng=lng_c),
                    height=280,
                    xaxis_title=T("click_xaxis"), yaxis_title=T("density_unit"),
                )
                st.plotly_chart(fig_click, use_container_width=True)

        with col_side:
            year_provs = sorted(filter_provs(province_stats[str(year_sp)], mainland_only),
                                key=lambda x: x["mean"], reverse=True)
            fig_rank = go.Figure(go.Bar(
                y=[p["name"] for p in year_provs[:15]][::-1],
                x=[p["mean"] for p in year_provs[:15]][::-1],
                orientation="h",
                marker=dict(color=[p["mean"] for p in year_provs[:15]][::-1], colorscale="YlOrBr"),
                text=[f"{p['mean']:.0f}" for p in year_provs[:15]][::-1],
                textposition="outside",
            ))
            fig_rank.update_layout(
                **{k: v for k, v in PLOTLY_LAYOUT.items() if k != "margin"},
                height=380,
                margin=dict(t=20, b=10, l=10, r=50),
                title=T("top15_title"), xaxis_title=T("density_unit"),
            )
            st.plotly_chart(fig_rank, use_container_width=True)

            valid_sp = data_sp[~np.isnan(data_sp) & (data_sp > 0)]
            fig_hist = px.histogram(
                x=valid_sp[valid_sp <= np.percentile(valid_sp, 99.5)],
                nbins=60,
                color_discrete_sequence=["#1e3a5f"],
                title=T("pixel_hist_title"),
            )
            fig_hist.update_layout(
                **PLOTLY_LAYOUT,
                height=280,
                xaxis_title=T("density_unit"), yaxis_title=T("pixel_count"),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    with spatial_compare:
        col_y1, col_y2 = st.columns(2)
        with col_y1:
            year_a = st.selectbox(T("compare_start"), YEARS, index=0, key="spatial_cmp_a")
        with col_y2:
            year_b = st.selectbox(T("compare_end"), YEARS, index=len(YEARS) - 1, key="spatial_cmp_b")

        if year_a != year_b:
            data_a, bounds, _, _, _ = load_year(year_a)
            data_b, _, _, _, _ = load_year(year_b)
            diff = data_b - data_a
            valid_diff = diff[~np.isnan(diff)]
            _, p99_val, clipped_n, total_n = diff_to_image_adaptive(diff)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric(T("compare_growth"), f"{np.sum(valid_diff > 0) / len(valid_diff) * 100:.1f}%")
            c2.metric(T("compare_decline"), f"{np.sum(valid_diff < 0) / len(valid_diff) * 100:.1f}%")
            c3.metric(T("compare_avg_change"), f"{np.nanmean(valid_diff):+.1f} {'人/km²' if lang == 'zh' else 'ppl/km²'}")
            c4.metric(T("compare_p99"), f"±{p99_val:.0f}")

            col_ma, col_diff, col_mb = st.columns(3)
            with col_ma:
                st.image(render_animation_frame_image(year_a, opacity), use_container_width=True,
                         caption=f"{year_a} {'年' if lang == 'zh' else ''}")
            with col_diff:
                st.image(render_diff_frame_image(year_a, year_b, opacity)[0], use_container_width=True,
                         caption=T("diff_map_caption", y0=year_a, y1=year_b))
                st.caption(T("compare_clip_note", n=clipped_n, pct=clipped_n / total_n * 100))
            with col_mb:
                st.image(render_animation_frame_image(year_b, opacity), use_container_width=True,
                         caption=f"{year_b} {'年' if lang == 'zh' else ''}")

            st.subheader(T("prov_change_header"))
            col_bar, col_hist = st.columns([3, 2])
            prov_a = {p["name"]: p for p in filter_provs(province_stats[str(year_a)], mainland_only)}
            prov_b = {p["name"]: p for p in filter_provs(province_stats[str(year_b)], mainland_only)}
            prov_diff_list = []
            for name in prov_a:
                if name in prov_b:
                    change = prov_b[name]["mean"] - prov_a[name]["mean"]
                    pct = (change / prov_a[name]["mean"] * 100) if prov_a[name]["mean"] > 0 else 0
                    pop_change = prov_b[name]["est_pop_wan"] - prov_a[name]["est_pop_wan"]
                    prov_diff_list.append({"省份": name, "密度变化": round(change, 1),
                                           "变化率%": round(pct, 1), "人口变化(万)": round(pop_change, 0)})
            prov_diff_list.sort(key=lambda x: x["密度变化"], reverse=True)

            with col_bar:
                fig_prov_diff = go.Figure(go.Bar(
                    y=[p["省份"] for p in prov_diff_list],
                    x=[p["密度变化"] for p in prov_diff_list],
                    orientation="h",
                    marker_color=["#c0392b" if p["密度变化"] > 0 else "#1e3a5f" for p in prov_diff_list],
                    text=[f"{p['密度变化']:+.0f}" for p in prov_diff_list],
                    textposition="outside",
                ))
                fig_prov_diff.add_vline(x=0, line_color="grey")
                fig_prov_diff.update_layout(
                    **PLOTLY_LAYOUT,
                    title=T("prov_diff_title"),
                    height=max(480, len(prov_diff_list) * 22),
                    margin=dict(t=40, b=10, l=10, r=50),
                    xaxis_title=T("density_unit"),
                )
                st.plotly_chart(fig_prov_diff, use_container_width=True)

            with col_hist:
                clipped_hist = valid_diff[(valid_diff > -p99_val * 1.5) & (valid_diff < p99_val * 1.5)]
                fig_hist = px.histogram(x=clipped_hist, nbins=100,
                                        title=T("pixel_change_hist"),
                                        color_discrete_sequence=["#64748b"])
                fig_hist.add_vline(x=0, line_dash="dash", line_color="#c0392b")
                fig_hist.update_layout(
                    **PLOTLY_LAYOUT,
                    height=280,
                    xaxis_title=T("density_unit"), yaxis_title=T("pixel_count"),
                )
                st.plotly_chart(fig_hist, use_container_width=True)
                st.dataframe(pd.DataFrame(prov_diff_list).head(8), use_container_width=True, hide_index=True)
        else:
            st.warning(T("diff_warning"))

    with spatial_centroid:
        st.subheader(T("centroid_header"))
        st.caption(T("centroid_caption"))

        col_centroid_map, col_centroid_info = st.columns([3, 2])
        with col_centroid_map:
            fig_centroid = go.Figure()
            fig_centroid.add_trace(go.Scattergeo(
                lon=df_centroid["重心经度"], lat=df_centroid["重心纬度"],
                mode="lines+markers+text",
                marker=dict(
                    size=df_centroid["年份"].apply(lambda y: 6 + (y - YEARS[0]) * 0.8),
                    color=df_centroid["年份"], colorscale=[[0, "#c8d6e5"], [0.5, "#5b7fa5"], [1, "#1e3a5f"]],
                    colorbar=dict(title=T("year_label")),
                ),
                text=df_centroid["年份"].astype(str),
                textposition="top right", textfont_size=8,
                line=dict(color="#333", width=1.5),
                hovertemplate=T("centroid_hover"),
            ))
            fig_centroid.update_geos(
                resolution=50,
                showcountries=True, countrycolor="grey",
                showsubunits=True,
                lonaxis_range=[100, 125], lataxis_range=[25, 42],
                projection_type="mercator",
            )
            fig_centroid.update_layout(
                **PLOTLY_LAYOUT,
                title=T("centroid_chart_title", y0=YEARS[0], y1=YEARS[-1]),
                height=480,
                margin=dict(t=40, b=10, l=10, r=10),
                geo=dict(bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_centroid, use_container_width=True)

        with col_centroid_info:
            st.markdown(T("centroid_start",
                          y0=YEARS[0],
                          lat=centroid_start["重心纬度"],
                          lon=centroid_start["重心经度"]))
            st.markdown(T("centroid_end",
                          y1=YEARS[-1],
                          lat=centroid_end["重心纬度"],
                          lon=centroid_end["重心经度"]))
            st.markdown(T("centroid_dir", dir=shift_dir))
            st.markdown(T("centroid_dist", km=shift_km))

            north_dir = T("shift_n") if dlat_km > 0 else T("shift_s")
            east_dir = T("shift_e") if dlon_km > 0 else T("shift_w")
            st.markdown(T("centroid_lat_dir", km=abs(dlat_km), dir=north_dir))
            st.markdown(T("centroid_lon_dir", km=abs(dlon_km), dir=east_dir))

            st.divider()
            st.markdown(T("centroid_meaning"))
            st.markdown(T("centroid_meaning_text"))

            # Centroid time series
            fig_latlon = make_subplots(specs=[[{"secondary_y": True}]])
            fig_latlon.add_trace(go.Scatter(
                x=df_centroid["年份"], y=df_centroid["重心纬度"],
                name=T("lat_label"), mode="lines+markers", line=dict(color="#1e3a5f", width=2),
            ), secondary_y=False)
            fig_latlon.add_trace(go.Scatter(
                x=df_centroid["年份"], y=df_centroid["重心经度"],
                name=T("lon_label"), mode="lines+markers", line=dict(color="#c0392b", width=2),
            ), secondary_y=True)
            fig_latlon.update_layout(
                **PLOTLY_LAYOUT,
                title=T("latlon_chart_title"), height=280,
                hovermode="x unified", legend=dict(orientation="h", y=1.15),
            )
            fig_latlon.update_yaxes(title_text=T("deg_n"), secondary_y=False)
            fig_latlon.update_yaxes(title_text=T("deg_e"), secondary_y=True)
            st.plotly_chart(fig_latlon, use_container_width=True)

        st.divider()
        st.subheader(T("region_shares_header"))
        st.caption(T("region_shares_caption"))

        df_regions = compute_regional_shares(province_stats, frozenset(exclude_for_rank))
        col_area, col_table = st.columns([3, 2])
        with col_area:
            fig_region = go.Figure()
            for region in ["东部", "中部", "西部", "东北"]:
                sub = df_regions[df_regions["区域"] == region]
                fig_region.add_trace(go.Scatter(
                    x=sub["年份"], y=sub["人口占比%"],
                    name=region_name(region), mode="lines+markers",
                    line=dict(color=REGION_COLORS.get(region, "#999"), width=2.5),
                ))
            fig_region.update_layout(
                **PLOTLY_LAYOUT,
                title=T("region_shares_chart_title"),
                height=380,
                xaxis_title=T("year_label"), yaxis_title=T("region_shares_yaxis"),
                hovermode="x unified",
                legend=dict(orientation="h", y=1.1),
            )
            st.plotly_chart(fig_region, use_container_width=True)

        with col_table:
            pivot = df_regions.pivot_table(index="年份", columns="区域", values="人口占比%")
            pivot = pivot.reindex(columns=["东部", "中部", "西部", "东北"])
            st.markdown(T("region_shares_table_title"))
            summary_rows = []
            for region in ["东部", "中部", "西部", "东北"]:
                r_start = pivot.loc[YEARS[0], region]
                r_end = pivot.loc[YEARS[-1], region]
                summary_rows.append({
                    T("region_col"): region_name(region),
                    f"{YEARS[0]}{'占比%' if lang == 'zh' else ' Share%'}": f"{r_start:.2f}",
                    f"{YEARS[-1]}{'占比%' if lang == 'zh' else ' Share%'}": f"{r_end:.2f}",
                    T("change_col"): f"{r_end - r_start:+.2f}",
                })
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

            st.divider()
            st.markdown(T("region_interp"))
            east_change = float(pivot.loc[YEARS[-1], "东部"]) - float(pivot.loc[YEARS[0], "东部"])
            if east_change > 0:
                st.markdown(T("east_rise", d=east_change))
            else:
                st.markdown(T("east_fall", d=east_change))


# ===================================================================
# TAB 4: Provincial heterogeneity — Regional Convergence
# ===================================================================
if _page == _nav_labels[3]:
    st.header(T("hetero_header"))
    st.caption(T("hetero_caption"))

    col_focus, col_peer, col_year = st.columns([2, 2, 1.3])
    with col_focus:
        focus_province = st.selectbox(T("focus_province"), available_names,
                                      index=focus_default_idx, key="hetero_focus")
    with col_peer:
        peer_candidates = [n for n in available_names if n != focus_province]
        peer_default = peer_candidates.index("江苏省") if "江苏省" in peer_candidates else 0
        compare_province = st.selectbox(T("compare_province"), peer_candidates,
                                        index=peer_default, key="hetero_peer")
    with col_year:
        year_dr = st.slider(T("focus_year"), YEARS[0], YEARS[-1], 2010, 1, key="hetero_year")

    scatter_mode = st.radio(
        T("scatter_mode"),
        [T("scatter_endpoint"), T("scatter_animation")],
        horizontal=True,
        key="hetero_scatter_mode",
    )

    if scatter_mode == T("scatter_endpoint"):
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
                mode="markers+text", name=region_name(region),
                marker=dict(
                    size=sub["2020人口(万)"].apply(lambda x: max(8, min(50, x / 300))),
                    color=REGION_COLORS.get(region, "#999"),
                    opacity=0.8,
                ),
                text=sub["标签"], textposition="top center", textfont_size=8,
                hovertemplate="%{customdata[0]}<br>"
                              + ("2002密度" if lang == "zh" else "2002 Density")
                              + ": %{x:.0f}<br>"
                              + ("增长率" if lang == "zh" else "Growth Rate")
                              + ": %{y:.1f}%<br>"
                              + ("2020人口" if lang == "zh" else "2020 Pop.")
                              + ": %{customdata[1]:.0f}"
                              + ("万" if lang == "zh" else "×10k")
                              + "<extra></extra>",
                customdata=list(zip(sub["省份"], sub["2020人口(万)"])),
            ))

        # Highlight focus province
        focus_row = scatter_df[scatter_df["省份"] == focus_province].iloc[0]
        fig_traj.add_trace(go.Scatter(
            x=[focus_row["2002年密度"]], y=[focus_row["增长率%"]],
            mode="markers",
            marker=dict(size=22, color="rgba(0,0,0,0)", line=dict(color="black", width=2.5)),
            name=T("focus_province_legend"), showlegend=False, hoverinfo="skip",
        ))

        # OLS regression line (β-convergence visual)
        x_log = np.log(scatter_df["2002年密度"].values)
        y_vals = scatter_df["增长率%"].values
        sl_h, intc_h, _, _, _ = sp_stats.linregress(x_log, y_vals)
        x_fit = np.linspace(scatter_df["2002年密度"].min() * 0.8, scatter_df["2002年密度"].max() * 1.2, 50)
        y_fit = intc_h + sl_h * np.log(x_fit)
        fig_traj.add_trace(go.Scatter(
            x=x_fit, y=y_fit, mode="lines",
            name=T("ols_trend", beta=sl_h),
            line=dict(color="grey", dash="dash", width=2),
        ))

        fig_traj.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5)
        fig_traj.update_layout(
            **PLOTLY_LAYOUT,
            title=T("beta_scatter_title"),
            height=480,
            xaxis_title=T("beta_scatter_xaxis"), yaxis_title=T("beta_scatter_yaxis"),
            xaxis_type="log",
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig_traj, use_container_width=True)
        direction_text = T("beta_caption_neg") if sl_h < 0 else T("beta_caption_pos")
        st.caption(T("beta_caption_full", beta=sl_h, direction=direction_text))
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
        fig_traj_anim.add_vline(x=float(df_growth["2002年密度"].median()), line_dash="dot",
                                line_color="#999", opacity=0.3)
        fig_traj_anim.update_layout(
            **PLOTLY_LAYOUT,
            title=T("anim_title", y0=YEARS[0]),
            height=480,
            xaxis_title=T("anim_xaxis"),
            yaxis_title=T("anim_yaxis", y0=YEARS[0]),
            xaxis_type="log",
            coloraxis_colorbar_title=T("abs_growth_label"),
            legend_title_text="",
        )
        fig_traj_anim.update_coloraxes(cmin=-color_abs, cmax=color_abs)
        if fig_traj_anim.layout.updatemenus:
            fig_traj_anim.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 700
            fig_traj_anim.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 350
        st.plotly_chart(fig_traj_anim, use_container_width=True)
        st.caption(T("anim_caption"))

    # Heatmap metric selector — map display labels back to internal column names
    heatmap_opt_mean = T("heatmap_opt_mean")
    heatmap_opt_growth = T("heatmap_opt_growth")
    heatmap_opt_yoy = T("heatmap_opt_yoy")
    matrix_metric_label = st.radio(T("heatmap_metric"),
                                   [heatmap_opt_mean, heatmap_opt_growth, heatmap_opt_yoy],
                                   horizontal=True, key="hetero_matrix_metric")
    # Map display label to internal key used in the logic below
    if matrix_metric_label == heatmap_opt_mean:
        matrix_metric = "平均密度"
    elif matrix_metric_label == heatmap_opt_growth:
        matrix_metric = "相对2002年增长率%"
    else:
        matrix_metric = "逐年变化"

    provs_2020 = sorted(filter_provs(province_stats["2020"], mainland_only),
                        key=lambda x: x["mean"], reverse=True)
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
        colorscale, zmin, zmax, bar_title = "OrRd", 0, 1200, T("heatmap_bar_density")
    elif matrix_metric == "相对2002年增长率%":
        zmax = max(abs(v) for row in z_matrix for v in row)
        colorscale, zmin, zmax, bar_title = "RdYlBu_r", -zmax * 0.3, zmax, T("heatmap_bar_pct")
    else:
        zmax = max(abs(v) for row in z_matrix for v in row if abs(v) < 500)
        colorscale, zmin, zmax, bar_title = "RdBu_r", -zmax, zmax, T("heatmap_bar_density")

    fig_hm = go.Figure(go.Heatmap(
        z=z_matrix, x=YEARS, y=prov_names_ordered, colorscale=colorscale, zmin=zmin, zmax=zmax,
        colorbar_title=bar_title,
        hovertemplate=("省份" if lang == "zh" else "Province")
                      + ": %{y}<br>"
                      + ("年份" if lang == "zh" else "Year")
                      + ": %{x}<br>"
                      + ("值" if lang == "zh" else "Value")
                      + ": %{z}<extra></extra>",
    ))
    fig_hm.update_layout(**{k: v for k, v in PLOTLY_LAYOUT.items() if k != "margin"},
                         height=max(480, len(prov_names_ordered) * 20),
                         margin=dict(t=20, b=20, l=10, r=10),
                         xaxis=dict(dtick=1), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_hm, use_container_width=True)

    # --- σ-convergence detail ---
    st.divider()
    st.subheader(T("sigma_detail_header"))
    col_sigma_detail, col_sigma_text = st.columns([5, 3])
    with col_sigma_detail:
        df_sigma_h = compute_sigma_convergence(province_stats, frozenset(exclude_for_rank))
        fig_sigma_h = make_subplots(specs=[[{"secondary_y": True}]])
        fig_sigma_h.add_trace(go.Scatter(
            x=df_sigma_h["年份"], y=df_sigma_h["省际标准差"],
            name=T("sigma_detail_trace1"), mode="lines+markers",
            line=dict(color="#1e3a5f", width=2.5),
        ), secondary_y=False)
        fig_sigma_h.add_trace(go.Scatter(
            x=df_sigma_h["年份"], y=df_sigma_h["变异系数CV"],
            name=T("sigma_detail_trace2"), mode="lines+markers",
            line=dict(color="#c0392b", width=2.5),
        ), secondary_y=True)
        fig_sigma_h.update_layout(
            **PLOTLY_LAYOUT,
            title=T("sigma_detail_title"),
            height=380,
            hovermode="x unified", legend=dict(orientation="h", y=1.12),
        )
        fig_sigma_h.update_yaxes(title_text=T("sigma_yaxis1"), secondary_y=False)
        fig_sigma_h.update_yaxes(title_text=T("cv_yaxis"), secondary_y=True)
        st.plotly_chart(fig_sigma_h, use_container_width=True)
    with col_sigma_text:
        cv_start = df_sigma_h[df_sigma_h["年份"] == YEARS[0]]["变异系数CV"].values[0]
        cv_end = df_sigma_h[df_sigma_h["年份"] == YEARS[-1]]["变异系数CV"].values[0]
        std_start = df_sigma_h[df_sigma_h["年份"] == YEARS[0]]["省际标准差"].values[0]
        std_end = df_sigma_h[df_sigma_h["年份"] == YEARS[-1]]["省际标准差"].values[0]
        st.markdown(T("sigma_judgment"))
        cv_dir = T("cv_converge") if cv_end < cv_start else T("cv_diverge")
        std_dir = T("std_shrink") if std_end < std_start else T("std_expand")
        st.markdown(f"- {'变异系数 CV' if lang == 'zh' else 'Coeff. of Variation CV'}: {cv_start:.4f} → {cv_end:.4f} ({cv_dir})")
        st.markdown(f"- {'标准差' if lang == 'zh' else 'Std. Dev.'}: {std_start:.2f} → {std_end:.2f} ({std_dir})")
        st.divider()
        st.markdown(T("sigma_meaning"))
        if cv_end < cv_start:
            st.markdown(T("sigma_conv_text"))
        else:
            st.markdown(T("sigma_div_text"))

    st.divider()
    st.subheader(T("focus_profile", prov=focus_province))

    prov_series = df_prov[df_prov["name"] == focus_province].sort_values("年份")
    prov_current = prov_series[prov_series["年份"] == year_dr].iloc[0]
    first_rank = rank_df[(rank_df["省份"] == focus_province) & (rank_df["年份"] == YEARS[0])]["排名"].values[0]
    cur_rank = rank_df[(rank_df["省份"] == focus_province) & (rank_df["年份"] == year_dr)]["排名"].values[0]
    total_growth = prov_series.iloc[-1]["mean"] - prov_series.iloc[0]["mean"]
    growth_rate = (total_growth / prov_series.iloc[0]["mean"] * 100) if prov_series.iloc[0]["mean"] > 0 else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(T("kpi_yr_density", yr=year_dr),
              f"{prov_current['mean']:.1f} {'人/km²' if lang == 'zh' else 'ppl/km²'}")
    k2.metric(T("kpi_pop_est"),
              f"{prov_current['est_pop_wan']/10000:.2f} {'亿' if lang == 'zh' else 'B'}")
    k3.metric(T("kpi_national_rank"),
              T("kpi_rank_val", rank=cur_rank),
              delta=f"{first_rank - cur_rank:+d} (vs {YEARS[0]})")
    k4.metric(T("kpi_total_growth"),
              f"{total_growth:+.1f} {'人/km²' if lang == 'zh' else 'ppl/km²'}")
    k5.metric(T("kpi_growth_rate"), f"{growth_rate:+.1f}%")

    col_ts, col_rank = st.columns([3, 2])
    with col_ts:
        fig_detail = make_subplots(specs=[[{"secondary_y": True}]])
        fig_detail.add_trace(go.Scatter(
            x=prov_series["年份"], y=prov_series["mean"],
            name=T("avg_density_trace"), mode="lines+markers",
            line=dict(color="#1e3a5f", width=2.5),
            fill="tozeroy", fillcolor="rgba(30,58,95,0.1)",
        ), secondary_y=False)
        fig_detail.add_trace(go.Scatter(
            x=prov_series["年份"], y=prov_series["est_pop_wan"] / 10000,
            name=T("pop_est_trace"), mode="lines+markers",
            line=dict(color="#c0392b", width=2),
        ), secondary_y=True)
        fig_detail.add_vline(x=year_dr, line_dash="dash", line_color="grey")
        fig_detail.update_layout(
            **PLOTLY_LAYOUT,
            title=T("province_trend_title", prov=focus_province),
            height=380,
            hovermode="x unified", legend=dict(orientation="h", y=1.1),
        )
        fig_detail.update_yaxes(title_text=T("density_unit"), secondary_y=False)
        fig_detail.update_yaxes(title_text=T("hundred_million"), secondary_y=True)
        st.plotly_chart(fig_detail, use_container_width=True)

    with col_rank:
        prov_rank_data = rank_df[rank_df["省份"] == focus_province]
        fig_bump = go.Figure(go.Scatter(
            x=prov_rank_data["年份"], y=prov_rank_data["排名"],
            mode="lines+markers+text",
            text=[f"#{r}" for r in prov_rank_data["排名"]],
            textposition="top center",
            line=dict(color="#1e3a5f", width=3), marker=dict(size=8),
        ))
        fig_bump.update_layout(
            **PLOTLY_LAYOUT,
            title=T("rank_chart_title", prov=focus_province),
            height=380,
            yaxis=dict(autorange="reversed", dtick=1, title=T("rank_yaxis")),
        )
        st.plotly_chart(fig_bump, use_container_width=True)

    col_stats, col_compare = st.columns([2, 3])
    with col_stats:
        metric_names = T("stats_table_metrics")
        st.dataframe(
            pd.DataFrame({
                T("stats_metric_col"): metric_names,
                T("stats_value_col"): [
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
            yoy_rows.append({
                T("yoy_year_col"): f"{int(prev['年份'])}→{int(cur['年份'])}",
                T("yoy_change_col"): f"{change:+.1f}",
                T("yoy_pct_col"): f"{pct:+.1f}%",
            })
        st.dataframe(pd.DataFrame(yoy_rows), use_container_width=True, height=250, hide_index=True)

    with col_compare:
        s1 = df_prov[df_prov["name"] == focus_province].sort_values("年份")
        s2 = df_prov[df_prov["name"] == compare_province].sort_values("年份")
        subplot_titles = T("compare_subplots")
        fig_cross = make_subplots(rows=1, cols=3, subplot_titles=subplot_titles)
        for idx, metric in enumerate(["mean", "est_pop_wan", "max"], 1):
            fig_cross.add_trace(go.Scatter(x=s1["年份"], y=s1[metric], name=focus_province,
                                           mode="lines+markers", line=dict(color="#1e3a5f", width=2),
                                           showlegend=(idx == 1)), row=1, col=idx)
            fig_cross.add_trace(go.Scatter(x=s2["年份"], y=s2[metric], name=compare_province,
                                           mode="lines+markers", line=dict(color="#c0392b", width=2),
                                           showlegend=(idx == 1)), row=1, col=idx)
        fig_cross.update_layout(
            **PLOTLY_LAYOUT,
            height=380,
            hovermode="x unified", legend=dict(orientation="h", y=1.1),
            title=f"{focus_province} vs {compare_province}",
        )
        st.plotly_chart(fig_cross, use_container_width=True)


# ===================================================================
# TAB 5: Research topics — Trade, Spatial, Market Integration focus
# ===================================================================
if _page == _nav_labels[4]:
    st.header(T("research_header"))
    st.caption(T("research_caption"))

    _topic_keys = [
        ("topic1", True),
        ("topic2", False),
        ("topic3", False),
        ("topic4", False),
        ("topic5", False),
        ("topic6", False),
    ]
    for _tk, _expanded in _topic_keys:
        with st.expander(T(f"{_tk}_title"), expanded=_expanded):
            _body = T(f"{_tk}_body")
            # Split body into main content and references
            if "**References**" in _body or "**参考文献**" in _body:
                _split_key = "**References**" if "**References**" in _body else "**参考文献**"
                _parts = _body.split(_split_key, 1)
                st.markdown(_parts[0])
                st.caption(f"{_split_key}{_parts[1]}")
            else:
                st.markdown(_body)

    st.divider()
    st.info(T("research_footer") % WORLDPOP_COLLECTION_URL)


# ===================================================================
# TAB 6: Data and methods
# ===================================================================
if _page == _nav_labels[5]:
    st.header(T("methods_header"))
    st.caption(T("methods_caption"))

    col_data, col_method = st.columns(2)
    with col_data:
        st.subheader(T("methods_data_sources"))
        st.markdown(T("methods_data_md",
                      col=WORLDPOP_COLLECTION_URL,
                      c2002=WORLDPOP_CHINA_2002_URL,
                      c2020=WORLDPOP_CHINA_2020_URL,
                      doi=WORLDPOP_DOI_URL,
                      cc=CC_BY_4_URL))
        st.subheader(T("methods_sample_scope"))
        st.markdown(T("methods_sample_md",
                      y0=YEARS[0], y1=YEARS[-1],
                      scope=scope_label_readable))

    with col_method:
        st.subheader(T("methods_processing"))
        st.markdown(T("methods_processing_md"))
        st.subheader(T("methods_limits"))
        st.markdown(T("methods_limits_md"))

    st.subheader(T("methods_variables"))
    var_names = T("methods_var_names")
    var_descs = T("methods_var_descs")
    st.dataframe(
        pd.DataFrame({
            T("methods_var_col"): var_names,
            T("methods_desc_col"): var_descs,
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader(T("methods_citation"))
    st.code(
        "WorldPop and CIESIN (2018). Global High Resolution Population Denominators Project. "
        "DOI: 10.5258/SOTON/WP00675",
        language="text",
    )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
scope_label = T("scope_short_mainland") if mainland_only else T("scope_short_all")
_fc1, _fc2 = st.columns([1, 1])
with _fc1:
    if lang == "zh":
        st.markdown(f"<span style='font-size:0.85rem;color:#6b6560;'>中国人口密度时空探索器 ({scope_label}) &nbsp;|&nbsp; Li Shen</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"<span style='font-size:0.85rem;color:#6b6560;'>China Population Density Explorer ({scope_label}) &nbsp;|&nbsp; Li Shen</span>", unsafe_allow_html=True)
with _fc2:
    st.markdown(
        f"<span style='font-size:0.85rem;color:#6b6560;'>"
        f"Data: <a href='{WORLDPOP_COLLECTION_URL}' target='_blank' style='color:#1e3a5f;'>WorldPop</a> &nbsp;|&nbsp; "
        f"DOI: <a href='{WORLDPOP_DOI_URL}' target='_blank' style='color:#1e3a5f;'>10.5258/SOTON/WP00675</a> &nbsp;|&nbsp; "
        f"<a href='{CC_BY_4_URL}' target='_blank' style='color:#1e3a5f;'>CC BY 4.0</a>"
        f"</span>",
        unsafe_allow_html=True,
    )
