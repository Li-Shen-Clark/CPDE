# China Population Density Spatiotemporal Explorer

**中国人口密度时空探索器 (2002-2020)**

An interactive Streamlit web application for exploring China's population density dynamics through the lens of spatial economics, market integration, and regional convergence.

Built on WorldPop UN-adjusted 1km population density rasters (downsampled to ~5km), covering 19 consecutive years (2002-2020) across 31 mainland provinces.

---

## Key Features

### Spatial Economics Analytics
- **Spatial Gini Coefficient** — pixel-level population inequality tracking over time
- **Population Concentration Index** — share of population in top 10% densest pixels
- **Population Centroid Trajectory** — weighted centroid drift direction and distance
- **Beta-Convergence Test** — OLS: ln(initial density) vs annual growth rate, with regional coloring (East/Central/West/Northeast)
- **Sigma-Convergence** — cross-sectional coefficient of variation (CV) time series

### Interactive Visualization
- **Folium Maps** with raster overlays and click-to-query pixel timeseries
- **Animated Spatial Reallocation** — year-by-year cumulative change maps with pre-rendered frames (flicker-free)
- **Regional Share Analysis** — East/Central/West/Northeast population share evolution
- **Provincial Deep Dive** — per-province density trends, rankings, and cross-province comparison

### Research Integration
- 6 research topic descriptions with methodology, data support, and literature references
- Topics ordered by relevance to trade, spatial economics, and market integration
- Each topic links to the platform's corresponding evidence panels

---

## Technical Stack

| Component | Technology |
|-----------|-----------|
| Web Framework | Streamlit |
| Raster Processing | rasterio, numpy |
| Spatial Statistics | rasterstats, geopandas, scipy |
| Interactive Maps | Folium + streamlit-folium |
| Charts | Plotly (scatter, heatmap, line, bar) |
| Animation | PIL pre-rendered frames + Python loop |
| Province Boundaries | GeoJSON (DataV.aliyun) |

---

## Data Pipeline

```
WorldPop 1km GeoTIFF (131MB each × 19 years)
        │
        ▼  preprocess.py
5km downsampled GeoTIFF (1.6MB each × 19 years)
        │
        ▼  preprocess_provinces.py
Province-level JSON stats ──────────────┐
National-level JSON stats ──────────────┤
        │                                │
        ▼  app.py                        ▼
   Raster Cube (19×900×1469)    Aggregated DataFrames
        │                                │
        └────────────┬───────────────────┘
                     ▼
        Streamlit Web Application
        ├── Summary (spatial economics narrative)
        ├── Agglomeration Dynamics (Gini, concentration, σ-convergence)
        ├── Core-Periphery Patterns (centroid, regional shares)
        ├── Regional Convergence (β-convergence, heatmaps)
        ├── Research Topics (trade/spatial focus)
        └── Data & Methods (variable definitions)
```

### Key Technical Decisions

1. **Latitude-corrected pixel area**: `area = (Δ° × 111.32)² × cos(lat)` instead of fixed 25 km², reducing population estimate error from ~39% to ~7%
2. **Dual-scope architecture**: mainland-only toggle flows through all views via `geometry_mask`
3. **Raster cube**: stacked (19, 900, 1469) numpy array for O(1) pixel timeseries lookup
4. **Pre-rendered animation frames**: PIL-based static frames eliminate Streamlit `st.rerun()` white-flash
5. **Adaptive colorscale**: P99-based symmetric bounds prevent extreme pixels from compressing the visual range

---

## Project Structure

```
china-population-density-explorer/
├── app.py                      # Main Streamlit application (~1600 lines)
├── preprocess.py               # 1km → 5km downsampling pipeline
├── preprocess_provinces.py     # Zonal statistics & national summary generation
├── china_provinces.geojson     # Province boundaries (34 features)
├── data_5km/                   # Processed data
│   ├── China_{year}_5km.tif    # 19 downsampled rasters (2002-2020)
│   ├── province_stats.json     # Per-province statistics (147KB)
│   └── national_stats.json     # National summary with dual scope (11KB)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/china-population-density-explorer.git
cd china-population-density-explorer

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

The app runs entirely on the included 5km data — no additional downloads needed.

---

## Reproducing from Raw Data

If you want to rebuild from the original 1km WorldPop rasters:

1. Download `China_{year}_1km_UNadj.tif` (2002-2020) from [WorldPop](https://hub.worldpop.org/geodata/listing?id=76)
2. Place them in the project root directory
3. Run the preprocessing pipeline:

```bash
python preprocess.py              # Downsample 1km → 5km
python preprocess_provinces.py    # Generate province & national stats
streamlit run app.py              # Launch the explorer
```

---

## Spatial Economics Indicators

| Indicator | Formula / Method | Interpretation |
|-----------|-----------------|----------------|
| Spatial Gini | Lorenz curve on pixel-level population (density × area) | Higher = more spatially concentrated |
| Top-K Concentration | Pop share in densest K% of pixels | Rising = agglomeration intensifying |
| Population Centroid | Σ(pop × coord) / Σ(pop) | Drift direction ≈ labor flow direction |
| β-Convergence | OLS slope of growth rate on ln(initial density) | Negative = catching-up; Positive = polarization |
| σ-Convergence | CV of cross-provincial density over time | Declining = provinces converging |
| Market Potential Proxy | Population-weighted mean density | Rising = expanding local market size |

---

## Data Source & Citation

**Data**: WorldPop UN-adjusted Population Density, 1km resolution, 2000-2020

- Official source: [WorldPop Global1 2000-2020](https://hub.worldpop.org/geodata/listing?id=76)
- DOI: [10.5258/SOTON/WP00675](https://doi.org/10.5258/SOTON/WP00675)
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

**Citation**:
```
WorldPop and CIESIN (2018). Global High Resolution Population Denominators Project.
DOI: 10.5258/SOTON/WP00675
```

---

## License

Code: MIT License. Data: CC BY 4.0 (WorldPop).
