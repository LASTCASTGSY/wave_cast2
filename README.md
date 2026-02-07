# CSDI v2: Wave Height Forecasting with ERA5 Spatial Covariates

A high-fidelity wave forecasting pipeline using Conditional Score-based Diffusion for Imputation (CSDI), enhanced with ERA5 atmospheric spatial covariates.

## Summary of Changes vs. V1

### 1. Deterministic Temporal Splitting
Replaced ratio-based splitting with strict chronological boundaries to prevent data leakage:
- **Train**: Oldest available data through December 31, 2022
- **Validation**: January 1, 2023 – December 31, 2023
- **Test**: January 1, 2024 – Present

###checking the data
python -c "import pickle; import pandas as pd; data = pickle.load(open('./data/wave/NDBC1_42001_processed.pk', 'rb')); df = pd.DataFrame(data['observed_data'], columns=data['feature_names']); df.insert(0, 'Timestamp', pd.to_datetime(data['timestamps'])); print(df.head(200).to_string())"

### 2. Multi-Horizon Forecasting
Support for multiple forecast horizons at 10-minute cadence (NDBC1):
- **6h**: 36 time steps
- **24h**: 144 time steps
- **48h**: 288 time steps
- **72h**: 432 time steps

Context window: 72 hours (432 steps)

### 3. ERA5 Spatial Covariate Integration
New `Wave_Dataset_ERA5_Forecasting` class with:
- Spatial grid extraction (3°×3° or 5°×5° centered on buoy)
- Variables: `u10`, `v10`, `msl` at 0.25° resolution
- **Strict temporal alignment** (causal constraint): ERA5 data only from [t-T_era : t]
- 2D CNN spatial encoder with cross-attention fusion

### 4. Enhanced Evaluation Metrics
- **Point Metrics**: RMSE, MAE per variable (un-normalized)
- **Probabilistic Metrics**: CRPS (Continuous Ranked Probability Score)
- **Calibration**: Empirical coverage of 90% prediction interval

### 5. Golden Run Artifact Generation
- `config.yaml`: Exact parameters used
- `model.pth`: Best-performing model checkpoint
- `visualizations/`: High-resolution forecast plots

## Installation

```bash
pip install torch numpy pandas matplotlib pyyaml tqdm
pip install linear-attention-transformer  # Optional, for linear attention
pip install xarray netCDF4  # For ERA5 data loading
```

## Directory Structure

```
csdi_v2/
├── config/
│   └── wave_base.yaml       # Default configuration
├── dataset_wave.py          # Dataset classes with temporal splits
├── main_model_wave.py       # CSDI models with ERA5 encoder
├── diff_models.py           # Diffusion backbone
├── exe_wave.py              # Execution script
├── utils.py                 # Training and evaluation utilities
└── README.md
```

## Usage

### Basic Imputation
```bash
python exe_wave.py --mode imputation --station 42001
```

### Single Horizon Forecasting
```bash
python exe_wave.py --mode forecasting --forecast_horizon 6 --station 42001
```

### Multi-Horizon Forecasting (6h, 24h, 48h, 72h)
```bash
python exe_wave.py --mode forecasting_multi --station 42001
```

### With ERA5 Covariates
```bash
python exe_wave.py --mode forecasting --use_era5 --era5_path ./data/era5 --forecast_horizon 24
```

### Golden Run (Full Pipeline)
```bash
python exe_wave.py --mode golden_run --station 42001
```

## Data Preparation

### NDBC Buoy Data
1. Download standard meteorological data from [NDBC](https://www.ndbc.noaa.gov/)
2. Preprocess using the provided function:

```python
from dataset_wave import preprocess_wave_data

preprocess_wave_data(
    raw_file="./data/42001h2010.txt",
    output_path="./data/wave",
    dataset_type="NDBC1",
    station="42001"
)
```

### ERA5 Data (Optional)
1. Download from [CDS](https://cds.climate.copernicus.eu/)
2. Required variables: `u10`, `v10`, `msl`
3. Save as NetCDF with dimensions: (time, latitude, longitude)

## Configuration

Key parameters in `config/wave_base.yaml`:

```yaml
train:
  epochs: 200
  batch_size: 16
  lr: 0.001

diffusion:
  layers: 4
  channels: 64
  nheads: 8
  num_steps: 50

forecasting:
  context_hours: 72
  horizons: [6, 24, 48, 72]

era5:
  enabled: false
  context_hours: 24
  spatial_window_deg: 3.0
```

## Model Architecture

### Standard CSDI (Imputation/Forecasting)
```
Input (B, 2, K, L) → Conv1d → ResidualBlocks → Output (B, K, L)
                         ↑
                    Side Info (time + feature embeddings)
```

### CSDI with ERA5 (Cross-Attention)
```
ERA5 Grid (B, T, C, H, W) → SpatialEncoder → (B, T, D)
                                                ↓
Buoy Data → CSDI → ResidualBlock + CrossAttention → Prediction
```

## Output Visualization

Forecast plots include:
- **Red points**: Observed context data
- **Blue crosses**: Ground truth targets
- **Green line**: Median prediction
- **Shaded green**: 90% prediction interval
- **Gray dashed**: Context/horizon boundary

## Mathematical Formulation

### Temporal Alignment (No-Cheating Rule)
For forecast origin $t$:
- Context $x$: Buoy data $[t - T_{ctx} : t]$
- Exogenous: ERA5 grids $[t - T_{era} : t]$ (causal only!)
- Target $y$: Future buoy values $[t+1 : t+H]$

### CRPS Calculation
$$\text{CRPS} = \frac{1}{|Q|} \sum_{q \in Q} \frac{\sum_{i} \rho_q(y_i - \hat{y}_i^{(q)}) \cdot \mathbf{1}_{mask}}{\sum_i |y_i| \cdot \mathbf{1}_{mask}}$$

where $\rho_q$ is the quantile loss function.

### 90% Coverage
$$\text{Coverage} = \frac{\sum_i \mathbf{1}[q_{0.05,i} \leq y_i \leq q_{0.95,i}] \cdot \mathbf{1}_{mask}}{\sum_i \mathbf{1}_{mask}}$$

## Feature Set

| Variable | Description | Units |
|----------|-------------|-------|
| WVHT | Significant Wave Height | m |
| WSPD | Wind Speed | m/s |
| WDIR | Wind Direction | degrees |
| DPD | Dominant Period | s |
| APD | Average Period | s |
| MWD | Mean Wave Direction | degrees |
| PRES | Atmospheric Pressure | hPa |
| ATMP | Air Temperature | °C |
| DEWP | Dew Point | °C |

## Citation

If you use this code, please cite:
- Original CSDI: Tashiro et al., "CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation" (NeurIPS 2021)
- MCD-TSF techniques if using ERA5 cross-attention

## License

MIT License
