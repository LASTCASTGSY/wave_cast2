"""
CSDI v2 Dataset Module for Wave Height Forecasting
===================================================
STATION 42002 CONFIGURATION
---------------------------
Train: 2017
Valid: 2019 (Partial data, ok for validation)
Test:  2018 (Golden Year - 99.9% Valid - GUARANTEES PLOTS)
"""

import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict, Union
import warnings
import math

# ============================================================================
# CONSTANTS
# ============================================================================

FEATURE_NAMES = ['WDIR', 'WSPD', 'WVHT', 'DPD', 'APD', 'MWD', 'PRES', 'ATMP', 'DEWP']
TARGET_DIM = len(FEATURE_NAMES)
WVHT_INDEX = FEATURE_NAMES.index('WVHT')  # Index 2

# Circular variables (degrees, wrap at 360)
CIRCULAR_VARS = ['WDIR', 'MWD']
CIRCULAR_INDICES = [FEATURE_NAMES.index(v) for v in CIRCULAR_VARS]

MINUTES_PER_STEP = 10
STEPS_PER_HOUR = 60 // MINUTES_PER_STEP

FORECAST_HORIZONS = {
    6: 6 * STEPS_PER_HOUR,
    24: 24 * STEPS_PER_HOUR,
    48: 48 * STEPS_PER_HOUR,
    72: 72 * STEPS_PER_HOUR,
}

CONTEXT_HOURS = 72
CONTEXT_STEPS = CONTEXT_HOURS * STEPS_PER_HOUR

# =========================================================
# DETERMINISTIC SPLIT BOUNDARIES (MAXIMUM DATA)
# =========================================================

# TRAIN: 13 Years of history (2010 -> 2022)
# Massive dataset for best possible learning
TRAIN_END_DATE = datetime(2022, 12, 31, 23, 59, 59)

# VALID: 2023 (Use the second-newest year for validation)
VAL_START_DATE = datetime(2023, 1, 1, 0, 0, 0)
VAL_END_DATE = datetime(2023, 12, 31, 23, 59, 59)

# TEST: 2024 (Test on the most recent data available)
TEST_START_DATE = datetime(2024, 1, 1, 0, 0, 0)

ERA5_VARIABLES = ['u10', 'v10', 'msl']
ERA5_RESOLUTION = 0.25


# ============================================================================
# CIRCULAR ENCODING/DECODING UTILITIES
# ============================================================================

def encode_circular(degrees: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Encode degrees to (sin, cos) representation."""
    radians = np.deg2rad(degrees)
    return np.sin(radians), np.cos(radians)

def decode_circular(sin_vals: np.ndarray, cos_vals: np.ndarray) -> np.ndarray:
    """Decode (sin, cos) back to degrees [0, 360)."""
    radians = np.arctan2(sin_vals, cos_vals)
    degrees = np.rad2deg(radians)
    degrees = np.mod(degrees, 360.0)
    return degrees

def angular_error(pred_deg: np.ndarray, true_deg: np.ndarray) -> np.ndarray:
    """Compute angular error in degrees (handles wrap-around)."""
    diff = np.abs(pred_deg - true_deg)
    return np.minimum(diff, 360.0 - diff)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def datetime_to_index(timestamps: np.ndarray, target_date: datetime) -> int:
    if isinstance(timestamps[0], (np.datetime64, pd.Timestamp)):
        target = np.datetime64(target_date)
    else:
        target = target_date
    idx = np.searchsorted(timestamps, target)
    return min(idx, len(timestamps) - 1)

def compute_missing_ratio(mask: np.ndarray, start_idx: int, end_idx: int) -> float:
    segment = mask[start_idx:end_idx]
    if len(segment) == 0: return 1.0
    return 1.0 - segment.mean()

def validate_sample(mask: np.ndarray, start_idx: int, length: int, 
                   context_steps: int, min_context_ratio: float = 0.3) -> bool:
    """
    Check if sample has enough valid data.
    FIX: Accepts 'context_steps' explicitly to handle NDBC1 vs NDBC2 correctly.
    """
    if start_idx + length > len(mask): return False
    context_mask = mask[start_idx:start_idx + context_steps, WVHT_INDEX]
    if len(context_mask) == 0: return False
    return context_mask.mean() >= min_context_ratio


# ============================================================================
# DATASET CLASSES
# ============================================================================

class Wave_Dataset(Dataset):
    def __init__(self, dataset_type: str = "NDBC1", mode: str = "train",
                 station: str = "42001", data_path: str = "./data/wave",
                 eval_length: int = None):
        self.dataset_type = dataset_type
        self.mode = mode
        self.station = station
        self.data_path = Path(data_path)
        
        if dataset_type == "NDBC1": self.steps_per_hour = 6
        else: self.steps_per_hour = 1
            
        if eval_length is None: self.eval_length = 72 * self.steps_per_hour
        else: self.eval_length = eval_length
            
        self.context_steps = 72 * self.steps_per_hour 
        self.feature_names = FEATURE_NAMES
        self.target_dim = TARGET_DIM
        self._load_data()
        self._create_deterministic_splits()
    
    def _load_data(self):
        data_file = self.data_path / f"{self.dataset_type}_{self.station}_processed.pk"
        if not data_file.exists():
            raise FileNotFoundError(f"Processed data file not found: {data_file}")
        with open(data_file, "rb") as f:
            data_dict = pickle.load(f)
            self.observed_data = data_dict['observed_data']
            self.observed_mask = data_dict['observed_mask']
            self.train_mean = data_dict['train_mean']
            self.train_std = data_dict['train_std']
            
            # Handle circular encoding metadata
            self.use_circular_encoding = data_dict.get('use_circular_encoding', False)
            # DEBUG: Print what we loaded
            print(f"\n[DEBUG] Loading data for {self.mode}:")
            print(f"  File loaded: {'imputed' if imputed_file.exists() else 'processed'}")
            print(f"  Data shape: {self.main_data.shape}")
            print(f"  use_circular_encoding in file: {data_dict.get('use_circular_encoding', 'NOT FOUND')}")
            print(f"  feature_names in file: {data_dict.get('feature_names', 'NOT FOUND')}")
            print(f"  expanded_feature_names in file: {data_dict.get('expanded_feature_names', 'NOT FOUND')}")
        

            self.original_feature_names = data_dict.get('feature_names', FEATURE_NAMES)
            self.expanded_feature_names = data_dict.get('expanded_feature_names', self.original_feature_names)
            self.circular_var_map = data_dict.get('circular_var_map', {})
            self.feature_to_expanded = data_dict.get('feature_to_expanded', {})
            
            # Update target_dim if using circular encoding
            if self.use_circular_encoding:
                self.target_dim = len(self.expanded_feature_names)
            else:
                self.target_dim = len(self.original_feature_names)
            
            if 'timestamps' in data_dict:
                self.timestamps = data_dict['timestamps']
            else:
                start_date = datetime(2010, 1, 1)
                self.timestamps = np.array([
                    start_date + timedelta(minutes=i * (60 // self.steps_per_hour))
                    for i in range(len(self.observed_data))
                ])
        
        # Debug: Print feature info
        print(f"\nDataset loaded: {self.mode}")
        print(f"  Original features: {len(self.original_feature_names)}")
        print(f"  Expanded features: {len(self.expanded_feature_names)}")
        print(f"  Circular encoding: {self.use_circular_encoding}")
        if self.use_circular_encoding:
            print(f"  Circular vars: {[k for k in self.circular_var_map.keys()]}")
    
    def _create_deterministic_splits(self):
        train_end_idx = datetime_to_index(self.timestamps, TRAIN_END_DATE)
        val_start_idx = datetime_to_index(self.timestamps, VAL_START_DATE)
        val_end_idx = datetime_to_index(self.timestamps, VAL_END_DATE)
        test_start_idx = datetime_to_index(self.timestamps, TEST_START_DATE)
        
        print(f"\n{self.mode.upper()} Split (42002 Optimized):")
        
        if self.mode == "train":
            start, end = 0, train_end_idx - self.eval_length + 1
            self.use_index = np.arange(start, max(start + 1, end), 1)
        elif self.mode == "valid":
            start, end = val_start_idx, val_end_idx - self.eval_length + 1
            self.use_index = np.arange(start, max(start + 1, end), 1)
        elif self.mode == "test":
            start, end = test_start_idx, len(self.timestamps) - self.eval_length + 1
            self.use_index = np.arange(start, max(start + 1, end), self.eval_length)
        
        self.cut_length = [0] * len(self.use_index)
        print(f"  Samples: {len(self.use_index)}")
    
    def __getitem__(self, org_index):
        index = self.use_index[org_index]
        obs_data = self.observed_data[index:index + self.eval_length]
        obs_mask = self.observed_mask[index:index + self.eval_length]
        if self.mode == "train":
            gt_mask = obs_mask.copy()
        else:
            gt_mask = obs_mask.copy()
            mask_window = 18 * (1 if self.dataset_type == "NDBC2" else 6)
            forecast_window = 6 * (1 if self.dataset_type == "NDBC2" else 6)
            for t in range(0, self.eval_length, mask_window):
                if t + forecast_window <= self.eval_length:
                    gt_mask[t:t+forecast_window, WVHT_INDEX] = 0
            gt_mask = gt_mask * obs_mask
        timepoints = np.arange(self.eval_length) * 1.0
        return {
            'observed_data': obs_data, 'observed_mask': obs_mask,
            'gt_mask': gt_mask, 'timepoints': timepoints,
            'cut_length': self.cut_length[org_index], 'hist_mask': obs_mask.copy()
        }
    
    def __len__(self): return len(self.use_index)


class Wave_Dataset_Forecasting(Dataset):
    def __init__(self, dataset_type: str = "NDBC1", mode: str = "train",
                 station: str = "42001", data_path: str = "./data/wave",
                 forecast_horizon_hours: int = 6, context_hours: int = CONTEXT_HOURS,
                 min_context_ratio: float = 0.3, stride: int = None):
        self.dataset_type = dataset_type
        self.mode = mode
        self.station = station
        self.data_path = Path(data_path)
        self.min_context_ratio = min_context_ratio
        
        if dataset_type == "NDBC1": self.steps_per_hour = STEPS_PER_HOUR
        else: self.steps_per_hour = 1
        
        self.context_steps = context_hours * self.steps_per_hour
        self.horizon_steps = forecast_horizon_hours * self.steps_per_hour
        self.seq_length = self.context_steps + self.horizon_steps
        
        if stride is None: self.stride = 1 if mode == "train" else self.horizon_steps
        else: self.stride = stride
        
        self._load_data()
        self._create_deterministic_splits()
    
    def _load_data(self):
        imputed_file = self.data_path / f"{self.dataset_type}_{self.station}_imputed.pk"
        processed_file = self.data_path / f"{self.dataset_type}_{self.station}_processed.pk"
        
        # DEBUG: Check which file exists and will be loaded
        file_loaded = None
        if imputed_file.exists():
            print(f"[WARNING] Imputed file exists: {imputed_file}")
            print(f"  This will be loaded instead of processed file (may not have circular encoding metadata)")
            with open(imputed_file, "rb") as f:
                data_dict = pickle.load(f)
                self.main_data = data_dict['imputed_data']
                self.mask_data = np.ones_like(self.main_data)
                self.train_mean = data_dict['train_mean']
                self.train_std = data_dict['train_std']
                file_loaded = "imputed"
        elif processed_file.exists():
            with open(processed_file, "rb") as f:
                data_dict = pickle.load(f)
                self.main_data = data_dict['observed_data']
                self.mask_data = data_dict['observed_mask']
                self.train_mean = data_dict['train_mean']
                self.train_std = data_dict['train_std']
                file_loaded = "processed"
        else:
            raise FileNotFoundError(f"No data found at {self.data_path}")
        
        # Handle circular encoding metadata
        self.use_circular_encoding = data_dict.get('use_circular_encoding', False)
        self.original_feature_names = data_dict.get('feature_names', FEATURE_NAMES)
        self.expanded_feature_names = data_dict.get('expanded_feature_names', self.original_feature_names)
        self.circular_var_map = data_dict.get('circular_var_map', {})
        self.feature_to_expanded = data_dict.get('feature_to_expanded', {})
        
        # DEBUG: Print what was loaded
        print(f"\n[DEBUG {self.mode}] Data loading:")
        print(f"  File loaded: {file_loaded}")
        print(f"  Data shape: {self.main_data.shape}")
        print(f"  use_circular_encoding in file: {data_dict.get('use_circular_encoding', 'NOT FOUND')}")
        if 'expanded_feature_names' in data_dict:
            print(f"  expanded_feature_names count: {len(data_dict['expanded_feature_names'])}")
        print(f"  Final use_circular_encoding: {self.use_circular_encoding}")
        print(f"  Final target_dim: {len(self.expanded_feature_names) if self.use_circular_encoding else len(self.original_feature_names)}")
        
        # Update target_dim if using circular encoding
        if self.use_circular_encoding:
            self.target_dim = len(self.expanded_feature_names)
        else:
            self.target_dim = len(self.original_feature_names)
        
        if 'timestamps' in data_dict:
            self.timestamps = pd.to_datetime(data_dict['timestamps'])
        else:
            start = datetime(2010, 1, 1)
            freq = f"{60 // self.steps_per_hour}min"
            self.timestamps = pd.date_range(start, periods=len(self.main_data), freq=freq)
    
    def _create_deterministic_splits(self):
        train_end_idx = datetime_to_index(self.timestamps.values, TRAIN_END_DATE)
        val_start_idx = datetime_to_index(self.timestamps.values, VAL_START_DATE)
        val_end_idx = datetime_to_index(self.timestamps.values, VAL_END_DATE)
        test_start_idx = datetime_to_index(self.timestamps.values, TEST_START_DATE)
        
        if self.mode == "train":
            start, end = 0, train_end_idx - self.seq_length + 1
        elif self.mode == "valid":
            start, end = val_start_idx, val_end_idx - self.seq_length + 1
        elif self.mode == "test":
            start, end = test_start_idx, len(self.timestamps) - self.seq_length + 1
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        candidates = np.arange(start, max(start + 1, end), self.stride)
        
        # Check context window size based on dataset type
        self.use_index = np.array([
            idx for idx in candidates
            if validate_sample(self.mask_data, idx, self.seq_length, self.context_steps, self.min_context_ratio)
        ])
        
        print(f"{self.mode.upper()} (42002 Optimized): {len(self.use_index)} valid samples")
        
        # TASK 4: Sanity prints for dataset configuration
        print(f"\n{'='*60}")
        print(f"DATASET CONFIGURATION: {self.mode.upper()}")
        print(f"{'='*60}")
        print(f"  dataset_type: {self.dataset_type}")
        print(f"  steps_per_hour: {self.steps_per_hour}")
        print(f"  context_steps: {self.context_steps}")
        print(f"  horizon_steps: {self.horizon_steps}")
        print(f"  seq_length: {self.seq_length}")
        print(f"  timestamps range: {self.timestamps[0]} to {self.timestamps[-1]}")
        print(f"  num_samples: {len(self.use_index)}")
        print(f"  target_dim: {self.target_dim}")
        print(f"  use_circular_encoding: {self.use_circular_encoding}")
        if self.use_circular_encoding:
            print(f"  expanded_features: {len(self.expanded_feature_names)}")
        print(f"{'='*60}\n")
    
    def __getitem__(self, org_index: int) -> Dict[str, np.ndarray]:
        index = self.use_index[org_index]
        data = self.main_data[index:index + self.seq_length].copy()
        mask = self.mask_data[index:index + self.seq_length].copy()
        
        # Initialize gt_mask: context is observed (conditioning), horizon is masked
        gt_mask = np.zeros_like(mask)
        gt_mask[:self.context_steps] = mask[:self.context_steps]
        
        # TASK 2: Only predict WVHT in forecast horizon (not all variables)
        # Derive WVHT index safely (handles circular encoding)
        if self.use_circular_encoding:
            # With circular encoding, find WVHT in expanded feature names
            if 'WVHT' in self.feature_to_expanded:
                wvht_idx = self.feature_to_expanded['WVHT'][0]
            else:
                # Fallback: count position accounting for circular vars
                wvht_idx = 0
                for feat_name in self.expanded_feature_names:
                    if feat_name == 'WVHT':
                        break
                    wvht_idx += 1
                # Safety check
                if wvht_idx >= len(self.expanded_feature_names):
                    raise ValueError(f"WVHT not found in expanded features: {self.expanded_feature_names}")
        else:
            # Without circular encoding, use original feature index
            wvht_idx = self.original_feature_names.index('WVHT')
        
        # observed_mask: only WVHT is observed in horizon (for scoring), all others masked
        observed_mask = mask.copy()
        observed_mask[self.context_steps:, :] = 0.0  # Mask all variables in horizon
        observed_mask[self.context_steps:, wvht_idx] = mask[self.context_steps:, wvht_idx]  # Unmask WVHT in horizon
        
        # DEBUG: Verify masks are correct (only print once per dataset)
        if not hasattr(self, '_mask_debug_printed'):
            print(f"\n[DEBUG {self.mode}] Mask verification (first sample):")
            print(f"  gt_mask context sum: {gt_mask[:self.context_steps].sum():.0f}")
            print(f"  gt_mask horizon sum: {gt_mask[self.context_steps:].sum():.0f} (should be ~0)")
            print(f"  observed_mask context sum: {observed_mask[:self.context_steps].sum():.0f}")
            print(f"  observed_mask horizon sum: {observed_mask[self.context_steps:].sum():.0f}")
            print(f"  observed_mask horizon WVHT sum: {observed_mask[self.context_steps:, wvht_idx].sum():.0f}")
            print(f"  WVHT index: {wvht_idx}")
            self._mask_debug_printed = True
        
        timepoints = np.arange(self.seq_length).astype(np.float32)
        
        return {
            'observed_data': data.astype(np.float32),
            'observed_mask': observed_mask.astype(np.float32),
            'gt_mask': gt_mask.astype(np.float32),
            'timepoints': timepoints,
            'context_end_idx': self.context_steps,
            'horizon_steps': self.horizon_steps,
        }
    
    def __len__(self) -> int: return len(self.use_index)
    def get_unnormalization_params(self): return self.train_mean, self.train_std


class Wave_Dataset_ERA5_Forecasting(Dataset):
    def __init__(self, dataset_type: str = "NDBC1", mode: str = "train",
                 station: str = "42001", data_path: str = "./data/wave",
                 era5_path: str = "./data/era5", forecast_horizon_hours: int = 6,
                 context_hours: int = CONTEXT_HOURS, era5_context_hours: int = 24,
                 spatial_window_deg: float = 3.0, min_context_ratio: float = 0.3,
                 stride: int = None):
        self.dataset_type = dataset_type
        self.mode = mode
        self.station = station
        self.data_path = Path(data_path)
        self.era5_path = Path(era5_path)
        self.min_context_ratio = min_context_ratio
        self.spatial_window_deg = spatial_window_deg
        
        if dataset_type == "NDBC1": self.steps_per_hour = STEPS_PER_HOUR
        else: self.steps_per_hour = 1
        
        self.context_steps = context_hours * self.steps_per_hour
        self.horizon_steps = forecast_horizon_hours * self.steps_per_hour
        self.seq_length = self.context_steps + self.horizon_steps
        self.era5_context_hours = era5_context_hours
        
        if stride is None: self.stride = 1 if mode == "train" else self.horizon_steps
        else: self.stride = stride
        
        self._load_buoy_data()
        self._load_era5_data()
        self._create_deterministic_splits()
    
    def _load_buoy_data(self):
        processed_file = self.data_path / f"{self.dataset_type}_{self.station}_processed.pk"
        imputed_file = self.data_path / f"{self.dataset_type}_{self.station}_imputed.pk"
        if imputed_file.exists():
            with open(imputed_file, "rb") as f:
                data_dict = pickle.load(f)
                self.buoy_data = data_dict['imputed_data']
                self.buoy_mask = np.ones_like(self.buoy_data)
        elif processed_file.exists():
            with open(processed_file, "rb") as f:
                data_dict = pickle.load(f)
                self.buoy_data = data_dict['observed_data']
                self.buoy_mask = data_dict['observed_mask']
        else: raise FileNotFoundError(f"No buoy data found at {self.data_path}")
        self.train_mean = data_dict['train_mean']
        self.train_std = data_dict['train_std']
        
        # Handle circular encoding metadata
        self.use_circular_encoding = data_dict.get('use_circular_encoding', False)
        self.original_feature_names = data_dict.get('feature_names', FEATURE_NAMES)
        self.expanded_feature_names = data_dict.get('expanded_feature_names', self.original_feature_names)
        self.circular_var_map = data_dict.get('circular_var_map', {})
        self.feature_to_expanded = data_dict.get('feature_to_expanded', {})
        
        # Update target_dim if using circular encoding
        if self.use_circular_encoding:
            self.target_dim = len(self.expanded_feature_names)
        else:
            self.target_dim = len(self.original_feature_names)
        
        if 'timestamps' in data_dict: self.timestamps = pd.to_datetime(data_dict['timestamps'])
        else: self.timestamps = pd.date_range(datetime(2010,1,1), periods=len(self.buoy_data), freq=f"{60//self.steps_per_hour}min")

    def _load_era5_data(self):
        self.era5_available = False
        era5_file = self.era5_path / f"era5_{self.station}.nc"
        try:
            import xarray as xr
            if era5_file.exists():
                ds = xr.open_dataset(era5_file)
                self.era5_timestamps = pd.to_datetime(ds.time.values)
                self.era5_lats = ds.latitude.values
                self.era5_lons = ds.longitude.values
                era5_arrays = []
                for var in ERA5_VARIABLES:
                    if var in ds: era5_arrays.append(ds[var].values)
                if era5_arrays:
                    self.era5_data = np.stack(era5_arrays, axis=1)
                    self.era5_available = True
                    self._compute_spatial_window()
                ds.close()
        except Exception: pass

    def _compute_spatial_window(self):
        buoy_lat, buoy_lon = 25.888, -89.658 # Default 42001
        half_window = self.spatial_window_deg / 2
        lat_mask = (self.era5_lats >= buoy_lat - half_window) & (self.era5_lats <= buoy_lat + half_window)
        lon_mask = (self.era5_lons >= buoy_lon - half_window) & (self.era5_lons <= buoy_lon + half_window)
        self.era5_lat_indices = np.where(lat_mask)[0]
        self.era5_lon_indices = np.where(lon_mask)[0]
        self.era5_h_size = len(self.era5_lat_indices)
        self.era5_w_size = len(self.era5_lon_indices)

    def _create_deterministic_splits(self):
        train_end_idx = datetime_to_index(self.timestamps.values, TRAIN_END_DATE)
        val_start_idx = datetime_to_index(self.timestamps.values, VAL_START_DATE)
        val_end_idx = datetime_to_index(self.timestamps.values, VAL_END_DATE)
        test_start_idx = datetime_to_index(self.timestamps.values, TEST_START_DATE)
        
        if self.mode == "train": start, end = 0, train_end_idx - self.seq_length + 1
        elif self.mode == "valid": start, end = val_start_idx, val_end_idx - self.seq_length + 1
        elif self.mode == "test": start, end = test_start_idx, len(self.timestamps) - self.seq_length + 1
        else: raise ValueError(f"Unknown mode: {self.mode}")
        
        candidates = np.arange(start, max(start + 1, end), self.stride)
        self.use_index = np.array([idx for idx in candidates if validate_sample(self.buoy_mask, idx, self.seq_length, self.context_steps, self.min_context_ratio)])
        print(f"{self.mode.upper()}: {len(self.use_index)} valid samples")

    def _get_era5_context(self, buoy_timestamp_end):
        if not self.era5_available:
            return np.zeros((self.era5_context_hours, len(ERA5_VARIABLES), 13, 13), dtype=np.float32), np.zeros(self.era5_context_hours)
        era5_end_idx = np.searchsorted(self.era5_timestamps, buoy_timestamp_end, side='right') - 1
        era5_start_idx = max(0, era5_end_idx - self.era5_context_hours + 1)
        era5_slice = self.era5_data[era5_start_idx:era5_end_idx + 1, :, self.era5_lat_indices[0]:self.era5_lat_indices[-1] + 1, self.era5_lon_indices[0]:self.era5_lon_indices[-1] + 1]
        actual_steps = era5_slice.shape[0]
        era5_mask = np.ones(self.era5_context_hours)
        if actual_steps < self.era5_context_hours:
            pad_size = self.era5_context_hours - actual_steps
            era5_slice = np.pad(era5_slice, ((pad_size, 0), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
            era5_mask[:pad_size] = 0
        return era5_slice.astype(np.float32), era5_mask.astype(np.float32)

    def __getitem__(self, org_index):
        index = self.use_index[org_index]
        buoy_data = self.buoy_data[index:index + self.seq_length].copy()
        buoy_mask = self.buoy_mask[index:index + self.seq_length].copy()
        gt_mask = np.zeros_like(buoy_mask)
        gt_mask[:self.context_steps] = buoy_mask[:self.context_steps]
        context_end_timestamp = self.timestamps[index + self.context_steps - 1]
        era5_data, era5_mask = self._get_era5_context(context_end_timestamp)
        return {
            'observed_data': buoy_data.astype(np.float32), 'observed_mask': buoy_mask.astype(np.float32),
            'gt_mask': gt_mask.astype(np.float32), 'era5_data': era5_data, 'era5_mask': era5_mask,
            'timepoints': np.arange(self.seq_length).astype(np.float32),
            'context_end_idx': self.context_steps, 'horizon_steps': self.horizon_steps
        }
    
    def __len__(self): return len(self.use_index)
    def get_era5_shape(self): return (self.era5_context_hours, len(ERA5_VARIABLES), self.era5_h_size, self.era5_w_size) if self.era5_available else (self.era5_context_hours, len(ERA5_VARIABLES), 13, 13)


def get_dataloader(datatype="NDBC1", device='cuda:0', batch_size=16, station="42001", data_path="./data/wave", task="imputation", forecast_horizon_hours=6, use_era5=False, era5_path="./data/era5", era5_context_hours=24, spatial_window_deg=3.0, num_workers=4, **kwargs):
    if task == "imputation":
        DatasetClass = Wave_Dataset
        dataset_kwargs = {'dataset_type': datatype, 'station': station, 'data_path': data_path}
    elif task == "forecasting":
        if use_era5:
            DatasetClass = Wave_Dataset_ERA5_Forecasting
            dataset_kwargs = {'dataset_type': datatype, 'station': station, 'data_path': data_path, 'era5_path': era5_path, 'forecast_horizon_hours': forecast_horizon_hours, 'era5_context_hours': era5_context_hours, 'spatial_window_deg': spatial_window_deg}
        else:
            DatasetClass = Wave_Dataset_Forecasting
            dataset_kwargs = {'dataset_type': datatype, 'station': station, 'data_path': data_path, 'forecast_horizon_hours': forecast_horizon_hours}
    else: raise ValueError(f"Unknown task: {task}")
    
    train_dataset = DatasetClass(mode='train', **dataset_kwargs)
    valid_dataset = DatasetClass(mode='valid', **dataset_kwargs)
    test_dataset = DatasetClass(mode='test', **dataset_kwargs)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    scaler = torch.from_numpy(train_dataset.train_std).to(device).float()
    mean_scaler = torch.from_numpy(train_dataset.train_mean).to(device).float()
    return train_loader, valid_loader, test_loader, scaler, mean_scaler

def preprocess_wave_data(raw_file, output_path, dataset_type="NDBC1", station="42001", use_circular_encoding=True):
    """
    Preprocess NDBC data with optional circular encoding for WDIR/MWD.
    
    Args:
        use_circular_encoding: If True, encode WDIR/MWD as (sin, cos) pairs
    """
    warnings.filterwarnings('ignore')
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(raw_file, delim_whitespace=True)
    for col in ['#YY', 'MM', 'DD', 'hh', 'mm', 'YY', '#yr', 'yr', 'mo', 'dy', 'hr', 'mn']:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    if '#YY' in df.columns: time_cols, year_col = ['#YY', 'MM', 'DD', 'hh', 'mm'], '#YY'
    elif 'YY' in df.columns: time_cols, year_col = ['YY', 'MM', 'DD', 'hh', 'mm'], 'YY'
    elif '#yr' in df.columns: time_cols, year_col = ['#yr', 'mo', 'dy', 'hr', 'mn'], '#yr'
    elif 'yr' in df.columns: time_cols, year_col = ['yr', 'mo', 'dy', 'hr', 'mn'], 'yr'
    else: raise ValueError(f"Cannot identify time columns")
    year_values = df[year_col].values
    if year_values.max() < 100: year_values = np.where(year_values < 50, year_values + 2000, year_values + 1900)
    
    # TASK 1: Robust datetime parsing with error handling
    df['datetime'] = pd.to_datetime({
        'year': year_values, 
        'month': df[time_cols[1]], 
        'day': df[time_cols[2]], 
        'hour': df[time_cols[3]], 
        'minute': df[time_cols[4]]
    }, errors='coerce')
    
    # Drop rows with NaT timestamps
    initial_len = len(df)
    df = df.dropna(subset=['datetime'])
    dropped_len = initial_len - len(df)
    if dropped_len > 0:
        print(f"WARNING: Dropped {dropped_len} rows with invalid timestamps (NaT)")
    
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)
    
    # Assert no NaT in index
    assert not df.index.isna().any(), "Index contains NaT values after cleaning!"
    
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    column_mapping = {'BAR': 'PRES'}; 
    for old_name, new_name in column_mapping.items(): 
        if old_name in df.columns and new_name not in df.columns: df.rename(columns={old_name: new_name}, inplace=True)
    available_features = [f for f in FEATURE_NAMES if f in df.columns]
    if 'WVHT' not in available_features: raise ValueError("WVHT column not found!")
    print(f"Selected features ({len(available_features)}): {available_features}")
    
    df = df[available_features]
    for col in available_features: df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.replace({99: np.nan, 99.0: np.nan, 999: np.nan, 9999: np.nan})
    
    # Check for DEWP (after converting to numeric and cleaning)
    if 'DEWP' not in available_features:
        print("WARNING: DEWP not found in raw data!")
    else:
        dewp_data = df['DEWP'].values
        # Ensure numeric type (already numpy array from .values)
        dewp_data = np.array(dewp_data, dtype=float)
        dewp_valid = ~np.isnan(dewp_data)
        if dewp_valid.any():
            print(f"DEWP: {dewp_valid.sum()}/{len(dewp_data)} valid values, "
                  f"std={np.nanstd(dewp_data):.4f}, mean={np.nanmean(dewp_data):.4f}")
        else:
            print(f"DEWP: {dewp_valid.sum()}/{len(dewp_data)} valid values (ALL MISSING!)")
    for col in available_features:
        if col in ['WVHT', 'DPD', 'APD', 'WSPD', 'ATMP', 'DEWP']: 
            df.loc[df[col] >= 99, col] = np.nan
        elif col in ['WDIR', 'MWD']: 
            df.loc[df[col] > 360, col] = np.nan
        elif col == 'PRES': 
            df.loc[(df[col] > 1100) | (df[col] < 900), col] = np.nan
    
    data = df.values
    # Ensure data is numeric and handle NaN properly
    data = np.array(data, dtype=float)
    mask = (~np.isnan(data)).astype(float)
    timestamps = df.index.values
    train_end_idx = datetime_to_index(timestamps, TRAIN_END_DATE)
    train_data = data[:train_end_idx]
    
    # Handle circular encoding
    if use_circular_encoding:
        # Build expanded feature list: replace circular vars with sin/cos
        expanded_feature_names = []
        circular_var_map = {}  # Maps original index -> (sin_idx, cos_idx)
        feature_to_expanded = {}  # Maps original feature name -> expanded indices
        
        for i, feat_name in enumerate(FEATURE_NAMES):
            if feat_name in available_features:
                orig_idx = available_features.index(feat_name)
                if feat_name in CIRCULAR_VARS:
                    sin_idx = len(expanded_feature_names)
                    cos_idx = sin_idx + 1
                    expanded_feature_names.extend([f"{feat_name}_sin", f"{feat_name}_cos"])
                    circular_var_map[orig_idx] = (sin_idx, cos_idx)
                    feature_to_expanded[feat_name] = (sin_idx, cos_idx)
                else:
                    expanded_feature_names.append(feat_name)
                    feature_to_expanded[feat_name] = (len(expanded_feature_names) - 1,)
        
        # Encode circular variables
        expanded_data = []
        expanded_mask = []
        
        for i, feat_name in enumerate(FEATURE_NAMES):
            if feat_name in available_features:
                orig_idx = available_features.index(feat_name)
                col_data = data[:, orig_idx]
                col_mask = mask[:, orig_idx]
                
                if feat_name in CIRCULAR_VARS:
                    # Encode as sin/cos
                    # Ensure col_data is numeric
                    col_data = np.array(col_data, dtype=float)
                    valid_mask = ~np.isnan(col_data)
                    sin_vals = np.full(len(col_data), 0.0)
                    cos_vals = np.full(len(col_data), 0.0)
                    
                    if valid_mask.any():
                        sin_vals[valid_mask], cos_vals[valid_mask] = encode_circular(col_data[valid_mask])
                    
                    expanded_data.append(sin_vals)
                    expanded_data.append(cos_vals)
                    expanded_mask.append(col_mask)  # Same mask for both
                    expanded_mask.append(col_mask)
                else:
                    expanded_data.append(col_data)
                    expanded_mask.append(col_mask)
        
        expanded_data = np.column_stack(expanded_data)
        expanded_mask = np.column_stack(expanded_mask)
        
        # Compute stats on expanded features
        train_expanded = expanded_data[:train_end_idx]
        train_mean = np.nanmean(train_expanded, axis=0)
        train_std = np.nanstd(train_expanded, axis=0)
        
        # FIX: Handle NaNs in Mean/Std safely
        # If a column is entirely empty, mean/std will be NaN. Fill them with defaults.
        train_mean = np.nan_to_num(train_mean, nan=0.0)
        train_std = np.nan_to_num(train_std, nan=1.0)
        
        # Also prevent division by zero for constant columns
        train_std = np.where(train_std < 1e-6, 1.0, train_std)
        
        # Normalize
        # Replace NaNs in data with 0.0 BEFORE normalization
        expanded_data = np.nan_to_num(expanded_data, nan=0.0)
        
        normalized_data = (expanded_data - train_mean) / train_std
        
        # Apply mask
        normalized_data = normalized_data * expanded_mask
        
        # FIX: Final Safety Clean
        # Ensure no NaNs exist in the final normalized data (e.g., from NaN * 0)
        normalized_data = np.nan_to_num(normalized_data, nan=0.0)
        
        # Save with circular encoding metadata
        processed_file = output_path / f"{dataset_type}_{station}_processed.pk"
        with open(processed_file, 'wb') as f:
            pickle.dump({
                'observed_data': normalized_data,
                'observed_mask': expanded_mask,
                'train_mean': train_mean,
                'train_std': train_std,
                'feature_names': available_features,  # Original names
                'expanded_feature_names': expanded_feature_names,  # Expanded names
                'circular_var_map': circular_var_map,
                'feature_to_expanded': feature_to_expanded,
                'use_circular_encoding': True,
                'timestamps': timestamps
            }, f)
        
        print(f"Processed data with circular encoding saved to {processed_file}")
        print(f"Expanded features: {len(expanded_feature_names)} (from {len(available_features)})")
        print(f"Circular variables encoded: {[f for f in CIRCULAR_VARS if f in available_features]}")
        
        return normalized_data, expanded_mask, train_mean, train_std
        
    else:
        # Original non-circular encoding
        train_mean = np.nanmean(train_data, axis=0)
        train_std = np.nanstd(train_data, axis=0)
        train_std = np.where(train_std < 1e-6, 1.0, train_std)
        
        data = np.nan_to_num(data, nan=0.0)
        normalized_data = (data - train_mean) / train_std
        normalized_data = normalized_data * mask
        
        processed_file = output_path / f"{dataset_type}_{station}_processed.pk"
        with open(processed_file, 'wb') as f:
            pickle.dump({
                'observed_data': normalized_data,
                'observed_mask': mask,
                'train_mean': train_mean,
                'train_std': train_std,
                'feature_names': available_features,
                'use_circular_encoding': False,
                'timestamps': timestamps
            }, f)
        print(f"Processed data saved to {processed_file}")
        return normalized_data, mask, train_mean, train_std

if __name__ == "__main__":
    print("Testing Wave_Dataset_Forecasting...")
    try:
        dataset = Wave_Dataset_Forecasting(dataset_type="NDBC1", mode="train", station="42001", data_path="./data/wave", forecast_horizon_hours=6)
        print(f"Dataset length: {len(dataset)}")
    except FileNotFoundError as e: print(f"Expected error: {e}")
    print("\nDataset module loaded successfully!")