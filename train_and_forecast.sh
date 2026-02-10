#!/bin/bash
# Full Training and Forecasting Pipeline for CSDI Wave Forecasting
# Usage: bash train_and_forecast.sh

# ============================================================================
# STEP 1: Preprocessing (if needed - uncomment if data needs reprocessing)
# ============================================================================
# python preprocess_ndbc_data.py \
#     --input_dir ./raw_data \
#     --output ./data/wave \
#     --dataset_type NDBC2 \
#     --station "42002" \
#     --use_circular_encoding

# ============================================================================
# STEP 2: Full Training (150 epochs recommended for good convergence)
# ============================================================================
echo "Starting full training..."
python exe_wave.py \
    --mode forecasting \
    --dataset_type NDBC2 \
    --station 42002 \
    --forecast_horizon 72 \
    --epochs 150 \
    --batch_size 16 \
    --data_path ./data/wave \
    --config config/wave_base.yaml \
    --device cuda:0 \
    --nsample 100

# ============================================================================
# STEP 3: Evaluation Only (if you want to evaluate a trained model)
# ============================================================================
# Uncomment and set --modelfolder to the trained model path
# python exe_wave.py \
#     --mode evaluate_only \
#     --dataset_type NDBC2 \
#     --station 42002 \
#     --forecast_horizon 72 \
#     --modelfolder ./save/forecast_NDBC2_42002_72h_YYYYMMDD_HHMMSS \
#     --data_path ./data/wave \
#     --config config/wave_base.yaml \
#     --device cuda:0 \
#     --nsample 100

echo "Training complete! Check ./save/ for results."
