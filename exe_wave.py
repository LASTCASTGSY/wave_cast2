"""
CSDI v2 Execution Script for Wave Height Forecasting
=====================================================

Summary of Changes vs. V1:
--------------------------
1. MULTI-HORIZON FORECASTING:
   - Support for 6h, 24h, 48h, and 72h forecast horizons
   - Single command to run all horizons sequentially

2. ERA5 INTEGRATION:
   - Optional ERA5 spatial covariate loading
   - Configurable spatial window and context length

3. GOLDEN RUN ARTIFACT GENERATION:
   - Automatic config.yaml export
   - Best model checkpoint based on validation CRPS
   - High-resolution visualization generation

4. DETERMINISTIC TEMPORAL SPLITS:
   - Train: ≤2022, Val: 2023, Test: 2024+
   - Prevents data leakage

Usage Examples:
---------------
# Run imputation
python exe_wave.py --mode imputation --station 42001

# Run single horizon forecasting
python exe_wave.py --mode forecasting --forecast_horizon 6 --station 42001

# Run multi-horizon forecasting (6h, 24h, 48h, 72h)
python exe_wave.py --mode forecasting_multi --station 42001

# Run with ERA5 covariates
python exe_wave.py --mode forecasting --use_era5 --era5_path ./data/era5

# Golden Run (full pipeline)
python exe_wave.py --mode golden_run --station 42001

Author: CSDI Wave Forecasting Project
Date: 2024
"""

import argparse
import torch
import datetime
import json
import yaml
import os
import sys
from pathlib import Path

from main_model_wave import CSDI_Wave, CSDI_Wave_Forecasting, get_model
from dataset_wave import get_dataloader, FORECAST_HORIZONS, CONTEXT_HOURS, CONTEXT_STEPS
from utils import train, evaluate, export_golden_run_artifacts


# ============================================================================
# ARGUMENT PARSER
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="CSDI v2 for Wave Height Forecasting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Basic configuration
    parser.add_argument("--config", type=str, default="config/wave_base.yaml",
                        help="Path to config file")
    parser.add_argument('--device', default='cuda:0', help='Device for training')
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Model/data configuration
    parser.add_argument("--modelfolder", type=str, default="",
                        help="Folder with pre-trained model (skip training)")
    parser.add_argument("--nsample", type=int, default=100,
                        help="Number of samples for probabilistic evaluation")
    parser.add_argument("--dataset_type", type=str, default="NDBC1",
                        choices=["NDBC1", "NDBC2"],
                        help="Dataset type (NDBC1=10-min, NDBC2=hourly)")
    parser.add_argument("--station", type=str, default="42001",
                        help="NDBC buoy station ID")
    parser.add_argument("--data_path", type=str, default="./data/wave",
                        help="Path to wave data directory")
    
    # Task configuration
    parser.add_argument("--mode", type=str, default="imputation",
                        choices=["imputation", "forecasting", "forecasting_multi", 
                                "golden_run", "evaluate_only"],
                        help="Task mode")
    parser.add_argument("--targetstrategy", type=str, default="mix",
                        choices=["mix", "random", "historical"],
                        help="Target masking strategy for imputation")
    parser.add_argument("--unconditional", action="store_true",
                        help="Use unconditional model")
    
    # Forecasting configuration
    parser.add_argument("--forecast_horizon", type=int, default=6,
                        choices=[6, 24, 48, 72],
                        help="Forecast horizon in hours")
    parser.add_argument("--context_hours", type=int, default=72,
                        help="Context window length in hours")
    
    # ERA5 configuration
    parser.add_argument("--use_era5", action="store_true",
                        help="Use ERA5 spatial covariates")
    parser.add_argument("--era5_path", type=str, default="./data/era5",
                        help="Path to ERA5 data directory")
    parser.add_argument("--era5_context_hours", type=int, default=24,
                        help="ERA5 context window in hours")
    parser.add_argument("--spatial_window_deg", type=float, default=3.0,
                        choices=[3.0, 5.0],
                        help="ERA5 spatial window size in degrees")
    
    # Training configuration
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override config epochs")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override config batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override config learning rate")
    parser.add_argument("--early_stopping", action="store_true",
                        help="Enable early stopping")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="./save",
                        help="Output directory for results")
    parser.add_argument("--run_name", type=str, default="",
                        help="Custom run name (default: auto-generated)")
    parser.add_argument("--no_vis", action="store_true",
                        help="Disable visualization generation")
    
    return parser.parse_args()


# ============================================================================
# CONFIGURATION LOADING
# ============================================================================

def load_config(args):
    """Load and merge configuration."""
    # Default config
    default_config = {
        "train": {
            "epochs": 200,
            "batch_size": 16,
            "lr": 1e-3,
            "itr_per_epoch": 100
        },
        "diffusion": {
            "layers": 4,
            "channels": 64,
            "nheads": 8,
            "diffusion_embedding_dim": 128,
            "beta_start": 0.0001,
            "beta_end": 0.5,
            "num_steps": 50,
            "schedule": "quad",
            "is_linear": False
        },
        "model": {
            "is_unconditional": False,
            "timeemb": 128,
            "featureemb": 16,
            "target_strategy": "mix",
            "cfg_dropout_prob": 0.0,
            "cfg_scale": 1.0
        }
    }
    
    # Try to load from file
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            file_config = yaml.safe_load(f)
            # Deep merge
            for key in file_config:
                if key in default_config and isinstance(default_config[key], dict):
                    default_config[key].update(file_config[key])
                else:
                    default_config[key] = file_config[key]
    
    config = default_config
    
    # Override from command line
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["train"]["lr"] = args.lr
    
    config["model"]["is_unconditional"] = args.unconditional
    config["model"]["target_strategy"] = args.targetstrategy
    
    return config


def get_era5_config(args):
    """Create ERA5 encoder configuration."""
    if not args.use_era5:
        return None
    
    return {
        "in_channels": 3,  # u10, v10, msl
        "hidden_channels": 64,
        "out_dim": 128,
        "num_blocks": 2
    }


# ============================================================================
# RUN FUNCTIONS
# ============================================================================

def run_imputation(args, config):
    """Run imputation task."""
    print("\n" + "="*60)
    print("RUNNING IMPUTATION TASK")
    print("="*60 + "\n")
    
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"imputation_{args.dataset_type}_{args.station}_{current_time}"
    foldername = os.path.join(args.output_dir, run_name)
    
    print(f"Output folder: {foldername}")
    os.makedirs(foldername, exist_ok=True)
    
    # Save config
    with open(os.path.join(foldername, "config.yaml"), "w") as f:
        yaml.dump(config, f)
    
    # Get dataloaders
    train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
        datatype=args.dataset_type,
        device=args.device,
        batch_size=config["train"]["batch_size"],
        station=args.station,
        data_path=args.data_path,
        task="imputation"
    )
    
    # Get circular encoding info from dataset
    train_dataset = train_loader.dataset
    use_circular = getattr(train_dataset, 'use_circular_encoding', False)
    feature_to_expanded = getattr(train_dataset, 'feature_to_expanded', None)
    
    # Determine target_dim based on circular encoding
    if use_circular:
        target_dim = train_dataset.target_dim
    else:
        target_dim = 9
    
    # Initialize model
    era5_config = get_era5_config(args)
    model = get_model(
        config, args.device, target_dim=target_dim,
        task="imputation",
        use_era5=args.use_era5,
        era5_config=era5_config
    ).to(args.device)
    
    # Verify model is on correct device
    model_device = next(model.parameters()).device
    print(f"\nModel device: {model_device}")
    
    # Train or load model
    if args.modelfolder == "":
        train(
            model,
            config["train"],
            train_loader,
            valid_loader=valid_loader,
            foldername=foldername,
            use_early_stopping=args.early_stopping,
            patience=args.patience
        )
    else:
        model.load_state_dict(
            torch.load(os.path.join(args.modelfolder, "model.pth"))
        )
    
    # Evaluate
    print("\nEvaluating imputation model...")
    metrics = evaluate(
        model,
        test_loader,
        nsample=args.nsample,
        scaler=scaler,
        mean_scaler=mean_scaler,
        foldername=foldername,
        generate_visualizations=not args.no_vis,
        use_circular_encoding=use_circular,
        feature_to_expanded=feature_to_expanded,
        debug=True,  # Enable debug prints
        dataset_type=args.dataset_type
    )
    
    return foldername, metrics


def run_forecasting(args, config, horizon_hours=None):
    """Run forecasting task for a single horizon."""
    if horizon_hours is None:
        horizon_hours = args.forecast_horizon
    
    print("\n" + "="*60)
    print(f"RUNNING FORECASTING TASK - {horizon_hours}h HORIZON")
    print("="*60 + "\n")
    
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    era5_tag = "_era5" if args.use_era5 else ""
    run_name = args.run_name or f"forecast_{args.dataset_type}_{args.station}_{horizon_hours}h{era5_tag}_{current_time}"
    foldername = os.path.join(args.output_dir, run_name)
    
    print(f"Output folder: {foldername}")
    os.makedirs(foldername, exist_ok=True)
    
    # Save config with horizon info
    config_with_horizon = config.copy()
    config_with_horizon["forecasting"] = {
        "horizon_hours": horizon_hours,
        "context_hours": args.context_hours,
        "use_era5": args.use_era5,
        "era5_context_hours": args.era5_context_hours if args.use_era5 else None,
        "spatial_window_deg": args.spatial_window_deg if args.use_era5 else None
    }
    
    with open(os.path.join(foldername, "config.yaml"), "w") as f:
        yaml.dump(config_with_horizon, f)
    
    # Get dataloaders
    train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
        datatype=args.dataset_type,
        device=args.device,
        batch_size=config["train"]["batch_size"],
        station=args.station,
        data_path=args.data_path,
        task="forecasting",
        forecast_horizon_hours=horizon_hours,
        use_era5=args.use_era5,
        era5_path=args.era5_path,
        era5_context_hours=args.era5_context_hours,
        spatial_window_deg=args.spatial_window_deg
    )
    
    # Get circular encoding info from dataset
    train_dataset = train_loader.dataset
    use_circular = getattr(train_dataset, 'use_circular_encoding', False)
    feature_to_expanded = getattr(train_dataset, 'feature_to_expanded', None)
    
    # Determine target_dim based on circular encoding
    if use_circular:
        target_dim = train_dataset.target_dim
    else:
        target_dim = 9
    
    # Initialize model
    era5_config = get_era5_config(args)
    model = get_model(
        config, args.device, target_dim=target_dim,
        task="forecasting",
        use_era5=args.use_era5,
        era5_config=era5_config
    ).to(args.device)
    
    # Verify model is on correct device
    model_device = next(model.parameters()).device
    print(f"\nModel device: {model_device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train or load model
    if args.modelfolder == "":
        train(
            model,
            config["train"],
            train_loader,
            valid_loader=valid_loader,
            foldername=foldername,
            use_early_stopping=args.early_stopping,
            patience=args.patience
        )
    else:
        model.load_state_dict(
            torch.load(os.path.join(args.modelfolder, "model.pth"))
        )
    
    # Evaluate
    print(f"\nEvaluating forecasting model for {horizon_hours}h horizon...")
    print(f"[DEBUG] Scaler shape: {scaler.shape}, Mean scaler shape: {mean_scaler.shape}")
    print(f"[DEBUG] Dataset target_dim: {train_dataset.target_dim}, use_circular: {use_circular}")
    metrics = evaluate(
        model,
        test_loader,
        nsample=args.nsample,
        scaler=scaler,
        mean_scaler=mean_scaler,
        foldername=foldername,
        generate_visualizations=not args.no_vis,
        use_circular_encoding=use_circular,
        feature_to_expanded=feature_to_expanded,
        debug=True,  # Enable debug prints
        dataset_type=args.dataset_type
    )
    
    return foldername, metrics


def run_multi_horizon_forecasting(args, config):
    """Run forecasting for all standard horizons (6h, 24h, 48h, 72h)."""
    print("\n" + "="*60)
    print("RUNNING MULTI-HORIZON FORECASTING")
    print(f"Horizons: {list(FORECAST_HORIZONS.keys())} hours")
    print("="*60 + "\n")
    
    all_results = {}
    
    for horizon_hours in [6, 24, 48, 72]:
        try:
            foldername, metrics = run_forecasting(args, config, horizon_hours)
            all_results[f"{horizon_hours}h"] = {
                "folder": foldername,
                "metrics": metrics
            }
        except Exception as e:
            print(f"\nError in {horizon_hours}h forecasting: {e}")
            all_results[f"{horizon_hours}h"] = {"error": str(e)}
    
    # Save summary
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_folder = os.path.join(
        args.output_dir, 
        f"multi_horizon_{args.station}_{current_time}"
    )
    os.makedirs(summary_folder, exist_ok=True)
    
    # Create summary table
    summary = {
        "station": args.station,
        "dataset_type": args.dataset_type,
        "use_era5": args.use_era5,
        "results": {}
    }
    
    print("\n" + "="*60)
    print("MULTI-HORIZON RESULTS SUMMARY")
    print("="*60)
    print(f"\n{'Horizon':<10} {'RMSE':<10} {'MAE':<10} {'CRPS':<10} {'Coverage90':<12}")
    print("-" * 52)
    
    for horizon, result in all_results.items():
        if "error" in result:
            print(f"{horizon:<10} ERROR: {result['error']}")
            summary["results"][horizon] = {"error": result["error"]}
        else:
            m = result["metrics"]["overall"]
            print(f"{horizon:<10} {m['rmse']:<10.4f} {m['mae']:<10.4f} {m['crps']:<10.4f} {m['coverage_90']:<12.4f}")
            summary["results"][horizon] = {
                "folder": result["folder"],
                "overall": m
            }
    
    with open(os.path.join(summary_folder, "multi_horizon_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_folder}")
    
    return summary_folder, all_results


def run_golden_run(args, config):
    """
    Run complete Golden Run pipeline.
    
    This includes:
    1. Training with best model checkpoint
    2. Multi-horizon evaluation
    3. Complete artifact generation
    """
    print("\n" + "="*60)
    print("GOLDEN RUN - Complete Wave Forecasting Pipeline")
    print(f"Station: {args.station}")
    print(f"ERA5: {'Enabled' if args.use_era5 else 'Disabled'}")
    print("="*60 + "\n")
    
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    golden_folder = os.path.join(
        args.output_dir,
        f"golden_run_{args.station}_{current_time}"
    )
    os.makedirs(golden_folder, exist_ok=True)
    
    # Run multi-horizon forecasting
    _, all_results = run_multi_horizon_forecasting(args, config)
    
    # Export Golden Run artifacts
    export_golden_run_artifacts(
        config=config,
        model_path=golden_folder,
        metrics=all_results,
        output_folder=golden_folder,
        additional_info={
            "station": args.station,
            "dataset_type": args.dataset_type,
            "use_era5": args.use_era5,
            "horizons": [6, 24, 48, 72]
        }
    )
    
    print("\n" + "="*60)
    print("GOLDEN RUN COMPLETE")
    print(f"All artifacts saved to: {golden_folder}")
    print("="*60)
    
    return golden_folder


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load configuration
    config = load_config(args)
    
    print("\n" + "="*60)
    print("CSDI v2 - Wave Height Forecasting")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Mode: {args.mode}")
    print(f"  Station: {args.station}")
    print(f"  Dataset: {args.dataset_type}")
    print(f"  Device: {args.device}")
    print(f"  ERA5: {'Enabled' if args.use_era5 else 'Disabled'}")
    if args.mode == "forecasting":
        print(f"  Horizon: {args.forecast_horizon}h")
    print(f"\nModel Config:")
    print(json.dumps(config, indent=2))
    
    # Check device
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    # Run selected mode
    if args.mode == "imputation":
        run_imputation(args, config)
    
    elif args.mode == "forecasting":
        run_forecasting(args, config)
    
    elif args.mode == "forecasting_multi":
        run_multi_horizon_forecasting(args, config)
    
    elif args.mode == "golden_run":
        run_golden_run(args, config)
    
    elif args.mode == "evaluate_only":
        if args.modelfolder == "":
            print("ERROR: --modelfolder required for evaluate_only mode")
            sys.exit(1)
        # Load existing model and evaluate
        # (Implementation depends on saved model type)
        print("Evaluate-only mode: Loading from", args.modelfolder)
    
    print("\n" + "="*60)
    print("EXECUTION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
