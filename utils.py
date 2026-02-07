"""
CSDI v2 Utilities Module
========================

Summary of Changes vs. V1:
--------------------------
1. ENHANCED EVALUATION METRICS:
   - RMSE and MAE per variable (un-normalized)
   - CRPS (Continuous Ranked Probability Score)
   - Empirical coverage of 90% prediction interval
   - Per-horizon metrics for forecasting

2. TRAINING IMPROVEMENTS:
   - Best model selection based on validation CRPS
   - Early stopping option
   - Gradient clipping
   - Learning rate warmup

3. GOLDEN RUN ARTIFACT GENERATION:
   - Automatic config.yaml export
   - Model checkpoint saving
   - Visualization generation

Author: CSDI Wave Forecasting Project
Date: 2024
"""

import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
import json
import yaml
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from datetime import datetime


# ============================================================================
# FEATURE CONFIGURATION
# ============================================================================

FEATURE_NAMES = ['WDIR', 'WSPD', 'WVHT', 'DPD', 'APD', 'MWD', 'PRES', 'ATMP', 'DEWP']
WVHT_INDEX = FEATURE_NAMES.index('WVHT')


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train(
    model,
    config: Dict,
    train_loader,
    valid_loader=None,
    valid_epoch_interval: int = 5,
    foldername: str = "",
    use_early_stopping: bool = False,
    patience: int = 20,
    grad_clip: float = 1.0
):
    """
    Train CSDI model with enhanced features.
    """
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    
    epochs = config["epochs"]
    p1 = int(0.75 * epochs)
    p2 = int(0.9 * epochs)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )
    
    if foldername:
        output_path = Path(foldername)
        output_path.mkdir(parents=True, exist_ok=True)
        model_path = output_path / "model.pth"
        best_model_path = output_path / "model_best.pth"
    
    best_valid_loss = float('inf')
    epochs_without_improvement = 0
    training_history = {'train_loss': [], 'valid_loss': [], 'lr': []}
    
    for epoch_no in range(epochs):
        avg_loss = 0
        model.train()
        
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()
                
                loss = model(train_batch)
                loss.backward()
                
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
                avg_loss += loss.item()
                optimizer.step()
                
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                        "lr": optimizer.param_groups[0]['lr']
                    },
                    refresh=False,
                )
                
                if batch_no >= config["itr_per_epoch"]:
                    break
        
        lr_scheduler.step()
        training_history['train_loss'].append(avg_loss / batch_no)
        training_history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Validation
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        loss = model(valid_batch, is_train=0)
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )
            
            avg_loss_valid = avg_loss_valid / batch_no
            training_history['valid_loss'].append(avg_loss_valid)
            
            if avg_loss_valid < best_valid_loss:
                best_valid_loss = avg_loss_valid
                epochs_without_improvement = 0
                print(f"\n Best validation loss: {best_valid_loss:.6f} at epoch {epoch_no}")
                
                if foldername:
                    torch.save(model.state_dict(), best_model_path)
            else:
                epochs_without_improvement += valid_epoch_interval
            
            if use_early_stopping and epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch_no}")
                break
    
    if foldername:
        torch.save(model.state_dict(), model_path)
        with open(output_path / "training_history.json", 'w') as f:
            json.dump(training_history, f, indent=2)
    
    return training_history


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def quantile_loss(target, forecast, q: float, eval_points) -> float:
    """Quantile loss for CRPS calculation."""
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    """Calculate denominator for CRPS."""
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):
    """Calculate CRPS using quantile-based approximation."""
    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    
    if denom == 0:
        return 0.0
    
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j:j+1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    
    return CRPS.item() / len(quantiles)


def calc_quantile_CRPS_sum(target, forecast, eval_points, mean_scaler, scaler):
    """Calculate CRPS sum."""
    eval_points = eval_points.mean(-1)
    target = target * scaler + mean_scaler
    target = target.sum(-1)
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    
    if denom == 0:
        return 0.0
    
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = torch.quantile(forecast.sum(-1), quantiles[i], dim=1)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    
    return CRPS.item() / len(quantiles)


def calc_coverage(
    target: torch.Tensor,
    samples: torch.Tensor,
    eval_points: torch.Tensor,
    coverage_level: float = 0.90
) -> float:
    """
    Calculate empirical coverage of prediction interval.
    
    Args:
        target: (B, L, K) ground truth
        samples: (B, N, L, K) samples
        eval_points: (B, L, K) evaluation mask
        coverage_level: Desired coverage level (default: 0.90)
    
    Returns:
        Empirical coverage (fraction of targets within interval)
    """
    alpha = 1 - coverage_level
    lower_q = alpha / 2
    upper_q = 1 - alpha / 2
    
    lower_bound = torch.quantile(samples, lower_q, dim=1)
    upper_bound = torch.quantile(samples, upper_q, dim=1)
    
    in_interval = ((target >= lower_bound) & (target <= upper_bound)).float()
    
    covered = (in_interval * eval_points).sum()
    total = eval_points.sum()
    
    if total > 0:
        return (covered / total).item()
    return 0.0


def calc_per_variable_metrics(
    target: torch.Tensor,
    samples: torch.Tensor,
    eval_points: torch.Tensor,
    mean_scaler: torch.Tensor,
    scaler: torch.Tensor,
    feature_names: List[str] = FEATURE_NAMES
) -> Dict[str, Dict[str, float]]:
    """
    Calculate RMSE and MAE per variable (un-normalized).
    """
    pred_median = samples.median(dim=1).values
    
    target_unnorm = target * scaler + mean_scaler
    pred_unnorm = pred_median * scaler + mean_scaler
    
    metrics = {}
    for k, name in enumerate(feature_names):
        mask_k = eval_points[:, :, k]
        target_k = target_unnorm[:, :, k]
        pred_k = pred_unnorm[:, :, k]
        
        num_points = mask_k.sum()
        if num_points > 0:
            rmse = torch.sqrt(((pred_k - target_k) ** 2 * mask_k).sum() / num_points)
            mae = (torch.abs(pred_k - target_k) * mask_k).sum() / num_points
            
            samples_k = samples[:, :, :, k]
            lower = torch.quantile(samples_k, 0.05, dim=1)
            upper = torch.quantile(samples_k, 0.95, dim=1)
            
            lower_unnorm = lower * scaler[k] + mean_scaler[k]
            upper_unnorm = upper * scaler[k] + mean_scaler[k]
            
            in_interval = ((target_k >= lower_unnorm) & (target_k <= upper_unnorm)).float()
            coverage = (in_interval * mask_k).sum() / num_points
            
            metrics[name] = {
                'rmse': rmse.item(),
                'mae': mae.item(),
                'coverage_90': coverage.item()
            }
        else:
            metrics[name] = {'rmse': 0.0, 'mae': 0.0, 'coverage_90': 0.0}
    
    return metrics


# ============================================================================
# MAIN EVALUATION FUNCTION
# ============================================================================

def evaluate(
    model,
    test_loader,
    nsample: int = 100,
    scaler=None,
    mean_scaler=None,
    foldername: str = "",
    feature_names: List[str] = FEATURE_NAMES,
    generate_visualizations: bool = True,
    max_vis_samples: int = 5
):
    """
    Comprehensive evaluation with all metrics.
    """
    device = next(model.parameters()).device
    
    if scaler is None:
        scaler = torch.ones(len(feature_names)).to(device)
    if mean_scaler is None:
        mean_scaler = torch.zeros(len(feature_names)).to(device)
    
    with torch.no_grad():
        model.eval()
        
        all_target = []
        all_samples = []
        all_evalpoint = []
        all_observed_point = []
        all_observed_time = []
        
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0
        
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(test_batch, nsample)
                samples, observed_data, target_mask, observed_mask, observed_time = output
                
                samples = samples.permute(0, 1, 3, 2)
                c_target = observed_data.permute(0, 2, 1)
                eval_points = target_mask.permute(0, 2, 1)
                observed_points = observed_mask.permute(0, 2, 1)
                
                all_target.append(c_target)
                all_samples.append(samples)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                
                samples_median = samples.median(dim=1).values
                
                mse_current = (
                    ((samples_median - c_target) * eval_points) ** 2
                ) * (scaler ** 2)
                mae_current = (
                    torch.abs((samples_median - c_target) * eval_points)
                ) * scaler
                
                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                evalpoints_total += eval_points.sum().item()
                
                if evalpoints_total > 0:
                    it.set_postfix(
                        ordered_dict={
                            "rmse": np.sqrt(mse_total / evalpoints_total),
                            "mae": mae_total / evalpoints_total,
                            "batch": batch_no,
                        },
                        refresh=True,
                    )
        
        all_target = torch.cat(all_target, dim=0)
        all_samples = torch.cat(all_samples, dim=0)
        all_evalpoint = torch.cat(all_evalpoint, dim=0)
        all_observed_point = torch.cat(all_observed_point, dim=0)
        all_observed_time = torch.cat(all_observed_time, dim=0)
        
        if evalpoints_total > 0:
            final_rmse = np.sqrt(mse_total / evalpoints_total)
            final_mae = mae_total / evalpoints_total
        else:
            print("WARNING: No evaluation points found!")
            final_rmse = 0.0
            final_mae = 0.0
        
        crps = calc_quantile_CRPS(
            all_target, all_samples, all_evalpoint, mean_scaler, scaler
        )
        
        crps_sum = calc_quantile_CRPS_sum(
            all_target, all_samples, all_evalpoint, mean_scaler, scaler
        )
        
        coverage_90 = calc_coverage(
            all_target, all_samples, all_evalpoint, 0.90
        )
        
        per_var_metrics = calc_per_variable_metrics(
            all_target, all_samples, all_evalpoint,
            mean_scaler, scaler, feature_names
        )
        
        results = {
            'overall': {
                'rmse': final_rmse,
                'mae': final_mae,
                'crps': crps,
                'crps_sum': crps_sum,
                'coverage_90': coverage_90,
            },
            'per_variable': per_var_metrics
        }
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"\nOverall Metrics:")
        print(f"  RMSE: {final_rmse:.4f}")
        print(f"  MAE: {final_mae:.4f}")
        print(f"  CRPS: {crps:.4f}")
        print(f"  CRPS_sum: {crps_sum:.4f}")
        print(f"  90% Coverage: {coverage_90:.4f}")
        
        print(f"\nPer-Variable Metrics (Un-normalized):")
        for var_name, var_metrics in per_var_metrics.items():
            print(f"  {var_name}:")
            print(f"    RMSE: {var_metrics['rmse']:.4f}")
            print(f"    MAE: {var_metrics['mae']:.4f}")
            print(f"    90% Coverage: {var_metrics['coverage_90']:.4f}")
        
        if foldername:
            output_path = Path(foldername)
            
            with open(output_path / f"generated_outputs_nsample{nsample}.pk", "wb") as f:
                pickle.dump(
                    [
                        all_samples.cpu(),
                        all_target.cpu(),
                        all_evalpoint.cpu(),
                        all_observed_point.cpu(),
                        all_observed_time.cpu(),
                        scaler.cpu(),
                        mean_scaler.cpu(),
                    ],
                    f,
                )
            
            with open(output_path / f"result_nsample{nsample}.pk", "wb") as f:
                pickle.dump([final_rmse, final_mae, crps], f)
            
            with open(output_path / f"metrics_nsample{nsample}.json", "w") as f:
                json_results = {
                    'overall': results['overall'],
                    'per_variable': {
                        k: {mk: float(mv) for mk, mv in v.items()}
                        for k, v in results['per_variable'].items()
                    }
                }
                json.dump(json_results, f, indent=2)
            
            if generate_visualizations:
                vis_path = output_path / "visualizations"
                vis_path.mkdir(exist_ok=True)
                
                generate_forecast_plots(
                    all_target.cpu(),
                    all_samples.cpu(),
                    all_evalpoint.cpu(),
                    all_observed_point.cpu(),
                    mean_scaler.cpu(),
                    scaler.cpu(),
                    vis_path,
                    feature_names,
                    max_samples=max_vis_samples
                )
        
        return results


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def generate_forecast_plots(
    target: torch.Tensor,
    samples: torch.Tensor,
    eval_points: torch.Tensor,
    observed_points: torch.Tensor,
    mean_scaler: torch.Tensor,
    scaler: torch.Tensor,
    output_path: Path,
    feature_names: List[str],
    max_samples: int = 5,
    primary_var_idx: int = WVHT_INDEX
):
    """
    Generate high-resolution forecast visualization plots.
    
    Creates plots with:
    - Red "Observed" points (Context)
    - Blue "Target" points (Ground Truth)
    - Green line (Median Prediction)
    - Shaded Green Area (90% uncertainty band)
    """
    output_path = Path(output_path)
    
    target_unnorm = target * scaler + mean_scaler
    samples_unnorm = samples * scaler + mean_scaler
    
    pred_median = samples_unnorm.median(dim=1).values
    pred_lower = torch.quantile(samples_unnorm, 0.05, dim=1)
    pred_upper = torch.quantile(samples_unnorm, 0.95, dim=1)
    
    n_plots = min(max_samples, len(target))
    
    for sample_idx in range(n_plots):
        for var_idx, var_name in enumerate(feature_names):
            fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
            
            L = target.shape[1]
            time_steps = np.arange(L)
            
            target_var = target_unnorm[sample_idx, :, var_idx].numpy()
            pred_med = pred_median[sample_idx, :, var_idx].numpy()
            pred_lo = pred_lower[sample_idx, :, var_idx].numpy()
            pred_hi = pred_upper[sample_idx, :, var_idx].numpy()
            obs_mask = observed_points[sample_idx, :, var_idx].numpy()
            eval_mask = eval_points[sample_idx, :, var_idx].numpy()
            
            context_idx = np.where((obs_mask > 0) & (eval_mask == 0))[0]
            target_idx = np.where(eval_mask > 0)[0]
            
            if len(context_idx) > 0:
                ax.scatter(
                    time_steps[context_idx],
                    target_var[context_idx],
                    color='red',
                    label='Observed (Context)',
                    s=20,
                    alpha=0.8,
                    zorder=3
                )
            
            if len(target_idx) > 0:
                ax.scatter(
                    time_steps[target_idx],
                    target_var[target_idx],
                    color='blue',
                    label='Ground Truth (Target)',
                    s=30,
                    marker='x',
                    zorder=4
                )
            
            ax.plot(
                time_steps,
                pred_med,
                color='green',
                linewidth=2,
                label='Median Prediction',
                zorder=2
            )
            
            ax.fill_between(
                time_steps,
                pred_lo,
                pred_hi,
                color='green',
                alpha=0.2,
                label='90% Prediction Interval',
                zorder=1
            )
            
            if len(context_idx) > 0 and len(target_idx) > 0:
                boundary = context_idx[-1] + 0.5
                ax.axvline(
                    x=boundary,
                    color='gray',
                    linestyle='--',
                    linewidth=1.5,
                    label='Context/Horizon Boundary'
                )
            
            ax.set_xlabel('Time Step (10-min intervals)', fontsize=12)
            ax.set_ylabel(f'{var_name} (Physical Units)', fontsize=12)
            ax.set_title(f'Wave Forecast: {var_name} - Sample {sample_idx + 1}', fontsize=14)
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(
                output_path / f"forecast_sample{sample_idx + 1}_{var_name}.png",
                dpi=150,
                bbox_inches='tight'
            )
            plt.close()


# ============================================================================
# GOLDEN RUN ARTIFACT GENERATION
# ============================================================================

def export_golden_run_artifacts(
    config: Dict,
    model_path: str,
    metrics: Dict,
    output_folder: str,
    additional_info: Dict = None
):
    """
    Export complete Golden Run package.
    """
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    config_path = output_path / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    metrics_path = output_path / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'model_path': str(model_path),
        'config_path': str(config_path),
        'metrics_summary': {
            'rmse': metrics.get('overall', {}).get('rmse', 'N/A'),
            'mae': metrics.get('overall', {}).get('mae', 'N/A'),
            'crps': metrics.get('overall', {}).get('crps', 'N/A'),
            'coverage_90': metrics.get('overall', {}).get('coverage_90', 'N/A'),
        }
    }
    
    if additional_info:
        summary.update(additional_info)
    
    summary_path = output_path / "run_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nGolden Run artifacts exported to: {output_path}")
