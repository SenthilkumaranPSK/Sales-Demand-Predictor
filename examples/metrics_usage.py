"""
Example usage of the performance metrics computation module.

This example demonstrates how to use the metrics module to evaluate
forecasting model performance.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from datetime import datetime
from src.training.metrics import (
    compute_mae,
    compute_rmse,
    compute_mape,
    PerformanceMetrics
)


def main():
    """Demonstrate metrics computation with example forecast data."""
    
    # Example: Weekly demand forecasts vs actual sales
    print("=" * 60)
    print("Demand Forecasting Performance Metrics Example")
    print("=" * 60)
    
    # Simulated predictions and actual values
    predictions = np.array([1200, 1350, 1100, 1450, 1300, 1250, 1400])
    actuals = np.array([1180, 1400, 1150, 1420, 1280, 1300, 1380])
    
    print("\nWeek | Predicted | Actual | Error")
    print("-" * 40)
    for i, (pred, actual) in enumerate(zip(predictions, actuals), 1):
        error = pred - actual
        print(f"  {i}  |   {pred:4d}    | {actual:4d}  | {error:+4d}")
    
    # Compute individual metrics
    print("\n" + "=" * 60)
    print("Performance Metrics")
    print("=" * 60)
    
    mae = compute_mae(predictions, actuals)
    print(f"\nMAE (Mean Absolute Error):           {mae:.2f} units")
    print("  → Average magnitude of forecast errors")
    
    rmse = compute_rmse(predictions, actuals)
    print(f"\nRMSE (Root Mean Squared Error):      {rmse:.2f} units")
    print("  → Penalizes large errors more heavily than MAE")
    
    mape = compute_mape(predictions, actuals)
    print(f"\nMAPE (Mean Absolute Percentage Error): {mape:.2f}%")
    print("  → Scale-independent accuracy measure")
    
    # Create comprehensive metrics object
    print("\n" + "=" * 60)
    print("Creating PerformanceMetrics Object")
    print("=" * 60)
    
    metrics = PerformanceMetrics(
        mae=mae,
        rmse=rmse,
        mape=mape,
        sample_size=len(predictions),
        evaluation_period=(datetime(2024, 1, 1), datetime(2024, 2, 28))
    )
    
    print(f"\nMetrics Object:")
    print(f"  MAE:              {metrics.mae:.2f}")
    print(f"  RMSE:             {metrics.rmse:.2f}")
    print(f"  MAPE:             {metrics.mape:.2f}%")
    print(f"  Sample Size:      {metrics.sample_size}")
    print(f"  Evaluation Period: {metrics.evaluation_period[0].date()} to {metrics.evaluation_period[1].date()}")
    
    # Compare two models
    print("\n" + "=" * 60)
    print("Model Comparison Example")
    print("=" * 60)
    
    # Model A (Custom model)
    predictions_a = np.array([1200, 1350, 1100, 1450, 1300, 1250, 1400])
    mae_a = compute_mae(predictions_a, actuals)
    rmse_a = compute_rmse(predictions_a, actuals)
    mape_a = compute_mape(predictions_a, actuals)
    
    # Model B (Benchmark model)
    predictions_b = np.array([1150, 1380, 1120, 1470, 1320, 1280, 1420])
    mae_b = compute_mae(predictions_b, actuals)
    rmse_b = compute_rmse(predictions_b, actuals)
    mape_b = compute_mape(predictions_b, actuals)
    
    print("\nModel A (Custom):")
    print(f"  MAE:  {mae_a:.2f} | RMSE: {rmse_a:.2f} | MAPE: {mape_a:.2f}%")
    
    print("\nModel B (Benchmark):")
    print(f"  MAE:  {mae_b:.2f} | RMSE: {rmse_b:.2f} | MAPE: {mape_b:.2f}%")
    
    # Calculate improvements
    mae_improvement = ((mae_b - mae_a) / mae_b) * 100
    rmse_improvement = ((rmse_b - rmse_a) / rmse_b) * 100
    mape_improvement = ((mape_b - mape_a) / mape_b) * 100
    
    print("\nModel A vs Model B:")
    print(f"  MAE Improvement:  {mae_improvement:+.2f}%")
    print(f"  RMSE Improvement: {rmse_improvement:+.2f}%")
    print(f"  MAPE Improvement: {mape_improvement:+.2f}%")
    
    if mae_improvement > 0:
        print("\n✓ Model A (Custom) performs better than Model B (Benchmark)")
    else:
        print("\n✗ Model B (Benchmark) performs better than Model A (Custom)")
    
    # Edge case handling
    print("\n" + "=" * 60)
    print("Edge Case Handling")
    print("=" * 60)
    
    # MAPE with zero actuals
    print("\nHandling zero actual values in MAPE:")
    predictions_with_zeros = np.array([100, 200, 300])
    actuals_with_zeros = np.array([110, 0, 310])
    
    try:
        mape_with_zeros = compute_mape(predictions_with_zeros, actuals_with_zeros)
        print(f"  MAPE (excluding zeros): {mape_with_zeros:.2f}%")
        print("  ✓ Zero actuals were automatically excluded")
    except ValueError as e:
        print(f"  Error: {e}")
    
    # All zeros
    print("\nHandling all zero actual values:")
    predictions_all_zeros = np.array([100, 200, 300])
    actuals_all_zeros = np.array([0, 0, 0])
    
    try:
        mape_all_zeros = compute_mape(predictions_all_zeros, actuals_all_zeros)
        print(f"  MAPE: {mape_all_zeros:.2f}%")
    except ValueError as e:
        print(f"  ✓ Error caught: {e}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
