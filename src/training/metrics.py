"""
Performance metrics computation module for demand forecasting models.

This module provides functions to calculate standard forecasting accuracy metrics:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)

And model comparison functionality:
- ComparisonReport for comparing custom vs benchmark models
- generate_comparison_report for creating comparison reports

Requirements: 2.2, 3.4, 8.2, 8.3
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, Optional
import numpy as np


@dataclass
class PerformanceMetrics:
    """
    Container for model performance metrics.
    
    Attributes:
        mae: Mean Absolute Error
        rmse: Root Mean Squared Error
        mape: Mean Absolute Percentage Error (as percentage, e.g., 5.0 for 5%)
        sample_size: Number of samples used for evaluation
        evaluation_period: Optional tuple of (start_date, end_date) for the evaluation
    """
    mae: float
    rmse: float
    mape: float
    sample_size: int
    evaluation_period: Optional[Tuple[datetime, datetime]] = None


@dataclass
class ComparisonReport:
    """
    Container for model comparison results between custom and benchmark models.
    
    Attributes:
        custom_metrics: Performance metrics for the custom model
        benchmark_metrics: Performance metrics for the benchmark model
        mae_improvement: Percentage improvement in MAE (positive means custom is better)
        rmse_improvement: Percentage improvement in RMSE (positive means custom is better)
        mape_improvement: Percentage improvement in MAPE (positive means custom is better)
        recommendation: Recommendation string ("Use custom model" or "Use benchmark")
    """
    custom_metrics: PerformanceMetrics
    benchmark_metrics: PerformanceMetrics
    mae_improvement: float
    rmse_improvement: float
    mape_improvement: float
    recommendation: str


def compute_mae(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """
    Compute Mean Absolute Error.
    
    MAE measures the average magnitude of errors in predictions, without considering
    their direction. It's the average absolute difference between predicted and actual values.
    
    Formula: MAE = (1/n) * Σ|predicted_i - actual_i|
    
    Args:
        predictions: Array of predicted values
        actuals: Array of actual values
        
    Returns:
        Mean Absolute Error as a float
        
    Raises:
        ValueError: If arrays are empty, have different lengths, or contain invalid values
        
    Examples:
        >>> compute_mae(np.array([100, 200, 300]), np.array([110, 190, 310]))
        10.0
    """
    # Validate inputs
    if len(predictions) == 0 or len(actuals) == 0:
        raise ValueError("Input arrays cannot be empty")
    
    if len(predictions) != len(actuals):
        raise ValueError(
            f"Predictions and actuals must have the same length. "
            f"Got predictions: {len(predictions)}, actuals: {len(actuals)}"
        )
    
    # Convert to numpy arrays if not already
    predictions = np.asarray(predictions, dtype=np.float64)
    actuals = np.asarray(actuals, dtype=np.float64)
    
    # Check for NaN or infinite values
    if np.any(np.isnan(predictions)) or np.any(np.isnan(actuals)):
        raise ValueError("Input arrays contain NaN values")
    
    if np.any(np.isinf(predictions)) or np.any(np.isinf(actuals)):
        raise ValueError("Input arrays contain infinite values")
    
    # Compute MAE
    mae = np.mean(np.abs(predictions - actuals))
    
    return float(mae)


def compute_rmse(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error.
    
    RMSE measures the square root of the average squared differences between predicted
    and actual values. It penalizes large errors more heavily than MAE.
    
    Formula: RMSE = sqrt((1/n) * Σ(predicted_i - actual_i)²)
    
    Args:
        predictions: Array of predicted values
        actuals: Array of actual values
        
    Returns:
        Root Mean Squared Error as a float
        
    Raises:
        ValueError: If arrays are empty, have different lengths, or contain invalid values
        
    Examples:
        >>> compute_rmse(np.array([100, 200, 300]), np.array([110, 190, 310]))
        10.0
    """
    # Validate inputs
    if len(predictions) == 0 or len(actuals) == 0:
        raise ValueError("Input arrays cannot be empty")
    
    if len(predictions) != len(actuals):
        raise ValueError(
            f"Predictions and actuals must have the same length. "
            f"Got predictions: {len(predictions)}, actuals: {len(actuals)}"
        )
    
    # Convert to numpy arrays if not already
    predictions = np.asarray(predictions, dtype=np.float64)
    actuals = np.asarray(actuals, dtype=np.float64)
    
    # Check for NaN or infinite values
    if np.any(np.isnan(predictions)) or np.any(np.isnan(actuals)):
        raise ValueError("Input arrays contain NaN values")
    
    if np.any(np.isinf(predictions)) or np.any(np.isinf(actuals)):
        raise ValueError("Input arrays contain infinite values")
    
    # Compute RMSE
    mse = np.mean(np.square(predictions - actuals))
    rmse = np.sqrt(mse)
    
    return float(rmse)


def compute_mape(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """
    Compute Mean Absolute Percentage Error.
    
    MAPE measures the average absolute percentage difference between predicted and
    actual values. It's scale-independent, making it useful for comparing forecasts
    across different products or scales.
    
    Formula: MAPE = (100/n) * Σ|(actual_i - predicted_i) / actual_i|
    
    Note: MAPE is undefined when actual values are zero. This function handles this
    by excluding zero actual values from the calculation and issuing a warning if
    any are found.
    
    Args:
        predictions: Array of predicted values
        actuals: Array of actual values
        
    Returns:
        Mean Absolute Percentage Error as a percentage (e.g., 5.0 for 5%)
        
    Raises:
        ValueError: If arrays are empty, have different lengths, contain invalid values,
                   or all actual values are zero
        
    Examples:
        >>> compute_mape(np.array([100, 200, 300]), np.array([110, 190, 310]))
        5.303030303030303
    """
    # Validate inputs
    if len(predictions) == 0 or len(actuals) == 0:
        raise ValueError("Input arrays cannot be empty")
    
    if len(predictions) != len(actuals):
        raise ValueError(
            f"Predictions and actuals must have the same length. "
            f"Got predictions: {len(predictions)}, actuals: {len(actuals)}"
        )
    
    # Convert to numpy arrays if not already
    predictions = np.asarray(predictions, dtype=np.float64)
    actuals = np.asarray(actuals, dtype=np.float64)
    
    # Check for NaN or infinite values
    if np.any(np.isnan(predictions)) or np.any(np.isnan(actuals)):
        raise ValueError("Input arrays contain NaN values")
    
    if np.any(np.isinf(predictions)) or np.any(np.isinf(actuals)):
        raise ValueError("Input arrays contain infinite values")
    
    # Handle division by zero: exclude zero actual values
    non_zero_mask = actuals != 0
    
    if not np.any(non_zero_mask):
        raise ValueError(
            "Cannot compute MAPE: all actual values are zero. "
            "MAPE is undefined when dividing by zero."
        )
    
    # Filter out zero actual values
    filtered_predictions = predictions[non_zero_mask]
    filtered_actuals = actuals[non_zero_mask]
    
    # Compute MAPE on non-zero actuals
    percentage_errors = np.abs((filtered_actuals - filtered_predictions) / filtered_actuals)
    mape = np.mean(percentage_errors) * 100
    
    return float(mape)


def generate_comparison_report(
    custom_metrics: PerformanceMetrics,
    benchmark_metrics: PerformanceMetrics
) -> ComparisonReport:
    """
    Generate a comparison report between custom and benchmark model performance.
    
    Calculates percentage improvements for each metric using the formula:
    improvement = ((benchmark_metric - custom_metric) / benchmark_metric) * 100
    
    Positive improvement means the custom model performs better (lower error).
    Negative improvement means the benchmark model performs better.
    
    The recommendation is based on the average improvement across all three metrics.
    If the average improvement is positive, recommend the custom model; otherwise,
    recommend the benchmark.
    
    Args:
        custom_metrics: Performance metrics for the custom model
        benchmark_metrics: Performance metrics for the benchmark model
        
    Returns:
        ComparisonReport with improvement percentages and recommendation
        
    Raises:
        ValueError: If benchmark metrics contain zero values (division by zero)
        
    Examples:
        >>> custom = PerformanceMetrics(mae=10.0, rmse=15.0, mape=5.0, sample_size=100)
        >>> benchmark = PerformanceMetrics(mae=12.0, rmse=18.0, mape=6.0, sample_size=100)
        >>> report = generate_comparison_report(custom, benchmark)
        >>> report.mae_improvement
        16.666666666666664
        >>> report.recommendation
        'Use custom model'
    """
    # Validate benchmark metrics are not zero (would cause division by zero)
    if benchmark_metrics.mae == 0:
        raise ValueError("Benchmark MAE is zero, cannot compute improvement percentage")
    if benchmark_metrics.rmse == 0:
        raise ValueError("Benchmark RMSE is zero, cannot compute improvement percentage")
    if benchmark_metrics.mape == 0:
        raise ValueError("Benchmark MAPE is zero, cannot compute improvement percentage")
    
    # Calculate percentage improvements
    # Formula: ((benchmark - custom) / benchmark) * 100
    # Positive value means custom is better (lower error)
    mae_improvement = ((benchmark_metrics.mae - custom_metrics.mae) / benchmark_metrics.mae) * 100
    rmse_improvement = ((benchmark_metrics.rmse - custom_metrics.rmse) / benchmark_metrics.rmse) * 100
    mape_improvement = ((benchmark_metrics.mape - custom_metrics.mape) / benchmark_metrics.mape) * 100
    
    # Generate recommendation based on average improvement
    # If average improvement is positive, custom model is better
    average_improvement = (mae_improvement + rmse_improvement + mape_improvement) / 3
    
    if average_improvement > 0:
        recommendation = "Use custom model"
    else:
        recommendation = "Use benchmark"
    
    return ComparisonReport(
        custom_metrics=custom_metrics,
        benchmark_metrics=benchmark_metrics,
        mae_improvement=mae_improvement,
        rmse_improvement=rmse_improvement,
        mape_improvement=mape_improvement,
        recommendation=recommendation
    )
