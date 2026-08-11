"""
Example usage of model comparison report generation.

This example demonstrates how to:
1. Create PerformanceMetrics for custom and benchmark models
2. Generate a comparison report
3. Interpret the results and recommendations

Requirements: 3.4, 8.3
"""

from datetime import datetime
from src.training.metrics import (
    PerformanceMetrics,
    ComparisonReport,
    generate_comparison_report
)


def example_basic_comparison():
    """Basic example: Custom model performs better than benchmark."""
    print("=" * 60)
    print("Example 1: Custom Model Outperforms Benchmark")
    print("=" * 60)
    
    # Custom model metrics (better performance)
    custom_metrics = PerformanceMetrics(
        mae=45.2,
        rmse=62.8,
        mape=3.8,
        sample_size=1000,
        evaluation_period=(datetime(2024, 1, 1), datetime(2024, 3, 31))
    )
    
    # Amazon Forecast benchmark metrics
    benchmark_metrics = PerformanceMetrics(
        mae=52.1,
        rmse=71.5,
        mape=4.3,
        sample_size=1000,
        evaluation_period=(datetime(2024, 1, 1), datetime(2024, 3, 31))
    )
    
    # Generate comparison report
    report = generate_comparison_report(custom_metrics, benchmark_metrics)
    
    # Display results
    print(f"\nCustom Model Metrics:")
    print(f"  MAE:  {custom_metrics.mae:.2f}")
    print(f"  RMSE: {custom_metrics.rmse:.2f}")
    print(f"  MAPE: {custom_metrics.mape:.2f}%")
    
    print(f"\nBenchmark Model Metrics:")
    print(f"  MAE:  {benchmark_metrics.mae:.2f}")
    print(f"  RMSE: {benchmark_metrics.rmse:.2f}")
    print(f"  MAPE: {benchmark_metrics.mape:.2f}%")
    
    print(f"\nPerformance Improvements:")
    print(f"  MAE Improvement:  {report.mae_improvement:+.2f}%")
    print(f"  RMSE Improvement: {report.rmse_improvement:+.2f}%")
    print(f"  MAPE Improvement: {report.mape_improvement:+.2f}%")
    
    print(f"\nRecommendation: {report.recommendation}")
    print()


def example_benchmark_better():
    """Example: Benchmark performs better than custom model."""
    print("=" * 60)
    print("Example 2: Benchmark Outperforms Custom Model")
    print("=" * 60)
    
    # Custom model metrics (worse performance)
    custom_metrics = PerformanceMetrics(
        mae=65.5,
        rmse=88.2,
        mape=5.9,
        sample_size=1000
    )
    
    # Benchmark metrics (better performance)
    benchmark_metrics = PerformanceMetrics(
        mae=52.1,
        rmse=71.5,
        mape=4.3,
        sample_size=1000
    )
    
    # Generate comparison report
    report = generate_comparison_report(custom_metrics, benchmark_metrics)
    
    # Display results
    print(f"\nCustom Model Metrics:")
    print(f"  MAE:  {custom_metrics.mae:.2f}")
    print(f"  RMSE: {custom_metrics.rmse:.2f}")
    print(f"  MAPE: {custom_metrics.mape:.2f}%")
    
    print(f"\nBenchmark Model Metrics:")
    print(f"  MAE:  {benchmark_metrics.mae:.2f}")
    print(f"  RMSE: {benchmark_metrics.rmse:.2f}")
    print(f"  MAPE: {benchmark_metrics.mape:.2f}%")
    
    print(f"\nPerformance Improvements:")
    print(f"  MAE Improvement:  {report.mae_improvement:+.2f}%")
    print(f"  RMSE Improvement: {report.rmse_improvement:+.2f}%")
    print(f"  MAPE Improvement: {report.mape_improvement:+.2f}%")
    
    print(f"\nRecommendation: {report.recommendation}")
    print(f"\nNote: Negative improvements indicate the custom model performs worse.")
    print()


def example_mixed_performance():
    """Example: Mixed performance (some metrics better, some worse)."""
    print("=" * 60)
    print("Example 3: Mixed Performance")
    print("=" * 60)
    
    # Custom model: better MAE and MAPE, worse RMSE
    custom_metrics = PerformanceMetrics(
        mae=45.0,   # Better
        rmse=75.0,  # Worse
        mape=3.5,   # Better
        sample_size=1000
    )
    
    # Benchmark metrics
    benchmark_metrics = PerformanceMetrics(
        mae=52.1,
        rmse=71.5,
        mape=4.3,
        sample_size=1000
    )
    
    # Generate comparison report
    report = generate_comparison_report(custom_metrics, benchmark_metrics)
    
    # Display results
    print(f"\nCustom Model Metrics:")
    print(f"  MAE:  {custom_metrics.mae:.2f} ✓ Better")
    print(f"  RMSE: {custom_metrics.rmse:.2f} ✗ Worse")
    print(f"  MAPE: {custom_metrics.mape:.2f}% ✓ Better")
    
    print(f"\nBenchmark Model Metrics:")
    print(f"  MAE:  {benchmark_metrics.mae:.2f}")
    print(f"  RMSE: {benchmark_metrics.rmse:.2f}")
    print(f"  MAPE: {benchmark_metrics.mape:.2f}%")
    
    print(f"\nPerformance Improvements:")
    print(f"  MAE Improvement:  {report.mae_improvement:+.2f}% (positive = better)")
    print(f"  RMSE Improvement: {report.rmse_improvement:+.2f}% (negative = worse)")
    print(f"  MAPE Improvement: {report.mape_improvement:+.2f}% (positive = better)")
    
    avg_improvement = (report.mae_improvement + report.rmse_improvement + report.mape_improvement) / 3
    print(f"\nAverage Improvement: {avg_improvement:+.2f}%")
    print(f"Recommendation: {report.recommendation}")
    print(f"\nNote: Recommendation is based on average improvement across all metrics.")
    print()


def example_large_improvement():
    """Example: Custom model shows significant improvement."""
    print("=" * 60)
    print("Example 4: Significant Improvement")
    print("=" * 60)
    
    # Custom model with domain-specific features (much better)
    custom_metrics = PerformanceMetrics(
        mae=25.0,
        rmse=35.0,
        mape=2.1,
        sample_size=1000
    )
    
    # Benchmark metrics
    benchmark_metrics = PerformanceMetrics(
        mae=52.1,
        rmse=71.5,
        mape=4.3,
        sample_size=1000
    )
    
    # Generate comparison report
    report = generate_comparison_report(custom_metrics, benchmark_metrics)
    
    # Display results
    print(f"\nCustom Model Metrics:")
    print(f"  MAE:  {custom_metrics.mae:.2f}")
    print(f"  RMSE: {custom_metrics.rmse:.2f}")
    print(f"  MAPE: {custom_metrics.mape:.2f}%")
    
    print(f"\nBenchmark Model Metrics:")
    print(f"  MAE:  {benchmark_metrics.mae:.2f}")
    print(f"  RMSE: {benchmark_metrics.rmse:.2f}")
    print(f"  MAPE: {benchmark_metrics.mape:.2f}%")
    
    print(f"\nPerformance Improvements:")
    print(f"  MAE Improvement:  {report.mae_improvement:+.2f}%")
    print(f"  RMSE Improvement: {report.rmse_improvement:+.2f}%")
    print(f"  MAPE Improvement: {report.mape_improvement:+.2f}%")
    
    print(f"\nRecommendation: {report.recommendation}")
    print(f"\nNote: Large improvements (>50%) indicate domain-specific features")
    print(f"      are providing significant value over automated approaches.")
    print()


def example_programmatic_usage():
    """Example: Using comparison report in automated decision making."""
    print("=" * 60)
    print("Example 5: Programmatic Usage")
    print("=" * 60)
    
    # Simulate multiple model comparisons
    models = [
        ("Product A", 45.2, 62.8, 3.8, 52.1, 71.5, 4.3),
        ("Product B", 65.5, 88.2, 5.9, 52.1, 71.5, 4.3),
        ("Product C", 48.0, 66.0, 4.0, 52.1, 71.5, 4.3),
    ]
    
    print("\nModel Selection Summary:")
    print("-" * 60)
    
    for product, c_mae, c_rmse, c_mape, b_mae, b_rmse, b_mape in models:
        custom = PerformanceMetrics(c_mae, c_rmse, c_mape, 1000)
        benchmark = PerformanceMetrics(b_mae, b_rmse, b_mape, 1000)
        
        report = generate_comparison_report(custom, benchmark)
        
        avg_improvement = (report.mae_improvement + report.rmse_improvement + report.mape_improvement) / 3
        
        print(f"\n{product}:")
        print(f"  Average Improvement: {avg_improvement:+.2f}%")
        print(f"  Decision: {report.recommendation}")
        
        # Programmatic decision making
        if avg_improvement > 10:
            print(f"  Action: Deploy custom model (significant improvement)")
        elif avg_improvement > 0:
            print(f"  Action: Deploy custom model (marginal improvement)")
        else:
            print(f"  Action: Use benchmark (custom model underperforms)")
    
    print()


if __name__ == "__main__":
    example_basic_comparison()
    example_benchmark_better()
    example_mixed_performance()
    example_large_improvement()
    example_programmatic_usage()
