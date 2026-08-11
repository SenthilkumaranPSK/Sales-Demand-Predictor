"""
Unit tests for performance metrics computation module.

Tests cover:
- Basic metric calculations (MAE, RMSE, MAPE)
- Edge cases (empty arrays, single values, division by zero)
- Error handling (mismatched lengths, invalid values)
- PerformanceMetrics dataclass

Requirements: 2.2, 8.2
"""

import pytest
import numpy as np
from datetime import datetime
from src.training.metrics import (
    compute_mae,
    compute_rmse,
    compute_mape,
    PerformanceMetrics
)


class TestComputeMAE:
    """Tests for Mean Absolute Error calculation."""
    
    def test_mae_perfect_predictions(self):
        """MAE should be 0 when predictions match actuals exactly."""
        predictions = np.array([100, 200, 300, 400])
        actuals = np.array([100, 200, 300, 400])
        
        mae = compute_mae(predictions, actuals)
        
        assert mae == 0.0
    
    def test_mae_basic_calculation(self):
        """MAE should correctly calculate average absolute error."""
        predictions = np.array([100, 200, 300])
        actuals = np.array([110, 190, 310])
        # Errors: |100-110| + |200-190| + |300-310| = 10 + 10 + 10 = 30
        # MAE = 30 / 3 = 10
        
        mae = compute_mae(predictions, actuals)
        
        assert mae == 10.0
    
    def test_mae_with_negative_errors(self):
        """MAE should handle both positive and negative errors."""
        predictions = np.array([100, 200, 300])
        actuals = np.array([90, 210, 295])
        # Errors: |100-90| + |200-210| + |300-295| = 10 + 10 + 5 = 25
        # MAE = 25 / 3 = 8.333...
        
        mae = compute_mae(predictions, actuals)
        
        assert abs(mae - 8.333333) < 0.001
    
    def test_mae_single_value(self):
        """MAE should work with single value arrays."""
        predictions = np.array([100])
        actuals = np.array([110])
        
        mae = compute_mae(predictions, actuals)
        
        assert mae == 10.0
    
    def test_mae_large_values(self):
        """MAE should handle large values correctly."""
        predictions = np.array([1000000, 2000000, 3000000])
        actuals = np.array([1100000, 1900000, 3100000])
        
        mae = compute_mae(predictions, actuals)
        
        assert mae == 100000.0
    
    def test_mae_with_floats(self):
        """MAE should work with floating point values."""
        predictions = np.array([1.5, 2.7, 3.9])
        actuals = np.array([1.6, 2.5, 4.0])
        # Errors: |1.5-1.6| + |2.7-2.5| + |3.9-4.0| = 0.1 + 0.2 + 0.1 = 0.4
        # MAE = 0.4 / 3 = 0.133...
        
        mae = compute_mae(predictions, actuals)
        
        assert abs(mae - 0.133333) < 0.001
    
    def test_mae_empty_arrays(self):
        """MAE should raise ValueError for empty arrays."""
        predictions = np.array([])
        actuals = np.array([])
        
        with pytest.raises(ValueError, match="Input arrays cannot be empty"):
            compute_mae(predictions, actuals)
    
    def test_mae_mismatched_lengths(self):
        """MAE should raise ValueError for arrays of different lengths."""
        predictions = np.array([100, 200, 300])
        actuals = np.array([100, 200])
        
        with pytest.raises(ValueError, match="must have the same length"):
            compute_mae(predictions, actuals)
    
    def test_mae_with_nan_values(self):
        """MAE should raise ValueError when arrays contain NaN."""
        predictions = np.array([100, np.nan, 300])
        actuals = np.array([110, 190, 310])
        
        with pytest.raises(ValueError, match="contain NaN values"):
            compute_mae(predictions, actuals)
    
    def test_mae_with_infinite_values(self):
        """MAE should raise ValueError when arrays contain infinite values."""
        predictions = np.array([100, np.inf, 300])
        actuals = np.array([110, 190, 310])
        
        with pytest.raises(ValueError, match="contain infinite values"):
            compute_mae(predictions, actuals)
    
    def test_mae_with_list_input(self):
        """MAE should accept Python lists and convert them to arrays."""
        predictions = [100, 200, 300]
        actuals = [110, 190, 310]
        
        mae = compute_mae(predictions, actuals)
        
        assert mae == 10.0


class TestComputeRMSE:
    """Tests for Root Mean Squared Error calculation."""
    
    def test_rmse_perfect_predictions(self):
        """RMSE should be 0 when predictions match actuals exactly."""
        predictions = np.array([100, 200, 300, 400])
        actuals = np.array([100, 200, 300, 400])
        
        rmse = compute_rmse(predictions, actuals)
        
        assert rmse == 0.0
    
    def test_rmse_basic_calculation(self):
        """RMSE should correctly calculate root mean squared error."""
        predictions = np.array([100, 200, 300])
        actuals = np.array([110, 190, 310])
        # Squared errors: (100-110)² + (200-190)² + (300-310)² = 100 + 100 + 100 = 300
        # MSE = 300 / 3 = 100
        # RMSE = sqrt(100) = 10
        
        rmse = compute_rmse(predictions, actuals)
        
        assert rmse == 10.0
    
    def test_rmse_penalizes_large_errors(self):
        """RMSE should penalize large errors more than MAE."""
        predictions = np.array([100, 200, 300])
        actuals = np.array([100, 200, 350])  # One large error
        # Squared errors: 0 + 0 + 2500 = 2500
        # MSE = 2500 / 3 = 833.333...
        # RMSE = sqrt(833.333...) = 28.867...
        
        rmse = compute_rmse(predictions, actuals)
        mae = compute_mae(predictions, actuals)
        
        # RMSE should be larger than MAE due to squaring
        assert rmse > mae
        assert abs(rmse - 28.867513) < 0.001
    
    def test_rmse_single_value(self):
        """RMSE should work with single value arrays."""
        predictions = np.array([100])
        actuals = np.array([110])
        
        rmse = compute_rmse(predictions, actuals)
        
        assert rmse == 10.0
    
    def test_rmse_with_floats(self):
        """RMSE should work with floating point values."""
        predictions = np.array([1.0, 2.0, 3.0])
        actuals = np.array([1.5, 2.5, 3.5])
        # Squared errors: 0.25 + 0.25 + 0.25 = 0.75
        # MSE = 0.75 / 3 = 0.25
        # RMSE = sqrt(0.25) = 0.5
        
        rmse = compute_rmse(predictions, actuals)
        
        assert rmse == 0.5
    
    def test_rmse_empty_arrays(self):
        """RMSE should raise ValueError for empty arrays."""
        predictions = np.array([])
        actuals = np.array([])
        
        with pytest.raises(ValueError, match="Input arrays cannot be empty"):
            compute_rmse(predictions, actuals)
    
    def test_rmse_mismatched_lengths(self):
        """RMSE should raise ValueError for arrays of different lengths."""
        predictions = np.array([100, 200, 300])
        actuals = np.array([100, 200])
        
        with pytest.raises(ValueError, match="must have the same length"):
            compute_rmse(predictions, actuals)
    
    def test_rmse_with_nan_values(self):
        """RMSE should raise ValueError when arrays contain NaN."""
        predictions = np.array([100, np.nan, 300])
        actuals = np.array([110, 190, 310])
        
        with pytest.raises(ValueError, match="contain NaN values"):
            compute_rmse(predictions, actuals)
    
    def test_rmse_with_infinite_values(self):
        """RMSE should raise ValueError when arrays contain infinite values."""
        predictions = np.array([100, np.inf, 300])
        actuals = np.array([110, 190, 310])
        
        with pytest.raises(ValueError, match="contain infinite values"):
            compute_rmse(predictions, actuals)


class TestComputeMAPE:
    """Tests for Mean Absolute Percentage Error calculation."""
    
    def test_mape_perfect_predictions(self):
        """MAPE should be 0 when predictions match actuals exactly."""
        predictions = np.array([100, 200, 300, 400])
        actuals = np.array([100, 200, 300, 400])
        
        mape = compute_mape(predictions, actuals)
        
        assert mape == 0.0
    
    def test_mape_basic_calculation(self):
        """MAPE should correctly calculate mean absolute percentage error."""
        predictions = np.array([100, 200, 300])
        actuals = np.array([110, 190, 310])
        # Percentage errors: |110-100|/110 + |190-200|/190 + |310-300|/310
        #                  = 10/110 + 10/190 + 10/310
        #                  = 0.0909 + 0.0526 + 0.0323 = 0.1758
        # MAPE = 0.1758 / 3 * 100 = 5.86%
        
        mape = compute_mape(predictions, actuals)
        
        assert abs(mape - 5.86) < 0.1
    
    def test_mape_scale_independence(self):
        """MAPE should be scale-independent."""
        # Same percentage errors but different scales
        predictions1 = np.array([100, 200, 300])
        actuals1 = np.array([110, 220, 330])
        
        predictions2 = np.array([1000, 2000, 3000])
        actuals2 = np.array([1100, 2200, 3300])
        
        mape1 = compute_mape(predictions1, actuals1)
        mape2 = compute_mape(predictions2, actuals2)
        
        # Both should have same MAPE (10% error)
        assert abs(mape1 - mape2) < 0.001
        assert abs(mape1 - 9.09) < 0.1
    
    def test_mape_single_value(self):
        """MAPE should work with single value arrays."""
        predictions = np.array([100])
        actuals = np.array([110])
        # Percentage error: |110-100|/110 = 10/110 = 0.0909
        # MAPE = 0.0909 * 100 = 9.09%
        
        mape = compute_mape(predictions, actuals)
        
        assert abs(mape - 9.09) < 0.1
    
    def test_mape_with_zero_actuals_excluded(self):
        """MAPE should exclude zero actual values from calculation."""
        predictions = np.array([100, 200, 300])
        actuals = np.array([110, 0, 310])  # One zero actual
        # Should only use first and third values
        # Percentage errors: |110-100|/110 + |310-300|/310
        #                  = 10/110 + 10/310 = 0.0909 + 0.0323 = 0.1232
        # MAPE = 0.1232 / 2 * 100 = 6.16%
        
        mape = compute_mape(predictions, actuals)
        
        assert abs(mape - 6.16) < 0.1
    
    def test_mape_all_zero_actuals(self):
        """MAPE should raise ValueError when all actual values are zero."""
        predictions = np.array([100, 200, 300])
        actuals = np.array([0, 0, 0])
        
        with pytest.raises(ValueError, match="all actual values are zero"):
            compute_mape(predictions, actuals)
    
    def test_mape_empty_arrays(self):
        """MAPE should raise ValueError for empty arrays."""
        predictions = np.array([])
        actuals = np.array([])
        
        with pytest.raises(ValueError, match="Input arrays cannot be empty"):
            compute_mape(predictions, actuals)
    
    def test_mape_mismatched_lengths(self):
        """MAPE should raise ValueError for arrays of different lengths."""
        predictions = np.array([100, 200, 300])
        actuals = np.array([100, 200])
        
        with pytest.raises(ValueError, match="must have the same length"):
            compute_mape(predictions, actuals)
    
    def test_mape_with_nan_values(self):
        """MAPE should raise ValueError when arrays contain NaN."""
        predictions = np.array([100, np.nan, 300])
        actuals = np.array([110, 190, 310])
        
        with pytest.raises(ValueError, match="contain NaN values"):
            compute_mape(predictions, actuals)
    
    def test_mape_with_infinite_values(self):
        """MAPE should raise ValueError when arrays contain infinite values."""
        predictions = np.array([100, np.inf, 300])
        actuals = np.array([110, 190, 310])
        
        with pytest.raises(ValueError, match="contain infinite values"):
            compute_mape(predictions, actuals)
    
    def test_mape_with_negative_values(self):
        """MAPE should handle negative values correctly."""
        predictions = np.array([-100, -200, -300])
        actuals = np.array([-110, -190, -310])
        
        # Should calculate percentage errors correctly with negative values
        mape = compute_mape(predictions, actuals)
        
        assert mape > 0  # MAPE should always be positive


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics dataclass."""
    
    def test_performance_metrics_creation(self):
        """PerformanceMetrics should be created with all required fields."""
        metrics = PerformanceMetrics(
            mae=10.5,
            rmse=12.3,
            mape=5.2,
            sample_size=1000
        )
        
        assert metrics.mae == 10.5
        assert metrics.rmse == 12.3
        assert metrics.mape == 5.2
        assert metrics.sample_size == 1000
        assert metrics.evaluation_period is None
    
    def test_performance_metrics_with_evaluation_period(self):
        """PerformanceMetrics should support optional evaluation period."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 12, 31)
        
        metrics = PerformanceMetrics(
            mae=10.5,
            rmse=12.3,
            mape=5.2,
            sample_size=1000,
            evaluation_period=(start_date, end_date)
        )
        
        assert metrics.evaluation_period == (start_date, end_date)
    
    def test_performance_metrics_equality(self):
        """PerformanceMetrics instances with same values should be equal."""
        metrics1 = PerformanceMetrics(
            mae=10.5,
            rmse=12.3,
            mape=5.2,
            sample_size=1000
        )
        metrics2 = PerformanceMetrics(
            mae=10.5,
            rmse=12.3,
            mape=5.2,
            sample_size=1000
        )
        
        assert metrics1 == metrics2


class TestMetricsIntegration:
    """Integration tests for metrics computation."""
    
    def test_all_metrics_on_same_data(self):
        """All metrics should be computable on the same dataset."""
        predictions = np.array([100, 200, 300, 400, 500])
        actuals = np.array([110, 190, 310, 390, 510])
        
        mae = compute_mae(predictions, actuals)
        rmse = compute_rmse(predictions, actuals)
        mape = compute_mape(predictions, actuals)
        
        # All metrics should be positive
        assert mae > 0
        assert rmse > 0
        assert mape > 0
        
        # RMSE should be >= MAE (due to squaring)
        assert rmse >= mae
    
    def test_create_performance_metrics_from_calculations(self):
        """PerformanceMetrics can be created from computed metrics."""
        predictions = np.array([100, 200, 300, 400, 500])
        actuals = np.array([110, 190, 310, 390, 510])
        
        mae = compute_mae(predictions, actuals)
        rmse = compute_rmse(predictions, actuals)
        mape = compute_mape(predictions, actuals)
        
        metrics = PerformanceMetrics(
            mae=mae,
            rmse=rmse,
            mape=mape,
            sample_size=len(predictions),
            evaluation_period=(datetime(2024, 1, 1), datetime(2024, 12, 31))
        )
        
        assert metrics.mae == mae
        assert metrics.rmse == rmse
        assert metrics.mape == mape
        assert metrics.sample_size == 5
    
    def test_metrics_with_realistic_forecast_data(self):
        """Test metrics with realistic demand forecasting scenario."""
        # Simulated weekly demand predictions vs actuals
        predictions = np.array([1200, 1350, 1100, 1450, 1300, 1250, 1400])
        actuals = np.array([1180, 1400, 1150, 1420, 1280, 1300, 1380])
        
        mae = compute_mae(predictions, actuals)
        rmse = compute_rmse(predictions, actuals)
        mape = compute_mape(predictions, actuals)
        
        # Verify metrics are in reasonable ranges for this data
        assert 0 < mae < 100  # Average error should be reasonable
        assert 0 < rmse < 150  # RMSE should be slightly higher than MAE
        assert 0 < mape < 10  # Percentage error should be single digits
        
        # Create comprehensive metrics object
        metrics = PerformanceMetrics(
            mae=mae,
            rmse=rmse,
            mape=mape,
            sample_size=len(predictions)
        )
        
        assert metrics.sample_size == 7



class TestGenerateComparisonReport:
    """Tests for model comparison report generation."""
    
    def test_comparison_report_custom_better(self):
        """Comparison report should recommend custom model when it performs better."""
        custom_metrics = PerformanceMetrics(
            mae=10.0,
            rmse=15.0,
            mape=5.0,
            sample_size=100
        )
        benchmark_metrics = PerformanceMetrics(
            mae=12.0,
            rmse=18.0,
            mape=6.0,
            sample_size=100
        )
        
        from src.training.metrics import generate_comparison_report
        report = generate_comparison_report(custom_metrics, benchmark_metrics)
        
        # Verify improvements are calculated correctly
        # MAE improvement: (12 - 10) / 12 * 100 = 16.67%
        assert abs(report.mae_improvement - 16.666666) < 0.001
        # RMSE improvement: (18 - 15) / 18 * 100 = 16.67%
        assert abs(report.rmse_improvement - 16.666666) < 0.001
        # MAPE improvement: (6 - 5) / 6 * 100 = 16.67%
        assert abs(report.mape_improvement - 16.666666) < 0.001
        
        # Should recommend custom model
        assert report.recommendation == "Use custom model"
        
        # Verify metrics are preserved
        assert report.custom_metrics == custom_metrics
        assert report.benchmark_metrics == benchmark_metrics
    
    def test_comparison_report_benchmark_better(self):
        """Comparison report should recommend benchmark when it performs better."""
        custom_metrics = PerformanceMetrics(
            mae=15.0,
            rmse=22.0,
            mape=8.0,
            sample_size=100
        )
        benchmark_metrics = PerformanceMetrics(
            mae=12.0,
            rmse=18.0,
            mape=6.0,
            sample_size=100
        )
        
        from src.training.metrics import generate_comparison_report
        report = generate_comparison_report(custom_metrics, benchmark_metrics)
        
        # Verify improvements are negative (custom is worse)
        # MAE improvement: (12 - 15) / 12 * 100 = -25%
        assert abs(report.mae_improvement - (-25.0)) < 0.001
        # RMSE improvement: (18 - 22) / 18 * 100 = -22.22%
        assert abs(report.rmse_improvement - (-22.222222)) < 0.001
        # MAPE improvement: (6 - 8) / 6 * 100 = -33.33%
        assert abs(report.mape_improvement - (-33.333333)) < 0.001
        
        # Should recommend benchmark
        assert report.recommendation == "Use benchmark"
    
    def test_comparison_report_mixed_performance(self):
        """Comparison report should handle mixed performance (some metrics better, some worse)."""
        custom_metrics = PerformanceMetrics(
            mae=10.0,  # Better
            rmse=20.0,  # Worse
            mape=5.0,   # Better
            sample_size=100
        )
        benchmark_metrics = PerformanceMetrics(
            mae=12.0,
            rmse=18.0,
            mape=6.0,
            sample_size=100
        )
        
        from src.training.metrics import generate_comparison_report
        report = generate_comparison_report(custom_metrics, benchmark_metrics)
        
        # MAE improvement: (12 - 10) / 12 * 100 = 16.67% (positive)
        assert report.mae_improvement > 0
        # RMSE improvement: (18 - 20) / 18 * 100 = -11.11% (negative)
        assert report.rmse_improvement < 0
        # MAPE improvement: (6 - 5) / 6 * 100 = 16.67% (positive)
        assert report.mape_improvement > 0
        
        # Average improvement: (16.67 - 11.11 + 16.67) / 3 = 7.41% (positive)
        # Should recommend custom model based on average
        assert report.recommendation == "Use custom model"
    
    def test_comparison_report_equal_performance(self):
        """Comparison report should handle equal performance."""
        custom_metrics = PerformanceMetrics(
            mae=10.0,
            rmse=15.0,
            mape=5.0,
            sample_size=100
        )
        benchmark_metrics = PerformanceMetrics(
            mae=10.0,
            rmse=15.0,
            mape=5.0,
            sample_size=100
        )
        
        from src.training.metrics import generate_comparison_report
        report = generate_comparison_report(custom_metrics, benchmark_metrics)
        
        # All improvements should be 0
        assert report.mae_improvement == 0.0
        assert report.rmse_improvement == 0.0
        assert report.mape_improvement == 0.0
        
        # Average is 0, so should recommend benchmark (not positive)
        assert report.recommendation == "Use benchmark"
    
    def test_comparison_report_large_improvement(self):
        """Comparison report should handle large improvements correctly."""
        custom_metrics = PerformanceMetrics(
            mae=5.0,
            rmse=8.0,
            mape=2.0,
            sample_size=100
        )
        benchmark_metrics = PerformanceMetrics(
            mae=20.0,
            rmse=30.0,
            mape=10.0,
            sample_size=100
        )
        
        from src.training.metrics import generate_comparison_report
        report = generate_comparison_report(custom_metrics, benchmark_metrics)
        
        # MAE improvement: (20 - 5) / 20 * 100 = 75%
        assert abs(report.mae_improvement - 75.0) < 0.001
        # RMSE improvement: (30 - 8) / 30 * 100 = 73.33%
        assert abs(report.rmse_improvement - 73.333333) < 0.001
        # MAPE improvement: (10 - 2) / 10 * 100 = 80%
        assert abs(report.mape_improvement - 80.0) < 0.001
        
        assert report.recommendation == "Use custom model"
    
    def test_comparison_report_small_differences(self):
        """Comparison report should handle small differences correctly."""
        custom_metrics = PerformanceMetrics(
            mae=10.0,
            rmse=15.0,
            mape=5.0,
            sample_size=100
        )
        benchmark_metrics = PerformanceMetrics(
            mae=10.1,
            rmse=15.1,
            mape=5.05,
            sample_size=100
        )
        
        from src.training.metrics import generate_comparison_report
        report = generate_comparison_report(custom_metrics, benchmark_metrics)
        
        # Small positive improvements
        assert 0 < report.mae_improvement < 2
        assert 0 < report.rmse_improvement < 2
        assert 0 < report.mape_improvement < 2
        
        assert report.recommendation == "Use custom model"
    
    def test_comparison_report_zero_benchmark_mae(self):
        """Comparison report should raise ValueError when benchmark MAE is zero."""
        custom_metrics = PerformanceMetrics(
            mae=10.0,
            rmse=15.0,
            mape=5.0,
            sample_size=100
        )
        benchmark_metrics = PerformanceMetrics(
            mae=0.0,  # Invalid
            rmse=15.0,
            mape=5.0,
            sample_size=100
        )
        
        from src.training.metrics import generate_comparison_report
        with pytest.raises(ValueError, match="Benchmark MAE is zero"):
            generate_comparison_report(custom_metrics, benchmark_metrics)
    
    def test_comparison_report_zero_benchmark_rmse(self):
        """Comparison report should raise ValueError when benchmark RMSE is zero."""
        custom_metrics = PerformanceMetrics(
            mae=10.0,
            rmse=15.0,
            mape=5.0,
            sample_size=100
        )
        benchmark_metrics = PerformanceMetrics(
            mae=10.0,
            rmse=0.0,  # Invalid
            mape=5.0,
            sample_size=100
        )
        
        from src.training.metrics import generate_comparison_report
        with pytest.raises(ValueError, match="Benchmark RMSE is zero"):
            generate_comparison_report(custom_metrics, benchmark_metrics)
    
    def test_comparison_report_zero_benchmark_mape(self):
        """Comparison report should raise ValueError when benchmark MAPE is zero."""
        custom_metrics = PerformanceMetrics(
            mae=10.0,
            rmse=15.0,
            mape=5.0,
            sample_size=100
        )
        benchmark_metrics = PerformanceMetrics(
            mae=10.0,
            rmse=15.0,
            mape=0.0,  # Invalid
            sample_size=100
        )
        
        from src.training.metrics import generate_comparison_report
        with pytest.raises(ValueError, match="Benchmark MAPE is zero"):
            generate_comparison_report(custom_metrics, benchmark_metrics)
    
    def test_comparison_report_with_evaluation_periods(self):
        """Comparison report should preserve evaluation periods from metrics."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 12, 31)
        
        custom_metrics = PerformanceMetrics(
            mae=10.0,
            rmse=15.0,
            mape=5.0,
            sample_size=100,
            evaluation_period=(start_date, end_date)
        )
        benchmark_metrics = PerformanceMetrics(
            mae=12.0,
            rmse=18.0,
            mape=6.0,
            sample_size=100,
            evaluation_period=(start_date, end_date)
        )
        
        from src.training.metrics import generate_comparison_report
        report = generate_comparison_report(custom_metrics, benchmark_metrics)
        
        # Verify evaluation periods are preserved
        assert report.custom_metrics.evaluation_period == (start_date, end_date)
        assert report.benchmark_metrics.evaluation_period == (start_date, end_date)
    
    def test_comparison_report_realistic_scenario(self):
        """Test comparison report with realistic demand forecasting metrics."""
        # Custom model trained with domain-specific features
        custom_metrics = PerformanceMetrics(
            mae=45.2,
            rmse=62.8,
            mape=3.8,
            sample_size=1000,
            evaluation_period=(datetime(2024, 1, 1), datetime(2024, 3, 31))
        )
        
        # Amazon Forecast benchmark
        benchmark_metrics = PerformanceMetrics(
            mae=52.1,
            rmse=71.5,
            mape=4.3,
            sample_size=1000,
            evaluation_period=(datetime(2024, 1, 1), datetime(2024, 3, 31))
        )
        
        from src.training.metrics import generate_comparison_report
        report = generate_comparison_report(custom_metrics, benchmark_metrics)
        
        # Custom model should show improvements
        assert report.mae_improvement > 0
        assert report.rmse_improvement > 0
        assert report.mape_improvement > 0
        
        # Should recommend custom model
        assert report.recommendation == "Use custom model"
        
        # Verify improvement percentages are reasonable (around 10-15%)
        assert 10 < report.mae_improvement < 20
        assert 10 < report.rmse_improvement < 20
        assert 10 < report.mape_improvement < 20


class TestComparisonReportDataclass:
    """Tests for ComparisonReport dataclass."""
    
    def test_comparison_report_creation(self):
        """ComparisonReport should be created with all required fields."""
        from src.training.metrics import ComparisonReport
        
        custom_metrics = PerformanceMetrics(
            mae=10.0,
            rmse=15.0,
            mape=5.0,
            sample_size=100
        )
        benchmark_metrics = PerformanceMetrics(
            mae=12.0,
            rmse=18.0,
            mape=6.0,
            sample_size=100
        )
        
        report = ComparisonReport(
            custom_metrics=custom_metrics,
            benchmark_metrics=benchmark_metrics,
            mae_improvement=16.67,
            rmse_improvement=16.67,
            mape_improvement=16.67,
            recommendation="Use custom model"
        )
        
        assert report.custom_metrics == custom_metrics
        assert report.benchmark_metrics == benchmark_metrics
        assert report.mae_improvement == 16.67
        assert report.rmse_improvement == 16.67
        assert report.mape_improvement == 16.67
        assert report.recommendation == "Use custom model"
    
    def test_comparison_report_equality(self):
        """ComparisonReport instances with same values should be equal."""
        from src.training.metrics import ComparisonReport
        
        custom_metrics = PerformanceMetrics(
            mae=10.0,
            rmse=15.0,
            mape=5.0,
            sample_size=100
        )
        benchmark_metrics = PerformanceMetrics(
            mae=12.0,
            rmse=18.0,
            mape=6.0,
            sample_size=100
        )
        
        report1 = ComparisonReport(
            custom_metrics=custom_metrics,
            benchmark_metrics=benchmark_metrics,
            mae_improvement=16.67,
            rmse_improvement=16.67,
            mape_improvement=16.67,
            recommendation="Use custom model"
        )
        
        report2 = ComparisonReport(
            custom_metrics=custom_metrics,
            benchmark_metrics=benchmark_metrics,
            mae_improvement=16.67,
            rmse_improvement=16.67,
            mape_improvement=16.67,
            recommendation="Use custom model"
        )
        
        assert report1 == report2
