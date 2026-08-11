"""
Unit tests for ForecastingEngine.

Tests forecast generation, confidence interval calculation, and error handling.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from io import BytesIO
import joblib

from src.inference.forecasting_engine import (
    ForecastingEngine,
    ForecastResult,
    ConfidenceInterval
)
from src.registry.model_registry import ModelMetadata
from sklearn.linear_model import LinearRegression
from prophet import Prophet


@pytest.fixture
def forecasting_engine():
    """Create a ForecastingEngine instance."""
    return ForecastingEngine()


@pytest.fixture
def mock_sklearn_model():
    """Create a mock scikit-learn model."""
    model = LinearRegression()
    # Train on dummy data with 4 features to match metadata
    # Features: price, is_holiday, day_of_week, month
    X = np.array([
        [100, 0, 0, 1],
        [105, 0, 1, 1],
        [110, 1, 2, 1],
        [115, 0, 3, 1],
        [120, 0, 4, 1]
    ])
    y = np.array([10, 20, 30, 40, 50])
    model.fit(X, y)
    return model


@pytest.fixture
def mock_prophet_model():
    """Create a mock Prophet model."""
    # Create simple training data
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    df = pd.DataFrame({
        'ds': dates,
        'y': np.random.randint(10, 100, size=30)
    })
    
    model = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
    model.fit(df)
    return model


@pytest.fixture
def mock_model_metadata():
    """Create mock model metadata."""
    return ModelMetadata(
        model_id='test_model_123',
        product_id='product_abc',
        model_type='custom',
        version=1,
        artifact_path='s3://bucket/path/model',
        training_dataset_id='dataset_123',
        mae=5.0,
        rmse=7.5,
        mape=10.0,
        hyperparameters={
            'algorithm': 'linear',
            'features': ['price', 'is_holiday', 'day_of_week', 'month']
        },
        created_at=datetime.now(),
        forecast_horizon=30
    )


def serialize_model(model, algorithm):
    """Helper to serialize a model."""
    buffer = BytesIO()
    model_package = {
        'model': model,
        'algorithm': algorithm,
        'serialization_version': '1.0'
    }
    joblib.dump(model_package, buffer)
    buffer.seek(0)
    return buffer.read()


class TestForecastingEngine:
    """Test suite for ForecastingEngine."""
    
    def test_generate_forecast_with_sklearn_model(
        self,
        forecasting_engine,
        mock_sklearn_model,
        mock_model_metadata
    ):
        """Test forecast generation with scikit-learn model."""
        # Serialize model
        model_artifact = serialize_model(mock_sklearn_model, 'linear')
        
        # Mock model registry
        with patch('src.inference.forecasting_engine.model_registry') as mock_registry:
            mock_registry.get_model.return_value = (model_artifact, mock_model_metadata)
            
            # Generate forecast
            result = forecasting_engine.generate_forecast(
                model_id='test_model_123',
                forecast_horizon=7,
                start_date=datetime(2024, 1, 1)
            )
            
            # Verify result structure
            assert isinstance(result, ForecastResult)
            assert result.model_id == 'test_model_123'
            assert result.product_id == 'product_abc'
            assert len(result.timestamps) == 7
            assert len(result.predictions) == 7
            
            # Verify confidence intervals
            assert '50%' in result.confidence_intervals
            assert '80%' in result.confidence_intervals
            assert '90%' in result.confidence_intervals
            
            for level, ci in result.confidence_intervals.items():
                assert isinstance(ci, ConfidenceInterval)
                assert len(ci.lower) == 7
                assert len(ci.upper) == 7
                # Lower should be less than upper
                for lower, upper in zip(ci.lower, ci.upper):
                    assert lower <= upper
            
            # Verify metadata
            assert 'algorithm' in result.metadata
            assert result.metadata['algorithm'] == 'linear'
            assert result.metadata['forecast_horizon'] == 7
    
    def test_generate_forecast_with_prophet_model(
        self,
        forecasting_engine,
        mock_prophet_model,
        mock_model_metadata
    ):
        """Test forecast generation with Prophet model."""
        # Update metadata for Prophet
        prophet_metadata = ModelMetadata(
            model_id='prophet_model_123',
            product_id='product_abc',
            model_type='custom',
            version=1,
            artifact_path='s3://bucket/path/model',
            training_dataset_id='dataset_123',
            mae=5.0,
            rmse=7.5,
            mape=10.0,
            hyperparameters={
                'algorithm': 'prophet',
                'features': []
            },
            created_at=datetime.now(),
            forecast_horizon=30
        )
        
        # Serialize model
        model_artifact = serialize_model(mock_prophet_model, 'prophet')
        
        # Mock model registry
        with patch('src.inference.forecasting_engine.model_registry') as mock_registry:
            mock_registry.get_model.return_value = (model_artifact, prophet_metadata)
            
            # Generate forecast
            result = forecasting_engine.generate_forecast(
                model_id='prophet_model_123',
                forecast_horizon=7,
                start_date=datetime(2024, 1, 1)
            )
            
            # Verify result structure
            assert isinstance(result, ForecastResult)
            assert len(result.predictions) == 7
            assert len(result.confidence_intervals) == 3
    
    def test_generate_forecast_with_future_features(
        self,
        forecasting_engine,
        mock_sklearn_model,
        mock_model_metadata
    ):
        """Test forecast generation with future features provided."""
        model_artifact = serialize_model(mock_sklearn_model, 'linear')
        
        future_features = {
            'prices': [100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0],
            'holidays': [False, False, True, False, False, False, True]
        }
        
        with patch('src.inference.forecasting_engine.model_registry') as mock_registry:
            mock_registry.get_model.return_value = (model_artifact, mock_model_metadata)
            
            result = forecasting_engine.generate_forecast(
                model_id='test_model_123',
                forecast_horizon=7,
                future_features=future_features,
                start_date=datetime(2024, 1, 1)
            )
            
            assert len(result.predictions) == 7
            assert len(result.timestamps) == 7
    
    def test_generate_forecast_invalid_horizon(self, forecasting_engine):
        """Test forecast generation with invalid horizon."""
        # Test horizon < 1
        with pytest.raises(ValueError, match="Invalid forecast_horizon"):
            forecasting_engine.generate_forecast(
                model_id='test_model',
                forecast_horizon=0
            )
        
        # Test horizon > 90
        with pytest.raises(ValueError, match="Invalid forecast_horizon"):
            forecasting_engine.generate_forecast(
                model_id='test_model',
                forecast_horizon=100
            )
    
    def test_generate_forecast_model_not_found(self, forecasting_engine):
        """Test forecast generation when model not found."""
        with patch('src.inference.forecasting_engine.model_registry') as mock_registry:
            mock_registry.get_model.side_effect = ValueError("Model not found: invalid_model")
            
            with pytest.raises(ValueError, match="Model not found"):
                forecasting_engine.generate_forecast(
                    model_id='invalid_model',
                    forecast_horizon=7
                )
    
    def test_confidence_intervals_ordering(
        self,
        forecasting_engine,
        mock_sklearn_model,
        mock_model_metadata
    ):
        """Test that confidence intervals are properly ordered (50% < 80% < 90%)."""
        model_artifact = serialize_model(mock_sklearn_model, 'linear')
        
        with patch('src.inference.forecasting_engine.model_registry') as mock_registry:
            mock_registry.get_model.return_value = (model_artifact, mock_model_metadata)
            
            result = forecasting_engine.generate_forecast(
                model_id='test_model_123',
                forecast_horizon=7,
                start_date=datetime(2024, 1, 1)
            )
            
            # For each prediction point, verify interval ordering
            for i in range(7):
                ci_50 = result.confidence_intervals['50%']
                ci_80 = result.confidence_intervals['80%']
                ci_90 = result.confidence_intervals['90%']
                
                # 50% interval should be narrowest
                width_50 = ci_50.upper[i] - ci_50.lower[i]
                width_80 = ci_80.upper[i] - ci_80.lower[i]
                width_90 = ci_90.upper[i] - ci_90.lower[i]
                
                assert width_50 <= width_80 <= width_90
    
    def test_confidence_intervals_non_negative(
        self,
        forecasting_engine,
        mock_sklearn_model,
        mock_model_metadata
    ):
        """Test that confidence intervals are non-negative for demand forecasting."""
        model_artifact = serialize_model(mock_sklearn_model, 'linear')
        
        with patch('src.inference.forecasting_engine.model_registry') as mock_registry:
            mock_registry.get_model.return_value = (model_artifact, mock_model_metadata)
            
            result = forecasting_engine.generate_forecast(
                model_id='test_model_123',
                forecast_horizon=7,
                start_date=datetime(2024, 1, 1)
            )
            
            # Verify all confidence interval bounds are non-negative
            for level, ci in result.confidence_intervals.items():
                for lower, upper in zip(ci.lower, ci.upper):
                    assert lower >= 0, f"Lower bound is negative: {lower}"
                    assert upper >= 0, f"Upper bound is negative: {upper}"
    
    def test_prepare_future_features_seasonality(self, forecasting_engine, mock_model_metadata):
        """Test that future features include correct seasonality."""
        start_date = datetime(2024, 3, 15)  # March 15, 2024 (Friday)
        forecast_horizon = 7
        
        future_df = forecasting_engine._prepare_future_features(
            start_date=start_date,
            forecast_horizon=forecast_horizon,
            future_features=None,
            metadata=mock_model_metadata
        )
        
        # Verify seasonality features exist
        assert 'day_of_week' in future_df.columns
        assert 'month' in future_df.columns
        assert 'quarter' in future_df.columns
        assert 'season' in future_df.columns
        
        # Verify first day (March 15, 2024 is Friday = 4)
        assert future_df.iloc[0]['day_of_week'] == 4
        assert future_df.iloc[0]['month'] == 3
        assert future_df.iloc[0]['quarter'] == 1
        assert future_df.iloc[0]['season'] == 'spring'
    
    def test_prepare_future_features_with_holidays(
        self,
        forecasting_engine,
        mock_model_metadata
    ):
        """Test future features with holiday indicators."""
        future_features = {
            'holidays': [False, False, True, False, False, False, True]
        }
        
        future_df = forecasting_engine._prepare_future_features(
            start_date=datetime(2024, 1, 1),
            forecast_horizon=7,
            future_features=future_features,
            metadata=mock_model_metadata
        )
        
        assert 'is_holiday' in future_df.columns
        assert future_df['is_holiday'].tolist() == [False, False, True, False, False, False, True]
    
    def test_prepare_future_features_with_prices(
        self,
        forecasting_engine,
        mock_model_metadata
    ):
        """Test future features with price data."""
        # Test with list of prices
        future_features = {
            'prices': [100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0]
        }
        
        future_df = forecasting_engine._prepare_future_features(
            start_date=datetime(2024, 1, 1),
            forecast_horizon=7,
            future_features=future_features,
            metadata=mock_model_metadata
        )
        
        assert 'price' in future_df.columns
        assert future_df['price'].tolist() == [100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0]
        
        # Test with single price value
        future_features = {'prices': 100.0}
        
        future_df = forecasting_engine._prepare_future_features(
            start_date=datetime(2024, 1, 1),
            forecast_horizon=7,
            future_features=future_features,
            metadata=mock_model_metadata
        )
        
        assert all(price == 100.0 for price in future_df['price'])
    
    def test_generate_multi_model_forecast(
        self,
        forecasting_engine,
        mock_sklearn_model,
        mock_model_metadata
    ):
        """Test multi-model forecast generation."""
        model_artifact = serialize_model(mock_sklearn_model, 'linear')
        
        # Create multiple model metadata
        model1 = mock_model_metadata
        model2 = ModelMetadata(
            model_id='test_model_456',
            product_id='product_abc',
            model_type='custom',
            version=2,
            artifact_path='s3://bucket/path/model2',
            training_dataset_id='dataset_123',
            mae=4.5,
            rmse=6.5,
            mape=9.0,
            hyperparameters={
                'algorithm': 'linear',
                'features': ['price', 'is_holiday']
            },
            created_at=datetime.now(),
            forecast_horizon=30
        )
        
        with patch('src.inference.forecasting_engine.model_registry') as mock_registry:
            mock_registry.list_models.return_value = [model1, model2]
            mock_registry.get_model.return_value = (model_artifact, model1)
            
            results = forecasting_engine.generate_multi_model_forecast(
                product_id='product_abc',
                forecast_horizon=7,
                start_date=datetime(2024, 1, 1)
            )
            
            # Verify results
            assert isinstance(results, dict)
            assert len(results) >= 1  # At least one model succeeded
            
            for model_id, forecast in results.items():
                assert isinstance(forecast, ForecastResult)
                assert len(forecast.predictions) == 7
    
    def test_generate_multi_model_forecast_no_models(self, forecasting_engine):
        """Test multi-model forecast when no models exist."""
        with patch('src.inference.forecasting_engine.model_registry') as mock_registry:
            mock_registry.list_models.return_value = []
            
            with pytest.raises(ValueError, match="No models found"):
                forecasting_engine.generate_multi_model_forecast(
                    product_id='nonexistent_product',
                    forecast_horizon=7
                )
    
    def test_timestamps_generation(
        self,
        forecasting_engine,
        mock_sklearn_model,
        mock_model_metadata
    ):
        """Test that timestamps are correctly generated for forecast horizon."""
        model_artifact = serialize_model(mock_sklearn_model, 'linear')
        start_date = datetime(2024, 1, 1, 12, 0, 0)
        
        with patch('src.inference.forecasting_engine.model_registry') as mock_registry:
            mock_registry.get_model.return_value = (model_artifact, mock_model_metadata)
            
            result = forecasting_engine.generate_forecast(
                model_id='test_model_123',
                forecast_horizon=5,
                start_date=start_date
            )
            
            # Verify timestamps
            expected_timestamps = [
                datetime(2024, 1, 1, 12, 0, 0),
                datetime(2024, 1, 2, 12, 0, 0),
                datetime(2024, 1, 3, 12, 0, 0),
                datetime(2024, 1, 4, 12, 0, 0),
                datetime(2024, 1, 5, 12, 0, 0)
            ]
            
            assert result.timestamps == expected_timestamps
    
    def test_metadata_in_result(
        self,
        forecasting_engine,
        mock_sklearn_model,
        mock_model_metadata
    ):
        """Test that result metadata contains expected fields."""
        model_artifact = serialize_model(mock_sklearn_model, 'linear')
        
        with patch('src.inference.forecasting_engine.model_registry') as mock_registry:
            mock_registry.get_model.return_value = (model_artifact, mock_model_metadata)
            
            result = forecasting_engine.generate_forecast(
                model_id='test_model_123',
                forecast_horizon=7,
                start_date=datetime(2024, 1, 1)
            )
            
            # Verify metadata fields
            assert 'algorithm' in result.metadata
            assert 'model_version' in result.metadata
            assert 'training_mae' in result.metadata
            assert 'training_rmse' in result.metadata
            assert 'training_mape' in result.metadata
            assert 'forecast_horizon' in result.metadata
            assert 'start_date' in result.metadata
            assert 'hyperparameters' in result.metadata
            
            # Verify values
            assert result.metadata['algorithm'] == 'linear'
            assert result.metadata['model_version'] == 1
            assert result.metadata['training_mae'] == 5.0
            assert result.metadata['training_rmse'] == 7.5
            assert result.metadata['training_mape'] == 10.0
