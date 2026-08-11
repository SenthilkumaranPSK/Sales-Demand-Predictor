"""
Unit tests for Amazon Forecast query functionality in ForecastingEngine.

Tests forecast generation from Amazon Forecast predictors, quantile extraction,
and confidence interval conversion.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from botocore.exceptions import ClientError

from src.inference.forecasting_engine import (
    ForecastingEngine,
    ForecastResult,
    ConfidenceInterval
)
from src.registry.model_registry import ModelMetadata


@pytest.fixture
def forecasting_engine():
    """Create a ForecastingEngine instance with mock AWS clients."""
    mock_forecast_client = Mock()
    mock_forecastquery_client = Mock()
    
    return ForecastingEngine(
        forecast_client=mock_forecast_client,
        forecastquery_client=mock_forecastquery_client
    )


@pytest.fixture
def mock_forecast_metadata():
    """Create mock metadata for Amazon Forecast model."""
    return ModelMetadata(
        model_id='forecast_model_123',
        product_id='product_abc',
        model_type='forecast',
        version=1,
        artifact_path='s3://bucket/path/forecast_model',
        training_dataset_id='dataset_123',
        mae=4.5,
        rmse=6.0,
        mape=8.5,
        hyperparameters={
            'predictor_arn': 'arn:aws:forecast:us-east-1:123456789012:predictor/test_predictor',
            'algorithm': 'AutoML'
        },
        created_at=datetime.now(),
        forecast_horizon=30
    )


@pytest.fixture
def mock_forecast_response():
    """Create mock response from Amazon Forecast query."""
    # Generate 7 days of predictions
    return {
        'Forecast': {
            'Predictions': {
                'p10': [
                    {'Timestamp': '2024-01-01T00:00:00', 'Value': 8.0},
                    {'Timestamp': '2024-01-02T00:00:00', 'Value': 9.0},
                    {'Timestamp': '2024-01-03T00:00:00', 'Value': 10.0},
                    {'Timestamp': '2024-01-04T00:00:00', 'Value': 11.0},
                    {'Timestamp': '2024-01-05T00:00:00', 'Value': 12.0},
                    {'Timestamp': '2024-01-06T00:00:00', 'Value': 13.0},
                    {'Timestamp': '2024-01-07T00:00:00', 'Value': 14.0}
                ],
                'p50': [
                    {'Timestamp': '2024-01-01T00:00:00', 'Value': 10.0},
                    {'Timestamp': '2024-01-02T00:00:00', 'Value': 12.0},
                    {'Timestamp': '2024-01-03T00:00:00', 'Value': 14.0},
                    {'Timestamp': '2024-01-04T00:00:00', 'Value': 16.0},
                    {'Timestamp': '2024-01-05T00:00:00', 'Value': 18.0},
                    {'Timestamp': '2024-01-06T00:00:00', 'Value': 20.0},
                    {'Timestamp': '2024-01-07T00:00:00', 'Value': 22.0}
                ],
                'p90': [
                    {'Timestamp': '2024-01-01T00:00:00', 'Value': 12.0},
                    {'Timestamp': '2024-01-02T00:00:00', 'Value': 15.0},
                    {'Timestamp': '2024-01-03T00:00:00', 'Value': 18.0},
                    {'Timestamp': '2024-01-04T00:00:00', 'Value': 21.0},
                    {'Timestamp': '2024-01-05T00:00:00', 'Value': 24.0},
                    {'Timestamp': '2024-01-06T00:00:00', 'Value': 27.0},
                    {'Timestamp': '2024-01-07T00:00:00', 'Value': 30.0}
                ]
            }
        }
    }


class TestAmazonForecastQuery:
    """Test suite for Amazon Forecast query functionality."""
    
    def test_generate_forecast_from_amazon_forecast(
        self,
        forecasting_engine,
        mock_forecast_metadata,
        mock_forecast_response
    ):
        """Test forecast generation from Amazon Forecast predictor."""
        # Mock model registry
        with patch('src.inference.forecasting_engine.model_registry') as mock_registry:
            mock_registry.get_model.return_value = (b'', mock_forecast_metadata)
            
            # Mock Forecast API calls
            forecasting_engine.forecast_client.create_forecast.return_value = {
                'ForecastArn': 'arn:aws:forecast:us-east-1:123456789012:forecast/test_forecast'
            }
            
            forecasting_engine.forecast_client.describe_forecast.return_value = {
                'Status': 'ACTIVE',
                'ForecastArn': 'arn:aws:forecast:us-east-1:123456789012:forecast/test_forecast'
            }
            
            forecasting_engine.forecastquery_client.query_forecast.return_value = mock_forecast_response
            
            # Generate forecast
            result = forecasting_engine.generate_forecast(
                model_id='forecast_model_123',
                forecast_horizon=7,
                start_date=datetime(2024, 1, 1)
            )
            
            # Verify result structure
            assert isinstance(result, ForecastResult)
            assert result.model_id == 'forecast_model_123'
            assert result.product_id == 'product_abc'
            assert len(result.timestamps) == 7
            assert len(result.predictions) == 7
            
            # Verify predictions are p50 values
            assert result.predictions == [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0]
            
            # Verify confidence intervals
            assert '50%' in result.confidence_intervals
            assert '80%' in result.confidence_intervals
            assert '90%' in result.confidence_intervals
            
            # Verify metadata
            assert result.metadata['algorithm'] == 'amazon_forecast'
            assert 'predictor_arn' in result.metadata
            assert 'forecast_arn' in result.metadata
    
    def test_convert_quantiles_to_confidence_intervals(self, forecasting_engine):
        """Test conversion of Forecast quantiles to confidence intervals."""
        p10_values = [
            {'Value': 8.0},
            {'Value': 9.0},
            {'Value': 10.0}
        ]
        
        p50_values = [
            {'Value': 10.0},
            {'Value': 12.0},
            {'Value': 14.0}
        ]
        
        p90_values = [
            {'Value': 12.0},
            {'Value': 15.0},
            {'Value': 18.0}
        ]
        
        confidence_intervals = forecasting_engine._convert_quantiles_to_confidence_intervals(
            p10_values=p10_values,
            p50_values=p50_values,
            p90_values=p90_values
        )
        
        # Verify all confidence levels present
        assert '50%' in confidence_intervals
        assert '80%' in confidence_intervals
        assert '90%' in confidence_intervals
        
        # Verify 80% CI uses p10 and p90 directly
        ci_80 = confidence_intervals['80%']
        assert ci_80.lower == [8.0, 9.0, 10.0]
        assert ci_80.upper == [12.0, 15.0, 18.0]
        
        # Verify confidence intervals are ordered (50% < 80% < 90%)
        for i in range(3):
            ci_50 = confidence_intervals['50%']
            ci_80 = confidence_intervals['80%']
            ci_90 = confidence_intervals['90%']
            
            width_50 = ci_50.upper[i] - ci_50.lower[i]
            width_80 = ci_80.upper[i] - ci_80.lower[i]
            width_90 = ci_90.upper[i] - ci_90.lower[i]
            
            assert width_50 <= width_80 <= width_90
    
    def test_convert_quantiles_without_p10_p90(self, forecasting_engine):
        """Test confidence interval conversion when p10/p90 not available."""
        p50_values = [
            {'Value': 10.0},
            {'Value': 12.0},
            {'Value': 14.0}
        ]
        
        confidence_intervals = forecasting_engine._convert_quantiles_to_confidence_intervals(
            p10_values=[],
            p50_values=p50_values,
            p90_values=[]
        )
        
        # Verify fallback intervals created
        assert '50%' in confidence_intervals
        assert '80%' in confidence_intervals
        assert '90%' in confidence_intervals
        
        # Verify intervals are symmetric around p50
        ci_50 = confidence_intervals['50%']
        assert len(ci_50.lower) == 3
        assert len(ci_50.upper) == 3
        
        # Verify all bounds are non-negative
        for level, ci in confidence_intervals.items():
            for lower, upper in zip(ci.lower, ci.upper):
                assert lower >= 0
                assert upper >= 0
                assert lower <= upper
    
    def test_extract_forecast_results(self, forecasting_engine, mock_forecast_response):
        """Test extraction of predictions and confidence intervals from Forecast response."""
        start_date = datetime(2024, 1, 1)
        forecast_horizon = 7
        
        timestamps, predictions, confidence_intervals = forecasting_engine._extract_forecast_results(
            forecast_data=mock_forecast_response['Forecast'],
            start_date=start_date,
            forecast_horizon=forecast_horizon
        )
        
        # Verify timestamps
        assert len(timestamps) == 7
        assert timestamps[0] == start_date
        assert timestamps[-1] == start_date + timedelta(days=6)
        
        # Verify predictions (p50 values)
        assert len(predictions) == 7
        assert predictions == [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0]
        
        # Verify confidence intervals
        assert len(confidence_intervals) == 3
        assert '50%' in confidence_intervals
        assert '80%' in confidence_intervals
        assert '90%' in confidence_intervals
    
    def test_extract_forecast_results_missing_p50(self, forecasting_engine):
        """Test error handling when p50 predictions missing."""
        forecast_data = {
            'Predictions': {
                'p10': [{'Value': 8.0}],
                'p90': [{'Value': 12.0}]
            }
        }
        
        with pytest.raises(RuntimeError, match="No p50 predictions found"):
            forecasting_engine._extract_forecast_results(
                forecast_data=forecast_data,
                start_date=datetime(2024, 1, 1),
                forecast_horizon=7
            )
    
    def test_create_forecast(self, forecasting_engine):
        """Test forecast creation from predictor."""
        predictor_arn = 'arn:aws:forecast:us-east-1:123456789012:predictor/test_predictor'
        product_id = 'product_abc'
        
        forecasting_engine.forecast_client.create_forecast.return_value = {
            'ForecastArn': 'arn:aws:forecast:us-east-1:123456789012:forecast/test_forecast'
        }
        
        forecast_arn = forecasting_engine._create_forecast(
            predictor_arn=predictor_arn,
            product_id=product_id
        )
        
        assert forecast_arn == 'arn:aws:forecast:us-east-1:123456789012:forecast/test_forecast'
        
        # Verify API call
        forecasting_engine.forecast_client.create_forecast.assert_called_once()
        call_args = forecasting_engine.forecast_client.create_forecast.call_args
        assert 'ForecastName' in call_args[1]
        assert 'PredictorArn' in call_args[1]
        assert call_args[1]['PredictorArn'] == predictor_arn
    
    def test_create_forecast_already_exists(self, forecasting_engine):
        """Test forecast creation when forecast already exists."""
        predictor_arn = 'arn:aws:forecast:us-east-1:123456789012:predictor/test_predictor'
        product_id = 'product_abc'
        
        # Mock ResourceAlreadyExistsException
        error_response = {'Error': {'Code': 'ResourceAlreadyExistsException'}}
        forecasting_engine.forecast_client.create_forecast.side_effect = ClientError(
            error_response,
            'CreateForecast'
        )
        
        forecasting_engine.forecast_client.describe_forecast.return_value = {
            'ForecastArn': 'arn:aws:forecast:us-east-1:123456789012:forecast/existing_forecast',
            'Status': 'ACTIVE'
        }
        
        forecast_arn = forecasting_engine._create_forecast(
            predictor_arn=predictor_arn,
            product_id=product_id
        )
        
        assert 'forecast' in forecast_arn
    
    def test_wait_for_forecast_completion_success(self, forecasting_engine):
        """Test waiting for forecast completion - success case."""
        forecast_arn = 'arn:aws:forecast:us-east-1:123456789012:forecast/test_forecast'
        
        # Mock describe_forecast to return ACTIVE immediately
        forecasting_engine.forecast_client.describe_forecast.return_value = {
            'Status': 'ACTIVE',
            'ForecastArn': forecast_arn
        }
        
        # Should complete without error
        forecasting_engine._wait_for_forecast_completion(forecast_arn)
        
        # Verify API was called
        forecasting_engine.forecast_client.describe_forecast.assert_called()
    
    def test_wait_for_forecast_completion_failure(self, forecasting_engine):
        """Test waiting for forecast completion - failure case."""
        forecast_arn = 'arn:aws:forecast:us-east-1:123456789012:forecast/test_forecast'
        
        # Mock describe_forecast to return CREATE_FAILED
        forecasting_engine.forecast_client.describe_forecast.return_value = {
            'Status': 'CREATE_FAILED',
            'Message': 'Forecast creation failed due to invalid data',
            'ForecastArn': forecast_arn
        }
        
        with pytest.raises(RuntimeError, match="Forecast creation failed"):
            forecasting_engine._wait_for_forecast_completion(forecast_arn)
    
    def test_query_forecast(self, forecasting_engine, mock_forecast_response):
        """Test querying forecast for predictions."""
        forecast_arn = 'arn:aws:forecast:us-east-1:123456789012:forecast/test_forecast'
        product_id = 'product_abc'
        start_date = datetime(2024, 1, 1)
        forecast_horizon = 7
        
        forecasting_engine.forecastquery_client.query_forecast.return_value = mock_forecast_response
        
        forecast_data = forecasting_engine._query_forecast(
            forecast_arn=forecast_arn,
            product_id=product_id,
            start_date=start_date,
            forecast_horizon=forecast_horizon
        )
        
        # Verify result
        assert 'Predictions' in forecast_data
        assert 'p50' in forecast_data['Predictions']
        
        # Verify API call
        forecasting_engine.forecastquery_client.query_forecast.assert_called_once()
        call_args = forecasting_engine.forecastquery_client.query_forecast.call_args
        assert call_args[1]['ForecastArn'] == forecast_arn
        assert call_args[1]['Filters']['item_id'] == product_id
    
    def test_query_forecast_not_found(self, forecasting_engine):
        """Test query forecast when forecast not found."""
        forecast_arn = 'arn:aws:forecast:us-east-1:123456789012:forecast/nonexistent'
        
        # Mock ResourceNotFoundException
        error_response = {'Error': {'Code': 'ResourceNotFoundException'}}
        forecasting_engine.forecastquery_client.query_forecast.side_effect = ClientError(
            error_response,
            'QueryForecast'
        )
        
        with pytest.raises(RuntimeError, match="Forecast not found"):
            forecasting_engine._query_forecast(
                forecast_arn=forecast_arn,
                product_id='product_abc',
                start_date=datetime(2024, 1, 1),
                forecast_horizon=7
            )
    
    def test_query_forecast_invalid_input(self, forecasting_engine):
        """Test query forecast with invalid input."""
        forecast_arn = 'arn:aws:forecast:us-east-1:123456789012:forecast/test_forecast'
        
        # Mock InvalidInputException
        error_response = {'Error': {'Code': 'InvalidInputException'}}
        forecasting_engine.forecastquery_client.query_forecast.side_effect = ClientError(
            error_response,
            'QueryForecast'
        )
        
        with pytest.raises(ValueError, match="Invalid query parameters"):
            forecasting_engine._query_forecast(
                forecast_arn=forecast_arn,
                product_id='product_abc',
                start_date=datetime(2024, 1, 1),
                forecast_horizon=7
            )
    
    def test_forecast_model_missing_predictor_arn(
        self,
        forecasting_engine,
        mock_forecast_metadata
    ):
        """Test error when predictor ARN missing from metadata."""
        # Remove predictor_arn from metadata
        bad_metadata = ModelMetadata(
            model_id='forecast_model_123',
            product_id='product_abc',
            model_type='forecast',
            version=1,
            artifact_path='s3://bucket/path/forecast_model',
            training_dataset_id='dataset_123',
            mae=4.5,
            rmse=6.0,
            mape=8.5,
            hyperparameters={},  # No predictor_arn
            created_at=datetime.now(),
            forecast_horizon=30
        )
        
        with patch('src.inference.forecasting_engine.model_registry') as mock_registry:
            mock_registry.get_model.return_value = (b'', bad_metadata)
            
            with pytest.raises(RuntimeError, match="Predictor ARN not found"):
                forecasting_engine.generate_forecast(
                    model_id='forecast_model_123',
                    forecast_horizon=7
                )
    
    def test_confidence_intervals_non_negative_forecast(
        self,
        forecasting_engine,
        mock_forecast_metadata,
        mock_forecast_response
    ):
        """Test that Forecast confidence intervals are non-negative."""
        with patch('src.inference.forecasting_engine.model_registry') as mock_registry:
            mock_registry.get_model.return_value = (b'', mock_forecast_metadata)
            
            forecasting_engine.forecast_client.create_forecast.return_value = {
                'ForecastArn': 'arn:aws:forecast:us-east-1:123456789012:forecast/test_forecast'
            }
            
            forecasting_engine.forecast_client.describe_forecast.return_value = {
                'Status': 'ACTIVE'
            }
            
            forecasting_engine.forecastquery_client.query_forecast.return_value = mock_forecast_response
            
            result = forecasting_engine.generate_forecast(
                model_id='forecast_model_123',
                forecast_horizon=7,
                start_date=datetime(2024, 1, 1)
            )
            
            # Verify all confidence interval bounds are non-negative
            for level, ci in result.confidence_intervals.items():
                for lower, upper in zip(ci.lower, ci.upper):
                    assert lower >= 0, f"Lower bound is negative: {lower}"
                    assert upper >= 0, f"Upper bound is negative: {upper}"
                    assert lower <= upper, f"Lower > upper: {lower} > {upper}"

