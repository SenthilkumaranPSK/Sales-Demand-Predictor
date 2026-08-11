"""
Unit tests for the Training Pipeline orchestration module.

Tests cover:
- Successful pipeline execution
- Error handling at each stage
- Retry logic with exponential backoff
- Model registration
- Version management
- Forecast predictor training
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch, call
from datetime import datetime
import numpy as np
import pandas as pd

from src.training.pipeline import (
    TrainingPipeline, 
    PipelineConfig, 
    PipelineResult,
    ForecastPredictorConfig,
    ForecastPredictorResult
)
from src.training.custom_model_trainer import ModelConfig, TrainingResult
from src.training.data_preparation import TrainingDataset, DataPreparationResult
from src.training.forecast_integration import ForecastImportResult
from src.training.metrics import PerformanceMetrics
from src.registry.model_registry import ModelMetadata


@pytest.fixture
def mock_data_preparation():
    """Mock TrainingDataPreparation instance."""
    mock = Mock()
    
    # Create mock dataset
    train_data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=100),
        'product_id': ['prod_1'] * 100,
        'sales_volume': np.random.randint(50, 200, 100),
        'price': np.random.uniform(10, 50, 100),
        'is_holiday': [False] * 100,
        'day_of_week': [i % 7 for i in range(100)],
        'month': [1] * 100,
        'quarter': [1] * 100
    })
    
    validation_data = pd.DataFrame({
        'timestamp': pd.date_range('2023-04-11', periods=25),
        'product_id': ['prod_1'] * 25,
        'sales_volume': np.random.randint(50, 200, 25),
        'price': np.random.uniform(10, 50, 25),
        'is_holiday': [False] * 25,
        'day_of_week': [i % 7 for i in range(25)],
        'month': [4] * 25,
        'quarter': [2] * 25
    })
    
    dataset = TrainingDataset(
        train_data=train_data,
        validation_data=validation_data,
        feature_columns=['price', 'is_holiday', 'day_of_week', 'month', 'quarter'],
        target_column='sales_volume',
        normalization_params={},
        metadata={'total_records': 125}
    )
    
    mock.prepare_training_data.return_value = DataPreparationResult(
        success=True,
        dataset=dataset,
        warnings=[]
    )
    
    return mock


@pytest.fixture
def mock_model_trainer():
    """Mock CustomModelTrainer instance."""
    mock = Mock()
    
    # Create mock training result
    metrics = PerformanceMetrics(
        mae=10.5,
        rmse=15.2,
        mape=5.3,
        sample_size=25
    )
    
    mock.train_model.return_value = TrainingResult(
        success=True,
        model_artifact=b'mock_model_artifact',
        metrics=metrics,
        model_type='random_forest',
        warnings=[]
    )
    
    return mock


@pytest.fixture
def mock_model_registry():
    """Mock ModelRegistry instance."""
    mock = Mock()
    
    # Mock register_model to return a model ID
    mock.register_model.return_value = 'prod_1_random_forest_20240115_abc123'
    
    # Mock list_models to return empty list (no existing models)
    mock.list_models.return_value = []
    
    return mock


@pytest.fixture
def pipeline_config():
    """Create a sample pipeline configuration."""
    return PipelineConfig(
        dataset_path='s3://bucket/historical-data/prod_1',
        product_id='prod_1',
        model_config=ModelConfig(
            algorithm='random_forest',
            hyperparameters={'n_estimators': 100},
            forecast_horizon=30
        ),
        train_split=0.8,
        target_column='sales_volume',
        training_dataset_id='dataset_v1',
        max_retries=3,
        retry_base_delay=0.1  # Short delay for testing
    )


class TestTrainingPipeline:
    """Test suite for TrainingPipeline class."""
    
    def test_successful_pipeline_execution(
        self,
        mock_data_preparation,
        mock_model_trainer,
        mock_model_registry,
        pipeline_config
    ):
        """Test successful execution of complete training pipeline."""
        # Create pipeline with mocked dependencies
        pipeline = TrainingPipeline(
            data_preparation=mock_data_preparation,
            model_trainer=mock_model_trainer,
            model_registry=mock_model_registry
        )
        
        # Execute pipeline
        result = pipeline.train_custom_model(pipeline_config)
        
        # Verify success
        assert result.success is True
        assert result.model_id is not None
        assert result.metadata is not None
        assert len(result.errors) == 0
        
        # Verify all stages were called
        mock_data_preparation.prepare_training_data.assert_called_once()
        mock_model_trainer.train_model.assert_called_once()
        mock_model_registry.register_model.assert_called_once()
        
        # Verify stage timings are recorded
        assert 'data_preparation' in result.stage_timings
        assert 'model_training' in result.stage_timings
        assert 'model_registration' in result.stage_timings
        
        # Verify execution time is positive
        assert result.execution_time > 0
    
    def test_data_preparation_failure(
        self,
        mock_data_preparation,
        mock_model_trainer,
        mock_model_registry,
        pipeline_config
    ):
        """Test pipeline handles data preparation failure."""
        # Configure mock to return failure
        mock_data_preparation.prepare_training_data.return_value = DataPreparationResult(
            success=False,
            errors=['Failed to load dataset from S3']
        )
        
        pipeline = TrainingPipeline(
            data_preparation=mock_data_preparation,
            model_trainer=mock_model_trainer,
            model_registry=mock_model_registry
        )
        
        result = pipeline.train_custom_model(pipeline_config)
        
        # Verify failure
        assert result.success is False
        assert len(result.errors) > 0
        assert 'Failed to load dataset from S3' in result.errors
        
        # Verify subsequent stages were not called
        mock_model_trainer.train_model.assert_not_called()
        mock_model_registry.register_model.assert_not_called()
    
    def test_model_training_failure(
        self,
        mock_data_preparation,
        mock_model_trainer,
        mock_model_registry,
        pipeline_config
    ):
        """Test pipeline handles model training failure."""
        # Configure mock to return failure
        mock_model_trainer.train_model.return_value = TrainingResult(
            success=False,
            errors=['Model training failed: insufficient data']
        )
        
        pipeline = TrainingPipeline(
            data_preparation=mock_data_preparation,
            model_trainer=mock_model_trainer,
            model_registry=mock_model_registry
        )
        
        result = pipeline.train_custom_model(pipeline_config)
        
        # Verify failure
        assert result.success is False
        assert len(result.errors) > 0
        assert 'Model training failed: insufficient data' in result.errors
        
        # Verify data preparation was called but registration was not
        mock_data_preparation.prepare_training_data.assert_called_once()
        mock_model_registry.register_model.assert_not_called()
    
    def test_model_registration_failure(
        self,
        mock_data_preparation,
        mock_model_trainer,
        mock_model_registry,
        pipeline_config
    ):
        """Test pipeline handles model registration failure."""
        # Configure mock to raise exception
        mock_model_registry.register_model.side_effect = RuntimeError(
            'Database connection failed'
        )
        
        pipeline = TrainingPipeline(
            data_preparation=mock_data_preparation,
            model_trainer=mock_model_trainer,
            model_registry=mock_model_registry
        )
        
        result = pipeline.train_custom_model(pipeline_config)
        
        # Verify failure
        assert result.success is False
        assert len(result.errors) > 0
        
        # Verify all stages were attempted
        mock_data_preparation.prepare_training_data.assert_called_once()
        mock_model_trainer.train_model.assert_called_once()
        mock_model_registry.register_model.assert_called()
    
    def test_retry_logic_with_transient_error(
        self,
        mock_data_preparation,
        mock_model_trainer,
        mock_model_registry,
        pipeline_config
    ):
        """Test retry logic with transient S3 error."""
        # Configure mock to fail twice then succeed
        call_count = 0
        
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception('RequestTimeout: S3 request timed out')
            return DataPreparationResult(
                success=True,
                dataset=mock_data_preparation.prepare_training_data.return_value.dataset
            )
        
        mock_data_preparation.prepare_training_data.side_effect = side_effect
        
        pipeline = TrainingPipeline(
            data_preparation=mock_data_preparation,
            model_trainer=mock_model_trainer,
            model_registry=mock_model_registry
        )
        
        result = pipeline.train_custom_model(pipeline_config)
        
        # Verify success after retries
        assert result.success is True
        assert call_count == 3  # Failed twice, succeeded on third attempt
    
    def test_retry_logic_exhausted(
        self,
        mock_data_preparation,
        mock_model_trainer,
        mock_model_registry,
        pipeline_config
    ):
        """Test retry logic when all attempts fail."""
        # Configure mock to always fail with transient error
        mock_data_preparation.prepare_training_data.side_effect = Exception(
            'SlowDown: S3 throttling'
        )
        
        pipeline = TrainingPipeline(
            data_preparation=mock_data_preparation,
            model_trainer=mock_model_trainer,
            model_registry=mock_model_registry
        )
        
        result = pipeline.train_custom_model(pipeline_config)
        
        # Verify failure after all retries
        assert result.success is False
        assert len(result.errors) > 0
        
        # Verify retry attempts (1 initial + 3 retries = 4 total)
        assert mock_data_preparation.prepare_training_data.call_count == 4
    
    def test_non_transient_error_no_retry(
        self,
        mock_data_preparation,
        mock_model_trainer,
        mock_model_registry,
        pipeline_config
    ):
        """Test that non-transient errors are not retried."""
        # Configure mock to fail with non-transient error
        mock_data_preparation.prepare_training_data.side_effect = ValueError(
            'Invalid dataset path'
        )
        
        pipeline = TrainingPipeline(
            data_preparation=mock_data_preparation,
            model_trainer=mock_model_trainer,
            model_registry=mock_model_registry
        )
        
        result = pipeline.train_custom_model(pipeline_config)
        
        # Verify failure without retries
        assert result.success is False
        
        # Verify only one attempt (no retries for non-transient errors)
        assert mock_data_preparation.prepare_training_data.call_count == 1
    
    def test_exponential_backoff_timing(
        self,
        mock_data_preparation,
        mock_model_trainer,
        mock_model_registry,
        pipeline_config
    ):
        """Test that retry delays follow exponential backoff."""
        call_times = []
        
        def side_effect(*args, **kwargs):
            call_times.append(time.time())
            if len(call_times) <= 2:
                raise Exception('Timeout: transient error')
            return DataPreparationResult(
                success=True,
                dataset=mock_data_preparation.prepare_training_data.return_value.dataset
            )
        
        mock_data_preparation.prepare_training_data.side_effect = side_effect
        
        pipeline = TrainingPipeline(
            data_preparation=mock_data_preparation,
            model_trainer=mock_model_trainer,
            model_registry=mock_model_registry
        )
        
        result = pipeline.train_custom_model(pipeline_config)
        
        # Verify success
        assert result.success is True
        
        # Verify exponential backoff delays
        # First retry: base_delay * 2^0 = 0.1s
        # Second retry: base_delay * 2^1 = 0.2s
        if len(call_times) >= 3:
            delay1 = call_times[1] - call_times[0]
            delay2 = call_times[2] - call_times[1]
            
            # Allow some tolerance for execution time
            assert delay1 >= 0.1 and delay1 < 0.3
            assert delay2 >= 0.2 and delay2 < 0.4
            assert delay2 > delay1  # Second delay should be longer
    
    def test_version_management(
        self,
        mock_data_preparation,
        mock_model_trainer,
        mock_model_registry,
        pipeline_config
    ):
        """Test that version numbers are incremented correctly."""
        # Configure mock to return existing models
        existing_metadata = ModelMetadata(
            model_id='prod_1_random_forest_old',
            product_id='prod_1',
            model_type='custom',
            version=2,
            artifact_path='s3://bucket/models/old',
            training_dataset_id='dataset_v0',
            mae=12.0,
            rmse=18.0,
            mape=6.0,
            hyperparameters={},
            created_at=datetime(2024, 1, 1),
            forecast_horizon=30
        )
        
        mock_model_registry.list_models.return_value = [existing_metadata]
        
        pipeline = TrainingPipeline(
            data_preparation=mock_data_preparation,
            model_trainer=mock_model_trainer,
            model_registry=mock_model_registry
        )
        
        result = pipeline.train_custom_model(pipeline_config)
        
        # Verify success
        assert result.success is True
        
        # Verify version was incremented
        call_args = mock_model_registry.register_model.call_args
        registered_metadata = call_args[1]['metadata']
        assert registered_metadata.version == 3  # Incremented from 2
    
    def test_warnings_propagation(
        self,
        mock_data_preparation,
        mock_model_trainer,
        mock_model_registry,
        pipeline_config
    ):
        """Test that warnings from stages are propagated to final result."""
        # Configure mocks to return warnings
        mock_data_preparation.prepare_training_data.return_value = DataPreparationResult(
            success=True,
            dataset=mock_data_preparation.prepare_training_data.return_value.dataset,
            warnings=['Missing some seasonality features']
        )
        
        mock_model_trainer.train_model.return_value = TrainingResult(
            success=True,
            model_artifact=b'mock_artifact',
            metrics=PerformanceMetrics(mae=10.0, rmse=15.0, mape=5.0, sample_size=25),
            model_type='random_forest',
            warnings=['Some features had low importance']
        )
        
        pipeline = TrainingPipeline(
            data_preparation=mock_data_preparation,
            model_trainer=mock_model_trainer,
            model_registry=mock_model_registry
        )
        
        result = pipeline.train_custom_model(pipeline_config)
        
        # Verify warnings are collected
        assert result.success is True
        assert len(result.warnings) == 2
        assert 'Missing some seasonality features' in result.warnings
        assert 'Some features had low importance' in result.warnings
    
    def test_model_id_generation(
        self,
        mock_data_preparation,
        mock_model_trainer,
        mock_model_registry,
        pipeline_config
    ):
        """Test that model IDs are generated correctly."""
        pipeline = TrainingPipeline(
            data_preparation=mock_data_preparation,
            model_trainer=mock_model_trainer,
            model_registry=mock_model_registry
        )
        
        result = pipeline.train_custom_model(pipeline_config)
        
        # Verify model ID format
        assert result.success is True
        assert result.model_id is not None
        
        # Model ID should contain product_id and algorithm
        # Format: {product_id}_{algorithm}_{timestamp}_{uuid}
        # But the registry returns its own ID, so just verify it's not empty
        assert len(result.model_id) > 0


class TestIsTransientError:
    """Test suite for transient error detection."""
    
    def test_s3_throttling_error(self):
        """Test S3 throttling errors are detected as transient."""
        pipeline = TrainingPipeline()
        
        error = Exception('SlowDown: Please reduce your request rate')
        assert pipeline._is_transient_error(error) is True
    
    def test_timeout_error(self):
        """Test timeout errors are detected as transient."""
        pipeline = TrainingPipeline()
        
        error = Exception('RequestTimeout: Your socket connection timed out')
        assert pipeline._is_transient_error(error) is True
    
    def test_connection_error(self):
        """Test connection errors are detected as transient."""
        pipeline = TrainingPipeline()
        
        error = Exception('Connection error: Unable to connect to database')
        assert pipeline._is_transient_error(error) is True
    
    def test_service_unavailable_error(self):
        """Test service unavailable errors are detected as transient."""
        pipeline = TrainingPipeline()
        
        error = Exception('503 Service Unavailable')
        assert pipeline._is_transient_error(error) is True
    
    def test_validation_error_not_transient(self):
        """Test validation errors are not detected as transient."""
        pipeline = TrainingPipeline()
        
        error = ValueError('Invalid dataset path')
        assert pipeline._is_transient_error(error) is False
    
    def test_generic_error_not_transient(self):
        """Test generic errors are not detected as transient."""
        pipeline = TrainingPipeline()
        
        error = Exception('Something went wrong')
        assert pipeline._is_transient_error(error) is False


@pytest.fixture
def sample_historical_data():
    """Create sample historical data for Forecast training."""
    return pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=100),
        'product_id': ['prod_1'] * 100,
        'sales_volume': np.random.randint(50, 200, 100),
        'price': np.random.uniform(10, 50, 100),
        'is_holiday': [False] * 100,
        'day_of_week': [i % 7 for i in range(100)],
        'month': [(i // 30) + 1 for i in range(100)],
        'quarter': [1] * 100
    })


@pytest.fixture
def mock_forecast_integration():
    """Mock AmazonForecastIntegration instance."""
    mock = Mock()
    
    # Mock import_dataset to return success
    mock.import_dataset.return_value = ForecastImportResult(
        success=True,
        dataset_group_arn='arn:aws:forecast:us-east-1:123456789012:dataset-group/test-group',
        dataset_arn='arn:aws:forecast:us-east-1:123456789012:dataset/test-dataset',
        import_job_arn='arn:aws:forecast:us-east-1:123456789012:dataset-import-job/test-job',
        s3_path='s3://bucket/forecast_datasets/prod_1/test.csv',
        record_count=100
    )
    
    # Mock wait_for_import_completion to return success
    mock.wait_for_import_completion.return_value = True
    
    return mock


@pytest.fixture
def mock_forecast_client():
    """Mock boto3 Forecast client."""
    mock = Mock()
    
    # Mock describe_predictor with a callable that tracks state
    describe_call_count = {'count': 0}
    
    def describe_predictor_side_effect(*args, **kwargs):
        describe_call_count['count'] += 1
        call_num = describe_call_count['count']
        
        if call_num == 1:
            # First call: predictor doesn't exist (for creation check)
            raise Exception('ResourceNotFoundException')
        elif call_num <= 3:
            # Calls 2-3: predictor is being created
            return {'Status': 'CREATE_IN_PROGRESS'}
        else:
            # Call 4+: predictor is active with metrics
            return {
                'Status': 'ACTIVE',
                'AccuracyMetrics': {
                    'RMSE': 12.5,
                    'WeightedQuantileLosses': [0.1, 0.2, 0.3]
                }
            }
    
    mock.describe_predictor.side_effect = describe_predictor_side_effect
    
    # Mock create_predictor
    mock.create_predictor.return_value = {
        'PredictorArn': 'arn:aws:forecast:us-east-1:123456789012:predictor/test-predictor'
    }
    
    return mock


@pytest.fixture
def forecast_predictor_config():
    """Create a sample forecast predictor configuration."""
    return ForecastPredictorConfig(
        dataset_path='s3://bucket/historical-data/prod_1.csv',
        product_id='prod_1',
        forecast_horizon=30,
        dataset_name='test-dataset',
        dataset_group_name='test-group',
        predictor_name='test-predictor',
        algorithm='auto',
        dataset_frequency='D',
        timestamp_format='yyyy-MM-dd HH:mm:ss',
        training_dataset_id='dataset_v1',
        max_retries=3,
        retry_base_delay=0.1,
        max_wait_seconds=300,
        poll_interval=1
    )


class TestForecastPredictorTraining:
    """Test suite for Forecast predictor training."""
    
    def test_successful_predictor_training(
        self,
        mock_forecast_integration,
        mock_forecast_client,
        forecast_predictor_config,
        sample_historical_data
    ):
        """Test successful execution of Forecast predictor training."""
        # Create pipeline with mocked dependencies
        with patch('src.training.pipeline.boto3.client') as mock_boto3:
            mock_boto3.return_value = mock_forecast_client
            
            pipeline = TrainingPipeline(
                forecast_integration=mock_forecast_integration,
                forecast_client=mock_forecast_client
            )
            
            # Mock _load_data_from_s3
            with patch.object(pipeline, '_load_data_from_s3', return_value=sample_historical_data):
                # Execute predictor training
                result = pipeline.train_forecast_predictor(forecast_predictor_config)
        
        # Verify success
        assert result.success is True
        assert result.predictor_arn is not None
        assert result.dataset_import_result is not None
        assert result.metrics is not None
        assert len(result.errors) == 0
        
        # Verify all stages were called
        mock_forecast_integration.import_dataset.assert_called_once()
        mock_forecast_integration.wait_for_import_completion.assert_called_once()
        mock_forecast_client.create_predictor.assert_called_once()
        
        # Verify stage timings are recorded
        assert 'load_data' in result.stage_timings
        assert 'dataset_import' in result.stage_timings
        assert 'wait_import' in result.stage_timings
        assert 'create_predictor' in result.stage_timings
        assert 'wait_predictor' in result.stage_timings
        assert 'get_metrics' in result.stage_timings
        
        # Verify execution time is positive
        assert result.execution_time > 0
    
    def test_dataset_import_failure(
        self,
        mock_forecast_integration,
        mock_forecast_client,
        forecast_predictor_config,
        sample_historical_data
    ):
        """Test predictor training handles dataset import failure."""
        # Configure mock to return failure
        mock_forecast_integration.import_dataset.return_value = ForecastImportResult(
            success=False,
            errors=['Failed to upload data to S3']
        )
        
        with patch('src.training.pipeline.boto3.client') as mock_boto3:
            mock_boto3.return_value = mock_forecast_client
            
            pipeline = TrainingPipeline(
                forecast_integration=mock_forecast_integration,
                forecast_client=mock_forecast_client
            )
            
            with patch.object(pipeline, '_load_data_from_s3', return_value=sample_historical_data):
                result = pipeline.train_forecast_predictor(forecast_predictor_config)
        
        # Verify failure
        assert result.success is False
        assert len(result.errors) > 0
        assert 'Failed to upload data to S3' in result.errors
        
        # Verify subsequent stages were not called
        mock_forecast_integration.wait_for_import_completion.assert_not_called()
        mock_forecast_client.create_predictor.assert_not_called()
    
    def test_import_job_timeout(
        self,
        mock_forecast_integration,
        mock_forecast_client,
        forecast_predictor_config,
        sample_historical_data
    ):
        """Test predictor training handles import job timeout."""
        # Configure mock to return timeout
        mock_forecast_integration.wait_for_import_completion.return_value = False
        
        with patch('src.training.pipeline.boto3.client') as mock_boto3:
            mock_boto3.return_value = mock_forecast_client
            
            pipeline = TrainingPipeline(
                forecast_integration=mock_forecast_integration,
                forecast_client=mock_forecast_client
            )
            
            with patch.object(pipeline, '_load_data_from_s3', return_value=sample_historical_data):
                result = pipeline.train_forecast_predictor(forecast_predictor_config)
        
        # Verify failure
        assert result.success is False
        assert len(result.errors) > 0
        assert 'Dataset import job failed or timed out' in result.errors
        
        # Verify predictor creation was not called
        mock_forecast_client.create_predictor.assert_not_called()
    
    def test_predictor_training_failure(
        self,
        mock_forecast_integration,
        mock_forecast_client,
        forecast_predictor_config,
        sample_historical_data
    ):
        """Test predictor training handles training failure."""
        # Configure mock to return failure status
        mock_forecast_client.describe_predictor.side_effect = [
            Exception('ResourceNotFoundException'),
            {'Status': 'CREATE_FAILED', 'Message': 'Insufficient data for training'}
        ]
        
        with patch('src.training.pipeline.boto3.client') as mock_boto3:
            mock_boto3.return_value = mock_forecast_client
            
            pipeline = TrainingPipeline(
                forecast_integration=mock_forecast_integration,
                forecast_client=mock_forecast_client
            )
            
            with patch.object(pipeline, '_load_data_from_s3', return_value=sample_historical_data):
                result = pipeline.train_forecast_predictor(forecast_predictor_config)
        
        # Verify failure
        assert result.success is False
        assert len(result.errors) > 0
        assert 'Predictor training failed or timed out' in result.errors
    
    def test_automl_configuration(
        self,
        mock_forecast_integration,
        mock_forecast_client,
        forecast_predictor_config,
        sample_historical_data
    ):
        """Test predictor is configured with AutoML when algorithm is 'auto'."""
        with patch('src.training.pipeline.boto3.client') as mock_boto3:
            mock_boto3.return_value = mock_forecast_client
            
            pipeline = TrainingPipeline(
                forecast_integration=mock_forecast_integration,
                forecast_client=mock_forecast_client
            )
            
            with patch.object(pipeline, '_load_data_from_s3', return_value=sample_historical_data):
                result = pipeline.train_forecast_predictor(forecast_predictor_config)
        
        # Verify create_predictor was called with PerformAutoML=True
        call_args = mock_forecast_client.create_predictor.call_args
        assert call_args[1]['PerformAutoML'] is True
        assert 'AlgorithmArn' not in call_args[1]
    
    def test_specific_algorithm_configuration(
        self,
        mock_forecast_integration,
        mock_forecast_client,
        forecast_predictor_config,
        sample_historical_data
    ):
        """Test predictor is configured with specific algorithm when provided."""
        # Change algorithm to specific ARN
        forecast_predictor_config.algorithm = 'arn:aws:forecast:::algorithm/ARIMA'
        
        with patch('src.training.pipeline.boto3.client') as mock_boto3:
            mock_boto3.return_value = mock_forecast_client
            
            pipeline = TrainingPipeline(
                forecast_integration=mock_forecast_integration,
                forecast_client=mock_forecast_client
            )
            
            with patch.object(pipeline, '_load_data_from_s3', return_value=sample_historical_data):
                result = pipeline.train_forecast_predictor(forecast_predictor_config)
        
        # Verify create_predictor was called with AlgorithmArn
        call_args = mock_forecast_client.create_predictor.call_args
        assert call_args[1]['AlgorithmArn'] == 'arn:aws:forecast:::algorithm/ARIMA'
        assert 'PerformAutoML' not in call_args[1]
    
    def test_forecast_horizon_parameter(
        self,
        mock_forecast_integration,
        mock_forecast_client,
        forecast_predictor_config,
        sample_historical_data
    ):
        """Test predictor is configured with correct forecast horizon."""
        with patch('src.training.pipeline.boto3.client') as mock_boto3:
            mock_boto3.return_value = mock_forecast_client
            
            pipeline = TrainingPipeline(
                forecast_integration=mock_forecast_integration,
                forecast_client=mock_forecast_client
            )
            
            with patch.object(pipeline, '_load_data_from_s3', return_value=sample_historical_data):
                result = pipeline.train_forecast_predictor(forecast_predictor_config)
        
        # Verify create_predictor was called with correct forecast horizon
        call_args = mock_forecast_client.create_predictor.call_args
        assert call_args[1]['ForecastHorizon'] == 30
    
    def test_exponential_backoff_polling(
        self,
        mock_forecast_integration,
        mock_forecast_client,
        forecast_predictor_config,
        sample_historical_data
    ):
        """Test predictor polling uses exponential backoff."""
        call_times = []
        
        def describe_predictor_side_effect(*args, **kwargs):
            call_times.append(time.time())
            if len(call_times) == 1:
                raise Exception('ResourceNotFoundException')
            elif len(call_times) <= 4:
                return {'Status': 'CREATE_IN_PROGRESS'}
            else:
                return {'Status': 'ACTIVE', 'AccuracyMetrics': {'RMSE': 12.5}}
        
        mock_forecast_client.describe_predictor.side_effect = describe_predictor_side_effect
        
        with patch('src.training.pipeline.boto3.client') as mock_boto3:
            mock_boto3.return_value = mock_forecast_client
            
            pipeline = TrainingPipeline(
                forecast_integration=mock_forecast_integration,
                forecast_client=mock_forecast_client
            )
            
            with patch.object(pipeline, '_load_data_from_s3', return_value=sample_historical_data):
                result = pipeline.train_forecast_predictor(forecast_predictor_config)
        
        # Verify success
        assert result.success is True
        
        # Verify exponential backoff (delays should increase)
        # We need at least 3 polling calls (after the initial ResourceNotFoundException check)
        if len(call_times) >= 4:
            # Skip first call (ResourceNotFoundException check)
            # Calculate delays between polling calls (calls 2, 3, 4)
            polling_delays = [call_times[i+1] - call_times[i] for i in range(1, min(4, len(call_times)-1))]
            
            # Verify that delays are increasing (with some tolerance for timing variations)
            # First delay should be around poll_interval (1 second)
            # Second delay should be around poll_interval * 1.5 (1.5 seconds)
            if len(polling_delays) >= 2:
                assert polling_delays[0] >= 0.9  # At least 0.9 seconds
                assert polling_delays[1] >= polling_delays[0] * 1.3  # At least 30% increase
    
    def test_metrics_retrieval(
        self,
        mock_forecast_integration,
        mock_forecast_client,
        forecast_predictor_config,
        sample_historical_data
    ):
        """Test predictor metrics are retrieved correctly."""
        with patch('src.training.pipeline.boto3.client') as mock_boto3:
            mock_boto3.return_value = mock_forecast_client
            
            pipeline = TrainingPipeline(
                forecast_integration=mock_forecast_integration,
                forecast_client=mock_forecast_client
            )
            
            with patch.object(pipeline, '_load_data_from_s3', return_value=sample_historical_data):
                result = pipeline.train_forecast_predictor(forecast_predictor_config)
        
        # Verify metrics were retrieved
        assert result.success is True
        assert result.metrics is not None
        assert 'rmse' in result.metrics
        assert 'wql' in result.metrics
        assert result.metrics['rmse'] == 12.5
    
    def test_empty_data_handling(
        self,
        mock_forecast_integration,
        mock_forecast_client,
        forecast_predictor_config
    ):
        """Test predictor training handles empty data."""
        empty_data = pd.DataFrame()
        
        with patch('src.training.pipeline.boto3.client') as mock_boto3:
            mock_boto3.return_value = mock_forecast_client
            
            pipeline = TrainingPipeline(
                forecast_integration=mock_forecast_integration,
                forecast_client=mock_forecast_client
            )
            
            with patch.object(pipeline, '_load_data_from_s3', return_value=empty_data):
                result = pipeline.train_forecast_predictor(forecast_predictor_config)
        
        # Verify failure
        assert result.success is False
        assert len(result.errors) > 0
        assert 'No data loaded from S3' in result.errors
    
    def test_retry_logic_with_transient_error(
        self,
        mock_forecast_integration,
        mock_forecast_client,
        forecast_predictor_config,
        sample_historical_data
    ):
        """Test retry logic with transient Forecast API error."""
        call_count = 0
        
        def import_dataset_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception('Throttling: Request rate exceeded')
            return ForecastImportResult(
                success=True,
                dataset_group_arn='arn:aws:forecast:us-east-1:123456789012:dataset-group/test-group',
                dataset_arn='arn:aws:forecast:us-east-1:123456789012:dataset/test-dataset',
                import_job_arn='arn:aws:forecast:us-east-1:123456789012:dataset-import-job/test-job',
                s3_path='s3://bucket/forecast_datasets/prod_1/test.csv',
                record_count=100
            )
        
        mock_forecast_integration.import_dataset.side_effect = import_dataset_side_effect
        
        with patch('src.training.pipeline.boto3.client') as mock_boto3:
            mock_boto3.return_value = mock_forecast_client
            
            pipeline = TrainingPipeline(
                forecast_integration=mock_forecast_integration,
                forecast_client=mock_forecast_client
            )
            
            with patch.object(pipeline, '_load_data_from_s3', return_value=sample_historical_data):
                result = pipeline.train_forecast_predictor(forecast_predictor_config)
        
        # Verify success after retries
        assert result.success is True
        assert call_count == 3  # Failed twice, succeeded on third attempt
