"""
Training pipeline orchestration module for the Demand Forecasting System.

This module provides the TrainingPipeline class that orchestrates the complete
training workflow from data loading to model registration, with proper error
handling and retry logic.

Requirements: 2.3, 2.4, 2.5, 10.3
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import time
import logging
from datetime import datetime, timezone
import uuid

from src.training.data_preparation import TrainingDataPreparation, DataPreparationResult
from src.training.custom_model_trainer import CustomModelTrainer, ModelConfig, TrainingResult
from src.training.forecast_integration import AmazonForecastIntegration, ForecastDatasetConfig, ForecastImportResult
from src.training.metrics import PerformanceMetrics
from src.registry.model_registry import ModelRegistry, ModelMetadata
from src.utils.logging_config import logger
import boto3
from botocore.exceptions import ClientError
from config.settings import settings


@dataclass
class PipelineConfig:
    """
    Configuration for training pipeline execution.
    
    Attributes:
        dataset_path: S3 path to historical dataset
        product_id: Product identifier for filtering data
        model_config: Configuration for model training
        train_split: Fraction of data for training (default: 0.8)
        target_column: Name of target column (default: 'sales_volume')
        training_dataset_id: Identifier for the training dataset
        max_retries: Maximum number of retry attempts for transient failures
        retry_base_delay: Base delay in seconds for exponential backoff
    """
    dataset_path: str
    product_id: str
    model_config: ModelConfig
    train_split: float = 0.8
    target_column: str = 'sales_volume'
    training_dataset_id: Optional[str] = None
    max_retries: int = 3
    retry_base_delay: float = 1.0


@dataclass
class ForecastPredictorConfig:
    """
    Configuration for Amazon Forecast predictor training.
    
    Attributes:
        dataset_path: S3 path to historical dataset
        product_id: Product identifier for filtering data
        forecast_horizon: Number of time steps to forecast
        dataset_name: Name for the Forecast dataset
        dataset_group_name: Name for the Forecast dataset group
        predictor_name: Name for the Forecast predictor
        algorithm: Algorithm to use ('auto' for AutoML or specific algorithm ARN)
        dataset_frequency: Data frequency (e.g., 'D' for daily)
        timestamp_format: Format for timestamp column
        training_dataset_id: Identifier for the training dataset
        max_retries: Maximum number of retry attempts for transient failures
        retry_base_delay: Base delay in seconds for exponential backoff
        max_wait_seconds: Maximum time to wait for predictor training (default: 7200)
        poll_interval: Seconds between status checks (default: 60)
    """
    dataset_path: str
    product_id: str
    forecast_horizon: int
    dataset_name: str
    dataset_group_name: str
    predictor_name: str
    algorithm: str = 'auto'
    dataset_frequency: str = 'D'
    timestamp_format: str = 'yyyy-MM-dd HH:mm:ss'
    training_dataset_id: Optional[str] = None
    max_retries: int = 3
    retry_base_delay: float = 1.0
    max_wait_seconds: int = 7200
    poll_interval: int = 60


@dataclass
class PipelineResult:
    """
    Result of training pipeline execution.
    
    Attributes:
        success: Whether pipeline execution succeeded
        model_id: Registered model ID (if successful)
        metadata: Model metadata (if successful)
        errors: List of error messages (if failed)
        warnings: List of warning messages
        execution_time: Total execution time in seconds
        stage_timings: Dictionary of stage names to execution times
    """
    success: bool
    model_id: Optional[str] = None
    metadata: Optional[ModelMetadata] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    stage_timings: Dict[str, float] = field(default_factory=dict)


@dataclass
class ForecastPredictorResult:
    """
    Result of Amazon Forecast predictor training.
    
    Attributes:
        success: Whether predictor training succeeded
        predictor_arn: ARN of the trained predictor
        dataset_import_result: Result of dataset import operation
        metrics: Performance metrics from Forecast (if available)
        errors: List of error messages (if failed)
        warnings: List of warning messages
        execution_time: Total execution time in seconds
        stage_timings: Dictionary of stage names to execution times
    """
    success: bool
    predictor_arn: Optional[str] = None
    dataset_import_result: Optional[ForecastImportResult] = None
    metrics: Optional[Dict[str, float]] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    stage_timings: Dict[str, float] = field(default_factory=dict)


class TrainingPipeline:
    """
    Orchestrates the complete training pipeline workflow.
    
    Responsibilities:
    - Load and prepare training data from S3
    - Apply feature engineering pipeline
    - Train custom forecasting model
    - Evaluate model performance on validation set
    - Register trained model in Model Registry
    - Implement error handling and retry logic for transient failures
    
    The pipeline executes the following stages:
    1. Data Preparation: Load dataset, apply feature engineering, split train/validation
    2. Model Training: Train model with configured hyperparameters
    3. Model Evaluation: Generate predictions and compute metrics
    4. Model Registration: Store model artifact and metadata in registry
    """
    
    def __init__(
        self,
        data_preparation: Optional[TrainingDataPreparation] = None,
        model_trainer: Optional[CustomModelTrainer] = None,
        model_registry: Optional[ModelRegistry] = None,
        forecast_integration: Optional[AmazonForecastIntegration] = None,
        forecast_client=None,
        role_arn: Optional[str] = None
    ):
        """
        Initialize the training pipeline.
        
        Args:
            data_preparation: Optional TrainingDataPreparation instance (for testing)
            model_trainer: Optional CustomModelTrainer instance (for testing)
            model_registry: Optional ModelRegistry instance (for testing)
            forecast_integration: Optional AmazonForecastIntegration instance (for testing)
            forecast_client: Optional boto3 Forecast client (for testing)
            role_arn: IAM role ARN for Forecast to access S3
        """
        self.data_preparation = data_preparation or TrainingDataPreparation()
        self.model_trainer = model_trainer or CustomModelTrainer()
        self.model_registry = model_registry or ModelRegistry()
        self.forecast_integration = forecast_integration or AmazonForecastIntegration(
            role_arn=role_arn
        )
        
        self.forecast_client = forecast_client or boto3.client(
            'forecast',
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key
        )
        
        logger.info("TrainingPipeline initialized")
    
    def train_custom_model(
        self,
        config: PipelineConfig
    ) -> PipelineResult:
        """
        Train custom forecasting model on historical dataset.
        
        This method orchestrates the complete training workflow:
        1. Load and prepare training data from S3
        2. Apply feature engineering (seasonality, preprocessing)
        3. Train custom model with configured hyperparameters
        4. Evaluate model on validation set
        5. Register trained model in Model Registry
        
        The pipeline implements retry logic with exponential backoff for
        transient failures (S3 access, database connections).
        
        Args:
            config: Pipeline configuration with dataset path and model config
            
        Returns:
            PipelineResult with model_id, metrics, and execution metadata
        """
        start_time = time.time()
        stage_timings = {}
        errors = []
        warnings = []
        
        try:
            logger.info(
                f"Starting training pipeline for product_id={config.product_id}, "
                f"algorithm={config.model_config.algorithm}"
            )
            
            # Stage 1: Data Preparation
            logger.info("Stage 1: Data Preparation")
            stage_start = time.time()
            
            data_result = self._execute_with_retry(
                lambda: self.data_preparation.prepare_training_data(
                    dataset_path=config.dataset_path,
                    product_id=config.product_id,
                    train_split=config.train_split,
                    target_column=config.target_column
                ),
                stage_name="data_preparation",
                max_retries=config.max_retries,
                base_delay=config.retry_base_delay
            )
            
            stage_timings['data_preparation'] = time.time() - stage_start
            
            if not data_result.success:
                logger.error(f"Data preparation failed: {data_result.errors}")
                return PipelineResult(
                    success=False,
                    errors=data_result.errors,
                    warnings=data_result.warnings,
                    execution_time=time.time() - start_time,
                    stage_timings=stage_timings
                )
            
            warnings.extend(data_result.warnings)
            dataset = data_result.dataset
            
            logger.info(
                f"Data preparation completed: {len(dataset.train_data)} train records, "
                f"{len(dataset.validation_data)} validation records"
            )
            
            # Stage 2: Model Training
            logger.info("Stage 2: Model Training")
            stage_start = time.time()
            
            training_result = self._execute_with_retry(
                lambda: self.model_trainer.train_model(
                    dataset=dataset,
                    config=config.model_config
                ),
                stage_name="model_training",
                max_retries=config.max_retries,
                base_delay=config.retry_base_delay
            )
            
            stage_timings['model_training'] = time.time() - stage_start
            
            if not training_result.success:
                logger.error(f"Model training failed: {training_result.errors}")
                return PipelineResult(
                    success=False,
                    errors=training_result.errors,
                    warnings=warnings + training_result.warnings,
                    execution_time=time.time() - start_time,
                    stage_timings=stage_timings
                )
            
            warnings.extend(training_result.warnings)
            
            logger.info(
                f"Model training completed: MAE={training_result.metrics.mae:.2f}, "
                f"RMSE={training_result.metrics.rmse:.2f}, MAPE={training_result.metrics.mape:.2f}%"
            )
            
            # Stage 3: Model Registration
            logger.info("Stage 3: Model Registration")
            stage_start = time.time()
            
            # Generate model ID
            model_id = self._generate_model_id(config.product_id, config.model_config.algorithm)
            
            # Determine version number
            version = self._get_next_version(config.product_id, "custom")
            
            # Create model metadata
            training_dataset_id = config.training_dataset_id or config.dataset_path
            
            metadata = ModelMetadata(
                model_id=model_id,
                product_id=config.product_id,
                model_type="custom",
                version=version,
                artifact_path="",  # Will be set by registry
                training_dataset_id=training_dataset_id,
                mae=training_result.metrics.mae,
                rmse=training_result.metrics.rmse,
                mape=training_result.metrics.mape,
                hyperparameters=config.model_config.hyperparameters,
                created_at=datetime.now(timezone.utc),
                forecast_horizon=config.model_config.forecast_horizon
            )
            
            # Register model with retry logic
            registered_model_id = self._execute_with_retry(
                lambda: self.model_registry.register_model(
                    model_artifact=training_result.model_artifact,
                    metadata=metadata
                ),
                stage_name="model_registration",
                max_retries=config.max_retries,
                base_delay=config.retry_base_delay
            )
            
            stage_timings['model_registration'] = time.time() - stage_start
            
            logger.info(f"Model registered successfully: {registered_model_id}")
            
            # Update metadata with registered model ID
            metadata.model_id = registered_model_id
            
            # Calculate total execution time
            execution_time = time.time() - start_time
            
            logger.info(
                f"Training pipeline completed successfully in {execution_time:.2f}s. "
                f"Model ID: {registered_model_id}"
            )
            
            return PipelineResult(
                success=True,
                model_id=registered_model_id,
                metadata=metadata,
                warnings=warnings,
                execution_time=execution_time,
                stage_timings=stage_timings
            )
            
        except Exception as e:
            error_msg = f"Training pipeline failed with unexpected error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            return PipelineResult(
                success=False,
                errors=[error_msg],
                warnings=warnings,
                execution_time=time.time() - start_time,
                stage_timings=stage_timings
            )
    
    def train_forecast_predictor(
        self,
        config: ForecastPredictorConfig
    ) -> ForecastPredictorResult:
        """
        Train Amazon Forecast predictor on historical dataset.
        
        This method orchestrates the complete Forecast predictor training workflow:
        1. Load historical data from S3
        2. Import dataset into Amazon Forecast
        3. Create and configure Forecast predictor (AutoML or specific algorithm)
        4. Poll for training completion with exponential backoff
        5. Retrieve predictor metrics from Forecast API
        
        The method implements retry logic with exponential backoff for
        transient failures (S3 access, Forecast API throttling).
        
        Args:
            config: Forecast predictor configuration with dataset path and parameters
            
        Returns:
            ForecastPredictorResult with predictor ARN, metrics, and execution metadata
        """
        start_time = time.time()
        stage_timings = {}
        errors = []
        warnings = []
        
        try:
            logger.info(
                f"Starting Forecast predictor training for product_id={config.product_id}, "
                f"forecast_horizon={config.forecast_horizon}, algorithm={config.algorithm}"
            )
            
            # Stage 1: Load historical data
            logger.info("Stage 1: Loading historical data")
            stage_start = time.time()
            
            import pandas as pd
            from io import StringIO
            
            # Load data from S3
            data = self._execute_with_retry(
                lambda: self._load_data_from_s3(config.dataset_path),
                stage_name="load_data",
                max_retries=config.max_retries,
                base_delay=config.retry_base_delay
            )
            
            stage_timings['load_data'] = time.time() - stage_start
            
            if data.empty:
                return ForecastPredictorResult(
                    success=False,
                    errors=["No data loaded from S3"],
                    execution_time=time.time() - start_time,
                    stage_timings=stage_timings
                )
            
            logger.info(f"Loaded {len(data)} records from S3")
            
            # Stage 2: Import dataset into Forecast
            logger.info("Stage 2: Importing dataset into Amazon Forecast")
            stage_start = time.time()
            
            dataset_config = ForecastDatasetConfig(
                dataset_name=config.dataset_name,
                dataset_group_name=config.dataset_group_name,
                dataset_frequency=config.dataset_frequency,
                timestamp_format=config.timestamp_format
            )
            
            import_result = self._execute_with_retry(
                lambda: self.forecast_integration.import_dataset(
                    data=data,
                    config=dataset_config,
                    product_id=config.product_id
                ),
                stage_name="dataset_import",
                max_retries=config.max_retries,
                base_delay=config.retry_base_delay
            )
            
            stage_timings['dataset_import'] = time.time() - stage_start
            
            if not import_result.success:
                logger.error(f"Dataset import failed: {import_result.errors}")
                return ForecastPredictorResult(
                    success=False,
                    dataset_import_result=import_result,
                    errors=import_result.errors,
                    warnings=import_result.warnings,
                    execution_time=time.time() - start_time,
                    stage_timings=stage_timings
                )
            
            warnings.extend(import_result.warnings)
            
            logger.info(
                f"Dataset import initiated: {import_result.record_count} records, "
                f"import_job_arn={import_result.import_job_arn}"
            )
            
            # Stage 3: Wait for dataset import completion
            logger.info("Stage 3: Waiting for dataset import completion")
            stage_start = time.time()
            
            import_success = self._execute_with_retry(
                lambda: self.forecast_integration.wait_for_import_completion(
                    import_job_arn=import_result.import_job_arn,
                    max_wait_seconds=config.max_wait_seconds,
                    poll_interval=config.poll_interval
                ),
                stage_name="wait_import",
                max_retries=config.max_retries,
                base_delay=config.retry_base_delay
            )
            
            stage_timings['wait_import'] = time.time() - stage_start
            
            if not import_success:
                return ForecastPredictorResult(
                    success=False,
                    dataset_import_result=import_result,
                    errors=["Dataset import job failed or timed out"],
                    warnings=warnings,
                    execution_time=time.time() - start_time,
                    stage_timings=stage_timings
                )
            
            logger.info("Dataset import completed successfully")
            
            # Stage 4: Create Forecast predictor
            logger.info("Stage 4: Creating Forecast predictor")
            stage_start = time.time()
            
            predictor_arn = self._execute_with_retry(
                lambda: self._create_predictor(
                    predictor_name=config.predictor_name,
                    forecast_horizon=config.forecast_horizon,
                    dataset_group_arn=import_result.dataset_group_arn,
                    algorithm=config.algorithm
                ),
                stage_name="create_predictor",
                max_retries=config.max_retries,
                base_delay=config.retry_base_delay
            )
            
            stage_timings['create_predictor'] = time.time() - stage_start
            
            logger.info(f"Predictor created: {predictor_arn}")
            
            # Stage 5: Wait for predictor training completion
            logger.info("Stage 5: Waiting for predictor training completion")
            stage_start = time.time()
            
            training_success = self._execute_with_retry(
                lambda: self._wait_for_predictor_completion(
                    predictor_arn=predictor_arn,
                    max_wait_seconds=config.max_wait_seconds,
                    poll_interval=config.poll_interval
                ),
                stage_name="wait_predictor",
                max_retries=config.max_retries,
                base_delay=config.retry_base_delay
            )
            
            stage_timings['wait_predictor'] = time.time() - stage_start
            
            if not training_success:
                return ForecastPredictorResult(
                    success=False,
                    predictor_arn=predictor_arn,
                    dataset_import_result=import_result,
                    errors=["Predictor training failed or timed out"],
                    warnings=warnings,
                    execution_time=time.time() - start_time,
                    stage_timings=stage_timings
                )
            
            logger.info("Predictor training completed successfully")
            
            # Stage 6: Retrieve predictor metrics
            logger.info("Stage 6: Retrieving predictor metrics")
            stage_start = time.time()
            
            metrics = self._execute_with_retry(
                lambda: self._get_predictor_metrics(predictor_arn),
                stage_name="get_metrics",
                max_retries=config.max_retries,
                base_delay=config.retry_base_delay
            )
            
            stage_timings['get_metrics'] = time.time() - stage_start
            
            logger.info(f"Retrieved predictor metrics: {metrics}")
            
            # Calculate total execution time
            execution_time = time.time() - start_time
            
            logger.info(
                f"Forecast predictor training completed successfully in {execution_time:.2f}s. "
                f"Predictor ARN: {predictor_arn}"
            )
            
            return ForecastPredictorResult(
                success=True,
                predictor_arn=predictor_arn,
                dataset_import_result=import_result,
                metrics=metrics,
                warnings=warnings,
                execution_time=execution_time,
                stage_timings=stage_timings
            )
            
        except Exception as e:
            error_msg = f"Forecast predictor training failed with unexpected error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            return ForecastPredictorResult(
                success=False,
                errors=[error_msg],
                warnings=warnings,
                execution_time=time.time() - start_time,
                stage_timings=stage_timings
            )
    
    def _load_data_from_s3(self, s3_path: str) -> 'pd.DataFrame':
        """
        Load historical data from S3.
        
        Args:
            s3_path: S3 URI (s3://bucket/key) or local file path
            
        Returns:
            DataFrame with historical data
        """
        import pandas as pd
        from io import StringIO
        
        if s3_path.startswith('s3://'):
            # Parse S3 URI
            parts = s3_path.replace('s3://', '').split('/', 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ''
            
            # Download from S3
            s3_client = boto3.client(
                's3',
                region_name=settings.aws_region,
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key
            )
            
            response = s3_client.get_object(Bucket=bucket, Key=key)
            content = response['Body'].read().decode('utf-8')
            
            # Parse CSV
            data = pd.read_csv(StringIO(content))
        else:
            # Load from local file
            data = pd.read_csv(s3_path)
        
        return data
    
    def _create_predictor(
        self,
        predictor_name: str,
        forecast_horizon: int,
        dataset_group_arn: str,
        algorithm: str
    ) -> str:
        """
        Create Amazon Forecast predictor.
        
        Args:
            predictor_name: Name for the predictor
            forecast_horizon: Number of time steps to forecast
            dataset_group_arn: ARN of the dataset group
            algorithm: Algorithm to use ('auto' for AutoML or specific algorithm ARN)
            
        Returns:
            Predictor ARN
        """
        try:
            # Check if predictor already exists
            try:
                predictor_arn = self._build_forecast_arn('predictor', predictor_name)
                response = self.forecast_client.describe_predictor(
                    PredictorArn=predictor_arn
                )
                
                logger.info(f"Predictor already exists: {predictor_arn}")
                return predictor_arn
                
            except (ClientError, Exception) as e:
                # Check if it's a ResourceNotFoundException
                error_str = str(e)
                if 'ResourceNotFoundException' not in error_str:
                    # If it's not a ResourceNotFoundException, re-raise
                    if isinstance(e, ClientError):
                        if e.response['Error']['Code'] != 'ResourceNotFoundException':
                            raise
                    else:
                        # For non-ClientError exceptions, only continue if it's ResourceNotFoundException
                        if 'ResourceNotFoundException' not in error_str:
                            raise
                
                # Predictor doesn't exist, create it
                logger.info(f"Creating new predictor: {predictor_name}")
                
                # Configure predictor parameters
                predictor_params = {
                    'PredictorName': predictor_name,
                    'ForecastHorizon': forecast_horizon,
                    'InputDataConfig': {
                        'DatasetGroupArn': dataset_group_arn
                    },
                    'FeaturizationConfig': {
                        'ForecastFrequency': 'D'
                    }
                }
                
                # Set algorithm configuration
                if algorithm == 'auto':
                    # Use AutoML
                    predictor_params['PerformAutoML'] = True
                else:
                    # Use specific algorithm
                    predictor_params['AlgorithmArn'] = algorithm
                
                response = self.forecast_client.create_predictor(**predictor_params)
                
                predictor_arn = response['PredictorArn']
                logger.info(f"Created predictor: {predictor_arn}")
                
                return predictor_arn
                
        except ClientError as e:
            logger.error(f"Failed to create predictor: {str(e)}")
            raise
    
    def _wait_for_predictor_completion(
        self,
        predictor_arn: str,
        max_wait_seconds: int,
        poll_interval: int
    ) -> bool:
        """
        Wait for predictor training to complete with exponential backoff.
        
        Polls the predictor status until it reaches a terminal state
        (ACTIVE or CREATE_FAILED). Uses exponential backoff for polling
        to reduce API calls.
        
        Args:
            predictor_arn: ARN of the predictor
            max_wait_seconds: Maximum time to wait in seconds
            poll_interval: Initial seconds between status checks
            
        Returns:
            True if training succeeded, False if failed
        """
        start_time = time.time()
        current_poll_interval = poll_interval
        max_poll_interval = 300  # Cap at 5 minutes
        
        logger.info(f"Waiting for predictor training to complete: {predictor_arn}")
        
        while True:
            try:
                response = self.forecast_client.describe_predictor(
                    PredictorArn=predictor_arn
                )
                
                status = response['Status']
                
                logger.info(f"Predictor status: {status}")
                
                if status == 'ACTIVE':
                    logger.info("Predictor training completed successfully")
                    return True
                
                elif status == 'CREATE_FAILED':
                    error_message = response.get('Message', 'Unknown error')
                    logger.error(f"Predictor training failed: {error_message}")
                    return False
                
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > max_wait_seconds:
                    logger.error(
                        f"Predictor training did not complete within {max_wait_seconds}s. "
                        f"Current status: {status}"
                    )
                    return False
                
                # Wait before next poll with exponential backoff
                time.sleep(current_poll_interval)
                
                # Increase poll interval exponentially (up to max)
                current_poll_interval = min(current_poll_interval * 1.5, max_poll_interval)
                
            except ClientError as e:
                logger.error(f"Error checking predictor status: {str(e)}")
                return False
    
    def _get_predictor_metrics(self, predictor_arn: str) -> Dict[str, float]:
        """
        Retrieve predictor performance metrics from Forecast API.
        
        Args:
            predictor_arn: ARN of the predictor
            
        Returns:
            Dictionary of metrics (RMSE, WAPE, etc.)
        """
        try:
            response = self.forecast_client.describe_predictor(
                PredictorArn=predictor_arn
            )
            
            # Extract metrics from predictor metadata
            metrics = {}
            
            # Get predictor evaluation metrics if available
            if 'PredictorExecutionDetails' in response:
                execution_details = response['PredictorExecutionDetails']
                
                if 'PredictorExecutions' in execution_details:
                    for execution in execution_details['PredictorExecutions']:
                        if 'TestWindows' in execution:
                            for window in execution['TestWindows']:
                                if 'Metrics' in window:
                                    window_metrics = window['Metrics']
                                    
                                    # Extract common metrics
                                    if 'RMSE' in window_metrics:
                                        metrics['rmse'] = window_metrics['RMSE']
                                    if 'WeightedQuantileLosses' in window_metrics:
                                        wql = window_metrics['WeightedQuantileLosses']
                                        if wql:
                                            # Use average weighted quantile loss
                                            metrics['wql'] = sum(wql) / len(wql)
                                    if 'ErrorMetrics' in window_metrics:
                                        error_metrics = window_metrics['ErrorMetrics']
                                        if 'WAPE' in error_metrics:
                                            metrics['wape'] = error_metrics['WAPE']
                                        if 'RMSE' in error_metrics:
                                            metrics['rmse'] = error_metrics['RMSE']
                                        if 'MASE' in error_metrics:
                                            metrics['mase'] = error_metrics['MASE']
            
            # If no metrics found in execution details, try accuracy metrics
            if not metrics and 'AccuracyMetrics' in response:
                accuracy_metrics = response['AccuracyMetrics']
                
                if 'RMSE' in accuracy_metrics:
                    metrics['rmse'] = accuracy_metrics['RMSE']
                if 'WeightedQuantileLosses' in accuracy_metrics:
                    wql = accuracy_metrics['WeightedQuantileLosses']
                    if wql:
                        metrics['wql'] = sum(wql) / len(wql)
            
            logger.info(f"Retrieved {len(metrics)} metrics from predictor")
            
            return metrics
            
        except ClientError as e:
            logger.error(f"Failed to retrieve predictor metrics: {str(e)}")
            # Return empty metrics rather than failing
            return {}
    
    def _build_forecast_arn(self, resource_type: str, resource_name: str) -> str:
        """
        Build Amazon Forecast ARN for a resource.
        
        ARN format: arn:aws:forecast:region:account-id:resource-type/resource-name
        
        Args:
            resource_type: Type of resource (e.g., 'predictor', 'dataset')
            resource_name: Name of the resource
            
        Returns:
            Full ARN string
        """
        # Get account ID from STS
        try:
            sts_client = boto3.client(
                'sts',
                region_name=settings.aws_region,
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key
            )
            account_id = sts_client.get_caller_identity()['Account']
        except Exception as e:
            logger.warning(f"Could not get account ID: {str(e)}. Using placeholder.")
            account_id = "123456789012"
        
        arn = f"arn:aws:forecast:{settings.aws_region}:{account_id}:{resource_type}/{resource_name}"
        
        return arn
    
    def _execute_with_retry(
        self,
        operation,
        stage_name: str,
        max_retries: int,
        base_delay: float
    ):
        """
        Execute an operation with exponential backoff retry logic.
        
        Implements retry logic for transient failures such as:
        - S3 access errors (throttling, temporary unavailability)
        - Database connection errors
        - Network timeouts
        
        The retry delay follows exponential backoff:
        delay = base_delay * (2 ^ attempt_number)
        
        Args:
            operation: Callable to execute
            stage_name: Name of the stage (for logging)
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds for exponential backoff
            
        Returns:
            Result of the operation
            
        Raises:
            Exception: If all retry attempts fail
        """
        attempt = 0
        last_exception = None
        
        while attempt <= max_retries:
            try:
                if attempt > 0:
                    logger.info(
                        f"Retry attempt {attempt}/{max_retries} for stage: {stage_name}"
                    )
                
                result = operation()
                
                if attempt > 0:
                    logger.info(
                        f"Stage {stage_name} succeeded after {attempt} retry attempts"
                    )
                
                return result
                
            except Exception as e:
                last_exception = e
                attempt += 1
                
                # Check if this is a transient error that should be retried
                if not self._is_transient_error(e):
                    logger.error(
                        f"Non-transient error in stage {stage_name}: {str(e)}. "
                        "Not retrying."
                    )
                    raise
                
                if attempt <= max_retries:
                    # Calculate exponential backoff delay
                    delay = base_delay * (2 ** (attempt - 1))
                    
                    logger.warning(
                        f"Transient error in stage {stage_name}: {str(e)}. "
                        f"Retrying in {delay:.2f}s (attempt {attempt}/{max_retries})"
                    )
                    
                    time.sleep(delay)
                else:
                    logger.error(
                        f"Stage {stage_name} failed after {max_retries} retry attempts. "
                        f"Last error: {str(e)}"
                    )
        
        # All retries exhausted
        raise last_exception
    
    def _is_transient_error(self, error: Exception) -> bool:
        """
        Determine if an error is transient and should be retried.
        
        Transient errors include:
        - S3 throttling errors (SlowDown, RequestTimeout)
        - Database connection errors
        - Network timeouts
        - Temporary service unavailability
        
        Args:
            error: Exception to check
            
        Returns:
            True if error is transient, False otherwise
        """
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        # S3 transient errors
        transient_keywords = [
            'slowdown',
            'requesttimeout',
            'timeout',
            'throttl',
            'connection',
            'temporarily unavailable',
            'service unavailable',
            'too many requests',
            '503',
            '429'
        ]
        
        # Check error message for transient keywords
        for keyword in transient_keywords:
            if keyword in error_str:
                return True
        
        # Check for specific exception types
        transient_types = [
            'ConnectionError',
            'TimeoutError',
            'OperationalError',  # Database connection errors
            'ClientError'  # May include S3 throttling
        ]
        
        if error_type in transient_types:
            return True
        
        return False
    
    def _generate_model_id(self, product_id: str, algorithm: str) -> str:
        """
        Generate a unique model ID.
        
        Format: {product_id}_{algorithm}_{timestamp}_{uuid}
        
        Args:
            product_id: Product identifier
            algorithm: Model algorithm name
            
        Returns:
            Unique model ID
        """
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        
        model_id = f"{product_id}_{algorithm}_{timestamp}_{unique_id}"
        
        return model_id
    
    def _get_next_version(self, product_id: str, model_type: str) -> int:
        """
        Get the next version number for a model.
        
        Args:
            product_id: Product identifier
            model_type: Model type ('custom' or 'forecast')
            
        Returns:
            Next version number (1 if no existing models)
        """
        try:
            # Get existing models for this product and type
            models = self.model_registry.list_models(
                product_id=product_id,
                model_type=model_type
            )
            
            if not models:
                return 1
            
            # Find maximum version number
            max_version = max(model.version for model in models)
            
            return max_version + 1
            
        except Exception as e:
            logger.warning(
                f"Could not determine next version number: {str(e)}. "
                "Defaulting to version 1."
            )
            return 1
    
    def register_forecast_predictor(
        self,
        predictor_arn: str,
        product_id: str,
        forecast_horizon: int,
        metrics: Dict[str, float],
        training_dataset_id: str,
        algorithm: str = 'auto'
    ) -> Optional[str]:
        """
        Register Amazon Forecast predictor in Model Registry.
        
        This method creates a model registry entry for a trained Forecast predictor,
        allowing it to be tracked alongside custom models for comparison and
        model selection.
        
        Args:
            predictor_arn: ARN of the trained Forecast predictor
            product_id: Product identifier
            forecast_horizon: Number of time steps forecasted
            metrics: Performance metrics from Forecast (RMSE, WAPE, etc.)
            training_dataset_id: Identifier for the training dataset
            algorithm: Algorithm used ('auto' for AutoML or specific algorithm)
            
        Returns:
            Registered model ID if successful, None if registration fails
        """
        try:
            logger.info(f"Registering Forecast predictor in Model Registry: {predictor_arn}")
            
            # Generate model ID
            model_id = self._generate_model_id(product_id, f"forecast_{algorithm}")
            
            # Determine version number
            version = self._get_next_version(product_id, "forecast")
            
            # Convert Forecast metrics to standard format
            # Forecast provides RMSE, WAPE, MASE, etc.
            # We need MAE, RMSE, MAPE for consistency
            mae = metrics.get('mae', metrics.get('MASE', 0.0))
            rmse = metrics.get('rmse', metrics.get('RMSE', 0.0))
            mape = metrics.get('mape', metrics.get('wape', 0.0) * 100)  # Convert WAPE to percentage
            
            # Create model metadata
            metadata = ModelMetadata(
                model_id=model_id,
                product_id=product_id,
                model_type="forecast",
                version=version,
                artifact_path=predictor_arn,  # Store predictor ARN as artifact path
                training_dataset_id=training_dataset_id,
                mae=mae,
                rmse=rmse,
                mape=mape,
                hyperparameters={'algorithm': algorithm, 'predictor_arn': predictor_arn},
                created_at=datetime.now(timezone.utc),
                forecast_horizon=forecast_horizon
            )
            
            # For Forecast predictors, we don't have a binary artifact
            # Use a placeholder that indicates this is a Forecast model
            placeholder_artifact = f"FORECAST_PREDICTOR:{predictor_arn}".encode('utf-8')
            
            # Register model with retry logic
            registered_model_id = self._execute_with_retry(
                lambda: self.model_registry.register_model(
                    model_artifact=placeholder_artifact,
                    metadata=metadata
                ),
                stage_name="forecast_model_registration",
                max_retries=3,
                base_delay=1.0
            )
            
            logger.info(
                f"Forecast predictor registered successfully: {registered_model_id}, "
                f"version={version}, RMSE={rmse:.2f}"
            )
            
            return registered_model_id
            
        except Exception as e:
            error_msg = f"Failed to register Forecast predictor in Model Registry: {str(e)}"
            logger.error(error_msg, exc_info=True)
            # Return None to indicate failure, but don't raise exception
            # This allows the pipeline to continue even if registration fails
            return None
    
    def train_with_forecast_fallback(
        self,
        custom_config: PipelineConfig,
        forecast_config: Optional[ForecastPredictorConfig] = None
    ) -> Dict[str, Any]:
        """
        Train both custom model and Forecast predictor with fallback logic.
        
        This method implements the benchmark-driven training workflow:
        1. Train custom model (primary)
        2. Train Forecast predictor (benchmark) - failures logged but don't block
        3. Register both models in Model Registry
        4. Generate comparison report if both succeed
        
        Forecast training failures are logged as warnings and do not prevent
        custom model training from completing successfully.
        
        Args:
            custom_config: Configuration for custom model training
            forecast_config: Optional configuration for Forecast predictor training
            
        Returns:
            Dictionary with results:
            - custom_result: PipelineResult for custom model
            - forecast_result: Optional ForecastPredictorResult for Forecast predictor
            - forecast_model_id: Optional model ID if Forecast predictor was registered
            - comparison: Optional comparison metrics if both models trained
        """
        results = {
            'custom_result': None,
            'forecast_result': None,
            'forecast_model_id': None,
            'comparison': None
        }
        
        # Train custom model (primary)
        logger.info("Training custom model (primary)")
        custom_result = self.train_custom_model(custom_config)
        results['custom_result'] = custom_result
        
        if not custom_result.success:
            logger.error("Custom model training failed. Skipping Forecast training.")
            return results
        
        logger.info(f"Custom model training succeeded: {custom_result.model_id}")
        
        # Train Forecast predictor (benchmark) if config provided
        if forecast_config:
            logger.info("Training Forecast predictor (benchmark)")
            
            try:
                forecast_result = self.train_forecast_predictor(forecast_config)
                results['forecast_result'] = forecast_result
                
                if forecast_result.success:
                    logger.info(
                        f"Forecast predictor training succeeded: {forecast_result.predictor_arn}"
                    )
                    
                    # Register Forecast predictor in Model Registry
                    forecast_model_id = self.register_forecast_predictor(
                        predictor_arn=forecast_result.predictor_arn,
                        product_id=custom_config.product_id,
                        forecast_horizon=custom_config.model_config.forecast_horizon,
                        metrics=forecast_result.metrics or {},
                        training_dataset_id=forecast_config.training_dataset_id or forecast_config.dataset_path,
                        algorithm=forecast_config.algorithm
                    )
                    
                    results['forecast_model_id'] = forecast_model_id
                    
                    if forecast_model_id:
                        logger.info(f"Forecast predictor registered: {forecast_model_id}")
                        
                        # Generate comparison if both models succeeded
                        if custom_result.metadata and forecast_result.metrics:
                            results['comparison'] = self._generate_comparison_metrics(
                                custom_result.metadata,
                                forecast_result.metrics
                            )
                    else:
                        logger.warning(
                            "Forecast predictor registration failed, but custom model training succeeded"
                        )
                else:
                    logger.warning(
                        f"Forecast predictor training failed: {forecast_result.errors}. "
                        "Custom model training succeeded and will be used."
                    )
                    
            except Exception as e:
                error_msg = f"Forecast predictor training failed with exception: {str(e)}"
                logger.warning(error_msg, exc_info=True)
                logger.info("Custom model training succeeded despite Forecast failure")
        
        return results
    
    def _generate_comparison_metrics(
        self,
        custom_metadata: ModelMetadata,
        forecast_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Generate comparison metrics between custom and Forecast models.
        
        Args:
            custom_metadata: Metadata for custom model
            forecast_metrics: Metrics from Forecast predictor
            
        Returns:
            Dictionary with comparison metrics
        """
        # Extract Forecast metrics in standard format
        forecast_rmse = forecast_metrics.get('rmse', forecast_metrics.get('RMSE', 0.0))
        forecast_mae = forecast_metrics.get('mae', forecast_metrics.get('MASE', 0.0))
        forecast_mape = forecast_metrics.get('mape', forecast_metrics.get('wape', 0.0) * 100)
        
        # Calculate improvements (positive = custom is better)
        rmse_improvement = ((forecast_rmse - custom_metadata.rmse) / forecast_rmse * 100) if forecast_rmse > 0 else 0
        mae_improvement = ((forecast_mae - custom_metadata.mae) / forecast_mae * 100) if forecast_mae > 0 else 0
        mape_improvement = ((forecast_mape - custom_metadata.mape) / forecast_mape * 100) if forecast_mape > 0 else 0
        
        comparison = {
            'custom': {
                'mae': custom_metadata.mae,
                'rmse': custom_metadata.rmse,
                'mape': custom_metadata.mape
            },
            'forecast': {
                'mae': forecast_mae,
                'rmse': forecast_rmse,
                'mape': forecast_mape
            },
            'improvement': {
                'mae_pct': mae_improvement,
                'rmse_pct': rmse_improvement,
                'mape_pct': mape_improvement
            },
            'recommendation': 'custom' if rmse_improvement > 0 else 'forecast'
        }
        
        logger.info(
            f"Model comparison: Custom RMSE={custom_metadata.rmse:.2f}, "
            f"Forecast RMSE={forecast_rmse:.2f}, "
            f"Improvement={rmse_improvement:.1f}%"
        )
        
        return comparison
