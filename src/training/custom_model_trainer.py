"""
Custom model trainer module for the Demand Forecasting System.

This module provides functionality to train custom forecasting models using
Prophet or scikit-learn, serialize models, generate predictions, and compute
performance metrics.

Requirements: 2.1, 2.2, 8.2
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Literal, Tuple
import pandas as pd
import numpy as np
import joblib
from io import BytesIO
import logging

# ML libraries
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso

from src.training.data_preparation import TrainingDataset
from src.training.metrics import (
    compute_mae,
    compute_rmse,
    compute_mape,
    PerformanceMetrics
)

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """
    Configuration for model training.
    
    Attributes:
        algorithm: Model algorithm to use
        hyperparameters: Algorithm-specific hyperparameters
        features: List of feature columns to use for training
        forecast_horizon: Number of time steps to forecast
    """
    algorithm: Literal["prophet", "random_forest", "gradient_boosting", "linear", "ridge", "lasso"]
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    features: List[str] = field(default_factory=list)
    forecast_horizon: int = 30


@dataclass
class TrainingResult:
    """
    Result of model training operation.
    
    Attributes:
        success: Whether training succeeded
        model_artifact: Serialized model bytes (if successful)
        metrics: Performance metrics on validation set (if successful)
        model_type: Type of model trained
        errors: List of error messages (if failed)
        warnings: List of warning messages
        metadata: Additional training metadata
    """
    success: bool
    model_artifact: Optional[bytes] = None
    metrics: Optional[PerformanceMetrics] = None
    model_type: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CustomModelTrainer:
    """
    Trains custom forecasting models using Prophet or scikit-learn.
    
    Responsibilities:
    - Train models with configured hyperparameters
    - Serialize trained models using joblib
    - Generate predictions on validation set for backtesting
    - Compute performance metrics (MAE, RMSE, MAPE)
    """
    
    def __init__(self):
        """Initialize the custom model trainer."""
        self.supported_algorithms = {
            "prophet": self._train_prophet,
            "random_forest": self._train_sklearn,
            "gradient_boosting": self._train_sklearn,
            "linear": self._train_sklearn,
            "ridge": self._train_sklearn,
            "lasso": self._train_sklearn
        }
    
    def train_model(
        self,
        dataset: TrainingDataset,
        config: ModelConfig
    ) -> TrainingResult:
        """
        Train a custom forecasting model.
        
        Args:
            dataset: Training dataset with train/validation split
            config: Model configuration with algorithm and hyperparameters
            
        Returns:
            TrainingResult with model artifact, metrics, and metadata
        """
        errors = []
        warnings = []
        
        try:
            # Validate inputs
            validation_errors = self._validate_inputs(dataset, config)
            if validation_errors:
                return TrainingResult(
                    success=False,
                    errors=validation_errors
                )
            
            # Check algorithm support
            if config.algorithm not in self.supported_algorithms:
                return TrainingResult(
                    success=False,
                    errors=[
                        f"Unsupported algorithm: {config.algorithm}. "
                        f"Supported: {list(self.supported_algorithms.keys())}"
                    ]
                )
            
            logger.info(f"Training {config.algorithm} model with config: {config}")
            
            # Train model using appropriate method
            train_method = self.supported_algorithms[config.algorithm]
            model, train_warnings = train_method(dataset, config)
            warnings.extend(train_warnings)
            
            # Generate predictions on validation set
            logger.info("Generating predictions on validation set")
            predictions = self._generate_predictions(
                model,
                dataset.validation_data,
                dataset.feature_columns,
                config.algorithm,
                config
            )
            
            # Get actual values from validation set
            actuals = dataset.validation_data[dataset.target_column].values
            
            # Compute performance metrics
            logger.info("Computing performance metrics")
            metrics = self._compute_metrics(predictions, actuals)
            
            # Serialize model
            logger.info("Serializing model")
            model_artifact = self._serialize_model(model, config.algorithm)
            
            # Prepare metadata
            metadata = {
                'algorithm': config.algorithm,
                'hyperparameters': config.hyperparameters,
                'features': config.features or dataset.feature_columns,
                'forecast_horizon': config.forecast_horizon,
                'train_records': len(dataset.train_data),
                'validation_records': len(dataset.validation_data),
                'target_column': dataset.target_column
            }
            
            logger.info(
                f"Training completed successfully. "
                f"MAE: {metrics.mae:.2f}, RMSE: {metrics.rmse:.2f}, MAPE: {metrics.mape:.2f}%"
            )
            
            return TrainingResult(
                success=True,
                model_artifact=model_artifact,
                metrics=metrics,
                model_type=config.algorithm,
                warnings=warnings,
                metadata=metadata
            )
            
        except Exception as e:
            error_msg = f"Model training failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return TrainingResult(
                success=False,
                errors=[error_msg]
            )
    
    def _validate_inputs(
        self,
        dataset: TrainingDataset,
        config: ModelConfig
    ) -> List[str]:
        """
        Validate training inputs.
        
        Args:
            dataset: Training dataset
            config: Model configuration
            
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        # Validate dataset
        if dataset.train_data.empty:
            errors.append("Training data is empty")
        
        if dataset.validation_data.empty:
            errors.append("Validation data is empty")
        
        if not dataset.feature_columns:
            errors.append("No feature columns specified")
        
        if not dataset.target_column:
            errors.append("No target column specified")
        
        # Validate target column exists
        if dataset.target_column not in dataset.train_data.columns:
            errors.append(
                f"Target column '{dataset.target_column}' not found in training data"
            )
        
        # Validate feature columns exist
        missing_features = [
            f for f in dataset.feature_columns
            if f not in dataset.train_data.columns
        ]
        if missing_features:
            errors.append(
                f"Feature columns not found in training data: {missing_features}"
            )
        
        # Validate config
        if config.forecast_horizon < 1:
            errors.append(
                f"Invalid forecast_horizon: {config.forecast_horizon}. Must be >= 1"
            )
        
        return errors
    
    def _train_prophet(
        self,
        dataset: TrainingDataset,
        config: ModelConfig
    ) -> Tuple[Prophet, List[str]]:
        """
        Train a Prophet model.
        
        Prophet requires data in specific format:
        - 'ds' column for timestamps
        - 'y' column for target values
        - Additional regressors for features
        
        Args:
            dataset: Training dataset
            config: Model configuration
            
        Returns:
            Tuple of (trained_model, warnings)
        """
        warnings = []
        
        # Prepare data in Prophet format
        train_df = dataset.train_data.copy()
        
        # Prophet requires 'ds' (datestamp) and 'y' (target) columns
        if 'timestamp' not in train_df.columns:
            raise ValueError("Prophet requires 'timestamp' column in training data")
        
        train_df['ds'] = pd.to_datetime(train_df['timestamp'])
        train_df['y'] = train_df[dataset.target_column]
        
        # Initialize Prophet with hyperparameters
        prophet_params = config.hyperparameters.copy()
        
        # Set default parameters if not specified
        if 'seasonality_mode' not in prophet_params:
            prophet_params['seasonality_mode'] = 'multiplicative'
        if 'yearly_seasonality' not in prophet_params:
            prophet_params['yearly_seasonality'] = True
        if 'weekly_seasonality' not in prophet_params:
            prophet_params['weekly_seasonality'] = True
        if 'daily_seasonality' not in prophet_params:
            prophet_params['daily_seasonality'] = False
        
        logger.info(f"Initializing Prophet with params: {prophet_params}")
        model = Prophet(**prophet_params)
        
        # Add additional regressors (features)
        features_to_use = config.features or dataset.feature_columns
        for feature in features_to_use:
            if feature in train_df.columns and feature not in ['timestamp', 'ds', 'y']:
                try:
                    model.add_regressor(feature)
                    logger.debug(f"Added regressor: {feature}")
                except Exception as e:
                    warning = f"Could not add regressor '{feature}': {str(e)}"
                    logger.warning(warning)
                    warnings.append(warning)
        
        # Fit the model
        logger.info("Fitting Prophet model")
        model.fit(train_df[['ds', 'y'] + features_to_use])
        
        return model, warnings
    
    def _train_sklearn(
        self,
        dataset: TrainingDataset,
        config: ModelConfig
    ) -> Tuple[Any, List[str]]:
        """
        Train a scikit-learn model.
        
        Args:
            dataset: Training dataset
            config: Model configuration
            
        Returns:
            Tuple of (trained_model, warnings)
        """
        warnings = []
        
        # Select features
        features_to_use = config.features or dataset.feature_columns
        
        # Prepare training data
        X_train = dataset.train_data[features_to_use].values
        y_train = dataset.train_data[dataset.target_column].values
        
        # Initialize model based on algorithm
        model_class = self._get_sklearn_model_class(config.algorithm)
        
        # Get hyperparameters
        hyperparams = config.hyperparameters.copy()
        
        # Set default random_state for reproducibility if not specified
        if config.algorithm in ['random_forest', 'gradient_boosting']:
            if 'random_state' not in hyperparams:
                hyperparams['random_state'] = 42
        
        logger.info(f"Initializing {config.algorithm} with params: {hyperparams}")
        model = model_class(**hyperparams)
        
        # Fit the model
        logger.info(f"Fitting {config.algorithm} model")
        model.fit(X_train, y_train)
        
        return model, warnings
    
    def _get_sklearn_model_class(self, algorithm: str):
        """
        Get scikit-learn model class for algorithm.
        
        Args:
            algorithm: Algorithm name
            
        Returns:
            Model class
        """
        model_classes = {
            'random_forest': RandomForestRegressor,
            'gradient_boosting': GradientBoostingRegressor,
            'linear': LinearRegression,
            'ridge': Ridge,
            'lasso': Lasso
        }
        
        return model_classes[algorithm]
    
    def _generate_predictions(
        self,
        model: Any,
        validation_data: pd.DataFrame,
        feature_columns: List[str],
        algorithm: str,
        config: ModelConfig
    ) -> np.ndarray:
        """
        Generate predictions on validation set.
        
        Args:
            model: Trained model
            validation_data: Validation dataset
            feature_columns: List of feature columns (default features from dataset)
            algorithm: Algorithm type
            config: Model configuration (contains custom features if specified)
            
        Returns:
            Array of predictions
        """
        # Use custom features if specified, otherwise use default feature columns
        features_to_use = config.features or feature_columns
        
        if algorithm == 'prophet':
            # Prophet requires specific format
            val_df = validation_data.copy()
            val_df['ds'] = pd.to_datetime(val_df['timestamp'])
            
            # Create future dataframe with features
            future_df = val_df[['ds'] + features_to_use].copy()
            
            # Generate predictions
            forecast = model.predict(future_df)
            predictions = forecast['yhat'].values
        else:
            # Scikit-learn models
            X_val = validation_data[features_to_use].values
            predictions = model.predict(X_val)
        
        return predictions
    
    def _compute_metrics(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray
    ) -> PerformanceMetrics:
        """
        Compute performance metrics.
        
        Args:
            predictions: Predicted values
            actuals: Actual values
            
        Returns:
            PerformanceMetrics with MAE, RMSE, MAPE
        """
        mae = compute_mae(predictions, actuals)
        rmse = compute_rmse(predictions, actuals)
        mape = compute_mape(predictions, actuals)
        
        return PerformanceMetrics(
            mae=mae,
            rmse=rmse,
            mape=mape,
            sample_size=len(predictions),
            evaluation_period=None
        )
    
    def _serialize_model(self, model: Any, algorithm: str) -> bytes:
        """
        Serialize model to bytes using joblib.
        
        Args:
            model: Trained model
            algorithm: Algorithm type
            
        Returns:
            Serialized model as bytes
        """
        buffer = BytesIO()
        
        # Create a dictionary with model and metadata
        model_package = {
            'model': model,
            'algorithm': algorithm,
            'serialization_version': '1.0'
        }
        
        joblib.dump(model_package, buffer)
        buffer.seek(0)
        
        return buffer.read()
    
    @staticmethod
    def deserialize_model(model_artifact: bytes) -> Tuple[Any, str]:
        """
        Deserialize model from bytes.
        
        Args:
            model_artifact: Serialized model bytes
            
        Returns:
            Tuple of (model, algorithm)
        """
        buffer = BytesIO(model_artifact)
        model_package = joblib.load(buffer)
        
        # Handle both old format (just model) and new format (dict with metadata)
        if isinstance(model_package, dict):
            model = model_package['model']
            algorithm = model_package['algorithm']
        else:
            # Legacy format: just the model
            model = model_package
            algorithm = 'unknown'
        
        return model, algorithm
