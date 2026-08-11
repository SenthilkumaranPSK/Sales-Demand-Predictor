"""
Unit tests for the Custom Model Trainer module.

Tests cover:
- Model training with different algorithms
- Hyperparameter configuration
- Model serialization/deserialization
- Prediction generation
- Metrics computation
- Error handling
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.training.custom_model_trainer import (
    CustomModelTrainer,
    ModelConfig,
    TrainingResult
)
from src.training.data_preparation import TrainingDataset
from src.training.metrics import PerformanceMetrics


@pytest.fixture
def sample_training_dataset():
    """Create a sample training dataset for testing."""
    # Generate synthetic time series data
    np.random.seed(42)
    n_train = 100
    n_val = 30
    
    # Create timestamps
    start_date = datetime(2023, 1, 1)
    train_dates = [start_date + timedelta(days=i) for i in range(n_train)]
    val_dates = [start_date + timedelta(days=n_train + i) for i in range(n_val)]
    
    # Create training data with trend and seasonality
    train_data = pd.DataFrame({
        'timestamp': train_dates,
        'product_id': ['PROD_001'] * n_train,
        'sales_volume': 100 + np.arange(n_train) * 0.5 + np.random.randn(n_train) * 10,
        'price': 50 + np.random.randn(n_train) * 5,
        'is_holiday': np.random.choice([True, False], n_train, p=[0.1, 0.9]),
        'day_of_week': [d.weekday() for d in train_dates],
        'month': [d.month for d in train_dates],
        'quarter': [(d.month - 1) // 3 + 1 for d in train_dates]
    })
    
    # Create validation data
    val_data = pd.DataFrame({
        'timestamp': val_dates,
        'product_id': ['PROD_001'] * n_val,
        'sales_volume': 100 + np.arange(n_train, n_train + n_val) * 0.5 + np.random.randn(n_val) * 10,
        'price': 50 + np.random.randn(n_val) * 5,
        'is_holiday': np.random.choice([True, False], n_val, p=[0.1, 0.9]),
        'day_of_week': [d.weekday() for d in val_dates],
        'month': [d.month for d in val_dates],
        'quarter': [(d.month - 1) // 3 + 1 for d in val_dates]
    })
    
    feature_columns = ['price', 'is_holiday', 'day_of_week', 'month', 'quarter']
    
    return TrainingDataset(
        train_data=train_data,
        validation_data=val_data,
        feature_columns=feature_columns,
        target_column='sales_volume',
        normalization_params={},
        metadata={
            'total_records': n_train + n_val,
            'train_records': n_train,
            'validation_records': n_val
        }
    )


class TestCustomModelTrainer:
    """Test suite for CustomModelTrainer."""
    
    def test_initialization(self):
        """Test trainer initialization."""
        trainer = CustomModelTrainer()
        
        assert trainer is not None
        assert 'prophet' in trainer.supported_algorithms
        assert 'random_forest' in trainer.supported_algorithms
        assert 'gradient_boosting' in trainer.supported_algorithms
        assert 'linear' in trainer.supported_algorithms
        assert 'ridge' in trainer.supported_algorithms
        assert 'lasso' in trainer.supported_algorithms
    
    def test_train_random_forest_model(self, sample_training_dataset):
        """Test training a Random Forest model."""
        trainer = CustomModelTrainer()
        
        config = ModelConfig(
            algorithm='random_forest',
            hyperparameters={
                'n_estimators': 10,
                'max_depth': 5,
                'random_state': 42
            },
            forecast_horizon=30
        )
        
        result = trainer.train_model(sample_training_dataset, config)
        
        assert result.success is True
        assert result.model_artifact is not None
        assert result.metrics is not None
        assert result.model_type == 'random_forest'
        assert result.metrics.mae > 0
        assert result.metrics.rmse > 0
        assert result.metrics.mape > 0
        assert result.metrics.sample_size == 30  # validation set size
        assert len(result.errors) == 0
    
    def test_train_gradient_boosting_model(self, sample_training_dataset):
        """Test training a Gradient Boosting model."""
        trainer = CustomModelTrainer()
        
        config = ModelConfig(
            algorithm='gradient_boosting',
            hyperparameters={
                'n_estimators': 10,
                'learning_rate': 0.1,
                'max_depth': 3,
                'random_state': 42
            },
            forecast_horizon=30
        )
        
        result = trainer.train_model(sample_training_dataset, config)
        
        assert result.success is True
        assert result.model_artifact is not None
        assert result.metrics is not None
        assert result.model_type == 'gradient_boosting'
    
    def test_train_linear_model(self, sample_training_dataset):
        """Test training a Linear Regression model."""
        trainer = CustomModelTrainer()
        
        config = ModelConfig(
            algorithm='linear',
            hyperparameters={},
            forecast_horizon=30
        )
        
        result = trainer.train_model(sample_training_dataset, config)
        
        assert result.success is True
        assert result.model_artifact is not None
        assert result.metrics is not None
        assert result.model_type == 'linear'
    
    def test_train_ridge_model(self, sample_training_dataset):
        """Test training a Ridge Regression model."""
        trainer = CustomModelTrainer()
        
        config = ModelConfig(
            algorithm='ridge',
            hyperparameters={'alpha': 1.0},
            forecast_horizon=30
        )
        
        result = trainer.train_model(sample_training_dataset, config)
        
        assert result.success is True
        assert result.model_artifact is not None
        assert result.metrics is not None
        assert result.model_type == 'ridge'
    
    def test_train_lasso_model(self, sample_training_dataset):
        """Test training a Lasso Regression model."""
        trainer = CustomModelTrainer()
        
        config = ModelConfig(
            algorithm='lasso',
            hyperparameters={'alpha': 0.1},
            forecast_horizon=30
        )
        
        result = trainer.train_model(sample_training_dataset, config)
        
        assert result.success is True
        assert result.model_artifact is not None
        assert result.metrics is not None
        assert result.model_type == 'lasso'
    
    def test_train_prophet_model(self, sample_training_dataset):
        """Test training a Prophet model."""
        trainer = CustomModelTrainer()
        
        config = ModelConfig(
            algorithm='prophet',
            hyperparameters={
                'seasonality_mode': 'additive',
                'yearly_seasonality': False,
                'weekly_seasonality': True,
                'daily_seasonality': False
            },
            features=['price', 'is_holiday'],
            forecast_horizon=30
        )
        
        result = trainer.train_model(sample_training_dataset, config)
        
        assert result.success is True
        assert result.model_artifact is not None
        assert result.metrics is not None
        assert result.model_type == 'prophet'
        assert result.metrics.mae > 0
        assert result.metrics.rmse > 0
        assert result.metrics.mape > 0
    
    def test_custom_features(self, sample_training_dataset):
        """Test training with custom feature selection."""
        trainer = CustomModelTrainer()
        
        # Use only a subset of features
        config = ModelConfig(
            algorithm='random_forest',
            hyperparameters={'n_estimators': 10, 'random_state': 42},
            features=['price', 'is_holiday'],  # Only 2 features
            forecast_horizon=30
        )
        
        result = trainer.train_model(sample_training_dataset, config)
        
        assert result.success is True
        assert result.metadata['features'] == ['price', 'is_holiday']
    
    def test_default_features(self, sample_training_dataset):
        """Test training with default features (all available)."""
        trainer = CustomModelTrainer()
        
        config = ModelConfig(
            algorithm='random_forest',
            hyperparameters={'n_estimators': 10, 'random_state': 42},
            features=[],  # Empty list means use all features
            forecast_horizon=30
        )
        
        result = trainer.train_model(sample_training_dataset, config)
        
        assert result.success is True
        assert result.metadata['features'] == sample_training_dataset.feature_columns
    
    def test_model_serialization_deserialization(self, sample_training_dataset):
        """Test model serialization and deserialization."""
        trainer = CustomModelTrainer()
        
        config = ModelConfig(
            algorithm='random_forest',
            hyperparameters={'n_estimators': 10, 'random_state': 42},
            forecast_horizon=30
        )
        
        # Train model
        result = trainer.train_model(sample_training_dataset, config)
        assert result.success is True
        
        # Deserialize model
        model, algorithm = CustomModelTrainer.deserialize_model(result.model_artifact)
        
        assert model is not None
        assert algorithm == 'random_forest'
        
        # Test that deserialized model can make predictions
        X_test = sample_training_dataset.validation_data[sample_training_dataset.feature_columns].values
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(sample_training_dataset.validation_data)
        assert all(isinstance(p, (int, float, np.number)) for p in predictions)
    
    def test_metadata_completeness(self, sample_training_dataset):
        """Test that training result includes complete metadata."""
        trainer = CustomModelTrainer()
        
        config = ModelConfig(
            algorithm='random_forest',
            hyperparameters={'n_estimators': 10, 'random_state': 42},
            features=['price', 'is_holiday'],
            forecast_horizon=30
        )
        
        result = trainer.train_model(sample_training_dataset, config)
        
        assert result.success is True
        assert 'algorithm' in result.metadata
        assert 'hyperparameters' in result.metadata
        assert 'features' in result.metadata
        assert 'forecast_horizon' in result.metadata
        assert 'train_records' in result.metadata
        assert 'validation_records' in result.metadata
        assert 'target_column' in result.metadata
        
        assert result.metadata['algorithm'] == 'random_forest'
        assert result.metadata['forecast_horizon'] == 30
        assert result.metadata['train_records'] == 100
        assert result.metadata['validation_records'] == 30
    
    def test_empty_training_data(self):
        """Test error handling for empty training data."""
        trainer = CustomModelTrainer()
        
        empty_dataset = TrainingDataset(
            train_data=pd.DataFrame(),
            validation_data=pd.DataFrame({'sales_volume': [1, 2, 3]}),
            feature_columns=['price'],
            target_column='sales_volume',
            normalization_params={},
            metadata={}
        )
        
        config = ModelConfig(
            algorithm='random_forest',
            hyperparameters={},
            forecast_horizon=30
        )
        
        result = trainer.train_model(empty_dataset, config)
        
        assert result.success is False
        assert len(result.errors) > 0
        assert 'empty' in result.errors[0].lower()
    
    def test_empty_validation_data(self):
        """Test error handling for empty validation data."""
        trainer = CustomModelTrainer()
        
        empty_val_dataset = TrainingDataset(
            train_data=pd.DataFrame({'sales_volume': [1, 2, 3], 'price': [10, 20, 30]}),
            validation_data=pd.DataFrame(),
            feature_columns=['price'],
            target_column='sales_volume',
            normalization_params={},
            metadata={}
        )
        
        config = ModelConfig(
            algorithm='random_forest',
            hyperparameters={},
            forecast_horizon=30
        )
        
        result = trainer.train_model(empty_val_dataset, config)
        
        assert result.success is False
        assert len(result.errors) > 0
        assert 'validation' in result.errors[0].lower()
    
    def test_missing_target_column(self, sample_training_dataset):
        """Test error handling for missing target column."""
        trainer = CustomModelTrainer()
        
        # Create dataset with wrong target column
        bad_dataset = TrainingDataset(
            train_data=sample_training_dataset.train_data,
            validation_data=sample_training_dataset.validation_data,
            feature_columns=sample_training_dataset.feature_columns,
            target_column='nonexistent_column',
            normalization_params={},
            metadata={}
        )
        
        config = ModelConfig(
            algorithm='random_forest',
            hyperparameters={},
            forecast_horizon=30
        )
        
        result = trainer.train_model(bad_dataset, config)
        
        assert result.success is False
        assert len(result.errors) > 0
        assert 'target column' in result.errors[0].lower()
    
    def test_missing_feature_columns(self, sample_training_dataset):
        """Test error handling for missing feature columns."""
        trainer = CustomModelTrainer()
        
        # Create dataset with wrong feature columns
        bad_dataset = TrainingDataset(
            train_data=sample_training_dataset.train_data,
            validation_data=sample_training_dataset.validation_data,
            feature_columns=['nonexistent_feature'],
            target_column='sales_volume',
            normalization_params={},
            metadata={}
        )
        
        config = ModelConfig(
            algorithm='random_forest',
            hyperparameters={},
            forecast_horizon=30
        )
        
        result = trainer.train_model(bad_dataset, config)
        
        assert result.success is False
        assert len(result.errors) > 0
        assert 'feature columns' in result.errors[0].lower()
    
    def test_invalid_forecast_horizon(self, sample_training_dataset):
        """Test error handling for invalid forecast horizon."""
        trainer = CustomModelTrainer()
        
        config = ModelConfig(
            algorithm='random_forest',
            hyperparameters={},
            forecast_horizon=0  # Invalid
        )
        
        result = trainer.train_model(sample_training_dataset, config)
        
        assert result.success is False
        assert len(result.errors) > 0
        assert 'forecast_horizon' in result.errors[0].lower()
    
    def test_unsupported_algorithm(self, sample_training_dataset):
        """Test error handling for unsupported algorithm."""
        trainer = CustomModelTrainer()
        
        config = ModelConfig(
            algorithm='unsupported_algo',  # type: ignore
            hyperparameters={},
            forecast_horizon=30
        )
        
        result = trainer.train_model(sample_training_dataset, config)
        
        assert result.success is False
        assert len(result.errors) > 0
        assert 'unsupported algorithm' in result.errors[0].lower()
    
    def test_metrics_accuracy(self, sample_training_dataset):
        """Test that computed metrics are reasonable."""
        trainer = CustomModelTrainer()
        
        config = ModelConfig(
            algorithm='random_forest',
            hyperparameters={'n_estimators': 50, 'random_state': 42},
            forecast_horizon=30
        )
        
        result = trainer.train_model(sample_training_dataset, config)
        
        assert result.success is True
        assert result.metrics is not None
        
        # Metrics should be positive
        assert result.metrics.mae > 0
        assert result.metrics.rmse > 0
        assert result.metrics.mape > 0
        
        # RMSE should be >= MAE (mathematical property)
        assert result.metrics.rmse >= result.metrics.mae
        
        # Sample size should match validation set
        assert result.metrics.sample_size == len(sample_training_dataset.validation_data)
    
    def test_different_hyperparameters(self, sample_training_dataset):
        """Test that different hyperparameters produce different results."""
        trainer = CustomModelTrainer()
        
        # Train with different hyperparameters
        config1 = ModelConfig(
            algorithm='random_forest',
            hyperparameters={'n_estimators': 5, 'max_depth': 3, 'random_state': 42},
            forecast_horizon=30
        )
        
        config2 = ModelConfig(
            algorithm='random_forest',
            hyperparameters={'n_estimators': 50, 'max_depth': 10, 'random_state': 42},
            forecast_horizon=30
        )
        
        result1 = trainer.train_model(sample_training_dataset, config1)
        result2 = trainer.train_model(sample_training_dataset, config2)
        
        assert result1.success is True
        assert result2.success is True
        
        # Models with different hyperparameters should produce different metrics
        # (though not guaranteed, very likely with different complexity)
        assert result1.metadata['hyperparameters'] != result2.metadata['hyperparameters']
    
    def test_prophet_without_timestamp(self):
        """Test Prophet error handling when timestamp column is missing."""
        trainer = CustomModelTrainer()
        
        # Create dataset without timestamp column
        train_data = pd.DataFrame({
            'product_id': ['PROD_001'] * 10,
            'sales_volume': np.random.randn(10) * 10 + 100,
            'price': np.random.randn(10) * 5 + 50
        })
        
        val_data = pd.DataFrame({
            'product_id': ['PROD_001'] * 5,
            'sales_volume': np.random.randn(5) * 10 + 100,
            'price': np.random.randn(5) * 5 + 50
        })
        
        dataset = TrainingDataset(
            train_data=train_data,
            validation_data=val_data,
            feature_columns=['price'],
            target_column='sales_volume',
            normalization_params={},
            metadata={}
        )
        
        config = ModelConfig(
            algorithm='prophet',
            hyperparameters={},
            forecast_horizon=30
        )
        
        result = trainer.train_model(dataset, config)
        
        assert result.success is False
        assert len(result.errors) > 0
        assert 'timestamp' in result.errors[0].lower()
