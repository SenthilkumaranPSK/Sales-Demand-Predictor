"""
Training data preparation module for the Demand Forecasting System.

This module provides functionality to prepare training data by loading historical
datasets from S3, applying feature engineering pipelines, and splitting into
train/validation sets.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import pandas as pd
import numpy as np
import boto3
from io import BytesIO
import logging

from src.features.seasonality import extract_seasonality_features
from src.features.preprocessing import FeaturePreprocessor, PreprocessingResult
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class TrainingDataset:
    """Training dataset with train/validation split."""
    train_data: pd.DataFrame
    validation_data: pd.DataFrame
    feature_columns: List[str]
    target_column: str
    normalization_params: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass
class DataPreparationResult:
    """Result of data preparation operation."""
    success: bool
    dataset: Optional[TrainingDataset] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class TrainingDataPreparation:
    """
    Prepares training data for model training.
    
    Responsibilities:
    - Load historical dataset from S3
    - Apply feature engineering pipeline (seasonality, preprocessing)
    - Split data into train/validation sets (80/20)
    - Create training dataset with features (sales, price, holidays, seasonality)
    """
    
    def __init__(self, s3_client=None, preprocessor=None):
        """
        Initialize the training data preparation service.
        
        Args:
            s3_client: Optional boto3 S3 client (for testing)
            preprocessor: Optional FeaturePreprocessor instance (for testing)
        """
        self.s3_client = s3_client or boto3.client(
            's3',
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key
        )
        self.preprocessor = preprocessor or FeaturePreprocessor()
        self.bucket_name = settings.s3_historical_datasets_bucket
    
    def prepare_training_data(
        self,
        dataset_path: str,
        product_id: Optional[str] = None,
        train_split: float = 0.8,
        target_column: str = 'sales_volume'
    ) -> DataPreparationResult:
        """
        Prepare training data from historical dataset.
        
        Args:
            dataset_path: S3 path to historical dataset (s3://bucket/path or path within bucket)
            product_id: Optional product ID to filter data
            train_split: Fraction of data for training (default: 0.8)
            target_column: Name of target column (default: 'sales_volume')
            
        Returns:
            DataPreparationResult with training dataset or errors
        """
        errors = []
        warnings = []
        
        try:
            # Validate train_split
            if not 0 < train_split < 1:
                return DataPreparationResult(
                    success=False,
                    errors=[f"train_split must be between 0 and 1, got {train_split}"]
                )
            
            # Load historical dataset from S3
            logger.info(f"Loading historical dataset from {dataset_path}")
            df = self._load_from_s3(dataset_path, product_id)
            
            if df.empty:
                return DataPreparationResult(
                    success=False,
                    errors=["Loaded dataset is empty"]
                )
            
            logger.info(f"Loaded {len(df)} records from S3")
            
            # Validate required columns
            validation_errors = self._validate_required_columns(df, target_column)
            if validation_errors:
                return DataPreparationResult(
                    success=False,
                    errors=validation_errors
                )
            
            # Apply feature engineering pipeline
            logger.info("Applying feature engineering pipeline")
            df_engineered = self._apply_feature_engineering(df)
            
            # Preprocess features
            logger.info("Preprocessing features")
            preprocessing_result = self.preprocessor.preprocess_features(
                df_engineered,
                normalize=True,
                fit_normalization=True
            )
            
            if not preprocessing_result.is_valid:
                return DataPreparationResult(
                    success=False,
                    errors=preprocessing_result.errors
                )
            
            df_processed = preprocessing_result.data
            
            # Sort by timestamp for proper train/validation split
            if 'timestamp' in df_processed.columns:
                df_processed = df_processed.sort_values('timestamp')
            
            # Split into train/validation sets
            logger.info(f"Splitting data with train_split={train_split}")
            train_data, validation_data = self._train_validation_split(
                df_processed,
                train_split
            )
            
            # Identify feature columns
            feature_columns = self._identify_feature_columns(df_processed, target_column)
            
            # Create training dataset
            dataset = TrainingDataset(
                train_data=train_data,
                validation_data=validation_data,
                feature_columns=feature_columns,
                target_column=target_column,
                normalization_params=preprocessing_result.normalization_params,
                metadata={
                    'total_records': len(df_processed),
                    'train_records': len(train_data),
                    'validation_records': len(validation_data),
                    'product_id': product_id,
                    'dataset_path': dataset_path,
                    'train_split': train_split
                }
            )
            
            logger.info(
                f"Successfully prepared training data: "
                f"{len(train_data)} train, {len(validation_data)} validation records"
            )
            
            return DataPreparationResult(
                success=True,
                dataset=dataset,
                warnings=warnings
            )
            
        except Exception as e:
            error_msg = f"Data preparation failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return DataPreparationResult(
                success=False,
                errors=[error_msg]
            )
    
    def _load_from_s3(
        self,
        dataset_path: str,
        product_id: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load historical dataset from S3.
        
        Args:
            dataset_path: S3 path (s3://bucket/path or path within bucket)
            product_id: Optional product ID to filter data
            
        Returns:
            DataFrame with historical data
            
        Raises:
            Exception: If S3 loading fails
        """
        # Parse S3 path
        if dataset_path.startswith('s3://'):
            # Extract bucket and key from full S3 path
            path_parts = dataset_path.replace('s3://', '').split('/', 1)
            bucket = path_parts[0]
            key_prefix = path_parts[1] if len(path_parts) > 1 else ''
        else:
            # Use configured bucket
            bucket = self.bucket_name
            key_prefix = dataset_path
        
        # List all parquet files under the path
        logger.info(f"Listing objects in s3://{bucket}/{key_prefix}")
        
        try:
            # Handle both file and directory paths
            if key_prefix.endswith('.parquet'):
                # Single file
                keys = [key_prefix]
            else:
                # Directory - list all parquet files
                response = self.s3_client.list_objects_v2(
                    Bucket=bucket,
                    Prefix=key_prefix
                )
                
                if 'Contents' not in response:
                    logger.warning(f"No objects found at s3://{bucket}/{key_prefix}")
                    return pd.DataFrame()
                
                keys = [
                    obj['Key'] for obj in response['Contents']
                    if obj['Key'].endswith('.parquet')
                ]
            
            if not keys:
                logger.warning(f"No parquet files found at s3://{bucket}/{key_prefix}")
                return pd.DataFrame()
            
            # Filter by product_id if specified
            if product_id:
                keys = [k for k in keys if f'product_id={product_id}' in k]
            
            if not keys:
                logger.warning(f"No files found for product_id={product_id}")
                return pd.DataFrame()
            
            # Load all parquet files and concatenate
            dataframes = []
            for key in keys:
                logger.debug(f"Loading s3://{bucket}/{key}")
                obj = self.s3_client.get_object(Bucket=bucket, Key=key)
                df = pd.read_parquet(BytesIO(obj['Body'].read()))
                dataframes.append(df)
            
            # Concatenate all dataframes
            if dataframes:
                result = pd.concat(dataframes, ignore_index=True)
                logger.info(f"Loaded {len(result)} records from {len(dataframes)} files")
                return result
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Failed to load data from S3: {str(e)}", exc_info=True)
            raise
    
    def _validate_required_columns(
        self,
        df: pd.DataFrame,
        target_column: str
    ) -> List[str]:
        """
        Validate that required columns are present.
        
        Args:
            df: DataFrame to validate
            target_column: Name of target column
            
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        required_columns = ['timestamp', 'product_id', target_column]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            errors.append(
                f"Missing required columns: {missing_columns}. "
                f"Available columns: {list(df.columns)}"
            )
        
        return errors
    
    def _apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering pipeline.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        result = df.copy()
        
        # Extract seasonality features if timestamp is present
        if 'timestamp' in result.columns:
            # Ensure timestamp is datetime
            if not pd.api.types.is_datetime64_any_dtype(result['timestamp']):
                result['timestamp'] = pd.to_datetime(result['timestamp'])
            
            # Extract seasonality features
            result = extract_seasonality_features(result, timestamp_column='timestamp')
            logger.info("Extracted seasonality features: day_of_week, month, quarter, season")
        
        return result
    
    def _train_validation_split(
        self,
        df: pd.DataFrame,
        train_split: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and validation sets.
        
        Uses temporal split: first train_split fraction for training,
        remaining for validation.
        
        Args:
            df: DataFrame to split (should be sorted by timestamp)
            train_split: Fraction of data for training
            
        Returns:
            Tuple of (train_data, validation_data)
        """
        split_index = int(len(df) * train_split)
        
        train_data = df.iloc[:split_index].copy()
        validation_data = df.iloc[split_index:].copy()
        
        return train_data, validation_data
    
    def _identify_feature_columns(
        self,
        df: pd.DataFrame,
        target_column: str
    ) -> List[str]:
        """
        Identify feature columns for model training.
        
        Args:
            df: DataFrame with all columns
            target_column: Name of target column
            
        Returns:
            List of feature column names
        """
        # Exclude non-feature columns
        exclude_columns = {
            target_column,
            'timestamp',
            'product_id',
            'season',  # Categorical, will be encoded separately if needed
            'year'  # Used for partitioning only
        }
        
        # Include all numeric and boolean columns except excluded ones
        feature_columns = []
        for col in df.columns:
            if col in exclude_columns:
                continue
            
            # Include numeric and boolean columns
            if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
                feature_columns.append(col)
        
        return feature_columns
    
    def load_dataset_for_inference(
        self,
        dataset_path: str,
        product_id: Optional[str] = None
    ) -> DataPreparationResult:
        """
        Load and prepare dataset for inference (no train/validation split).
        
        Args:
            dataset_path: S3 path to historical dataset
            product_id: Optional product ID to filter data
            
        Returns:
            DataPreparationResult with full dataset
        """
        errors = []
        
        try:
            # Load historical dataset from S3
            logger.info(f"Loading dataset for inference from {dataset_path}")
            df = self._load_from_s3(dataset_path, product_id)
            
            if df.empty:
                return DataPreparationResult(
                    success=False,
                    errors=["Loaded dataset is empty"]
                )
            
            # Apply feature engineering pipeline
            df_engineered = self._apply_feature_engineering(df)
            
            # Preprocess features (without fitting normalization)
            preprocessing_result = self.preprocessor.preprocess_features(
                df_engineered,
                normalize=True,
                fit_normalization=False
            )
            
            if not preprocessing_result.is_valid:
                return DataPreparationResult(
                    success=False,
                    errors=preprocessing_result.errors
                )
            
            df_processed = preprocessing_result.data
            
            # Create dataset without split
            dataset = TrainingDataset(
                train_data=df_processed,
                validation_data=pd.DataFrame(),
                feature_columns=self._identify_feature_columns(df_processed, 'sales_volume'),
                target_column='sales_volume',
                normalization_params=preprocessing_result.normalization_params,
                metadata={
                    'total_records': len(df_processed),
                    'product_id': product_id,
                    'dataset_path': dataset_path
                }
            )
            
            return DataPreparationResult(
                success=True,
                dataset=dataset
            )
            
        except Exception as e:
            error_msg = f"Dataset loading failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return DataPreparationResult(
                success=False,
                errors=[error_msg]
            )
