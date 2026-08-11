"""
Amazon Forecast integration module for the Demand Forecasting System.

This module provides functionality to integrate with Amazon Forecast service,
including dataset creation, data import, and predictor training.

Requirements: 3.1, 3.2
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import time
import logging
from datetime import datetime, timezone
import pandas as pd
import boto3
from botocore.exceptions import ClientError
from io import StringIO

from config.settings import settings
from src.utils.logging_config import logger


@dataclass
class ForecastDatasetConfig:
    """
    Configuration for Amazon Forecast dataset creation.
    
    Attributes:
        dataset_name: Name for the Forecast dataset
        dataset_group_name: Name for the Forecast dataset group
        domain: Forecast domain (e.g., 'CUSTOM', 'RETAIL', 'INVENTORY_PLANNING')
        dataset_frequency: Data frequency (e.g., 'D' for daily, 'H' for hourly)
        timestamp_format: Format for timestamp column (e.g., 'yyyy-MM-dd HH:mm:ss')
    """
    dataset_name: str
    dataset_group_name: str
    domain: str = 'CUSTOM'
    dataset_frequency: str = 'D'
    timestamp_format: str = 'yyyy-MM-dd HH:mm:ss'


@dataclass
class ForecastImportResult:
    """
    Result of Amazon Forecast dataset import operation.
    
    Attributes:
        success: Whether import succeeded
        dataset_group_arn: ARN of the dataset group
        dataset_arn: ARN of the dataset
        import_job_arn: ARN of the import job
        s3_path: S3 path where data was uploaded
        record_count: Number of records imported
        errors: List of error messages (if failed)
        warnings: List of warning messages
    """
    success: bool
    dataset_group_arn: Optional[str] = None
    dataset_arn: Optional[str] = None
    import_job_arn: Optional[str] = None
    s3_path: Optional[str] = None
    record_count: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class AmazonForecastIntegration:
    """
    Integration with Amazon Forecast service for benchmark model training.
    
    Responsibilities:
    - Upload historical data to S3 in Forecast-compatible CSV format
    - Create Forecast dataset group and dataset
    - Import historical data into Forecast dataset
    - Configure target time series (sales_volume) and related features
    
    Amazon Forecast requires data in specific CSV format with columns:
    - timestamp: Date/time in specified format
    - target_value: The value to forecast (sales_volume)
    - item_id: Unique identifier for the item (product_id)
    - Related features: price, is_holiday, seasonality features
    """
    
    def __init__(
        self,
        forecast_client=None,
        s3_client=None,
        role_arn: Optional[str] = None
    ):
        """
        Initialize Amazon Forecast integration.
        
        Args:
            forecast_client: Optional boto3 Forecast client (for testing)
            s3_client: Optional boto3 S3 client (for testing)
            role_arn: IAM role ARN for Forecast to access S3 (required for production)
        """
        self.forecast_client = forecast_client or boto3.client(
            'forecast',
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key
        )
        
        self.s3_client = s3_client or boto3.client(
            's3',
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key
        )
        
        self.role_arn = role_arn
        self.bucket_name = settings.s3_historical_datasets_bucket
        
        logger.info("AmazonForecastIntegration initialized")
    
    def import_dataset(
        self,
        data: pd.DataFrame,
        config: ForecastDatasetConfig,
        product_id: str
    ) -> ForecastImportResult:
        """
        Import historical data into Amazon Forecast.
        
        This method performs the complete dataset import workflow:
        1. Convert data to Forecast-compatible CSV format
        2. Upload CSV to S3
        3. Create dataset group (if not exists)
        4. Create dataset with schema
        5. Create and start dataset import job
        
        Args:
            data: Historical time-series data with required columns
            config: Configuration for dataset creation
            product_id: Product identifier for filtering/naming
            
        Returns:
            ForecastImportResult with ARNs and status
        """
        errors = []
        warnings = []
        
        try:
            logger.info(
                f"Starting Forecast dataset import for product_id={product_id}, "
                f"dataset_group={config.dataset_group_name}"
            )
            
            # Step 1: Validate and convert data to Forecast format
            logger.info("Step 1: Converting data to Forecast format")
            forecast_data = self._convert_to_forecast_format(data, product_id)
            
            if forecast_data.empty:
                return ForecastImportResult(
                    success=False,
                    errors=["No data to import after conversion"],
                    record_count=0
                )
            
            logger.info(f"Converted {len(forecast_data)} records to Forecast format")
            
            # Step 2: Upload data to S3
            logger.info("Step 2: Uploading data to S3")
            s3_path = self._upload_to_s3(
                forecast_data,
                config.dataset_name,
                product_id
            )
            
            logger.info(f"Data uploaded to {s3_path}")
            
            # Step 3: Create dataset group
            logger.info("Step 3: Creating dataset group")
            dataset_group_arn = self._create_dataset_group(
                config.dataset_group_name,
                config.domain
            )
            
            logger.info(f"Dataset group ARN: {dataset_group_arn}")
            
            # Step 4: Create dataset
            logger.info("Step 4: Creating dataset")
            dataset_arn = self._create_dataset(
                config.dataset_name,
                config.domain,
                config.dataset_frequency
            )
            
            logger.info(f"Dataset ARN: {dataset_arn}")
            
            # Step 5: Create dataset import job
            logger.info("Step 5: Creating dataset import job")
            import_job_arn = self._create_import_job(
                dataset_arn,
                s3_path,
                config.dataset_name,
                config.timestamp_format
            )
            
            logger.info(f"Import job ARN: {import_job_arn}")
            
            logger.info(
                f"Forecast dataset import initiated successfully. "
                f"Import job: {import_job_arn}"
            )
            
            return ForecastImportResult(
                success=True,
                dataset_group_arn=dataset_group_arn,
                dataset_arn=dataset_arn,
                import_job_arn=import_job_arn,
                s3_path=s3_path,
                record_count=len(forecast_data),
                warnings=warnings
            )
            
        except Exception as e:
            error_msg = f"Forecast dataset import failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            return ForecastImportResult(
                success=False,
                errors=[error_msg],
                warnings=warnings,
                record_count=0
            )
    
    def _convert_to_forecast_format(
        self,
        data: pd.DataFrame,
        product_id: str
    ) -> pd.DataFrame:
        """
        Convert historical data to Amazon Forecast CSV format.
        
        Forecast requires specific column names and format:
        - timestamp: Date/time column
        - target_value: The value to forecast (sales_volume)
        - item_id: Unique identifier (product_id)
        
        Related time series features (optional):
        - price: Numeric price value
        - is_holiday: Binary holiday indicator (0/1)
        - day_of_week: Day of week (0-6)
        - month: Month (1-12)
        - quarter: Quarter (1-4)
        
        Args:
            data: Historical data with standard column names
            product_id: Product identifier
            
        Returns:
            DataFrame in Forecast-compatible format
        """
        # Create a copy to avoid modifying original
        forecast_data = data.copy()
        
        # Validate required columns
        required_columns = ['timestamp', 'sales_volume', 'product_id']
        missing_columns = [col for col in required_columns if col not in forecast_data.columns]
        
        if missing_columns:
            raise ValueError(
                f"Missing required columns for Forecast: {missing_columns}"
            )
        
        # Filter for specific product if needed
        if 'product_id' in forecast_data.columns:
            forecast_data = forecast_data[forecast_data['product_id'] == product_id].copy()
        
        # Rename columns to Forecast format
        column_mapping = {
            'timestamp': 'timestamp',
            'sales_volume': 'target_value',
            'product_id': 'item_id'
        }
        
        # Select and rename core columns
        forecast_columns = ['timestamp', 'target_value', 'item_id']
        forecast_data = forecast_data.rename(columns=column_mapping)
        
        # Add related features if available
        related_features = []
        
        if 'price' in data.columns:
            forecast_data['price'] = data['price']
            related_features.append('price')
        
        if 'is_holiday' in data.columns:
            # Convert boolean to integer (0/1)
            forecast_data['is_holiday'] = data['is_holiday'].astype(int)
            related_features.append('is_holiday')
        
        if 'day_of_week' in data.columns:
            forecast_data['day_of_week'] = data['day_of_week']
            related_features.append('day_of_week')
        
        if 'month' in data.columns:
            forecast_data['month'] = data['month']
            related_features.append('month')
        
        if 'quarter' in data.columns:
            forecast_data['quarter'] = data['quarter']
            related_features.append('quarter')
        
        # Select final columns in correct order
        final_columns = forecast_columns + related_features
        forecast_data = forecast_data[final_columns]
        
        # Sort by timestamp
        forecast_data = forecast_data.sort_values('timestamp')
        
        # Remove duplicates
        forecast_data = forecast_data.drop_duplicates(subset=['timestamp', 'item_id'])
        
        logger.info(
            f"Converted data to Forecast format: {len(forecast_data)} records, "
            f"columns: {list(forecast_data.columns)}"
        )
        
        return forecast_data
    
    def _upload_to_s3(
        self,
        data: pd.DataFrame,
        dataset_name: str,
        product_id: str
    ) -> str:
        """
        Upload Forecast-formatted data to S3 as CSV.
        
        Args:
            data: DataFrame in Forecast format
            dataset_name: Name of the dataset
            product_id: Product identifier
            
        Returns:
            S3 URI (s3://bucket/key)
        """
        # Generate S3 key
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        s3_key = f"forecast_datasets/{product_id}/{dataset_name}_{timestamp}.csv"
        
        # Convert DataFrame to CSV string
        csv_buffer = StringIO()
        data.to_csv(csv_buffer, index=False, date_format='%Y-%m-%d %H:%M:%S')
        csv_content = csv_buffer.getvalue()
        
        # Upload to S3
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=csv_content.encode('utf-8'),
                ContentType='text/csv'
            )
            
            s3_path = f"s3://{self.bucket_name}/{s3_key}"
            logger.info(f"Uploaded {len(data)} records to {s3_path}")
            
            return s3_path
            
        except ClientError as e:
            logger.error(f"Failed to upload data to S3: {str(e)}")
            raise
    
    def _create_dataset_group(
        self,
        dataset_group_name: str,
        domain: str
    ) -> str:
        """
        Create Amazon Forecast dataset group.
        
        A dataset group is a container for datasets and predictors.
        If the dataset group already exists, returns its ARN.
        
        Args:
            dataset_group_name: Name for the dataset group
            domain: Forecast domain (e.g., 'CUSTOM', 'RETAIL')
            
        Returns:
            Dataset group ARN
        """
        try:
            # Check if dataset group already exists
            try:
                response = self.forecast_client.describe_dataset_group(
                    DatasetGroupArn=self._build_arn('dataset-group', dataset_group_name)
                )
                
                dataset_group_arn = response['DatasetGroupArn']
                logger.info(f"Dataset group already exists: {dataset_group_arn}")
                
                return dataset_group_arn
                
            except ClientError as e:
                if e.response['Error']['Code'] != 'ResourceNotFoundException':
                    raise
                
                # Dataset group doesn't exist, create it
                logger.info(f"Creating new dataset group: {dataset_group_name}")
                
                response = self.forecast_client.create_dataset_group(
                    DatasetGroupName=dataset_group_name,
                    Domain=domain
                )
                
                dataset_group_arn = response['DatasetGroupArn']
                logger.info(f"Created dataset group: {dataset_group_arn}")
                
                return dataset_group_arn
                
        except ClientError as e:
            logger.error(f"Failed to create dataset group: {str(e)}")
            raise
    
    def _create_dataset(
        self,
        dataset_name: str,
        domain: str,
        dataset_frequency: str
    ) -> str:
        """
        Create Amazon Forecast dataset with schema.
        
        Defines the schema for target time series data including:
        - timestamp: Required timestamp field
        - target_value: Required target field (sales_volume)
        - item_id: Required item identifier (product_id)
        - Related features: price, is_holiday, seasonality features
        
        Args:
            dataset_name: Name for the dataset
            domain: Forecast domain
            dataset_frequency: Data frequency (e.g., 'D' for daily)
            
        Returns:
            Dataset ARN
        """
        try:
            # Check if dataset already exists
            try:
                response = self.forecast_client.describe_dataset(
                    DatasetArn=self._build_arn('dataset', dataset_name)
                )
                
                dataset_arn = response['DatasetArn']
                logger.info(f"Dataset already exists: {dataset_arn}")
                
                return dataset_arn
                
            except ClientError as e:
                if e.response['Error']['Code'] != 'ResourceNotFoundException':
                    raise
                
                # Dataset doesn't exist, create it
                logger.info(f"Creating new dataset: {dataset_name}")
                
                # Define schema for target time series
                schema = {
                    'Attributes': [
                        {
                            'AttributeName': 'timestamp',
                            'AttributeType': 'timestamp'
                        },
                        {
                            'AttributeName': 'target_value',
                            'AttributeType': 'float'
                        },
                        {
                            'AttributeName': 'item_id',
                            'AttributeType': 'string'
                        },
                        {
                            'AttributeName': 'price',
                            'AttributeType': 'float'
                        },
                        {
                            'AttributeName': 'is_holiday',
                            'AttributeType': 'integer'
                        },
                        {
                            'AttributeName': 'day_of_week',
                            'AttributeType': 'integer'
                        },
                        {
                            'AttributeName': 'month',
                            'AttributeType': 'integer'
                        },
                        {
                            'AttributeName': 'quarter',
                            'AttributeType': 'integer'
                        }
                    ]
                }
                
                response = self.forecast_client.create_dataset(
                    DatasetName=dataset_name,
                    Domain=domain,
                    DatasetType='TARGET_TIME_SERIES',
                    DataFrequency=dataset_frequency,
                    Schema=schema
                )
                
                dataset_arn = response['DatasetArn']
                logger.info(f"Created dataset: {dataset_arn}")
                
                return dataset_arn
                
        except ClientError as e:
            logger.error(f"Failed to create dataset: {str(e)}")
            raise
    
    def _create_import_job(
        self,
        dataset_arn: str,
        s3_path: str,
        dataset_name: str,
        timestamp_format: str
    ) -> str:
        """
        Create Amazon Forecast dataset import job.
        
        The import job loads data from S3 into the Forecast dataset.
        This is an asynchronous operation that may take several minutes.
        
        Args:
            dataset_arn: ARN of the dataset to import into
            s3_path: S3 URI of the CSV file
            dataset_name: Name of the dataset (for job naming)
            timestamp_format: Format for timestamp column
            
        Returns:
            Import job ARN
        """
        try:
            # Generate unique import job name
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            import_job_name = f"{dataset_name}_import_{timestamp}"
            
            # Validate role ARN is provided
            if not self.role_arn:
                raise ValueError(
                    "IAM role ARN is required for Forecast to access S3. "
                    "Provide role_arn when initializing AmazonForecastIntegration."
                )
            
            logger.info(f"Creating import job: {import_job_name}")
            
            response = self.forecast_client.create_dataset_import_job(
                DatasetImportJobName=import_job_name,
                DatasetArn=dataset_arn,
                DataSource={
                    'S3Config': {
                        'Path': s3_path,
                        'RoleArn': self.role_arn
                    }
                },
                TimestampFormat=timestamp_format
            )
            
            import_job_arn = response['DatasetImportJobArn']
            logger.info(f"Created import job: {import_job_arn}")
            
            return import_job_arn
            
        except ClientError as e:
            logger.error(f"Failed to create import job: {str(e)}")
            raise
    
    def _build_arn(self, resource_type: str, resource_name: str) -> str:
        """
        Build Amazon Forecast ARN for a resource.
        
        ARN format: arn:aws:forecast:region:account-id:resource-type/resource-name
        
        Args:
            resource_type: Type of resource (e.g., 'dataset', 'dataset-group')
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
    
    def wait_for_import_completion(
        self,
        import_job_arn: str,
        max_wait_seconds: int = 3600,
        poll_interval: int = 60
    ) -> bool:
        """
        Wait for dataset import job to complete.
        
        Polls the import job status until it reaches a terminal state
        (ACTIVE or CREATE_FAILED).
        
        Args:
            import_job_arn: ARN of the import job
            max_wait_seconds: Maximum time to wait in seconds
            poll_interval: Seconds between status checks
            
        Returns:
            True if import succeeded, False if failed
        """
        start_time = time.time()
        
        logger.info(f"Waiting for import job to complete: {import_job_arn}")
        
        while True:
            try:
                response = self.forecast_client.describe_dataset_import_job(
                    DatasetImportJobArn=import_job_arn
                )
                
                status = response['Status']
                
                logger.info(f"Import job status: {status}")
                
                if status == 'ACTIVE':
                    logger.info("Import job completed successfully")
                    return True
                
                elif status == 'CREATE_FAILED':
                    error_message = response.get('Message', 'Unknown error')
                    logger.error(f"Import job failed: {error_message}")
                    return False
                
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > max_wait_seconds:
                    logger.error(
                        f"Import job did not complete within {max_wait_seconds}s. "
                        f"Current status: {status}"
                    )
                    return False
                
                # Wait before next poll
                time.sleep(poll_interval)
                
            except ClientError as e:
                logger.error(f"Error checking import job status: {str(e)}")
                return False
