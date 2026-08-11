"""Setup AWS resources (S3 buckets) for the demand forecasting system."""
import boto3
from botocore.exceptions import ClientError
import sys
import os

# Add parent directory to path to import config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.settings import settings
from src.utils.logging_config import logger


def create_s3_bucket(bucket_name: str, region: str) -> bool:
    """
    Create an S3 bucket with versioning enabled.
    
    Args:
        bucket_name: Name of the bucket to create
        region: AWS region for the bucket
        
    Returns:
        True if bucket was created or already exists, False otherwise
    """
    s3_client = boto3.client('s3', region_name=region)
    
    try:
        # Check if bucket already exists
        try:
            s3_client.head_bucket(Bucket=bucket_name)
            logger.info(f"Bucket '{bucket_name}' already exists")
            return True
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code != '404':
                raise
        
        # Create bucket
        if region == 'us-east-1':
            # us-east-1 doesn't require LocationConstraint
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={'LocationConstraint': region}
            )
        
        logger.info(f"Bucket '{bucket_name}' created successfully")
        
        # Enable versioning for model artifacts bucket
        if 'model-artifacts' in bucket_name:
            s3_client.put_bucket_versioning(
                Bucket=bucket_name,
                VersioningConfiguration={'Status': 'Enabled'}
            )
            logger.info(f"Versioning enabled for bucket '{bucket_name}'")
        
        # Add lifecycle policy for historical datasets
        if 'historical-data' in bucket_name:
            lifecycle_policy = {
                'Rules': [
                    {
                        'Id': 'TransitionToIA',
                        'Status': 'Enabled',
                        'Transitions': [
                            {
                                'Days': 90,
                                'StorageClass': 'STANDARD_IA'
                            }
                        ],
                        'Filter': {'Prefix': ''}
                    }
                ]
            }
            s3_client.put_bucket_lifecycle_configuration(
                Bucket=bucket_name,
                LifecycleConfiguration=lifecycle_policy
            )
            logger.info(f"Lifecycle policy applied to bucket '{bucket_name}'")
        
        return True
        
    except ClientError as e:
        logger.error(f"Error creating bucket '{bucket_name}': {e}")
        return False


def create_cloudwatch_log_group(log_group_name: str, region: str) -> bool:
    """
    Create CloudWatch log group.
    
    Args:
        log_group_name: Name of the log group
        region: AWS region
        
    Returns:
        True if log group was created or already exists, False otherwise
    """
    logs_client = boto3.client('logs', region_name=region)
    
    try:
        # Check if log group exists
        try:
            logs_client.describe_log_groups(logGroupNamePrefix=log_group_name)
            logger.info(f"Log group '{log_group_name}' already exists")
            return True
        except ClientError:
            pass
        
        # Create log group
        logs_client.create_log_group(logGroupName=log_group_name)
        
        # Set retention policy (30 days)
        logs_client.put_retention_policy(
            logGroupName=log_group_name,
            retentionInDays=30
        )
        
        logger.info(f"Log group '{log_group_name}' created successfully")
        return True
        
    except ClientError as e:
        logger.error(f"Error creating log group '{log_group_name}': {e}")
        return False


def main():
    """Main setup function."""
    logger.info("Starting AWS resources setup...")
    
    region = settings.aws_region
    success = True
    
    # Create S3 buckets
    logger.info("Creating S3 buckets...")
    if not create_s3_bucket(settings.s3_historical_datasets_bucket, region):
        success = False
    
    if not create_s3_bucket(settings.s3_model_artifacts_bucket, region):
        success = False
    
    # Create CloudWatch log group
    logger.info("Creating CloudWatch log group...")
    if not create_cloudwatch_log_group(settings.cloudwatch_log_group, region):
        success = False
    
    if success:
        logger.info("AWS resources setup completed successfully")
        logger.info(f"Historical datasets bucket: {settings.s3_historical_datasets_bucket}")
        logger.info(f"Model artifacts bucket: {settings.s3_model_artifacts_bucket}")
        logger.info(f"CloudWatch log group: {settings.cloudwatch_log_group}")
    else:
        logger.error("AWS resources setup completed with errors")
        sys.exit(1)


if __name__ == "__main__":
    main()
