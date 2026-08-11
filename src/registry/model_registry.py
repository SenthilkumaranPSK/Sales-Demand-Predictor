"""Model Registry for storing and retrieving trained model artifacts and metadata."""
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any, Literal
import boto3
from botocore.exceptions import ClientError
import json
from sqlalchemy import text
from src.registry.database import db_manager
from config.settings import settings
from src.utils.logging_config import logger


@dataclass
class ModelMetadata:
    """Metadata for a registered model."""
    model_id: str
    product_id: str
    model_type: Literal["custom", "forecast"]
    version: int
    artifact_path: str  # S3 path
    training_dataset_id: str
    mae: float
    rmse: float
    mape: float
    hyperparameters: Dict[str, Any]
    created_at: datetime
    forecast_horizon: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for JSON serialization."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data


class ModelRegistry:
    """
    Manages storage and retrieval of trained model artifacts and metadata.
    
    Model artifacts are stored in S3 with versioning enabled.
    Model metadata is stored in RDS PostgreSQL with indexing for efficient queries.
    """
    
    def __init__(self):
        """Initialize Model Registry with S3 and database connections."""
        self.s3_client = boto3.client(
            's3',
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key
        )
        self.bucket_name = settings.s3_model_artifacts_bucket
        logger.info(f"ModelRegistry initialized with bucket: {self.bucket_name}")
    
    def register_model(
        self,
        model_artifact: bytes,
        metadata: ModelMetadata
    ) -> str:
        """
        Store model artifact and metadata.
        
        Args:
            model_artifact: Serialized model (pickle or joblib)
            metadata: Version, metrics, training config, dataset reference
            
        Returns:
            model_id: Unique identifier for registered model
            
        Raises:
            ValueError: If metadata is invalid
            RuntimeError: If storage operation fails
        """
        try:
            # Validate metadata
            if not metadata.model_id:
                raise ValueError("model_id cannot be empty")
            if not metadata.product_id:
                raise ValueError("product_id cannot be empty")
            if metadata.model_type not in ["custom", "forecast"]:
                raise ValueError(f"Invalid model_type: {metadata.model_type}")
            if metadata.version < 1:
                raise ValueError(f"Invalid version: {metadata.version}")
            
            # Store model artifact to S3
            s3_key = f"{metadata.product_id}/{metadata.model_type}/v{metadata.version}/{metadata.model_id}"
            
            logger.info(f"Uploading model artifact to S3: {s3_key}")
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=model_artifact,
                Metadata={
                    'model_id': metadata.model_id,
                    'product_id': metadata.product_id,
                    'model_type': metadata.model_type,
                    'version': str(metadata.version)
                }
            )
            
            # Update metadata with S3 path
            metadata.artifact_path = f"s3://{self.bucket_name}/{s3_key}"
            
            # Store metadata in database
            with db_manager.get_session() as session:
                insert_query = text("""
                    INSERT INTO models (
                        model_id, product_id, model_type, version,
                        artifact_s3_path, training_dataset_id,
                        mae, rmse, mape, hyperparameters, forecast_horizon
                    ) VALUES (
                        :model_id, :product_id, :model_type, :version,
                        :artifact_s3_path, :training_dataset_id,
                        :mae, :rmse, :mape, :hyperparameters, :forecast_horizon
                    )
                    ON CONFLICT (product_id, model_type, version)
                    DO UPDATE SET
                        artifact_s3_path = EXCLUDED.artifact_s3_path,
                        training_dataset_id = EXCLUDED.training_dataset_id,
                        mae = EXCLUDED.mae,
                        rmse = EXCLUDED.rmse,
                        mape = EXCLUDED.mape,
                        hyperparameters = EXCLUDED.hyperparameters,
                        forecast_horizon = EXCLUDED.forecast_horizon
                """)
                
                session.execute(insert_query, {
                    'model_id': metadata.model_id,
                    'product_id': metadata.product_id,
                    'model_type': metadata.model_type,
                    'version': metadata.version,
                    'artifact_s3_path': metadata.artifact_path,
                    'training_dataset_id': metadata.training_dataset_id,
                    'mae': metadata.mae,
                    'rmse': metadata.rmse,
                    'mape': metadata.mape,
                    'hyperparameters': json.dumps(metadata.hyperparameters),
                    'forecast_horizon': metadata.forecast_horizon
                })
            
            logger.info(f"Model registered successfully: {metadata.model_id}")
            return metadata.model_id
            
        except ValueError:
            # Re-raise validation errors without wrapping
            raise
        except ClientError as e:
            error_msg = f"S3 error while registering model: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f"Error registering model: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def get_model(self, model_id: str) -> Tuple[bytes, ModelMetadata]:
        """
        Retrieve model artifact and metadata by ID.
        
        Args:
            model_id: Unique identifier for the model
            
        Returns:
            Tuple of (model_artifact, metadata)
            
        Raises:
            ValueError: If model_id is not found
            RuntimeError: If retrieval operation fails
        """
        try:
            # Retrieve metadata from database
            with db_manager.get_session() as session:
                query = text("""
                    SELECT 
                        model_id, product_id, model_type, version,
                        artifact_s3_path, training_dataset_id,
                        mae, rmse, mape, hyperparameters, 
                        created_at, forecast_horizon
                    FROM models
                    WHERE model_id = :model_id
                """)
                
                result = session.execute(query, {'model_id': model_id}).fetchone()
                
                if not result:
                    raise ValueError(f"Model not found: {model_id}")
                
                # Parse result into ModelMetadata
                metadata = ModelMetadata(
                    model_id=result[0],
                    product_id=result[1],
                    model_type=result[2],
                    version=result[3],
                    artifact_path=result[4],
                    training_dataset_id=result[5],
                    mae=float(result[6]),
                    rmse=float(result[7]),
                    mape=float(result[8]),
                    hyperparameters=json.loads(result[9]) if result[9] else {},
                    created_at=result[10],
                    forecast_horizon=result[11]
                )
            
            # Retrieve model artifact from S3
            s3_path = metadata.artifact_path
            if not s3_path.startswith('s3://'):
                raise ValueError(f"Invalid S3 path: {s3_path}")
            
            # Parse S3 path
            path_parts = s3_path.replace('s3://', '').split('/', 1)
            bucket = path_parts[0]
            key = path_parts[1]
            
            logger.info(f"Downloading model artifact from S3: {key}")
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            model_artifact = response['Body'].read()
            
            logger.info(f"Model retrieved successfully: {model_id}")
            return model_artifact, metadata
            
        except ClientError as e:
            error_msg = f"S3 error while retrieving model: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        except ValueError:
            raise
        except Exception as e:
            error_msg = f"Error retrieving model: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def list_models(
        self,
        product_id: Optional[str] = None,
        model_type: Optional[Literal["custom", "forecast"]] = None
    ) -> List[ModelMetadata]:
        """
        List available models with optional filtering.
        
        Args:
            product_id: Optional filter by product ID
            model_type: Optional filter by model type ('custom' or 'forecast')
            
        Returns:
            List of ModelMetadata objects
        """
        try:
            # Build query with optional filters
            query_parts = ["""
                SELECT 
                    model_id, product_id, model_type, version,
                    artifact_s3_path, training_dataset_id,
                    mae, rmse, mape, hyperparameters,
                    created_at, forecast_horizon
                FROM models
                WHERE 1=1
            """]
            
            params = {}
            
            if product_id:
                query_parts.append("AND product_id = :product_id")
                params['product_id'] = product_id
            
            if model_type:
                query_parts.append("AND model_type = :model_type")
                params['model_type'] = model_type
            
            query_parts.append("ORDER BY created_at DESC")
            
            query = text(' '.join(query_parts))
            
            # Execute query
            with db_manager.get_session() as session:
                results = session.execute(query, params).fetchall()
            
            # Parse results into ModelMetadata objects
            models = []
            for row in results:
                metadata = ModelMetadata(
                    model_id=row[0],
                    product_id=row[1],
                    model_type=row[2],
                    version=row[3],
                    artifact_path=row[4],
                    training_dataset_id=row[5],
                    mae=float(row[6]),
                    rmse=float(row[7]),
                    mape=float(row[8]),
                    hyperparameters=json.loads(row[9]) if row[9] else {},
                    created_at=row[10],
                    forecast_horizon=row[11]
                )
                models.append(metadata)
            
            logger.info(f"Listed {len(models)} models (product_id={product_id}, model_type={model_type})")
            return models
            
        except Exception as e:
            error_msg = f"Error listing models: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def get_latest_model(
        self,
        product_id: str,
        model_type: Literal["custom", "forecast"]
    ) -> Tuple[str, ModelMetadata]:
        """
        Get most recent model version for a product.
        
        Args:
            product_id: Product identifier
            model_type: Model type ('custom' or 'forecast')
            
        Returns:
            Tuple of (model_id, metadata) for the latest version
            
        Raises:
            ValueError: If no model found for the given product and type
        """
        try:
            query = text("""
                SELECT 
                    model_id, product_id, model_type, version,
                    artifact_s3_path, training_dataset_id,
                    mae, rmse, mape, hyperparameters,
                    created_at, forecast_horizon
                FROM models
                WHERE product_id = :product_id AND model_type = :model_type
                ORDER BY version DESC, created_at DESC
                LIMIT 1
            """)
            
            with db_manager.get_session() as session:
                result = session.execute(query, {
                    'product_id': product_id,
                    'model_type': model_type
                }).fetchone()
                
                if not result:
                    raise ValueError(
                        f"No model found for product_id={product_id}, model_type={model_type}"
                    )
                
                # Parse result into ModelMetadata
                metadata = ModelMetadata(
                    model_id=result[0],
                    product_id=result[1],
                    model_type=result[2],
                    version=result[3],
                    artifact_path=result[4],
                    training_dataset_id=result[5],
                    mae=float(result[6]),
                    rmse=float(result[7]),
                    mape=float(result[8]),
                    hyperparameters=json.loads(result[9]) if result[9] else {},
                    created_at=result[10],
                    forecast_horizon=result[11]
                )
            
            logger.info(
                f"Retrieved latest model: {metadata.model_id} "
                f"(product_id={product_id}, model_type={model_type}, version={metadata.version})"
            )
            return metadata.model_id, metadata
            
        except ValueError:
            raise
        except Exception as e:
            error_msg = f"Error getting latest model: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e


# Global model registry instance
model_registry = ModelRegistry()
