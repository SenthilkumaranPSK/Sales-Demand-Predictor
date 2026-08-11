"""Application configuration settings."""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # AWS Configuration
    aws_region: str = "us-east-1"
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    
    # S3 Buckets
    s3_historical_datasets_bucket: str = "demand-forecasting-historical-data"
    s3_model_artifacts_bucket: str = "demand-forecasting-model-artifacts"
    
    # Database Configuration
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "model_registry"
    db_user: str = "postgres"
    db_password: str = ""
    db_pool_size: int = 10
    db_max_overflow: int = 20
    
    # CloudWatch Logging
    cloudwatch_log_group: str = "/aws/demand-forecasting"
    cloudwatch_log_stream: str = "api"
    
    # API Configuration
    api_key_header: str = "X-API-Key"
    api_keys: Optional[str] = None  # Comma-separated list of valid API keys
    rate_limit_per_minute: int = 1000
    max_concurrent_requests: int = 100
    
    # Application Settings
    log_level: str = "INFO"
    environment: str = "development"
    api_version: str = "1.0.0"
    
    class Config:
        env_file = "config/.env"
        case_sensitive = False
    
    @property
    def database_url(self) -> str:
        """Construct database connection URL."""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"


# Global settings instance
settings = Settings()
