"""API request/response schemas."""
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Literal


class ConfidenceIntervalModel(BaseModel):
    """Confidence interval model for API responses."""
    level: str = Field(..., description="Confidence level (e.g., '50%', '80%', '90%')")
    lower: List[float] = Field(..., description="Lower bound values")
    upper: List[float] = Field(..., description="Upper bound values")

    class Config:
        json_schema_extra = {
            "example": {
                "level": "80%",
                "lower": [95.2, 98.1, 102.3],
                "upper": [105.8, 108.9, 112.7]
            }
        }


class ForecastRequest(BaseModel):
    """Request model for forecast generation."""
    product_id: str = Field(
        ...,
        description="Product identifier for which to generate forecast",
        min_length=1,
        max_length=255,
        example="PROD-12345"
    )
    forecast_horizon: int = Field(
        ...,
        description="Number of days to forecast (1-90)",
        ge=1,
        le=90,
        example=30
    )
    model_id: Optional[str] = Field(
        None,
        description="Specific model ID to use. If None, uses latest custom model for the product",
        example="model_prod12345_v1_20250115"
    )
    include_benchmark: bool = Field(
        False,
        description="Whether to include benchmark (Amazon Forecast) model predictions"
    )
    future_features: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional future feature values (holidays, prices)",
        example={
            "holidays": [False, False, True, False],
            "prices": [19.99, 19.99, 17.99, 19.99]
        }
    )

    class Config:
        json_schema_extra = {
            "example": {
                "product_id": "PROD-12345",
                "forecast_horizon": 30,
                "model_id": None,
                "include_benchmark": True,
                "future_features": {
                    "holidays": [False] * 30,
                    "prices": [19.99] * 30
                }
            }
        }


class ForecastResponse(BaseModel):
    """Response model for forecast generation."""
    forecast_id: str = Field(..., description="Unique identifier for this forecast")
    product_id: str = Field(..., description="Product identifier")
    model_id: str = Field(..., description="Model used for forecast")
    timestamps: List[datetime] = Field(..., description="Future timestamps for predictions")
    predictions: List[float] = Field(..., description="Point predictions (mean/median)")
    confidence_intervals: Dict[str, ConfidenceIntervalModel] = Field(
        ...,
        description="Confidence intervals at different levels (50%, 80%, 90%)"
    )
    benchmark: Optional[Dict[str, Any]] = Field(
        None,
        description="Benchmark model forecast (if include_benchmark=True)"
    )
    metadata: Dict[str, Any] = Field(
        ...,
        description="Additional forecast metadata (algorithm, metrics, etc.)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "forecast_id": "forecast_20250115_123456",
                "product_id": "PROD-12345",
                "model_id": "model_prod12345_v1_20250115",
                "timestamps": ["2025-01-16T00:00:00", "2025-01-17T00:00:00"],
                "predictions": [100.5, 105.2],
                "confidence_intervals": {
                    "50%": {
                        "level": "50%",
                        "lower": [95.2, 99.8],
                        "upper": [105.8, 110.6]
                    },
                    "80%": {
                        "level": "80%",
                        "lower": [90.3, 94.9],
                        "upper": [110.7, 115.5]
                    },
                    "90%": {
                        "level": "90%",
                        "lower": [85.4, 90.0],
                        "upper": [115.6, 120.4]
                    }
                },
                "benchmark": None,
                "metadata": {
                    "algorithm": "random_forest",
                    "model_version": 1,
                    "training_mae": 5.2,
                    "training_rmse": 7.8,
                    "training_mape": 4.5
                }
            }
        }


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: datetime
    environment: str
    version: str
    checks: Dict[str, Any]


class ModelListResponse(BaseModel):
    """Response model for listing models."""
    models: List[Dict[str, Any]] = Field(..., description="List of available models")
    total_count: int = Field(..., description="Total number of models")

    class Config:
        json_schema_extra = {
            "example": {
                "models": [
                    {
                        "model_id": "model_prod12345_v1_20250115",
                        "product_id": "PROD-12345",
                        "model_type": "custom",
                        "version": 1,
                        "mae": 5.2,
                        "rmse": 7.8,
                        "mape": 4.5,
                        "created_at": "2025-01-15T10:30:00Z"
                    }
                ],
                "total_count": 1
            }
        }


class ModelMetadataResponse(BaseModel):
    """Response model for model metadata."""
    model_id: str = Field(..., description="Unique model identifier")
    product_id: str = Field(..., description="Product identifier")
    model_type: str = Field(..., description="Model type (custom or forecast)")
    version: int = Field(..., description="Model version number")
    artifact_path: str = Field(..., description="S3 path to model artifact")
    training_dataset_id: str = Field(..., description="Training dataset identifier")
    mae: float = Field(..., description="Mean Absolute Error")
    rmse: float = Field(..., description="Root Mean Squared Error")
    mape: float = Field(..., description="Mean Absolute Percentage Error")
    hyperparameters: Dict[str, Any] = Field(..., description="Model hyperparameters")
    created_at: datetime = Field(..., description="Model creation timestamp")
    forecast_horizon: int = Field(..., description="Forecast horizon in days")

    class Config:
        json_schema_extra = {
            "example": {
                "model_id": "model_prod12345_v1_20250115",
                "product_id": "PROD-12345",
                "model_type": "custom",
                "version": 1,
                "artifact_path": "s3://models/PROD-12345/custom/v1/model_prod12345_v1_20250115",
                "training_dataset_id": "dataset_20250115_123456",
                "mae": 5.2,
                "rmse": 7.8,
                "mape": 4.5,
                "hyperparameters": {
                    "algorithm": "random_forest",
                    "n_estimators": 100,
                    "max_depth": 10
                },
                "created_at": "2025-01-15T10:30:00Z",
                "forecast_horizon": 30
            }
        }


class DataIngestionRequest(BaseModel):
    """Request model for data ingestion."""
    data: Union[List[Dict[str, Any]], str] = Field(
        ...,
        description="Historical sales data as list of records or CSV/JSON string"
    )
    format: Literal["csv", "json", "auto"] = Field(
        "auto",
        description="Data format (csv, json, or auto for detection)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "data": [
                    {
                        "timestamp": "2025-01-01T00:00:00",
                        "product_id": "PROD-12345",
                        "sales_volume": 100.5,
                        "price": 19.99,
                        "is_holiday": False,
                        "day_of_week": 0,
                        "month": 1,
                        "quarter": 1
                    }
                ],
                "format": "auto"
            }
        }


class DataIngestionResponse(BaseModel):
    """Response model for data ingestion."""
    success: bool = Field(..., description="Whether ingestion succeeded")
    record_count: int = Field(..., description="Number of records ingested")
    s3_path: Optional[str] = Field(None, description="S3 path where data was stored")
    errors: List[str] = Field(default_factory=list, description="List of error messages")
    ingestion_time_seconds: float = Field(..., description="Time taken for ingestion")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "record_count": 1000,
                "s3_path": "s3://historical-data/dataset_20250115_123456",
                "errors": [],
                "ingestion_time_seconds": 2.5
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: Dict[str, Any]

    class Config:
        json_schema_extra = {
            "example": {
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Invalid input parameters",
                    "details": {"field": "forecast_horizon", "issue": "must be between 1 and 90"},
                    "timestamp": "2025-01-15T10:30:00Z"
                }
            }
        }