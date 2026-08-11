"""FastAPI application entry point."""
from fastapi import FastAPI, status, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from sqlalchemy import text
import time
from typing import Dict, Any, Optional, List, Union, Literal

from config.settings import settings
from src.utils.logging_config import logger
from src.registry.database import db_manager
from src.inference.forecasting_engine import ForecastingEngine, forecasting_engine, ConfidenceInterval
from src.registry.model_registry import ModelRegistry, model_registry
from src.api.auth import verify_api_key
from src.api.exceptions import (
    ValidationError,
    ModelNotFoundError,
    ForecastGenerationError,
    ServiceUnavailableError,
    APIException
)
from src.api.error_handlers import register_error_handlers
from src.api.rate_limiter import RateLimitMiddleware, rate_limiter


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info(
        f"Starting Demand Forecasting System API - "
        f"Environment: {settings.environment}"
    )
    yield
    logger.info("Shutting down Demand Forecasting System API")
    db_manager.close()


# Create FastAPI application
app = FastAPI(
    title="Demand Forecasting System API",
    lifespan=lifespan,
    description="""
    Scalable ML platform for demand prediction using historical sales data, 
    seasonality patterns, holiday effects, and price impacts.
    
    ## Features
    
    * **Forecast Generation**: Generate demand forecasts with confidence intervals
    * **Model Management**: Access custom and benchmark models
    * **Multi-Model Comparison**: Compare predictions from multiple models
    * **Health Monitoring**: Real-time service health checks
    
    ## Authentication
    
    All endpoints (except health checks) require API key authentication via the `X-API-Key` header.
    """,
    version=settings.api_version,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_tags=[
        {
            "name": "Forecasting",
            "description": "Generate demand forecasts using trained models"
        },
        {
            "name": "Models",
            "description": "Manage and query model registry"
        },
        {
            "name": "Data",
            "description": "Data ingestion and management"
        },
        {
            "name": "Health",
            "description": "Service health and status monitoring"
        },
        {
            "name": "Root",
            "description": "API root and information"
        }
    ]
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=False,  # Cannot use credentials with wildcard origins
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting middleware
app.add_middleware(RateLimitMiddleware, rate_limiter=rate_limiter)

# Register error handlers
register_error_handlers(app)


# ============================================================================
# Pydantic Models for Request/Response
# ============================================================================

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
    """
    Request model for forecast generation.
    
    **Validates: Requirement 5.1**
    """
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
    """
    Response model for forecast generation.
    
    **Validates: Requirement 5.2, 5.3**
    """
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


# ============================================================================
# Dependency Injection
# ============================================================================

def get_forecasting_engine() -> ForecastingEngine:
    """
    Dependency injection for ForecastingEngine.
    
    Returns the global forecasting engine instance.
    This allows for easy mocking in tests.
    
    **Validates: Requirement 5.1**
    """
    return forecasting_engine


def get_model_registry() -> ModelRegistry:
    """
    Dependency injection for ModelRegistry.
    
    Returns the global model registry instance.
    This allows for easy mocking in tests.
    
    **Validates: Requirement 5.1**
    """
    return model_registry


# ============================================================================
# Health Check Models and Endpoints
# ============================================================================


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: datetime
    environment: str
    version: str
    checks: Dict[str, Any]


@app.get(
    "/api/v1/health",
    response_model=HealthCheckResponse,
    status_code=status.HTTP_200_OK,
    tags=["Health"]
)
async def health_check() -> HealthCheckResponse:
    """
    Health check endpoint.
    
    Returns service status and component health within 1 second.
    
    **Validates: Requirement 10.5**
    """
    start_time = time.time()
    
    checks = {
        "api": "healthy",
        "database": "unknown",
        "response_time_ms": 0
    }
    
    # Check database connectivity
    try:
        with db_manager.get_session() as session:
            session.execute(text("SELECT 1"))
        checks["database"] = "healthy"
    except ConnectionError as e:
        checks["database"] = "unhealthy"
        checks["database_error"] = str(e)
        logger.error(f"Database health check failed: {e}")
        # Raise ServiceUnavailableError if database is critical
        raise ServiceUnavailableError(
            message="Database service is unavailable",
            details={
                "component": "database",
                "error": str(e)
            }
        )
    except Exception as e:
        checks["database"] = "unhealthy"
        checks["database_error"] = str(e)
        logger.error(f"Database health check failed: {e}")
    
    # Calculate response time
    response_time_ms = (time.time() - start_time) * 1000
    checks["response_time_ms"] = round(response_time_ms, 2)
    
    # Determine overall status
    overall_status = "healthy" if checks["database"] == "healthy" else "degraded"
    
    response = HealthCheckResponse(
        status=overall_status,
        timestamp=datetime.now(timezone.utc),
        environment=settings.environment,
        version=settings.api_version,
        checks=checks
    )
    
    # Log health check
    logger.info(
        f"Health check completed: status={overall_status}, "
        f"response_time={response_time_ms:.2f}ms"
    )
    
    return response


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Demand Forecasting System API",
        "version": settings.api_version,
        "docs": "/api/docs"
    }


@app.post(
    "/api/v1/forecast",
    response_model=ForecastResponse,
    status_code=status.HTTP_200_OK,
    tags=["Forecasting"],
    summary="Generate demand forecast",
    description="""
    Generate demand forecast for a product using trained models.
    
    **Validates: Requirements 5.1, 5.2, 5.3, 5.5, 7.2**
    
    This endpoint accepts a forecast request with product ID, forecast horizon,
    and optional model selection. It returns predictions with confidence intervals
    at 50%, 80%, and 90% levels.
    
    **Authentication**: Requires valid API key in X-API-Key header.
    
    **Performance Target**: Response within 3 seconds for horizons up to 90 days.
    """
)
async def generate_forecast(
    request: ForecastRequest,
    engine: ForecastingEngine = Depends(get_forecasting_engine),
    registry: ModelRegistry = Depends(get_model_registry),
    api_key: str = Depends(verify_api_key)
) -> ForecastResponse:
    """
    Generate demand forecast for a product.
    
    **Validates: Requirements 5.1, 5.2, 5.3, 5.5, 7.2**
    
    Args:
        request: Forecast request with product_id, forecast_horizon, model_id, etc.
        engine: Forecasting engine instance (injected)
        registry: Model registry instance (injected)
        api_key: Validated API key (injected)
        
    Returns:
        ForecastResponse with predictions, confidence intervals, and metadata
        
    Raises:
        HTTPException: 401 for invalid/missing API key, 400 for invalid parameters, 
                      404 for missing model, 500 for errors
    """
    start_time = time.time()
    
    try:
        logger.info(
            f"Forecast request received: product_id={request.product_id}, "
            f"horizon={request.forecast_horizon}, model_id={request.model_id}, "
            f"include_benchmark={request.include_benchmark}"
        )
        
        # Determine which model to use
        if request.model_id:
            # Use specified model
            model_id = request.model_id
            logger.info(f"Using specified model: {model_id}")
        else:
            # Use latest custom model for the product
            try:
                model_id, _ = registry.get_latest_model(
                    product_id=request.product_id,
                    model_type="custom"
                )
                logger.info(f"Using latest custom model: {model_id}")
            except ValueError as e:
                logger.error(f"No custom model found for product {request.product_id}: {e}")
                raise ModelNotFoundError(
                    message=f"No trained model found for product {request.product_id}. Please train a model first.",
                    details={
                        "product_id": request.product_id,
                        "model_type": "custom"
                    }
                )
        
        # Generate forecast using the selected model
        try:
            forecast_result = engine.generate_forecast(
                model_id=model_id,
                forecast_horizon=request.forecast_horizon,
                future_features=request.future_features,
                start_date=None  # Use current time
            )
        except ValueError as e:
            logger.error(f"Validation error during forecast generation: {e}")
            raise ValidationError(
                message=str(e),
                details={
                    "product_id": request.product_id,
                    "forecast_horizon": request.forecast_horizon,
                    "model_id": model_id
                }
            )
        except RuntimeError as e:
            logger.error(f"Runtime error during forecast generation: {e}")
            raise ForecastGenerationError(
                message="Failed to generate forecast",
                details={
                    "error": str(e),
                    "product_id": request.product_id,
                    "model_id": model_id
                }
            )
        
        # Generate benchmark forecast if requested
        benchmark_result = None
        if request.include_benchmark:
            try:
                # Try to get latest benchmark model
                benchmark_model_id, _ = registry.get_latest_model(
                    product_id=request.product_id,
                    model_type="forecast"
                )
                logger.info(f"Generating benchmark forecast with model: {benchmark_model_id}")
                
                benchmark_forecast = engine.generate_forecast(
                    model_id=benchmark_model_id,
                    forecast_horizon=request.forecast_horizon,
                    future_features=request.future_features,
                    start_date=None
                )
                
                # Convert benchmark forecast to dictionary format
                benchmark_result = {
                    "model_id": benchmark_forecast.model_id,
                    "predictions": benchmark_forecast.predictions,
                    "confidence_intervals": {
                        level: {
                            "level": ci.level,
                            "lower": ci.lower,
                            "upper": ci.upper
                        }
                        for level, ci in benchmark_forecast.confidence_intervals.items()
                    },
                    "metadata": benchmark_forecast.metadata
                }
                
            except ValueError as e:
                logger.warning(f"No benchmark model found for product {request.product_id}: {e}")
                # Continue without benchmark
            except Exception as e:
                logger.warning(f"Failed to generate benchmark forecast: {e}")
                # Continue without benchmark
        
        # Generate unique forecast ID
        forecast_id = f"forecast_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{request.product_id}"
        
        # Convert confidence intervals to response format
        confidence_intervals_response = {}
        for level, ci in forecast_result.confidence_intervals.items():
            confidence_intervals_response[level] = ConfidenceIntervalModel(
                level=ci.level,
                lower=ci.lower,
                upper=ci.upper
            )
        
        # Calculate response time
        response_time_ms = (time.time() - start_time) * 1000
        
        # Create response
        response = ForecastResponse(
            forecast_id=forecast_id,
            product_id=forecast_result.product_id,
            model_id=forecast_result.model_id,
            timestamps=forecast_result.timestamps,
            predictions=forecast_result.predictions,
            confidence_intervals=confidence_intervals_response,
            benchmark=benchmark_result,
            metadata=forecast_result.metadata
        )
        
        logger.info(
            f"Forecast generated successfully: forecast_id={forecast_id}, "
            f"predictions={len(forecast_result.predictions)}, "
            f"response_time={response_time_ms:.2f}ms"
        )
        
        # Check if response time exceeds target (3 seconds)
        if response_time_ms > 3000:
            logger.warning(
                f"Forecast response time exceeded target: {response_time_ms:.2f}ms > 3000ms"
            )
        
        return response
        
    except APIException:
        # Re-raise custom API exceptions to be handled by error handlers
        raise
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Catch any unexpected errors
        logger.error(f"Unexpected error in forecast endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An internal error occurred while generating forecast",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            }
        )


# ============================================================================
# Model Management Endpoints
# ============================================================================


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


@app.get(
    "/api/v1/models",
    response_model=ModelListResponse,
    status_code=status.HTTP_200_OK,
    tags=["Models"],
    summary="List available models",
    description="""
    List all available models with optional filtering by product ID and model type.
    
    **Validates: Requirement 10.5**
    
    **Authentication**: Requires valid API key in X-API-Key header.
    
    **Query Parameters**:
    - product_id: Optional filter by product identifier
    - model_type: Optional filter by model type (custom or forecast)
    """
)
async def list_models(
    product_id: Optional[str] = None,
    model_type: Optional[str] = None,
    registry: ModelRegistry = Depends(get_model_registry),
    api_key: str = Depends(verify_api_key)
) -> ModelListResponse:
    """
    List available models with optional filtering.
    
    **Validates: Requirement 10.5**
    
    Args:
        product_id: Optional filter by product ID
        model_type: Optional filter by model type
        registry: Model registry instance (injected)
        api_key: Validated API key (injected)
        
    Returns:
        ModelListResponse with list of models and total count
        
    Raises:
        HTTPException: 400 for invalid model_type, 401 for invalid API key, 500 for errors
    """
    try:
        # Validate model_type if provided
        if model_type and model_type not in ["custom", "forecast"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": {
                        "code": "VALIDATION_ERROR",
                        "message": f"Invalid model_type: {model_type}. Must be 'custom' or 'forecast'",
                        "details": {
                            "model_type": model_type,
                            "valid_values": ["custom", "forecast"]
                        },
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                }
            )
        
        logger.info(
            f"List models request: product_id={product_id}, model_type={model_type}"
        )
        
        # Get models from registry
        models = registry.list_models(
            product_id=product_id,
            model_type=model_type
        )
        
        # Convert to response format
        models_data = []
        for model in models:
            models_data.append({
                "model_id": model.model_id,
                "product_id": model.product_id,
                "model_type": model.model_type,
                "version": model.version,
                "mae": model.mae,
                "rmse": model.rmse,
                "mape": model.mape,
                "created_at": model.created_at.isoformat(),
                "forecast_horizon": model.forecast_horizon
            })
        
        response = ModelListResponse(
            models=models_data,
            total_count=len(models_data)
        )
        
        logger.info(f"Listed {len(models_data)} models successfully")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing models: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An internal error occurred while listing models",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            }
        )


@app.get(
    "/api/v1/models/{model_id}",
    response_model=ModelMetadataResponse,
    status_code=status.HTTP_200_OK,
    tags=["Models"],
    summary="Get model metadata",
    description="""
    Retrieve detailed metadata for a specific model by ID.
    
    **Validates: Requirement 10.5**
    
    **Authentication**: Requires valid API key in X-API-Key header.
    """
)
async def get_model_metadata(
    model_id: str,
    registry: ModelRegistry = Depends(get_model_registry),
    api_key: str = Depends(verify_api_key)
) -> ModelMetadataResponse:
    """
    Get model metadata by ID.
    
    **Validates: Requirement 10.5**
    
    Args:
        model_id: Unique model identifier
        registry: Model registry instance (injected)
        api_key: Validated API key (injected)
        
    Returns:
        ModelMetadataResponse with complete model metadata
        
    Raises:
        HTTPException: 404 for model not found, 401 for invalid API key, 500 for errors
    """
    try:
        logger.info(f"Get model metadata request: model_id={model_id}")
        
        # Get model from registry (only need metadata, not artifact)
        try:
            _, metadata = registry.get_model(model_id)
        except ValueError as e:
            logger.warning(f"Model not found: {model_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": {
                        "code": "MODEL_NOT_FOUND",
                        "message": f"Model not found: {model_id}",
                        "details": {"model_id": model_id},
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                }
            )
        
        # Convert to response format
        response = ModelMetadataResponse(
            model_id=metadata.model_id,
            product_id=metadata.product_id,
            model_type=metadata.model_type,
            version=metadata.version,
            artifact_path=metadata.artifact_path,
            training_dataset_id=metadata.training_dataset_id,
            mae=metadata.mae,
            rmse=metadata.rmse,
            mape=metadata.mape,
            hyperparameters=metadata.hyperparameters,
            created_at=metadata.created_at,
            forecast_horizon=metadata.forecast_horizon
        )
        
        logger.info(f"Retrieved model metadata successfully: {model_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model metadata: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An internal error occurred while retrieving model metadata",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            }
        )


# ============================================================================
# Data Ingestion Endpoints
# ============================================================================


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


def get_data_ingestion_service():
    """
    Dependency injection for DataIngestionService.
    
    Returns the data ingestion service instance.
    This allows for easy mocking in tests.
    """
    from src.data.ingestion import DataIngestionService
    return DataIngestionService()


@app.post(
    "/api/v1/data/ingest",
    response_model=DataIngestionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Data"],
    summary="Ingest historical sales data",
    description="""
    Ingest historical sales data with validation and S3 storage.
    
    **Validates: Requirement 10.5**
    
    Accepts data in CSV or JSON format with automatic format detection.
    Data is validated for schema compliance and stored in S3 for model training.
    
    **Authentication**: Requires valid API key in X-API-Key header.
    
    **Performance Target**: 5 seconds for 1M records, 60 seconds for 5M records.
    """
)
async def ingest_data(
    request: DataIngestionRequest,
    ingestion_service = Depends(get_data_ingestion_service),
    api_key: str = Depends(verify_api_key)
) -> DataIngestionResponse:
    """
    Ingest historical sales data.
    
    **Validates: Requirement 10.5**
    
    Args:
        request: Data ingestion request with data and format
        ingestion_service: Data ingestion service instance (injected)
        api_key: Validated API key (injected)
        
    Returns:
        DataIngestionResponse with success status, record count, and S3 path
        
    Raises:
        HTTPException: 400 for validation errors, 401 for invalid API key, 500 for errors
    """
    try:
        logger.info(f"Data ingestion request received: format={request.format}")
        
        # Ingest data
        result = ingestion_service.ingest_batch(
            data=request.data,
            format=request.format
        )
        
        # If ingestion failed, return error response with 400 status
        if not result.success:
            logger.warning(f"Data ingestion failed: {result.errors}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": {
                        "code": "VALIDATION_ERROR",
                        "message": "Data ingestion failed due to validation errors",
                        "details": {
                            "errors": result.errors,
                            "record_count": result.record_count
                        },
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                }
            )
        
        # Create response
        response = DataIngestionResponse(
            success=result.success,
            record_count=result.record_count,
            s3_path=result.s3_path,
            errors=result.errors,
            ingestion_time_seconds=result.ingestion_time_seconds
        )
        
        logger.info(
            f"Data ingestion completed: success={result.success}, "
            f"records={result.record_count}, time={result.ingestion_time_seconds:.2f}s"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during data ingestion: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An internal error occurred during data ingestion",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            }
        )


