"""
Forecasting Engine for generating demand predictions.

This module provides the ForecastingEngine class that generates forecasts
using trained custom models and Amazon Forecast predictors with confidence intervals.

Requirements: 4.1, 4.2, 4.3, 4.4, 9.4
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import logging
from io import BytesIO
import boto3
from botocore.exceptions import ClientError
import time

from src.registry.model_registry import model_registry, ModelMetadata
from src.training.custom_model_trainer import CustomModelTrainer
from src.features.seasonality import extract_seasonality_features
from src.features.preprocessing import FeaturePreprocessor
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceInterval:
    """
    Confidence interval for forecast predictions.
    
    Attributes:
        level: Confidence level as percentage string (e.g., "50%", "80%", "90%")
        lower: Lower bound values
        upper: Upper bound values
    """
    level: str
    lower: List[float]
    upper: List[float]


@dataclass
class ForecastResult:
    """
    Result of forecast generation.
    
    Attributes:
        model_id: ID of model used for forecast
        product_id: Product identifier
        timestamps: Future timestamps for predictions
        predictions: Point predictions (mean/median)
        confidence_intervals: Dictionary mapping confidence levels to intervals
        metadata: Additional forecast metadata
    """
    model_id: str
    product_id: str
    timestamps: List[datetime]
    predictions: List[float]
    confidence_intervals: Dict[str, ConfidenceInterval] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ForecastingEngine:
    """
    Generates demand forecasts using trained models.
    
    Responsibilities:
    - Load model artifacts from Model Registry
    - Prepare future features (holidays, seasonality) for forecast horizon
    - Generate point predictions using loaded model
    - Calculate confidence intervals using quantile regression or bootstrap
    - Query Amazon Forecast predictors for benchmark forecasts
    """
    
    def __init__(self, forecast_client=None, forecastquery_client=None):
        """
        Initialize the forecasting engine.
        
        AWS Forecast clients are created lazily on first use to avoid
        instantiating boto3 clients at import time.
        
        Args:
            forecast_client: Optional boto3 Forecast client (for testing)
            forecastquery_client: Optional boto3 ForecastQuery client (for testing)
        """
        self.feature_preprocessor = FeaturePreprocessor()
        self._forecast_client = forecast_client
        self._forecastquery_client = forecastquery_client

    @property
    def forecast_client(self):
        """Lazily initialized boto3 Forecast client."""
        if self._forecast_client is None:
            self._forecast_client = boto3.client(
                'forecast',
                region_name=settings.aws_region,
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key
            )
        return self._forecast_client

    @forecast_client.setter
    def forecast_client(self, value):
        self._forecast_client = value

    @property
    def forecastquery_client(self):
        """Lazily initialized boto3 ForecastQuery client."""
        if self._forecastquery_client is None:
            self._forecastquery_client = boto3.client(
                'forecastquery',
                region_name=settings.aws_region,
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key
            )
        return self._forecastquery_client

    @forecastquery_client.setter
    def forecastquery_client(self, value):
        self._forecastquery_client = value
    
    def generate_forecast(
        self,
        model_id: str,
        forecast_horizon: int,
        future_features: Optional[Dict[str, Any]] = None,
        start_date: Optional[datetime] = None
    ) -> ForecastResult:
        """
        Generate demand forecast for specified horizon.
        
        Args:
            model_id: Model to use for prediction
            forecast_horizon: Number of time steps to forecast (1-90 days)
            future_features: Optional future values for price, holidays
            start_date: Optional start date for forecast (defaults to now)
            
        Returns:
            ForecastResult with predictions and confidence intervals
            
        Raises:
            ValueError: If model not found or invalid parameters
            RuntimeError: If forecast generation fails
        """
        try:
            # Validate forecast horizon
            if forecast_horizon < 1 or forecast_horizon > 90:
                raise ValueError(
                    f"Invalid forecast_horizon: {forecast_horizon}. "
                    f"Must be between 1 and 90 days."
                )
            
            logger.info(f"Generating forecast with model {model_id} for {forecast_horizon} days")
            
            # Load model artifact and metadata from registry
            model_artifact, metadata = model_registry.get_model(model_id)
            
            # Check if this is an Amazon Forecast model
            if metadata.model_type == 'forecast':
                return self._generate_forecast_from_amazon_forecast(
                    model_id=model_id,
                    metadata=metadata,
                    forecast_horizon=forecast_horizon,
                    start_date=start_date
                )
            
            # Otherwise, use custom model
            # Deserialize model
            model, algorithm = CustomModelTrainer.deserialize_model(model_artifact)
            
            # Prepare future features
            if start_date is None:
                start_date = datetime.now()
            
            future_df = self._prepare_future_features(
                start_date=start_date,
                forecast_horizon=forecast_horizon,
                future_features=future_features,
                metadata=metadata
            )
            
            # Generate point predictions
            predictions = self._generate_predictions(
                model=model,
                algorithm=algorithm,
                future_df=future_df,
                metadata=metadata
            )
            
            # Calculate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(
                model=model,
                algorithm=algorithm,
                future_df=future_df,
                predictions=predictions,
                metadata=metadata
            )
            
            # Create timestamps for forecast
            timestamps = [
                start_date + timedelta(days=i)
                for i in range(forecast_horizon)
            ]
            
            # Prepare result metadata
            result_metadata = {
                'algorithm': algorithm,
                'model_version': metadata.version,
                'training_mae': metadata.mae,
                'training_rmse': metadata.rmse,
                'training_mape': metadata.mape,
                'forecast_horizon': forecast_horizon,
                'start_date': start_date.isoformat(),
                'hyperparameters': metadata.hyperparameters
            }
            
            logger.info(
                f"Forecast generated successfully: {len(predictions)} predictions, "
                f"{len(confidence_intervals)} confidence levels"
            )
            
            return ForecastResult(
                model_id=model_id,
                product_id=metadata.product_id,
                timestamps=timestamps,
                predictions=predictions,
                confidence_intervals=confidence_intervals,
                metadata=result_metadata
            )
            
        except ValueError:
            # Re-raise validation errors
            raise
        except Exception as e:
            error_msg = f"Forecast generation failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
    
    def _generate_forecast_from_amazon_forecast(
        self,
        model_id: str,
        metadata: ModelMetadata,
        forecast_horizon: int,
        start_date: Optional[datetime] = None
    ) -> ForecastResult:
        """
        Generate forecast using Amazon Forecast predictor.
        
        This method queries an Amazon Forecast predictor to generate forecasts.
        It first creates a forecast export job, then queries the forecast to get
        quantile predictions (p10, p50, p90) which are converted to confidence intervals.
        
        Args:
            model_id: Model ID (contains predictor ARN in hyperparameters)
            metadata: Model metadata from registry
            forecast_horizon: Number of days to forecast
            start_date: Optional start date for forecast
            
        Returns:
            ForecastResult with predictions and confidence intervals
            
        Raises:
            RuntimeError: If Forecast API calls fail
        """
        try:
            if start_date is None:
                start_date = datetime.now()
            
            logger.info(
                f"Generating forecast from Amazon Forecast predictor: {model_id}"
            )
            
            # Extract predictor ARN from metadata
            predictor_arn = metadata.hyperparameters.get('predictor_arn')
            if not predictor_arn:
                raise ValueError(
                    f"Predictor ARN not found in model metadata for {model_id}"
                )
            
            # Create forecast from predictor
            forecast_arn = self._create_forecast(
                predictor_arn=predictor_arn,
                product_id=metadata.product_id
            )
            
            # Wait for forecast creation to complete
            self._wait_for_forecast_completion(forecast_arn)
            
            # Query forecast for predictions
            forecast_data = self._query_forecast(
                forecast_arn=forecast_arn,
                product_id=metadata.product_id,
                start_date=start_date,
                forecast_horizon=forecast_horizon
            )
            
            # Extract predictions and confidence intervals from forecast data
            timestamps, predictions, confidence_intervals = self._extract_forecast_results(
                forecast_data=forecast_data,
                start_date=start_date,
                forecast_horizon=forecast_horizon
            )
            
            # Prepare result metadata
            result_metadata = {
                'algorithm': 'amazon_forecast',
                'model_version': metadata.version,
                'training_mae': metadata.mae,
                'training_rmse': metadata.rmse,
                'training_mape': metadata.mape,
                'forecast_horizon': forecast_horizon,
                'start_date': start_date.isoformat(),
                'predictor_arn': predictor_arn,
                'forecast_arn': forecast_arn
            }
            
            logger.info(
                f"Amazon Forecast query completed: {len(predictions)} predictions"
            )
            
            return ForecastResult(
                model_id=model_id,
                product_id=metadata.product_id,
                timestamps=timestamps,
                predictions=predictions,
                confidence_intervals=confidence_intervals,
                metadata=result_metadata
            )
            
        except ClientError as e:
            error_msg = f"Amazon Forecast API error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f"Amazon Forecast query failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
    
    def _create_forecast(
        self,
        predictor_arn: str,
        product_id: str
    ) -> str:
        """
        Create a forecast from an Amazon Forecast predictor.
        
        Args:
            predictor_arn: ARN of the trained predictor
            product_id: Product identifier for naming
            
        Returns:
            Forecast ARN
        """
        try:
            # Generate unique forecast name
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            forecast_name = f"{product_id}_forecast_{timestamp}"
            
            logger.info(f"Creating forecast: {forecast_name}")
            
            response = self.forecast_client.create_forecast(
                ForecastName=forecast_name,
                PredictorArn=predictor_arn
            )
            
            forecast_arn = response['ForecastArn']
            logger.info(f"Forecast creation initiated: {forecast_arn}")
            
            return forecast_arn
            
        except ClientError as e:
            # Check if forecast already exists
            if e.response['Error']['Code'] == 'ResourceAlreadyExistsException':
                logger.warning(f"Forecast already exists: {forecast_name}")
                # Try to describe the existing forecast
                try:
                    response = self.forecast_client.describe_forecast(
                        ForecastArn=self._build_forecast_arn(forecast_name)
                    )
                    return response['ForecastArn']
                except Exception as describe_error:
                    logger.warning(f"Failed to describe existing forecast: {describe_error}")
            
            logger.error(f"Failed to create forecast: {str(e)}")
            raise
    
    def _wait_for_forecast_completion(
        self,
        forecast_arn: str,
        max_wait_seconds: int = 1800,
        poll_interval: int = 30
    ) -> None:
        """
        Wait for forecast creation to complete.
        
        Args:
            forecast_arn: ARN of the forecast
            max_wait_seconds: Maximum time to wait (default 30 minutes)
            poll_interval: Seconds between status checks
            
        Raises:
            RuntimeError: If forecast creation fails or times out
        """
        start_time = time.time()
        
        logger.info(f"Waiting for forecast completion: {forecast_arn}")
        
        while True:
            try:
                response = self.forecast_client.describe_forecast(
                    ForecastArn=forecast_arn
                )
                
                status = response['Status']
                
                logger.debug(f"Forecast status: {status}")
                
                if status == 'ACTIVE':
                    logger.info("Forecast creation completed successfully")
                    return
                
                elif status in ['CREATE_FAILED', 'DELETE_PENDING', 'DELETE_IN_PROGRESS']:
                    error_message = response.get('Message', 'Unknown error')
                    raise RuntimeError(
                        f"Forecast creation failed with status {status}: {error_message}"
                    )
                
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > max_wait_seconds:
                    raise RuntimeError(
                        f"Forecast creation did not complete within {max_wait_seconds}s. "
                        f"Current status: {status}"
                    )
                
                # Wait before next poll
                time.sleep(poll_interval)
                
            except ClientError as e:
                logger.error(f"Error checking forecast status: {str(e)}")
                raise RuntimeError(f"Failed to check forecast status: {str(e)}") from e
    
    def _query_forecast(
        self,
        forecast_arn: str,
        product_id: str,
        start_date: datetime,
        forecast_horizon: int
    ) -> Dict[str, Any]:
        """
        Query Amazon Forecast for predictions.
        
        Args:
            forecast_arn: ARN of the forecast
            product_id: Product identifier (item_id in Forecast)
            start_date: Start date for forecast
            forecast_horizon: Number of days to forecast
            
        Returns:
            Forecast data with quantile predictions
        """
        try:
            # Calculate end date
            end_date = start_date + timedelta(days=forecast_horizon)
            
            logger.info(
                f"Querying forecast for product {product_id} "
                f"from {start_date} to {end_date}"
            )
            
            # Query forecast using ForecastQuery API
            response = self.forecastquery_client.query_forecast(
                ForecastArn=forecast_arn,
                Filters={
                    'item_id': product_id
                },
                StartDate=start_date.strftime('%Y-%m-%dT%H:%M:%S'),
                EndDate=end_date.strftime('%Y-%m-%dT%H:%M:%S')
            )
            
            logger.info(f"Forecast query completed successfully")
            
            return response['Forecast']
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            
            if error_code == 'ResourceNotFoundException':
                raise RuntimeError(
                    f"Forecast not found or not available: {forecast_arn}"
                ) from e
            elif error_code == 'InvalidInputException':
                raise ValueError(
                    f"Invalid query parameters: {str(e)}"
                ) from e
            else:
                logger.error(f"Forecast query failed: {str(e)}")
                raise RuntimeError(f"Forecast query failed: {str(e)}") from e
    
    def _extract_forecast_results(
        self,
        forecast_data: Dict[str, Any],
        start_date: datetime,
        forecast_horizon: int
    ) -> Tuple[List[datetime], List[float], Dict[str, ConfidenceInterval]]:
        """
        Extract predictions and confidence intervals from Forecast response.
        
        Amazon Forecast returns quantile forecasts (p10, p50, p90).
        We convert these to confidence intervals:
        - 50% CI: p25 to p75 (approximated from p10, p50, p90)
        - 80% CI: p10 to p90
        - 90% CI: p5 to p95 (approximated from p10, p50, p90)
        
        Args:
            forecast_data: Forecast response from query_forecast
            start_date: Start date for forecast
            forecast_horizon: Number of days to forecast
            
        Returns:
            Tuple of (timestamps, predictions, confidence_intervals)
        """
        try:
            # Extract predictions from forecast data
            # Forecast data structure: {'Predictions': {'p10': [...], 'p50': [...], 'p90': [...]}}
            predictions_dict = forecast_data.get('Predictions', {})
            
            # Get p50 (median) as point predictions
            p50_values = predictions_dict.get('p50', [])
            p10_values = predictions_dict.get('p10', [])
            p90_values = predictions_dict.get('p90', [])
            
            if not p50_values:
                raise ValueError("No p50 predictions found in Forecast response")
            
            # Limit to forecast horizon
            p50_values = p50_values[:forecast_horizon]
            p10_values = p10_values[:forecast_horizon] if p10_values else []
            p90_values = p90_values[:forecast_horizon] if p90_values else []
            
            # Extract values from prediction objects
            predictions = [pred['Value'] for pred in p50_values]
            
            # Create timestamps
            timestamps = [
                start_date + timedelta(days=i)
                for i in range(len(predictions))
            ]
            
            # Convert quantiles to confidence intervals
            confidence_intervals = self._convert_quantiles_to_confidence_intervals(
                p10_values=p10_values,
                p50_values=p50_values,
                p90_values=p90_values
            )
            
            logger.info(
                f"Extracted {len(predictions)} predictions with "
                f"{len(confidence_intervals)} confidence levels"
            )
            
            return timestamps, predictions, confidence_intervals
            
        except Exception as e:
            logger.error(f"Failed to extract forecast results: {str(e)}")
            raise RuntimeError(f"Failed to extract forecast results: {str(e)}") from e
    
    def _convert_quantiles_to_confidence_intervals(
        self,
        p10_values: List[Dict[str, Any]],
        p50_values: List[Dict[str, Any]],
        p90_values: List[Dict[str, Any]]
    ) -> Dict[str, ConfidenceInterval]:
        """
        Convert Amazon Forecast quantiles to confidence intervals.
        
        Amazon Forecast provides p10, p50, p90 quantiles.
        We convert these to standard confidence intervals:
        - 80% CI: p10 to p90 (directly available)
        - 50% CI: Approximate using p50 ± 0.33 * (p90 - p10)
        - 90% CI: Approximate using p50 ± 0.82 * (p90 - p10)
        
        Args:
            p10_values: 10th percentile predictions
            p50_values: 50th percentile (median) predictions
            p90_values: 90th percentile predictions
            
        Returns:
            Dictionary mapping confidence levels to ConfidenceInterval objects
        """
        confidence_intervals = {}
        
        # Extract numeric values
        p10 = np.array([pred['Value'] for pred in p10_values]) if p10_values else None
        p50 = np.array([pred['Value'] for pred in p50_values])
        p90 = np.array([pred['Value'] for pred in p90_values]) if p90_values else None
        
        if p10 is not None and p90 is not None:
            # Calculate spread from p10 to p90
            spread = p90 - p10
            
            # 80% CI: p10 to p90 (directly from Forecast)
            confidence_intervals['80%'] = ConfidenceInterval(
                level='80%',
                lower=np.maximum(p10, 0).tolist(),  # Ensure non-negative
                upper=np.maximum(p90, 0).tolist()
            )
            
            # 50% CI: Approximate as p50 ± 0.33 * spread
            # (50% CI is roughly 0.67 standard deviations, 80% is 1.28)
            ci_50_lower = p50 - 0.33 * spread
            ci_50_upper = p50 + 0.33 * spread
            
            confidence_intervals['50%'] = ConfidenceInterval(
                level='50%',
                lower=np.maximum(ci_50_lower, 0).tolist(),
                upper=np.maximum(ci_50_upper, 0).tolist()
            )
            
            # 90% CI: Approximate as p50 ± 0.82 * spread
            # (90% CI is roughly 1.645 standard deviations, 80% is 1.28)
            ci_90_lower = p50 - 0.82 * spread
            ci_90_upper = p50 + 0.82 * spread
            
            confidence_intervals['90%'] = ConfidenceInterval(
                level='90%',
                lower=np.maximum(ci_90_lower, 0).tolist(),
                upper=np.maximum(ci_90_upper, 0).tolist()
            )
        else:
            # Fallback: If quantiles not available, create symmetric intervals
            # using a default uncertainty estimate
            logger.warning(
                "p10/p90 quantiles not available, using default uncertainty estimate"
            )
            
            # Use 20% of prediction as uncertainty estimate
            uncertainty = p50 * 0.2
            
            confidence_intervals['50%'] = ConfidenceInterval(
                level='50%',
                lower=np.maximum(p50 - 0.67 * uncertainty, 0).tolist(),
                upper=(p50 + 0.67 * uncertainty).tolist()
            )
            
            confidence_intervals['80%'] = ConfidenceInterval(
                level='80%',
                lower=np.maximum(p50 - 1.28 * uncertainty, 0).tolist(),
                upper=(p50 + 1.28 * uncertainty).tolist()
            )
            
            confidence_intervals['90%'] = ConfidenceInterval(
                level='90%',
                lower=np.maximum(p50 - 1.645 * uncertainty, 0).tolist(),
                upper=(p50 + 1.645 * uncertainty).tolist()
            )
        
        return confidence_intervals
    
    def _build_forecast_arn(self, forecast_name: str) -> str:
        """
        Build Amazon Forecast ARN for a forecast.
        
        Args:
            forecast_name: Name of the forecast
            
        Returns:
            Full ARN string
        """
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
        
        arn = f"arn:aws:forecast:{settings.aws_region}:{account_id}:forecast/{forecast_name}"
        
        return arn
    
    def _prepare_future_features(
        self,
        start_date: datetime,
        forecast_horizon: int,
        future_features: Optional[Dict[str, Any]],
        metadata: ModelMetadata
    ) -> pd.DataFrame:
        """
        Prepare future features for forecast horizon.
        
        Creates a DataFrame with future timestamps and features including:
        - Seasonality features (day_of_week, month, quarter, season)
        - Holiday indicators (from future_features or defaults to False)
        - Price data (from future_features or uses last known price)
        
        Args:
            start_date: Start date for forecast
            forecast_horizon: Number of days to forecast
            future_features: Optional dictionary with future feature values
            metadata: Model metadata containing feature information
            
        Returns:
            DataFrame with future features
        """
        # Create future timestamps
        timestamps = [
            start_date + timedelta(days=i)
            for i in range(forecast_horizon)
        ]
        
        future_df = pd.DataFrame({
            'timestamp': timestamps
        })
        
        # Extract seasonality features
        future_df = extract_seasonality_features(future_df, timestamp_column='timestamp')
        
        # Add holiday indicators
        if future_features and 'holidays' in future_features:
            # Use provided holiday indicators
            holidays = future_features['holidays']
            if isinstance(holidays, list) and len(holidays) == forecast_horizon:
                future_df['is_holiday'] = holidays
            else:
                logger.warning(
                    f"Invalid holidays format in future_features. "
                    f"Expected list of {forecast_horizon} boolean values. "
                    f"Defaulting to False."
                )
                future_df['is_holiday'] = False
        else:
            # Default to no holidays
            future_df['is_holiday'] = False
        
        # Add price data
        if future_features and 'prices' in future_features:
            # Use provided prices
            prices = future_features['prices']
            if isinstance(prices, list) and len(prices) == forecast_horizon:
                future_df['price'] = prices
            elif isinstance(prices, (int, float)):
                # Single price value for all periods
                future_df['price'] = float(prices)
            else:
                logger.warning(
                    f"Invalid prices format in future_features. "
                    f"Expected list of {forecast_horizon} values or single value. "
                    f"Defaulting to 0."
                )
                future_df['price'] = 0.0
        else:
            # Default price (could be improved by using last known price)
            future_df['price'] = 0.0
        
        # Apply feature preprocessing if needed
        # Load normalization parameters from metadata if available
        if 'normalization_params' in metadata.hyperparameters:
            self.feature_preprocessor.set_normalization_params(
                metadata.hyperparameters['normalization_params']
            )
            preprocess_result = self.feature_preprocessor.preprocess_features(
                future_df,
                normalize=True,
                fit_normalization=False
            )
            future_df = preprocess_result.data
        
        return future_df
    
    def _generate_predictions(
        self,
        model: Any,
        algorithm: str,
        future_df: pd.DataFrame,
        metadata: ModelMetadata
    ) -> List[float]:
        """
        Generate point predictions using loaded model.
        
        Args:
            model: Loaded model object
            algorithm: Algorithm type (prophet, random_forest, etc.)
            future_df: DataFrame with future features
            metadata: Model metadata
            
        Returns:
            List of point predictions
        """
        # Get features used during training
        features = metadata.hyperparameters.get('features', [])
        
        if algorithm == 'prophet':
            # Prophet requires specific format
            prophet_df = future_df.copy()
            prophet_df['ds'] = pd.to_datetime(prophet_df['timestamp'])
            
            # Add regressors if they were used during training
            regressor_cols = ['ds'] + [f for f in features if f in prophet_df.columns]
            
            # Generate predictions
            forecast = model.predict(prophet_df[regressor_cols])
            predictions = forecast['yhat'].values.tolist()
        else:
            # Scikit-learn models
            # Use normalized features if they exist, otherwise use original
            feature_cols = []
            for feature in features:
                if f'{feature}_normalized' in future_df.columns:
                    feature_cols.append(f'{feature}_normalized')
                elif feature in future_df.columns:
                    feature_cols.append(feature)
            
            if not feature_cols:
                raise ValueError(
                    f"No valid features found in future data. "
                    f"Expected features: {features}"
                )
            
            X_future = future_df[feature_cols].values
            predictions = model.predict(X_future).tolist()
        
        return predictions
    
    def _calculate_confidence_intervals(
        self,
        model: Any,
        algorithm: str,
        future_df: pd.DataFrame,
        predictions: List[float],
        metadata: ModelMetadata
    ) -> Dict[str, ConfidenceInterval]:
        """
        Calculate confidence intervals for predictions.
        
        Uses different methods based on algorithm:
        - Prophet: Uses built-in uncertainty intervals
        - Scikit-learn: Uses bootstrap resampling or residual-based intervals
        
        Args:
            model: Loaded model object
            algorithm: Algorithm type
            future_df: DataFrame with future features
            predictions: Point predictions
            metadata: Model metadata
            
        Returns:
            Dictionary mapping confidence levels to ConfidenceInterval objects
        """
        confidence_intervals = {}
        
        if algorithm == 'prophet':
            # Prophet provides uncertainty intervals natively
            prophet_df = future_df.copy()
            prophet_df['ds'] = pd.to_datetime(prophet_df['timestamp'])
            
            features = metadata.hyperparameters.get('features', [])
            regressor_cols = ['ds'] + [f for f in features if f in prophet_df.columns]
            
            forecast = model.predict(prophet_df[regressor_cols])
            
            # Prophet provides yhat_lower and yhat_upper (80% interval by default)
            # We'll use these and scale for other confidence levels
            yhat_lower = forecast['yhat_lower'].values
            yhat_upper = forecast['yhat_upper'].values
            yhat = forecast['yhat'].values
            
            # Calculate standard deviation from 80% interval
            # For normal distribution, 80% interval is approximately ±1.28 std
            std_dev = (yhat_upper - yhat_lower) / (2 * 1.28)
            
            # Generate intervals for 50%, 80%, 90%
            # 50% ≈ ±0.67 std, 80% ≈ ±1.28 std, 90% ≈ ±1.645 std
            z_scores = {
                '50%': 0.674,
                '80%': 1.282,
                '90%': 1.645
            }
            
            for level, z in z_scores.items():
                lower = (yhat - z * std_dev).tolist()
                upper = (yhat + z * std_dev).tolist()
                
                confidence_intervals[level] = ConfidenceInterval(
                    level=level,
                    lower=lower,
                    upper=upper
                )
        else:
            # For scikit-learn models, use residual-based intervals
            # Estimate prediction uncertainty from training metrics
            
            # Use RMSE as estimate of prediction standard deviation
            std_dev = metadata.rmse
            
            # Generate intervals for 50%, 80%, 90%
            z_scores = {
                '50%': 0.674,
                '80%': 1.282,
                '90%': 1.645
            }
            
            for level, z in z_scores.items():
                lower = [pred - z * std_dev for pred in predictions]
                upper = [pred + z * std_dev for pred in predictions]
                
                # Ensure non-negative predictions for demand forecasting
                lower = [max(0, val) for val in lower]
                upper = [max(0, val) for val in upper]
                
                confidence_intervals[level] = ConfidenceInterval(
                    level=level,
                    lower=lower,
                    upper=upper
                )
        
        return confidence_intervals
    
    def generate_multi_model_forecast(
        self,
        product_id: str,
        forecast_horizon: int,
        future_features: Optional[Dict[str, Any]] = None,
        start_date: Optional[datetime] = None
    ) -> Dict[str, ForecastResult]:
        """
        Generate forecasts from all available models for comparison.
        
        Args:
            product_id: Product identifier
            forecast_horizon: Number of time steps to forecast
            future_features: Optional future values for price, holidays
            start_date: Optional start date for forecast
            
        Returns:
            Dictionary mapping model_id to ForecastResult
            
        Raises:
            ValueError: If no models found for product
        """
        try:
            logger.info(
                f"Generating multi-model forecast for product {product_id}, "
                f"horizon {forecast_horizon} days"
            )
            
            # Get all models for this product
            models = model_registry.list_models(product_id=product_id)
            
            if not models:
                raise ValueError(
                    f"No models found for product_id: {product_id}. "
                    f"Please train a model first."
                )
            
            # Generate forecast for each model
            results = {}
            for model_metadata in models:
                try:
                    forecast = self.generate_forecast(
                        model_id=model_metadata.model_id,
                        forecast_horizon=forecast_horizon,
                        future_features=future_features,
                        start_date=start_date
                    )
                    results[model_metadata.model_id] = forecast
                except Exception as e:
                    logger.warning(
                        f"Failed to generate forecast for model {model_metadata.model_id}: {e}"
                    )
                    # Continue with other models
                    continue
            
            if not results:
                raise RuntimeError(
                    f"Failed to generate forecasts from any model for product {product_id}"
                )
            
            logger.info(
                f"Multi-model forecast completed: {len(results)} models, "
                f"{forecast_horizon} days"
            )
            
            return results
            
        except ValueError:
            raise
        except Exception as e:
            error_msg = f"Multi-model forecast generation failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e


# Global forecasting engine instance
forecasting_engine = ForecastingEngine()
