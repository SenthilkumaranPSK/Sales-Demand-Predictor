"""
Seasonality feature extraction for the Demand Forecasting System.

This module provides functions to extract temporal features from timestamps,
including day of week, month, quarter, and season.
"""

from typing import Union, Literal
import pandas as pd
from datetime import datetime


def extract_seasonality_features(
    data: pd.DataFrame,
    timestamp_column: str = 'timestamp'
) -> pd.DataFrame:
    """
    Extract seasonality features from timestamp column.
    
    Extracts the following features:
    - day_of_week: Integer 0-6 (Monday=0, Sunday=6)
    - month: Integer 1-12
    - quarter: Integer 1-4
    - season: Categorical string ('spring', 'summer', 'fall', 'winter')
    
    Args:
        data: DataFrame containing timestamp column
        timestamp_column: Name of the timestamp column (default: 'timestamp')
        
    Returns:
        DataFrame with added seasonality feature columns
        
    Raises:
        ValueError: If timestamp column is missing or not datetime type
    """
    if timestamp_column not in data.columns:
        raise ValueError(f"Timestamp column '{timestamp_column}' not found in data")
    
    # Ensure timestamp column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(data[timestamp_column]):
        raise ValueError(f"Column '{timestamp_column}' must be datetime type")
    
    # Create a copy to avoid modifying the original
    result = data.copy()
    
    # Extract day of week (0=Monday, 6=Sunday)
    result['day_of_week'] = result[timestamp_column].dt.dayofweek
    
    # Extract month (1-12)
    result['month'] = result[timestamp_column].dt.month
    
    # Extract quarter (1-4)
    result['quarter'] = result[timestamp_column].dt.quarter
    
    # Extract season based on month
    result['season'] = result['month'].apply(_month_to_season)
    
    return result


def _month_to_season(month: int) -> Literal['spring', 'summer', 'fall', 'winter']:
    """
    Convert month number to season name.
    
    Mapping:
    - Spring: March (3), April (4), May (5)
    - Summer: June (6), July (7), August (8)
    - Fall: September (9), October (10), November (11)
    - Winter: December (12), January (1), February (2)
    
    Args:
        month: Month number (1-12)
        
    Returns:
        Season name as string
    """
    if month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    elif month in [9, 10, 11]:
        return 'fall'
    else:  # month in [12, 1, 2]
        return 'winter'
