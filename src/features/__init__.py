"""
Feature engineering module for the Demand Forecasting System.

This module provides feature extraction and transformation functions
for time-series forecasting, including seasonality features.
"""

from src.features.seasonality import extract_seasonality_features

__all__ = ['extract_seasonality_features']
