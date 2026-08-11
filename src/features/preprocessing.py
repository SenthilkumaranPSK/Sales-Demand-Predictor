"""
Feature preprocessing module for the Demand Forecasting System.

This module provides feature preprocessing pipelines for holiday indicators,
price data, and feature normalization for model training.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, field


@dataclass
class PreprocessingResult:
    """Result of feature preprocessing."""
    data: pd.DataFrame
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    normalization_params: Dict[str, Dict[str, float]] = field(default_factory=dict)


class FeaturePreprocessor:
    """
    Preprocesses features for model training.
    
    Handles:
    - Holiday indicator validation (boolean)
    - Price data validation (numeric, non-negative)
    - Feature normalization (standardization)
    """
    
    def __init__(self):
        """Initialize the feature preprocessor."""
        self.normalization_params: Dict[str, Dict[str, float]] = {}
    
    def preprocess_features(
        self,
        data: pd.DataFrame,
        normalize: bool = True,
        fit_normalization: bool = True
    ) -> PreprocessingResult:
        """
        Preprocess features for model training.
        
        Validates and normalizes holiday indicators and price data.
        
        Args:
            data: DataFrame containing features
            normalize: Whether to normalize numeric features
            fit_normalization: Whether to fit normalization parameters (True for training, False for inference)
            
        Returns:
            PreprocessingResult with processed data and validation status
        """
        errors = []
        processed_data = data.copy()
        
        # Validate holiday indicators
        if 'is_holiday' in processed_data.columns:
            holiday_errors = self._validate_holiday_indicators(processed_data)
            errors.extend(holiday_errors)
            
            # Convert to boolean if needed
            if not holiday_errors:
                processed_data = self._convert_holiday_to_boolean(processed_data)
        
        # Validate price data
        if 'price' in processed_data.columns:
            price_errors = self._validate_price_data(processed_data)
            errors.extend(price_errors)
        
        # Normalize features if requested
        if normalize and not errors:
            processed_data, norm_params = self._normalize_features(
                processed_data,
                fit=fit_normalization
            )
            if fit_normalization:
                self.normalization_params = norm_params
        
        is_valid = len(errors) == 0
        
        return PreprocessingResult(
            data=processed_data,
            is_valid=is_valid,
            errors=errors,
            normalization_params=self.normalization_params.copy()
        )
    
    def _validate_holiday_indicators(self, data: pd.DataFrame) -> List[str]:
        """
        Validate holiday indicators are boolean or boolean-compatible.
        
        Args:
            data: DataFrame containing is_holiday column
            
        Returns:
            List of error messages
        """
        errors = []
        
        if 'is_holiday' not in data.columns:
            return errors
        
        # Check if already boolean
        if pd.api.types.is_bool_dtype(data['is_holiday']):
            return errors
        
        # Check if values can be converted to boolean
        unique_values = data['is_holiday'].dropna().unique()
        valid_bool_values = {True, False, 0, 1, '0', '1', 'true', 'false', 'True', 'False'}
        
        invalid_values = [val for val in unique_values if val not in valid_bool_values]
        if invalid_values:
            errors.append(
                f"is_holiday contains invalid values: {invalid_values}. "
                f"Must be boolean or convertible to boolean (0, 1, 'true', 'false')"
            )
        
        # Check for missing values
        if data['is_holiday'].isna().any():
            missing_count = data['is_holiday'].isna().sum()
            errors.append(
                f"is_holiday contains {missing_count} missing values. "
                f"All holiday indicators must be specified."
            )
        
        return errors
    
    def _convert_holiday_to_boolean(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert holiday indicators to boolean type.
        
        Args:
            data: DataFrame containing is_holiday column
            
        Returns:
            DataFrame with is_holiday as boolean
        """
        result = data.copy()
        
        if 'is_holiday' in result.columns:
            if not pd.api.types.is_bool_dtype(result['is_holiday']):
                # Convert string values
                if pd.api.types.is_object_dtype(result['is_holiday']):
                    result['is_holiday'] = result['is_holiday'].map({
                        'true': True, 'True': True, '1': True, 1: True,
                        'false': False, 'False': False, '0': False, 0: False
                    })
                # Convert numeric values
                elif pd.api.types.is_numeric_dtype(result['is_holiday']):
                    result['is_holiday'] = result['is_holiday'].astype(bool)
        
        return result
    
    def _validate_price_data(self, data: pd.DataFrame) -> List[str]:
        """
        Validate price data is numeric and non-negative.
        
        Args:
            data: DataFrame containing price column
            
        Returns:
            List of error messages
        """
        errors = []
        
        if 'price' not in data.columns:
            return errors
        
        # Check if numeric
        if not pd.api.types.is_numeric_dtype(data['price']):
            errors.append(
                f"price must be numeric type, got {data['price'].dtype}"
            )
            return errors
        
        # Check for missing values
        if data['price'].isna().any():
            missing_count = data['price'].isna().sum()
            errors.append(
                f"price contains {missing_count} missing values. "
                f"All price values must be specified."
            )
        
        # Check for negative values
        negative_mask = data['price'] < 0
        if negative_mask.any():
            negative_count = negative_mask.sum()
            min_price = data['price'].min()
            errors.append(
                f"price contains {negative_count} negative values (min: {min_price}). "
                f"All prices must be non-negative."
            )
        
        # Check for infinite values
        if np.isinf(data['price']).any():
            inf_count = np.isinf(data['price']).sum()
            errors.append(
                f"price contains {inf_count} infinite values. "
                f"All prices must be finite."
            )
        
        return errors
    
    def _normalize_features(
        self,
        data: pd.DataFrame,
        fit: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
        """
        Normalize numeric features using standardization (z-score).
        
        Normalization formula: (x - mean) / std
        
        Args:
            data: DataFrame containing features
            fit: Whether to fit normalization parameters
            
        Returns:
            Tuple of (normalized DataFrame, normalization parameters)
        """
        result = data.copy()
        norm_params = {}
        
        # Features to normalize (numeric features only)
        numeric_features = ['price', 'sales_volume']
        
        for feature in numeric_features:
            if feature not in result.columns:
                continue
            
            if not pd.api.types.is_numeric_dtype(result[feature]):
                continue
            
            if fit:
                # Compute mean and std
                mean = result[feature].mean()
                std = result[feature].std()
                
                # Handle zero std (constant feature)
                if std == 0 or pd.isna(std):
                    std = 1.0
                
                norm_params[feature] = {'mean': float(mean), 'std': float(std)}
            else:
                # Use existing normalization parameters
                if feature not in self.normalization_params:
                    continue
                norm_params[feature] = self.normalization_params[feature]
            
            # Apply normalization
            mean = norm_params[feature]['mean']
            std = norm_params[feature]['std']
            result[f'{feature}_normalized'] = (result[feature] - mean) / std
        
        return result, norm_params
    
    def denormalize_feature(
        self,
        values: np.ndarray,
        feature_name: str
    ) -> np.ndarray:
        """
        Denormalize feature values back to original scale.
        
        Args:
            values: Normalized values
            feature_name: Name of the feature
            
        Returns:
            Denormalized values
        """
        if feature_name not in self.normalization_params:
            return values
        
        params = self.normalization_params[feature_name]
        mean = params['mean']
        std = params['std']
        
        return values * std + mean
    
    def get_normalization_params(self) -> Dict[str, Dict[str, float]]:
        """
        Get normalization parameters.
        
        Returns:
            Dictionary mapping feature names to normalization parameters
        """
        return self.normalization_params.copy()
    
    def set_normalization_params(self, params: Dict[str, Dict[str, float]]) -> None:
        """
        Set normalization parameters.
        
        Args:
            params: Dictionary mapping feature names to normalization parameters
        """
        self.normalization_params = params.copy()
