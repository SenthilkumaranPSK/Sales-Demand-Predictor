"""
Example usage of ForecastingEngine for generating demand forecasts.

This example demonstrates:
1. Generating forecasts using a trained model
2. Providing future features (prices, holidays)
3. Generating multi-model forecasts for comparison
4. Accessing confidence intervals
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
from src.inference.forecasting_engine import forecasting_engine
from src.registry.model_registry import model_registry


def example_basic_forecast():
    """Example: Generate a basic forecast using a trained model."""
    print("=" * 60)
    print("Example 1: Basic Forecast Generation")
    print("=" * 60)
    
    try:
        # Get the latest model for a product
        model_id, metadata = model_registry.get_latest_model(
            product_id='product_001',
            model_type='custom'
        )
        
        print(f"\nUsing model: {model_id}")
        print(f"Model version: {metadata.version}")
        print(f"Training metrics - MAE: {metadata.mae:.2f}, RMSE: {metadata.rmse:.2f}, MAPE: {metadata.mape:.2f}%")
        
        # Generate forecast for next 30 days
        result = forecasting_engine.generate_forecast(
            model_id=model_id,
            forecast_horizon=30,
            start_date=datetime(2024, 2, 1)
        )
        
        print(f"\nForecast generated successfully!")
        print(f"Product: {result.product_id}")
        print(f"Forecast horizon: {len(result.predictions)} days")
        print(f"\nFirst 5 predictions:")
        for i in range(min(5, len(result.predictions))):
            timestamp = result.timestamps[i]
            prediction = result.predictions[i]
            ci_50 = result.confidence_intervals['50%']
            ci_90 = result.confidence_intervals['90%']
            
            print(f"  {timestamp.strftime('%Y-%m-%d')}: {prediction:.2f} "
                  f"(50% CI: [{ci_50.lower[i]:.2f}, {ci_50.upper[i]:.2f}], "
                  f"90% CI: [{ci_90.lower[i]:.2f}, {ci_90.upper[i]:.2f}])")
        
        print(f"\nConfidence intervals available: {list(result.confidence_intervals.keys())}")
        
    except ValueError as e:
        print(f"\nError: {e}")
        print("Make sure you have trained a model first using the training pipeline.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")


def example_forecast_with_future_features():
    """Example: Generate forecast with future price and holiday information."""
    print("\n" + "=" * 60)
    print("Example 2: Forecast with Future Features")
    print("=" * 60)
    
    try:
        # Get the latest model
        model_id, metadata = model_registry.get_latest_model(
            product_id='product_001',
            model_type='custom'
        )
        
        # Define future features for next 14 days
        forecast_horizon = 14
        
        # Future prices (e.g., planned price changes)
        future_prices = [99.99] * 7 + [89.99] * 7  # Price drop after 7 days
        
        # Future holidays (e.g., known holiday calendar)
        future_holidays = [False] * 6 + [True] + [False] * 7  # Holiday on day 7
        
        future_features = {
            'prices': future_prices,
            'holidays': future_holidays
        }
        
        print(f"\nGenerating forecast with future features:")
        print(f"  - Price schedule: ${future_prices[0]:.2f} for 7 days, then ${future_prices[7]:.2f}")
        print(f"  - Holiday on day 7")
        
        # Generate forecast
        result = forecasting_engine.generate_forecast(
            model_id=model_id,
            forecast_horizon=forecast_horizon,
            future_features=future_features,
            start_date=datetime(2024, 2, 1)
        )
        
        print(f"\nForecast generated successfully!")
        print(f"\nPredictions with future features:")
        for i in range(len(result.predictions)):
            timestamp = result.timestamps[i]
            prediction = result.predictions[i]
            price = future_prices[i]
            holiday = "Holiday" if future_holidays[i] else "Regular"
            
            print(f"  {timestamp.strftime('%Y-%m-%d')}: {prediction:.2f} units "
                  f"(Price: ${price:.2f}, {holiday})")
        
    except ValueError as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nUnexpected error: {e}")


def example_multi_model_forecast():
    """Example: Generate forecasts from multiple models for comparison."""
    print("\n" + "=" * 60)
    print("Example 3: Multi-Model Forecast Comparison")
    print("=" * 60)
    
    try:
        product_id = 'product_001'
        forecast_horizon = 7
        
        print(f"\nGenerating forecasts from all available models for {product_id}")
        
        # Generate forecasts from all models
        results = forecasting_engine.generate_multi_model_forecast(
            product_id=product_id,
            forecast_horizon=forecast_horizon,
            start_date=datetime(2024, 2, 1)
        )
        
        print(f"\nGenerated forecasts from {len(results)} models:")
        
        # Compare models
        for model_id, forecast in results.items():
            print(f"\n  Model: {model_id}")
            print(f"    Algorithm: {forecast.metadata.get('algorithm', 'unknown')}")
            print(f"    Training MAE: {forecast.metadata.get('training_mae', 0):.2f}")
            print(f"    Training RMSE: {forecast.metadata.get('training_rmse', 0):.2f}")
            print(f"    Average prediction: {sum(forecast.predictions) / len(forecast.predictions):.2f}")
            print(f"    Min prediction: {min(forecast.predictions):.2f}")
            print(f"    Max prediction: {max(forecast.predictions):.2f}")
        
        # Compare first day predictions across models
        print(f"\n  First day predictions comparison:")
        for model_id, forecast in results.items():
            first_pred = forecast.predictions[0]
            ci_90 = forecast.confidence_intervals['90%']
            print(f"    {model_id}: {first_pred:.2f} "
                  f"(90% CI: [{ci_90.lower[0]:.2f}, {ci_90.upper[0]:.2f}])")
        
    except ValueError as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nUnexpected error: {e}")


def example_confidence_intervals():
    """Example: Working with confidence intervals."""
    print("\n" + "=" * 60)
    print("Example 4: Understanding Confidence Intervals")
    print("=" * 60)
    
    try:
        # Get the latest model
        model_id, metadata = model_registry.get_latest_model(
            product_id='product_001',
            model_type='custom'
        )
        
        # Generate forecast
        result = forecasting_engine.generate_forecast(
            model_id=model_id,
            forecast_horizon=7,
            start_date=datetime(2024, 2, 1)
        )
        
        print(f"\nConfidence Intervals Explanation:")
        print(f"  - 50% CI: There's a 50% probability the actual value falls within this range")
        print(f"  - 80% CI: There's an 80% probability the actual value falls within this range")
        print(f"  - 90% CI: There's a 90% probability the actual value falls within this range")
        
        print(f"\nDay 1 forecast breakdown:")
        prediction = result.predictions[0]
        print(f"  Point prediction: {prediction:.2f}")
        
        for level in ['50%', '80%', '90%']:
            ci = result.confidence_intervals[level]
            lower = ci.lower[0]
            upper = ci.upper[0]
            width = upper - lower
            print(f"  {level} interval: [{lower:.2f}, {upper:.2f}] (width: {width:.2f})")
        
        print(f"\nInterpretation:")
        print(f"  - We're most confident the actual demand will be around {prediction:.2f}")
        print(f"  - There's a 90% chance it will be between "
              f"{result.confidence_intervals['90%'].lower[0]:.2f} and "
              f"{result.confidence_intervals['90%'].upper[0]:.2f}")
        
    except ValueError as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nUnexpected error: {e}")


def example_error_handling():
    """Example: Error handling scenarios."""
    print("\n" + "=" * 60)
    print("Example 5: Error Handling")
    print("=" * 60)
    
    # Test invalid forecast horizon
    print("\n1. Testing invalid forecast horizon:")
    try:
        forecasting_engine.generate_forecast(
            model_id='any_model',
            forecast_horizon=100  # Invalid: > 90
        )
    except ValueError as e:
        print(f"   ✓ Caught expected error: {e}")
    
    # Test non-existent model
    print("\n2. Testing non-existent model:")
    try:
        forecasting_engine.generate_forecast(
            model_id='nonexistent_model_12345',
            forecast_horizon=7
        )
    except ValueError as e:
        print(f"   ✓ Caught expected error: {e}")
    
    # Test non-existent product
    print("\n3. Testing non-existent product:")
    try:
        forecasting_engine.generate_multi_model_forecast(
            product_id='nonexistent_product',
            forecast_horizon=7
        )
    except ValueError as e:
        print(f"   ✓ Caught expected error: {e}")
    
    print("\n✓ All error handling tests passed!")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("ForecastingEngine Usage Examples")
    print("=" * 60)
    print("\nNote: These examples require trained models in the Model Registry.")
    print("Run the training pipeline examples first if you haven't already.")
    
    # Run examples
    example_basic_forecast()
    example_forecast_with_future_features()
    example_multi_model_forecast()
    example_confidence_intervals()
    example_error_handling()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
