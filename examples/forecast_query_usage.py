"""
Example usage of Amazon Forecast query functionality in ForecastingEngine.

This example demonstrates how to generate forecasts from Amazon Forecast predictors
that have been registered in the Model Registry.
"""

from datetime import datetime
from src.inference.forecasting_engine import forecasting_engine
from src.registry.model_registry import model_registry


def example_query_amazon_forecast():
    """
    Example: Query Amazon Forecast predictor for demand predictions.
    
    Prerequisites:
    - Amazon Forecast predictor has been trained (see training_pipeline_forecast_usage.py)
    - Predictor has been registered in Model Registry with model_type='forecast'
    - Predictor ARN is stored in model metadata hyperparameters
    """
    
    print("=" * 60)
    print("Amazon Forecast Query Example")
    print("=" * 60)
    
    # List available Forecast models
    print("\n1. Listing available Amazon Forecast models...")
    forecast_models = model_registry.list_models(model_type='forecast')
    
    if not forecast_models:
        print("No Amazon Forecast models found in registry.")
        print("Please train a Forecast predictor first using training_pipeline_forecast_usage.py")
        return
    
    print(f"Found {len(forecast_models)} Forecast model(s):")
    for model in forecast_models:
        print(f"  - {model.model_id} (product: {model.product_id}, MAE: {model.mae:.2f})")
    
    # Select first model for demonstration
    model = forecast_models[0]
    model_id = model.model_id
    product_id = model.product_id
    
    print(f"\n2. Generating forecast using model: {model_id}")
    print(f"   Product: {product_id}")
    print(f"   Predictor ARN: {model.hyperparameters.get('predictor_arn', 'N/A')}")
    
    # Generate 7-day forecast
    forecast_horizon = 7
    start_date = datetime(2024, 1, 1)
    
    print(f"\n3. Querying Amazon Forecast for {forecast_horizon}-day forecast...")
    print(f"   Start date: {start_date.strftime('%Y-%m-%d')}")
    
    try:
        result = forecasting_engine.generate_forecast(
            model_id=model_id,
            forecast_horizon=forecast_horizon,
            start_date=start_date
        )
        
        print("\n4. Forecast Results:")
        print(f"   Model ID: {result.model_id}")
        print(f"   Product ID: {result.product_id}")
        print(f"   Algorithm: {result.metadata['algorithm']}")
        print(f"   Number of predictions: {len(result.predictions)}")
        
        print("\n5. Predictions:")
        for timestamp, prediction in zip(result.timestamps, result.predictions):
            print(f"   {timestamp.strftime('%Y-%m-%d')}: {prediction:.2f}")
        
        print("\n6. Confidence Intervals:")
        for level, ci in result.confidence_intervals.items():
            print(f"\n   {level} Confidence Interval:")
            for i, (timestamp, lower, upper) in enumerate(zip(result.timestamps, ci.lower, ci.upper)):
                print(f"     {timestamp.strftime('%Y-%m-%d')}: [{lower:.2f}, {upper:.2f}]")
        
        print("\n7. Forecast Metadata:")
        print(f"   Training MAE: {result.metadata['training_mae']:.2f}")
        print(f"   Training RMSE: {result.metadata['training_rmse']:.2f}")
        print(f"   Training MAPE: {result.metadata['training_mape']:.2f}%")
        print(f"   Forecast ARN: {result.metadata.get('forecast_arn', 'N/A')}")
        
        print("\n✓ Forecast generation completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error generating forecast: {str(e)}")
        print("\nPossible causes:")
        print("  - Forecast predictor not found or not active")
        print("  - AWS credentials not configured")
        print("  - Predictor ARN missing from model metadata")
        print("  - Network connectivity issues")


def example_compare_custom_vs_forecast():
    """
    Example: Compare custom model forecast with Amazon Forecast.
    
    This demonstrates generating forecasts from both custom and Forecast models
    for the same product to compare performance.
    """
    
    print("\n" + "=" * 60)
    print("Custom vs Amazon Forecast Comparison")
    print("=" * 60)
    
    # Select a product that has both custom and Forecast models
    product_id = 'product_abc'
    
    print(f"\n1. Generating forecasts for product: {product_id}")
    
    try:
        # Generate multi-model forecast
        results = forecasting_engine.generate_multi_model_forecast(
            product_id=product_id,
            forecast_horizon=7,
            start_date=datetime(2024, 1, 1)
        )
        
        print(f"\n2. Generated forecasts from {len(results)} model(s):")
        
        custom_results = []
        forecast_results = []
        
        for model_id, result in results.items():
            algorithm = result.metadata.get('algorithm', 'unknown')
            print(f"   - {model_id}: {algorithm}")
            
            if algorithm == 'amazon_forecast':
                forecast_results.append(result)
            else:
                custom_results.append(result)
        
        print("\n3. Comparison:")
        
        if custom_results and forecast_results:
            custom = custom_results[0]
            forecast = forecast_results[0]
            
            print("\n   Custom Model:")
            print(f"     Algorithm: {custom.metadata['algorithm']}")
            print(f"     MAE: {custom.metadata['training_mae']:.2f}")
            print(f"     RMSE: {custom.metadata['training_rmse']:.2f}")
            print(f"     Sample predictions: {custom.predictions[:3]}")
            
            print("\n   Amazon Forecast:")
            print(f"     Algorithm: {forecast.metadata['algorithm']}")
            print(f"     MAE: {forecast.metadata['training_mae']:.2f}")
            print(f"     RMSE: {forecast.metadata['training_rmse']:.2f}")
            print(f"     Sample predictions: {forecast.predictions[:3]}")
            
            # Calculate prediction differences
            diffs = [abs(c - f) for c, f in zip(custom.predictions, forecast.predictions)]
            avg_diff = sum(diffs) / len(diffs)
            
            print(f"\n   Average prediction difference: {avg_diff:.2f}")
            
            if custom.metadata['training_mae'] < forecast.metadata['training_mae']:
                print("   → Custom model has better MAE")
            else:
                print("   → Amazon Forecast has better MAE")
        
        else:
            print("   Need both custom and Forecast models for comparison")
        
        print("\n✓ Comparison completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error during comparison: {str(e)}")


def example_handle_forecast_errors():
    """
    Example: Demonstrate error handling for Amazon Forecast queries.
    
    Shows how to handle common errors when querying Forecast predictors.
    """
    
    print("\n" + "=" * 60)
    print("Amazon Forecast Error Handling")
    print("=" * 60)
    
    # Test 1: Missing predictor ARN
    print("\n1. Testing missing predictor ARN...")
    try:
        # This would fail if model doesn't have predictor_arn in metadata
        result = forecasting_engine.generate_forecast(
            model_id='invalid_forecast_model',
            forecast_horizon=7
        )
    except ValueError as e:
        print(f"   ✓ Caught expected error: {str(e)}")
    except Exception as e:
        print(f"   ✓ Caught error: {type(e).__name__}: {str(e)}")
    
    # Test 2: Invalid forecast horizon
    print("\n2. Testing invalid forecast horizon...")
    try:
        result = forecasting_engine.generate_forecast(
            model_id='any_model',
            forecast_horizon=100  # > 90 days
        )
    except ValueError as e:
        print(f"   ✓ Caught expected error: {str(e)}")
    
    # Test 3: Forecast not found
    print("\n3. Testing forecast not found...")
    print("   (This would occur if predictor exists but forecast creation fails)")
    print("   Error would be: RuntimeError: Forecast not found or not available")
    
    print("\n✓ Error handling demonstration completed!")


if __name__ == '__main__':
    # Run examples
    example_query_amazon_forecast()
    example_compare_custom_vs_forecast()
    example_handle_forecast_errors()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)

