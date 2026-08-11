# Implementation Plan: Demand Forecasting System

## Overview

This implementation plan builds a scalable demand forecasting system using Python, FastAPI, AWS services (S3, RDS, Lambda, SageMaker, Amazon Forecast), and modern ML libraries. The system follows a modular architecture with data ingestion, model training, inference serving, and visualization components. Implementation proceeds incrementally with property-based testing to validate correctness properties from the design.

## Tasks

- [x] 1. Set up project structure and core infrastructure
  - Create Python project with virtual environment and dependency management (requirements.txt or poetry)
  - Set up project directory structure: `src/`, `tests/`, `config/`, `scripts/`
  - Configure AWS SDK (boto3) with credentials and region settings
  - Create S3 buckets for historical datasets and model artifacts
  - Set up RDS PostgreSQL database for Model Registry with connection pooling
  - Initialize database schema for models table (model_id, product_id, model_type, version, metrics, etc.)
  - Configure logging with structured JSON output to CloudWatch Logs
  - Create health check endpoint skeleton
  - _Requirements: 7.1, 7.4, 7.5, 10.1_

- [x] 2. Implement Data Ingestion Service
  - [x] 2.1 Create data validation module
    - Implement schema validation for required columns (timestamp, product_id, sales_volume, price, is_holiday, seasonality features)
    - Implement data type validation (datetime, string, numeric, boolean)
    - Implement value range validation (non-negative sales/price, valid date ranges)
    - Implement duplicate detection for (product_id, timestamp) pairs
    - Create ValidationResult and ValidationError data classes
    - _Requirements: 1.1, 1.2, 1.3, 9.5_
  
  - [ ]* 2.2 Write property test for schema validation completeness
    - **Property 1: Schema Validation Completeness**
    - **Validates: Requirements 1.1, 9.5**
    - Generate random datasets with missing columns, incorrect types, invalid ranges
    - Verify validation identifies all violations
  
  - [ ]* 2.3 Write property test for validation error message accuracy
    - **Property 2: Validation Error Message Accuracy**
    - **Validates: Requirements 1.2**
    - Generate datasets with known violations
    - Verify error messages contain correct field names and error types
  
  - [ ]* 2.4 Write property test for valid data acceptance
    - **Property 3: Valid Data Acceptance**
    - **Validates: Requirements 1.3**
    - Generate random valid datasets with all required fields
    - Verify acceptance without validation errors
  
  - [x] 2.5 Implement batch ingestion with CSV and JSON parsing
    - Create DataIngestionService class with ingest_batch method
    - Implement CSV parsing using pandas
    - Implement JSON parsing with validation
    - Add format detection and error handling
    - Store validated data to S3 in Parquet format partitioned by product_id and year
    - Implement 5-second performance target for 1M records
    - _Requirements: 1.4, 1.5, 7.4_
  
  - [ ]* 2.6 Write property test for format parsing round-trip
    - **Property 4: Format Parsing Round-Trip**
    - **Validates: Requirements 1.5**
    - Generate random valid datasets
    - Serialize to CSV/JSON and parse back
    - Verify equivalence of original and parsed data
  
  - [ ]* 2.7 Write unit tests for data ingestion edge cases
    - Test empty datasets, single-record datasets
    - Test maximum batch size (5M records)
    - Test malformed CSV/JSON handling

- [x] 3. Checkpoint - Verify data ingestion functionality
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Implement feature engineering module
  - [x] 4.1 Create seasonality feature extraction
    - Implement extract_seasonality_features function
    - Extract day_of_week (0-6) from timestamp
    - Extract month (1-12) from timestamp
    - Extract quarter (1-4) from timestamp
    - Extract season categorical (spring, summer, fall, winter)
    - _Requirements: 9.1, 9.4_
  
  - [ ]* 4.2 Write property test for seasonality feature extraction
    - **Property 10: Seasonality Feature Extraction**
    - **Validates: Requirements 9.1**
    - Generate random timestamps across multiple years
    - Verify day_of_week, month, quarter match calendar position
  
  - [x] 4.3 Implement holiday and price feature processing
    - Create feature preprocessing pipeline
    - Validate holiday indicators are boolean
    - Validate price data is numeric and non-negative
    - Implement feature normalization for model training
    - _Requirements: 9.2, 9.3_

- [x] 5. Implement Model Registry
  - [x] 5.1 Create Model Registry database interface
    - Implement ModelRegistry class with register_model, get_model, list_models, get_latest_model methods
    - Create ModelMetadata dataclass
    - Implement database connection with connection pooling using psycopg2 or SQLAlchemy
    - Implement model artifact storage to S3 with versioning
    - Implement model metadata storage to RDS PostgreSQL
    - Add indexing on (product_id, model_type) and created_at
    - _Requirements: 2.3, 7.5, 8.5_
  
  - [ ]* 5.2 Write unit tests for Model Registry operations
    - Test model registration and retrieval
    - Test version management (100+ versions per model)
    - Test filtering by product_id and model_type
    - Test latest model retrieval

- [x] 6. Implement performance metrics calculation
  - [x] 6.1 Create metrics computation module
    - Implement compute_mae function (Mean Absolute Error)
    - Implement compute_rmse function (Root Mean Squared Error)
    - Implement compute_mape function (Mean Absolute Percentage Error)
    - Create PerformanceMetrics dataclass
    - Handle edge cases (empty arrays, division by zero in MAPE)
    - _Requirements: 2.2, 8.2_
  
  - [ ]* 6.2 Write property test for metric calculation correctness
    - **Property 5: Metric Calculation Correctness**
    - **Validates: Requirements 2.2, 8.2**
    - Generate random prediction/actual pairs
    - Verify MAE, RMSE, MAPE match mathematical definitions
  
  - [x] 6.3 Implement model comparison report generation
    - Create generate_comparison_report function
    - Calculate percentage improvements for MAE, RMSE, MAPE
    - Generate recommendation (use custom vs. benchmark)
    - Create ComparisonReport dataclass
    - _Requirements: 3.4, 8.3_
  
  - [ ]* 6.4 Write property test for comparison report calculation
    - **Property 6: Comparison Report Calculation**
    - **Validates: Requirements 3.4, 8.3**
    - Generate random metric pairs
    - Verify improvement calculations match formula: ((benchmark - custom) / benchmark) * 100

- [x] 7. Checkpoint - Verify metrics and registry functionality
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Implement Custom Model Training Pipeline
  - [x] 8.1 Create training data preparation
    - Implement train/validation split (80/20)
    - Load historical dataset from S3
    - Apply feature engineering pipeline
    - Create training dataset with features (sales, price, holidays, seasonality)
    - _Requirements: 2.1, 9.1, 9.2, 9.3_
  
  - [x] 8.2 Implement custom model trainer
    - Create CustomModelTrainer class
    - Implement model training using Prophet or scikit-learn
    - Configure hyperparameters from ModelConfig
    - Implement model serialization using joblib or pickle
    - Generate predictions on validation set for backtesting
    - Compute performance metrics (MAE, RMSE, MAPE)
    - _Requirements: 2.1, 2.2, 8.2_
  
  - [x] 8.3 Implement training pipeline orchestration
    - Create TrainingPipeline class with train_custom_model method
    - Orchestrate data loading, feature engineering, training, evaluation
    - Register trained model in Model Registry with metadata
    - Implement error handling and logging for training failures
    - Add retry logic for transient failures
    - _Requirements: 2.3, 2.4, 2.5, 10.3_
  
  - [ ]* 8.4 Write unit tests for training pipeline
    - Test training with synthetic data
    - Test model registration after successful training
    - Test error handling for training failures
    - Test retraining with updated datasets

- [x] 9. Implement Amazon Forecast integration
  - [x] 9.1 Create Amazon Forecast dataset import
    - Implement dataset upload to S3 in Forecast-compatible format
    - Create Forecast dataset group using boto3
    - Import historical data into Forecast dataset
    - Configure target time series (sales_volume) and related features (price, holiday, seasonality)
    - _Requirements: 3.1, 3.2_
  
  - [x] 9.2 Implement Forecast predictor training
    - Create train_forecast_predictor method in TrainingPipeline
    - Configure Forecast predictor with AutoML or specific algorithm
    - Set forecast horizon parameter
    - Poll for training completion with exponential backoff
    - Retrieve predictor metrics from Forecast API
    - _Requirements: 3.1, 3.2, 3.3_
  
  - [x] 9.3 Implement Forecast error handling and fallback
    - Add try-except blocks for Forecast API calls
    - Log Forecast training failures without blocking custom model training
    - Implement exponential backoff retry logic for transient failures
    - Register Forecast predictor in Model Registry when successful
    - _Requirements: 3.5, 10.3_
  
  - [ ]* 9.4 Write property test for exponential backoff retry timing
    - **Property 12: Exponential Backoff Retry Timing**
    - **Validates: Requirements 10.3**
    - Generate random failure sequences
    - Verify retry delays follow formula: base_delay * (2 ^ attempt_number)
  
  - [ ]* 9.5 Write integration tests for Amazon Forecast
    - Test predictor creation with mock Forecast API
    - Test training completion polling
    - Test error handling for Forecast failures

- [x] 10. Checkpoint - Verify training pipeline functionality
  - Ensure all tests pass, ask the user if questions arise.

- [x] 11. Implement Forecasting Engine
  - [x] 11.1 Create forecast generation for custom models
    - Implement ForecastingEngine class with generate_forecast method
    - Load model artifact from Model Registry
    - Prepare future features (holidays, seasonality) for forecast horizon
    - Generate point predictions using loaded model
    - Implement confidence interval calculation using quantile regression or bootstrap
    - Create ForecastResult dataclass with predictions and confidence intervals (50%, 80%, 90%)
    - _Requirements: 4.1, 4.3, 4.4, 9.4_
  
  - [x] 11.2 Create forecast generation for Amazon Forecast models
    - Implement Forecast query using boto3 create_forecast and query_forecast APIs
    - Extract quantile forecasts (p10, p50, p90) from Forecast response
    - Convert Forecast quantiles to confidence intervals
    - Handle Forecast API errors and unavailability
    - _Requirements: 4.1, 4.2, 4.3_
  
  - [x] 11.3 Implement multi-model forecast generation
    - Create generate_multi_model_forecast method
    - Retrieve all available models for product from Model Registry
    - Generate forecasts from custom and benchmark models
    - Return dictionary mapping model_id to ForecastResult
    - _Requirements: 4.2, 8.1_
  
  - [ ]* 11.4 Write property test for forecast result structure completeness
    - **Property 7: Forecast Result Structure Completeness**
    - **Validates: Requirements 4.3, 5.3**
    - Generate random forecast results
    - Verify all required fields present (timestamps, predictions, confidence intervals, metadata)
  
  - [x] 11.5 Implement forecast validation and error handling
    - Validate forecast_horizon is between 1 and 90 days
    - Return descriptive errors for missing models
    - Implement 2-second response time target for 90-day horizon
    - _Requirements: 4.4, 4.5, 10.4_
  
  - [ ]* 11.6 Write unit tests for Forecasting Engine
    - Test forecast generation with mock models
    - Test confidence interval calculation
    - Test multi-model forecast generation
    - Test error handling for missing models

- [x] 12. Implement Inference API
  - [x] 12.1 Create FastAPI application structure
    - Initialize FastAPI app with CORS middleware
    - Configure OpenAPI documentation
    - Set up request/response models using Pydantic (ForecastRequest, ForecastResponse)
    - Implement dependency injection for ForecastingEngine and Model Registry
    - _Requirements: 5.1, 5.2, 5.3_
  
  - [x] 12.2 Implement forecast endpoint
    - Create POST /api/v1/forecast endpoint
    - Validate request parameters (product_id, forecast_horizon, model_id)
    - Call ForecastingEngine.generate_forecast or generate_multi_model_forecast
    - Format response with predictions, confidence intervals, metadata
    - Implement 3-second response time target
    - _Requirements: 5.1, 5.2, 5.3, 7.2_
  
  - [ ]* 12.3 Write property test for API request parameter validation
    - **Property 8: API Request Parameter Validation**
    - **Validates: Requirements 5.1, 10.4**
    - Generate random valid/invalid API requests
    - Verify validation identifies invalid product_ids, out-of-range horizons, missing fields
  
  - [ ]* 12.4 Write property test for API error response format
    - **Property 9: API Error Response Format**
    - **Validates: Requirements 5.4**
    - Generate random invalid requests
    - Verify HTTP 400 status with JSON error response containing descriptive message
  
  - [x] 12.5 Implement API authentication
    - Add API key authentication middleware
    - Validate X-API-Key header on each request
    - Return HTTP 401 for missing or invalid API keys
    - Integrate with AWS API Gateway for key management
    - _Requirements: 5.5_
  
  - [x] 12.6 Implement error handling and HTTP status codes
    - Create error handler middleware
    - Map validation errors to HTTP 400
    - Map missing resources to HTTP 404
    - Map internal errors to HTTP 500
    - Map service unavailability to HTTP 503
    - Return structured JSON error responses with error code, message, details
    - _Requirements: 5.4, 5.7, 10.2_
  
  - [ ]* 12.7 Write property test for HTTP status code mapping
    - **Property 11: HTTP Status Code Mapping**
    - **Validates: Requirements 10.2**
    - Generate random error conditions
    - Verify correct HTTP status codes (400, 404, 500, 503)
  
  - [x] 12.8 Implement rate limiting
    - Add rate limiting middleware (1000 requests/minute)
    - Return HTTP 429 when rate limit exceeded
    - Implement concurrent request limiting (100 concurrent)
    - _Requirements: 5.6, 7.2, 7.3_
  
  - [x] 12.9 Create additional API endpoints
    - Implement GET /api/v1/forecast/{forecast_id} for retrieving cached forecasts
    - Implement GET /api/v1/models for listing available models
    - Implement GET /api/v1/models/{model_id} for model metadata
    - Implement POST /api/v1/data/ingest for data ingestion
    - Implement GET /api/v1/health for health checks (1-second response)
    - _Requirements: 10.5_
  
  - [ ]* 12.10 Write integration tests for Inference API
    - Test end-to-end forecast request flow
    - Test authentication with valid/invalid API keys
    - Test rate limiting behavior
    - Test error responses for various failure scenarios

- [x] 13. Checkpoint - Verify Inference API functionality
  - Ensure all tests pass, ask the user if questions arise.

- [x] 14. Implement Analytics Dashboard
  - [x] 14.1 Set up React application structure
    - Initialize React project with TypeScript
    - Configure build system (Vite or Create React App)
    - Set up routing with React Router
    - Configure API client with axios or fetch
    - Create dashboard layout components
    - _Requirements: 6.1, 6.2, 6.3, 6.4_
  
  - [x] 14.2 Implement time-series chart component
    - Create ForecastChart component using Recharts or D3.js
    - Display historical sales data as line chart
    - Display forecast predictions as line chart
    - Render confidence intervals as shaded regions (50%, 80%, 90%)
    - Add axis labels, legend, and tooltips
    - _Requirements: 6.1, 6.2_
  
  - [x] 14.3 Implement model comparison view
    - Create ModelComparison component
    - Display custom model forecast and benchmark forecast side-by-side
    - Show performance metrics table (MAE, RMSE, MAPE)
    - Highlight better-performing model
    - _Requirements: 6.3, 6.4, 8.4_
  
  - [x] 14.4 Implement product selector and controls
    - Create product dropdown selector
    - Create forecast horizon slider (1-90 days)
    - Create date range filter
    - Create model type filter (custom, benchmark, both)
    - Wire controls to API requests
    - _Requirements: 6.5, 6.6_
  
  - [x] 14.5 Implement API integration
    - Create API service module for calling Inference API
    - Implement forecast data fetching with loading states
    - Implement error handling and error message display
    - Implement 5-second response time target for data fetching and rendering
    - Add API key configuration
    - _Requirements: 6.5, 6.7_
  
  - [ ]* 14.6 Write unit tests for dashboard components
    - Test chart rendering with mock data
    - Test user interactions (product selection, horizon adjustment)
    - Test error message display
    - Test API integration with mock responses

- [x] 15. Implement AWS Lambda deployment
  - [x] 15.1 Create Lambda function for Inference API
    - Package FastAPI application for Lambda using Mangum adapter
    - Create Lambda deployment package with dependencies
    - Configure Lambda function with appropriate memory and timeout settings
    - Set up environment variables for database connection, S3 buckets, API keys
    - _Requirements: 7.2_
  
  - [x] 15.2 Create Lambda function for Data Ingestion
    - Package Data Ingestion Service for Lambda
    - Configure S3 trigger for automatic ingestion on file upload
    - Set up Lambda execution role with S3 and RDS permissions
    - _Requirements: 7.4_
  
  - [x] 15.3 Configure API Gateway
    - Create API Gateway REST API
    - Configure routes to Lambda functions
    - Set up API key authentication
    - Configure CORS settings
    - Enable request/response logging
    - _Requirements: 5.5_

- [x] 16. Implement AWS Step Functions for Training Pipeline
  - [x] 16.1 Create Step Functions state machine
    - Define state machine for training workflow (data load → feature engineering → custom training → Forecast training → comparison)
    - Configure Lambda functions for each training step
    - Configure SageMaker training job integration for custom models
    - Add error handling and retry logic to state machine
    - _Requirements: 2.1, 2.5, 3.1_
  
  - [x] 16.2 Create training orchestration Lambda
    - Implement Lambda function to trigger Step Functions execution
    - Pass training configuration and dataset references
    - Monitor training progress and handle completion notifications
    - _Requirements: 2.4_

- [x] 17. Implement monitoring and logging
  - [x] 17.1 Configure CloudWatch Logs
    - Set up log groups for each component (API, Training, Ingestion)
    - Configure structured JSON logging format
    - Add log retention policies
    - _Requirements: 10.1_
  
  - [x] 17.2 Implement health check endpoints
    - Create health check for Inference API (service status, database connectivity)
    - Create health check for Forecasting Engine (model availability)
    - Ensure 1-second response time for health checks
    - _Requirements: 10.5_
  
  - [x] 17.3 Set up CloudWatch metrics and alarms
    - Create custom metrics for API latency, error rates, request counts
    - Create alarms for high error rates, slow responses, service unavailability
    - Configure SNS notifications for alarm triggers
    - _Requirements: 7.2, 7.3_

- [x] 18. Deploy Analytics Dashboard
  - [x] 18.1 Build and deploy dashboard to S3
    - Build React application for production
    - Upload build artifacts to S3 bucket
    - Configure S3 bucket for static website hosting
    - _Requirements: 6.1_
  
  - [x] 18.2 Configure CloudFront CDN
    - Create CloudFront distribution pointing to S3 bucket
    - Configure caching policies for static assets
    - Set up custom domain and SSL certificate
    - _Requirements: 6.5_

- [x] 19. Final integration and end-to-end testing
  - [ ]* 19.1 Write end-to-end tests for complete workflows
    - Test complete training pipeline: data ingestion → training → model registration
    - Test complete inference workflow: API request → model loading → prediction → response
    - Test model comparison workflow: train custom + benchmark → generate comparison report
    - Test dashboard workflow: user interaction → API calls → visualization
  
  - [ ]* 19.2 Perform load testing
    - Test Inference API with 1000 requests/minute sustained load
    - Test data ingestion with 5 million record batches
    - Test concurrent forecast requests with 100 simultaneous users
    - Verify performance targets are met
    - _Requirements: 7.1, 7.2, 7.3, 7.4_
  
  - [x] 19.3 Verify all requirements are satisfied
    - Review each requirement's acceptance criteria
    - Confirm all functionality is implemented and tested
    - Document any deviations or limitations

- [x] 20. Final checkpoint - Complete system verification
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- Integration tests validate AWS service interactions
- The system uses Python with FastAPI, boto3, scikit-learn/Prophet, and React
- AWS services: S3, RDS PostgreSQL, Lambda, SageMaker, Amazon Forecast, API Gateway, Step Functions, CloudWatch
- Checkpoints ensure incremental validation throughout implementation
