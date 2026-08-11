# Demand Forecasting System

A scalable machine learning platform that predicts future product demand using historical sales data, seasonality patterns, holiday effects, and price impacts.

## Features

- **Forecast Generation**: Generate demand forecasts with confidence intervals (50%, 80%, 90%)
- **Model Management**: Register, version, and query custom and Amazon Forecast benchmark models
- **Multi-Model Comparison**: Compare predictions across all trained models for a product
- **Data Ingestion**: Validated batch ingestion of historical sales data (CSV/JSON) with S3 storage
- **Security**: API key authentication (`X-API-Key` header) and per-client rate limiting
- **Health Monitoring**: Real-time service health checks

## Setup

### Prerequisites
- Python 3.9+ (developed on 3.11)
- AWS Account with credentials configured
- PostgreSQL database (RDS)

### Installation

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp config/.env.example config/.env
# Edit config/.env with your AWS credentials and database settings
```

4. Initialize database:
```bash
python scripts/init_db.py
```

## Project Structure

```
.
├── src/                    # Source code
│   ├── api/               # FastAPI application (routes, auth, rate limiting, error handling)
│   ├── data/              # Data ingestion and validation services
│   ├── features/          # Feature preprocessing and seasonality extraction
│   ├── training/          # Model training pipeline and Amazon Forecast integration
│   ├── inference/         # Forecasting engine with confidence intervals
│   ├── registry/          # Model registry (S3 artifacts + PostgreSQL metadata)
│   └── utils/             # Shared utilities and logging
├── tests/                 # Unit, integration, and performance test suite
├── examples/              # Usage examples for each component
├── config/                # Settings and environment configuration
├── scripts/               # Setup and initialization scripts
├── .kiro/specs/           # Design, requirements, and task specifications
└── requirements.txt       # Python dependencies
```

## API Endpoints

| Method | Path                      | Auth | Description                                      |
|--------|---------------------------|------|--------------------------------------------------|
| GET    | `/api/v1/health`          | No   | Service health check                             |
| POST   | `/api/v1/forecast`        | Yes  | Generate demand forecast with confidence intervals |
| GET    | `/api/v1/models`          | Yes  | List models (filter by product_id / model_type)  |
| GET    | `/api/v1/models/{id}`     | Yes  | Get detailed model metadata                      |
| POST   | `/api/v1/data/ingest`     | Yes  | Ingest historical sales data (CSV/JSON)          |
| GET    | `/api/docs`               | No   | Interactive OpenAPI documentation (Swagger UI)   |

All endpoints except health checks require an API key in the `X-API-Key` header. If no `API_KEYS` environment variable is configured, a development key is used (see `src/api/auth.py`).

## Running the Application

### Development Server
```bash
uvicorn src.api.main:app --reload
```

### Health Check
```bash
curl http://localhost:8000/api/v1/health
```

### Generate a Forecast
```bash
curl -X POST http://localhost:8000/api/v1/forecast \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <your-api-key>" \
  -d '{"product_id": "PROD-12345", "forecast_horizon": 30}'
```

## Testing

Run the full test suite:
```bash
pytest
```

Run tests by category (defined in `pytest.ini`):
```bash
pytest -m unit          # Unit tests
pytest -m integration   # Integration tests (AWS/mocked services)
pytest -m property      # Property-based tests using Hypothesis
pytest -m slow          # Slow/perfomance tests
```

## Examples

Run usage examples for individual components:
```bash
python examples/forecasting_engine_usage.py
python examples/training_pipeline_usage.py
python examples/model_registry_usage.py
```

## AWS Resources

This system uses the following AWS services:
- **S3**: Historical datasets and model artifacts
- **RDS PostgreSQL**: Model registry database
- **Lambda**: API endpoints and data ingestion
- **SageMaker**: Custom model training
- **Amazon Forecast**: Benchmark model training
- **CloudWatch**: Logging and monitoring
