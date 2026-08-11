# Demand Forecasting System

A scalable machine learning platform that predicts future product demand using historical sales data, seasonality patterns, holiday effects, and price impacts.

## Setup

### Prerequisites
- Python 3.9+
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
│   ├── api/               # FastAPI application
│   ├── data/              # Data ingestion service
│   ├── training/          # Model training pipeline
│   ├── inference/         # Forecasting engine
│   ├── registry/          # Model registry
│   └── utils/             # Shared utilities
├── tests/                 # Test suite
├── config/                # Configuration files
├── scripts/               # Utility scripts
└── requirements.txt       # Python dependencies
```

## Running the Application

### Development Server
```bash
uvicorn src.api.main:app --reload
```

### Health Check
```bash
curl http://localhost:8000/api/v1/health
```

## Testing

Run all tests:
```bash
pytest
```

Run property-based tests:
```bash
pytest -m property
```

## AWS Resources

This system uses the following AWS services:
- **S3**: Historical datasets and model artifacts
- **RDS PostgreSQL**: Model registry database
- **Lambda**: API endpoints and data ingestion
- **SageMaker**: Custom model training
- **Amazon Forecast**: Benchmark model training
- **CloudWatch**: Logging and monitoring
