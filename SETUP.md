# Setup Guide - Demand Forecasting System

This guide walks through setting up the Demand Forecasting System infrastructure.

## Prerequisites

1. **Python 3.9+** installed
2. **AWS Account** with appropriate permissions:
   - S3 bucket creation
   - RDS database access
   - CloudWatch Logs access
3. **PostgreSQL Database** (RDS or local)
4. **AWS CLI** configured with credentials

## Step 1: Clone and Setup Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
make install
# Or: pip install -r requirements.txt
```

## Step 2: Configure Environment Variables

```bash
# Copy example environment file
cp config/.env.example config/.env

# Edit config/.env with your settings
# Required variables:
# - AWS credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
# - Database connection (DB_HOST, DB_PASSWORD, etc.)
# - S3 bucket names (must be globally unique)
```

### Important Configuration Notes

- **S3 Bucket Names**: Must be globally unique across all AWS accounts
- **Database**: Can use AWS RDS PostgreSQL or local PostgreSQL instance
- **AWS Region**: Ensure all resources are in the same region

## Step 3: Create AWS Resources

```bash
# Create S3 buckets and CloudWatch log group
make setup-aws
# Or: python scripts/setup_aws_resources.py
```

This script creates:
- S3 bucket for historical datasets (with lifecycle policy)
- S3 bucket for model artifacts (with versioning enabled)
- CloudWatch log group for application logs

## Step 4: Initialize Database

```bash
# Create database and schema
make init-db
# Or: python scripts/init_db.py
```

This script:
- Creates the database if it doesn't exist
- Creates the `models` table for Model Registry
- Creates the `training_jobs` table for tracking training pipelines
- Sets up indexes for efficient querying

## Step 5: Verify Installation

```bash
# Run tests to verify setup
make test
# Or: pytest

# Start development server
make run
# Or: uvicorn src.api.main:app --reload
```

## Step 6: Test Health Check

```bash
# In another terminal, test the health check endpoint
curl http://localhost:8000/api/v1/health

# Expected response:
# {
#   "status": "healthy",
#   "timestamp": "2025-01-15T10:30:00",
#   "environment": "development",
#   "version": "1.0.0",
#   "checks": {
#     "api": "healthy",
#     "database": "healthy",
#     "response_time_ms": 45.23
#   }
# }
```

## Troubleshooting

### Database Connection Issues

If you see database connection errors:

1. Verify database credentials in `config/.env`
2. Check database is running and accessible
3. Verify security group rules (for RDS)
4. Test connection manually:
   ```bash
   psql -h <DB_HOST> -U <DB_USER> -d <DB_NAME>
   ```

### AWS Permission Issues

If AWS resource creation fails:

1. Verify AWS credentials are configured:
   ```bash
   aws sts get-caller-identity
   ```
2. Check IAM permissions for S3, CloudWatch
3. Verify bucket names are globally unique

### Import Errors

If you see Python import errors:

1. Ensure virtual environment is activated
2. Verify all dependencies are installed:
   ```bash
   pip list
   ```
3. Check Python version (3.9+ required)

## Next Steps

After successful setup:

1. **Task 2**: Implement Data Ingestion Service
2. **Task 4**: Implement Feature Engineering
3. **Task 5**: Implement Model Registry

See `tasks.md` for the complete implementation plan.

## AWS Resource Cleanup

To remove AWS resources when done:

```bash
# Delete S3 buckets (WARNING: This deletes all data)
aws s3 rb s3://<bucket-name> --force

# Delete CloudWatch log group
aws logs delete-log-group --log-group-name /aws/demand-forecasting
```

## Production Deployment

For production deployment:

1. Update `config/.env` with production settings
2. Set `ENVIRONMENT=production`
3. Configure proper CORS origins in `src/api/main.py`
4. Set up API Gateway for authentication
5. Deploy to AWS Lambda using deployment scripts (Task 15)
6. Configure CloudWatch alarms (Task 17)
