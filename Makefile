.PHONY: help install setup-aws init-db test run clean

help:
	@echo "Demand Forecasting System - Available Commands"
	@echo ""
	@echo "  make install     - Install Python dependencies"
	@echo "  make setup-aws   - Create AWS resources (S3 buckets, CloudWatch)"
	@echo "  make init-db     - Initialize database schema"
	@echo "  make test        - Run all tests"
	@echo "  make test-unit   - Run unit tests only"
	@echo "  make test-prop   - Run property-based tests only"
	@echo "  make run         - Start development server"
	@echo "  make clean       - Remove generated files"

install:
	pip install -r requirements.txt

setup-aws:
	python scripts/setup_aws_resources.py

init-db:
	python scripts/init_db.py

test:
	pytest

test-unit:
	pytest -m unit

test-prop:
	pytest -m property

run:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
