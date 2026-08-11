"""
Performance tests for data ingestion service.

Tests the 5-second performance target for 1M records.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock

from src.data.ingestion import DataIngestionService
from src.data.validation import DataValidator


class TestDataIngestionPerformance:
    """Performance test suite for DataIngestionService."""
    
    @pytest.fixture
    def mock_s3_client(self):
        """Create a mock S3 client for testing."""
        mock_client = Mock()
        mock_client.put_object = Mock(return_value={'ETag': 'test-etag'})
        return mock_client
    
    @pytest.fixture
    def validator(self):
        """Create a DataValidator instance for testing."""
        return DataValidator()
    
    @pytest.fixture
    def ingestion_service(self, mock_s3_client, validator):
        """Create a DataIngestionService instance with mocked S3."""
        return DataIngestionService(s3_client=mock_s3_client, validator=validator)
    
    def generate_large_dataset(self, num_records: int) -> pd.DataFrame:
        """
        Generate a large valid dataset for performance testing.
        
        Args:
            num_records: Number of records to generate
            
        Returns:
            DataFrame with specified number of records
        """
        # Generate dates - ensure unique (product_id, timestamp) pairs
        # Use 10 products, so each product gets num_records/10 unique timestamps
        start_date = datetime(2020, 1, 1)
        dates = []
        product_ids = []
        
        num_products = 10
        records_per_product = num_records // num_products
        
        for product_idx in range(num_products):
            product_id = f'PROD_{product_idx:03d}'
            for day_idx in range(records_per_product):
                dates.append(start_date + timedelta(days=day_idx))
                product_ids.append(product_id)
        
        # Handle any remaining records
        remaining = num_records - len(dates)
        for i in range(remaining):
            dates.append(start_date + timedelta(days=records_per_product + i))
            product_ids.append(f'PROD_{i % num_products:03d}')
        
        # Generate random sales and prices
        np.random.seed(42)  # For reproducibility
        sales_volumes = np.random.uniform(50, 500, num_records)
        prices = np.random.uniform(5, 50, num_records)
        
        # Generate holiday indicators (10% holidays)
        is_holiday = np.random.choice([True, False], num_records, p=[0.1, 0.9])
        
        # Generate seasonality features
        day_of_week = [d.weekday() for d in dates]
        month = [d.month for d in dates]
        quarter = [(d.month - 1) // 3 + 1 for d in dates]
        
        return pd.DataFrame({
            'timestamp': dates,
            'product_id': product_ids,
            'sales_volume': sales_volumes,
            'price': prices,
            'is_holiday': is_holiday,
            'day_of_week': day_of_week,
            'month': month,
            'quarter': quarter
        })
    
    @pytest.mark.slow
    def test_ingest_1m_records_within_5_seconds(self, ingestion_service):
        """
        Test that ingesting 1M records completes within 5 seconds.
        
        This is a critical performance requirement from the design document.
        """
        # Generate 1M records
        large_dataset = self.generate_large_dataset(1_000_000)
        
        # Ingest the data
        result = ingestion_service.ingest_batch(large_dataset, format="auto")
        
        # Verify success
        assert result.success is True
        assert result.record_count == 1_000_000
        
        # Verify performance target
        assert result.ingestion_time_seconds <= 5.0, \
            f"Ingestion took {result.ingestion_time_seconds:.2f}s, exceeds 5s target"
        
        print(f"\n✓ Successfully ingested 1M records in {result.ingestion_time_seconds:.2f} seconds")
    
    @pytest.mark.slow
    def test_ingest_100k_records_performance(self, ingestion_service):
        """Test ingestion performance with 100K records."""
        dataset = self.generate_large_dataset(100_000)
        
        result = ingestion_service.ingest_batch(dataset, format="auto")
        
        assert result.success is True
        assert result.record_count == 100_000
        
        # Should be much faster than 5 seconds for 100K records
        assert result.ingestion_time_seconds <= 1.0, \
            f"Ingestion of 100K records took {result.ingestion_time_seconds:.2f}s"
        
        print(f"\n✓ Successfully ingested 100K records in {result.ingestion_time_seconds:.2f} seconds")
    
    @pytest.mark.slow
    def test_ingest_10k_records_performance(self, ingestion_service):
        """Test ingestion performance with 10K records."""
        dataset = self.generate_large_dataset(10_000)
        
        result = ingestion_service.ingest_batch(dataset, format="auto")
        
        assert result.success is True
        assert result.record_count == 10_000
        
        # Should be very fast for 10K records
        assert result.ingestion_time_seconds <= 0.5, \
            f"Ingestion of 10K records took {result.ingestion_time_seconds:.2f}s"
        
        print(f"\n✓ Successfully ingested 10K records in {result.ingestion_time_seconds:.2f} seconds")
    
    @pytest.mark.slow
    def test_csv_parsing_performance_1m_records(self, ingestion_service):
        """Test CSV parsing performance with 1M records."""
        dataset = self.generate_large_dataset(1_000_000)
        csv_string = dataset.to_csv(index=False)
        
        result = ingestion_service.ingest_batch(csv_string, format="csv")
        
        assert result.success is True
        assert result.record_count == 1_000_000
        
        # CSV parsing might be slightly slower, but should still be reasonable
        assert result.ingestion_time_seconds <= 10.0, \
            f"CSV parsing of 1M records took {result.ingestion_time_seconds:.2f}s"
        
        print(f"\n✓ Successfully parsed and ingested 1M records from CSV in {result.ingestion_time_seconds:.2f} seconds")
    
    @pytest.mark.slow
    def test_validation_performance_1m_records(self, validator):
        """Test validation performance with 1M records."""
        dataset = self.generate_large_dataset(1_000_000)
        
        start_time = datetime.now()
        result = validator.validate_schema(dataset)
        end_time = datetime.now()
        
        validation_time = (end_time - start_time).total_seconds()
        
        assert result.is_valid is True
        assert result.record_count == 1_000_000
        
        # Validation should be fast
        assert validation_time <= 2.0, \
            f"Validation of 1M records took {validation_time:.2f}s"
        
        print(f"\n✓ Successfully validated 1M records in {validation_time:.2f} seconds")
