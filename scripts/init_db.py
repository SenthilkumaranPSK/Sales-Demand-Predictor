"""Initialize database schema for Model Registry."""
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import sys
import os

# Add parent directory to path to import config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.settings import settings
from src.utils.logging_config import logger


def create_database():
    """Create database if it doesn't exist."""
    try:
        # Connect to PostgreSQL server (default database)
        conn = psycopg2.connect(
            host=settings.db_host,
            port=settings.db_port,
            user=settings.db_user,
            password=settings.db_password,
            database='postgres'
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (settings.db_name,)
        )
        exists = cursor.fetchone()
        
        if not exists:
            cursor.execute(f"CREATE DATABASE {settings.db_name}")
            logger.info(f"Database '{settings.db_name}' created successfully")
        else:
            logger.info(f"Database '{settings.db_name}' already exists")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error creating database: {e}")
        raise


def create_schema():
    """Create Model Registry schema."""
    schema_sql = """
    -- Models table for storing model metadata
    CREATE TABLE IF NOT EXISTS models (
        model_id VARCHAR(255) PRIMARY KEY,
        product_id VARCHAR(255) NOT NULL,
        model_type VARCHAR(50) NOT NULL CHECK (model_type IN ('custom', 'forecast')),
        version INTEGER NOT NULL,
        artifact_s3_path VARCHAR(512) NOT NULL,
        training_dataset_id VARCHAR(255) NOT NULL,
        mae DECIMAL(10, 4),
        rmse DECIMAL(10, 4),
        mape DECIMAL(10, 4),
        hyperparameters JSONB,
        forecast_horizon INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT unique_product_type_version UNIQUE (product_id, model_type, version)
    );
    
    -- Indexes for efficient querying
    CREATE INDEX IF NOT EXISTS idx_product_type ON models(product_id, model_type);
    CREATE INDEX IF NOT EXISTS idx_created ON models(created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_product_id ON models(product_id);
    
    -- Training jobs table for tracking training pipeline executions
    CREATE TABLE IF NOT EXISTS training_jobs (
        job_id VARCHAR(255) PRIMARY KEY,
        product_id VARCHAR(255) NOT NULL,
        dataset_id VARCHAR(255) NOT NULL,
        model_type VARCHAR(50) NOT NULL,
        status VARCHAR(50) NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed')),
        error_message TEXT,
        started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        completed_at TIMESTAMP,
        model_id VARCHAR(255) REFERENCES models(model_id)
    );
    
    CREATE INDEX IF NOT EXISTS idx_training_status ON training_jobs(status);
    CREATE INDEX IF NOT EXISTS idx_training_product ON training_jobs(product_id);
    """
    
    try:
        conn = psycopg2.connect(
            host=settings.db_host,
            port=settings.db_port,
            user=settings.db_user,
            password=settings.db_password,
            database=settings.db_name
        )
        cursor = conn.cursor()
        
        # Execute schema creation
        cursor.execute(schema_sql)
        conn.commit()
        
        logger.info("Database schema created successfully")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error creating schema: {e}")
        raise


def main():
    """Main initialization function."""
    logger.info("Starting database initialization...")
    
    try:
        create_database()
        create_schema()
        logger.info("Database initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
