"""Database connection management with connection pooling."""
from sqlalchemy import create_engine, pool
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from typing import Generator
from config.settings import settings
from src.utils.logging_config import logger


class DatabaseManager:
    """Manages database connections with connection pooling."""
    
    def __init__(self):
        """Initialize database engine with connection pooling."""
        self.engine = create_engine(
            settings.database_url,
            poolclass=pool.QueuePool,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_pre_ping=True,  # Verify connections before using
            pool_recycle=3600,   # Recycle connections after 1 hour
            echo=settings.log_level == "DEBUG"
        )
        
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        logger.info(
            f"Database connection pool initialized: "
            f"pool_size={settings.db_pool_size}, "
            f"max_overflow={settings.db_max_overflow}"
        )
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Get a database session with automatic cleanup.
        
        Yields:
            SQLAlchemy Session object
            
        Example:
            with db_manager.get_session() as session:
                result = session.query(Model).all()
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def close(self):
        """Close all database connections."""
        self.engine.dispose()
        logger.info("Database connection pool closed")


# Global database manager instance
db_manager = DatabaseManager()
