"""Tests for health check endpoint."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from src.api.main import app


client = TestClient(app)


@pytest.mark.unit
def test_health_check_success():
    """Test health check endpoint returns 200 when all services are healthy."""
    with patch('src.api.main.db_manager.get_session') as mock_session:
        # Mock successful database connection
        mock_session.return_value.__enter__ = MagicMock()
        mock_session.return_value.__exit__ = MagicMock()
        
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] in ["healthy", "degraded"]
        assert "timestamp" in data
        assert "environment" in data
        assert "version" in data
        assert "checks" in data
        assert "response_time_ms" in data["checks"]


@pytest.mark.unit
def test_health_check_response_time():
    """Test health check responds within 1 second (Requirement 10.5)."""
    import time
    
    with patch('src.api.main.db_manager.get_session') as mock_session:
        mock_session.return_value.__enter__ = MagicMock()
        mock_session.return_value.__exit__ = MagicMock()
        
        start_time = time.time()
        response = client.get("/api/v1/health")
        elapsed_time = time.time() - start_time
        
        assert response.status_code == 200
        assert elapsed_time < 1.0, f"Health check took {elapsed_time:.2f}s, should be < 1s"


@pytest.mark.unit
def test_health_check_database_failure():
    """Test health check returns degraded status when database is unavailable."""
    with patch('src.api.main.db_manager.get_session') as mock_session:
        # Mock database connection failure
        mock_session.return_value.__enter__.side_effect = Exception("Database connection failed")
        
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "degraded"
        assert data["checks"]["database"] == "unhealthy"
        assert "database_error" in data["checks"]


@pytest.mark.unit
def test_root_endpoint():
    """Test root endpoint returns API information."""
    response = client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "message" in data
    assert "version" in data
    assert "docs" in data
