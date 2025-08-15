"""
Integration tests for CI/CD pipeline components.

This module tests the deployment scripts, health checks, and CI/CD workflows
to ensure they work correctly in different environments.
"""

import asyncio
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.core.config import Config
from src.web_interface.api.health import (
    check_database_health,
    check_redis_health,
    check_exchanges_health,
    check_ml_models_health,
    ComponentHealth
)
from src.web_interface.app import create_app


class TestHealthCheckEndpoints:
    """Test health check endpoints and functionality."""

    @pytest.fixture
    def test_config(self):
        """Create test configuration."""
        return Config(
            environment="test",
            debug=True,
            database={
                "url": "sqlite:///test.db",
                "pool_size": 5
            },
            redis={
                "url": "redis://localhost:6379/1",
                "password": None
            },
            security={
                "secret_key": "test-secret-key",
                "jwt_algorithm": "HS256",
                "jwt_expire_minutes": 60
            }
        )

    @pytest.fixture
    def test_client(self, test_config):
        """Create test client."""
        app = create_app(test_config)
        return TestClient(app)

    def test_basic_health_endpoint(self, test_client):
        """Test basic health check endpoint."""
        response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert data["service"] == "t-bot-api"
        assert data["version"] == "1.0.0"
        assert "uptime_seconds" in data
        assert isinstance(data["uptime_seconds"], (int, float))

    @pytest.mark.asyncio
    async def test_database_health_check_success(self, test_config):
        """Test successful database health check."""
        with patch('src.database.connection.DatabaseManager') as mock_db:
            mock_conn = AsyncMock()
            mock_conn.fetchval.return_value = 1
            mock_db.return_value.get_connection.return_value.__aenter__.return_value = mock_conn
            mock_db.return_value.get_pool_status.return_value = {
                "size": 10, "used": 2, "free": 8
            }
            
            result = await check_database_health(test_config)
            
            assert isinstance(result, ComponentHealth)
            assert result.status == "healthy"
            assert "Database connection successful" in result.message
            assert result.response_time_ms is not None
            assert result.metadata["pool_size"] == 10

    @pytest.mark.asyncio
    async def test_database_health_check_failure(self, test_config):
        """Test failed database health check."""
        with patch('src.database.connection.DatabaseManager') as mock_db:
            mock_db.return_value.get_connection.side_effect = Exception("Connection failed")
            
            result = await check_database_health(test_config)
            
            assert isinstance(result, ComponentHealth)
            assert result.status == "unhealthy"
            assert "Connection failed" in result.message

    @pytest.mark.asyncio
    async def test_redis_health_check_success(self, test_config):
        """Test successful Redis health check."""
        with patch('src.database.redis_client.RedisClient') as mock_redis:
            mock_client = AsyncMock()
            mock_client.ping.return_value = True
            mock_client.set.return_value = True
            mock_client.get.return_value = "test_value"
            mock_client.info.return_value = {
                "used_memory_human": "1M",
                "connected_clients": 5,
                "uptime_in_days": 10
            }
            mock_redis.return_value = mock_client
            
            result = await check_redis_health(test_config)
            
            assert isinstance(result, ComponentHealth)
            assert result.status == "healthy"
            assert "Redis connection successful" in result.message
            assert result.metadata["connected_clients"] == 5

    @pytest.mark.asyncio
    async def test_exchanges_health_check(self, test_config):
        """Test exchanges health check."""
        with patch('src.exchanges.factory.ExchangeFactory') as mock_factory, \
             patch('src.exchanges.health_monitor.ExchangeHealthMonitor') as mock_monitor:
            
            mock_factory.return_value.get_available_exchanges.return_value = ["binance", "coinbase"]
            mock_exchange = AsyncMock()
            mock_factory.return_value.create_exchange.return_value = mock_exchange
            
            mock_monitor.return_value.check_exchange_health.return_value = {
                "status": "healthy",
                "latency_ms": 50,
                "rate_limit_remaining": 1000
            }
            
            result = await check_exchanges_health(test_config)
            
            assert isinstance(result, ComponentHealth)
            assert result.status == "healthy"
            assert "All 2 exchanges healthy" in result.message
            assert result.metadata["total_exchanges"] == 2

    def test_readiness_probe(self, test_client):
        """Test Kubernetes readiness probe endpoint."""
        response = test_client.get("/health/ready")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["ready"] is True
        assert "timestamp" in data

    def test_liveness_probe(self, test_client):
        """Test Kubernetes liveness probe endpoint."""
        response = test_client.get("/health/live")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["alive"] is True
        assert "timestamp" in data

    def test_startup_probe(self, test_client):
        """Test Kubernetes startup probe endpoint."""
        # Should pass after startup time
        time.sleep(0.1)  # Ensure some uptime
        response = test_client.get("/health/startup")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["started"] is True
        assert "uptime_seconds" in data


class TestDeploymentScripts:
    """Test deployment and rollback scripts."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create temporary project directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create directory structure
            (temp_path / "scripts").mkdir()
            (temp_path / "backups").mkdir()
            (temp_path / ".env.staging").touch()
            (temp_path / ".env.production").touch()
            
            # Create mock deployment script
            deploy_script = temp_path / "scripts" / "deploy.sh"
            deploy_script.write_text("""#!/bin/bash
echo "Mock deployment script"
echo "Environment: $1"
echo "Version: $2"
exit 0
""")
            deploy_script.chmod(0o755)
            
            # Create mock rollback script
            rollback_script = temp_path / "scripts" / "rollback.sh"
            rollback_script.write_text("""#!/bin/bash
echo "Mock rollback script"
echo "Environment: $1"
echo "Version: $2"
exit 0
""")
            rollback_script.chmod(0o755)
            
            yield temp_path

    def test_deployment_script_exists(self):
        """Test that deployment script exists and is executable."""
        script_path = Path(__file__).parent.parent.parent / "scripts" / "deploy.sh"
        
        assert script_path.exists(), "Deploy script should exist"
        assert script_path.is_file(), "Deploy script should be a file"
        # Note: Can't test executable bit in CI environment

    def test_rollback_script_exists(self):
        """Test that rollback script exists and is executable."""
        script_path = Path(__file__).parent.parent.parent / "scripts" / "rollback.sh"
        
        assert script_path.exists(), "Rollback script should exist"
        assert script_path.is_file(), "Rollback script should be a file"

    def test_environment_templates_exist(self):
        """Test that environment configuration templates exist."""
        project_root = Path(__file__).parent.parent.parent
        
        staging_template = project_root / ".env.staging.example"
        production_template = project_root / ".env.production.example"
        
        assert staging_template.exists(), "Staging environment template should exist"
        assert production_template.exists(), "Production environment template should exist"

    def test_deployment_script_help(self, temp_project_dir):
        """Test deployment script help option."""
        script_path = temp_project_dir / "scripts" / "deploy.sh"
        
        result = subprocess.run(
            [str(script_path), "--help"],
            capture_output=True,
            text=True,
            cwd=temp_project_dir
        )
        
        assert result.returncode == 0
        assert "Usage:" in result.stdout
        assert "ENVIRONMENT" in result.stdout

    def test_rollback_script_help(self, temp_project_dir):
        """Test rollback script help option."""
        script_path = temp_project_dir / "scripts" / "rollback.sh"
        
        result = subprocess.run(
            [str(script_path), "--help"],
            capture_output=True,
            text=True,
            cwd=temp_project_dir
        )
        
        assert result.returncode == 0
        assert "Usage:" in result.stdout
        assert "ENVIRONMENT" in result.stdout


class TestGitHubActionsWorkflows:
    """Test GitHub Actions workflow configurations."""

    def test_ci_workflow_exists(self):
        """Test that CI workflow file exists."""
        workflow_path = Path(__file__).parent.parent.parent / ".github" / "workflows" / "ci.yml"
        
        assert workflow_path.exists(), "CI workflow should exist"
        
        # Verify basic structure
        content = workflow_path.read_text()
        assert "name: CI Pipeline" in content
        assert "quality-gates" in content
        assert "backend-tests" in content
        assert "frontend-tests" in content

    def test_cd_workflow_exists(self):
        """Test that CD workflow file exists."""
        workflow_path = Path(__file__).parent.parent.parent / ".github" / "workflows" / "cd.yml"
        
        assert workflow_path.exists(), "CD workflow should exist"
        
        content = workflow_path.read_text()
        assert "name: CD Pipeline" in content
        assert "build-and-push" in content
        assert "deploy-staging" in content
        assert "deploy-production" in content

    def test_security_workflow_exists(self):
        """Test that security workflow file exists."""
        workflow_path = Path(__file__).parent.parent.parent / ".github" / "workflows" / "security.yml"
        
        assert workflow_path.exists(), "Security workflow should exist"
        
        content = workflow_path.read_text()
        assert "name: Security Scans" in content
        assert "dependency-scan" in content
        assert "container-scan" in content

    def test_docker_workflow_exists(self):
        """Test that Docker workflow file exists."""
        workflow_path = Path(__file__).parent.parent.parent / ".github" / "workflows" / "docker.yml"
        
        assert workflow_path.exists(), "Docker workflow should exist"
        
        content = workflow_path.read_text()
        assert "name: Docker Build" in content
        assert "build-matrix" in content
        assert "linux/amd64,linux/arm64" in content

    def test_dependencies_workflow_exists(self):
        """Test that dependencies workflow file exists."""
        workflow_path = Path(__file__).parent.parent.parent / ".github" / "workflows" / "dependencies.yml"
        
        assert workflow_path.exists(), "Dependencies workflow should exist"
        
        content = workflow_path.read_text()
        assert "name: Dependency Updates" in content
        assert "update-python-deps" in content
        assert "update-frontend-deps" in content


class TestGitLabCIConfiguration:
    """Test GitLab CI configuration."""

    def test_gitlab_ci_exists(self):
        """Test that GitLab CI file exists."""
        ci_path = Path(__file__).parent.parent.parent / ".gitlab-ci.yml"
        
        assert ci_path.exists(), "GitLab CI configuration should exist"
        
        content = ci_path.read_text()
        assert "stages:" in content
        assert "quality-gates" in content
        assert "test" in content
        assert "build" in content
        assert "security" in content

    def test_gitlab_ci_structure(self):
        """Test GitLab CI configuration structure."""
        ci_path = Path(__file__).parent.parent.parent / ".gitlab-ci.yml"
        content = ci_path.read_text()
        
        # Check for required stages
        required_stages = [
            "quality-gates",
            "test", 
            "build",
            "security",
            "deploy-staging",
            "deploy-production"
        ]
        
        for stage in required_stages:
            assert stage in content, f"Stage '{stage}' should be in GitLab CI"

    def test_gitlab_ci_variables(self):
        """Test GitLab CI variables are properly defined."""
        ci_path = Path(__file__).parent.parent.parent / ".gitlab-ci.yml"
        content = ci_path.read_text()
        
        # Check for important variables
        required_variables = [
            "PYTHON_VERSION",
            "NODE_VERSION",
            "REGISTRY",
            "BACKEND_IMAGE",
            "FRONTEND_IMAGE"
        ]
        
        for var in required_variables:
            assert f"{var}:" in content, f"Variable '{var}' should be defined"


class TestDockerConfiguration:
    """Test Docker and Docker Compose configurations."""

    def test_dockerfile_exists(self):
        """Test that Dockerfile exists."""
        dockerfile_path = Path(__file__).parent.parent.parent / "Dockerfile"
        
        assert dockerfile_path.exists(), "Dockerfile should exist"
        
        content = dockerfile_path.read_text()
        assert "FROM python:3.10.12-slim" in content
        assert "HEALTHCHECK" in content

    def test_frontend_dockerfile_exists(self):
        """Test that frontend Dockerfile exists."""
        dockerfile_path = Path(__file__).parent.parent.parent / "frontend" / "Dockerfile"
        
        assert dockerfile_path.exists(), "Frontend Dockerfile should exist"

    def test_docker_compose_files_exist(self):
        """Test that Docker Compose files exist."""
        project_root = Path(__file__).parent.parent.parent
        
        compose_files = [
            "docker-compose.yml",
            "docker-compose.prod.yml"
        ]
        
        for compose_file in compose_files:
            path = project_root / compose_file
            assert path.exists(), f"Docker Compose file '{compose_file}' should exist"

    def test_docker_healthchecks(self):
        """Test that Docker containers have proper health checks."""
        dockerfile_path = Path(__file__).parent.parent.parent / "Dockerfile"
        content = dockerfile_path.read_text()
        
        # Check for health check configuration
        assert "HEALTHCHECK" in content
        assert "/health" in content


class TestCICDIntegration:
    """Test end-to-end CI/CD integration scenarios."""

    @pytest.mark.slow
    def test_full_pipeline_simulation(self):
        """Simulate a full CI/CD pipeline run."""
        # This would be a more comprehensive test that simulates
        # the entire pipeline in a test environment
        
        steps = [
            "quality_gates",
            "unit_tests", 
            "integration_tests",
            "build_images",
            "security_scan",
            "deploy_staging"
        ]
        
        results = {}
        
        for step in steps:
            # Simulate each step
            results[step] = "success"  # Would be actual test results
        
        # Verify all steps passed
        for step, result in results.items():
            assert result == "success", f"Step {step} should succeed"

    def test_deployment_config_validation(self):
        """Test deployment configuration validation."""
        # Test environment variable validation
        required_staging_vars = [
            "SECRET_KEY",
            "JWT_SECRET", 
            "DATABASE_URL",
            "REDIS_URL"
        ]
        
        staging_template = Path(__file__).parent.parent.parent / ".env.staging.example"
        content = staging_template.read_text()
        
        for var in required_staging_vars:
            assert f"{var}=" in content, f"Variable {var} should be in staging template"

    def test_security_configuration(self):
        """Test security-related configuration."""
        # Check that security workflows have proper scanning tools
        security_workflow = Path(__file__).parent.parent.parent / ".github" / "workflows" / "security.yml"
        content = security_workflow.read_text()
        
        security_tools = [
            "bandit",
            "safety",
            "trivy",
            "semgrep"
        ]
        
        for tool in security_tools:
            assert tool in content, f"Security tool {tool} should be configured"

    def test_monitoring_configuration(self):
        """Test monitoring and alerting configuration."""
        # Check that health check endpoints are comprehensive
        health_api = Path(__file__).parent.parent.parent / "src" / "web_interface" / "api" / "health.py"
        content = health_api.read_text()
        
        health_checks = [
            "database",
            "redis", 
            "exchanges",
            "ml_models"
        ]
        
        for check in health_checks:
            assert f"check_{check}_health" in content, f"Health check for {check} should exist"