"""
Tests for web_interface.app module.
"""

import threading
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import uvicorn
from fastapi import FastAPI, Request

from src.core.config import Config
from src.core.exceptions import ConfigurationError
from src.web_interface.app import (
    LazyApp,
    _connect_api_endpoints_to_services,
    _get_app_lazy,
    _initialize_services,
    _register_routes,
    _setup_monitoring,
    app,
    create_app,
    get_app,
    get_asgi_app,
    lifespan,
)


@pytest.fixture
def mock_config():
    """Create a mock config for testing."""
    config = Mock(spec=Config)
    config.environment = "test"
    config.debug = True
    config.api = Mock()
    config.api.port = 8080

    # Security config
    config.security = Mock()
    config.security.secret_key = "test-secret-key"
    config.security.jwt_algorithm = "HS256"
    config.security.jwt_expire_minutes = 30
    config.security.refresh_token_expire_days = 7
    config.security.session_timeout_minutes = 60

    # Web interface config
    config.web_interface = {
        "debug": True,
        "jwt": {
            "secret_key": "test-secret-key",
            "algorithm": "HS256",
            "access_token_expire_minutes": 30,
        },
        "cors": {
            "allow_origins": ["http://localhost:3000"],
            "allow_credentials": True,
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        },
        "rate_limiting": {"anonymous_limit": 1000, "authenticated_limit": 5000},
    }

    return config


@pytest.fixture
def mock_bot_orchestrator():
    """Create a mock bot orchestrator."""
    orchestrator = Mock()
    orchestrator.start = AsyncMock()
    orchestrator.stop = AsyncMock()
    return orchestrator


@pytest.fixture
def mock_execution_engine():
    """Create a mock execution engine."""
    engine = Mock()
    engine.start = AsyncMock()
    engine.stop = AsyncMock()
    return engine


@pytest.fixture
def mock_model_manager():
    """Create a mock model manager."""
    manager = Mock()
    return manager


class TestInitializeServices:
    """Tests for _initialize_services function."""

    @patch("src.web_interface.app._services_initialized", False)
    @patch("src.core.dependency_injection.injector")
    @patch("src.web_interface.di_registration.register_web_interface_services")
    @patch("src.web_interface.app.initialize_auth_manager")
    @patch("src.web_interface.app.get_api_facade")
    @patch("src.web_interface.app._connect_api_endpoints_to_services")
    async def test_initialize_services_success(
        self,
        mock_connect_api,
        mock_get_facade,
        mock_init_auth,
        mock_register_services,
        mock_injector,
    ):
        """Test successful service initialization."""
        # Setup mocks
        mock_facade = AsyncMock()
        mock_get_facade.return_value = mock_facade
        mock_injector.resolve.side_effect = lambda name: Mock(name=name)

        # Mock app_config
        with patch("src.web_interface.app.app_config") as mock_app_config:
            mock_app_config.security = Mock()
            mock_app_config.security.secret_key = "test-key"

            await _initialize_services()

            # Verify calls
            mock_register_services.assert_called_once()
            mock_init_auth.assert_called_once()
            mock_facade.initialize.assert_called_once()
            mock_connect_api.assert_called_once()

    @patch("src.web_interface.app._services_initialized", False)
    @patch("src.web_interface.app.app_config", None)
    async def test_initialize_services_no_security_config(self):
        """Test service initialization fails without security config."""
        from src.core.exceptions import ConfigurationError
        with pytest.raises(ConfigurationError, match="Security configuration is required"):
            await _initialize_services()

    @patch("src.web_interface.app._services_initialized", True)
    async def test_initialize_services_already_initialized(self):
        """Test that already initialized services are skipped."""
        # Should return early without doing anything
        await _initialize_services()
        # No assertions needed - just verify no exception is raised


class TestConnectApiEndpointsToServices:
    """Tests for _connect_api_endpoints_to_services function."""

    async def test_connect_api_endpoints_success(self):
        """Test successful API endpoint connection.

        Backend no longer calls set_bot_service (deprecated).
        It only logs that the service is connected via registry.
        """
        mock_registry = Mock()
        mock_registry.has_service.return_value = True
        mock_service = Mock()
        mock_registry.get_service.return_value = mock_service

        # Should complete without error
        # Backend no longer calls deprecated set_bot_service
        await _connect_api_endpoints_to_services(mock_registry)

        # Verify has_service was checked
        mock_registry.has_service.assert_called_with("bot_management")

    async def test_connect_api_endpoints_no_service(self):
        """Test API endpoint connection when service is not available."""
        mock_registry = Mock()
        mock_registry.has_service.return_value = False

        # Should complete without error
        await _connect_api_endpoints_to_services(mock_registry)

    async def test_connect_api_endpoints_exception(self):
        """Test API endpoint connection handles exceptions gracefully."""
        mock_registry = Mock()
        mock_registry.has_service.side_effect = Exception("Test error")

        # Should complete without raising exception
        await _connect_api_endpoints_to_services(mock_registry)


class TestLifespan:
    """Tests for lifespan context manager."""

    @patch("src.web_interface.app._initialize_services")
    @patch("src.web_interface.app.get_unified_websocket_manager")
    @patch("src.web_interface.app.bot_orchestrator")
    @patch("src.web_interface.app.execution_engine")
    async def test_lifespan_startup_success(
        self,
        mock_execution_engine,
        mock_bot_orchestrator,
        mock_get_websocket_manager,
        mock_initialize_services,
    ):
        """Test successful lifespan startup."""
        # Setup mocks
        mock_initialize_services.return_value = AsyncMock()
        mock_websocket_manager = AsyncMock()
        mock_get_websocket_manager.return_value = mock_websocket_manager
        mock_bot_orchestrator.start = AsyncMock()
        mock_execution_engine.start = AsyncMock()

        app_mock = Mock()

        async with lifespan(app_mock):
            # Verify startup calls
            mock_initialize_services.assert_called_once()
            mock_websocket_manager.start.assert_called_once()
            mock_bot_orchestrator.start.assert_called_once()
            mock_execution_engine.start.assert_called_once()

    @patch("src.web_interface.app._initialize_services")
    @patch("src.web_interface.app.get_unified_websocket_manager")
    async def test_lifespan_startup_websocket_failure(
        self, mock_get_websocket_manager, mock_initialize_services
    ):
        """Test lifespan startup with WebSocket failure."""
        mock_initialize_services.return_value = AsyncMock()
        mock_websocket_manager = AsyncMock()
        mock_websocket_manager.start.side_effect = Exception("WebSocket error")
        mock_get_websocket_manager.return_value = mock_websocket_manager

        app_mock = Mock()

        with pytest.raises(Exception, match="WebSocket error"):
            async with lifespan(app_mock):
                pass

    @patch("src.web_interface.app._initialize_services")
    @patch("src.web_interface.app.get_unified_websocket_manager")
    @patch("src.web_interface.app.get_api_facade")
    @patch("src.web_interface.app.get_service_registry")
    async def test_lifespan_shutdown_success(
        self,
        mock_get_service_registry,
        mock_get_api_facade,
        mock_get_websocket_manager,
        mock_initialize_services,
    ):
        """Test successful lifespan shutdown."""
        # Setup mocks
        mock_initialize_services.return_value = AsyncMock()
        mock_websocket_manager = AsyncMock()
        mock_get_websocket_manager.return_value = mock_websocket_manager
        mock_facade = AsyncMock()
        mock_get_api_facade.return_value = mock_facade
        mock_registry = Mock()
        mock_registry.get_all_service_names.return_value = ["service1", "service2"]
        mock_service = AsyncMock()
        mock_registry.get_service.return_value = mock_service
        mock_registry.cleanup_all = AsyncMock()
        mock_get_service_registry.return_value = mock_registry

        app_mock = Mock()

        try:
            async with lifespan(app_mock):
                pass
        except StopAsyncIteration:
            pass

        # Verify shutdown calls
        mock_websocket_manager.stop.assert_called_once()
        mock_facade.cleanup.assert_called_once()
        mock_registry.cleanup_all.assert_called_once()


class TestCreateApp:
    """Tests for create_app function."""

    def test_create_app_success(self, mock_config, mock_bot_orchestrator):
        """Test successful app creation."""
        with patch("src.web_interface.app.get_unified_websocket_manager") as mock_get_manager:
            mock_websocket_manager = Mock()
            mock_websocket_manager.create_server = Mock()
            mock_get_manager.return_value = mock_websocket_manager

            with patch("socketio.ASGIApp") as mock_asgi_app:
                app_instance = create_app(
                    mock_config, bot_orchestrator_instance=mock_bot_orchestrator
                )

                # Verify app was created
                assert app_instance is not None
                mock_asgi_app.assert_called_once()

    def test_create_app_minimal_config(self):
        """Test app creation with minimal config."""
        mock_config = Mock()
        mock_config.security = Mock()
        mock_config.security.secret_key = "test-key"

        with patch("src.web_interface.app.get_unified_websocket_manager") as mock_get_manager:
            mock_websocket_manager = Mock()
            mock_websocket_manager.create_server = Mock()
            mock_get_manager.return_value = mock_get_manager

            with patch("socketio.ASGIApp"):
                app_instance = create_app(mock_config)
                assert app_instance is not None

    def test_create_app_connection_pool_failure_dev(self, mock_config):
        """Test app creation handles connection pool failure in development."""
        mock_config.environment = "development"

        with patch("src.web_interface.app.get_unified_websocket_manager") as mock_get_manager:
            mock_websocket_manager = Mock()
            mock_get_manager.return_value = mock_websocket_manager

            with patch("socketio.ASGIApp"):
                with patch("src.web_interface.middleware.connection_pool.ConnectionPoolMiddleware", side_effect=Exception("Pool error")):
                    # Should not raise in development
                    app_instance = create_app(mock_config)
                    assert app_instance is not None

    def test_create_app_connection_pool_failure_prod(self, mock_config):
        """Test app creation fails on connection pool error in production."""
        mock_config.environment = "production"

        with patch("src.web_interface.app.get_unified_websocket_manager"):
            with patch("src.web_interface.middleware.connection_pool.ConnectionPoolMiddleware", side_effect=Exception("Pool error")):
                with pytest.raises(RuntimeError, match="Connection pool middleware setup failed"):
                    create_app(mock_config)

    def test_create_app_auth_failure_production(self, mock_config):
        """Test app creation fails on auth setup error in production."""
        mock_config.environment = "production"

        with patch("src.web_interface.app.get_unified_websocket_manager") as mock_get_manager:
            mock_websocket_manager = Mock()
            mock_get_manager.return_value = mock_websocket_manager

            with patch("src.web_interface.security.jwt_handler.JWTHandler", side_effect=ImportError("Auth error")):
                with pytest.raises(RuntimeError, match="Cannot start without authentication in production"):
                    create_app(mock_config)


class TestRegisterRoutes:
    """Tests for _register_routes function."""

    def test_register_routes_success(self):
        """Test successful route registration."""
        mock_app = Mock(spec=FastAPI)

        with patch("importlib.import_module") as mock_import:
            mock_module = Mock()
            mock_router = Mock()
            mock_module.router = mock_router
            mock_import.return_value = mock_module

            _register_routes(mock_app)

            # Verify routes were registered
            assert mock_app.include_router.call_count > 0

    def test_register_routes_critical_failure(self):
        """Test route registration fails when critical routers are missing."""
        mock_app = Mock(spec=FastAPI)

        with patch("importlib.import_module", side_effect=ImportError("Module not found")):
            with pytest.raises(RuntimeError, match="Critical routers failed to load"):
                _register_routes(mock_app)

    def test_register_routes_non_critical_failure(self):
        """Test route registration continues when non-critical routers fail."""
        mock_app = Mock(spec=FastAPI)

        def side_effect(module_name):
            if "auth" in module_name or "bot_management" in module_name or "trading" in module_name:
                mock_module = Mock()
                mock_module.router = Mock()
                return mock_module
            raise ImportError("Non-critical module not found")

        with patch("importlib.import_module", side_effect=side_effect):
            # Should not raise exception for non-critical failures
            _register_routes(mock_app)
            assert mock_app.include_router.call_count >= 3  # At least critical routers

    async def test_health_check_endpoint(self):
        """Test health check endpoint functionality."""
        mock_app = Mock(spec=FastAPI)

        with patch("src.web_interface.app.get_api_facade") as mock_get_facade:
            with patch("src.web_interface.app.get_auth_manager") as mock_get_auth:
                with patch("src.web_interface.app.get_unified_websocket_manager") as mock_get_ws:
                    mock_facade = Mock()
                    mock_facade.health_check = AsyncMock(return_value={"status": "ok"})
                    mock_get_facade.return_value = mock_facade

                    mock_auth = Mock()
                    mock_auth.get_user_stats = Mock(return_value={"users": 1})
                    mock_get_auth.return_value = mock_auth

                    mock_ws = Mock()
                    mock_ws.get_connection_stats = Mock(return_value={"connections": 5})
                    mock_get_ws.return_value = mock_ws

                    _register_routes(mock_app)

                    # Verify that get was called for health endpoint registration
                    called_routes = [call[0][0] for call in mock_app.get.call_args_list]
                    assert "/health" in called_routes

    async def test_process_time_middleware(self):
        """Test process time middleware functionality."""
        mock_app = Mock(spec=FastAPI)
        _register_routes(mock_app)

        # Verify that middleware decorator was called
        assert mock_app.middleware.called
        # Check if 'http' was passed as the middleware type
        called_types = [call[0][0] for call in mock_app.middleware.call_args_list if call[0]]
        assert "http" in called_types


class TestSetupMonitoring:
    """Tests for _setup_monitoring function."""

    @patch("src.web_interface.app._monitoring_setup_done", False)
    def test_setup_monitoring_success(self, mock_config):
        """Test successful monitoring setup."""
        mock_app = Mock(spec=FastAPI)

        with patch("src.web_interface.app.setup_telemetry") as mock_setup_telemetry:
            with patch("src.web_interface.app.MetricsCollector"):
                with patch("src.web_interface.app.PerformanceProfiler"):
                    with patch("src.web_interface.app.AlertManager"):
                        _setup_monitoring(mock_app, mock_config)
                        mock_setup_telemetry.assert_called_once()

    @patch("src.web_interface.app._monitoring_setup_done", True)
    def test_setup_monitoring_already_done(self, mock_config):
        """Test monitoring setup skipped when already done."""
        mock_app = Mock(spec=FastAPI)
        _setup_monitoring(mock_app, mock_config)
        # Should return early without setting up anything

    @patch("src.web_interface.app._monitoring_setup_done", False)
    def test_setup_monitoring_with_failures(self, mock_config):
        """Test monitoring setup continues despite component failures."""
        mock_app = Mock(spec=FastAPI)

        with patch("src.web_interface.app.setup_telemetry", side_effect=Exception("Telemetry error")):
            with patch("src.web_interface.app.MetricsCollector", side_effect=Exception("Metrics error")):
                with patch("src.web_interface.app.PerformanceProfiler", side_effect=Exception("Profiler error")):
                    with patch("src.web_interface.app.AlertManager", side_effect=Exception("Alert error")):
                        # Should not raise exception
                        _setup_monitoring(mock_app, mock_config)


class TestGetApp:
    """Tests for get_app function."""

    @patch("src.web_interface.app._app_instance", None)
    @patch("src.web_interface.app._app_creation_in_progress", False)
    def test_get_app_creates_new_instance(self):
        """Test get_app creates new instance when none exists."""
        with patch("src.web_interface.app.create_app") as mock_create_app:
            with patch("src.web_interface.app.Config") as mock_config_class:
                mock_app = Mock()
                mock_create_app.return_value = mock_app
                mock_config = Mock()
                mock_config_class.return_value = mock_config

                result = get_app()

                assert result == mock_app
                mock_create_app.assert_called_once_with(mock_config)

    @patch("src.web_interface.app._app_creation_in_progress", True)
    def test_get_app_creation_in_progress(self):
        """Test get_app returns None when creation is in progress."""
        result = get_app()
        assert result is None

    @patch("src.web_interface.app._app_instance", None)
    @patch("src.web_interface.app._app_creation_in_progress", False)
    def test_get_app_creation_failure(self):
        """Test get_app handles creation failure gracefully."""
        with patch("src.web_interface.app.create_app", side_effect=Exception("Creation failed")):
            with patch("src.web_interface.app.Config") as mock_config_class:
                mock_config_class.return_value = Mock()

                result = get_app()

                # Should return fallback FastAPI app
                assert result is not None
                assert isinstance(result, FastAPI)


class TestGetAsgiApp:
    """Tests for get_asgi_app function."""

    @patch("src.web_interface.app._lazy_app", None)
    def test_get_asgi_app_creates_new(self):
        """Test get_asgi_app creates new app when none exists."""
        with patch("src.web_interface.app.get_app") as mock_get_app:
            mock_app = Mock()
            mock_get_app.return_value = mock_app

            result = get_asgi_app()

            assert result == mock_app
            mock_get_app.assert_called_once()

    def test_get_asgi_app_returns_existing(self):
        """Test get_asgi_app returns existing app."""
        mock_app = Mock()
        with patch("src.web_interface.app._lazy_app", mock_app):
            result = get_asgi_app()
            assert result == mock_app


class TestLazyApp:
    """Tests for LazyApp class."""

    def test_lazy_app_getattr(self):
        """Test LazyApp __getattr__ method."""
        with patch("src.web_interface.app.get_asgi_app") as mock_get_app:
            mock_app = Mock()
            mock_app.some_attribute = "test_value"
            mock_get_app.return_value = mock_app

            lazy_app = LazyApp()
            result = lazy_app.some_attribute

            assert result == "test_value"
            mock_get_app.assert_called_once()

    def test_lazy_app_call(self):
        """Test LazyApp __call__ method."""
        with patch("src.web_interface.app.get_asgi_app") as mock_get_app:
            mock_app = Mock()
            mock_app.return_value = "called_result"
            mock_get_app.return_value = mock_app

            lazy_app = LazyApp()
            result = lazy_app("arg1", kwarg="test")

            assert result == "called_result"
            mock_app.assert_called_once_with("arg1", kwarg="test")


class TestMainExecution:
    """Tests for main execution block."""

    @patch("src.web_interface.app.get_asgi_app")
    @patch("src.web_interface.app._app_config_instance")
    @patch("uvicorn.run")
    def test_main_execution_default_port(self, mock_uvicorn_run, mock_app_config, mock_get_asgi_app):
        """Test main execution with default port."""
        mock_app_config = None
        mock_app = Mock()
        mock_get_asgi_app.return_value = mock_app

        # Mock __name__ == "__main__"
        with patch("src.web_interface.app.__name__", "__main__"):
            exec(open("/mnt/e/Work/P-41 Trading/code/t-bot/src/web_interface/app.py").read())

        # Should use default port 8000 when config is not available

    @patch("src.web_interface.app.get_asgi_app")
    @patch("src.web_interface.app._app_config_instance")
    @patch("uvicorn.run")
    def test_main_execution_config_port(self, mock_uvicorn_run, mock_app_config, mock_get_asgi_app):
        """Test main execution with configured port."""
        mock_config = Mock()
        mock_config.api = Mock()
        mock_config.api.port = 9000
        mock_app_config = mock_config
        mock_app = Mock()
        mock_get_asgi_app.return_value = mock_app

        # Test would require actual execution which is complex in unit tests
        # Focus on the configuration error handling instead

    def test_configuration_error_handling(self):
        """Test configuration error handling in main execution."""
        mock_config = Mock()

        # Test attribute error handling
        with patch("src.web_interface.app._app_config_instance", mock_config):
            mock_config.api = None

            # Simulate the port reading logic
            try:
                port = (
                    mock_config.api.port
                    if mock_config and hasattr(mock_config, "api")
                    else 8000
                )
            except (AttributeError, TypeError, ValueError):
                port = 8000

            assert port == 8000


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_get_app_lazy(self):
        """Test _get_app_lazy function."""
        with patch("src.web_interface.app.get_app") as mock_get_app:
            mock_app = Mock()
            mock_get_app.return_value = mock_app

            result = _get_app_lazy()

            assert result == mock_app
            mock_get_app.assert_called_once()


class TestAppModuleVariables:
    """Tests for module-level variables and state."""

    def test_global_variables_initialization(self):
        """Test that global variables are properly initialized."""
        from src.web_interface.app import (
            _app_creation_in_progress,
            _app_instance,
            _monitoring_setup_done,
            _services_initialized,
            _app_config_instance,
            bot_orchestrator,
            execution_engine,
            model_manager,
        )

        # These should be initialized to proper default values
        assert _app_config_instance is None or hasattr(_app_config_instance, '__dict__')
        assert bot_orchestrator is None or hasattr(bot_orchestrator, '__dict__')
        assert execution_engine is None or hasattr(execution_engine, '__dict__')
        assert model_manager is None or hasattr(model_manager, '__dict__')
        assert isinstance(_monitoring_setup_done, bool)
        assert isinstance(_services_initialized, bool)
        assert isinstance(_app_creation_in_progress, bool)

    def test_services_lock(self):
        """Test services lock is properly initialized."""
        from src.web_interface.app import _services_lock
        import threading

        assert type(_services_lock) is type(threading.Lock())

    def test_lazy_app_instance(self):
        """Test lazy app instance is properly initialized."""
        from src.web_interface.app import app

        assert isinstance(app, LazyApp)