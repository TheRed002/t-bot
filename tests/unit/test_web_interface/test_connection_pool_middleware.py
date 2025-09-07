"""
Test cases for web_interface connection_pool middleware.

This module tests the connection pool functionality for database and Redis connections
used in the web interface middleware.
"""

import asyncio
import os
import pytest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock, MagicMock, patch
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.pool import NullPool, AsyncAdaptedQueuePool
from fastapi import Request

from src.core.config import Config
from src.core.exceptions import ValidationError, ServiceError
from src.web_interface.middleware.connection_pool import (
    PoolAsyncUnitOfWork,
    ConnectionPoolManager,
    ConnectionPoolMiddleware,
    ConnectionHealthMonitor,
    get_global_pool_manager,
    set_global_pool_manager,
    get_db_connection,
    get_redis_connection,
    get_uow
)


class TestPoolAsyncUnitOfWork:
    """Test PoolAsyncUnitOfWork functionality."""

    @pytest.fixture
    def session_factory(self):
        """Create mock session factory."""
        mock_session = AsyncMock(spec=AsyncSession)
        factory = Mock(return_value=mock_session)
        return factory

    @pytest.fixture
    def uow(self, session_factory):
        """Create PoolAsyncUnitOfWork instance."""
        return PoolAsyncUnitOfWork(session_factory)

    async def test_init(self, session_factory):
        """Test UoW initialization."""
        uow = PoolAsyncUnitOfWork(session_factory)
        assert uow.session_factory == session_factory
        assert uow.session is None

    async def test_async_context_manager_success(self, uow):
        """Test successful context manager usage."""
        mock_session = AsyncMock()
        uow.session_factory.return_value = mock_session
        
        async with uow as context:
            assert context == uow
            assert uow.session == mock_session
        
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()

    async def test_async_context_manager_exception(self, uow):
        """Test context manager with exception."""
        mock_session = AsyncMock()
        uow.session_factory.return_value = mock_session
        
        with pytest.raises(ValueError):
            async with uow:
                raise ValueError("test error")
        
        mock_session.rollback.assert_called_once()
        mock_session.close.assert_called_once()

    async def test_commit(self, uow):
        """Test commit method."""
        mock_session = AsyncMock()
        uow.session = mock_session
        
        await uow.commit()
        mock_session.commit.assert_called_once()

    async def test_commit_no_session(self, uow):
        """Test commit with no session."""
        await uow.commit()  # Should not raise

    async def test_rollback(self, uow):
        """Test rollback method."""
        mock_session = AsyncMock()
        uow.session = mock_session
        
        await uow.rollback()
        mock_session.rollback.assert_called_once()

    async def test_rollback_no_session(self, uow):
        """Test rollback with no session."""
        await uow.rollback()  # Should not raise

    async def test_close(self, uow):
        """Test close method."""
        mock_session = AsyncMock()
        uow.session = mock_session
        
        await uow.close()
        mock_session.close.assert_called_once()
        assert uow.session is None

    async def test_close_no_session(self, uow):
        """Test close with no session."""
        await uow.close()  # Should not raise


class TestConnectionPoolManager:
    """Test ConnectionPoolManager functionality."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        config = Mock(spec=Config)
        config.debug = False
        config.db_pool_min = 5
        config.db_pool_max = 20
        config.db_pool_timeout = 30.0
        config.db_idle_timeout = 300.0
        config.db_connection_lifetime = 3600.0
        config.redis_pool_min = 5
        config.redis_pool_max = 50
        config.redis_pool_timeout = 10.0
        return config

    @pytest.fixture
    def pool_manager(self, mock_config):
        """Create ConnectionPoolManager instance."""
        return ConnectionPoolManager(mock_config)

    def test_init(self, mock_config):
        """Test pool manager initialization."""
        manager = ConnectionPoolManager(mock_config)
        assert manager.config == mock_config
        assert manager._async_engine is None
        assert manager._async_session_factory is None
        assert manager.redis_client is None
        assert not manager._initialized

    def test_pool_config_defaults(self, mock_config):
        """Test default pool configuration."""
        manager = ConnectionPoolManager(mock_config)
        
        assert manager.db_pool_config["min_connections"] == 5
        assert manager.db_pool_config["max_connections"] == 20
        assert manager.redis_pool_config["min_connections"] == 5
        assert manager.redis_pool_config["max_connections"] == 50

    @patch.dict(os.environ, {"PYTEST_CURRENT_TEST": "test"})
    async def test_initialize_with_pytest(self, pool_manager):
        """Test initialization in pytest environment."""
        with patch.object(pool_manager, '_initialize_database_pool') as mock_db_init, \
             patch.object(pool_manager, '_initialize_redis_pool') as mock_redis_init:
            await pool_manager.initialize()
            
            mock_db_init.assert_called_once()
            mock_redis_init.assert_called_once()
            assert pool_manager._initialized

    async def test_initialize_already_initialized(self, pool_manager):
        """Test initialization when already initialized."""
        pool_manager._initialized = True
        
        with patch.object(pool_manager, '_initialize_database_pool') as mock_db_init:
            await pool_manager.initialize()
            mock_db_init.assert_not_called()

    async def test_initialize_failure(self, pool_manager):
        """Test initialization failure."""
        with patch.object(pool_manager, '_initialize_database_pool', side_effect=Exception("DB error")):
            with pytest.raises(Exception, match="DB error"):
                await pool_manager.initialize()

    @patch("src.web_interface.middleware.connection_pool.create_async_engine")
    async def test_initialize_database_pool_success(self, mock_create_engine, pool_manager):
        """Test successful database pool initialization."""
        mock_config = pool_manager.config
        mock_config.get_async_database_url = Mock(return_value="postgresql+asyncpg://test")
        
        mock_engine = AsyncMock()
        mock_connection = AsyncMock()
        mock_engine.begin.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        with patch("src.web_interface.middleware.connection_pool.SQLALCHEMY_AVAILABLE", True):
            await pool_manager._initialize_database_pool()
        
        assert pool_manager._async_engine == mock_engine
        mock_connection.execute.assert_called_once()

    async def test_initialize_database_pool_no_url(self, pool_manager):
        """Test database pool initialization with no URL."""
        with patch("src.web_interface.middleware.connection_pool.SQLALCHEMY_AVAILABLE", True):
            await pool_manager._initialize_database_pool()
        
        assert pool_manager._async_engine is None

    @patch("src.web_interface.middleware.connection_pool.RedisClient")
    async def test_initialize_redis_pool_success(self, mock_redis_client_class, pool_manager):
        """Test successful Redis pool initialization."""
        mock_redis_client = AsyncMock()
        mock_redis_client_class.return_value = mock_redis_client
        
        await pool_manager._initialize_redis_pool()
        
        mock_redis_client.connect.assert_called_once()
        mock_redis_client.ping.assert_called_once()
        assert pool_manager.redis_client == mock_redis_client

    @patch("src.web_interface.middleware.connection_pool.RedisClient")
    async def test_initialize_redis_pool_failure(self, mock_redis_client_class, pool_manager):
        """Test Redis pool initialization failure."""
        mock_redis_client = AsyncMock()
        mock_redis_client.connect.side_effect = Exception("Redis error")
        mock_redis_client_class.return_value = mock_redis_client
        
        await pool_manager._initialize_redis_pool()
        assert pool_manager.redis_client is None

    async def test_get_db_connection_not_initialized(self, pool_manager):
        """Test getting DB connection when not initialized."""
        mock_session = AsyncMock()
        mock_session_factory = Mock()
        
        # Create a proper async context manager mock
        async_context_mock = AsyncMock()
        async_context_mock.__aenter__ = AsyncMock(return_value=mock_session)
        async_context_mock.__aexit__ = AsyncMock(return_value=None)
        mock_session_factory.return_value = async_context_mock
        
        with patch.object(pool_manager, 'initialize') as mock_init:
            # Mock the initialization to set up the session factory
            def mock_initialize():
                pool_manager._initialized = True
                pool_manager._async_session_factory = mock_session_factory
            
            mock_init.side_effect = mock_initialize
            
            async with pool_manager.get_db_connection():
                pass
            mock_init.assert_called_once()

    async def test_get_db_connection_no_factory(self, pool_manager):
        """Test getting DB connection with no session factory."""
        pool_manager._initialized = True
        
        with pytest.raises(RuntimeError, match="Async database pool not initialized"):
            async with pool_manager.get_db_connection():
                pass

    async def test_get_db_connection_success(self, pool_manager):
        """Test successful DB connection retrieval."""
        pool_manager._initialized = True
        mock_session = AsyncMock()
        mock_session_factory = Mock()
        
        # Create a proper async context manager mock
        async_context_mock = AsyncMock()
        async_context_mock.__aenter__ = AsyncMock(return_value=mock_session)
        async_context_mock.__aexit__ = AsyncMock(return_value=None)
        mock_session_factory.return_value = async_context_mock
        pool_manager._async_session_factory = mock_session_factory
        
        async with pool_manager.get_db_connection() as session:
            assert session == mock_session
        
        mock_session.commit.assert_called_once()

    async def test_get_db_connection_exception(self, pool_manager):
        """Test DB connection with exception."""
        pool_manager._initialized = True
        mock_session = AsyncMock()
        mock_session_factory = Mock()
        
        # Create a proper async context manager mock
        async_context_mock = AsyncMock()
        async_context_mock.__aenter__ = AsyncMock(return_value=mock_session)
        async_context_mock.__aexit__ = AsyncMock(return_value=None)
        mock_session_factory.return_value = async_context_mock
        pool_manager._async_session_factory = mock_session_factory
        
        with pytest.raises(ValueError):
            async with pool_manager.get_db_connection():
                raise ValueError("test error")
        
        mock_session.rollback.assert_called_once()

    async def test_get_uow_success(self, pool_manager):
        """Test successful UoW retrieval."""
        pool_manager._initialized = True
        mock_uow = Mock()
        mock_uow.__aenter__ = AsyncMock(return_value=mock_uow)
        mock_uow.__aexit__ = AsyncMock()
        pool_manager._uow_factory = Mock(return_value=mock_uow)
        
        async with pool_manager.get_uow() as uow:
            assert uow == mock_uow

    async def test_get_redis_connection_success(self, pool_manager):
        """Test successful Redis connection retrieval."""
        pool_manager._initialized = True
        mock_redis = Mock()
        pool_manager.redis_client = mock_redis
        
        async with pool_manager.get_redis_connection() as redis:
            assert redis == mock_redis

    async def test_get_redis_connection_not_initialized(self, pool_manager):
        """Test Redis connection when client not initialized."""
        pool_manager._initialized = True
        
        with pytest.raises(RuntimeError, match="Redis client not initialized"):
            async with pool_manager.get_redis_connection():
                pass

    async def test_health_check_success(self, pool_manager):
        """Test successful health check."""
        pool_manager._initialized = True
        
        # Mock database connection
        mock_session = AsyncMock()
        mock_result = Mock()
        mock_result.scalar.return_value = 1
        mock_session.execute.return_value = mock_result
        mock_session_factory = Mock()
        
        # Create a proper async context manager mock
        async_context_mock = AsyncMock()
        async_context_mock.__aenter__ = AsyncMock(return_value=mock_session)
        async_context_mock.__aexit__ = AsyncMock(return_value=None)
        mock_session_factory.return_value = async_context_mock
        pool_manager._async_session_factory = mock_session_factory
        
        # Mock Redis client
        mock_redis = AsyncMock()
        mock_redis.ping.return_value = True
        mock_redis.info.return_value = {"connected_clients": 5, "used_memory_human": "1MB"}
        pool_manager.redis_client = mock_redis
        
        result = await pool_manager.health_check()
        
        assert result["database"]["status"] == "healthy"
        assert result["redis"]["status"] == "healthy"

    async def test_close_pools(self, pool_manager):
        """Test closing connection pools."""
        mock_engine = AsyncMock()
        mock_redis = AsyncMock()
        pool_manager._async_engine = mock_engine
        pool_manager.redis_client = mock_redis
        pool_manager._initialized = True
        
        await pool_manager.close()
        
        mock_engine.dispose.assert_called_once()
        mock_redis.disconnect.assert_called_once()
        assert not pool_manager._initialized


class TestConnectionPoolMiddleware:
    """Test ConnectionPoolMiddleware functionality."""

    @pytest.fixture
    def mock_app(self):
        """Create mock FastAPI app."""
        return Mock()

    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        return Mock(spec=Config)

    @pytest.fixture
    def middleware(self, mock_app, mock_config):
        """Create middleware instance."""
        return ConnectionPoolMiddleware(mock_app, mock_config)

    def test_init(self, mock_app, mock_config):
        """Test middleware initialization."""
        middleware = ConnectionPoolMiddleware(mock_app, mock_config)
        assert middleware.config == mock_config
        assert isinstance(middleware.pool_manager, ConnectionPoolManager)

    async def test_dispatch_initializes_pool(self, middleware):
        """Test dispatch initializes pool manager."""
        mock_request = Mock()
        mock_request.state = Mock()
        mock_call_next = AsyncMock(return_value=Mock())
        
        with patch.object(middleware.pool_manager, 'initialize') as mock_init:
            await middleware.dispatch(mock_request, mock_call_next)
            mock_init.assert_called_once()

    async def test_dispatch_sets_request_state(self, middleware):
        """Test dispatch sets pool manager in request state."""
        mock_request = Mock()
        mock_request.state = Mock()
        mock_call_next = AsyncMock(return_value=Mock())
        middleware.pool_manager._initialized = True
        
        await middleware.dispatch(mock_request, mock_call_next)
        
        assert mock_request.state.pool_manager == middleware.pool_manager
        assert hasattr(mock_request.state, 'get_db_connection')
        assert hasattr(mock_request.state, 'get_redis_connection')
        assert hasattr(mock_request.state, 'get_uow')

    async def test_startup(self, middleware):
        """Test startup event handler."""
        with patch.object(middleware.pool_manager, 'initialize') as mock_init:
            await middleware.startup()
            mock_init.assert_called_once()

    async def test_shutdown(self, middleware):
        """Test shutdown event handler."""
        with patch.object(middleware.pool_manager, 'close') as mock_close:
            await middleware.shutdown()
            mock_close.assert_called_once()


class TestGlobalPoolManager:
    """Test global pool manager functions."""

    def test_get_global_pool_manager_none(self):
        """Test getting global pool manager when none set."""
        with patch('src.web_interface.middleware.connection_pool._global_pool_manager', None):
            assert get_global_pool_manager() is None

    def test_set_and_get_global_pool_manager(self):
        """Test setting and getting global pool manager."""
        mock_manager = Mock()
        set_global_pool_manager(mock_manager)
        assert get_global_pool_manager() == mock_manager

    async def test_get_db_connection_dependency(self):
        """Test database connection dependency function."""
        mock_connection = Mock()
        
        # Mock the pool manager's get_db_connection method to return the connection directly
        mock_manager = AsyncMock()
        
        @asynccontextmanager
        async def mock_get_db_connection():
            yield mock_connection
            
        mock_manager.get_db_connection = mock_get_db_connection
        
        with patch('src.web_interface.middleware.connection_pool.get_global_pool_manager', return_value=mock_manager):
            async for conn in get_db_connection():
                assert conn == mock_connection

    async def test_get_db_connection_no_pool_manager(self):
        """Test database connection dependency with no pool manager."""
        with patch('src.web_interface.middleware.connection_pool.get_global_pool_manager', return_value=None):
            with pytest.raises(RuntimeError, match="Connection pool not initialized"):
                async for _ in get_db_connection():
                    pass

    async def test_get_redis_connection_dependency(self):
        """Test Redis connection dependency function."""
        mock_connection = Mock()
        
        # Mock the pool manager's get_redis_connection method to return the connection directly
        mock_manager = AsyncMock()
        
        @asynccontextmanager
        async def mock_get_redis_connection():
            yield mock_connection
            
        mock_manager.get_redis_connection = mock_get_redis_connection
        
        with patch('src.web_interface.middleware.connection_pool.get_global_pool_manager', return_value=mock_manager):
            async for conn in get_redis_connection():
                assert conn == mock_connection

    async def test_get_uow_dependency(self):
        """Test UoW dependency function."""
        mock_uow = Mock()
        
        # Mock the pool manager's get_uow method to return the uow directly
        mock_manager = AsyncMock()
        
        @asynccontextmanager
        async def mock_get_uow():
            yield mock_uow
            
        mock_manager.get_uow = mock_get_uow
        
        with patch('src.web_interface.middleware.connection_pool.get_global_pool_manager', return_value=mock_manager):
            async for uow in get_uow():
                assert uow == mock_uow


class TestConnectionHealthMonitor:
    """Test ConnectionHealthMonitor functionality."""

    @pytest.fixture
    def mock_pool_manager(self):
        """Create mock pool manager."""
        return AsyncMock()

    @pytest.fixture
    def health_monitor(self, mock_pool_manager):
        """Create health monitor instance."""
        return ConnectionHealthMonitor(mock_pool_manager)

    def test_init(self, mock_pool_manager):
        """Test health monitor initialization."""
        monitor = ConnectionHealthMonitor(mock_pool_manager)
        assert monitor.pool_manager == mock_pool_manager
        assert monitor.monitoring_enabled
        assert monitor.check_interval == 60.0
        assert monitor._monitor_task is None

    async def test_start_monitoring(self, health_monitor):
        """Test starting monitoring."""
        with patch('asyncio.create_task') as mock_create_task:
            await health_monitor.start_monitoring()
            mock_create_task.assert_called_once()
            assert health_monitor.monitoring_enabled

    async def test_start_monitoring_already_started(self, health_monitor):
        """Test starting monitoring when already started."""
        health_monitor._monitor_task = Mock()
        
        with patch('asyncio.create_task') as mock_create_task:
            await health_monitor.start_monitoring()
            mock_create_task.assert_not_called()

    async def test_stop_monitoring(self, health_monitor):
        """Test stopping monitoring."""
        # Create a future that will be cancelled
        mock_task = asyncio.Future()
        mock_task.cancel()  # Pre-cancel the task
        
        # Override cancel method to track calls
        cancel_called = Mock()
        original_cancel = mock_task.cancel
        def mock_cancel():
            cancel_called()
            return original_cancel()
        mock_task.cancel = mock_cancel
        
        health_monitor._monitor_task = mock_task
        
        await health_monitor.stop_monitoring()
        
        assert not health_monitor.monitoring_enabled
        cancel_called.assert_called_once()

    async def test_stop_monitoring_not_started(self, health_monitor):
        """Test stopping monitoring when not started."""
        await health_monitor.stop_monitoring()  # Should not raise

    async def test_monitoring_loop_cancelled(self, health_monitor):
        """Test monitoring loop handles cancellation."""
        health_monitor.monitoring_enabled = True
        
        with patch.object(health_monitor, '_perform_health_checks', side_effect=asyncio.CancelledError):
            with patch('asyncio.sleep') as mock_sleep:
                await health_monitor._monitoring_loop()
                mock_sleep.assert_not_called()

    async def test_monitoring_loop_exception(self, health_monitor):
        """Test monitoring loop handles exceptions."""
        health_monitor.monitoring_enabled = True
        
        # Create a side effect that raises exception first, then disables monitoring
        def side_effect_exception():
            if health_monitor.monitoring_enabled:
                health_monitor.monitoring_enabled = False  # Stop after first exception
                raise Exception("test error")
        
        with patch.object(health_monitor, '_perform_health_checks', side_effect=side_effect_exception):
            with patch('asyncio.sleep') as mock_sleep:
                await health_monitor._monitoring_loop()
                mock_sleep.assert_called_once()

    async def test_perform_health_checks_success(self, health_monitor, mock_pool_manager):
        """Test successful health checks."""
        health_results = {
            "database": {"status": "healthy"},
            "redis": {"status": "healthy"}
        }
        pool_stats = {
            "database": {"pool_info": {"size": 10, "checked_out": 2}},
            "redis": {"pool_info": {}}
        }
        
        mock_pool_manager.health_check.return_value = health_results
        mock_pool_manager.get_pool_stats.return_value = pool_stats
        
        await health_monitor._perform_health_checks()
        
        mock_pool_manager.health_check.assert_called_once()
        mock_pool_manager.get_pool_stats.assert_called_once()

    async def test_perform_health_checks_unhealthy(self, health_monitor, mock_pool_manager):
        """Test health checks with unhealthy status."""
        health_results = {
            "database": {"status": "unhealthy", "details": {"error": "Connection failed"}},
            "redis": {"status": "unhealthy", "details": {"error": "Redis down"}}
        }
        
        mock_pool_manager.health_check.return_value = health_results
        mock_pool_manager.get_pool_stats.return_value = {}
        
        with patch.object(health_monitor, '_alert_connection_issue') as mock_alert:
            await health_monitor._perform_health_checks()
            assert mock_alert.call_count == 2

    async def test_check_pool_utilization_high_usage(self, health_monitor):
        """Test pool utilization check with high usage."""
        pool_stats = {
            "database": {
                "pool_info": {"size": 10, "checked_out": 9}  # 90% utilization
            },
            "redis": {
                "pool_info": {"connected_clients": 5}
            }
        }
        
        with patch.object(health_monitor, '_alert_high_utilization') as mock_alert:
            await health_monitor._check_pool_utilization(pool_stats)
            mock_alert.assert_called_once_with("database", 0.9)

    async def test_alert_connection_issue(self, health_monitor):
        """Test connection issue alerting."""
        details = {"error": "Connection failed"}
        await health_monitor._alert_connection_issue("database", details)
        # This is mainly for logging, no assertions needed

    async def test_alert_high_utilization(self, health_monitor):
        """Test high utilization alerting."""
        await health_monitor._alert_high_utilization("database", 0.9)
        # This is mainly for logging, no assertions needed