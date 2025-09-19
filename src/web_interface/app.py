"""
Main FastAPI application for T-Bot Trading System.

This module creates and configures the FastAPI application with all
middleware, routes, and dependencies for the trading system web interface.
"""

# Global instances with thread safety
import os
import threading
from contextlib import asynccontextmanager
from typing import Any

import socketio
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import Config
from src.core.exceptions import ConfigurationError
from src.core.logging import correlation_context, get_logger
from src.monitoring import (
    AlertManager,
    MetricsCollector,
    NotificationConfig,
    OpenTelemetryConfig,
    PerformanceProfiler,
    instrument_fastapi,
    set_global_alert_manager,
    set_global_profiler,
    set_global_trading_tracer,
    setup_telemetry,
)

# New unified architecture imports
from src.web_interface.auth import get_auth_manager, initialize_auth_manager
from src.web_interface.constants import (
    ANONYMOUS_RATE_LIMIT,
    AUTHENTICATED_RATE_LIMIT,
    DEFAULT_JWT_EXPIRE_MINUTES,
    DEFAULT_PORT,
    DEFAULT_SESSION_TIMEOUT_MINUTES,
    DEV_ORIGINS,
)
from src.web_interface.facade import get_api_facade, get_service_registry

# Import middleware
from src.web_interface.middleware.auth import AuthMiddleware
from src.web_interface.middleware.correlation import CorrelationMiddleware
from src.web_interface.middleware.decimal_precision import (
    DecimalPrecisionMiddleware,
    DecimalValidationMiddleware,
)
from src.web_interface.middleware.error_handler import ErrorHandlerMiddleware
from src.web_interface.middleware.rate_limit import RateLimitMiddleware
from src.web_interface.middleware.security import SecurityMiddleware
from src.web_interface.versioning import VersioningMiddleware, VersionRoutingMiddleware
from src.web_interface.websockets import get_unified_websocket_manager

# Initialize logger
logger = get_logger(__name__)

app_config: "Config | None" = None
bot_orchestrator = None
execution_engine = None
model_manager = None
_monitoring_setup_done = False
_services_initialized = False
_services_lock = threading.Lock()


async def _initialize_services():
    """Initialize service layer and facade."""
    global _services_initialized

    with _services_lock:
        if _services_initialized:
            return

        # Initialize dependency injection container
        from src.core.dependency_injection import injector
        from src.web_interface.di_registration import register_web_interface_services

        # Register core services with DI container
        if app_config:
            injector.register_service("Config", app_config, singleton=True)
        if execution_engine:
            injector.register_service("ExecutionEngine", execution_engine, singleton=True)
            # If execution_engine provides ExecutionService interface, register it
            if hasattr(execution_engine, 'record_trade_execution'):
                injector.register_service("ExecutionService", execution_engine, singleton=True)
            # If execution_engine provides ExecutionOrchestrationService interface, register it
            if hasattr(execution_engine, 'execute_order_from_data'):
                injector.register_service("ExecutionOrchestrationService", execution_engine, singleton=True)
        if bot_orchestrator:
            injector.register_service("BotOrchestrator", bot_orchestrator, singleton=True)
        if model_manager:
            injector.register_service("ModelManager", model_manager, singleton=True)

        # Register web interface services using DI
        register_web_interface_services(injector)

        # Initialize authentication manager
        if app_config and hasattr(app_config, "security"):
            auth_config = {
                "jwt": {
                    "secret_key": app_config.security.secret_key,
                    "algorithm": getattr(app_config.security, "jwt_algorithm", "HS256"),
                    "access_token_expire_minutes": getattr(
                        app_config.security, "jwt_expire_minutes", DEFAULT_JWT_EXPIRE_MINUTES
                    ),
                    "refresh_token_expire_days": getattr(
                        app_config.security, "refresh_token_expire_days", 7
                    ),
                },
                "session": {
                    "timeout_minutes": getattr(app_config.security, "session_timeout_minutes", DEFAULT_SESSION_TIMEOUT_MINUTES)
                },
            }
            initialize_auth_manager(auth_config)
        else:
            from src.core.exceptions import ConfigurationError
            raise ConfigurationError("Security configuration is required for authentication", config_section="security")

        # Get services from DI container
        registry = injector.resolve("WebServiceRegistry")

        # Register services with registry using proper DI-created instances
        trading_service = injector.resolve("TradingService")
        bot_service = injector.resolve("BotManagementService")
        market_service = injector.resolve("MarketDataService")
        portfolio_service = injector.resolve("PortfolioService")
        risk_service = injector.resolve("RiskService")
        strategy_service = injector.resolve("StrategyServiceImpl")

        registry.register_service("trading", trading_service)
        registry.register_service("bot_management", bot_service)
        registry.register_service("market_data", market_service)
        registry.register_service("portfolio", portfolio_service)
        registry.register_service("risk_management", risk_service)
        registry.register_service("strategies", strategy_service)

        # Initialize API facade
        facade = get_api_facade()
        await facade.initialize()

        # Connect API endpoints to services for backward compatibility
        await _connect_api_endpoints_to_services(registry)

        _services_initialized = True


async def _connect_api_endpoints_to_services(registry):
    """Connect API endpoints to registered services for backward compatibility."""
    try:
        # Connect bot management API to the bot management service
        if registry.has_service("bot_management"):
            bot_mgmt_service = registry.get_service("bot_management")
            # Import here to avoid circular imports
            from src.web_interface.api import bot_management

            bot_management.set_bot_service(bot_mgmt_service)
            logger.info("Connected bot management API to service")

        # Add other API connections here as needed
        # Example: trading API, portfolio API, etc.

    except Exception as e:
        logger.error(f"Error connecting API endpoints to services: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.

    Handles startup and shutdown of the trading system components.
    """
    # Startup
    startup_correlation_id = correlation_context.generate_correlation_id()
    with correlation_context.correlation_context(startup_correlation_id):
        logger.info("Starting T-Bot web interface with unified architecture")

    try:
        # Initialize service layer
        await _initialize_services()

        # Initialize unified WebSocket manager
        try:
            websocket_manager = get_unified_websocket_manager()
            await websocket_manager.start()
            with correlation_context.correlation_context(startup_correlation_id):
                logger.info("Unified WebSocket manager started")
        except Exception as e:
            with correlation_context.correlation_context(startup_correlation_id):
                logger.error(f"Failed to start WebSocket manager: {e}")
                raise

        # Initialize trading system components
        if bot_orchestrator:
            await bot_orchestrator.start()
            with correlation_context.correlation_context(startup_correlation_id):
                logger.info("Bot orchestrator started")

        if execution_engine:
            await execution_engine.start()
            with correlation_context.correlation_context(startup_correlation_id):
                logger.info("Execution engine started")

        with correlation_context.correlation_context(startup_correlation_id):
            logger.info("T-Bot web interface startup completed")

        yield

    finally:
        # Shutdown
        shutdown_correlation_id = correlation_context.generate_correlation_id()
        with correlation_context.correlation_context(shutdown_correlation_id):
            logger.info("Shutting down T-Bot web interface")

        try:
            # Stop trading system components in reverse order of startup
            if execution_engine and hasattr(execution_engine, "stop"):
                try:
                    await execution_engine.stop()
                    with correlation_context.correlation_context(shutdown_correlation_id):
                        logger.info("Execution engine stopped")
                except Exception as e:
                    with correlation_context.correlation_context(shutdown_correlation_id):
                        logger.error(f"Error stopping execution engine: {e}")

            if bot_orchestrator and hasattr(bot_orchestrator, "stop"):
                try:
                    await bot_orchestrator.stop()
                    with correlation_context.correlation_context(shutdown_correlation_id):
                        logger.info("Bot orchestrator stopped")
                except Exception as e:
                    with correlation_context.correlation_context(shutdown_correlation_id):
                        logger.error(f"Error stopping bot orchestrator: {e}")

            # Stop unified WebSocket manager
            try:
                websocket_manager = get_unified_websocket_manager()
                await websocket_manager.stop()
                with correlation_context.correlation_context(shutdown_correlation_id):
                    logger.info("Unified WebSocket manager stopped")
            except Exception as e:
                with correlation_context.correlation_context(shutdown_correlation_id):
                    logger.error(f"Error stopping WebSocket manager: {e}")

            # Cleanup API facade and service registry
            try:
                facade = get_api_facade()
                await facade.cleanup()
                with correlation_context.correlation_context(shutdown_correlation_id):
                    logger.info("API facade cleaned up")
            except Exception as e:
                with correlation_context.correlation_context(shutdown_correlation_id):
                    logger.error(f"Error cleaning up API facade: {e}")

            try:
                registry = get_service_registry()
                # Cleanup all registered services safely
                service_names = []
                try:
                    service_names = registry.get_all_service_names()
                except Exception as e:
                    logger.warning(f"Could not get service names: {e}")

                for service_name in service_names:
                    try:
                        service = registry.get_service(service_name)
                        if hasattr(service, "cleanup"):
                            await service.cleanup()
                    except Exception as e:
                        logger.warning(f"Error cleaning up service {service_name}: {e}")

                # Clear all services using the cleanup method
                try:
                    await registry.cleanup_all()
                except Exception as e:
                    logger.warning(f"Error during registry cleanup_all: {e}")

                with correlation_context.correlation_context(shutdown_correlation_id):
                    logger.info("Service registry cleaned up")
            except Exception as e:
                with correlation_context.correlation_context(shutdown_correlation_id):
                    logger.error(f"Error cleaning up service registry: {e}")

            with correlation_context.correlation_context(shutdown_correlation_id):
                logger.info("T-Bot web interface shutdown completed")

        except Exception as e:
            with correlation_context.correlation_context(shutdown_correlation_id):
                logger.error(f"Error during shutdown: {e}")


def create_app(
    config: Config,
    bot_orchestrator_instance: "Any | None" = None,
    execution_engine_instance: "Any | None" = None,
    model_manager_instance: "Any | None" = None,
) -> Any:
    """
    Create and configure FastAPI application with Socket.IO.

    Args:
        config: Application configuration
        bot_orchestrator_instance: Bot orchestrator instance
        execution_engine_instance: Execution engine instance
        model_manager_instance: Model manager instance

    Returns:
        Combined ASGI app with FastAPI and Socket.IO
    """
    global app_config, bot_orchestrator, execution_engine, model_manager

    # Store global instances
    app_config = config
    bot_orchestrator = bot_orchestrator_instance
    execution_engine = execution_engine_instance
    model_manager = model_manager_instance

    # Create FastAPI app
    # Handle missing web_interface config gracefully
    web_config = getattr(
        config,
        "web_interface",
        {
            "debug": getattr(config, "debug", False),
            "jwt": {
                "secret_key": config.security.secret_key or os.environ.get("JWT_SECRET_KEY"),
                "algorithm": getattr(config.security, "jwt_algorithm", "HS256"),
                "access_token_expire_minutes": getattr(config.security, "jwt_expire_minutes", DEFAULT_JWT_EXPIRE_MINUTES),
            },
            "cors": {
                "allow_origins": DEV_ORIGINS,
                "allow_credentials": True,
                "allow_methods": ["*"],
                "allow_headers": ["*"],
            },
            "rate_limiting": {"anonymous_limit": ANONYMOUS_RATE_LIMIT, "authenticated_limit": AUTHENTICATED_RATE_LIMIT},
        },
    )

    fastapi_app = FastAPI(
        title="T-Bot Trading System API",
        description="Advanced cryptocurrency trading bot with ML-powered strategies",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
        debug=web_config.get("debug", False),
    )

    # Configure CORS
    cors_config = web_config.get("cors", {})
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_config.get("allow_origins", DEV_ORIGINS[:1]),
        allow_credentials=cors_config.get("allow_credentials", True),
        allow_methods=cors_config.get("allow_methods", ["*"]),
        allow_headers=cors_config.get("allow_headers", ["*"]),
        expose_headers=cors_config.get(
            "expose_headers",
            ["X-Process-Time", "X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"],
        ),
    )

    # Add custom middleware (order matters!)
    # 1. Error handling (outermost)
    fastapi_app.add_middleware(ErrorHandlerMiddleware, debug=web_config.get("debug", False))

    # 2. API Versioning - must be early in the chain
    fastapi_app.add_middleware(VersioningMiddleware)
    fastapi_app.add_middleware(VersionRoutingMiddleware)

    # 3. Correlation ID middleware for request tracking
    fastapi_app.add_middleware(CorrelationMiddleware)

    # 4. Security middleware
    fastapi_app.add_middleware(SecurityMiddleware, enable_csp=True, enable_input_validation=False)

    # 5. Connection pooling for performance
    try:
        from src.web_interface.middleware.connection_pool import (
            ConnectionPoolMiddleware,
            set_global_pool_manager,
        )

        # Create and set up connection pool middleware
        connection_pool_middleware = ConnectionPoolMiddleware(fastapi_app, config)
        set_global_pool_manager(connection_pool_middleware.pool_manager)

        # Add the middleware instance to FastAPI
        fastapi_app.add_middleware(ConnectionPoolMiddleware, config=config)
    except Exception as e:
        error_msg = f"Connection pool middleware setup failed: {e}"
        logger.error(error_msg)
        # In production, connection pooling is critical for performance
        if getattr(config, "environment", "development") == "production":
            raise RuntimeError(error_msg)

    # 6. Decimal precision middleware for financial data integrity
    try:
        fastapi_app.add_middleware(DecimalValidationMiddleware)
        fastapi_app.add_middleware(DecimalPrecisionMiddleware)
    except Exception as e:
        logger.warning(f"Decimal precision middleware setup failed: {e}")

    # 7. Rate limiting
    try:
        fastapi_app.add_middleware(RateLimitMiddleware, config=config)
    except Exception as e:
        logger.warning(f"Rate limiting middleware setup failed: {e}")

    # 8. Authentication middleware
    try:
        from src.web_interface.security.jwt_handler import JWTHandler

        jwt_handler = JWTHandler(config)
        fastapi_app.add_middleware(AuthMiddleware, jwt_handler=jwt_handler)
    except ImportError as e:
        logger.error(f"Auth middleware setup failed - authentication disabled: {e}")
        if config.environment == "production":
            raise RuntimeError("Cannot start without authentication in production")
    except Exception as e:
        logger.error(f"Auth middleware setup failed: {e}")
        if config.environment == "production":
            raise RuntimeError("Cannot start without authentication in production")

    # Add routes
    _register_routes(fastapi_app)

    # Add startup and shutdown events (additional to lifespan)
    # Events are now handled in the lifespan context manager

    # Setup monitoring infrastructure
    _setup_monitoring(fastapi_app, config)

    # Create unified WebSocket server and wrap FastAPI app
    websocket_manager = get_unified_websocket_manager()

    # Create Socket.IO server with CORS configuration
    cors_origins = cors_config.get("allow_origins", DEV_ORIGINS[:1])
    websocket_manager.create_server(cors_allowed_origins=cors_origins)

    # Wrap FastAPI app with Socket.IO
    combined_app = socketio.ASGIApp(
        websocket_manager.sio, other_asgi_app=fastapi_app, socketio_path="/socket.io/"
    )

    # Generate correlation ID for app creation completion
    app_creation_correlation_id = correlation_context.generate_correlation_id()
    with correlation_context.correlation_context(app_creation_correlation_id):
        logger.info("FastAPI application with unified WebSocket manager created")
    return combined_app


def _register_routes(app: FastAPI) -> None:
    """
    Register all API routes.

    Args:
        app: FastAPI application
    """

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        health_status = {
            "status": "healthy",
            "service": "t-bot-api",
            "version": "2.0.0",
            "architecture": "unified",
            "components": {},
        }

        try:
            facade = get_api_facade()
            if hasattr(facade, "health_check"):
                health_status["components"]["api_facade"] = await facade.health_check()
            else:
                health_status["components"]["api_facade"] = {"status": "available"}
        except Exception as e:
            health_status["components"]["api_facade"] = {"status": "error", "error": str(e)}

        try:
            auth_manager = get_auth_manager()
            if hasattr(auth_manager, "get_user_stats"):
                health_status["components"]["auth_manager"] = auth_manager.get_user_stats()
            else:
                health_status["components"]["auth_manager"] = {"status": "available"}
        except Exception as e:
            health_status["components"]["auth_manager"] = {"status": "error", "error": str(e)}

        try:
            websocket_manager = get_unified_websocket_manager()
            if hasattr(websocket_manager, "get_connection_stats"):
                health_status["components"]["websocket_connections"] = (
                    websocket_manager.get_connection_stats()
                )
            else:
                health_status["components"]["websocket_connections"] = {"status": "available"}
        except Exception as e:
            health_status["components"]["websocket_connections"] = {
                "status": "error",
                "error": str(e),
            }

        # If any component is in error, mark overall status as degraded
        if any(comp.get("status") == "error" for comp in health_status["components"].values()):
            health_status["status"] = "degraded"

        return health_status

    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "message": "T-Bot Trading System API - Unified Architecture",
            "version": "2.0.0",
            "api_versions": ["v1", "v1.1", "v2"],
            "default_version": "v2",
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
            "versioning": {
                "header": "X-API-Version",
                "path": "/api/v{version}/...",
                "query": "?api_version=v{version}",
            },
        }

    # API version info endpoint
    @app.get("/api/versions")
    async def api_versions():
        """Get API version information."""
        from src.web_interface.versioning import get_version_manager

        version_manager = get_version_manager()
        versions = version_manager.list_versions()

        return {
            "versions": [
                {
                    "version": v.version,
                    "status": v.status.value,
                    "features": list(v.features),
                    "release_date": v.release_date.isoformat() if v.release_date else None,
                    "deprecation_date": (
                        v.deprecation_date.isoformat() if v.deprecation_date else None
                    ),
                }
                for v in versions
            ],
            "default": version_manager.get_default_version().version,
            "latest": version_manager.get_latest_version().version,
        }

    # Register all API routers
    routers = [
        ("src.web_interface.api.auth", "router", "/api/auth", ["Authentication"]),
        (
            "src.web_interface.api.auth",
            "router",
            "/auth",
            ["Authentication Legacy"],
        ),  # Backward compatibility
        ("src.web_interface.api.health", "router", "/api/health", ["Health"]),
        ("src.web_interface.api.bot_management", "router", "/api/bot", ["Bot Management"]),
        ("src.web_interface.api.portfolio", "router", "/api/portfolio", ["Portfolio"]),
        ("src.web_interface.api.trading", "router", "/api/trading", ["Trading"]),
        ("src.web_interface.api.strategies", "router", "/api/strategies", ["Strategies"]),
        ("src.web_interface.api.risk", "router", "/api/risk", ["Risk Management"]),
        ("src.web_interface.api.ml_models", "router", "/api/ml", ["Machine Learning"]),
        ("src.web_interface.api.monitoring", "router", "/api/monitoring", ["Monitoring"]),
        ("src.web_interface.api.playground", "router", "/api/playground", ["Playground"]),
        ("src.web_interface.api.optimization", "router", "/api/optimization", ["Optimization"]),
        # New comprehensive API endpoints
        ("src.web_interface.api.analytics", "router", "", ["Analytics"]),
        ("src.web_interface.api.capital", "router", "", ["Capital Management"]),
        ("src.web_interface.api.data", "router", "", ["Data Management"]),
        ("src.web_interface.api.exchanges", "router", "", ["Exchange Management"]),
    ]

    # Track critical routers that must be available
    critical_routers = {
        "src.web_interface.api.auth",
        "src.web_interface.api.bot_management",
        "src.web_interface.api.trading",
    }
    failed_critical = []

    for module_path, router_name, prefix, tags in routers:
        try:
            import importlib

            module = importlib.import_module(module_path)
            router = getattr(module, router_name)
            app.include_router(router, prefix=prefix, tags=tags)
            logger.debug(f"Registered router: {module_path}")
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to import {module_path}: {e}")
            if module_path in critical_routers:
                failed_critical.append(module_path)

    # If any critical routers failed, raise an error
    if failed_critical:
        raise RuntimeError(f"Critical routers failed to load: {', '.join(failed_critical)}")

    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        """Add process time header to all responses."""
        import time

        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response


def _setup_monitoring(fastapi_app: FastAPI, config: Config) -> None:
    """
    Setup monitoring infrastructure for the application.

    Args:
        fastapi_app: FastAPI application
        config: Application configuration
    """
    global _monitoring_setup_done

    if _monitoring_setup_done:
        logger.debug("Monitoring infrastructure already setup, skipping")
        return

    try:
        # Generate correlation ID for monitoring setup
        monitoring_correlation_id = correlation_context.generate_correlation_id()
        with correlation_context.correlation_context(monitoring_correlation_id):
            logger.info("Setting up monitoring infrastructure")

        # Setup OpenTelemetry tracing
        try:
            telemetry_config = OpenTelemetryConfig(
                service_name="tbot-trading-system",
                service_version="1.0.0",
                environment=getattr(config, "environment", "development"),
                jaeger_enabled=False,  # Disabled by default - enable when Jaeger server is available
                console_enabled=False,  # Explicitly disable console exporter to prevent log pollution
            )
            trading_tracer = setup_telemetry(telemetry_config)
            set_global_trading_tracer(trading_tracer)
            instrument_fastapi(fastapi_app, telemetry_config)
            with correlation_context.correlation_context(monitoring_correlation_id):
                logger.info("OpenTelemetry tracing configured")
        except Exception as e:
            with correlation_context.correlation_context(monitoring_correlation_id):
                logger.warning(f"Failed to setup OpenTelemetry: {e}")

        # Setup metrics collection
        try:
            MetricsCollector()
            with correlation_context.correlation_context(monitoring_correlation_id):
                logger.info("Prometheus metrics collector configured")
        except Exception as e:
            with correlation_context.correlation_context(monitoring_correlation_id):
                logger.warning(f"Failed to setup metrics collector: {e}")

        # Setup performance profiler
        try:
            profiler = PerformanceProfiler(enable_memory_tracking=True, enable_cpu_profiling=True)
            set_global_profiler(profiler)
            with correlation_context.correlation_context(monitoring_correlation_id):
                logger.info("Performance profiler configured")
        except Exception as e:
            with correlation_context.correlation_context(monitoring_correlation_id):
                logger.warning(f"Failed to setup performance profiler: {e}")

        # Setup alert manager
        try:
            notification_config = NotificationConfig(
                email_from="tbot-alerts@example.com",
                email_to=["admin@example.com"],
                slack_webhook_url=getattr(config, "slack_webhook_url", ""),
                webhook_urls=getattr(config, "webhook_urls", []),
            )
            alert_manager = AlertManager(notification_config)
            set_global_alert_manager(alert_manager)
            with correlation_context.correlation_context(monitoring_correlation_id):
                logger.info("Alert manager configured")
        except Exception as e:
            with correlation_context.correlation_context(monitoring_correlation_id):
                logger.warning(f"Failed to setup alert manager: {e}")

        with correlation_context.correlation_context(monitoring_correlation_id):
            logger.info("Monitoring infrastructure setup completed")
        _monitoring_setup_done = True

    except Exception as e:
        with correlation_context.correlation_context(monitoring_correlation_id):
            logger.error(f"Error setting up monitoring infrastructure: {e}")


# Router creators removed - using direct imports instead


# Module-level app instance variables - DO NOT CREATE APP HERE
_app_instance = None
_app_config_instance = None
_app_creation_in_progress = False


def get_app():
    """Get or create the FastAPI app instance."""
    global _app_instance, _app_config_instance, _app_creation_in_progress

    # Prevent multiple simultaneous app creation
    if _app_creation_in_progress:
        logger.warning("App creation already in progress, returning None to prevent duplication")
        return None

    if _app_instance is None:
        _app_creation_in_progress = True
        try:
            # Generate correlation ID for app creation
            app_creation_correlation_id = correlation_context.generate_correlation_id()
            with correlation_context.correlation_context(app_creation_correlation_id):
                logger.info("Creating single application instance")
            _app_config_instance = Config()
            _app_instance = create_app(_app_config_instance)
            with correlation_context.correlation_context(app_creation_correlation_id):
                logger.info("Module-level combined FastAPI+Socket.IO app created successfully")
        except Exception as e:
            app_error_correlation_id = correlation_context.generate_correlation_id()
            with correlation_context.correlation_context(app_error_correlation_id):
                logger.error(f"Failed to create module-level app: {e}")
                # Create a minimal app as fallback
                _app_instance = FastAPI(title="T-Bot Trading System API (Fallback)")
                logger.warning("Created fallback FastAPI app")
        finally:
            _app_creation_in_progress = False

    return _app_instance


# Export app for ASGI servers - but delay creation until first access
def _get_app_lazy():
    """Lazy app getter that only creates app when needed."""
    return get_app()


# Lazy app creation for ASGI servers
_lazy_app = None


def get_asgi_app():
    """Get or create ASGI app for deployment."""
    global _lazy_app
    if _lazy_app is None:
        _lazy_app = get_app()
    return _lazy_app


# Export app for ASGI servers - uvicorn expects 'app' variable
# Use a lazy getter property to ensure app is created when accessed
class LazyApp:
    """Lazy app wrapper that creates app on first access."""

    def __getattr__(self, name):
        """Create and return app on first attribute access."""
        return getattr(get_asgi_app(), name)

    def __call__(self, *args, **kwargs):
        """Create and call app on first invocation."""
        return get_asgi_app()(*args, **kwargs)


app = LazyApp()


# Run the server when executed directly
if __name__ == "__main__":
    import uvicorn

    # Create app instance for running
    app_instance = get_asgi_app()

    # Get port from config or use default
    try:
        port = (
            _app_config_instance.api.port
            if _app_config_instance and hasattr(_app_config_instance, "api")
            else DEFAULT_PORT
        )
    except (AttributeError, TypeError, ValueError) as e:
        logger.warning(f"Failed to read port from config: {e}, using default port {DEFAULT_PORT}")
        port = DEFAULT_PORT
    except Exception as e:
        logger.error(f"Unexpected error reading configuration: {e}")
        raise ConfigurationError(
            "Failed to read application configuration",
            config_section="api.port",
            suggested_action="Check configuration file format and accessibility",
        ) from e
    host = "0.0.0.0"

    logger.info(f"Starting web server on {host}:{port}")

    uvicorn.run(app_instance, host=host, port=port, reload=False, log_level="info")
