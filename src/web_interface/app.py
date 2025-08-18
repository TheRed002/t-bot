"""
Main FastAPI application for T-Bot Trading System.

This module creates and configures the FastAPI application with all
middleware, routes, and dependencies for the trading system web interface.
"""

from contextlib import asynccontextmanager

import socketio
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import Config
from src.core.logging import correlation_context, get_logger
from src.monitoring.alerting import AlertManager, NotificationConfig, set_global_alert_manager
from src.monitoring.metrics import MetricsCollector
from src.monitoring.performance import PerformanceProfiler, set_global_profiler
from src.monitoring.telemetry import (
    OpenTelemetryConfig,
    instrument_fastapi,
    set_global_trading_tracer,
    setup_telemetry,
)
from src.web_interface.middleware.auth import AuthMiddleware
from src.web_interface.middleware.error_handler import ErrorHandlerMiddleware
from src.web_interface.middleware.rate_limit import RateLimitMiddleware
from src.web_interface.security.auth import init_auth

logger = get_logger(__name__)

# Global instances
app_config: Config = None
bot_orchestrator = None
execution_engine = None
model_manager = None
_monitoring_setup_done = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.

    Handles startup and shutdown of the trading system components.
    """
    # Startup
    # Generate correlation ID for startup sequence
    startup_correlation_id = correlation_context.generate_correlation_id()
    with correlation_context.correlation_context(startup_correlation_id):
        logger.info("Starting T-Bot web interface")

    try:
        # Initialize authentication system
        init_auth(app_config)

        # Start Socket.IO background tasks
        try:
            from src.web_interface.socketio_manager import socketio_manager

            await socketio_manager.start_background_tasks()
            with correlation_context.correlation_context(startup_correlation_id):
                logger.info("Socket.IO background tasks started")
        except Exception as e:
            with correlation_context.correlation_context(startup_correlation_id):
                logger.error(f"Failed to start Socket.IO background tasks: {e}")

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
        # Generate correlation ID for shutdown sequence
        shutdown_correlation_id = correlation_context.generate_correlation_id()
        with correlation_context.correlation_context(shutdown_correlation_id):
            logger.info("Shutting down T-Bot web interface")

        try:
            # Stop trading system components
            if execution_engine:
                await execution_engine.stop()
                with correlation_context.correlation_context(shutdown_correlation_id):
                    logger.info("Execution engine stopped")

            if bot_orchestrator:
                await bot_orchestrator.stop()
                with correlation_context.correlation_context(shutdown_correlation_id):
                    logger.info("Bot orchestrator stopped")

            # Stop Socket.IO background tasks
            try:
                from src.web_interface.socketio_manager import socketio_manager

                await socketio_manager.stop_background_tasks()
                with correlation_context.correlation_context(shutdown_correlation_id):
                    logger.info("Socket.IO background tasks stopped")
            except Exception as e:
                with correlation_context.correlation_context(shutdown_correlation_id):
                    logger.error(f"Failed to stop Socket.IO background tasks: {e}")

            with correlation_context.correlation_context(shutdown_correlation_id):
                logger.info("T-Bot web interface shutdown completed")

        except Exception as e:
            with correlation_context.correlation_context(shutdown_correlation_id):
                logger.error(f"Error during shutdown: {e}")


def create_app(
    config: Config,
    bot_orchestrator_instance=None,
    execution_engine_instance=None,
    model_manager_instance=None,
):
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
            "debug": config.debug,
            "jwt": {
                "secret_key": config.security.secret_key,
                "algorithm": config.security.jwt_algorithm,
                "access_token_expire_minutes": config.security.jwt_expire_minutes,
            },
            "cors": {
                "allow_origins": ["http://localhost:3000", "http://testserver"],
                "allow_credentials": True,
                "allow_methods": ["*"],
                "allow_headers": ["*"],
            },
            "rate_limiting": {"anonymous_limit": 1000, "authenticated_limit": 5000},
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
        allow_origins=cors_config.get("allow_origins", ["http://localhost:3000"]),
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

    # 1.5. Correlation ID middleware for request tracking
    from src.web_interface.middleware.correlation import CorrelationMiddleware

    fastapi_app.add_middleware(CorrelationMiddleware)

    # 2. Security middleware
    from src.web_interface.middleware.security import SecurityMiddleware

    fastapi_app.add_middleware(SecurityMiddleware, enable_csp=True, enable_input_validation=True)

    # 2. Connection pooling for performance
    from src.web_interface.middleware.connection_pool import (
        ConnectionPoolMiddleware,
        set_global_pool_manager,
    )

    connection_pool_middleware = ConnectionPoolMiddleware(fastapi_app, config)
    fastapi_app.add_middleware(ConnectionPoolMiddleware, config=config)
    set_global_pool_manager(connection_pool_middleware.pool_manager)

    # 3. Decimal precision middleware for financial data integrity
    from src.web_interface.middleware.decimal_precision import (
        DecimalPrecisionMiddleware,
        DecimalValidationMiddleware,
    )

    fastapi_app.add_middleware(DecimalValidationMiddleware)
    fastapi_app.add_middleware(DecimalPrecisionMiddleware)

    # 4. Rate limiting
    fastapi_app.add_middleware(RateLimitMiddleware, config=config)

    # 5. Authentication (innermost)
    from src.web_interface.security.jwt_handler import JWTHandler

    jwt_handler = JWTHandler(config)
    fastapi_app.add_middleware(AuthMiddleware, jwt_handler=jwt_handler)

    # Add routes
    _register_routes(fastapi_app)

    # Add startup and shutdown events (additional to lifespan)
    # Events are now handled in the lifespan context manager

    # Setup monitoring infrastructure
    _setup_monitoring(fastapi_app, config)

    # Create Socket.IO server and wrap FastAPI app
    from src.web_interface.socketio_manager import socketio_manager

    # Create Socket.IO server with CORS configuration
    cors_origins = cors_config.get("allow_origins", ["http://localhost:3000"])
    socketio_manager.create_server(cors_allowed_origins=cors_origins)

    # Wrap FastAPI app with Socket.IO
    combined_app = socketio.ASGIApp(
        socketio_manager.sio, other_asgi_app=fastapi_app, socketio_path="/socket.io/"
    )

    # Generate correlation ID for app creation completion
    app_creation_correlation_id = correlation_context.generate_correlation_id()
    with correlation_context.correlation_context(app_creation_correlation_id):
        logger.info("FastAPI application with Socket.IO created and configured")
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
        return {"status": "healthy", "service": "t-bot-api", "version": "1.0.0"}

    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "message": "T-Bot Trading System API",
            "version": "1.0.0",
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
        }

    # Authentication routes
    try:
        from src.web_interface.api.auth import router as auth_router

        app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
    except ImportError as e:
        logger.warning(f"Failed to import auth router: {e}")

    # Bot management routes
    try:
        from src.web_interface.api.bot_management import router as bot_router

        app.include_router(bot_router, prefix="/api/bots", tags=["Bot Management"])
    except ImportError as e:
        logger.warning(f"Failed to import bot management router: {e}")

    # Portfolio routes
    try:
        from src.web_interface.api.portfolio import router as portfolio_router

        app.include_router(portfolio_router, prefix="/api/portfolio", tags=["Portfolio"])
    except ImportError as e:
        logger.warning(f"Failed to import portfolio router: {e}")

    # Trading routes
    try:
        from src.web_interface.api.trading import router as trading_router

        app.include_router(trading_router, prefix="/api/trading", tags=["Trading"])
    except ImportError as e:
        logger.warning(f"Failed to import trading router: {e}")

    # Strategy routes
    try:
        from src.web_interface.api.strategies import router as strategy_router

        app.include_router(strategy_router, prefix="/api/strategies", tags=["Strategies"])
    except ImportError as e:
        logger.warning(f"Failed to import strategies router: {e}")

    # Risk management routes
    try:
        from src.web_interface.api.risk import router as risk_router

        app.include_router(risk_router, prefix="/api/risk", tags=["Risk Management"])
    except ImportError as e:
        logger.warning(f"Failed to import risk router: {e}")

    # ML model routes
    try:
        from src.web_interface.api.ml_models import router as ml_router

        app.include_router(ml_router, prefix="/api/ml", tags=["Machine Learning"])
    except ImportError as e:
        logger.warning(f"Failed to import ML models router: {e}")

    # Monitoring routes
    try:
        from src.web_interface.api.monitoring import router as monitoring_router

        app.include_router(monitoring_router, prefix="/api/monitoring", tags=["Monitoring"])
    except ImportError as e:
        logger.warning(f"Failed to import monitoring router: {e}")

    # Playground routes
    try:
        from src.web_interface.api.playground import router as playground_router

        app.include_router(playground_router, prefix="/api/playground", tags=["Playground"])
    except ImportError as e:
        logger.warning(f"Failed to import playground router: {e}")

    # Optimization routes
    try:
        from src.web_interface.api.optimization import router as optimization_router

        app.include_router(optimization_router, prefix="/api/optimization", tags=["Optimization"])
    except ImportError as e:
        logger.warning(f"Failed to import optimization router: {e}")

    # Don't add Socket.IO here - it will be wrapped around the entire app
    pass

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
            metrics_collector = MetricsCollector()
            # Store globally for access
            import src.monitoring.metrics

            src.monitoring.metrics._global_collector = metrics_collector
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


# API route placeholders (will be implemented in the next steps)
def create_auth_router():
    """Create authentication router - placeholder."""
    from fastapi import APIRouter

    router = APIRouter()

    @router.get("/status")
    async def auth_status():
        return {"status": "auth router placeholder"}

    return router


def create_bot_router():
    """Create bot management router - placeholder."""
    from fastapi import APIRouter

    router = APIRouter()

    @router.get("/")
    async def list_bots():
        return {"bots": [], "message": "bot router placeholder"}

    return router


def create_portfolio_router():
    """Create portfolio router - placeholder."""
    from fastapi import APIRouter

    router = APIRouter()

    @router.get("/summary")
    async def portfolio_summary():
        return {"summary": {}, "message": "portfolio router placeholder"}

    return router


def create_trading_router():
    """Create trading router - placeholder."""
    from fastapi import APIRouter

    router = APIRouter()

    @router.get("/orders")
    async def list_orders():
        return {"orders": [], "message": "trading router placeholder"}

    return router


def create_strategy_router():
    """Create strategy router - placeholder."""
    from fastapi import APIRouter

    router = APIRouter()

    @router.get("/")
    async def list_strategies():
        return {"strategies": [], "message": "strategy router placeholder"}

    return router


def create_risk_router():
    """Create risk management router - placeholder."""
    from fastapi import APIRouter

    router = APIRouter()

    @router.get("/metrics")
    async def risk_metrics():
        return {"metrics": {}, "message": "risk router placeholder"}

    return router


def create_ml_router():
    """Create ML model router - placeholder."""
    from fastapi import APIRouter

    router = APIRouter()

    @router.get("/models")
    async def list_models():
        return {"models": [], "message": "ml router placeholder"}

    return router


# Store router creators for dynamic import
ROUTER_CREATORS = {
    "auth": create_auth_router,
    "bot_management": create_bot_router,
    "portfolio": create_portfolio_router,
    "trading": create_trading_router,
    "strategies": create_strategy_router,
    "risk": create_risk_router,
    "ml_models": create_ml_router,
}


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


# Create a property-like access for the app
class AppProxy:
    def __getattr__(self, name):
        app_instance = get_app()
        if app_instance is None:
            raise RuntimeError("Application instance is not available")
        return getattr(app_instance, name)


app = AppProxy()


# Run the server when executed directly
if __name__ == "__main__":
    import uvicorn

    # Get port from config or use default
    try:
        port = (
            _app_config_instance.api.port
            if _app_config_instance and hasattr(_app_config_instance, "api")
            else 8000
        )
    except:
        port = 8000
    host = "0.0.0.0"

    logger.info(f"Starting web server on {host}:{port}")

    uvicorn.run("src.web_interface.app:app", host=host, port=port, reload=True, log_level="info")
