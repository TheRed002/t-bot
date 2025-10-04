"""
Main application entry point for the trading bot framework.

This module provides the main application startup, configuration loading,
component registry, graceful shutdown handling, and health check endpoints.

Features:
- Configuration loading with validation
- Database connection initialization
- Component registry setup
- Graceful shutdown handling
- Health check endpoint setup
- Error handling and logging configuration
"""

import asyncio
import signal
import sys
from contextlib import asynccontextmanager
from typing import Any

from src.core.config import Config
from src.core.exceptions import ConfigurationError
from src.core.logging import get_logger, setup_logging


class Application:
    """Main application class for the trading bot framework."""

    def __init__(self):
        self.config: Config | None = None
        self.logger = get_logger(__name__)
        self.components: dict[str, Any] = {}
        self.shutdown_event = asyncio.Event()
        self.health_status = {"status": "starting", "components": {}}

    async def initialize(self) -> None:
        """Initialize the application with configuration and components."""
        try:
            # Setup logging first
            self._setup_logging()

            # Set a correlation ID for this initialization
            from src.core.logging import correlation_context

            correlation_id = correlation_context.generate_correlation_id()
            correlation_context.set_correlation_id(correlation_id)

            self.logger.info("Starting application initialization", correlation_id=correlation_id)

            # Load and validate configuration
            await self._load_configuration()

            # Initialize core components
            await self._initialize_components()

            # Setup signal handlers for graceful shutdown
            self._setup_signal_handlers()

            # Update health status
            self.health_status["status"] = "running"

            self.logger.info(
                "Application initialized successfully",
                app_name=self.config.app_name,
                environment=self.config.environment,
            )

        except Exception as e:
            self.logger.error(
                "Application initialization failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        import os
        from pathlib import Path

        # Setup logging based on environment
        environment = (
            getattr(self.config, "environment", "development") if self.config else "development"
        )

        # Determine log file path - write to file in ALL environments for ELK
        logs_dir = Path(os.getenv("LOGS_DIR", "./logs"))
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = str(logs_dir / f"tbot-{environment}.log")

        setup_logging(environment=environment, log_file=log_file)

        # Enable debug logging only in development mode
        if self.config and self.config.debug and environment == "development":
            from src.core.logging import setup_debug_logging

            setup_debug_logging()
            self.logger.debug("Debug logging enabled for development environment")

        self.logger.info("Logging system initialized", log_file=log_file)

    async def _load_configuration(self) -> None:
        """Load and validate application configuration."""
        try:
            self.config = Config()

            # Validate configuration
            self.config.validate()

            self.logger.info(
                "Configuration loaded successfully",
                environment=self.config.environment,
                debug_mode=self.config.debug,
            )

        except Exception as e:
            raise ConfigurationError(
                f"Failed to load configuration: {e!s}",
                error_code="CONFIG_LOAD_ERROR",
                details={"error": str(e)},
            )

    async def _initialize_components(self) -> None:
        """Initialize core application components."""
        try:
            # Initialize dependency injection container first
            await self._initialize_dependency_injection()

            # Initialize database connections (placeholder for P-002)
            await self._initialize_database()

            # Initialize monitoring and analytics
            await self._initialize_monitoring()
            await self._initialize_analytics()

            # Initialize exchange connections (placeholder for P-003)
            await self._initialize_exchanges()

            # Initialize risk management (placeholder for P-008)
            await self._initialize_risk_management()

            # Initialize strategies (placeholder for P-011)
            await self._initialize_strategies()

            # Initialize ML models (placeholder for P-017)
            await self._initialize_ml_models()

            self.logger.info("All components initialized successfully")

        except Exception as e:
            self.logger.error(
                "Component initialization failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    async def _initialize_database(self) -> None:
        """Initialize database connections and seed data in development."""
        # Skip database in development/mock mode for now
        import os

        if os.getenv("MOCK_MODE") or self.config.environment == "development":
            self.logger.info("Skipping database initialization in development/mock mode")
            self.health_status["components"]["database"] = "mock"
            return

        try:
            # Initialize database connection
            from src.database.connection import init_database

            await init_database(self.config)
            self.logger.info("Database connection initialized")

            # Run database seeding in development mode
            if self.config.environment == "development":
                self.logger.info("Running database seeding for development environment...")
                try:
                    from src.database.seed_data import DatabaseSeeder

                    seeder = DatabaseSeeder(self.config)
                    await seeder.seed_all()
                    self.logger.info("Database seeding completed")
                except Exception as e:
                    # Don't fail if seeding fails, just log the error
                    self.logger.warning(f"Database seeding failed (non-critical): {e}")

            self.health_status["components"]["database"] = "initialized"

        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            self.health_status["components"]["database"] = "error"
            # Continue without database for now in development
            if self.config.environment != "development":
                raise

    async def _initialize_dependency_injection(self) -> None:
        """Initialize dependency injection container."""
        try:
            from src.core.dependency_injection import DependencyInjector

            # Create and configure DI container
            self.components["injector"] = DependencyInjector()

            # Register database dependencies first
            from src.database.di_registration import register_database_services

            register_database_services(self.components["injector"])

            # Register monitoring dependencies
            from src.monitoring.di_registration import register_monitoring_services

            register_monitoring_services(self.components["injector"])

            # Register core service dependencies in dependency order
            try:
                # State management (needed by many services)
                from src.state.di_registration import register_state_services
                register_state_services(self.components["injector"].get_container())
                self.logger.info("State management dependencies registered")
            except Exception as e:
                self.logger.warning(f"Failed to register state dependencies: {e}")

            try:
                # Data services (needed by analytics and trading)
                from src.data.di_registration import register_data_services
                register_data_services(self.components["injector"])
                self.logger.info("Data service dependencies registered")
            except Exception as e:
                self.logger.warning(f"Failed to register data dependencies: {e}")

            try:
                # Risk management (needed by execution and strategies)
                from src.risk_management.di_registration import register_risk_management_services
                register_risk_management_services(self.components["injector"])
                self.logger.info("Risk management dependencies registered")
            except Exception as e:
                self.logger.warning(f"Failed to register risk dependencies: {e}")

            try:
                # Execution services
                from src.execution.di_registration import register_execution_module
                register_execution_module(self.components["injector"], self.config)
                self.logger.info("Execution module dependencies registered")
            except Exception as e:
                self.logger.warning(f"Failed to register execution dependencies: {e}")

            try:
                # Bot management services
                from src.bot_management.di_registration import register_bot_management_services
                register_bot_management_services(self.components["injector"])
                self.logger.info("Bot management dependencies registered")
            except Exception as e:
                self.logger.warning(f"Failed to register bot management dependencies: {e}")

            try:
                # Strategies services
                from src.strategies.di_registration import register_strategies_dependencies
                register_strategies_dependencies(self.components["injector"].get_container())
                self.logger.info("Strategies dependencies registered")
            except Exception as e:
                self.logger.warning(f"Failed to register strategies dependencies: {e}")

            try:
                # Optimization services (needed by web interface and strategies)
                from src.optimization.di_registration import register_optimization_dependencies
                register_optimization_dependencies(self.components["injector"])
                self.logger.info("Optimization dependencies registered")
            except Exception as e:
                self.logger.warning(f"Failed to register optimization dependencies: {e}")

            try:
                # ML services (needed by optimization and strategies)
                from src.ml.di_registration import register_ml_services
                register_ml_services(
                    self.components["injector"].get_container(),
                    self.config.to_dict() if hasattr(self.config, 'to_dict') else {}
                )
                self.logger.info("ML dependencies registered")
            except Exception as e:
                self.logger.warning(f"Failed to register ML dependencies: {e}")

            try:
                # Backtesting services (needed by optimization)
                from src.backtesting.di_registration import register_backtesting_services
                register_backtesting_services(self.components["injector"])
                self.logger.info("Backtesting dependencies registered")
            except Exception as e:
                self.logger.warning(f"Failed to register backtesting dependencies: {e}")

            try:
                # Analytics services (needed by many modules)
                from src.analytics.di_registration import register_analytics_services
                register_analytics_services(self.components["injector"])
                self.logger.info("Analytics dependencies registered")
            except Exception as e:
                self.logger.warning(f"Failed to register analytics dependencies: {e}")

            try:
                # Web interface dependencies (needs most other services)
                from src.web_interface.di_registration import register_web_interface_services
                register_web_interface_services(self.components["injector"])
                self.logger.info("Web interface dependencies registered")
            except Exception as e:
                self.logger.warning(f"Failed to register web interface dependencies: {e}")

            self.health_status["components"]["dependency_injection"] = "initialized"
            self.logger.info("Dependency injection container initialized")

        except Exception as e:
            self.logger.error(f"Dependency injection initialization failed: {e}")
            self.health_status["components"]["dependency_injection"] = "error"
            raise

    async def _initialize_monitoring(self) -> None:
        """Initialize monitoring system."""
        try:

            # Get injector
            injector = self.components.get("injector")
            if not injector:
                raise RuntimeError("Dependency injector not initialized")

            # Register monitoring services and get composite service
            from src.monitoring.di_registration import register_monitoring_services
            register_monitoring_services(injector)
            self.logger.info("About to resolve MonitoringServiceInterface...")
            monitoring_service = injector.resolve("MonitoringServiceInterface")
            self.logger.info("MonitoringServiceInterface resolved, calling start_monitoring()...")
            await monitoring_service.start_monitoring()
            self.logger.info("start_monitoring() completed successfully")

            self.components["monitoring_service"] = monitoring_service
            self.health_status["components"]["monitoring"] = "initialized"
            self.logger.info("Monitoring system initialized")

        except Exception as e:
            self.logger.error(f"Monitoring initialization failed: {e}")
            self.health_status["components"]["monitoring"] = "error"
            # Continue without monitoring in development
            if self.config.environment != "development":
                raise

    async def _initialize_analytics(self) -> None:
        """Initialize analytics system."""
        try:
            from src.analytics import get_analytics_service
            from src.analytics.di_registration import register_analytics_services

            # Get injector
            injector = self.components.get("injector")
            if not injector:
                raise RuntimeError("Dependency injector not initialized")

            # Register analytics services
            register_analytics_services(injector)

            # Create analytics service
            analytics_service = get_analytics_service(injector)
            await analytics_service.start()

            self.components["analytics_service"] = analytics_service
            self.health_status["components"]["analytics"] = "initialized"
            self.logger.info("Analytics system initialized")

        except Exception as e:
            self.logger.error(f"Analytics initialization failed: {e}")
            self.health_status["components"]["analytics"] = "error"
            # Continue without analytics in development
            if self.config.environment != "development":
                raise

    async def _initialize_exchanges(self) -> None:
        """Initialize exchange connections."""
        try:
            from src.exchanges import ExchangeFactory, register_exchanges

            # Create exchange factory
            self.components["exchange_factory"] = ExchangeFactory(self.config)

            # Register all available exchanges
            register_exchanges(self.components["exchange_factory"])

            # Initialize supported exchanges from configuration
            supported_exchanges = self.config.exchange.enabled_exchanges if hasattr(self.config, 'exchange') else []
            for exchange_name in supported_exchanges:
                if self.components["exchange_factory"].is_exchange_supported(exchange_name):
                    try:
                        await self.components["exchange_factory"].create_exchange(exchange_name)
                        self.logger.info(f"Initialized exchange: {exchange_name}")
                    except Exception as e:
                        self.logger.error(f"Failed to initialize exchange {exchange_name}: {e!s}")
                else:
                    self.logger.warning(f"Exchange {exchange_name} not supported")

            self.health_status["components"]["exchanges"] = "initialized"
            self.logger.info("Exchange initialization completed")

        except Exception as e:
            self.logger.error(f"Exchange initialization failed: {e!s}")
            self.health_status["components"]["exchanges"] = "error"
            raise

    async def _initialize_risk_management(self) -> None:
        """Initialize risk management system."""
        try:
            from src.risk_management import RiskManager

            # Get injector
            injector = self.components.get("injector")

            # Create risk manager with proper dependencies
            self.components["risk_manager"] = RiskManager(config=self.config)

            # Validate risk parameters
            await self.components["risk_manager"].validate_risk_parameters()

            self.health_status["components"]["risk_management"] = "initialized"
            self.logger.info("Risk management system initialized successfully")

        except Exception as e:
            self.logger.error(f"Risk management initialization failed: {e!s}")
            self.health_status["components"]["risk_management"] = "error"
            raise

    async def _initialize_strategies(self) -> None:
        """Initialize trading strategies."""
        try:
            from src.strategies import StrategyConfigurationManager, StrategyFactory

            # Create strategy configuration manager
            self.components["strategy_config_manager"] = StrategyConfigurationManager()

            # Create strategy factory
            self.components["strategy_factory"] = StrategyFactory()

            # Note: StrategyFactory dependencies are managed through DI container
            # No manual setter methods needed - the factory uses injected dependencies

            # Load and create strategies from configuration
            available_strategies = self.components[
                "strategy_config_manager"
            ].get_available_strategies()
            for strategy_name in available_strategies:
                try:
                    # Load strategy configuration
                    config = self.components["strategy_config_manager"].load_strategy_config(
                        strategy_name
                    )

                    # Create strategy instance
                    self.components["strategy_factory"].create_strategy(
                        strategy_name, config.model_dump()
                    )

                    self.logger.info(f"Initialized strategy: {strategy_name}")

                except Exception as e:
                    self.logger.error(f"Failed to initialize strategy {strategy_name}: {e!s}")

            self.health_status["components"]["strategies"] = "initialized"
            self.logger.info("Strategy initialization completed successfully")

        except Exception as e:
            self.logger.error(f"Strategy initialization failed: {e!s}")
            self.health_status["components"]["strategies"] = "error"
            raise

    async def _initialize_ml_models(self) -> None:
        """Initialize machine learning models."""
        # Placeholder for P-017 ML implementation
        self.logger.info("ML models initialization placeholder - will be implemented in P-017")
        self.health_status["components"]["ml_models"] = "initialized"

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            self.logger.info(
                f"Received signal {signum}, initiating graceful shutdown",
                signal=signum,
            )
            asyncio.create_task(self.shutdown())

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def shutdown(self) -> None:
        """Gracefully shutdown the application."""
        try:
            self.logger.info("Starting graceful shutdown")
            self.health_status["status"] = "shutting_down"

            # Set shutdown event
            self.shutdown_event.set()

            # Shutdown components in reverse order
            await self._shutdown_ml_models()
            await self._shutdown_strategies()
            await self._shutdown_risk_management()
            await self._shutdown_exchanges()
            await self._shutdown_analytics()
            await self._shutdown_monitoring()
            await self._shutdown_database()
            await self._shutdown_error_handlers()

            self.health_status["status"] = "stopped"
            self.logger.info("Graceful shutdown completed")

        except Exception as e:
            self.logger.error(
                "Error during shutdown",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    async def _shutdown_database(self) -> None:
        """Shutdown database connections."""
        # Placeholder for P-002 database shutdown
        self.logger.info("Database shutdown placeholder - will be implemented in P-002")
        self.health_status["components"]["database"] = "shutdown"

    async def _shutdown_exchanges(self) -> None:
        """Shutdown exchange connections."""
        try:
            if "exchange_factory" in self.components:
                await self.components["exchange_factory"].disconnect_all()
                self.logger.info("Disconnected all exchanges")

            self.health_status["components"]["exchanges"] = "shutdown"

        except Exception as e:
            self.logger.error(f"Exchange shutdown error: {e!s}")
            self.health_status["components"]["exchanges"] = "error"

    async def _shutdown_risk_management(self) -> None:
        """Shutdown risk management system."""
        try:
            if "risk_manager" in self.components:
                # Get final risk summary before shutdown
                risk_summary = await self.components[
                    "risk_manager"
                ].get_comprehensive_risk_summary()
                self.logger.info("Final risk summary", risk_summary=risk_summary)

                # Clear components
                del self.components["risk_manager"]

            self.health_status["components"]["risk_management"] = "shutdown"
            self.logger.info("Risk management system shutdown completed")

        except Exception as e:
            self.logger.error(f"Risk management shutdown failed: {e!s}")
            self.health_status["components"]["risk_management"] = "error"

    async def _shutdown_strategies(self) -> None:
        """Shutdown trading strategies."""
        try:
            if "strategy_factory" in self.components:
                # Strategy factory cleanup handled by DI container
                self.logger.info("Strategy factory cleanup delegated to DI container")
            else:
                self.logger.warning("Strategy factory not found during shutdown")

        except Exception as e:
            self.logger.error(f"Strategy shutdown failed: {e!s}")
            # Continue with shutdown even if strategy shutdown fails
        self.health_status["components"]["strategies"] = "shutdown"

    async def _shutdown_ml_models(self) -> None:
        """Shutdown machine learning models."""
        # Placeholder for P-017 ML shutdown
        self.logger.info("ML models shutdown placeholder - will be implemented in P-017")
        self.health_status["components"]["ml_models"] = "shutdown"

    async def _shutdown_analytics(self) -> None:
        """Shutdown analytics system."""
        try:
            if "analytics_service" in self.components:
                await self.components["analytics_service"].stop()
                self.logger.info("Analytics service stopped")

            self.health_status["components"]["analytics"] = "shutdown"

        except Exception as e:
            self.logger.error(f"Analytics shutdown failed: {e}")
            self.health_status["components"]["analytics"] = "error"

    async def _shutdown_monitoring(self) -> None:
        """Shutdown monitoring system."""
        try:
            if "monitoring_service" in self.components:
                await self.components["monitoring_service"].stop()
                self.logger.info("Monitoring service stopped")

            self.health_status["components"]["monitoring"] = "shutdown"

        except Exception as e:
            self.logger.error(f"Monitoring shutdown failed: {e}")
            self.health_status["components"]["monitoring"] = "error"

    async def _shutdown_error_handlers(self) -> None:
        """Shutdown all error handlers and cleanup resources."""
        try:
            from src.error_handling.decorators import (
                get_active_handler_count,
                shutdown_all_error_handlers,
            )

            active_count = get_active_handler_count()
            if active_count > 0:
                self.logger.info(f"Shutting down {active_count} active error handlers")
                await shutdown_all_error_handlers()
                self.logger.info("All error handlers shut down successfully")

            self.health_status["components"]["error_handlers"] = "shutdown"

        except Exception as e:
            self.logger.error(f"Error handler shutdown failed: {e!s}")
            self.health_status["components"]["error_handlers"] = "error"

    def get_health_status(self) -> dict[str, Any]:
        """Get current application health status."""
        return self.health_status.copy()

    async def _perform_health_checks(self) -> None:
        """Perform periodic health checks on application components."""
        try:
            # Check component health status
            unhealthy_components = []
            for component, status in self.health_status.get("components", {}).items():
                if status == "error":
                    unhealthy_components.append(component)

            if unhealthy_components:
                self.logger.warning(
                    f"Unhealthy components detected: {', '.join(unhealthy_components)}"
                )

            # Update overall status
            if unhealthy_components:
                self.health_status["status"] = "degraded"
            elif self.health_status["status"] == "running":
                self.health_status["status"] = "healthy"

            # Log periodic status
            component_count = len(self.health_status.get("components", {}))
            self.logger.debug(
                f"Health check completed: {component_count} components, status: {self.health_status['status']}"
            )

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")

    async def run(self) -> None:
        """Run the main application loop."""
        try:
            # Initialize application
            await self.initialize()

            # Main application loop - monitor components and handle events
            loop_count = 0
            while not self.shutdown_event.is_set():
                try:
                    # Perform periodic health checks and maintenance
                    if loop_count % 60 == 0:  # Every 60 seconds
                        await self._perform_health_checks()

                    # Yield control and wait for shutdown event
                    await asyncio.sleep(1)
                    loop_count += 1

                except asyncio.CancelledError:
                    self.logger.info("Main loop cancelled, initiating shutdown")
                    break
                except Exception as e:
                    self.logger.error(f"Error in main loop: {e}")
                    await asyncio.sleep(5)  # Prevent tight error loops

        except Exception as e:
            self.logger.error(
                "Application run failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise
        finally:
            await self.shutdown()


@asynccontextmanager
async def application_context():
    """Context manager for application lifecycle."""
    app = Application()
    try:
        await app.initialize()
        yield app
    finally:
        await app.shutdown()


async def main() -> None:
    """Main application entry point."""
    logger = get_logger(__name__)
    try:
        async with application_context() as app:
            await app.run()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
