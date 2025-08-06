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
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from src.core.config import Config
from src.core.logging import get_logger, setup_logging, correlation_context
from src.core.exceptions import TradingBotError, ConfigurationError


class Application:
    """Main application class for the trading bot framework."""
    
    def __init__(self):
        self.config: Optional[Config] = None
        self.logger = get_logger(__name__)
        self.components: Dict[str, Any] = {}
        self.shutdown_event = asyncio.Event()
        self.health_status = {"status": "starting", "components": {}}
    
    async def initialize(self) -> None:
        """Initialize the application with configuration and components."""
        try:
            # Setup logging first
            self._setup_logging()
            
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
                app_version=self.config.app_version,
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
        # TODO: Remove in production - Debug logging setup
        if self.config and self.config.debug:
            from src.core.logging import setup_debug_logging
            setup_debug_logging()
        
        # Setup logging based on environment
        environment = getattr(self.config, 'environment', 'development') if self.config else 'development'
        setup_logging(environment=environment)
        
        self.logger.info("Logging system initialized")
    
    async def _load_configuration(self) -> None:
        """Load and validate application configuration."""
        try:
            self.config = Config()
            
            # Generate configuration schema for validation
            self.config.generate_schema()
            
            self.logger.info(
                "Configuration loaded successfully",
                environment=self.config.environment,
                debug_mode=self.config.debug,
            )
            
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load configuration: {str(e)}",
                error_code="CONFIG_LOAD_ERROR",
                details={"error": str(e)}
            )
    
    async def _initialize_components(self) -> None:
        """Initialize core application components."""
        try:
            # Initialize database connections (placeholder for P-002)
            await self._initialize_database()
            
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
        """Initialize database connections."""
        # Placeholder for P-002 database implementation
        self.logger.info("Database initialization placeholder - will be implemented in P-002")
        self.health_status["components"]["database"] = "initialized"
    
    async def _initialize_exchanges(self) -> None:
        """Initialize exchange connections."""
        try:
            from src.exchanges import ExchangeFactory, register_exchanges
            
            # Create exchange factory
            self.components["exchange_factory"] = ExchangeFactory(self.config)
            
            # Register all available exchanges
            register_exchanges(self.components["exchange_factory"])
            
            # Initialize supported exchanges from configuration
            supported_exchanges = self.config.exchanges.supported_exchanges
            for exchange_name in supported_exchanges:
                if self.components["exchange_factory"].is_exchange_supported(exchange_name):
                    try:
                        exchange = await self.components["exchange_factory"].create_exchange(exchange_name)
                        self.logger.info(f"Initialized exchange: {exchange_name}")
                    except Exception as e:
                        self.logger.error(f"Failed to initialize exchange {exchange_name}: {str(e)}")
                else:
                    self.logger.warning(f"Exchange {exchange_name} not supported")
            
            self.health_status["components"]["exchanges"] = "initialized"
            self.logger.info("Exchange initialization completed")
            
        except Exception as e:
            self.logger.error(f"Exchange initialization failed: {str(e)}")
            self.health_status["components"]["exchanges"] = "error"
            raise
    
    async def _initialize_risk_management(self) -> None:
        """Initialize risk management system."""
        try:
            from src.risk_management import RiskManager
            
            # Create risk manager
            self.components["risk_manager"] = RiskManager(self.config)
            
            # Validate risk parameters
            await self.components["risk_manager"].validate_risk_parameters()
            
            self.health_status["components"]["risk_management"] = "initialized"
            self.logger.info("Risk management system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Risk management initialization failed: {str(e)}")
            self.health_status["components"]["risk_management"] = "error"
            raise
    
    async def _initialize_strategies(self) -> None:
        """Initialize trading strategies."""
        # Placeholder for P-011 strategy implementation
        self.logger.info("Strategy initialization placeholder - will be implemented in P-011")
        self.health_status["components"]["strategies"] = "initialized"
    
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
            await self._shutdown_database()
            
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
            self.logger.error(f"Exchange shutdown error: {str(e)}")
            self.health_status["components"]["exchanges"] = "error"
    
    async def _shutdown_risk_management(self) -> None:
        """Shutdown risk management system."""
        try:
            if "risk_manager" in self.components:
                # Get final risk summary before shutdown
                risk_summary = await self.components["risk_manager"].get_comprehensive_risk_summary()
                self.logger.info("Final risk summary", risk_summary=risk_summary)
                
                # Clear components
                del self.components["risk_manager"]
            
            self.health_status["components"]["risk_management"] = "shutdown"
            self.logger.info("Risk management system shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Risk management shutdown failed: {str(e)}")
            self.health_status["components"]["risk_management"] = "error"
    
    async def _shutdown_strategies(self) -> None:
        """Shutdown trading strategies."""
        # Placeholder for P-011 strategy shutdown
        self.logger.info("Strategy shutdown placeholder - will be implemented in P-011")
        self.health_status["components"]["strategies"] = "shutdown"
    
    async def _shutdown_ml_models(self) -> None:
        """Shutdown machine learning models."""
        # Placeholder for P-017 ML shutdown
        self.logger.info("ML models shutdown placeholder - will be implemented in P-017")
        self.health_status["components"]["ml_models"] = "shutdown"
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current application health status."""
        return self.health_status.copy()
    
    async def run(self) -> None:
        """Run the main application loop."""
        try:
            # Initialize application
            await self.initialize()
            
            # Main application loop
            while not self.shutdown_event.is_set():
                # TODO: Remove in production - Main loop placeholder
                self.logger.debug("Main application loop iteration")
                await asyncio.sleep(1)
            
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
    try:
        async with application_context() as app:
            await app.run()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"Application failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 