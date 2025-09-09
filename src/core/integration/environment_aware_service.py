"""
Environment-aware service integration base class.

This module provides the base functionality for integrating all T-Bot services
with the sandbox/live environment-aware exchange system, ensuring seamless
operation across different trading environments.
"""

import asyncio
from collections.abc import Callable
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel, Field

from src.core.base.service import BaseService
from src.core.config.environment import ExchangeEnvironment
from src.core.logging import get_logger

logger = get_logger(__name__)


class EnvironmentMode(Enum):
    """Environment operation modes."""
    SANDBOX_ONLY = "sandbox_only"
    LIVE_ONLY = "live_only"
    HYBRID = "hybrid"
    AUTO_SWITCH = "auto_switch"


class EnvironmentContext(BaseModel):
    """Context information for environment-aware operations."""

    exchange_name: str = Field(..., description="Name of the exchange")
    environment: ExchangeEnvironment = Field(..., description="Current environment")
    is_production: bool = Field(default=False, description="Whether in production mode")
    risk_level: str = Field(default="normal", description="Risk tolerance level for this environment")
    max_order_value: Decimal | None = Field(default=None, description="Maximum order value for this environment")
    enable_paper_trading: bool = Field(default=False, description="Whether to enable paper trading mode")
    validation_level: str = Field(default="standard", description="Level of validation (relaxed/standard/strict)")


class EnvironmentAwareServiceInterface(Protocol):
    """Interface for environment-aware services."""

    async def switch_environment(
        self,
        environment: str | ExchangeEnvironment,
        exchange: str | None = None
    ) -> bool:
        """Switch trading environment."""
        ...

    async def validate_environment_operation(
        self,
        operation: str,
        context: EnvironmentContext
    ) -> bool:
        """Validate if operation is safe in current environment."""
        ...

    def get_environment_context(self, exchange: str) -> EnvironmentContext:
        """Get current environment context for an exchange."""
        ...


class EnvironmentAwareServiceMixin:
    """
    Mixin class providing environment-aware functionality to services.
    
    This mixin adds environment awareness capabilities to any service,
    allowing it to operate transparently across sandbox and live environments.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._environment_contexts: dict[str, EnvironmentContext] = {}
        self._environment_mode = EnvironmentMode.HYBRID
        self._environment_switch_callbacks: list[Callable] = []

    def register_environment_switch_callback(self, callback: Callable) -> None:
        """Register callback to be called when environment switches."""
        if callback not in self._environment_switch_callbacks:
            self._environment_switch_callbacks.append(callback)

    def unregister_environment_switch_callback(self, callback: Callable) -> None:
        """Unregister environment switch callback."""
        if callback in self._environment_switch_callbacks:
            self._environment_switch_callbacks.remove(callback)

    async def _notify_environment_switch(
        self,
        old_env: EnvironmentContext,
        new_env: EnvironmentContext
    ) -> None:
        """Notify all registered callbacks of environment switch."""
        for callback in self._environment_switch_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(old_env, new_env)
                else:
                    callback(old_env, new_env)
            except Exception as e:
                logger.warning(f"Environment switch callback failed: {e}")

    def get_environment_context(self, exchange: str) -> EnvironmentContext:
        """Get current environment context for an exchange."""
        if exchange not in self._environment_contexts:
            # Create default context based on config
            config = getattr(self, "config", None)
            if config and hasattr(config, "get_environment_exchange_config"):
                exchange_config = config.get_environment_exchange_config(exchange)

                context = EnvironmentContext(
                    exchange_name=exchange,
                    environment=ExchangeEnvironment(exchange_config.get("environment_mode", "sandbox")),
                    is_production=exchange_config.get("is_production", False),
                    risk_level="strict" if exchange_config.get("is_production") else "relaxed",
                    validation_level="strict" if exchange_config.get("is_production") else "standard"
                )

                self._environment_contexts[exchange] = context
            else:
                # Fallback to sandbox
                context = EnvironmentContext(
                    exchange_name=exchange,
                    environment=ExchangeEnvironment.SANDBOX,
                    is_production=False,
                    risk_level="relaxed",
                    validation_level="standard"
                )
                self._environment_contexts[exchange] = context

        return self._environment_contexts[exchange]

    async def switch_environment(
        self,
        environment: str | ExchangeEnvironment,
        exchange: str | None = None
    ) -> bool:
        """
        Switch trading environment for specific exchange or globally.
        
        Args:
            environment: Target environment
            exchange: Specific exchange name, None for global switch
            
        Returns:
            bool: True if switch successful
        """
        try:
            # Normalize environment
            if isinstance(environment, str):
                environment = ExchangeEnvironment(environment.lower())

            # Get affected exchanges
            exchanges_to_switch = [exchange] if exchange else list(self._environment_contexts.keys())

            # If no contexts exist yet, get default exchanges
            if not exchanges_to_switch and not exchange:
                exchanges_to_switch = ["binance", "coinbase", "okx"]
            elif not exchanges_to_switch and exchange:
                exchanges_to_switch = [exchange]

            switch_results = []

            for exch in exchanges_to_switch:
                old_context = self.get_environment_context(exch)

                # Validate switch is safe
                if not await self._validate_environment_switch(old_context, environment):
                    logger.error(f"Environment switch validation failed for {exch}")
                    switch_results.append(False)
                    continue

                # Create new context
                new_context = EnvironmentContext(
                    exchange_name=exch,
                    environment=environment,
                    is_production=environment in (ExchangeEnvironment.LIVE, ExchangeEnvironment.PRODUCTION),
                    risk_level="strict" if environment in (ExchangeEnvironment.LIVE, ExchangeEnvironment.PRODUCTION) else "relaxed",
                    validation_level="strict" if environment in (ExchangeEnvironment.LIVE, ExchangeEnvironment.PRODUCTION) else "standard",
                    max_order_value=Decimal("1000") if environment == ExchangeEnvironment.SANDBOX else None
                )

                # Store new context
                self._environment_contexts[exch] = new_context

                # Notify callbacks
                await self._notify_environment_switch(old_context, new_context)

                # Update service-specific environment settings
                await self._update_service_environment(new_context)

                switch_results.append(True)
                logger.info(f"Switched {exch} environment to {environment.value}")

            return all(switch_results)

        except Exception as e:
            logger.error(f"Failed to switch environment: {e}")
            return False

    async def _validate_environment_switch(
        self,
        current_context: EnvironmentContext,
        target_environment: ExchangeEnvironment
    ) -> bool:
        """Validate that environment switch is safe."""
        # Check for production safeguards
        if target_environment in (ExchangeEnvironment.LIVE, ExchangeEnvironment.PRODUCTION):
            # Validate credentials are configured
            config = getattr(self, "config", None)
            if config and hasattr(config, "validate_production_credentials"):
                if not config.validate_production_credentials(current_context.exchange_name):
                    logger.error(f"Production credentials not configured for {current_context.exchange_name}")
                    return False

            # Check for pending operations in sandbox
            if hasattr(self, "_has_pending_operations"):
                if await self._has_pending_operations(current_context.exchange_name):
                    logger.warning(f"Pending operations found for {current_context.exchange_name}")
                    return False

        return True

    async def _update_service_environment(self, context: EnvironmentContext) -> None:
        """Update service-specific settings based on new environment context."""
        # Override in subclasses to implement service-specific logic
        pass

    async def validate_environment_operation(
        self,
        operation: str,
        context: EnvironmentContext | None = None,
        exchange: str | None = None
    ) -> bool:
        """
        Validate if an operation is allowed in current environment.
        
        Args:
            operation: Name of the operation to validate
            context: Optional environment context
            exchange: Exchange name if context not provided
            
        Returns:
            bool: True if operation is allowed
        """
        if context is None and exchange:
            context = self.get_environment_context(exchange)
        elif context is None:
            logger.error("Either context or exchange must be provided")
            return False

        # Production environment validations
        if context.is_production:
            # Stricter validation for production
            if operation in ("place_order", "cancel_order", "modify_position"):
                # Check additional production safeguards
                return await self._validate_production_operation(operation, context)

        # Sandbox environment validations
        else:
            # More lenient validation for sandbox
            if operation == "place_order" and context.max_order_value:
                # Check order value limits for sandbox
                return True  # Additional checks would be done at operation level

        return True

    async def _validate_production_operation(
        self,
        operation: str,
        context: EnvironmentContext
    ) -> bool:
        """Validate production operations with additional safety checks."""
        # Override in subclasses for service-specific production validation
        return True

    def get_environment_specific_config(self, exchange: str, config_key: str, default: Any = None) -> Any:
        """Get environment-specific configuration value."""
        context = self.get_environment_context(exchange)

        # Build environment-specific config key
        env_prefix = "live" if context.is_production else "sandbox"
        full_key = f"{env_prefix}_{config_key}"

        # Try to get from config
        config = getattr(self, "config", None)
        if config and hasattr(config, "get"):
            return config.get(full_key, config.get(config_key, default))

        return default

    def is_environment_ready(self, exchange: str) -> bool:
        """Check if environment is ready for operations."""
        context = self.get_environment_context(exchange)

        # Basic readiness checks
        if context.is_production:
            # Production requires valid credentials
            config = getattr(self, "config", None)
            if config and hasattr(config, "validate_production_credentials"):
                return config.validate_production_credentials(exchange)

        # Sandbox is generally always ready
        return True

    async def get_environment_health_status(self) -> dict[str, Any]:
        """Get health status of all environment contexts."""
        health_status = {}

        for exchange, context in self._environment_contexts.items():
            health_status[exchange] = {
                "environment": context.environment.value,
                "is_production": context.is_production,
                "ready": self.is_environment_ready(exchange),
                "risk_level": context.risk_level,
                "validation_level": context.validation_level,
                "last_updated": datetime.now(timezone.utc).isoformat()
            }

        return health_status


class EnvironmentAwareService(BaseService, EnvironmentAwareServiceMixin):
    """
    Base class for services that need environment awareness.
    
    Combines BaseService with EnvironmentAwareServiceMixin to provide
    a complete environment-aware service foundation.
    """

    def __init__(
        self,
        name: str,
        config: dict[str, Any] | None = None,
        correlation_id: str | None = None
    ):
        """Initialize environment-aware service."""
        super().__init__(name=name, config=config, correlation_id=correlation_id)

        # Initialize environment contexts based on config
        self._initialize_environment_contexts()

    def _initialize_environment_contexts(self) -> None:
        """Initialize environment contexts from configuration."""
        if not hasattr(self, "config") or not self.config:
            return

        # Get supported exchanges
        supported_exchanges = ["binance", "coinbase", "okx"]

        for exchange in supported_exchanges:
            try:
                # Get environment context from config
                self.get_environment_context(exchange)
                logger.debug(f"Initialized environment context for {exchange}")
            except Exception as e:
                logger.warning(f"Failed to initialize context for {exchange}: {e}")

    async def _do_start(self) -> None:
        """Start the environment-aware service."""
        await super()._do_start()

        # Validate all environment contexts are ready
        for exchange in self._environment_contexts:
            if not self.is_environment_ready(exchange):
                logger.warning(f"Environment not ready for {exchange}")

    async def _do_stop(self) -> None:
        """Stop the environment-aware service."""
        # Clear environment contexts
        self._environment_contexts.clear()
        self._environment_switch_callbacks.clear()

        await super()._do_stop()

    async def get_service_health(self) -> dict[str, Any]:
        """Get comprehensive service health including environment status."""
        base_health = await super().get_service_health() if hasattr(super(), "get_service_health") else {}

        environment_health = await self.get_environment_health_status()

        return {
            **base_health,
            "environment_contexts": environment_health,
            "environment_mode": self._environment_mode.value,
            "ready_exchanges": [
                exchange for exchange in self._environment_contexts
                if self.is_environment_ready(exchange)
            ]
        }
