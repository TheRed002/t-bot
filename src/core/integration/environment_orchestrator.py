"""
Environment-Aware Integration Orchestrator.

This module provides the main orchestrator for integrating all T-Bot services
with the sandbox/live environment-aware exchange system. It coordinates
environment switches across all services and maintains system coherence.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any

from src.core.config.environment import ExchangeEnvironment, TradingEnvironment
from src.core.exceptions import ServiceError, ValidationError
from src.core.integration.environment_aware_service import (
    EnvironmentAwareService,
    EnvironmentContext,
)
from src.core.logging import get_logger

logger = get_logger(__name__)


class EnvironmentIntegrationOrchestrator(EnvironmentAwareService):
    """
    Orchestrates environment-aware integration across all T-Bot services.
    
    This orchestrator ensures that all services work cohesively when switching
    between sandbox and live environments, maintaining data consistency,
    risk controls, and operational integrity.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        correlation_id: str | None = None
    ):
        super().__init__(
            name="EnvironmentIntegrationOrchestrator",
            config=config,
            correlation_id=correlation_id
        )

        # Service registry for environment-aware services
        self._registered_services: dict[str, Any] = {}
        self._service_dependencies: dict[str, set[str]] = {}
        self._environment_switch_locks: dict[str, asyncio.Lock] = {}
        self._global_environment_state: dict[str, Any] = {}

        # Integration status tracking
        self._integration_status: dict[str, dict[str, Any]] = {}
        self._health_monitors: dict[str, Any] = {}

    async def _do_start(self) -> None:
        """Start the integration orchestrator."""
        await super()._do_start()

        # Initialize environment locks for each exchange
        for exchange in ["binance", "coinbase", "okx"]:
            self._environment_switch_locks[exchange] = asyncio.Lock()

        # Initialize global environment state
        self._global_environment_state = {
            "current_global_environment": TradingEnvironment.SANDBOX,
            "exchange_environments": {
                "binance": ExchangeEnvironment.SANDBOX,
                "coinbase": ExchangeEnvironment.SANDBOX,
                "okx": ExchangeEnvironment.SANDBOX,
            },
            "last_environment_switch": None,
            "pending_switches": {},
            "switch_history": [],
        }

        logger.info("Environment Integration Orchestrator started")

    def register_service(
        self,
        service_name: str,
        service_instance: Any,
        dependencies: list[str] | None = None
    ) -> None:
        """
        Register an environment-aware service with the orchestrator.
        
        Args:
            service_name: Unique name for the service
            service_instance: Service instance implementing environment awareness
            dependencies: List of service names this service depends on
        """
        if service_name in self._registered_services:
            logger.warning(f"Service {service_name} already registered, overwriting")

        self._registered_services[service_name] = service_instance
        self._service_dependencies[service_name] = set(dependencies or [])

        # Initialize integration status
        self._integration_status[service_name] = {
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "current_environments": {},
            "last_environment_switch": None,
            "switch_count": 0,
            "health_status": "unknown",
            "ready": False,
        }

        # Register environment switch callback if supported
        if hasattr(service_instance, "register_environment_switch_callback"):
            service_instance.register_environment_switch_callback(
                self._on_service_environment_switch
            )

        logger.info(f"Registered environment-aware service: {service_name}")

    async def switch_global_environment(
        self,
        target_environment: str | TradingEnvironment,
        confirm_production: bool = False
    ) -> dict[str, Any]:
        """
        Switch the global trading environment across all services and exchanges.
        
        Args:
            target_environment: Target global environment
            confirm_production: Required confirmation for production switches
            
        Returns:
            Dict containing switch results and status
        """
        if isinstance(target_environment, str):
            target_environment = TradingEnvironment(target_environment.lower())

        logger.info(f"Initiating global environment switch to {target_environment.value}")

        # Production safety check
        if target_environment == TradingEnvironment.LIVE and not confirm_production:
            raise ValidationError(
                "Production environment switch requires explicit confirmation"
            )

        switch_id = f"global_switch_{datetime.now().timestamp()}"
        switch_results = {
            "switch_id": switch_id,
            "target_environment": target_environment.value,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "success": False,
            "exchanges_switched": [],
            "services_updated": [],
            "errors": [],
            "warnings": [],
        }

        try:
            # Pre-switch validation
            validation_result = await self._validate_global_environment_switch(target_environment)
            if not validation_result["valid"]:
                switch_results["errors"].extend(validation_result["errors"])
                switch_results["warnings"].extend(validation_result["warnings"])
                raise ValidationError(f"Global environment switch validation failed: {validation_result}")

            # Get exchange-specific environments based on global environment
            exchange_environments = self._map_global_to_exchange_environments(target_environment)

            # Switch each exchange environment
            for exchange, exchange_env in exchange_environments.items():
                try:
                    exchange_result = await self.switch_exchange_environment(
                        exchange, exchange_env, part_of_global_switch=True
                    )

                    if exchange_result["success"]:
                        switch_results["exchanges_switched"].append(exchange)
                    else:
                        switch_results["errors"].append(
                            f"Failed to switch {exchange}: {exchange_result.get('error')}"
                        )

                except Exception as e:
                    error_msg = f"Exception switching {exchange}: {e}"
                    switch_results["errors"].append(error_msg)
                    logger.error(error_msg)

            # Update global environment state if any exchanges succeeded
            if switch_results["exchanges_switched"]:
                self._global_environment_state["current_global_environment"] = target_environment
                self._global_environment_state["last_environment_switch"] = switch_results["start_time"]

                # Add to switch history
                self._global_environment_state["switch_history"].append({
                    "switch_id": switch_id,
                    "timestamp": switch_results["start_time"],
                    "target_environment": target_environment.value,
                    "success": len(switch_results["exchanges_switched"]) > 0,
                    "exchanges_affected": switch_results["exchanges_switched"],
                })

            # Determine overall success
            total_exchanges = len(exchange_environments)
            switched_exchanges = len(switch_results["exchanges_switched"])

            if switched_exchanges == total_exchanges:
                switch_results["success"] = True
                logger.info(f"Global environment switch to {target_environment.value} completed successfully")
            elif switched_exchanges > 0:
                switch_results["success"] = False
                switch_results["warnings"].append(
                    f"Partial success: {switched_exchanges}/{total_exchanges} exchanges switched"
                )
                logger.warning(f"Partial global environment switch: {switched_exchanges}/{total_exchanges}")
            else:
                switch_results["success"] = False
                logger.error(f"Global environment switch to {target_environment.value} failed completely")

            switch_results["end_time"] = datetime.now(timezone.utc).isoformat()
            return switch_results

        except Exception as e:
            switch_results["success"] = False
            switch_results["end_time"] = datetime.now(timezone.utc).isoformat()
            switch_results["errors"].append(f"Global switch exception: {e}")
            logger.error(f"Global environment switch failed: {e}")
            raise ServiceError(f"Global environment switch failed: {e}")

    async def switch_exchange_environment(
        self,
        exchange: str,
        target_environment: str | ExchangeEnvironment,
        part_of_global_switch: bool = False
    ) -> dict[str, Any]:
        """
        Switch environment for a specific exchange across all services.
        
        Args:
            exchange: Exchange name
            target_environment: Target exchange environment
            part_of_global_switch: Whether this is part of a global switch
            
        Returns:
            Dict containing switch results and status
        """
        if isinstance(target_environment, str):
            target_environment = ExchangeEnvironment(target_environment.lower())

        exchange = exchange.lower()
        switch_id = f"exchange_switch_{exchange}_{datetime.now().timestamp()}"

        logger.info(f"Switching {exchange} environment to {target_environment.value}")

        # Acquire lock for this exchange to prevent concurrent switches
        async with self._environment_switch_locks.get(exchange, asyncio.Lock()):
            switch_results = {
                "switch_id": switch_id,
                "exchange": exchange,
                "target_environment": target_environment.value,
                "start_time": datetime.now(timezone.utc).isoformat(),
                "success": False,
                "services_updated": [],
                "errors": [],
                "warnings": [],
                "part_of_global_switch": part_of_global_switch,
            }

            try:
                # Pre-switch validation
                validation_result = await self._validate_exchange_environment_switch(
                    exchange, target_environment
                )
                if not validation_result["valid"]:
                    switch_results["errors"].extend(validation_result["errors"])
                    switch_results["warnings"].extend(validation_result["warnings"])

                    if validation_result["errors"]:  # Only fail on errors, not warnings
                        raise ValidationError("Exchange environment switch validation failed")

                # Get ordered list of services based on dependencies
                service_order = self._get_service_switch_order()

                # Switch each service to the new environment
                successful_services = []
                failed_services = []

                for service_name in service_order:
                    try:
                        service_instance = self._registered_services[service_name]

                        # Switch service environment if it supports it
                        if hasattr(service_instance, "switch_environment"):
                            success = await service_instance.switch_environment(
                                target_environment, exchange
                            )

                            if success:
                                successful_services.append(service_name)
                                self._integration_status[service_name]["switch_count"] += 1
                                self._integration_status[service_name]["last_environment_switch"] = switch_results["start_time"]

                                # Update service's environment status
                                if "current_environments" not in self._integration_status[service_name]:
                                    self._integration_status[service_name]["current_environments"] = {}
                                self._integration_status[service_name]["current_environments"][exchange] = target_environment.value
                            else:
                                failed_services.append(service_name)
                                switch_results["errors"].append(f"Service {service_name} failed to switch environment")
                        else:
                            switch_results["warnings"].append(f"Service {service_name} does not support environment switching")

                    except Exception as e:
                        error_msg = f"Exception switching service {service_name}: {e}"
                        failed_services.append(service_name)
                        switch_results["errors"].append(error_msg)
                        logger.error(error_msg)

                switch_results["services_updated"] = successful_services

                # Update global environment state
                if successful_services:
                    self._global_environment_state["exchange_environments"][exchange] = target_environment

                # Determine success
                total_services = len(self._registered_services)
                successful_count = len(successful_services)

                if successful_count == total_services:
                    switch_results["success"] = True
                    logger.info(f"Exchange {exchange} environment switch completed successfully")
                elif successful_count > 0:
                    switch_results["success"] = False  # Partial failure
                    switch_results["warnings"].append(
                        f"Partial success: {successful_count}/{total_services} services switched"
                    )
                    logger.warning(f"Partial exchange environment switch for {exchange}")
                else:
                    switch_results["success"] = False
                    logger.error(f"Exchange {exchange} environment switch failed completely")

                # Post-switch validation and cleanup
                await self._post_switch_validation(exchange, target_environment)

                switch_results["end_time"] = datetime.now(timezone.utc).isoformat()
                return switch_results

            except Exception as e:
                switch_results["success"] = False
                switch_results["end_time"] = datetime.now(timezone.utc).isoformat()
                switch_results["errors"].append(f"Exchange switch exception: {e}")
                logger.error(f"Exchange {exchange} environment switch failed: {e}")
                raise ServiceError(f"Exchange environment switch failed: {e}")

    async def get_integrated_health_status(self) -> dict[str, Any]:
        """
        Get comprehensive health status across all environment-aware services.
        
        Returns:
            Dict containing integrated health status information
        """
        health_status = {
            "orchestrator": {
                "status": "healthy",
                "registered_services": len(self._registered_services),
                "global_environment": self._global_environment_state.get("current_global_environment", "unknown"),
                "last_updated": datetime.now(timezone.utc).isoformat(),
            },
            "services": {},
            "exchanges": {},
            "global_status": "healthy",
            "issues": [],
            "warnings": [],
        }

        # Check each registered service
        for service_name, service_instance in self._registered_services.items():
            try:
                # Get service-specific health if available
                if hasattr(service_instance, "get_service_health"):
                    service_health = await service_instance.get_service_health()
                else:
                    service_health = {"status": "unknown", "reason": "no health check available"}

                # Get environment health if available
                if hasattr(service_instance, "get_environment_health_status"):
                    env_health = await service_instance.get_environment_health_status()
                    service_health["environment_health"] = env_health

                health_status["services"][service_name] = {
                    "health": service_health,
                    "integration_status": self._integration_status.get(service_name, {}),
                }

                # Check for issues
                if isinstance(service_health.get("status"), str):
                    if service_health["status"].lower() in ["unhealthy", "error", "failed"]:
                        health_status["issues"].append(f"Service {service_name} is unhealthy")
                        health_status["global_status"] = "degraded"
                    elif service_health["status"].lower() in ["degraded", "warning"]:
                        health_status["warnings"].append(f"Service {service_name} is degraded")
                        if health_status["global_status"] == "healthy":
                            health_status["global_status"] = "degraded"

            except Exception as e:
                health_status["services"][service_name] = {
                    "health": {"status": "error", "error": str(e)},
                    "integration_status": self._integration_status.get(service_name, {}),
                }
                health_status["issues"].append(f"Failed to get health for service {service_name}: {e}")
                health_status["global_status"] = "degraded"

        # Check exchange-specific health
        for exchange in ["binance", "coinbase", "okx"]:
            exchange_health = {
                "current_environment": self._global_environment_state["exchange_environments"].get(exchange, "unknown"),
                "services_ready": 0,
                "total_services": len(self._registered_services),
                "last_environment_switch": None,
            }

            # Count ready services for this exchange
            for service_name, service_instance in self._registered_services.items():
                try:
                    if hasattr(service_instance, "is_environment_ready"):
                        if service_instance.is_environment_ready(exchange):
                            exchange_health["services_ready"] += 1
                except (AttributeError, TypeError, ValueError) as e:
                    self.logger.debug(f"Service {service_name} doesn't support environment readiness check: {e}")

            # Calculate readiness percentage
            if exchange_health["total_services"] > 0:
                exchange_health["readiness_percentage"] = (
                    exchange_health["services_ready"] / exchange_health["total_services"] * 100
                )
            else:
                exchange_health["readiness_percentage"] = 0

            health_status["exchanges"][exchange] = exchange_health

        # Set overall global status based on issues
        if health_status["issues"]:
            health_status["global_status"] = "unhealthy"
        elif health_status["warnings"]:
            health_status["global_status"] = "degraded"

        return health_status

    async def get_environment_status_summary(self) -> dict[str, Any]:
        """Get summary of current environment configuration and status."""
        return {
            "global_environment": self._global_environment_state.get("current_global_environment"),
            "exchange_environments": self._global_environment_state.get("exchange_environments", {}),
            "registered_services": list(self._registered_services.keys()),
            "last_global_switch": self._global_environment_state.get("last_environment_switch"),
            "switch_history_count": len(self._global_environment_state.get("switch_history", [])),
            "pending_switches": self._global_environment_state.get("pending_switches", {}),
            "integration_status": {
                service_name: {
                    "ready": status.get("ready", False),
                    "current_environments": status.get("current_environments", {}),
                    "switch_count": status.get("switch_count", 0),
                }
                for service_name, status in self._integration_status.items()
            },
            "summary_generated_at": datetime.now(timezone.utc).isoformat(),
        }

    async def validate_environment_consistency(self) -> dict[str, Any]:
        """Validate consistency across all environment-aware services."""
        validation_results = {
            "consistent": True,
            "validation_timestamp": datetime.now(timezone.utc).isoformat(),
            "service_validations": {},
            "exchange_consistency": {},
            "issues_found": [],
            "warnings": [],
        }

        # Validate each service's environment consistency
        for service_name, service_instance in self._registered_services.items():
            try:
                service_validation = {"status": "unknown", "details": {}}

                # Service-specific consistency validation
                if hasattr(service_instance, "validate_environment_consistency"):
                    # Check each exchange
                    for exchange in ["binance", "coinbase", "okx"]:
                        try:
                            exchange_validation = await service_instance.validate_environment_consistency(exchange)
                            service_validation["details"][exchange] = exchange_validation

                            if not exchange_validation.get("is_consistent", True):
                                validation_results["consistent"] = False
                                validation_results["issues_found"].extend(
                                    exchange_validation.get("issues_found", [])
                                )
                        except Exception as e:
                            service_validation["details"][exchange] = {
                                "error": str(e),
                                "is_consistent": False
                            }
                            validation_results["issues_found"].append(
                                f"Consistency check failed for {service_name}/{exchange}: {e}"
                            )
                            validation_results["consistent"] = False

                validation_results["service_validations"][service_name] = service_validation

            except Exception as e:
                validation_results["service_validations"][service_name] = {
                    "status": "error",
                    "error": str(e)
                }
                validation_results["issues_found"].append(f"Service validation failed for {service_name}: {e}")
                validation_results["consistent"] = False

        # Cross-service consistency checks
        await self._validate_cross_service_consistency(validation_results)

        return validation_results

    # Helper methods

    async def _validate_global_environment_switch(
        self, target_environment: TradingEnvironment
    ) -> dict[str, Any]:
        """Validate global environment switch feasibility."""
        validation = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "requirements": [],
        }

        # Check if any services are currently processing critical operations
        for service_name, service_instance in self._registered_services.items():
            if hasattr(service_instance, "has_pending_operations"):
                try:
                    for exchange in ["binance", "coinbase", "okx"]:
                        if await service_instance.has_pending_operations(exchange):
                            validation["warnings"].append(
                                f"Service {service_name} has pending operations on {exchange}"
                            )
                except (AttributeError, TypeError, ValueError) as e:
                    self.logger.debug(f"Service {service_name} doesn't support pending operations check: {e}")

        # Production-specific validations
        if target_environment == TradingEnvironment.LIVE:
            # Check credentials configuration
            if hasattr(self, "config") and self.config:
                for exchange in ["binance", "coinbase", "okx"]:
                    if hasattr(self.config, "validate_production_credentials"):
                        if not self.config.validate_production_credentials(exchange):
                            validation["errors"].append(
                                f"Production credentials not configured for {exchange}"
                            )
                            validation["valid"] = False

        return validation

    async def _validate_exchange_environment_switch(
        self, exchange: str, target_environment: ExchangeEnvironment
    ) -> dict[str, Any]:
        """Validate exchange-specific environment switch."""
        validation = {
            "valid": True,
            "errors": [],
            "warnings": [],
        }

        # Check if exchange is currently healthy
        for service_name, service_instance in self._registered_services.items():
            if hasattr(service_instance, "is_environment_ready"):
                try:
                    if not service_instance.is_environment_ready(exchange):
                        validation["warnings"].append(
                            f"Service {service_name} reports {exchange} environment not ready"
                        )
                except (AttributeError, TypeError, ValueError) as e:
                    self.logger.debug(f"Service {service_name} environment readiness check failed: {e}")

        return validation

    def _map_global_to_exchange_environments(
        self, global_env: TradingEnvironment
    ) -> dict[str, ExchangeEnvironment]:
        """Map global environment to exchange-specific environments."""
        if global_env == TradingEnvironment.LIVE:
            return {
                "binance": ExchangeEnvironment.LIVE,
                "coinbase": ExchangeEnvironment.LIVE,
                "okx": ExchangeEnvironment.LIVE,
            }
        elif global_env == TradingEnvironment.SANDBOX:
            return {
                "binance": ExchangeEnvironment.SANDBOX,
                "coinbase": ExchangeEnvironment.SANDBOX,
                "okx": ExchangeEnvironment.SANDBOX,
            }
        else:
            # Hybrid or mock - default to sandbox
            return {
                "binance": ExchangeEnvironment.SANDBOX,
                "coinbase": ExchangeEnvironment.SANDBOX,
                "okx": ExchangeEnvironment.SANDBOX,
            }

    def _get_service_switch_order(self) -> list[str]:
        """Get ordered list of services for environment switching based on dependencies."""
        # Simple topological sort based on dependencies
        ordered_services = []
        remaining_services = set(self._registered_services.keys())

        while remaining_services:
            # Find services with no unmet dependencies
            ready_services = []
            for service in remaining_services:
                dependencies = self._service_dependencies.get(service, set())
                if dependencies.issubset(set(ordered_services)):
                    ready_services.append(service)

            if not ready_services:
                # Circular dependency or missing dependency - add remaining in arbitrary order
                ready_services = list(remaining_services)

            # Add ready services to order
            for service in ready_services:
                ordered_services.append(service)
                remaining_services.remove(service)

        return ordered_services

    async def _post_switch_validation(
        self, exchange: str, target_environment: ExchangeEnvironment
    ) -> None:
        """Perform post-switch validation and cleanup."""
        # Validate that services actually switched
        for service_name, service_instance in self._registered_services.items():
            try:
                if hasattr(service_instance, "get_environment_context"):
                    context = service_instance.get_environment_context(exchange)
                    if context.environment != target_environment:
                        logger.warning(
                            f"Service {service_name} context shows {context.environment} "
                            f"but expected {target_environment} for {exchange}"
                        )
            except Exception as e:
                logger.warning(f"Post-switch validation failed for {service_name}: {e}")

    async def _validate_cross_service_consistency(self, validation_results: dict[str, Any]) -> None:
        """Validate consistency across services."""
        # Check that all services report the same environment for each exchange
        for exchange in ["binance", "coinbase", "okx"]:
            reported_environments = set()

            for service_name, service_instance in self._registered_services.items():
                try:
                    if hasattr(service_instance, "get_environment_context"):
                        context = service_instance.get_environment_context(exchange)
                        reported_environments.add(context.environment.value)
                except (AttributeError, TypeError, ValueError) as e:
                    self.logger.debug(f"Service {service_name} doesn't support environment context: {e}")

            if len(reported_environments) > 1:
                validation_results["consistent"] = False
                validation_results["issues_found"].append(
                    f"Services report different environments for {exchange}: {reported_environments}"
                )

    async def _on_service_environment_switch(
        self, old_env: EnvironmentContext, new_env: EnvironmentContext
    ) -> None:
        """Handle service environment switch callback."""
        logger.info(
            f"Service environment switch: {old_env.exchange_name} "
            f"{old_env.environment.value} -> {new_env.environment.value}"
        )

        # Update integration status if we can identify the service
        # This would require additional context in a real implementation
