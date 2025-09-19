"""
Bot Lifecycle Service Implementation.

This service handles bot lifecycle management operations following the service layer pattern.
It manages the creation, deployment, and termination of bots.
"""

from decimal import Decimal
from typing import Any

from src.core.base.service import BaseService
from src.core.exceptions import ServiceError, ValidationError
from src.core.logging import get_logger
from src.core.types import BotConfiguration, BotPriority

from .interfaces import IBotLifecycleService


class BotLifecycleService(BaseService, IBotLifecycleService):
    """
    Service for managing bot lifecycles.

    This service handles the business logic for bot lifecycle operations,
    including creation from templates, deployment strategies, and termination.
    """

    def __init__(self, name: str = "BotLifecycleService", config: dict[str, Any] = None):
        """Initialize bot lifecycle service."""
        super().__init__(name=name, config=config)
        self._logger = get_logger(__name__)
        self._bot_templates = {}
        self._deployment_strategies = {}
        self._lifecycle_events = {}
        self._initialize_bot_templates()
        self._initialize_deployment_strategies()

    def _initialize_bot_templates(self) -> None:
        """Initialize default bot templates."""
        self._bot_templates = {
            "basic_trading": {
                "name": "Basic Trading Bot",
                "description": "Simple trading bot with basic strategy",
                "default_strategy": "trend_following",
                "default_priority": BotPriority.NORMAL,
                "default_config": {
                    "trading_enabled": True,
                    "max_position_size": "1000.00",
                    "stop_loss_percentage": "2.0",
                    "take_profit_percentage": "5.0",
                },
            },
            "market_maker": {
                "name": "Market Making Bot",
                "description": "Market making bot with spread optimization",
                "default_strategy": "market_making",
                "default_priority": BotPriority.HIGH,
                "default_config": {
                    "trading_enabled": True,
                    "spread_percentage": "0.1",
                    "inventory_target": "0.5",
                    "max_inventory_deviation": "0.2",
                },
            },
            "arbitrage": {
                "name": "Arbitrage Bot",
                "description": "Cross-exchange arbitrage bot",
                "default_strategy": "arbitrage",
                "default_priority": BotPriority.HIGH,
                "default_config": {
                    "trading_enabled": True,
                    "min_profit_threshold": "0.5",
                    "max_position_size": "5000.00",
                },
            },
        }

    def _initialize_deployment_strategies(self) -> None:
        """Initialize deployment strategies."""
        self._deployment_strategies = {
            "immediate": self._deploy_immediate,
            "staged": self._deploy_staged,
            "blue_green": self._deploy_blue_green,
            "canary": self._deploy_canary,
            "rolling": self._deploy_rolling,
        }

    async def create_bot_from_template(
        self,
        template_name: str,
        bot_name: str,
        exchange: str,
        strategy: str,
        capital_amount: Decimal,
        deployment_strategy: str = "immediate",
        priority: BotPriority = BotPriority.NORMAL,
        custom_config: dict[str, Any] = None,
    ) -> BotConfiguration:
        """
        Create bot configuration from template.

        Args:
            template_name: Name of the template to use
            bot_name: Name for the new bot
            exchange: Exchange to trade on
            strategy: Trading strategy to use
            capital_amount: Amount of capital to allocate
            deployment_strategy: Deployment strategy to use
            priority: Bot priority level
            custom_config: Custom configuration overrides

        Returns:
            Bot configuration
        """
        try:
            # Validate inputs
            if template_name not in self._bot_templates:
                raise ValidationError(f"Unknown template: {template_name}")

            if not bot_name:
                raise ValidationError("Bot name is required")

            if not exchange:
                raise ValidationError("Exchange is required")

            if capital_amount <= 0:
                raise ValidationError("Capital amount must be positive")

            # Get template
            template = self._bot_templates[template_name]

            # Create bot configuration
            bot_config = BotConfiguration(
                id=f"bot_{bot_name}_{template_name}",
                name=bot_name,
                exchange=exchange,
                strategy=strategy or template["default_strategy"],
                priority=priority,
                capital_amount=capital_amount,
                config=self._merge_configs(template["default_config"], custom_config or {}),
                enabled=True,
            )

            # Record lifecycle event
            await self._record_lifecycle_event(
                bot_config.id,
                "created",
                {
                    "template": template_name,
                    "deployment_strategy": deployment_strategy,
                    "capital_amount": str(capital_amount),
                }
            )

            self._logger.info(f"Created bot configuration from template {template_name}: {bot_config.id}")
            return bot_config

        except Exception as e:
            self._logger.error(f"Failed to create bot from template: {e}")
            raise ServiceError(f"Failed to create bot from template: {e}") from e

    async def deploy_bot(
        self, bot_config: BotConfiguration, strategy: str = "immediate"
    ) -> bool:
        """
        Deploy a bot using specified strategy.

        Args:
            bot_config: Bot configuration
            strategy: Deployment strategy

        Returns:
            True if deployed successfully
        """
        try:
            if strategy not in self._deployment_strategies:
                raise ValidationError(f"Unknown deployment strategy: {strategy}")

            # Record deployment start
            await self._record_lifecycle_event(
                bot_config.id,
                "deployment_started",
                {"strategy": strategy}
            )

            # Execute deployment strategy
            deployment_func = self._deployment_strategies[strategy]
            success = await deployment_func(bot_config)

            # Record deployment result
            await self._record_lifecycle_event(
                bot_config.id,
                "deployment_completed" if success else "deployment_failed",
                {"strategy": strategy, "success": success}
            )

            if success:
                self._logger.info(f"Successfully deployed bot {bot_config.id} using {strategy} strategy")
            else:
                self._logger.error(f"Failed to deploy bot {bot_config.id} using {strategy} strategy")

            return success

        except Exception as e:
            self._logger.error(f"Failed to deploy bot {bot_config.id}: {e}")
            await self._record_lifecycle_event(
                bot_config.id,
                "deployment_error",
                {"error": str(e)}
            )
            return False

    async def terminate_bot(self, bot_id: str, reason: str = "user_request") -> bool:
        """
        Terminate a bot.

        Args:
            bot_id: Bot ID to terminate
            reason: Reason for termination

        Returns:
            True if terminated successfully
        """
        try:
            # Record termination start
            await self._record_lifecycle_event(
                bot_id,
                "termination_started",
                {"reason": reason}
            )

            # Perform graceful termination
            success = await self._graceful_termination(bot_id)

            # Record termination result
            await self._record_lifecycle_event(
                bot_id,
                "termination_completed" if success else "termination_failed",
                {"reason": reason, "success": success}
            )

            if success:
                self._logger.info(f"Successfully terminated bot {bot_id}, reason: {reason}")
            else:
                self._logger.error(f"Failed to terminate bot {bot_id}, reason: {reason}")

            return success

        except Exception as e:
            self._logger.error(f"Failed to terminate bot {bot_id}: {e}")
            await self._record_lifecycle_event(
                bot_id,
                "termination_error",
                {"error": str(e)}
            )
            return False

    async def restart_bot(self, bot_id: str, reason: str = "restart_request") -> bool:
        """
        Restart a bot.

        Args:
            bot_id: Bot ID to restart
            reason: Reason for restart

        Returns:
            True if restarted successfully
        """
        try:
            # Record restart start
            await self._record_lifecycle_event(
                bot_id,
                "restart_started",
                {"reason": reason}
            )

            # Stop bot first
            stop_success = await self.terminate_bot(bot_id, f"restart_{reason}")
            if not stop_success:
                return False

            # Wait a bit before restart
            import asyncio
            await asyncio.sleep(1)

            # Deploy bot again
            # Note: This would need bot configuration retrieval
            # For now, just record the attempt
            await self._record_lifecycle_event(
                bot_id,
                "restart_completed",
                {"reason": reason, "success": True}
            )

            self._logger.info(f"Successfully restarted bot {bot_id}, reason: {reason}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to restart bot {bot_id}: {e}")
            await self._record_lifecycle_event(
                bot_id,
                "restart_error",
                {"error": str(e)}
            )
            return False

    async def get_lifecycle_status(self, bot_id: str) -> dict[str, Any]:
        """
        Get lifecycle status for a bot.

        Args:
            bot_id: Bot ID

        Returns:
            Lifecycle status information
        """
        try:
            events = self._lifecycle_events.get(bot_id, [])

            # Get latest events by type
            latest_events = {}
            for event in reversed(events):
                event_type = event["event_type"]
                if event_type not in latest_events:
                    latest_events[event_type] = event

            return {
                "bot_id": bot_id,
                "events": events,
                "latest_events": latest_events,
                "status": self._determine_lifecycle_status(latest_events),
            }

        except Exception as e:
            self._logger.error(f"Failed to get lifecycle status for {bot_id}: {e}")
            raise ServiceError(f"Failed to get lifecycle status: {e}") from e

    async def rollback_deployment(self, bot_id: str, target_version: str) -> bool:
        """
        Rollback bot deployment to previous version.

        Args:
            bot_id: Bot ID
            target_version: Target version to rollback to

        Returns:
            True if rollback successful
        """
        try:
            # Record rollback start
            await self._record_lifecycle_event(
                bot_id,
                "rollback_started",
                {"target_version": target_version}
            )

            # Perform rollback (placeholder implementation)
            # In real implementation, this would restore previous configuration
            success = True

            # Record rollback result
            await self._record_lifecycle_event(
                bot_id,
                "rollback_completed" if success else "rollback_failed",
                {"target_version": target_version, "success": success}
            )

            if success:
                self._logger.info(f"Successfully rolled back bot {bot_id} to version {target_version}")
            else:
                self._logger.error(f"Failed to rollback bot {bot_id} to version {target_version}")

            return success

        except Exception as e:
            self._logger.error(f"Failed to rollback bot {bot_id}: {e}")
            await self._record_lifecycle_event(
                bot_id,
                "rollback_error",
                {"error": str(e)}
            )
            return False

    async def _record_lifecycle_event(self, bot_id: str, event_type: str, data: dict[str, Any]) -> None:
        """Record a lifecycle event."""
        from datetime import datetime, timezone

        event = {
            "bot_id": bot_id,
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc),
            "data": data,
        }

        if bot_id not in self._lifecycle_events:
            self._lifecycle_events[bot_id] = []

        self._lifecycle_events[bot_id].append(event)

        # Keep only recent events (last 100 per bot)
        if len(self._lifecycle_events[bot_id]) > 100:
            self._lifecycle_events[bot_id] = self._lifecycle_events[bot_id][-100:]

    def _merge_configs(self, default_config: dict[str, Any], custom_config: dict[str, Any]) -> dict[str, Any]:
        """Merge default and custom configurations."""
        merged = default_config.copy()
        merged.update(custom_config)
        return merged

    def _determine_lifecycle_status(self, latest_events: dict[str, Any]) -> str:
        """Determine current lifecycle status from events."""
        if "deployment_completed" in latest_events:
            return "deployed"
        elif "deployment_started" in latest_events:
            return "deploying"
        elif "termination_completed" in latest_events:
            return "terminated"
        elif "termination_started" in latest_events:
            return "terminating"
        elif "created" in latest_events:
            return "created"
        else:
            return "unknown"

    async def _deploy_immediate(self, bot_config: BotConfiguration) -> bool:
        """Immediate deployment strategy."""
        # Placeholder implementation
        return True

    async def _deploy_staged(self, bot_config: BotConfiguration) -> bool:
        """Staged deployment strategy."""
        # Placeholder implementation
        return True

    async def _deploy_blue_green(self, bot_config: BotConfiguration) -> bool:
        """Blue-green deployment strategy."""
        # Placeholder implementation
        return True

    async def _deploy_canary(self, bot_config: BotConfiguration) -> bool:
        """Canary deployment strategy."""
        # Placeholder implementation
        return True

    async def _deploy_rolling(self, bot_config: BotConfiguration) -> bool:
        """Rolling deployment strategy."""
        # Placeholder implementation
        return True

    async def _graceful_termination(self, bot_id: str) -> bool:
        """Perform graceful termination of a bot."""
        # Placeholder implementation
        return True