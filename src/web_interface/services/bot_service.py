"""
Bot service for web interface business logic.

This service handles all bot management-related business logic that was previously
embedded in controllers, ensuring proper separation of concerns.
"""

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from src.core.base import BaseComponent
from src.core.exceptions import ServiceError, ValidationError
from src.core.types import BotConfiguration, BotPriority
from src.utils.validation import validate_symbol
from src.utils.web_interface_utils import safe_format_currency, safe_format_percentage
from src.web_interface.interfaces import WebBotServiceInterface


class WebBotService(BaseComponent):
    """Service handling bot management business logic for web interface."""

    def __init__(self, bot_facade=None):
        super().__init__()
        self.bot_controller = bot_facade  # bot_facade is actually BotManagementController

    async def initialize(self) -> None:
        """Initialize the service."""
        self.logger.info("Web bot service initialized")

    async def cleanup(self) -> None:
        """Cleanup the service."""
        self.logger.info("Web bot service cleaned up")

    async def validate_bot_configuration(self, config_data: dict[str, Any]) -> dict[str, Any]:
        """Validate bot configuration with web-specific business logic."""
        try:
            validation_errors = []

            # Validate required fields
            required_fields = [
                "bot_name",
                "bot_type",
                "strategy_name",
                "exchanges",
                "symbols",
                "allocated_capital",
                "risk_percentage",
            ]
            for field in required_fields:
                if field not in config_data or config_data[field] is None:
                    validation_errors.append(f"Field '{field}' is required")

            # Business logic: validate symbols
            symbols = config_data.get("symbols", [])
            for symbol in symbols:
                try:
                    validate_symbol(symbol)
                except ValidationError as e:
                    validation_errors.append(f"Invalid symbol '{symbol}': {e}")

            # Business logic: validate capital allocation
            allocated_capital = config_data.get("allocated_capital", 0)
            if isinstance(allocated_capital, (int, float, Decimal)):
                if Decimal(str(allocated_capital)) <= 0:
                    validation_errors.append("Allocated capital must be greater than 0")
                elif Decimal(str(allocated_capital)) < Decimal("100"):
                    validation_errors.append("Minimum allocated capital is $100")

            # Business logic: validate risk percentage
            risk_percentage = config_data.get("risk_percentage", 0)
            if isinstance(risk_percentage, (int, float, Decimal)):
                risk_decimal = Decimal(str(risk_percentage))
                if risk_decimal <= 0 or risk_decimal > 1:
                    validation_errors.append("Risk percentage must be between 0 and 1")

            # Business logic: validate exchanges
            supported_exchanges = ["binance", "coinbase", "okx"]
            exchanges = config_data.get("exchanges", [])
            for exchange in exchanges:
                if exchange.lower() not in supported_exchanges:
                    validation_errors.append(f"Unsupported exchange: {exchange}")

            return {
                "valid": len(validation_errors) == 0,
                "errors": validation_errors,
                "validated_data": config_data if len(validation_errors) == 0 else None,
            }

        except Exception as e:
            self.logger.error(f"Error validating bot configuration: {e}")
            raise ServiceError(f"Failed to validate bot configuration: {e}")

    async def format_bot_response(self, bot_data: dict[str, Any]) -> dict[str, Any]:
        """Format bot data for web response."""
        try:
            # Business logic: format financial values
            allocated_capital = bot_data.get("allocated_capital", Decimal("0"))
            risk_percentage = bot_data.get("risk_percentage", Decimal("0"))

            formatted_capital = safe_format_currency(allocated_capital)
            formatted_risk = safe_format_percentage(risk_percentage)

            return {
                "success": True,
                "message": "Bot operation completed successfully",
                "bot_id": bot_data.get("bot_id"),
                "bot_name": bot_data.get("bot_name"),
                "status": bot_data.get("status", "unknown"),
                "allocated_capital": formatted_capital,
                "risk_percentage": formatted_risk,
                "auto_started": bot_data.get("auto_start", False),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Error formatting bot response: {e}")
            raise ServiceError(f"Failed to format bot response: {e}")

    async def get_formatted_bot_list(self, filters: dict[str, Any] = None) -> list[dict[str, Any]]:
        """Get bot list with web-specific formatting and filtering."""
        try:
            filters = filters or {}

            if self.bot_controller:
                bots_response = await self.bot_controller.list_bots()
                bots = bots_response.get("bots", [])
            else:
                # Mock data for development
                bots = [
                    {
                        "bot_id": "bot_001",
                        "bot_name": "Momentum Trader",
                        "status": "running",
                        "allocated_capital": 5000.00,
                        "metrics": {
                            "total_pnl": 123.45,
                            "total_trades": 25,
                            "win_rate": 0.72,
                            "last_trade_time": datetime.now(timezone.utc),
                        },
                        "uptime": "2 days, 3 hours",
                    },
                    {
                        "bot_id": "bot_002",
                        "bot_name": "Arbitrage Hunter",
                        "status": "stopped",
                        "allocated_capital": 10000.00,
                        "metrics": {
                            "total_pnl": -45.67,
                            "total_trades": 15,
                            "win_rate": 0.60,
                            "last_trade_time": datetime.now(timezone.utc),
                        },
                        "uptime": "0 seconds",
                    },
                ]

            # Business logic: apply filters and format data
            formatted_bots = []
            status_counts = {"running": 0, "stopped": 0, "error": 0}

            for bot_data in bots:
                bot_status = bot_data.get("status", "unknown").lower()

                # Apply status filter if specified
                status_filter = filters.get("status_filter")
                if status_filter and bot_status != status_filter.lower():
                    continue

                # Count by status
                if bot_status in status_counts:
                    status_counts[bot_status] += 1

                # Business logic: format bot summary
                formatted_bot = {
                    "bot_id": bot_data.get("bot_id", ""),
                    "bot_name": bot_data.get("bot_name", ""),
                    "status": bot_data.get("status", "unknown"),
                    "allocated_capital": Decimal(str(bot_data.get("allocated_capital", 0))),
                    "current_pnl": bot_data.get("metrics", {}).get("total_pnl"),
                    "total_trades": bot_data.get("metrics", {}).get("total_trades"),
                    "win_rate": bot_data.get("metrics", {}).get("win_rate"),
                    "last_trade": bot_data.get("metrics", {}).get("last_trade_time"),
                    "uptime": bot_data.get("uptime"),
                }
                formatted_bots.append(formatted_bot)

            return {
                "bots": formatted_bots,
                "total": len(formatted_bots),
                "status_counts": status_counts,
            }

        except Exception as e:
            self.logger.error(f"Error getting formatted bot list: {e}")
            raise ServiceError(f"Failed to get formatted bot list: {e}")

    async def calculate_bot_metrics(self, bot_id: str) -> dict[str, Any]:
        """Calculate bot metrics with web-specific business logic."""
        try:
            if self.bot_controller:
                bot_status_response = await self.bot_controller.get_bot_status(bot_id)
                bot_status = bot_status_response.get("bot", {})
            else:
                # Mock data for development
                bot_status = {
                    "bot_id": bot_id,
                    "status": "running",
                    "metrics": {
                        "total_trades": 150,
                        "winning_trades": 108,
                        "losing_trades": 42,
                        "total_pnl": 1234.56,
                        "unrealized_pnl": 89.12,
                        "realized_pnl": 1145.44,
                        "win_rate": 0.72,
                        "profit_factor": 2.1,
                        "sharpe_ratio": 1.35,
                        "max_drawdown": -345.67,
                    },
                    "uptime_seconds": 86400 * 7,  # 7 days
                    "last_trade_time": datetime.now(timezone.utc),
                }

            # Business logic: calculate additional metrics
            metrics = bot_status.get("metrics", {})
            total_trades = metrics.get("total_trades", 0)
            winning_trades = metrics.get("winning_trades", 0)
            losing_trades = metrics.get("losing_trades", 0)

            # Calculate derived metrics
            win_rate = (winning_trades / total_trades) if total_trades > 0 else 0.0
            loss_rate = (losing_trades / total_trades) if total_trades > 0 else 0.0

            # Format uptime
            uptime_seconds = bot_status.get("uptime_seconds", 0)
            days = uptime_seconds // 86400
            hours = (uptime_seconds % 86400) // 3600
            minutes = (uptime_seconds % 3600) // 60

            if days > 0:
                uptime_str = f"{days} days, {hours} hours"
            elif hours > 0:
                uptime_str = f"{hours} hours, {minutes} minutes"
            else:
                uptime_str = f"{minutes} minutes"

            return {
                "bot_id": bot_id,
                "performance": {
                    "total_pnl": metrics.get("total_pnl", Decimal("0")),
                    "realized_pnl": metrics.get("realized_pnl", Decimal("0")),
                    "unrealized_pnl": metrics.get("unrealized_pnl", Decimal("0")),
                    "total_trades": total_trades,
                    "winning_trades": winning_trades,
                    "losing_trades": losing_trades,
                    "win_rate": win_rate,
                    "loss_rate": loss_rate,
                    "profit_factor": metrics.get("profit_factor", 0.0),
                    "sharpe_ratio": metrics.get("sharpe_ratio", 0.0),
                    "max_drawdown": metrics.get("max_drawdown", Decimal("0")),
                },
                "operational": {
                    "status": bot_status.get("status", "unknown"),
                    "uptime": uptime_str,
                    "uptime_seconds": uptime_seconds,
                    "last_trade_time": bot_status.get("last_trade_time"),
                    "health_score": self._calculate_health_score(metrics),
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Error calculating bot metrics for {bot_id}: {e}")
            raise ServiceError(f"Failed to calculate bot metrics: {e}")

    async def validate_bot_operation(self, bot_id: str, operation: str) -> dict[str, Any]:
        """Validate bot operation with web-specific checks."""
        try:
            validation_errors = []

            # Business logic: validate operation type
            allowed_operations = ["start", "stop", "pause", "resume", "restart", "delete"]
            if operation not in allowed_operations:
                validation_errors.append(f"Invalid operation: {operation}")

            # Business logic: check bot existence (if controller available)
            if self.bot_controller:
                try:
                    bot_status_response = await self.bot_controller.get_bot_status(bot_id)
                    bot_status = bot_status_response.get("bot", {})
                    current_status = bot_status.get("status", "unknown")

                    # Business logic: validate operation against current status
                    if operation == "start" and current_status == "running":
                        validation_errors.append("Bot is already running")
                    elif operation == "stop" and current_status == "stopped":
                        validation_errors.append("Bot is already stopped")
                    elif operation == "delete" and current_status == "running":
                        validation_errors.append("Cannot delete running bot. Stop it first.")

                except Exception as e:
                    # If bot doesn't exist or other error, continue with validation
                    self.logger.debug(f"Error validating bot {bot_id}: {e}")
                    validation_errors.append(f"Bot with ID {bot_id} not found")

            return {
                "valid": len(validation_errors) == 0,
                "errors": validation_errors,
                "operation": operation,
                "bot_id": bot_id,
            }

        except Exception as e:
            self.logger.error(f"Error validating bot operation {operation} for {bot_id}: {e}")
            raise ServiceError(f"Failed to validate bot operation: {e}")

    async def create_bot_configuration(
        self, request_data: dict[str, Any], user_id: str
    ) -> BotConfiguration:
        """Create bot configuration object with business logic validation."""
        try:
            # Business logic: generate unique bot ID
            bot_id = f"bot_{uuid.uuid4().hex[:8]}"

            # Business logic: calculate max position size based on capital
            allocated_capital = Decimal(str(request_data["allocated_capital"]))
            max_position_size = allocated_capital * Decimal("0.1")  # 10% max per position

            # Create configuration with proper field mapping
            bot_config = BotConfiguration(
                bot_id=bot_id,
                name=request_data["bot_name"],
                version="1.0.0",
                bot_type=request_data["bot_type"],
                strategy_name=request_data["strategy_name"],
                exchanges=request_data["exchanges"],
                symbols=request_data["symbols"],
                allocated_capital=allocated_capital,
                max_position_size=max_position_size,
                risk_percentage=float(request_data["risk_percentage"]),
                priority=request_data.get("priority", BotPriority.NORMAL),
                auto_start=request_data.get("auto_start", False),
                strategy_config=request_data.get("configuration", {}),
                metadata={"created_by": user_id},
                created_at=datetime.now(timezone.utc),
            )

            return bot_config

        except Exception as e:
            self.logger.error(f"Error creating bot configuration: {e}")
            raise ServiceError(f"Failed to create bot configuration: {e}")

    async def update_bot_configuration(
        self, bot_id: str, update_data: dict[str, Any], user_id: str
    ) -> dict[str, Any]:
        """Update bot configuration with business logic (moved from controller)."""
        try:
            # Get current bot status to access configuration
            current_status = await self.get_bot_status_through_service(bot_id)
            current_config = current_status.get("state", {}).get("configuration", {})

            # Business logic: Apply updates with validation
            updated_fields = {}

            if "bot_name" in update_data and update_data["bot_name"] is not None:
                # Business logic: validate bot name
                if len(update_data["bot_name"]) < 3:
                    raise ValidationError("Bot name must be at least 3 characters")
                current_config["bot_name"] = update_data["bot_name"]
                updated_fields["bot_name"] = update_data["bot_name"]

            if "allocated_capital" in update_data and update_data["allocated_capital"] is not None:
                # Business logic: validate capital allocation
                new_capital = Decimal(str(update_data["allocated_capital"]))
                if new_capital <= 0:
                    raise ValidationError("Allocated capital must be positive")

                # Business logic: check if capital is available (mock validation)
                # In real implementation, check against available capital
                current_config["allocated_capital"] = new_capital
                updated_fields["allocated_capital"] = new_capital

            if "risk_percentage" in update_data and update_data["risk_percentage"] is not None:
                # Business logic: validate risk percentage
                risk_pct = Decimal(str(update_data["risk_percentage"]))
                if risk_pct <= 0 or risk_pct > 1:
                    raise ValidationError("Risk percentage must be between 0 and 1")

                current_config["risk_percentage"] = risk_pct
                updated_fields["risk_percentage"] = risk_pct

            if "priority" in update_data and update_data["priority"] is not None:
                # Business logic: validate priority
                priority = update_data["priority"]
                if hasattr(priority, "value"):
                    priority_value = priority.value
                else:
                    priority_value = str(priority)

                current_config["priority"] = priority_value
                updated_fields["priority"] = priority_value

            if "configuration" in update_data and update_data["configuration"] is not None:
                # Business logic: merge configurations
                if "configuration" not in current_config:
                    current_config["configuration"] = {}
                current_config["configuration"].update(update_data["configuration"])
                updated_fields["configuration"] = update_data["configuration"]

            # Business logic: validate final configuration
            if "allocated_capital" in updated_fields and "risk_percentage" in updated_fields:
                # Ensure risk is reasonable for capital amount
                max_risk_amount = updated_fields["allocated_capital"] * updated_fields["risk_percentage"]
                if max_risk_amount > updated_fields["allocated_capital"] * Decimal("0.1"):
                    self.logger.warning(f"High risk configuration for bot {bot_id}")

            # TODO: Apply the actual configuration update
            # This would involve calling the bot facade to update the configuration
            # For now, we just return the updated fields

            self.logger.info(f"Bot configuration updated: {bot_id}", updated_fields=updated_fields, updated_by=user_id)

            return {
                "success": True,
                "message": "Bot updated successfully",
                "bot_id": bot_id,
                "updated_fields": updated_fields,
            }

        except ValidationError:
            raise
        except ServiceError:
            raise
        except Exception as e:
            self.logger.error(f"Error updating bot configuration {bot_id}: {e}")
            raise ServiceError(f"Failed to update bot configuration: {e}")

    def _calculate_health_score(self, metrics: dict[str, Any]) -> float:
        """Calculate bot health score based on metrics (business logic)."""
        try:
            score = 100.0  # Start with perfect score

            # Deduct points for poor performance
            win_rate = metrics.get("win_rate", 0.0)
            if win_rate < 0.5:
                score -= (0.5 - win_rate) * 100  # Deduct up to 50 points

            # Deduct points for negative PnL
            total_pnl = metrics.get("total_pnl", 0)
            if isinstance(total_pnl, (int, float, Decimal)) and Decimal(str(total_pnl)) < 0:
                score -= min(abs(Decimal(str(total_pnl))) / 100, 30)  # Deduct up to 30 points

            # Deduct points for high drawdown
            max_drawdown = metrics.get("max_drawdown", 0)
            if isinstance(max_drawdown, (int, float, Decimal)) and float(max_drawdown) < 0:
                drawdown_percent = abs(float(max_drawdown)) / 1000  # Assume $1000 base
                score -= min(drawdown_percent * 50, 20)  # Deduct up to 20 points

            return max(0.0, min(100.0, score))  # Clamp between 0 and 100

        except Exception as e:
            self.logger.warning(f"Error calculating bot health score: {e}")
            return 50.0  # Return neutral score on error

    def health_check(self) -> dict[str, Any]:
        """Perform health check and return status."""
        return {
            "service": "WebBotService",
            "status": "healthy",
            "bot_controller_available": self.bot_controller is not None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_service_info(self) -> dict[str, Any]:
        """Get service information and capabilities."""
        return {
            "service": "WebBotService",
            "description": "Web bot service handling bot management business logic",
            "capabilities": [
                "bot_configuration_validation",
                "bot_response_formatting",
                "bot_list_management",
                "bot_metrics_calculation",
                "bot_operation_validation",
            ],
            "version": "1.0.0",
        }

    async def create_bot_through_service(self, bot_config) -> str:
        """Create bot through service layer (wraps controller call)."""
        try:
            if self.bot_controller:
                # Extract parameters from BotConfiguration to match controller interface
                if hasattr(bot_config, 'model_dump'):
                    config_dict = bot_config.model_dump()
                else:
                    config_dict = dict(bot_config) if not isinstance(bot_config, dict) else bot_config

                response = await self.bot_controller.create_bot(
                    template_name=config_dict.get("template_name", "default"),
                    bot_name=config_dict.get("name", config_dict.get("bot_name", "")),
                    exchange=config_dict.get("exchanges", ["binance"])[0] if config_dict.get("exchanges") else "binance",
                    strategy=config_dict.get("strategy_id", config_dict.get("strategy_name", "")),
                    capital_amount=str(config_dict.get("allocated_capital", config_dict.get("max_capital", "1000"))),
                    deployment_strategy=config_dict.get("deployment_strategy", "immediate"),
                    priority=config_dict.get("priority", "normal"),
                    custom_config=config_dict.get("strategy_config", {})
                )
                return response.get("bot_id", response.get("data", {}).get("bot_id", ""))
            else:
                # Mock implementation for development
                mock_bot_id = f"bot_{uuid.uuid4().hex[:8]}"
                self.logger.info(f"Mock bot created: {mock_bot_id}")
                return mock_bot_id
        except Exception as e:
            self.logger.error(f"Error creating bot through service: {e}")
            raise ServiceError(f"Failed to create bot: {e}")

    async def get_bot_status_through_service(self, bot_id: str) -> dict[str, Any]:
        """Get bot status through service layer (wraps controller call)."""
        try:
            if self.bot_controller:
                response = await self.bot_controller.get_bot_status(bot_id)
                return response.get("bot", response.get("data", {}))
            else:
                # Mock implementation for development
                return {
                    "bot_id": bot_id,
                    "status": "running",
                    "state": {
                        "configuration": {},
                        "metrics": {},
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
        except Exception as e:
            self.logger.error(f"Error getting bot status through service: {e}")
            raise ServiceError(f"Failed to get bot status: {e}")

    async def start_bot_through_service(self, bot_id: str) -> bool:
        """Start bot through service layer (wraps controller call)."""
        try:
            if self.bot_controller:
                response = await self.bot_controller.start_bot(bot_id)
                return response.get("success", False)
            else:
                # Mock implementation for development
                self.logger.info(f"Mock bot started: {bot_id}")
                return True
        except Exception as e:
            self.logger.error(f"Error starting bot through service: {e}")
            raise ServiceError(f"Failed to start bot: {e}")

    async def stop_bot_through_service(self, bot_id: str) -> bool:
        """Stop bot through service layer (wraps controller call)."""
        try:
            if self.bot_controller:
                response = await self.bot_controller.stop_bot(bot_id)
                return response.get("success", False)
            else:
                # Mock implementation for development
                self.logger.info(f"Mock bot stopped: {bot_id}")
                return True
        except Exception as e:
            self.logger.error(f"Error stopping bot through service: {e}")
            raise ServiceError(f"Failed to stop bot: {e}")

    async def delete_bot_through_service(self, bot_id: str, force: bool = False) -> bool:
        """Delete bot through service layer (wraps controller call)."""
        try:
            if self.bot_controller and hasattr(self.bot_controller, "terminate_bot"):
                # Use terminate_bot method which is the proper delete operation
                reason = "force_delete" if force else "user_delete"
                response = await self.bot_controller.terminate_bot(bot_id, reason)
                return response.get("success", False)
            elif self.bot_controller and force:
                # Fallback: stop bot if terminate not available
                stop_response = await self.bot_controller.stop_bot(bot_id)
                self.logger.warning(f"Bot termination not available - bot {bot_id} stopped instead")
                return stop_response.get("success", False)
            else:
                # Mock implementation for development
                self.logger.info(f"Mock bot deleted: {bot_id}")
                return True
        except Exception as e:
            self.logger.error(f"Error deleting bot through service: {e}")
            raise ServiceError(f"Failed to delete bot: {e}")

    async def list_bots_through_service(self) -> list[dict[str, Any]]:
        """List bots through service layer (wraps controller call)."""
        try:
            if self.bot_controller:
                response = await self.bot_controller.list_bots()
                return response.get("bots", response.get("data", []))
            else:
                # Mock implementation for development
                return [
                    {
                        "bot_id": "bot_001",
                        "bot_name": "Mock Bot 1",
                        "status": "running",
                        "allocated_capital": 5000.00,
                    },
                    {
                        "bot_id": "bot_002",
                        "bot_name": "Mock Bot 2",
                        "status": "stopped",
                        "allocated_capital": 10000.00,
                    },
                ]
        except Exception as e:
            self.logger.error(f"Error listing bots through service: {e}")
            raise ServiceError(f"Failed to list bots: {e}")

    def get_controller_health_check(self) -> dict[str, Any]:
        """Get controller health check through service layer (wraps controller call)."""
        try:
            if self.bot_controller and hasattr(self.bot_controller, "health_check"):
                return self.bot_controller.health_check()
            else:
                return {
                    "status": "healthy",
                    "controller_available": self.bot_controller is not None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
        except Exception as e:
            self.logger.error(f"Error getting controller health check: {e}")
            return {
                "status": "error",
                "error": str(e),
                "controller_available": False,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    def _get_execution_service(self):
        """Get execution service through dependency injection (for integration)."""
        try:
            from src.core.dependency_injection import DependencyInjector

            injector = DependencyInjector.get_instance()
            if injector and injector.has_service("ExecutionService"):
                return injector.resolve("ExecutionService")
        except Exception as e:
            self.logger.warning(f"Could not get execution service: {e}")
        return None

    async def start_bot_with_execution_integration(self, bot_id: str) -> bool:
        """Start bot with execution service integration."""
        try:
            # Start bot through facade first
            success = await self.start_bot_through_service(bot_id)

            if success:
                # Integrate with execution service if available
                execution_service = self._get_execution_service()
                if execution_service and hasattr(execution_service, "start_bot_execution"):
                    try:
                        # Get bot configuration for execution context
                        bot_status = await self.get_bot_status_through_service(bot_id)
                        bot_config = bot_status.get("state", {}).get("configuration", {})
                        await execution_service.start_bot_execution(bot_id, bot_config)
                        self.logger.info(f"Execution service integration started for bot {bot_id}")
                    except Exception as exec_error:
                        self.logger.warning(
                            f"Failed to start execution service for bot {bot_id}: {exec_error}"
                        )
                        # Don't fail the overall operation if execution integration fails

            return success

        except Exception as e:
            self.logger.error(f"Error starting bot with execution integration: {e}")
            raise ServiceError(f"Failed to start bot with execution integration: {e}")

    async def stop_bot_with_execution_integration(self, bot_id: str) -> bool:
        """Stop bot with execution service integration."""
        try:
            # Notify execution service first if available
            execution_service = self._get_execution_service()
            if execution_service and hasattr(execution_service, "stop_bot_execution"):
                try:
                    await execution_service.stop_bot_execution(bot_id)
                    self.logger.info(f"Execution service integration stopped for bot {bot_id}")
                except Exception as exec_error:
                    self.logger.warning(f"Failed to stop execution service for bot {bot_id}: {exec_error}")

            # Stop bot through facade
            success = await self.stop_bot_through_service(bot_id)
            return success

        except Exception as e:
            self.logger.error(f"Error stopping bot with execution integration: {e}")
            raise ServiceError(f"Failed to stop bot with execution integration: {e}")

    async def pause_bot_through_service(self, bot_id: str) -> bool:
        """Pause bot through service layer (wraps controller call)."""
        try:
            if self.bot_controller and hasattr(self.bot_controller, "pause_bot"):
                response = await self.bot_controller.pause_bot(bot_id)
                return response.get("success", False)
            else:
                # Mock implementation for development - would be implemented when pause is supported
                self.logger.warning(f"Bot pause not yet implemented - mock paused: {bot_id}")
                return False
        except Exception as e:
            self.logger.error(f"Error pausing bot through service: {e}")
            raise ServiceError(f"Failed to pause bot: {e}")

    async def resume_bot_through_service(self, bot_id: str) -> bool:
        """Resume bot through service layer (wraps controller call)."""
        try:
            if self.bot_controller and hasattr(self.bot_controller, "resume_bot"):
                response = await self.bot_controller.resume_bot(bot_id)
                return response.get("success", False)
            else:
                # Mock implementation for development - would be implemented when resume is supported
                self.logger.warning(f"Bot resume not yet implemented - mock resumed: {bot_id}")
                return False
        except Exception as e:
            self.logger.error(f"Error resuming bot through service: {e}")
            raise ServiceError(f"Failed to resume bot: {e}")
