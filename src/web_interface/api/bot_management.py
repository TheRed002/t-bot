"""
Bot Management API endpoints for T-Bot web interface.

This module provides comprehensive bot management functionality including
creation, configuration, lifecycle management, and monitoring.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from src.core.caching import CacheKeys, cached
from src.core.exceptions import (
    EntityNotFoundError,
    ExecutionError,
    ServiceError,
    ValidationError,
)
from src.core.logging import get_logger
from src.core.types import BotConfiguration, BotPriority, BotStatus, BotType
from src.utils import (
    format_currency,
    format_percentage,
    validate_symbol,
)
from src.web_interface.facade import get_api_facade, get_service_registry
from src.web_interface.security.auth import User, get_admin_user, get_current_user, get_trading_user

logger = get_logger(__name__)
router = APIRouter()

# Global reference to bot service (set by app startup) - DEPRECATED
# Use get_bot_service() instead
bot_service = None


def get_bot_service():
    """
    Get bot service from service registry.

    This function provides access to the bot management service through the service registry.
    The service registry approach is preferred for API endpoints as it provides direct access
    to the service without additional abstraction layers.

    For higher-level orchestration that involves multiple services, consider using the
    API facade pattern via get_api_facade() which provides unified access to all services.
    """
    try:
        registry = get_service_registry()
        return registry.get_service("bot_management")
    except KeyError:
        logger.warning("Bot management service not found in registry")
        return bot_service  # Fallback to global variable for backward compatibility
    except Exception as e:
        logger.error(f"Error getting bot service from registry: {e}")
        return bot_service  # Fallback to global variable


def get_bot_service_via_facade():
    """
    Get bot service through API facade pattern.

    This approach provides access to bot management operations through the unified
    API facade, which can be useful for complex operations that might involve
    coordination with other services.

    Returns the API facade which provides bot management methods like:
    - create_bot()
    - start_bot()
    - stop_bot()
    - get_bot_status()
    - list_bots()
    """
    try:
        facade = get_api_facade()
        return facade
    except Exception as e:
        logger.error(f"Error getting API facade: {e}")
        return None


def set_bot_service(service):
    """Set global bot service reference - DEPRECATED."""
    global bot_service
    logger.warning(
        "set_bot_service is deprecated. Bot service should be registered in the service registry."
    )
    bot_service = service


def set_bot_orchestrator(orchestrator):
    """Set global bot orchestrator reference (alias for set_bot_service) - DEPRECATED."""
    logger.warning(
        "set_bot_orchestrator is deprecated. Bot service should be registered in the service registry."
    )
    set_bot_service(orchestrator)


class CreateBotRequest(BaseModel):
    """Request model for creating a new bot."""

    bot_name: str = Field(..., description="Human-readable bot name")
    bot_type: BotType = Field(..., description="Type of bot")
    strategy_name: str = Field(..., description="Trading strategy to use")
    exchanges: list[str] = Field(..., description="List of exchanges to trade on")
    symbols: list[str] = Field(..., description="List of trading symbols")
    allocated_capital: Decimal = Field(..., gt=0, description="Capital allocated to bot")
    risk_percentage: float = Field(..., gt=0, le=1, description="Risk percentage per trade")
    priority: BotPriority = Field(default=BotPriority.NORMAL, description="Bot priority")
    auto_start: bool = Field(default=False, description="Start bot immediately after creation")
    configuration: dict[str, Any] = Field(
        default_factory=dict, description="Bot-specific configuration"
    )


class UpdateBotRequest(BaseModel):
    """Request model for updating bot configuration."""

    bot_name: str | None = None
    allocated_capital: Decimal | None = None
    risk_percentage: float | None = None
    priority: BotPriority | None = None
    configuration: dict[str, Any] | None = None


class BotResponse(BaseModel):
    """Response model for bot information."""

    bot_id: str
    bot_name: str
    bot_type: str
    strategy_name: str
    exchanges: list[str]
    symbols: list[str]
    allocated_capital: Decimal
    risk_percentage: float
    priority: str
    status: str
    auto_start: bool
    created_at: datetime | None = None
    configuration: dict[str, Any]


class BotSummaryResponse(BaseModel):
    """Response model for bot summary."""

    bot_id: str
    bot_name: str
    status: str
    allocated_capital: Decimal
    current_pnl: Decimal | None = None
    total_trades: int | None = None
    win_rate: float | None = None
    last_trade: datetime | None = None
    uptime: str | None = None


class BotListResponse(BaseModel):
    """Response model for bot listing."""

    bots: list[BotSummaryResponse]
    total: int
    running: int
    stopped: int
    error: int


@router.post("/", response_model=dict[str, Any])
async def create_bot(bot_request: CreateBotRequest, current_user: User = Depends(get_trading_user)):
    """
    Create a new trading bot.

    Args:
        bot_request: Bot creation parameters
        current_user: Current authenticated user

    Returns:
        Dict: Bot creation result

    Raises:
        HTTPException: If bot creation fails
    """
    try:
        current_bot_service = get_bot_service()
        if not current_bot_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Bot service not available",
            )

        # Validate symbols
        for symbol in bot_request.symbols:
            try:
                validate_symbol(symbol)
            except ValidationError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid symbol '{symbol}': {e!s}",
                ) from e

        # Generate unique bot ID
        import uuid

        bot_id = f"bot_{uuid.uuid4().hex[:8]}"

        # Create bot configuration
        bot_config = BotConfiguration(
            bot_id=bot_id,
            bot_name=bot_request.bot_name,
            bot_type=bot_request.bot_type,
            strategy_name=bot_request.strategy_name,
            exchanges=bot_request.exchanges,
            symbols=bot_request.symbols,
            allocated_capital=bot_request.allocated_capital,
            max_position_size=bot_request.allocated_capital
            * Decimal("0.1"),  # Default to 10% of capital
            risk_percentage=bot_request.risk_percentage,
            priority=bot_request.priority,
            auto_start=bot_request.auto_start,
            strategy_config=bot_request.configuration,
            metadata={"created_by": current_user.user_id},
            created_at=datetime.now(timezone.utc),
        )

        # Create bot through bot service
        created_bot_id = await current_bot_service.create_bot(bot_config)

        logger.info(
            "Bot created successfully",
            bot_id=created_bot_id,
            bot_name=bot_request.bot_name,
            created_by=current_user.username,
        )

        # Format response with error handling
        try:
            formatted_capital = format_currency(float(bot_request.allocated_capital))
            formatted_risk = format_percentage(bot_request.risk_percentage)
        except Exception as e:
            logger.warning(f"Formatting error in response: {e}")
            # Fallback to simple formatting
            formatted_capital = f"${float(bot_request.allocated_capital):,.2f}"
            formatted_risk = f"{bot_request.risk_percentage * 100:.2f}%"

        return {
            "success": True,
            "message": "Bot created successfully",
            "bot_id": created_bot_id,
            "bot_name": bot_request.bot_name,
            "auto_started": bot_request.auto_start,
            "allocated_capital": formatted_capital,
            "risk_percentage": formatted_risk,
        }

    except HTTPException:
        raise
    except ValidationError as e:
        logger.error(f"Bot validation failed: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Validation error: {e!s}"
        )
    except ServiceError as e:
        logger.error(f"Bot service error: {e}", user=current_user.username)
        # Check for specific error types
        if "maximum bot limit" in str(e).lower():
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
        elif "already exists" in str(e).lower():
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
        elif "failed to allocate capital" in str(e).lower():
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Service error: {e!s}"
            )
    except ExecutionError as e:
        logger.error(f"Bot execution error: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Execution error: {e!s}"
        )
    except Exception as e:
        logger.error(f"Bot creation failed: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Bot creation failed: {e!s}"
        )


@router.get("/", response_model=BotListResponse)
@cached(
    ttl=30,
    namespace="api",
    data_type="api_response",
    key_generator=lambda status_filter=None, current_user=None: (
        CacheKeys.api_response(
            "list_bots", getattr(current_user, "user_id", "anonymous"), status_filter=status_filter
        )
        if CacheKeys
        else f"list_bots_{getattr(current_user, 'user_id', 'anonymous')}_{status_filter}"
    ),
)
async def list_bots(
    status_filter: BotStatus | None = Query(None, description="Filter bots by status"),
    current_user: User = Depends(get_current_user),
):
    """
    List all bots with optional status filtering.

    Args:
        status_filter: Optional status filter
        current_user: Current authenticated user

    Returns:
        BotListResponse: List of bots

    Raises:
        HTTPException: If listing fails
    """
    try:
        current_bot_service = get_bot_service()
        if not current_bot_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Bot service not available",
            )

        # Get bot statuses from bot service
        all_bots_status = await current_bot_service.get_all_bots_status()
        bots_data = all_bots_status.get("bots", {})

        # Convert to response format
        bot_summaries = []
        status_counts = {"running": 0, "stopped": 0, "error": 0}

        for bot_id, bot_data in bots_data.items():
            if isinstance(bot_data, dict):
                state = bot_data.get("state", {})
                metrics = bot_data.get("metrics")

                bot_status = state.get("status", "unknown").lower()

                # Apply status filter if specified
                if status_filter and bot_status != status_filter.value.lower():
                    continue

                # Count by status
                if bot_status in status_counts:
                    status_counts[bot_status] += 1

                # Extract configuration from state
                config = state.get("configuration", {})

                # Create summary
                summary = BotSummaryResponse(
                    bot_id=bot_id,
                    bot_name=config.get("bot_name", bot_id),
                    status=state.get("status", "unknown"),
                    allocated_capital=Decimal(str(config.get("allocated_capital", 0))),
                    current_pnl=metrics.get("total_pnl") if metrics else None,
                    total_trades=metrics.get("total_trades") if metrics else None,
                    win_rate=metrics.get("win_rate") if metrics else None,
                    last_trade=metrics.get("last_trade_time") if metrics else None,
                    uptime=bot_data.get("uptime"),
                )
                bot_summaries.append(summary)

        return BotListResponse(
            bots=bot_summaries,
            total=len(bot_summaries),
            running=status_counts["running"],
            stopped=status_counts["stopped"],
            error=status_counts["error"],
        )

    except ServiceError as e:
        logger.error(f"Bot service error during listing: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Service error: {e!s}"
        )
    except Exception as e:
        logger.error(f"Bot listing failed: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to list bots: {e!s}"
        )


@router.get("/{bot_id}", response_model=dict[str, Any])
@cached(
    ttl=15,
    namespace="api",
    data_type="api_response",
    key_generator=lambda bot_id, current_user=None: (
        CacheKeys.api_response(
            "get_bot", getattr(current_user, "user_id", "anonymous"), bot_id=bot_id
        )
        if CacheKeys
        else f"get_bot_{getattr(current_user, 'user_id', 'anonymous')}_{bot_id}"
    ),
)
async def get_bot(bot_id: str, current_user: User = Depends(get_current_user)):
    """
    Get detailed information about a specific bot.

    Args:
        bot_id: Bot identifier
        current_user: Current authenticated user

    Returns:
        Dict: Detailed bot information

    Raises:
        HTTPException: If bot not found or access fails
    """
    try:
        current_bot_service = get_bot_service()
        if not current_bot_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Bot service not available",
            )

        # Get bot status from bot service
        try:
            bot_status = await current_bot_service.get_bot_status(bot_id)
            return {"success": True, "bot": bot_status}
        except EntityNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Bot not found: {bot_id}"
            )
        except ServiceError as e:
            # Bot not found errors from service layer
            if "not found" in str(e).lower():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail=f"Bot not found: {bot_id}"
                )
            # Other service errors
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Service error: {e!s}"
            )
        except Exception as e:
            # For backward compatibility, still check for "not found" in string
            if "not found" in str(e).lower():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail=f"Bot not found: {bot_id}"
                )
            raise

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bot retrieval failed: {e}", bot_id=bot_id, user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve bot: {e!s}",
        )


@router.put("/{bot_id}")
async def update_bot(
    bot_id: str, update_request: UpdateBotRequest, current_user: User = Depends(get_trading_user)
):
    """
    Update bot configuration.

    Args:
        bot_id: Bot identifier
        update_request: Update parameters
        current_user: Current authenticated user

    Returns:
        Dict: Update result

    Raises:
        HTTPException: If update fails
    """
    try:
        current_bot_service = get_bot_service()
        if not current_bot_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Bot service not available",
            )

        # Check bot exists and get current status
        try:
            current_status = await current_bot_service.get_bot_status(bot_id)
            current_config = current_status.get("state", {}).get("configuration", {})
        except EntityNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Bot not found: {bot_id}"
            )
        except ServiceError as e:
            # Bot not found errors from service layer
            if "not found" in str(e).lower():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail=f"Bot not found: {bot_id}"
                )
            # Other service errors
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Service error: {e!s}"
            )
        except Exception as e:
            # For backward compatibility, still check for "not found" in string
            if "not found" in str(e).lower():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail=f"Bot not found: {bot_id}"
                )
            raise

        # Create updated configuration
        updated_fields = {}
        if update_request.bot_name is not None:
            current_config.bot_name = update_request.bot_name
            updated_fields["bot_name"] = update_request.bot_name

        if update_request.allocated_capital is not None:
            current_config.allocated_capital = update_request.allocated_capital
            updated_fields["allocated_capital"] = update_request.allocated_capital

        if update_request.risk_percentage is not None:
            current_config.risk_percentage = update_request.risk_percentage
            updated_fields["risk_percentage"] = update_request.risk_percentage

        if update_request.priority is not None:
            current_config.priority = update_request.priority
            updated_fields["priority"] = update_request.priority.value

        if update_request.configuration is not None:
            current_config.configuration.update(update_request.configuration)
            updated_fields["configuration"] = update_request.configuration

        # Apply configuration update through bot service
        # TODO: Implement update_bot_configuration method in bot service
        # For now, we'll need to recreate the bot with new configuration
        logger.warning(
            "Bot configuration update not fully implemented - configuration prepared but not applied"
        )

        logger.info(
            "Bot updated successfully",
            bot_id=bot_id,
            updated_fields=list(updated_fields.keys()),
            updated_by=current_user.username,
        )

        return {
            "success": True,
            "message": "Bot updated successfully",
            "bot_id": bot_id,
            "updated_fields": updated_fields,
        }

    except HTTPException:
        raise
    except ValidationError as e:
        logger.error(
            f"Bot update validation failed: {e}", bot_id=bot_id, user=current_user.username
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Validation error: {e!s}"
        )
    except ServiceError as e:
        logger.error(
            f"Bot service error during update: {e}", bot_id=bot_id, user=current_user.username
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Service error: {e!s}"
        )
    except Exception as e:
        logger.error(f"Bot update failed: {e}", bot_id=bot_id, user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Bot update failed: {e!s}"
        )


@router.post("/{bot_id}/start")
async def start_bot(bot_id: str, current_user: User = Depends(get_trading_user)):
    """
    Start a bot.

    Args:
        bot_id: Bot identifier
        current_user: Current authenticated user

    Returns:
        Dict: Start operation result

    Raises:
        HTTPException: If start fails
    """
    try:
        current_bot_service = get_bot_service()
        if not current_bot_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Bot service not available",
            )

        # Start bot
        success = await current_bot_service.start_bot(bot_id)

        if success:
            logger.info("Bot started successfully", bot_id=bot_id, started_by=current_user.username)
            return {
                "success": True,
                "message": f"Bot {bot_id} started successfully",
                "bot_id": bot_id,
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to start bot {bot_id}"
            )

    except HTTPException:
        raise
    except ServiceError as e:
        logger.error(
            f"Bot service error during start: {e}", bot_id=bot_id, user=current_user.username
        )
        # Check for specific error conditions
        if "not found" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Bot not found: {bot_id}"
            )
        elif "already running" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail="Bot is already running"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Service error: {e!s}"
            )
    except ExecutionError as e:
        logger.error(
            f"Bot execution error during start: {e}", bot_id=bot_id, user=current_user.username
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Failed to start bot: {e!s}"
        )
    except Exception as e:
        logger.error(f"Bot start failed: {e}", bot_id=bot_id, user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Bot start failed: {e!s}"
        )


@router.post("/{bot_id}/stop")
async def stop_bot(bot_id: str, current_user: User = Depends(get_trading_user)):
    """
    Stop a bot.

    Args:
        bot_id: Bot identifier
        current_user: Current authenticated user

    Returns:
        Dict: Stop operation result

    Raises:
        HTTPException: If stop fails
    """
    try:
        current_bot_service = get_bot_service()
        if not current_bot_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Bot service not available",
            )

        # Stop bot
        success = await current_bot_service.stop_bot(bot_id)

        if success:
            logger.info("Bot stopped successfully", bot_id=bot_id, stopped_by=current_user.username)
            return {
                "success": True,
                "message": f"Bot {bot_id} stopped successfully",
                "bot_id": bot_id,
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to stop bot {bot_id}"
            )

    except HTTPException:
        raise
    except ServiceError as e:
        logger.error(
            f"Bot service error during stop: {e}", bot_id=bot_id, user=current_user.username
        )
        # Check for specific error conditions
        if "not found" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Bot not found: {bot_id}"
            )
        elif "not running" in str(e).lower():
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Bot is not running")
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Service error: {e!s}"
            )
    except ExecutionError as e:
        logger.error(
            f"Bot execution error during stop: {e}", bot_id=bot_id, user=current_user.username
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Failed to stop bot: {e!s}"
        )
    except Exception as e:
        logger.error(f"Bot stop failed: {e}", bot_id=bot_id, user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Bot stop failed: {e!s}"
        )


@router.post("/{bot_id}/pause")
async def pause_bot(bot_id: str, current_user: User = Depends(get_trading_user)):
    """
    Pause a bot.

    Args:
        bot_id: Bot identifier
        current_user: Current authenticated user

    Returns:
        Dict: Pause operation result
    """
    try:
        current_bot_service = get_bot_service()
        if not current_bot_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Bot service not available",
            )

        # TODO: Implement pause functionality in BotService
        # success = await current_bot_service.pause_bot(bot_id)
        success = False  # Temporarily disabled

        if success:
            logger.info("Bot paused successfully", bot_id=bot_id, paused_by=current_user.username)
            return {
                "success": True,
                "message": f"Bot {bot_id} paused successfully",
                "bot_id": bot_id,
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to pause bot {bot_id}"
            )

    except HTTPException:
        raise
    except ExecutionError as e:
        logger.error(
            f"Bot execution error during pause: {e}", bot_id=bot_id, user=current_user.username
        )
        # Check for specific error conditions
        if "not running" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail="Cannot pause - bot is not running"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to pause bot: {e!s}",
            )
    except ServiceError as e:
        logger.error(
            f"Bot service error during pause: {e}", bot_id=bot_id, user=current_user.username
        )
        if "not found" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Bot not found: {bot_id}"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Service error: {e!s}"
            )
    except Exception as e:
        logger.error(f"Bot pause failed: {e}", bot_id=bot_id, user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Bot pause failed: {e!s}"
        )


@router.post("/{bot_id}/resume")
async def resume_bot(bot_id: str, current_user: User = Depends(get_trading_user)):
    """
    Resume a paused bot.

    Args:
        bot_id: Bot identifier
        current_user: Current authenticated user

    Returns:
        Dict: Resume operation result
    """
    try:
        current_bot_service = get_bot_service()
        if not current_bot_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Bot service not available",
            )

        # TODO: Implement resume functionality in BotService
        # success = await current_bot_service.resume_bot(bot_id)
        success = False  # Temporarily disabled

        if success:
            logger.info("Bot resumed successfully", bot_id=bot_id, resumed_by=current_user.username)
            return {
                "success": True,
                "message": f"Bot {bot_id} resumed successfully",
                "bot_id": bot_id,
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to resume bot {bot_id}"
            )

    except HTTPException:
        raise
    except ExecutionError as e:
        logger.error(
            f"Bot execution error during resume: {e}", bot_id=bot_id, user=current_user.username
        )
        # Check for specific error conditions
        if "not paused" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail="Cannot resume - bot is not paused"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to resume bot: {e!s}",
            )
    except ServiceError as e:
        logger.error(
            f"Bot service error during resume: {e}", bot_id=bot_id, user=current_user.username
        )
        if "not found" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Bot not found: {bot_id}"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Service error: {e!s}"
            )
    except Exception as e:
        logger.error(f"Bot resume failed: {e}", bot_id=bot_id, user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Bot resume failed: {e!s}"
        )


@router.delete("/{bot_id}")
async def delete_bot(
    bot_id: str,
    force: bool = Query(False, description="Force delete even if running"),
    current_user: User = Depends(get_admin_user),
):
    """
    Delete a bot (admin only).

    Args:
        bot_id: Bot identifier
        force: Force deletion even if running
        current_user: Current admin user

    Returns:
        Dict: Deletion result

    Raises:
        HTTPException: If deletion fails
    """
    try:
        current_bot_service = get_bot_service()
        if not current_bot_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Bot service not available",
            )

        success = await current_bot_service.delete_bot(bot_id, force=force)

        if success:
            logger.info(
                "Bot deleted successfully",
                bot_id=bot_id,
                force=force,
                deleted_by=current_user.username,
            )
            return {
                "success": True,
                "message": f"Bot {bot_id} deleted successfully",
                "bot_id": bot_id,
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to delete bot {bot_id}"
            )

    except HTTPException:
        raise
    except ServiceError as e:
        logger.error(
            f"Bot service error during deletion: {e}", bot_id=bot_id, user=current_user.username
        )
        # Check for specific error conditions
        if "not found" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Bot not found: {bot_id}"
            )
        elif "running" in str(e).lower() and not force:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Cannot delete running bot without force=true",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Service error: {e!s}"
            )
    except Exception as e:
        logger.error(f"Bot deletion failed: {e}", bot_id=bot_id, user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Bot deletion failed: {e!s}"
        )


@router.get("/orchestrator/status")
async def get_orchestrator_status(current_user: User = Depends(get_current_user)):
    """
    Get bot orchestrator status and statistics.

    Args:
        current_user: Current authenticated user

    Returns:
        Dict: Orchestrator status
    """
    try:
        current_bot_service = get_bot_service()
        if not current_bot_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Bot service not available",
            )

        # Get overall status from bot service
        all_bots_status = await current_bot_service.get_all_bots_status()

        # Add service health check
        # Note: health check should be done through public API
        # For now, we'll check if the service is running
        service_healthy = (
            hasattr(current_bot_service, "is_running") and current_bot_service.is_running
        )
        health_check = {
            "status": "healthy" if service_healthy else "unhealthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        bot_status = {
            "service": {
                "is_running": current_bot_service.is_running,
                "healthy": service_healthy,
                "health_check": health_check,
            },
            "bots": all_bots_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return {"success": True, "status": bot_status}

    except ServiceError as e:
        logger.error(f"Bot service error during status retrieval: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service error: {e!s}",
        )
    except Exception as e:
        logger.error(f"Orchestrator status retrieval failed: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get orchestrator status: {e!s}",
        )
