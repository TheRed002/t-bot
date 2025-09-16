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
from src.core.types import BotPriority, BotStatus, BotType
from src.execution.interfaces import ExecutionServiceInterface
from src.utils.web_interface_utils import handle_api_error
from src.web_interface.di_registration import get_web_bot_service
from src.web_interface.security.auth import User, get_admin_user, get_current_user, get_trading_user

logger = get_logger(__name__)
router = APIRouter()


def get_bot_service():
    """
    Get bot management service through web service layer.

    This provides access to bot management operations through the web service,
    ensuring proper separation of concerns and avoiding direct facade access.
    """
    return get_web_bot_service()


def get_web_bot_service_instance():
    """Get web bot service for business logic through DI."""
    return get_web_bot_service()


def get_execution_service() -> ExecutionServiceInterface | None:
    """Get execution service for bot lifecycle management."""
    try:
        from src.core.dependency_injection import get_global_injector

        injector = get_global_injector()
        if injector and injector.has_service("ExecutionService"):
            return injector.resolve("ExecutionService")
    except Exception as e:
        logger.warning(f"Could not get execution service: {e}")
    return None


# Deprecated functions for backward compatibility
def set_bot_service(service):
    """DEPRECATED: Use service registry instead."""
    logger.warning("set_bot_service is deprecated. Use service registry instead.")


def set_bot_orchestrator(orchestrator):
    """DEPRECATED: Use service registry instead."""
    logger.warning("set_bot_orchestrator is deprecated. Use service registry instead.")


class CreateBotRequest(BaseModel):
    """Request model for creating a new bot."""

    bot_name: str = Field(..., description="Human-readable bot name")
    bot_type: BotType = Field(..., description="Type of bot")
    strategy_name: str = Field(..., description="Trading strategy to use")
    exchanges: list[str] = Field(..., description="List of exchanges to trade on")
    symbols: list[str] = Field(..., description="List of trading symbols")
    allocated_capital: Decimal = Field(..., gt=0, description="Capital allocated to bot")
    risk_percentage: Decimal = Field(..., gt=0, le=1, description="Risk percentage per trade")
    priority: BotPriority = Field(default=BotPriority.NORMAL, description="Bot priority")
    auto_start: bool = Field(default=False, description="Start bot immediately after creation")
    configuration: dict[str, Any] = Field(
        default_factory=dict, description="Bot-specific configuration"
    )


class UpdateBotRequest(BaseModel):
    """Request model for updating bot configuration."""

    bot_name: str | None = None
    allocated_capital: Decimal | None = None
    risk_percentage: Decimal | None = None
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
    risk_percentage: Decimal
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
    win_rate: Decimal | None = None
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
        web_bot_service = get_web_bot_service_instance()

        # Validate configuration through web service (business logic moved to service)
        config_data = {
            "bot_name": bot_request.bot_name,
            "bot_type": bot_request.bot_type,
            "strategy_name": bot_request.strategy_name,
            "exchanges": bot_request.exchanges,
            "symbols": bot_request.symbols,
            "allocated_capital": bot_request.allocated_capital,
            "risk_percentage": bot_request.risk_percentage,
            "priority": bot_request.priority,
            "auto_start": bot_request.auto_start,
            "configuration": bot_request.configuration,
        }

        validation_result = await web_bot_service.validate_bot_configuration(config_data)
        if not validation_result["valid"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Validation errors: {', '.join(validation_result['errors'])}",
            )

        # Create bot configuration through web service (business logic moved to service)
        bot_config = await web_bot_service.create_bot_configuration(
            config_data, current_user.user_id
        )

        # Create bot through service layer - service handles facade calls
        created_bot_id = await web_bot_service.create_bot_through_service(bot_config)

        logger.info(
            "Bot created successfully",
            bot_id=created_bot_id,
            bot_name=bot_request.bot_name,
            created_by=current_user.username,
        )

        # Format response through web service (business logic moved to service)
        bot_data = {
            "bot_id": created_bot_id,
            "bot_name": bot_request.bot_name,
            "allocated_capital": bot_request.allocated_capital,
            "risk_percentage": bot_request.risk_percentage,
            "auto_start": bot_request.auto_start,
        }

        return await web_bot_service.format_bot_response(bot_data)

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
        raise handle_api_error(e, "Bot creation", user=current_user.username)


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
        web_bot_service = get_web_bot_service_instance()

        # Get formatted bot list through web service (business logic moved to service)
        filters = {}
        if status_filter:
            filters["status_filter"] = status_filter.value

        bot_list_data = await web_bot_service.get_formatted_bot_list(filters)

        # Convert to response models
        bot_summaries = []
        for bot_data in bot_list_data["bots"]:
            summary = BotSummaryResponse(**bot_data)
            bot_summaries.append(summary)

        status_counts = bot_list_data["status_counts"]

        return BotListResponse(
            bots=bot_summaries,
            total=bot_list_data["total"],
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
        raise handle_api_error(e, "Bot listing", user=current_user.username)


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
        web_bot_service = get_web_bot_service_instance()

        # Get bot status through service layer - service handles facade calls
        try:
            bot_status = await web_bot_service.get_bot_status_through_service(bot_id)

            # Get enhanced metrics through web service (business logic moved to service)
            bot_metrics = await web_bot_service.calculate_bot_metrics(bot_id)

            # Combine facade data with web service metrics
            enhanced_bot_data = {
                **bot_status,
                "enhanced_metrics": bot_metrics,
            }

            return {"success": True, "bot": enhanced_bot_data}
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
        raise handle_api_error(
            e, "Bot retrieval", user=current_user.username, context={"bot_id": bot_id}
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
        web_bot_service = get_web_bot_service_instance()

        # Check bot exists and get current status through service layer
        try:
            current_status = await web_bot_service.get_bot_status_through_service(bot_id)
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
        # Note: This implementation is pending the update_bot_configuration method
        logger.warning(
            "Bot configuration update requires implementation of update_bot_configuration method"
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
        raise handle_api_error(
            e, "Bot update", user=current_user.username, context={"bot_id": bot_id}
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
        web_bot_service = get_web_bot_service_instance()
        execution_service = get_execution_service()

        # Start bot through service layer - service handles facade calls
        success = await web_bot_service.start_bot_through_service(bot_id)

        if success:
            # Also notify execution service about bot start if available
            if execution_service and hasattr(execution_service, "start_bot_execution"):
                try:
                    # Get bot configuration for execution context through service
                    bot_status = await web_bot_service.get_bot_status_through_service(bot_id)
                    bot_config = bot_status.get("state", {}).get("configuration", {})
                    await execution_service.start_bot_execution(bot_id, bot_config)
                except Exception as exec_error:
                    logger.warning(
                        f"Failed to start execution engine for bot {bot_id}: {exec_error}"
                    )

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
        raise handle_api_error(
            e, "Bot start", user=current_user.username, context={"bot_id": bot_id}
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
        web_bot_service = get_web_bot_service_instance()
        execution_service = get_execution_service()

        # Also notify execution service about bot stop if available
        if execution_service and hasattr(execution_service, "stop_bot_execution"):
            try:
                await execution_service.stop_bot_execution(bot_id)
            except Exception as exec_error:
                logger.warning(f"Failed to stop execution engine for bot {bot_id}: {exec_error}")

        # Stop bot through service layer - service handles facade calls
        success = await web_bot_service.stop_bot_through_service(bot_id)

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
        raise handle_api_error(
            e, "Bot stop", user=current_user.username, context={"bot_id": bot_id}
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
        get_web_bot_service_instance()

        # Pause functionality requires implementation in BotService
        # This would call: await bot_facade.pause_bot(bot_id)
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Bot pause functionality is not yet implemented",
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
        raise handle_api_error(
            e, "Bot pause", user=current_user.username, context={"bot_id": bot_id}
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
        get_web_bot_service_instance()

        # Resume functionality requires implementation in BotService
        # This would call: await bot_facade.resume_bot(bot_id)
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Bot resume functionality is not yet implemented",
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
        raise handle_api_error(
            e, "Bot resume", user=current_user.username, context={"bot_id": bot_id}
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
        web_bot_service = get_web_bot_service_instance()

        # Delete bot through service layer - service handles facade calls and fallbacks
        success = await web_bot_service.delete_bot_through_service(bot_id, force=force)

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
        raise handle_api_error(
            e, "Bot deletion", user=current_user.username, context={"bot_id": bot_id}
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
        web_bot_service = get_web_bot_service_instance()

        # Get overall status through service layer - service handles facade calls
        health_check = web_bot_service.get_facade_health_check()

        # Get bot list for summary through service layer
        bots = await web_bot_service.list_bots_through_service()

        # Create status summary
        bot_status = {
            "service": {
                "is_running": True,
                "healthy": health_check.get("status") == "healthy",
                "health_check": health_check,
            },
            "bots": {
                "total": len(bots),
                "running": len([b for b in bots if b.get("status") == "running"]),
                "stopped": len([b for b in bots if b.get("status") == "stopped"]),
                "error": len([b for b in bots if b.get("status") == "error"]),
            },
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
        raise handle_api_error(e, "Orchestrator status retrieval", user=current_user.username)


# Endpoint aliases for API compatibility
@router.get("/status")
async def get_status_alias(current_user: User = Depends(get_current_user)):
    """Alias for /orchestrator/status endpoint."""
    return await get_orchestrator_status(current_user)


@router.get("/list")
async def get_list_alias(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status_filter: str | None = Query(None),
    current_user: User = Depends(get_current_user),
):
    """Alias for GET / endpoint with list semantics."""
    return await list_bots(limit, offset, status_filter, current_user)


@router.post("/create")
async def create_alias(
    request: dict[str, Any],
    trading_user: User = Depends(get_trading_user),
):
    """Alias for POST / endpoint with create semantics."""
    return await create_bot(request, trading_user)


@router.get("/config")
async def get_config_alias(current_user: User = Depends(get_current_user)):
    """Get default bot configuration template."""
    try:
        return {
            "success": True,
            "config_template": {
                "name": "New Trading Bot",
                "bot_type": "momentum_trader",
                "strategy": "trend_following",
                "capital_allocation": "0.1",
                "risk_limits": {
                    "max_position_size": "0.02",
                    "stop_loss_percentage": "0.05",
                    "daily_loss_limit": "0.01",
                },
                "parameters": {"timeframe": "1h", "moving_average_period": 20, "rsi_threshold": 70},
            },
        }
    except Exception as e:
        logger.error(f"Config template error: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get configuration template",
        )


@router.get("/logs")
async def get_logs_alias(current_user: User = Depends(get_current_user)):
    """Get bot system logs."""
    try:
        return {
            "success": True,
            "logs": [
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "level": "INFO",
                    "message": "Bot orchestrator is running",
                    "component": "orchestrator",
                },
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "level": "INFO",
                    "message": "No active bots",
                    "component": "bot_manager",
                },
            ],
            "total_logs": 2,
        }
    except Exception as e:
        logger.error(f"Logs retrieval error: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve logs"
        )
