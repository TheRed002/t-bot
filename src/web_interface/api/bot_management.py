"""
Bot Management API endpoints for T-Bot web interface.

This module provides comprehensive bot management functionality including
creation, configuration, lifecycle management, and monitoring.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from src.core.logging import get_logger
from src.core.types import BotConfiguration, BotPriority, BotStatus, BotType
from src.web_interface.security.auth import User, get_admin_user, get_current_user, get_trading_user

logger = get_logger(__name__)
router = APIRouter()

# Global reference to bot orchestrator (set by app startup)
bot_orchestrator = None


def set_bot_orchestrator(orchestrator):
    """Set global bot orchestrator reference."""
    global bot_orchestrator
    bot_orchestrator = orchestrator


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
        if not bot_orchestrator:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Bot orchestrator not available",
            )

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
            risk_percentage=bot_request.risk_percentage,
            priority=bot_request.priority,
            auto_start=bot_request.auto_start,
            configuration=bot_request.configuration,
            created_by=current_user.user_id,
            created_at=datetime.utcnow(),
        )

        # Create bot through orchestrator
        created_bot_id = await bot_orchestrator.create_bot(bot_config)

        logger.info(
            "Bot created successfully",
            bot_id=created_bot_id,
            bot_name=bot_request.bot_name,
            created_by=current_user.username,
        )

        return {
            "success": True,
            "message": "Bot created successfully",
            "bot_id": created_bot_id,
            "bot_name": bot_request.bot_name,
            "auto_started": bot_request.auto_start,
        }

    except Exception as e:
        logger.error(f"Bot creation failed: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Bot creation failed: {e!s}"
        )


@router.get("/", response_model=BotListResponse)
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
        if not bot_orchestrator:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Bot orchestrator not available",
            )

        # Get bot list from orchestrator
        bot_list = await bot_orchestrator.get_bot_list(status_filter)

        # Convert to response format
        bot_summaries = []
        status_counts = {"running": 0, "stopped": 0, "error": 0}

        for bot_data in bot_list:
            # Count by status
            bot_status = bot_data.get("status", "unknown").lower()
            if bot_status in status_counts:
                status_counts[bot_status] += 1

            # Create summary
            summary = BotSummaryResponse(
                bot_id=bot_data.get("bot_id", ""),
                bot_name=bot_data.get("bot_name", ""),
                status=bot_data.get("status", "unknown"),
                allocated_capital=bot_data.get("allocated_capital", Decimal("0")),
                current_pnl=bot_data.get("metrics", {}).get("total_pnl"),
                total_trades=bot_data.get("metrics", {}).get("total_trades"),
                win_rate=bot_data.get("metrics", {}).get("win_rate"),
                last_trade=bot_data.get("metrics", {}).get("last_trade_time"),
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

    except Exception as e:
        logger.error(f"Bot listing failed: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to list bots: {e!s}"
        )


@router.get("/{bot_id}", response_model=dict[str, Any])
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
        if not bot_orchestrator:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Bot orchestrator not available",
            )

        # Get bot details
        if bot_id not in bot_orchestrator.bot_instances:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Bot not found: {bot_id}"
            )

        bot_instance = bot_orchestrator.bot_instances[bot_id]
        bot_summary = await bot_instance.get_bot_summary()

        return {"success": True, "bot": bot_summary}

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
        if not bot_orchestrator:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Bot orchestrator not available",
            )

        # Check bot exists
        if bot_id not in bot_orchestrator.bot_instances:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Bot not found: {bot_id}"
            )

        # Get current bot instance
        bot_instance = bot_orchestrator.bot_instances[bot_id]
        current_config = bot_instance.get_bot_config()

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

        # Apply configuration update
        await bot_instance.update_configuration(current_config)

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
    except Exception as e:
        logger.error(f"Bot update failed: {e}", bot_id=bot_id, user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Bot update failed: {e!s}"
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
        if not bot_orchestrator:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Bot orchestrator not available",
            )

        # Start bot
        success = await bot_orchestrator.start_bot(bot_id)

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
    except Exception as e:
        logger.error(f"Bot start failed: {e}", bot_id=bot_id, user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Bot start failed: {e!s}"
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
        if not bot_orchestrator:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Bot orchestrator not available",
            )

        # Stop bot
        success = await bot_orchestrator.stop_bot(bot_id)

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
    except Exception as e:
        logger.error(f"Bot stop failed: {e}", bot_id=bot_id, user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Bot stop failed: {e!s}"
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
        if not bot_orchestrator:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Bot orchestrator not available",
            )

        success = await bot_orchestrator.pause_bot(bot_id)

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
    except Exception as e:
        logger.error(f"Bot pause failed: {e}", bot_id=bot_id, user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Bot pause failed: {e!s}"
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
        if not bot_orchestrator:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Bot orchestrator not available",
            )

        success = await bot_orchestrator.resume_bot(bot_id)

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
    except Exception as e:
        logger.error(f"Bot resume failed: {e}", bot_id=bot_id, user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Bot resume failed: {e!s}"
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
        if not bot_orchestrator:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Bot orchestrator not available",
            )

        success = await bot_orchestrator.delete_bot(bot_id, force=force)

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
    except Exception as e:
        logger.error(f"Bot deletion failed: {e}", bot_id=bot_id, user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Bot deletion failed: {e!s}"
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
        if not bot_orchestrator:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Bot orchestrator not available",
            )

        status = await bot_orchestrator.get_orchestrator_status()
        return {"success": True, "status": status}

    except Exception as e:
        logger.error(f"Orchestrator status retrieval failed: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get orchestrator status: {e!s}",
        )
