"""
Playground API endpoints for T-Bot web interface.

This module provides playground functionality for strategy testing, backtesting,
and simulation with different parameters, models, and settings.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from src.core.logging import get_logger
from src.core.types import BotType


class TimeInterval(Enum):
    """Time interval enumeration for data intervals."""

    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    HOUR_12 = "12h"
    DAY_1 = "1d"
    WEEK_1 = "1w"


from src.web_interface.security.auth import User, get_current_user, get_trading_user

logger = get_logger(__name__)
router = APIRouter()

# Global references (set by app startup)
bot_orchestrator = None
backtesting_engine = None
strategy_factory = None


def set_dependencies(orchestrator, backtest_engine, strat_factory):
    """Set global dependencies."""
    global bot_orchestrator, backtesting_engine, strategy_factory
    bot_orchestrator = orchestrator
    backtesting_engine = backtest_engine
    strategy_factory = strat_factory


class PlaygroundConfigurationRequest(BaseModel):
    """Request model for playground configuration."""

    # Bot Configuration
    bot_name: str = Field(..., description="Name for the playground bot")
    bot_type: BotType = Field(
        default=BotType.STRATEGY, description="Bot type (strategy/arbitrage/market_maker)"
    )

    # Symbol and Exchange Configuration
    symbols: list[str] = Field(..., description="Trading symbols")
    exchanges: list[str] = Field(..., description="Exchanges to use")

    # Capital Management
    allocated_capital: Decimal = Field(..., gt=0, description="Capital allocated")
    risk_percentage: float = Field(..., gt=0, le=1, description="Risk per trade")

    # Strategy Configuration
    strategy_name: str = Field(..., description="Strategy to use")
    strategy_parameters: dict[str, Any] = Field(
        default_factory=dict, description="Strategy parameters"
    )

    # Risk Settings
    stop_loss_percentage: float = Field(default=0.02, description="Stop loss percentage")
    take_profit_percentage: float = Field(default=0.04, description="Take profit percentage")
    max_position_size: Decimal = Field(default=Decimal("0.1"), description="Maximum position size")

    # Model Selection
    ml_model_id: str | None = Field(None, description="ML model to use")
    enable_ai_features: bool = Field(default=True, description="Enable AI features")

    # Execution Mode
    sandbox_mode: bool = Field(default=True, description="Run in sandbox mode")
    use_historical_data: bool = Field(default=True, description="Use historical data")

    # Backtesting Configuration (optional)
    start_date: datetime | None = Field(None, description="Backtest start date")
    end_date: datetime | None = Field(None, description="Backtest end date")
    time_interval: TimeInterval = Field(default=TimeInterval.HOUR_1, description="Data interval")


class PlaygroundSessionResponse(BaseModel):
    """Response model for playground session."""

    session_id: str
    bot_id: str
    status: str
    configuration: dict[str, Any]
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    is_backtest: bool
    is_sandbox: bool


class PlaygroundStatusResponse(BaseModel):
    """Response model for playground status."""

    session_id: str
    bot_id: str
    status: str  # running, paused, completed, failed
    progress: float  # 0.0 to 1.0
    current_step: str
    total_trades: int
    successful_trades: int
    current_pnl: Decimal
    start_time: datetime
    runtime_minutes: float
    memory_usage_mb: float
    cpu_usage_percent: float
    last_update: datetime


class PlaygroundResultResponse(BaseModel):
    """Response model for playground results."""

    session_id: str
    bot_id: str
    final_pnl: Decimal
    final_pnl_percentage: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    sharpe_ratio: float | None
    max_drawdown: Decimal
    max_drawdown_percentage: float
    total_fees: Decimal
    average_trade_duration: float  # minutes
    profit_factor: float
    completed_at: datetime
    runtime_minutes: float


class PlaygroundLogEntry(BaseModel):
    """Model for playground log entry."""

    timestamp: datetime
    level: str
    message: str
    bot_id: str
    session_id: str
    context: dict[str, Any] = Field(default_factory=dict)


# In-memory storage for active playground sessions
active_sessions: dict[str, dict[str, Any]] = {}


@router.post("/sessions", response_model=PlaygroundSessionResponse)
async def create_playground_session(
    request: PlaygroundConfigurationRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_trading_user),
):
    """
    Create a new playground session for testing strategies and parameters.

    This endpoint creates a new playground session where users can test different
    strategies, parameters, and models either in sandbox mode or with historical data.
    """
    try:
        # Generate unique identifiers
        session_id = str(uuid4())
        bot_id = f"playground_{session_id[:8]}"

        # Validate strategy exists
        if not strategy_factory or not strategy_factory.is_strategy_available(
            request.strategy_name
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Strategy '{request.strategy_name}' not available",
            )

        # Create configuration
        configuration = {
            "bot_name": request.bot_name,
            "bot_type": request.bot_type,
            "symbols": request.symbols,
            "exchanges": request.exchanges,
            "allocated_capital": float(request.allocated_capital),
            "risk_percentage": request.risk_percentage,
            "strategy_name": request.strategy_name,
            "strategy_parameters": request.strategy_parameters,
            "stop_loss_percentage": request.stop_loss_percentage,
            "take_profit_percentage": request.take_profit_percentage,
            "max_position_size": float(request.max_position_size),
            "ml_model_id": request.ml_model_id,
            "enable_ai_features": request.enable_ai_features,
            "sandbox_mode": request.sandbox_mode,
            "use_historical_data": request.use_historical_data,
            "start_date": request.start_date.isoformat() if request.start_date else None,
            "end_date": request.end_date.isoformat() if request.end_date else None,
            "time_interval": request.time_interval,
            "user_id": user.id,
        }

        # Determine if this is a backtest
        is_backtest = request.use_historical_data and request.start_date and request.end_date

        # Store session information
        session_data = {
            "session_id": session_id,
            "bot_id": bot_id,
            "status": "created",
            "configuration": configuration,
            "created_at": datetime.now(timezone.utc),
            "is_backtest": is_backtest,
            "is_sandbox": request.sandbox_mode,
            "user_id": user.id,
            "logs": [],
            "metrics": {},
        }

        active_sessions[session_id] = session_data

        logger.info(f"Created playground session {session_id} for user {user.username}")

        return PlaygroundSessionResponse(
            session_id=session_id,
            bot_id=bot_id,
            status="created",
            configuration=configuration,
            created_at=session_data["created_at"],
            is_backtest=is_backtest,
            is_sandbox=request.sandbox_mode,
        )

    except Exception as e:
        logger.error(f"Error creating playground session: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create playground session: {e!s}",
        )


@router.post("/sessions/{session_id}/start")
async def start_playground_session(
    session_id: str,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_trading_user),
):
    """
    Start a playground session.

    This will either start a bot in sandbox mode or begin a backtest with historical data.
    """
    try:
        if session_id not in active_sessions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Playground session not found"
            )

        session = active_sessions[session_id]

        # Verify user ownership
        if session["user_id"] != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this session",
            )

        # Check if already started
        if session["status"] in ["running", "completed"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Session is already {session['status']}",
            )

        # Update session status
        session["status"] = "starting"
        session["started_at"] = datetime.now(timezone.utc)

        # Start the playground session in background
        background_tasks.add_task(_run_playground_session, session_id)

        logger.info(f"Starting playground session {session_id}")

        return {"message": "Playground session started", "session_id": session_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting playground session {session_id}: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start playground session: {e!s}",
        )


@router.get("/sessions/{session_id}/status", response_model=PlaygroundStatusResponse)
async def get_playground_session_status(
    session_id: str,
    user: User = Depends(get_current_user),
):
    """
    Get the current status of a playground session.
    """
    try:
        if session_id not in active_sessions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Playground session not found"
            )

        session = active_sessions[session_id]

        # Verify user ownership
        if session["user_id"] != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this session",
            )

        metrics = session.get("metrics", {})
        start_time = session.get("started_at", session["created_at"])
        runtime_minutes = (datetime.now(timezone.utc) - start_time).total_seconds() / 60

        return PlaygroundStatusResponse(
            session_id=session_id,
            bot_id=session["bot_id"],
            status=session["status"],
            progress=metrics.get("progress", 0.0),
            current_step=metrics.get("current_step", "Initializing"),
            total_trades=metrics.get("total_trades", 0),
            successful_trades=metrics.get("successful_trades", 0),
            current_pnl=Decimal(str(metrics.get("current_pnl", "0"))),
            start_time=start_time,
            runtime_minutes=runtime_minutes,
            memory_usage_mb=metrics.get("memory_usage_mb", 0.0),
            cpu_usage_percent=metrics.get("cpu_usage_percent", 0.0),
            last_update=datetime.now(timezone.utc),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting playground session status {session_id}: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get session status: {e!s}",
        )


@router.get("/sessions/{session_id}/results", response_model=PlaygroundResultResponse)
async def get_playground_session_results(
    session_id: str,
    user: User = Depends(get_current_user),
):
    """
    Get the final results of a completed playground session.
    """
    try:
        if session_id not in active_sessions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Playground session not found"
            )

        session = active_sessions[session_id]

        # Verify user ownership
        if session["user_id"] != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this session",
            )

        # Check if session is completed
        if session["status"] != "completed":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Session not yet completed"
            )

        results = session.get("results", {})

        return PlaygroundResultResponse(
            session_id=session_id,
            bot_id=session["bot_id"],
            final_pnl=Decimal(str(results.get("final_pnl", "0"))),
            final_pnl_percentage=results.get("final_pnl_percentage", 0.0),
            total_trades=results.get("total_trades", 0),
            winning_trades=results.get("winning_trades", 0),
            losing_trades=results.get("losing_trades", 0),
            win_rate=results.get("win_rate", 0.0),
            sharpe_ratio=results.get("sharpe_ratio"),
            max_drawdown=Decimal(str(results.get("max_drawdown", "0"))),
            max_drawdown_percentage=results.get("max_drawdown_percentage", 0.0),
            total_fees=Decimal(str(results.get("total_fees", "0"))),
            average_trade_duration=results.get("average_trade_duration", 0.0),
            profit_factor=results.get("profit_factor", 0.0),
            completed_at=session.get("completed_at", datetime.now(timezone.utc)),
            runtime_minutes=results.get("runtime_minutes", 0.0),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting playground session results {session_id}: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get session results: {e!s}",
        )


@router.get("/sessions/{session_id}/logs", response_model=list[PlaygroundLogEntry])
async def get_playground_session_logs(
    session_id: str,
    limit: int = Query(default=100, le=1000, description="Maximum number of logs to return"),
    level: str | None = Query(default=None, description="Filter by log level"),
    user: User = Depends(get_current_user),
):
    """
    Get logs from a playground session.
    """
    try:
        if session_id not in active_sessions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Playground session not found"
            )

        session = active_sessions[session_id]

        # Verify user ownership
        if session["user_id"] != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this session",
            )

        logs = session.get("logs", [])

        # Filter by level if specified
        if level:
            logs = [log for log in logs if log.get("level", "").upper() == level.upper()]

        # Limit results
        logs = logs[-limit:]

        return [
            PlaygroundLogEntry(
                timestamp=log["timestamp"],
                level=log["level"],
                message=log["message"],
                bot_id=session["bot_id"],
                session_id=session_id,
                context=log.get("context", {}),
            )
            for log in logs
        ]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting playground session logs {session_id}: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get session logs: {e!s}",
        )


@router.post("/sessions/{session_id}/stop")
async def stop_playground_session(
    session_id: str,
    user: User = Depends(get_trading_user),
):
    """
    Stop a running playground session.
    """
    try:
        if session_id not in active_sessions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Playground session not found"
            )

        session = active_sessions[session_id]

        # Verify user ownership
        if session["user_id"] != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this session",
            )

        # Update session status
        session["status"] = "stopping"

        # If running a bot, stop it
        if not session["is_backtest"] and bot_orchestrator:
            try:
                await bot_orchestrator.stop_bot(session["bot_id"])
            except Exception as e:
                logger.warning(f"Error stopping playground bot {session['bot_id']}: {e!s}")

        session["status"] = "stopped"
        session["completed_at"] = datetime.now(timezone.utc)

        logger.info(f"Stopped playground session {session_id}")

        return {"message": "Playground session stopped", "session_id": session_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping playground session {session_id}: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop playground session: {e!s}",
        )


@router.delete("/sessions/{session_id}")
async def delete_playground_session(
    session_id: str,
    user: User = Depends(get_trading_user),
):
    """
    Delete a playground session and clean up resources.
    """
    try:
        if session_id not in active_sessions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Playground session not found"
            )

        session = active_sessions[session_id]

        # Verify user ownership
        if session["user_id"] != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this session",
            )

        # Stop session if running
        if session["status"] in ["running", "starting"]:
            try:
                await stop_playground_session(session_id, user)
            except Exception as e:
                logger.warning(f"Error stopping session during deletion: {e!s}")

        # Clean up bot if exists
        if not session["is_backtest"] and bot_orchestrator:
            try:
                await bot_orchestrator.remove_bot(session["bot_id"])
            except Exception as e:
                logger.warning(f"Error removing playground bot {session['bot_id']}: {e!s}")

        # Remove from active sessions
        del active_sessions[session_id]

        logger.info(f"Deleted playground session {session_id}")

        return {"message": "Playground session deleted", "session_id": session_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting playground session {session_id}: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete playground session: {e!s}",
        )


@router.get("/sessions", response_model=list[PlaygroundSessionResponse])
async def list_playground_sessions(
    user: User = Depends(get_current_user),
    status_filter: str | None = Query(default=None, description="Filter by status"),
    limit: int = Query(default=50, le=100, description="Maximum number of sessions to return"),
):
    """
    List playground sessions for the current user.
    """
    try:
        user_sessions = []

        for session_id, session in active_sessions.items():
            if session["user_id"] != user.id:
                continue

            if status_filter and session["status"] != status_filter:
                continue

            user_sessions.append(
                PlaygroundSessionResponse(
                    session_id=session_id,
                    bot_id=session["bot_id"],
                    status=session["status"],
                    configuration=session["configuration"],
                    created_at=session["created_at"],
                    started_at=session.get("started_at"),
                    completed_at=session.get("completed_at"),
                    is_backtest=session["is_backtest"],
                    is_sandbox=session["is_sandbox"],
                )
            )

        # Sort by creation date (newest first) and limit
        user_sessions.sort(key=lambda x: x.created_at, reverse=True)
        return user_sessions[:limit]

    except Exception as e:
        logger.error(f"Error listing playground sessions: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list playground sessions: {e!s}",
        )


async def _run_playground_session(session_id: str):
    """
    Background task to run a playground session.

    This function handles the actual execution of the playground session,
    either running a bot or performing a backtest.
    """
    try:
        session = active_sessions.get(session_id)
        if not session:
            logger.error(f"Session {session_id} not found during execution")
            return

        config = session["configuration"]

        # Add initial log
        _add_session_log(session_id, "INFO", "Starting playground session")

        session["status"] = "running"

        if session["is_backtest"]:
            await _run_backtest_session(session_id, config)
        else:
            await _run_sandbox_session(session_id, config)

    except Exception as e:
        logger.error(f"Error running playground session {session_id}: {e!s}")
        session = active_sessions.get(session_id)
        if session:
            session["status"] = "failed"
            session["completed_at"] = datetime.now(timezone.utc)
            _add_session_log(session_id, "ERROR", f"Session failed: {e!s}")


async def _run_backtest_session(session_id: str, config: dict[str, Any]):
    """Run a backtest session."""
    try:
        session = active_sessions[session_id]
        _add_session_log(session_id, "INFO", "Starting backtest")

        # Update metrics
        session["metrics"]["current_step"] = "Initializing backtest"
        session["metrics"]["progress"] = 0.1

        # Mock backtest execution (in real implementation, use BacktestEngine)
        start_date = (
            datetime.fromisoformat(config["start_date"])
            if config["start_date"]
            else datetime.now(timezone.utc) - timedelta(days=30)
        )
        end_date = (
            datetime.fromisoformat(config["end_date"])
            if config["end_date"]
            else datetime.now(timezone.utc)
        )

        # Simulate backtest progress
        total_days = (end_date - start_date).days
        for day in range(total_days):
            current_date = start_date + timedelta(days=day)
            progress = (day + 1) / total_days

            session["metrics"]["progress"] = progress
            session["metrics"]["current_step"] = f"Processing {current_date.strftime('%Y-%m-%d')}"

            # Simulate some trading activity
            if day % 3 == 0:  # Trade every 3 days
                session["metrics"]["total_trades"] = session["metrics"].get("total_trades", 0) + 1
                if day % 6 == 0:  # 50% win rate
                    session["metrics"]["successful_trades"] = (
                        session["metrics"].get("successful_trades", 0) + 1
                    )
                    pnl_change = float(config["allocated_capital"]) * 0.02  # 2% gain
                else:
                    pnl_change = -float(config["allocated_capital"]) * 0.01  # 1% loss

                current_pnl = session["metrics"].get("current_pnl", 0) + pnl_change
                session["metrics"]["current_pnl"] = current_pnl

                _add_session_log(
                    session_id,
                    "INFO",
                    f"Executed trade on {current_date.strftime('%Y-%m-%d')}: {'Profit' if pnl_change > 0 else 'Loss'} ${pnl_change:.2f}",
                )

            # Small delay to simulate processing time
            await asyncio.sleep(0.1)

        # Finalize results
        metrics = session["metrics"]
        total_trades = metrics.get("total_trades", 0)
        successful_trades = metrics.get("successful_trades", 0)
        current_pnl = metrics.get("current_pnl", 0)

        session["results"] = {
            "final_pnl": current_pnl,
            "final_pnl_percentage": (
                (current_pnl / float(config["allocated_capital"])) * 100
                if float(config["allocated_capital"]) > 0
                else 0
            ),
            "total_trades": total_trades,
            "winning_trades": successful_trades,
            "losing_trades": total_trades - successful_trades,
            "win_rate": (successful_trades / total_trades) if total_trades > 0 else 0,
            "sharpe_ratio": 1.5 if current_pnl > 0 else -0.5,  # Mock value
            "max_drawdown": abs(current_pnl * 0.3),  # Mock 30% of PnL
            "max_drawdown_percentage": 15.0,  # Mock value
            "total_fees": float(config["allocated_capital"]) * 0.001,  # 0.1% in fees
            "average_trade_duration": 1440.0,  # 1 day in minutes
            "profit_factor": 2.0 if current_pnl > 0 else 0.5,  # Mock value
            "runtime_minutes": (datetime.now(timezone.utc) - session["started_at"]).total_seconds()
            / 60,
        }

        session["status"] = "completed"
        session["completed_at"] = datetime.now(timezone.utc)

        _add_session_log(session_id, "INFO", f"Backtest completed. Final P&L: ${current_pnl:.2f}")

    except Exception as e:
        logger.error(f"Error running backtest session {session_id}: {e!s}")
        session["status"] = "failed"
        _add_session_log(session_id, "ERROR", f"Backtest failed: {e!s}")
        raise


async def _run_sandbox_session(session_id: str, config: dict[str, Any]):
    """Run a sandbox session with live bot."""
    try:
        session = active_sessions[session_id]
        _add_session_log(session_id, "INFO", "Starting sandbox bot")

        # Update metrics
        session["metrics"]["current_step"] = "Creating sandbox bot"
        session["metrics"]["progress"] = 0.1

        # In real implementation, create and start a sandbox bot
        # For now, we'll simulate bot activity

        session["bot_id"]

        # Mock bot running for demonstration
        runtime_minutes = 0
        while runtime_minutes < 60 and session["status"] == "running":  # Run for 1 hour max
            session["metrics"]["current_step"] = f"Bot running for {runtime_minutes:.1f} minutes"
            session["metrics"]["progress"] = min(runtime_minutes / 60, 1.0)

            # Simulate occasional trades
            if runtime_minutes > 0 and runtime_minutes % 10 == 0:  # Trade every 10 minutes
                session["metrics"]["total_trades"] = session["metrics"].get("total_trades", 0) + 1
                if runtime_minutes % 20 == 0:  # 50% win rate
                    session["metrics"]["successful_trades"] = (
                        session["metrics"].get("successful_trades", 0) + 1
                    )
                    pnl_change = float(config["allocated_capital"]) * 0.005  # 0.5% gain
                else:
                    pnl_change = -float(config["allocated_capital"]) * 0.003  # 0.3% loss

                current_pnl = session["metrics"].get("current_pnl", 0) + pnl_change
                session["metrics"]["current_pnl"] = current_pnl

                _add_session_log(
                    session_id,
                    "INFO",
                    f"Bot executed trade: {'Profit' if pnl_change > 0 else 'Loss'} ${pnl_change:.2f}",
                )

            await asyncio.sleep(60)  # Wait 1 minute
            runtime_minutes += 1

        # Session completed or stopped
        if session["status"] == "running":
            session["status"] = "completed"
            session["completed_at"] = datetime.now(timezone.utc)
            _add_session_log(session_id, "INFO", "Sandbox session completed")

    except Exception as e:
        logger.error(f"Error running sandbox session {session_id}: {e!s}")
        session["status"] = "failed"
        _add_session_log(session_id, "ERROR", f"Sandbox session failed: {e!s}")
        raise


def _add_session_log(
    session_id: str, level: str, message: str, context: dict[str, Any] | None = None
):
    """Add a log entry to a session."""
    try:
        session = active_sessions.get(session_id)
        if not session:
            return

        if "logs" not in session:
            session["logs"] = []

        log_entry = {
            "timestamp": datetime.now(timezone.utc),
            "level": level,
            "message": message,
            "context": context or {},
        }

        session["logs"].append(log_entry)

        # Keep only the last 1000 logs to prevent memory issues
        if len(session["logs"]) > 1000:
            session["logs"] = session["logs"][-1000:]

    except Exception as e:
        logger.error(f"Error adding log to session {session_id}: {e!s}")


# Additional playground features for comprehensive functionality


class PlaygroundConfigurationModel(BaseModel):
    """Enhanced playground configuration model."""

    id: str | None = None
    name: str
    description: str | None = None
    symbols: list[str]
    positionSizing: dict[str, Any]
    tradingSide: str
    riskSettings: dict[str, Any]
    portfolioSettings: dict[str, Any]
    strategy: dict[str, Any]
    model: dict[str, Any] | None = None
    timeframe: str
    createdAt: datetime | None = None
    updatedAt: datetime | None = None


class ABTestModel(BaseModel):
    """A/B test model."""

    id: str | None = None
    name: str
    configurations: dict[str, PlaygroundConfigurationModel]
    status: str = "setup"
    results: dict[str, Any] | None = None
    createdAt: datetime | None = None


class BatchOptimizationModel(BaseModel):
    """Batch optimization model."""

    id: str | None = None
    name: str
    description: str | None = None
    configurations: list[PlaygroundConfigurationModel]
    status: str = "pending"
    settings: dict[str, Any]
    results: dict[str, Any] | None = None
    startTime: datetime | None = None
    endTime: datetime | None = None


# Storage for configurations, A/B tests, and batches
playground_configurations: dict[str, PlaygroundConfigurationModel] = {}
ab_tests: dict[str, ABTestModel] = {}
batch_optimizations: dict[str, BatchOptimizationModel] = {}


@router.get("/configurations", response_model=list[PlaygroundConfigurationModel])
async def get_configurations(
    user: User = Depends(get_current_user),
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=100),
):
    """Get all playground configurations for the user."""
    try:
        user_configs = [config for config in playground_configurations.values()]

        # Simple pagination
        start = (page - 1) * limit
        end = start + limit

        return user_configs[start:end]
    except Exception as e:
        logger.error(f"Error getting configurations: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get configurations: {e!s}",
        )


@router.post("/configurations", response_model=PlaygroundConfigurationModel)
async def save_configuration(
    config: PlaygroundConfigurationModel,
    user: User = Depends(get_current_user),
):
    """Save a playground configuration."""
    try:
        if not config.id:
            config.id = str(uuid4())
            config.createdAt = datetime.now(timezone.utc)

        config.updatedAt = datetime.now(timezone.utc)
        playground_configurations[config.id] = config

        logger.info(f"Saved playground configuration {config.id} for user {user.username}")
        return config
    except Exception as e:
        logger.error(f"Error saving configuration: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save configuration: {e!s}",
        )


@router.delete("/configurations/{config_id}")
async def delete_configuration(
    config_id: str,
    user: User = Depends(get_current_user),
):
    """Delete a playground configuration."""
    try:
        if config_id not in playground_configurations:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Configuration not found"
            )

        del playground_configurations[config_id]
        logger.info(f"Deleted playground configuration {config_id}")
        return {"message": "Configuration deleted", "id": config_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting configuration: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete configuration: {e!s}",
        )


@router.post("/executions/{execution_id}/control")
async def control_execution(
    execution_id: str,
    action: str,
    user: User = Depends(get_current_user),
):
    """Control execution (pause, resume, stop)."""
    try:
        # Find session by execution ID
        session = None
        session_id = None
        for sid, sess in active_sessions.items():
            if sess.get("bot_id") == execution_id or sid == execution_id:
                session = sess
                session_id = sid
                break

        if not session:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Execution not found")

        if session["user_id"] != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to control this execution",
            )

        if action == "pause":
            session["status"] = "paused"
            _add_session_log(session_id, "INFO", "Execution paused")
        elif action == "resume":
            session["status"] = "running"
            _add_session_log(session_id, "INFO", "Execution resumed")
        elif action == "stop":
            session["status"] = "stopped"
            session["completed_at"] = datetime.now(timezone.utc)
            _add_session_log(session_id, "INFO", "Execution stopped")
        elif action == "restart":
            session["status"] = "running"
            session["started_at"] = datetime.now(timezone.utc)
            session["metrics"] = {"progress": 0.0}
            _add_session_log(session_id, "INFO", "Execution restarted")

        return {"message": f"Execution {action}ed", "execution_id": execution_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error controlling execution: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to control execution: {e!s}",
        )


@router.get("/executions")
async def get_executions(
    user: User = Depends(get_current_user),
    status_filter: list[str] | None = Query(default=None),
    mode_filter: list[str] | None = Query(default=None),
):
    """Get all executions for the user with optional filters."""
    try:
        user_executions = []

        for session_id, session in active_sessions.items():
            if session["user_id"] != user.id:
                continue

            if status_filter and session["status"] not in status_filter:
                continue

            mode = (
                "historical"
                if session["is_backtest"]
                else ("sandbox" if session["is_sandbox"] else "live")
            )
            if mode_filter and mode not in mode_filter:
                continue

            execution = {
                "id": session_id,
                "configurationId": session.get("configuration", {}).get("id", ""),
                "mode": mode,
                "status": session["status"],
                "progress": session.get("metrics", {}).get("progress", 0),
                "startTime": session.get("started_at"),
                "endTime": session.get("completed_at"),
                "metrics": session.get("results", {}),
                "trades": [],  # Would be populated from actual trade history
                "logs": session.get("logs", [])[-50:],  # Last 50 logs
            }

            user_executions.append(execution)

        return user_executions
    except Exception as e:
        logger.error(f"Error getting executions: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get executions: {e!s}",
        )


@router.post("/ab-tests", response_model=ABTestModel)
async def create_ab_test(
    ab_test: ABTestModel,
    user: User = Depends(get_current_user),
):
    """Create a new A/B test."""
    try:
        ab_test.id = str(uuid4())
        ab_test.createdAt = datetime.now(timezone.utc)
        ab_tests[ab_test.id] = ab_test

        logger.info(f"Created A/B test {ab_test.id} for user {user.username}")
        return ab_test
    except Exception as e:
        logger.error(f"Error creating A/B test: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create A/B test: {e!s}",
        )


@router.post("/ab-tests/{ab_test_id}/run")
async def run_ab_test(
    ab_test_id: str,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
):
    """Run an A/B test."""
    try:
        if ab_test_id not in ab_tests:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="A/B test not found")

        ab_test = ab_tests[ab_test_id]
        ab_test.status = "running"

        # Run A/B test in background
        background_tasks.add_task(_run_ab_test, ab_test_id)

        return {"message": "A/B test started", "id": ab_test_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running A/B test: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to run A/B test: {e!s}",
        )


@router.post("/batches", response_model=BatchOptimizationModel)
async def start_batch_optimization(
    batch: BatchOptimizationModel,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
):
    """Start batch optimization."""
    try:
        batch.id = str(uuid4())
        batch.startTime = datetime.now(timezone.utc)
        batch.status = "running"
        batch_optimizations[batch.id] = batch

        # Run batch optimization in background
        background_tasks.add_task(_run_batch_optimization, batch.id)

        logger.info(f"Started batch optimization {batch.id} for user {user.username}")
        return batch
    except Exception as e:
        logger.error(f"Error starting batch optimization: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start batch optimization: {e!s}",
        )


@router.get("/batches/{batch_id}/progress")
async def get_batch_progress(
    batch_id: str,
    user: User = Depends(get_current_user),
):
    """Get batch optimization progress."""
    try:
        if batch_id not in batch_optimizations:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Batch optimization not found"
            )

        batch = batch_optimizations[batch_id]

        # Mock progress calculation
        if batch.status == "running":
            runtime_minutes = (datetime.now(timezone.utc) - batch.startTime).total_seconds() / 60
            estimated_total = len(batch.configurations) * 2  # 2 minutes per config
            progress = min(runtime_minutes / estimated_total * 100, 95)  # Cap at 95% until complete
        elif batch.status == "completed":
            progress = 100
        else:
            progress = 0

        return {
            "completedConfigurations": int(progress / 100 * len(batch.configurations)),
            "totalConfigurations": len(batch.configurations),
            "progress": progress,
            "currentStage": "Optimizing parameters" if batch.status == "running" else batch.status,
            "estimatedTimeRemaining": (
                max(0, len(batch.configurations) * 2 - runtime_minutes)
                if batch.status == "running"
                else 0
            ),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting batch progress: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get batch progress: {e!s}",
        )


async def _run_ab_test(ab_test_id: str):
    """Background task to run A/B test."""
    try:
        ab_test = ab_tests.get(ab_test_id)
        if not ab_test:
            return

        # Simulate A/B test execution
        await asyncio.sleep(30)  # Simulate test duration

        # Mock results
        ab_test.results = {
            "significanceLevel": 0.05,
            "pValue": 0.023,
            "confidenceInterval": [1.2, 4.8],
            "winner": "treatment",
            "effect": {"magnitude": 3.2, "direction": "positive", "metric": "Total Return"},
        }

        ab_test.status = "completed"
        logger.info(f"A/B test {ab_test_id} completed")
    except Exception as e:
        logger.error(f"Error running A/B test {ab_test_id}: {e!s}")
        if ab_test_id in ab_tests:
            ab_tests[ab_test_id].status = "failed"


async def _run_batch_optimization(batch_id: str):
    """Background task to run batch optimization."""
    try:
        batch = batch_optimizations.get(batch_id)
        if not batch:
            return

        total_configs = len(batch.configurations)
        results = []

        # Simulate processing each configuration
        for i, _config in enumerate(batch.configurations):
            # Mock results for each configuration
            mock_metrics = {
                "totalReturn": (i - total_configs / 2) * 2,  # Varied returns
                "sharpeRatio": 0.5 + i * 0.1,
                "maxDrawdown": 5 + i * 0.5,
                "winRate": 45 + i * 2,
                "totalTrades": 50 + i * 10,
                "profitFactor": 0.8 + i * 0.1,
            }

            results.append(
                {
                    "configurationId": f"config_{i}",
                    "metrics": mock_metrics,
                    "rank": i + 1,
                    "parameters": {
                        "stopLoss": 1 + i * 0.1,
                        "takeProfit": 2 + i * 0.2,
                        "positionSize": 1 + i * 0.1,
                    },
                    "overfittingRisk": "Low" if i % 3 == 0 else "Medium" if i % 3 == 1 else "High",
                }
            )

            await asyncio.sleep(2)  # Simulate processing time

        # Sort by Sharpe ratio and assign ranks
        results.sort(key=lambda x: x["metrics"]["sharpeRatio"], reverse=True)
        for i, result in enumerate(results):
            result["rank"] = i + 1

        batch.results = {
            "bestConfiguration": batch.configurations[0] if batch.configurations else None,
            "results": results,
        }

        batch.status = "completed"
        batch.endTime = datetime.now(timezone.utc)

        logger.info(f"Batch optimization {batch_id} completed")
    except Exception as e:
        logger.error(f"Error running batch optimization {batch_id}: {e!s}")
        if batch_id in batch_optimizations:
            batch_optimizations[batch_id].status = "failed"
