"""
Shared bot state management utilities.

Extracted from duplicated bot state handling code across multiple services.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from enum import Enum

from src.core.logging import get_logger
from src.core.types import BotState, BotStatus, BotConfiguration, BotMetrics
from src.core.exceptions import ValidationError, ServiceError

logger = get_logger(__name__)


class BotStateTransition:
    """Manages valid bot state transitions."""

    # Valid state transitions
    VALID_TRANSITIONS = {
        BotStatus.STOPPED: [BotStatus.STARTING],
        BotStatus.STARTING: [BotStatus.RUNNING, BotStatus.ERROR, BotStatus.STOPPED],
        BotStatus.RUNNING: [BotStatus.PAUSED, BotStatus.STOPPING, BotStatus.ERROR],
        BotStatus.PAUSED: [BotStatus.RUNNING, BotStatus.STOPPING],
        BotStatus.STOPPING: [BotStatus.STOPPED, BotStatus.ERROR],
        BotStatus.ERROR: [BotStatus.STARTING, BotStatus.STOPPED],
        BotStatus.DEAD: []  # Dead bots cannot transition
    }

    @classmethod
    def is_valid_transition(cls, from_status: BotStatus, to_status: BotStatus) -> bool:
        """Check if a state transition is valid."""
        try:
            return to_status in cls.VALID_TRANSITIONS.get(from_status, [])
        except Exception as e:
            logger.error(f"Error checking state transition: {e}")
            return False

    @classmethod
    def get_valid_next_states(cls, current_status: BotStatus) -> list[BotStatus]:
        """Get list of valid next states from current state."""
        return cls.VALID_TRANSITIONS.get(current_status, [])

    @classmethod
    def validate_transition(cls, bot_id: str, from_status: BotStatus, to_status: BotStatus) -> None:
        """
        Validate a state transition and raise exception if invalid.

        Args:
            bot_id: Bot identifier for error messages
            from_status: Current bot status
            to_status: Desired bot status

        Raises:
            ValidationError: If transition is invalid
        """
        if not cls.is_valid_transition(from_status, to_status):
            valid_states = cls.get_valid_next_states(from_status)
            raise ValidationError(
                f"Invalid state transition for bot {bot_id}: "
                f"{from_status.value} -> {to_status.value}. "
                f"Valid transitions: {[s.value for s in valid_states]}"
            )


class BotStateManager:
    """Centralized bot state management with consistency checks."""

    def __init__(self):
        self._bot_states: Dict[str, BotState] = {}
        self._state_history: Dict[str, list[Dict[str, Any]]] = {}
        self._locks: Dict[str, asyncio.Lock] = {}

    async def get_bot_state(self, bot_id: str) -> Optional[BotState]:
        """Get current bot state."""
        return self._bot_states.get(bot_id)

    async def update_bot_state(
        self,
        bot_id: str,
        new_status: BotStatus,
        metrics: Optional[BotMetrics] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> BotState:
        """
        Update bot state with validation and history tracking.

        Args:
            bot_id: Bot identifier
            new_status: New bot status
            metrics: Optional updated metrics
            context: Additional context for the state change

        Returns:
            Updated bot state

        Raises:
            ValidationError: If state transition is invalid
        """
        # Get or create lock for this bot
        if bot_id not in self._locks:
            self._locks[bot_id] = asyncio.Lock()

        async with self._locks[bot_id]:
            current_state = self._bot_states.get(bot_id)

            if current_state:
                # Validate state transition
                BotStateTransition.validate_transition(
                    bot_id, current_state.status, new_status
                )

                # Update existing state
                updated_state = BotState(
                    bot_id=bot_id,
                    status=new_status,
                    metrics=metrics or current_state.metrics,
                    last_updated=datetime.now(timezone.utc),
                    error_message=context.get('error_message') if context else None
                )
            else:
                # Create new state
                updated_state = BotState(
                    bot_id=bot_id,
                    status=new_status,
                    metrics=metrics,
                    last_updated=datetime.now(timezone.utc),
                    error_message=context.get('error_message') if context else None
                )

            # Store updated state
            self._bot_states[bot_id] = updated_state

            # Record state change in history
            await self._record_state_change(bot_id, updated_state, context)

            logger.info(f"Bot {bot_id} state updated to {new_status.value}")
            return updated_state

    async def _record_state_change(
        self,
        bot_id: str,
        new_state: BotState,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record state change in history."""
        try:
            if bot_id not in self._state_history:
                self._state_history[bot_id] = []

            history_entry = {
                'timestamp': new_state.last_updated.isoformat(),
                'status': new_state.status.value,
                'context': context or {},
                'metrics_snapshot': {
                    'total_executions': new_state.metrics.total_executions if new_state.metrics else 0,
                    'error_count': new_state.metrics.error_count if new_state.metrics else 0,
                    'total_pnl': float(new_state.metrics.total_pnl) if new_state.metrics else 0.0
                } if new_state.metrics else None
            }

            self._state_history[bot_id].append(history_entry)

            # Keep only last 100 state changes per bot
            if len(self._state_history[bot_id]) > 100:
                self._state_history[bot_id] = self._state_history[bot_id][-100:]

        except Exception as e:
            logger.error(f"Error recording state change for bot {bot_id}: {e}")

    async def get_state_history(self, bot_id: str, limit: int = 50) -> list[Dict[str, Any]]:
        """Get state change history for a bot."""
        history = self._state_history.get(bot_id, [])
        return history[-limit:] if limit > 0 else history

    async def remove_bot_state(self, bot_id: str) -> bool:
        """Remove bot state and history."""
        try:
            if bot_id in self._locks:
                async with self._locks[bot_id]:
                    self._bot_states.pop(bot_id, None)
                    self._state_history.pop(bot_id, None)
                # Clean up the lock
                del self._locks[bot_id]
            else:
                self._bot_states.pop(bot_id, None)
                self._state_history.pop(bot_id, None)

            logger.info(f"Removed state for bot {bot_id}")
            return True

        except Exception as e:
            logger.error(f"Error removing state for bot {bot_id}: {e}")
            return False

    def get_all_bot_states(self) -> Dict[str, BotState]:
        """Get all current bot states."""
        return self._bot_states.copy()

    def get_bots_by_status(self, status: BotStatus) -> list[str]:
        """Get list of bot IDs with specified status."""
        return [
            bot_id for bot_id, state in self._bot_states.items()
            if state.status == status
        ]

    async def cleanup_stale_states(self, max_age_hours: int = 24) -> int:
        """
        Clean up stale bot states.

        Args:
            max_age_hours: Maximum age of states to keep

        Returns:
            Number of states cleaned up
        """
        try:
            from datetime import timedelta

            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
            cleaned_count = 0

            # Identify stale states
            stale_bot_ids = []
            for bot_id, state in self._bot_states.items():
                if state.last_updated < cutoff_time:
                    stale_bot_ids.append(bot_id)

            # Remove stale states
            for bot_id in stale_bot_ids:
                await self.remove_bot_state(bot_id)
                cleaned_count += 1

            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} stale bot states")

            return cleaned_count

        except Exception as e:
            logger.error(f"Error cleaning up stale states: {e}")
            return 0


class BotConfigurationValidator:
    """Validates bot configurations with common rules."""

    @staticmethod
    def validate_bot_configuration(config: BotConfiguration) -> Dict[str, Any]:
        """
        Validate a bot configuration.

        Args:
            config: Bot configuration to validate

        Returns:
            Validation result with issues and warnings
        """
        issues = []
        warnings = []

        try:
            # Basic field validation
            if not config.name or not isinstance(config.name, str):
                issues.append("Bot name must be a non-empty string")

            if not config.strategy_id or not isinstance(config.strategy_id, str):
                issues.append("Strategy ID must be a non-empty string")

            # Symbol validation
            if not config.symbols or not isinstance(config.symbols, list):
                issues.append("Symbols must be a non-empty list")
            elif len(config.symbols) == 0:
                issues.append("At least one symbol must be specified")

            # Exchange validation
            if not config.exchange or not isinstance(config.exchange, str):
                issues.append("Exchange must be specified")

            # Capital validation
            try:
                if hasattr(config, 'capital_allocation') and config.capital_allocation:
                    capital = float(config.capital_allocation)
                    if capital <= 0:
                        issues.append("Capital allocation must be positive")
                    elif capital < 100:
                        warnings.append("Capital allocation is very low (< 100)")
                    elif capital > 1000000:
                        warnings.append("Capital allocation is very high (> 1M)")
            except (ValueError, TypeError):
                issues.append("Invalid capital allocation format")

            # Risk parameters validation
            if hasattr(config, 'risk_parameters') and config.risk_parameters:
                risk_params = config.risk_parameters

                if 'max_position_size' in risk_params:
                    try:
                        max_pos = float(risk_params['max_position_size'])
                        if max_pos <= 0 or max_pos > 100:
                            issues.append("Max position size must be between 0 and 100 percent")
                    except (ValueError, TypeError):
                        issues.append("Invalid max position size format")

                if 'stop_loss_percent' in risk_params:
                    try:
                        stop_loss = float(risk_params['stop_loss_percent'])
                        if stop_loss <= 0 or stop_loss > 50:
                            warnings.append("Stop loss percent should be between 0-50%")
                    except (ValueError, TypeError):
                        issues.append("Invalid stop loss percent format")

            # Strategy parameters validation
            if hasattr(config, 'strategy_parameters') and config.strategy_parameters:
                strategy_params = config.strategy_parameters

                # Common parameter checks
                if 'timeframe' in strategy_params:
                    valid_timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
                    if strategy_params['timeframe'] not in valid_timeframes:
                        warnings.append(f"Unusual timeframe: {strategy_params['timeframe']}")

            return {
                'valid': len(issues) == 0,
                'issues': issues,
                'warnings': warnings,
                'config_summary': {
                    'name': config.name,
                    'strategy_id': config.strategy_id,
                    'exchange': config.exchange,
                    'symbol_count': len(config.symbols) if config.symbols else 0
                }
            }

        except Exception as e:
            logger.error(f"Error validating bot configuration: {e}")
            return {
                'valid': False,
                'issues': [f"Validation error: {e}"],
                'warnings': [],
                'config_summary': {}
            }

    @staticmethod
    def validate_configuration_update(
        current_config: BotConfiguration,
        update_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate configuration update against current configuration.

        Args:
            current_config: Current bot configuration
            update_data: Proposed updates

        Returns:
            Validation result for the update
        """
        issues = []
        warnings = []

        try:
            # Check for dangerous changes
            dangerous_fields = ['exchange', 'strategy_id']
            for field in dangerous_fields:
                if field in update_data:
                    current_value = getattr(current_config, field, None)
                    new_value = update_data[field]
                    if current_value != new_value:
                        warnings.append(
                            f"Changing {field} from '{current_value}' to '{new_value}' "
                            "may require bot restart"
                        )

            # Validate capital changes
            if 'capital_allocation' in update_data:
                try:
                    new_capital = float(update_data['capital_allocation'])
                    current_capital = float(getattr(current_config, 'capital_allocation', 0))

                    if new_capital <= 0:
                        issues.append("Capital allocation must be positive")
                    elif new_capital > current_capital * 10:
                        warnings.append("Capital increase is very large (>10x)")
                    elif new_capital < current_capital * 0.1:
                        warnings.append("Capital decrease is very large (>90%)")

                except (ValueError, TypeError):
                    issues.append("Invalid capital allocation format")

            # Validate symbol changes
            if 'symbols' in update_data:
                new_symbols = update_data['symbols']
                if not isinstance(new_symbols, list) or len(new_symbols) == 0:
                    issues.append("Symbols must be a non-empty list")
                else:
                    current_symbols = set(current_config.symbols or [])
                    new_symbols_set = set(new_symbols)

                    added_symbols = new_symbols_set - current_symbols
                    removed_symbols = current_symbols - new_symbols_set

                    if added_symbols:
                        warnings.append(f"Adding symbols: {list(added_symbols)}")
                    if removed_symbols:
                        warnings.append(f"Removing symbols: {list(removed_symbols)}")

            return {
                'valid': len(issues) == 0,
                'issues': issues,
                'warnings': warnings,
                'change_summary': {
                    'fields_changed': list(update_data.keys()),
                    'requires_restart': any(
                        field in update_data for field in ['exchange', 'strategy_id']
                    )
                }
            }

        except Exception as e:
            logger.error(f"Error validating configuration update: {e}")
            return {
                'valid': False,
                'issues': [f"Update validation error: {e}"],
                'warnings': [],
                'change_summary': {}
            }


class BotStateMetrics:
    """Collect and analyze bot state metrics."""

    def __init__(self):
        self._state_metrics: Dict[str, Dict[str, Any]] = {}

    def record_state_duration(self, bot_id: str, status: BotStatus, duration_seconds: float) -> None:
        """Record how long a bot spent in a particular state."""
        try:
            if bot_id not in self._state_metrics:
                self._state_metrics[bot_id] = {}

            status_key = f"duration_{status.value}"
            if status_key not in self._state_metrics[bot_id]:
                self._state_metrics[bot_id][status_key] = []

            self._state_metrics[bot_id][status_key].append(duration_seconds)

            # Keep only last 100 durations
            if len(self._state_metrics[bot_id][status_key]) > 100:
                self._state_metrics[bot_id][status_key] = self._state_metrics[bot_id][status_key][-100:]

        except Exception as e:
            logger.error(f"Error recording state duration for bot {bot_id}: {e}")

    def get_state_statistics(self, bot_id: str) -> Dict[str, Any]:
        """Get state statistics for a bot."""
        try:
            if bot_id not in self._state_metrics:
                return {}

            stats = {}
            bot_metrics = self._state_metrics[bot_id]

            for key, durations in bot_metrics.items():
                if key.startswith('duration_') and durations:
                    status = key.replace('duration_', '')
                    stats[status] = {
                        'avg_duration': sum(durations) / len(durations),
                        'min_duration': min(durations),
                        'max_duration': max(durations),
                        'total_time': sum(durations),
                        'occurrences': len(durations)
                    }

            return stats

        except Exception as e:
            logger.error(f"Error getting state statistics for bot {bot_id}: {e}")
            return {}


# Global state manager instance
_global_state_manager: Optional[BotStateManager] = None


def get_bot_state_manager() -> BotStateManager:
    """Get or create global bot state manager."""
    global _global_state_manager

    if _global_state_manager is None:
        _global_state_manager = BotStateManager()

    return _global_state_manager