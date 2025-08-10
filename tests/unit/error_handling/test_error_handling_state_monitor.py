"""
Unit tests for state monitor functionality.

Tests state validation, reconciliation, and consistency monitoring.
"""

from unittest.mock import patch

import pytest

from src.core.config import Config
from src.error_handling.state_monitor import StateMonitor, StateValidationResult


class TestStateValidationResult:
    """Test state validation result."""

    def test_state_validation_result_creation(self):
        """Test state validation result creation."""
        result = StateValidationResult(
            is_consistent=True, discrepancies=[], component="test_component", severity="low"
        )

        assert result.is_consistent is True
        assert result.discrepancies == []
        assert result.component == "test_component"
        assert result.severity == "low"
        assert result.validation_time is not None

    def test_state_validation_result_with_errors(self):
        """Test state validation result with errors."""
        discrepancies = [
            {"type": "balance_mismatch", "expected": 1000, "actual": 950},
            {"type": "position_mismatch", "expected": 5, "actual": 4},
        ]

        result = StateValidationResult(
            is_consistent=False, discrepancies=discrepancies, component="portfolio", severity="high"
        )

        assert result.is_consistent is False
        assert len(result.discrepancies) == 2
        assert result.component == "portfolio"
        assert result.severity == "high"


class TestStateMonitor:
    """Test state monitor functionality."""

    @pytest.fixture
    def config(self):
        """Provide test configuration."""
        return Config()

    @pytest.fixture
    def state_monitor(self, config):
        """Provide state monitor instance."""
        return StateMonitor(config)

    def test_state_monitor_initialization(self, config):
        """Test state monitor initialization."""
        monitor = StateMonitor(config)
        assert monitor.config == config
        assert monitor.last_validation_results == {}
        assert monitor.state_history == []
        assert monitor.reconciliation_attempts == {}
        assert len(monitor.consistency_checks) > 0

    @pytest.mark.asyncio
    async def test_validate_state_consistency(self, state_monitor):
        """Test state consistency validation."""
        # Mock the consistency check
        with patch.object(state_monitor, "_perform_consistency_check") as mock_check:
            mock_check.return_value = {
                "is_consistent": True,
                "discrepancies": [],
                "severity": "low",
            }

            result = await state_monitor.validate_state_consistency("test_component")

            assert isinstance(result, StateValidationResult)
            assert result.is_consistent is True
            assert result.component == "test_component"
            assert result.severity == "low"

    @pytest.mark.asyncio
    async def test_validate_state_consistency_with_errors(self, state_monitor):
        """Test state consistency validation with errors."""
        # Mock the consistency check to return errors
        with patch.object(state_monitor, "_perform_consistency_check") as mock_check:
            mock_check.return_value = {
                "is_consistent": False,
                "discrepancies": [{"type": "test_error"}],
                "severity": "high",
            }

            result = await state_monitor.validate_state_consistency("test_component")

            assert isinstance(result, StateValidationResult)
            assert result.is_consistent is False
            assert len(result.discrepancies) > 0
            assert result.severity == "high"

    @pytest.mark.asyncio
    async def test_reconcile_state(self, state_monitor):
        """Test state reconciliation."""
        discrepancies = [{"type": "test_discrepancy"}]

        # Mock the reconciliation method to return True
        with patch.object(
            state_monitor, "_reconcile_portfolio_balances", return_value=True
        ) as mock_reconcile:
            result = await state_monitor.reconcile_state("portfolio_balance_sync", discrepancies)

            assert result is True
            mock_reconcile.assert_called_once_with(discrepancies)

    def test_get_state_summary(self, state_monitor):
        """Test getting state summary."""
        # Add some validation results
        state_monitor.state_history = [
            StateValidationResult(is_consistent=True, component="test1"),
            StateValidationResult(is_consistent=False, component="test2"),
            StateValidationResult(is_consistent=True, component="test3"),
        ]

        summary = state_monitor.get_state_summary()

        # Check that the summary contains the expected keys
        assert "total_validations" in summary
        assert "recent_inconsistencies" in summary
        assert "last_validation_results" in summary
        assert "reconciliation_attempts" in summary

    def test_get_state_history(self, state_monitor):
        """Test getting state history."""
        # Add some validation results
        state_monitor.state_history = [
            StateValidationResult(is_consistent=True, component="test1"),
            StateValidationResult(is_consistent=False, component="test2"),
        ]

        history = state_monitor.get_state_history(hours=24)

        assert len(history) == 2
        assert isinstance(history[0], StateValidationResult)

    @pytest.mark.asyncio
    async def test_state_monitor_integration(self, state_monitor):
        """Test state monitor integration."""
        # Mock the consistency check
        with patch.object(state_monitor, "_perform_consistency_check") as mock_check:
            mock_check.return_value = {
                "is_consistent": True,
                "discrepancies": [],
                "severity": "low",
            }

            # Test validation
            validation_result = await state_monitor.validate_state_consistency("test_component")
            assert validation_result.is_consistent is True

            # Test summary
            summary = state_monitor.get_state_summary()
            assert "total_validations" in summary

            # Test history
            history = state_monitor.get_state_history()
            assert len(history) > 0
