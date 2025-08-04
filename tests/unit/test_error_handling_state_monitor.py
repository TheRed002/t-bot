"""
Unit tests for state monitor functionality.

Tests state validation, reconciliation, and consistency monitoring.
"""

import pytest
import asyncio
import time
from datetime import datetime, timezone
from unittest.mock import patch, AsyncMock, MagicMock
from src.error_handling.state_monitor import (
    StateValidationResult, StateMonitor
)
from src.core.exceptions import TradingBotError, StateConsistencyError
from src.core.config import Config


class TestStateValidationResult:
    """Test state validation result."""
    
    def test_state_validation_result_creation(self):
        """Test state validation result creation."""
        result = StateValidationResult(
            is_valid=True,
            errors=[],
            warnings=["Minor inconsistency detected"],
            timestamp=datetime.now(timezone.utc),
            component="database",
            validation_type="consistency"
        )
        
        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == ["Minor inconsistency detected"]
        assert result.timestamp is not None
        assert result.component == "database"
        assert result.validation_type == "consistency"
    
    def test_state_validation_result_with_errors(self):
        """Test state validation result with errors."""
        result = StateValidationResult(
            is_valid=False,
            errors=["Data corruption detected", "Inconsistent state"],
            warnings=[],
            timestamp=datetime.now(timezone.utc),
            component="exchange",
            validation_type="integrity"
        )
        
        assert result.is_valid is False
        assert len(result.errors) == 2
        assert "Data corruption detected" in result.errors
        assert "Inconsistent state" in result.errors


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
        assert monitor.validation_history == []
        assert monitor.audit_trail == []
        assert monitor.transactions == {}
    
    @pytest.mark.asyncio
    async def test_validate_state_consistency(self, state_monitor):
        """Test state consistency validation."""
        state_data = {
            "orders": {"count": 5, "total_value": 1000.0},
            "positions": {"count": 3, "total_value": 500.0},
            "balances": {"available": 2000.0, "reserved": 1500.0}
        }
        
        result = await state_monitor.validate_state_consistency(state_data)
        
        assert isinstance(result, StateValidationResult)
        assert result.component == "state_monitor"
        assert result.validation_type == "consistency"
        assert result.timestamp is not None
    
    @pytest.mark.asyncio
    async def test_validate_state_consistency_with_errors(self, state_monitor):
        """Test state consistency validation with errors."""
        # Inconsistent state data
        state_data = {
            "orders": {"count": 5, "total_value": 1000.0},
            "positions": {"count": 3, "total_value": 500.0},
            "balances": {"available": 2000.0, "reserved": 1500.0},
            "inconsistent_field": "should_not_exist"
        }
        
        result = await state_monitor.validate_state_consistency(state_data)
        
        assert isinstance(result, StateValidationResult)
        # Should detect inconsistency
        assert not result.is_valid or len(result.errors) > 0 or len(result.warnings) > 0
    
    @pytest.mark.asyncio
    async def test_reconcile_state(self, state_monitor):
        """Test state reconciliation."""
        current_state = {
            "orders": {"count": 5, "total_value": 1000.0},
            "positions": {"count": 3, "total_value": 500.0}
        }
        
        expected_state = {
            "orders": {"count": 4, "total_value": 800.0},
            "positions": {"count": 3, "total_value": 500.0}
        }
        
        reconciliation_result = await state_monitor.reconcile_state(current_state, expected_state)
        
        assert reconciliation_result is not None
        assert "reconciled_state" in reconciliation_result
        assert "actions_taken" in reconciliation_result
    
    @pytest.mark.asyncio
    async def test_detect_corruption(self, state_monitor):
        """Test corruption detection."""
        corrupted_state = {
            "orders": {"count": -1, "total_value": "invalid_value"},
            "positions": {"count": 3, "total_value": 500.0}
        }
        
        corruption_result = await state_monitor.detect_corruption(corrupted_state)
        
        assert corruption_result is not None
        assert "is_corrupted" in corruption_result
        assert "corruption_details" in corruption_result
    
    def test_add_audit_entry(self, state_monitor):
        """Test adding audit trail entry."""
        entry = {
            "action": "order_placed",
            "timestamp": datetime.now(timezone.utc),
            "user_id": "test_user",
            "details": {"order_id": "123", "symbol": "BTCUSDT"}
        }
        
        state_monitor.add_audit_entry(entry)
        
        assert len(state_monitor.audit_trail) == 1
        assert state_monitor.audit_trail[0]["action"] == "order_placed"
    
    def test_get_audit_trail(self, state_monitor):
        """Test getting audit trail."""
        # Add some audit entries
        entries = [
            {"action": "order_placed", "timestamp": datetime.now(timezone.utc)},
            {"action": "order_cancelled", "timestamp": datetime.now(timezone.utc)}
        ]
        
        for entry in entries:
            state_monitor.add_audit_entry(entry)
        
        audit_trail = state_monitor.get_audit_trail()
        assert len(audit_trail) == 2
    
    def test_clear_audit_trail(self, state_monitor):
        """Test clearing audit trail."""
        # Add some audit entries
        entry = {"action": "test", "timestamp": datetime.now(timezone.utc)}
        state_monitor.add_audit_entry(entry)
        
        # Clear the trail
        state_monitor.clear_audit_trail()
        
        assert len(state_monitor.audit_trail) == 0
    
    @pytest.mark.asyncio
    async def test_begin_transaction(self, state_monitor):
        """Test beginning a transaction."""
        transaction_id = await state_monitor.begin_transaction("test_transaction")
        
        assert transaction_id is not None
        assert transaction_id in state_monitor.transactions
    
    @pytest.mark.asyncio
    async def test_commit_transaction(self, state_monitor):
        """Test committing a transaction."""
        transaction_id = await state_monitor.begin_transaction("test_transaction")
        
        result = await state_monitor.commit_transaction(transaction_id)
        assert result is True
        assert transaction_id not in state_monitor.transactions
    
    @pytest.mark.asyncio
    async def test_rollback_transaction(self, state_monitor):
        """Test rolling back a transaction."""
        transaction_id = await state_monitor.begin_transaction("test_transaction")
        
        result = await state_monitor.rollback_transaction(transaction_id)
        assert result is True
        assert transaction_id not in state_monitor.transactions
    
    def test_get_validation_summary(self, state_monitor):
        """Test getting validation summary."""
        summary = state_monitor.get_validation_summary()
        
        assert "total_validations" in summary
        assert "successful_validations" in summary
        assert "failed_validations" in summary
        assert "last_validation" in summary
    
    def test_get_validation_history(self, state_monitor):
        """Test getting validation history."""
        history = state_monitor.get_validation_history()
        
        assert isinstance(history, list)
        # Initially empty
        assert len(history) == 0
    
    @pytest.mark.asyncio
    async def test_state_monitor_integration(self, state_monitor):
        """Test state monitor integration scenarios."""
        # Test full validation and reconciliation workflow
        state_data = {
            "orders": {"count": 5, "total_value": 1000.0},
            "positions": {"count": 3, "total_value": 500.0}
        }
        
        # Validate state
        validation_result = await state_monitor.validate_state_consistency(state_data)
        assert isinstance(validation_result, StateValidationResult)
        
        # Add audit entry
        audit_entry = {
            "action": "state_validation",
            "timestamp": datetime.now(timezone.utc),
            "details": {"validation_result": validation_result.is_valid}
        }
        state_monitor.add_audit_entry(audit_entry)
        
        # Check audit trail
        audit_trail = state_monitor.get_audit_trail()
        assert len(audit_trail) == 1
        
        # Get summary
        summary = state_monitor.get_validation_summary()
        assert "total_validations" in summary 