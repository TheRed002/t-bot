"""
Tests for simplified state consistency module.
"""

import pytest

from src.core.exceptions import StateConsistencyError
from src.state.consistency import (
    process_state_change,
    raise_state_error,
    validate_state_data,
)


@pytest.mark.unit
class TestValidateStateData:
    """Test validate_state_data function."""

    def test_validate_none_data(self):
        """Test validation with None data."""
        result = validate_state_data("test", None)
        
        assert result["is_valid"] is False
        assert "Data cannot be None" in result["errors"]
        assert result["data_type"] == "test"

    def test_validate_trade_state_valid(self):
        """Test validation with valid trade state."""
        trade_data = {
            "trade_id": "12345",
            "symbol": "BTCUSD",
            "side": "BUY"
        }
        
        result = validate_state_data("trade_state", trade_data)
        
        assert result["is_valid"] is True
        assert result["errors"] == []
        assert result["data_type"] == "trade_state"

    def test_validate_trade_state_missing_fields(self):
        """Test validation with missing required fields."""
        trade_data = {
            "trade_id": "12345"
            # Missing symbol and side
        }
        
        result = validate_state_data("trade_state", trade_data)
        
        assert result["is_valid"] is False
        assert len(result["errors"]) == 2
        assert "Missing required field: symbol" in result["errors"]
        assert "Missing required field: side" in result["errors"]

    def test_validate_other_data_type(self):
        """Test validation with non-trade state data."""
        data = {"some": "data"}
        
        result = validate_state_data("other", data)
        
        assert result["is_valid"] is True
        assert result["errors"] == []


@pytest.mark.unit 
class TestRaiseStateError:
    """Test raise_state_error function."""

    def test_raise_state_error_basic(self):
        """Test basic error raising."""
        with pytest.raises(StateConsistencyError) as exc_info:
            raise_state_error("Test error")
        
        assert "Test error" in str(exc_info.value)
        assert exc_info.value.context["error_source"] == "state_consistency"

    def test_raise_state_error_with_context(self):
        """Test error raising with context."""
        context = {"extra": "info"}
        
        with pytest.raises(StateConsistencyError) as exc_info:
            raise_state_error("Test error", context)
        
        assert exc_info.value.context["extra"] == "info"
        assert exc_info.value.context["error_source"] == "state_consistency"


@pytest.mark.unit
class TestProcessStateChange:
    """Test process_state_change function."""

    @pytest.mark.asyncio
    async def test_process_sync_function(self):
        """Test processing with synchronous function."""
        def sync_processor(data):
            return f"processed_{data}"
        
        result = await process_state_change("test", sync_processor)
        assert result == "processed_test"

    @pytest.mark.asyncio
    async def test_process_async_function(self):
        """Test processing with asynchronous function."""
        async def async_processor(data):
            return f"async_processed_{data}"
        
        result = await process_state_change("test", async_processor)
        assert result == "async_processed_test"

    @pytest.mark.asyncio
    async def test_process_non_callable(self):
        """Test processing with non-callable."""
        with pytest.raises(StateConsistencyError) as exc_info:
            await process_state_change("test", "not_callable")
        
        assert "Processor must be callable" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_process_error_handling(self):
        """Test error handling in processing."""
        def error_processor(data):
            raise ValueError("Processing error")
        
        with pytest.raises(StateConsistencyError) as exc_info:
            await process_state_change("test", error_processor)
        
        assert "Processing failed" in str(exc_info.value)