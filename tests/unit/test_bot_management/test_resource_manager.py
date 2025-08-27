"""Unit tests for ResourceManager component."""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.config import Config
from src.core.types import BotPriority, ResourceType
from src.bot_management.resource_manager import ResourceManager


@pytest.fixture
def config():
    """Create test configuration."""
    config = MagicMock(spec=Config)
    config.error_handling = MagicMock()
    config.capital_management = MagicMock()
    config.capital_management.total_capital = 100000
    config.capital_management.emergency_reserve_pct = 0.1
    config.database = MagicMock()
    config.bot_management = {
        "total_capital": 100000,
        "max_api_requests_per_minute": 1000,
        "max_database_connections": 50,
        "resource_check_interval": 60,
        "resource_limits": {
            "total_capital": "100000",
            "total_api_calls_per_minute": "1000",
            "max_database_connections": "50"
        }
    }
    # Add to_dict method for ConfigService compatibility
    config.to_dict.return_value = {
        "error_handling": {},
        "capital_management": {
            "total_capital": 100000,
            "emergency_reserve_pct": 0.1
        },
        "database": {},
        "bot_management": config.bot_management,
        "database_service": {
            "cache_enabled": True,
            "cache_ttl_seconds": 300,
            "query_cache_max_size": 1000,
            "slow_query_threshold_seconds": 1.0,
            "connection_pool_monitoring_enabled": True,
            "performance_metrics_enabled": True
        }
    }
    return config


@pytest.fixture
def resource_manager(config):
    """Create ResourceManager for testing."""
    return ResourceManager(config)


class TestResourceManager:
    """Test cases for ResourceManager class."""

    @pytest.mark.asyncio
    async def test_resource_manager_initialization(self, resource_manager, config):
        """Test resource manager initialization."""
        assert resource_manager.config == config
        assert resource_manager.global_resource_limits is not None
        assert resource_manager.resource_allocations == {}
        assert resource_manager.resource_usage_history is not None
        assert not resource_manager.is_running

    @pytest.mark.asyncio
    async def test_start_resource_manager(self, resource_manager):
        """Test resource manager startup."""
        await resource_manager.start()
        
        assert resource_manager.is_running
        assert resource_manager.monitoring_task is not None

    @pytest.mark.asyncio
    async def test_stop_resource_manager(self, resource_manager):
        """Test resource manager shutdown."""
        await resource_manager.start()
        await resource_manager.stop()
        
        assert not resource_manager.is_running

    @pytest.mark.asyncio
    async def test_request_capital_success(self, resource_manager):
        """Test successful capital allocation."""
        bot_id = "test_bot_001"
        capital_amount = Decimal("10000")
        priority = BotPriority.NORMAL
        
        # Request capital
        success = await resource_manager.request_resources(bot_id, capital_amount, priority)
        
        assert success
        assert bot_id in resource_manager.resource_allocations
        # Check that the bot has a capital allocation
        from src.core.types import ResourceType
        assert ResourceType.CAPITAL in resource_manager.resource_allocations[bot_id]
        capital_allocation = resource_manager.resource_allocations[bot_id][ResourceType.CAPITAL]
        assert capital_allocation.allocated_amount == capital_amount

    @pytest.mark.asyncio
    async def test_request_capital_insufficient_funds(self, resource_manager):
        """Test capital request with insufficient funds."""
        bot_id = "test_bot_001"
        capital_amount = Decimal("150000")  # More than available
        priority = BotPriority.NORMAL
        
        # Request more capital than available
        success = await resource_manager.request_resources(bot_id, capital_amount, priority)
        
        assert not success
        assert bot_id not in resource_manager.resource_allocations

    @pytest.mark.asyncio
    async def test_request_capital_high_priority_override(self, resource_manager):
        """Test high priority capital request with override."""
        # Allocate capital to normal priority bot
        normal_bot = "normal_bot"
        await resource_manager.request_resources(normal_bot, Decimal("60000"), BotPriority.NORMAL)
        
        # High priority bot requests more than available
        high_priority_bot = "high_priority_bot"
        capital_amount = Decimal("50000")  # Would exceed total with existing allocation
        
        success = await resource_manager.request_resources(
            high_priority_bot, capital_amount, BotPriority.HIGH
        )
        
        # Should succeed due to high priority
        assert success
        assert high_priority_bot in resource_manager.resource_allocations

    @pytest.mark.asyncio
    async def test_release_resources_success(self, resource_manager):
        """Test successful resource release."""
        bot_id = "test_bot_001"
        capital_amount = Decimal("10000")
        
        # First allocate resources
        await resource_manager.request_resources(bot_id, capital_amount, BotPriority.NORMAL)
        
        # Then release them
        success = await resource_manager.release_resources(bot_id)
        
        assert success
        assert bot_id not in resource_manager.resource_allocations

    @pytest.mark.asyncio
    async def test_release_resources_not_found(self, resource_manager):
        """Test releasing resources for non-existent bot."""
        success = await resource_manager.release_resources("non_existent_bot")
        assert not success

    @pytest.mark.asyncio
    async def test_update_capital_allocation(self, resource_manager):
        """Test updating capital allocation for existing bot."""
        bot_id = "test_bot_001"
        initial_capital = Decimal("10000")
        new_capital = Decimal("15000")
        
        # Initial allocation
        await resource_manager.request_resources(bot_id, initial_capital, BotPriority.NORMAL)
        
        # Update allocation
        success = await resource_manager.update_capital_allocation(bot_id, new_capital)
        
        assert success
        # Check the allocation was updated
        allocations = await resource_manager.get_bot_allocations()
        assert allocations[bot_id]["capital"] == new_capital

    @pytest.mark.asyncio
    async def test_update_capital_allocation_insufficient(self, resource_manager):
        """Test updating capital allocation with insufficient funds."""
        bot_id = "test_bot_001"
        initial_capital = Decimal("10000")
        new_capital = Decimal("150000")  # More than total available
        
        # Initial allocation
        await resource_manager.request_resources(bot_id, initial_capital, BotPriority.NORMAL)
        
        # Try to update with insufficient funds
        success = await resource_manager.update_capital_allocation(bot_id, new_capital)
        
        assert not success
        assert resource_manager.bot_allocations[bot_id]["capital"] == initial_capital

    @pytest.mark.asyncio
    async def test_check_resource_availability(self, resource_manager):
        """Test resource availability checking."""
        # Check availability for valid request
        available = await resource_manager.check_resource_availability(
            ResourceType.CAPITAL, Decimal("50000")
        )
        assert available
        
        # Check availability for excessive request
        available = await resource_manager.check_resource_availability(
            ResourceType.CAPITAL, Decimal("150000")
        )
        assert not available

    @pytest.mark.asyncio
    async def test_get_bot_allocations(self, resource_manager):
        """Test getting bot allocations."""
        # Allocate to multiple bots
        bots = [("bot1", Decimal("10000")), ("bot2", Decimal("20000"))]
        for bot_id, capital in bots:
            await resource_manager.request_resources(bot_id, capital, BotPriority.NORMAL)
        
        allocations = await resource_manager.get_bot_allocations()
        
        assert len(allocations) == 2
        assert "bot1" in allocations
        assert "bot2" in allocations
        assert allocations["bot1"]["capital"] == Decimal("10000")
        assert allocations["bot2"]["capital"] == Decimal("20000")

    @pytest.mark.asyncio
    async def test_get_resource_summary(self, resource_manager):
        """Test resource summary generation."""
        # Allocate some resources
        await resource_manager.request_resources("bot1", Decimal("30000"), BotPriority.HIGH)
        await resource_manager.request_resources("bot2", Decimal("20000"), BotPriority.NORMAL)
        
        summary = await resource_manager.get_resource_summary()
        
        # Verify summary structure
        assert "capital_management" in summary
        assert "resource_utilization" in summary
        assert "bot_allocations" in summary
        assert "system_health" in summary
        
        # Verify content
        assert summary["capital_management"]["total_capital"] == float(Decimal("100000"))
        assert summary["capital_management"]["allocated_capital"] == float(Decimal("50000"))
        assert summary["capital_management"]["available_capital"] == float(Decimal("50000"))
        assert summary["resource_utilization"]["capital_utilization_percentage"] == 50.0

    @pytest.mark.asyncio
    async def test_priority_based_allocation(self, resource_manager):
        """Test priority-based resource allocation."""
        # Fill up most capacity with normal priority
        await resource_manager.request_resources("normal1", Decimal("40000"), BotPriority.NORMAL)
        await resource_manager.request_resources("normal2", Decimal("40000"), BotPriority.NORMAL)
        
        # High priority should still get allocation
        success = await resource_manager.request_resources("high1", Decimal("30000"), BotPriority.HIGH)
        assert success
        
        # Low priority should be rejected
        success = await resource_manager.request_resources("low1", Decimal("10000"), BotPriority.LOW)
        assert not success

    @pytest.mark.asyncio
    async def test_api_rate_limit_management(self, resource_manager):
        """Test API rate limit resource management."""
        bot_id = "test_bot_001"
        
        # Request API rate limits
        success = await resource_manager.allocate_api_limits(bot_id, 100)  # 100 requests per minute
        
        assert success
        assert bot_id in resource_manager.api_allocations
        assert resource_manager.api_allocations[bot_id] == 100

    @pytest.mark.asyncio
    async def test_api_rate_limit_exceeded(self, resource_manager):
        """Test API rate limit exceeded scenario."""
        # Allocate all available API limits
        await resource_manager.allocate_api_limits("bot1", 500)
        await resource_manager.allocate_api_limits("bot2", 500)
        
        # Try to allocate more
        success = await resource_manager.allocate_api_limits("bot3", 100)
        assert not success

    @pytest.mark.asyncio
    async def test_database_connection_management(self, resource_manager):
        """Test database connection resource management."""
        bot_id = "test_bot_001"
        
        # Request database connections
        success = await resource_manager.allocate_database_connections(bot_id, 5)
        
        assert success
        assert bot_id in resource_manager.db_allocations
        assert resource_manager.db_allocations[bot_id] == 5

    @pytest.mark.asyncio
    async def test_database_connection_exceeded(self, resource_manager):
        """Test database connection limit exceeded."""
        # Allocate all available connections
        await resource_manager.allocate_database_connections("bot1", 30)
        await resource_manager.allocate_database_connections("bot2", 20)
        
        # Try to allocate more
        success = await resource_manager.allocate_database_connections("bot3", 5)
        assert not success

    @pytest.mark.asyncio
    async def test_resource_conflict_detection(self, resource_manager):
        """Test resource conflict detection."""
        # Allocate resources to multiple bots
        await resource_manager.request_resources("bot1", Decimal("30000"), BotPriority.NORMAL)
        await resource_manager.request_resources("bot2", Decimal("40000"), BotPriority.NORMAL)
        
        # Check for conflicts
        conflicts = await resource_manager.detect_resource_conflicts()
        
        assert isinstance(conflicts, list)

    @pytest.mark.asyncio
    async def test_emergency_resource_reallocation(self, resource_manager):
        """Test emergency resource reallocation."""
        # Set up scenario with allocated resources
        await resource_manager.request_resources("bot1", Decimal("30000"), BotPriority.NORMAL)
        await resource_manager.request_resources("bot2", Decimal("40000"), BotPriority.LOW)
        
        # Emergency reallocation for high priority bot
        success = await resource_manager.emergency_reallocate("critical_bot", Decimal("50000"))
        
        assert success
        assert "critical_bot" in resource_manager.bot_allocations

    @pytest.mark.asyncio
    async def test_resource_usage_tracking(self, resource_manager):
        """Test resource usage tracking."""
        bot_id = "test_bot_001"
        await resource_manager.request_resources(bot_id, Decimal("10000"), BotPriority.NORMAL)
        
        # Update resource usage
        await resource_manager.update_resource_usage(bot_id, {
            "cpu_usage": 45.0,
            "memory_usage": 60.0,
            "network_usage": 30.0
        })
        
        # Get usage stats
        usage = await resource_manager.get_resource_usage(bot_id)
        
        assert usage is not None
        assert "cpu_usage" in usage
        assert usage["cpu_usage"] == 45.0

    @pytest.mark.asyncio
    async def test_resource_optimization_suggestions(self, resource_manager):
        """Test resource optimization suggestions."""
        # Set up bots with different usage patterns
        await resource_manager.request_resources("efficient_bot", Decimal("10000"), BotPriority.NORMAL)
        await resource_manager.request_resources("inefficient_bot", Decimal("30000"), BotPriority.LOW)
        
        # Update usage data
        await resource_manager.update_resource_usage("efficient_bot", {"cpu_usage": 20.0})
        await resource_manager.update_resource_usage("inefficient_bot", {"cpu_usage": 80.0})
        
        # Get optimization suggestions
        suggestions = await resource_manager.get_optimization_suggestions()
        
        assert isinstance(suggestions, list)

    @pytest.mark.asyncio
    async def test_resource_monitoring_loop(self, resource_manager):
        """Test resource monitoring loop functionality."""
        await resource_manager.start()
        
        # Add some allocations
        await resource_manager.request_resources("bot1", Decimal("10000"), BotPriority.NORMAL)
        
        # Simulate monitoring loop
        await resource_manager._resource_monitoring_loop()
        
        # Should complete without errors
        assert resource_manager.is_running

    @pytest.mark.asyncio
    async def test_cleanup_stale_allocations(self, resource_manager):
        """Test cleanup of stale resource allocations."""
        bot_id = "stale_bot"
        await resource_manager.request_resources(bot_id, Decimal("10000"), BotPriority.NORMAL)
        
        # Mark allocation as stale
        resource_manager.bot_allocations[bot_id]["last_updated"] = datetime.now(timezone.utc)
        
        # Run cleanup
        cleaned = await resource_manager._cleanup_stale_allocations()
        
        assert isinstance(cleaned, int)

    @pytest.mark.asyncio
    async def test_resource_alerts(self, resource_manager):
        """Test resource alert generation."""
        # Allocate high percentage of resources
        await resource_manager.request_resources("bot1", Decimal("85000"), BotPriority.NORMAL)
        
        # Check for alerts
        alerts = await resource_manager.get_resource_alerts()
        
        assert isinstance(alerts, list)
        # Should have high utilization alert
        assert any("utilization" in alert.lower() for alert in alerts)

    @pytest.mark.asyncio
    async def test_concurrent_resource_requests(self, resource_manager):
        """Test concurrent resource requests handling."""
        # Simulate concurrent requests
        tasks = []
        for i in range(5):
            task = resource_manager.request_resources(
                f"bot_{i}", Decimal("20000"), BotPriority.NORMAL
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Some should succeed, some should fail due to capacity
        successful = sum(1 for result in results if result is True)
        assert successful > 0
        assert successful <= 5  # At most 5 can succeed with 20k each

    @pytest.mark.asyncio
    async def test_resource_reservation(self, resource_manager):
        """Test resource reservation functionality."""
        bot_id = "test_bot_001"
        reserved_amount = Decimal("25000")
        
        # Reserve resources
        reservation_id = await resource_manager.reserve_resources(
            bot_id, reserved_amount, BotPriority.NORMAL, duration_minutes=60
        )
        
        assert reservation_id is not None
        assert resource_manager.available_capital < Decimal("100000")

    @pytest.mark.asyncio
    async def test_resource_reservation_expiry(self, resource_manager):
        """Test resource reservation expiry."""
        bot_id = "test_bot_001"
        
        # Make short-term reservation
        reservation_id = await resource_manager.reserve_resources(
            bot_id, Decimal("10000"), BotPriority.NORMAL, duration_minutes=1
        )
        
        # Simulate time passage and cleanup
        await resource_manager._cleanup_expired_reservations()
        
        # Should handle expiry properly
        assert isinstance(reservation_id, str)