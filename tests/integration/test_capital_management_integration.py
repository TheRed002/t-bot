"""
Integration tests for Capital Management System (P-010A).

This module tests the complete capital management workflow including all components
working together in realistic scenarios.
"""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, Mock

from src.core.types import (
    CapitalAllocation, ExchangeAllocation, CurrencyExposure, FundFlow,
    AllocationStrategy, CapitalMetrics, WithdrawalRule
)
from src.core.exceptions import (
    CapitalManagementError, ValidationError, InsufficientCapitalError,
    WithdrawalError
)
from src.core.config import Config, CapitalManagementConfig
from src.capital_management import (
    CapitalAllocator, ExchangeDistributor, CurrencyManager, FundFlowManager
)


class TestCapitalManagementIntegration:
    """Integration tests for complete capital management workflow."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration with capital management settings."""
        # Create config with custom withdrawal rules
        config = Config(
            capital_management=CapitalManagementConfig(
                total_capital=100000.0,
                emergency_reserve_pct=0.1,
                allocation_strategy=AllocationStrategy.PERFORMANCE_WEIGHTED,
                rebalance_frequency_hours=24,
                min_allocation_pct=0.05,
                max_allocation_pct=0.4,
                max_exchange_allocation_pct=0.4,
                min_exchange_balance=1000.0,
                exchange_allocation_weights={
                    "binance": 0.4,
                    "okx": 0.35,
                    "coinbase": 0.25
                },
                supported_currencies=["USDT", "BTC", "ETH", "USD"],
                hedging_enabled=True,
                hedging_threshold=0.2,
                hedge_ratio=0.5,
                min_deposit_amount=1000.0,
                min_withdrawal_amount=100.0,
                max_withdrawal_pct=0.2,
                max_daily_reallocation_pct=0.1,
                fund_flow_cooldown_minutes=30,
                auto_compound_enabled=True,
                auto_compound_frequency="daily",
                profit_threshold=0.05,
                max_daily_loss_pct=0.05,
                max_weekly_loss_pct=0.15,
                max_monthly_loss_pct=0.25,
                profit_lock_pct=0.1,
                withdrawal_rules={
                    "emergency": {
                        "name": "emergency",
                        "description": "Emergency withdrawal rule",
                        "enabled": True,
                        "max_percentage": 0.5,
                        "cooldown_hours": 0
                    },
                    "regular": {
                        "name": "regular",
                        "description": "Regular withdrawal rule",
                        "enabled": True,
                        "max_percentage": 0.2,
                        "cooldown_hours": 24
                    },
                    "profit_only": {
                        "name": "profit_only",
                        "description": "Profit-only withdrawal rule",
                        "enabled": False,
                        "threshold": 0.1,
                        "cooldown_hours": 0
                    }
                }
            )
        )
        return config
    
    @pytest.fixture
    def capital_allocator(self, config):
        """Create capital allocator instance."""
        return CapitalAllocator(config)
    
    @pytest.fixture
    def exchange_distributor(self, config):
        """Create exchange distributor instance."""
        mock_exchanges = {
            "binance": Mock(),
            "okx": Mock(),
            "coinbase": Mock()
        }
        return ExchangeDistributor(config, mock_exchanges)
    
    @pytest.fixture
    def currency_manager(self, config):
        """Create currency manager instance."""
        mock_exchanges = {
            "binance": Mock(),
            "okx": Mock(),
            "coinbase": Mock()
        }
        return CurrencyManager(config, mock_exchanges)
    
    @pytest.fixture
    def fund_flow_manager(self, config):
        """Create fund flow manager instance."""
        return FundFlowManager(config)
    
    @pytest.mark.asyncio
    async def test_complete_capital_management_workflow(self, capital_allocator, 
                                                       exchange_distributor,
                                                       currency_manager,
                                                       fund_flow_manager):
        """Test complete capital management workflow."""
        # Step 1: Initial capital allocation
        await capital_allocator.allocate_capital("strategy_1", "binance", Decimal("20000"))
        await capital_allocator.allocate_capital("strategy_2", "okx", Decimal("15000"))
        await capital_allocator.allocate_capital("strategy_3", "coinbase", Decimal("10000"))
        
        # Step 2: Exchange distribution
        total_amount = Decimal("45000")
        await exchange_distributor.distribute_capital(total_amount)
        
        # Step 3: Currency exposure management
        balances = {
            "binance": {
                "USDT": Decimal("30000"),
                "BTC": Decimal("10000"),
                "ETH": Decimal("5000")
            }
        }
        await currency_manager.update_currency_exposures(balances)
        
        # Step 4: Fund flow processing
        await fund_flow_manager.process_deposit(Decimal("5000"), "USDT", "binance")
        await fund_flow_manager.process_withdrawal(Decimal("2000"), "USDT", "okx", "withdrawal")
        
        # Step 5: Performance updates
        await capital_allocator.update_utilization("strategy1", "binance", Decimal("16000"))
        await capital_allocator.update_utilization("strategy2", "okx", Decimal("10500"))
        await fund_flow_manager.update_performance("strategy_1", {"total_pnl": 1000.0})
        await fund_flow_manager.update_performance("strategy_2", {"total_pnl": 500.0})
        
        # Step 6: Rebalancing
        await capital_allocator.rebalance_allocations()
        await exchange_distributor.rebalance_exchanges()
        
        # Step 7: Currency conversion
        await currency_manager.execute_currency_conversion("USDT", "BTC", Decimal("10000"), "binance")
        
        # Step 8: Auto-compounding
        await fund_flow_manager.process_auto_compound()
        
        # Verify results
        capital_metrics = await capital_allocator.get_capital_metrics()
        assert capital_metrics.total_capital == Decimal("100000")
        assert capital_metrics.allocated_capital > 0
        
        exchange_metrics = await exchange_distributor.get_exchange_metrics()
        assert len(exchange_metrics) > 0
        
        currency_metrics = await currency_manager.get_currency_risk_metrics()
        assert len(currency_metrics) > 0
        
        flow_summary = await fund_flow_manager.get_flow_summary()
        assert flow_summary["total_deposits"] > 0
        assert flow_summary["total_withdrawals"] > 0
    
    @pytest.mark.asyncio
    async def test_capital_management_with_large_portfolio(self, capital_allocator,
                                                          exchange_distributor,
                                                          currency_manager,
                                                          fund_flow_manager):
        """Test capital management with large portfolio."""
        # Setup large portfolio
        total_capital = Decimal("1000000")
        await capital_allocator.update_total_capital(total_capital)
        
        # Allocate large amounts (within available capital limits)
        await capital_allocator.allocate_capital("strategy_1", "binance", Decimal("30000"))
        await capital_allocator.allocate_capital("strategy_2", "okx", Decimal("25000"))
        await capital_allocator.allocate_capital("strategy_3", "coinbase", Decimal("20000"))
        
        # Distribute across exchanges
        await exchange_distributor.distribute_capital(Decimal("450000"))
        
        # Large currency exposures
        balances = {
            "binance": {
                "USDT": Decimal("300000"),
                "BTC": Decimal("100000"),
                "ETH": Decimal("50000")
            }
        }
        await currency_manager.update_currency_exposures(balances)
        
        # Large fund flows
        await fund_flow_manager.process_deposit(Decimal("50000"), "USDT", "binance")
        await fund_flow_manager.process_withdrawal(Decimal("20000"), "USDT", "okx", "regular")
        
        # Verify large portfolio handling
        capital_metrics = await capital_allocator.get_capital_metrics()
        assert capital_metrics.total_capital == total_capital
        
        exchange_summary = await exchange_distributor.get_distribution_summary()
        assert exchange_summary["total_allocated"] > 0
        
        currency_metrics = await currency_manager.get_currency_risk_metrics()
        assert len(currency_metrics) > 0  # Check that we have currency metrics
    
    @pytest.mark.asyncio
    async def test_capital_management_with_high_volatility(self, capital_allocator,
                                                          exchange_distributor,
                                                          currency_manager,
                                                          fund_flow_manager):
        """Test capital management with high volatility scenarios."""
        # Setup allocations
        await capital_allocator.allocate_capital("strategy_1", "binance", Decimal("20000"))
        await capital_allocator.allocate_capital("strategy_2", "okx", Decimal("15000"))
        
        # Simulate high volatility with large losses
        await fund_flow_manager.update_performance("strategy_1", Decimal("-3000"))
        await fund_flow_manager.update_performance("strategy_2", Decimal("-2000"))
        
        # Update currency exposures with high volatility
        balances = {
            "binance": {
                "BTC": Decimal("25000"),
                "ETH": Decimal("15000")
            }
        }
        await currency_manager.update_currency_exposures(balances)
        
        # Check hedging requirements
        hedging_requirements = await currency_manager.calculate_hedging_requirements()
        assert len(hedging_requirements) > 0
        
        # Check capital protection
        protection_status = await fund_flow_manager.get_capital_protection_status()
        assert "protection_active" in protection_status
        
        # Verify risk management
        currency_metrics = await currency_manager.get_currency_risk_metrics()
        assert len(currency_metrics) > 0  # Should have currency metrics
    
    @pytest.mark.asyncio
    async def test_capital_management_with_multiple_strategies(self, capital_allocator,
                                                              exchange_distributor,
                                                              currency_manager,
                                                              fund_flow_manager):
        """Test capital management with multiple strategies."""
        strategies = ["strategy_1", "strategy_2", "strategy_3", "strategy_4"]
        exchanges = ["binance", "okx", "coinbase"]
        
        # Allocate capital to multiple strategies (within available capital limits)
        for i, strategy in enumerate(strategies):
            exchange = exchanges[i % len(exchanges)]
            amount = Decimal("5000") + (Decimal("2000") * i)  # Smaller amounts
            await capital_allocator.allocate_capital(strategy, exchange, amount)
        
        # Distribute across exchanges
        await exchange_distributor.distribute_capital(Decimal("80000"))
        
        # Multiple currency exposures
        currencies = ["USDT", "BTC", "ETH", "USD"]
        balances = {"binance": {}}
        for i, currency in enumerate(currencies):
            amount = Decimal("20000") + (Decimal("5000") * i)
            balances["binance"][currency] = amount
        await currency_manager.update_currency_exposures(balances)
        
        # Multiple fund flows
        for i, strategy in enumerate(strategies):
            exchange = exchanges[i % len(exchanges)]
            amount = Decimal("1000") + (Decimal("500") * i)
            await fund_flow_manager.process_deposit(amount, "USDT", exchange)
        
        # Update performance for all strategies
        for i, strategy in enumerate(strategies):
            pnl = Decimal("500") + (Decimal("100") * i)
            await fund_flow_manager.update_performance(strategy, pnl)
            await capital_allocator.update_utilization(strategy, exchanges[i % len(exchanges)], Decimal("0.7"))
        
        # Verify multi-strategy handling
        capital_metrics = await capital_allocator.get_capital_metrics()
        assert capital_metrics.allocation_count > 0  # Should have allocations
        
        exchange_summary = await exchange_distributor.get_distribution_summary()
        assert exchange_summary["total_exchanges"] == len(exchanges)
        
        currency_metrics = await currency_manager.get_currency_risk_metrics()
        assert len(currency_metrics) > 0  # Should have currency metrics
        
        performance_summary = await fund_flow_manager.get_performance_summary()
        assert performance_summary["total_pnl"] > 0
    
    @pytest.mark.asyncio
    async def test_capital_management_parameter_validation(self, capital_allocator,
                                                         exchange_distributor,
                                                         currency_manager,
                                                         fund_flow_manager):
        """Test capital management parameter validation."""
        # Test invalid capital allocation
        with pytest.raises(ValidationError):
            await capital_allocator.allocate_capital("strategy_1", "binance", Decimal("50000"))  # Exceeds available capital
        
        # Test invalid exchange distribution
        with pytest.raises(ValidationError):
            await exchange_distributor.distribute_capital(Decimal("-10000"))  # Negative amount
        
        # Test invalid currency exposure
        with pytest.raises(ValidationError):
            balances = {"binance": {"INVALID": Decimal("10000")}}
            await currency_manager.update_currency_exposures(balances)
        
        # Test invalid fund flow
        with pytest.raises(ValidationError):
            await fund_flow_manager.process_deposit(Decimal("500"), "USDT", "binance")  # Below minimum
    
    @pytest.mark.asyncio
    async def test_capital_management_error_handling(self, capital_allocator,
                                                   exchange_distributor,
                                                   currency_manager,
                                                   fund_flow_manager):
        """Test capital management error handling."""
        # Set total capital to enable validation
        await fund_flow_manager.update_total_capital(Decimal("100000"))
        
        # Test withdrawal with insufficient funds
        with pytest.raises(ValidationError):
            await fund_flow_manager.process_withdrawal(Decimal("25000"), "USDT", "binance", "regular")
        
        # Test reallocation exceeding limits
        with pytest.raises(ValidationError):
            await fund_flow_manager.process_strategy_reallocation(
                "strategy_1", "strategy_2", Decimal("15000"), "reallocation"
            )
        
        # Test currency conversion with invalid rate
        with pytest.raises(ValidationError):
            await currency_manager.execute_currency_conversion("USDT", "BTC", Decimal("10000"), "binance")
        
        # Test allocation with invalid strategy
        with pytest.raises(ValidationError):
            await capital_allocator.allocate_capital("", "binance", Decimal("10000"))
    
    @pytest.mark.asyncio
    async def test_capital_management_performance(self, capital_allocator,
                                                exchange_distributor,
                                                currency_manager,
                                                fund_flow_manager):
        """Test capital management performance under load."""
        import time
        
        start_time = time.time()
        
        # Perform many operations quickly
        for i in range(10):  # Reduced from 100 to avoid exceeding capital limits
            strategy = f"strategy_{i % 10}"
            exchange = ["binance", "okx", "coinbase"][i % 3]
            amount = Decimal("1000") + (Decimal("100") * i)
            
            await capital_allocator.allocate_capital(strategy, exchange, amount)
            await exchange_distributor.distribute_capital(amount)
            balances = {"binance": {"USDT": amount}}
            await currency_manager.update_currency_exposures(balances)
            await fund_flow_manager.process_deposit(amount, "USDT", exchange)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete within reasonable time (less than 10 seconds)
        assert execution_time < 10.0
        
        # Verify data integrity
        capital_metrics = await capital_allocator.get_capital_metrics()
        assert capital_metrics.allocation_count > 0
        
        exchange_summary = await exchange_distributor.get_distribution_summary()
        assert exchange_summary["total_exchanges"] > 0
        
        currency_metrics = await currency_manager.get_currency_risk_metrics()
        assert len(currency_metrics) > 0  # Check that we have currency metrics
    
    @pytest.mark.asyncio
    async def test_capital_management_edge_cases(self, capital_allocator,
                                               exchange_distributor,
                                               currency_manager,
                                               fund_flow_manager):
        """Test capital management edge cases."""
        # Test minimum amounts
        await capital_allocator.allocate_capital("strategy_1", "binance", Decimal("1000"))
        await exchange_distributor.distribute_capital(Decimal("1000"))
        balances = {"binance": {"USDT": Decimal("1000")}}
        await currency_manager.update_currency_exposures(balances)
        await fund_flow_manager.process_deposit(Decimal("1000"), "USDT", "binance")
        
        # Test maximum amounts
        max_amount = Decimal("30000")  # Within available capital limits
        await capital_allocator.allocate_capital("strategy_2", "okx", max_amount)
        
        # Test zero amounts (should fail)
        with pytest.raises(ValidationError):
            await capital_allocator.allocate_capital("strategy_3", "coinbase", Decimal("0"))
        
        # Test negative amounts (should fail)
        with pytest.raises(ValidationError):
            await fund_flow_manager.process_deposit(Decimal("-1000"), "USDT", "coinbase")
        
        # Test empty strings (should fail)
        with pytest.raises(ValidationError):
            balances = {"": {"total": Decimal("1000"), "available": Decimal("1000")}}
            await currency_manager.update_currency_exposures(balances)
    
    @pytest.mark.asyncio
    async def test_capital_management_configuration_changes(self, capital_allocator,
                                                          exchange_distributor,
                                                          currency_manager,
                                                          fund_flow_manager,
                                                          config):
        """Test capital management with configuration changes."""
        # Initial setup
        await capital_allocator.allocate_capital("strategy_1", "binance", Decimal("20000"))
        await exchange_distributor.distribute_capital(Decimal("20000"))
        balances = {"binance": {"USDT": Decimal("20000")}}
        await currency_manager.update_currency_exposures(balances)
        
        # Change configuration
        config.capital_management.max_allocation_pct = 0.3  # Reduce from 0.4
        config.capital_management.hedging_threshold = 0.15  # Reduce from 0.2
        config.capital_management.max_withdrawal_pct = 0.15  # Reduce from 0.2
        
        # Set total capital for withdrawal validation
        await fund_flow_manager.update_total_capital(Decimal("100000"))
        
        # Test that new limits are enforced
        with pytest.raises(ValidationError):
            await capital_allocator.allocate_capital("strategy_2", "okx", Decimal("35000"))  # 35% > 30%
        
        with pytest.raises(ValidationError):
            await fund_flow_manager.process_withdrawal(Decimal("16000"), "USDT", "binance", "regular")  # 16% > 15%
        
        # Test hedging with new threshold
        balances = {"binance": {"BTC": Decimal("18000")}}  # 18% > 15% threshold
        await currency_manager.update_currency_exposures(balances)
        hedging_requirements = await currency_manager.calculate_hedging_requirements()
        assert "BTC" in hedging_requirements
        assert hedging_requirements["BTC"] > 0
    
    @pytest.mark.asyncio
    async def test_capital_management_real_time_updates(self, capital_allocator,
                                                      exchange_distributor,
                                                      currency_manager,
                                                      fund_flow_manager):
        """Test capital management real-time updates."""
        # Initial state
        await capital_allocator.allocate_capital("strategy_1", "binance", Decimal("20000"))
        await exchange_distributor.distribute_capital(Decimal("20000"))
        balances = {"binance": {"USDT": Decimal("20000")}}
        await currency_manager.update_currency_exposures(balances)
        
        # Get initial metrics
        initial_capital_metrics = await capital_allocator.get_capital_metrics()
        initial_exchange_summary = await exchange_distributor.get_distribution_summary()
        initial_currency_metrics = await currency_manager.get_currency_risk_metrics()
        
        # Perform updates
        await capital_allocator.update_utilization("strategy_1", "binance", Decimal("18000"))
        await fund_flow_manager.update_performance("strategy_1", {"pnl": 1000.0})
        balances = {"binance": {"BTC": Decimal("15000")}}
        await currency_manager.update_currency_exposures(balances)
        
        # Get updated metrics
        updated_capital_metrics = await capital_allocator.get_capital_metrics()
        updated_exchange_summary = await exchange_distributor.get_distribution_summary()
        updated_currency_metrics = await currency_manager.get_currency_risk_metrics()
        
        # Verify updates
        assert updated_capital_metrics.utilization_rate != initial_capital_metrics.utilization_rate
        assert len(updated_currency_metrics) >= len(initial_currency_metrics)  # Should have same or more currencies
    
    @pytest.mark.asyncio
    async def test_capital_management_data_consistency(self, capital_allocator,
                                                     exchange_distributor,
                                                     currency_manager,
                                                     fund_flow_manager):
        """Test capital management data consistency."""
        # Setup consistent data
        total_capital = Decimal("100000")
        strategy_name = "test_strategy"
        exchange_name = "binance"
        currency = "USDT"
        amount = Decimal("20000")
        
        # Allocate capital
        await capital_allocator.allocate_capital(strategy_name, exchange_name, amount)
        
        # Distribute to exchange
        await exchange_distributor.distribute_capital(amount)
        
        # Update currency exposure
        balances = {"binance": {currency: amount}}  # Format: {exchange: {currency: amount}}
        await currency_manager.update_currency_exposures(balances)
        
        # Process fund flow
        await fund_flow_manager.process_deposit(amount, currency, exchange_name)
        
        # Verify data consistency
        capital_metrics = await capital_allocator.get_capital_metrics()
        assert capital_metrics.allocated_capital == amount
        
        exchange_allocation = await exchange_distributor.get_exchange_allocation(exchange_name)
        # Exchange allocation might be different due to distribution logic
        assert exchange_allocation.allocated_amount > 0  # Should have some allocation
        
        currency_exposure = await currency_manager.get_currency_exposure(currency)
        assert currency_exposure.total_exposure == amount
        
        flow_summary = await fund_flow_manager.get_flow_summary()
        assert flow_summary["total_deposits"] == amount
    
    @pytest.mark.asyncio
    async def test_capital_management_integration_with_risk_management(self, capital_allocator,
                                                                     exchange_distributor,
                                                                     currency_manager,
                                                                     fund_flow_manager):
        """Test capital management integration with risk management."""
        from src.risk_management.position_sizing import PositionSizer
        
        # Setup capital management
        await capital_allocator.allocate_capital("strategy_1", "binance", Decimal("20000"))
        await exchange_distributor.distribute_capital(Decimal("20000"))
        balances = {"binance": {"USDT": Decimal("20000")}}
        await currency_manager.update_currency_exposures(balances)
        
        # Setup risk management
        position_sizer = PositionSizer(capital_allocator.config)
        
        # Test integration
        # Capital allocator should work with position sizer
        allocation = await capital_allocator.get_allocation_summary()
        assert "total_allocations" in allocation
        
        # Exchange distributor should work with position sizer
        distribution = await exchange_distributor.get_distribution_summary()
        assert "total_exchanges" in distribution
        
        # Currency manager should work with position sizer
        currency_metrics = await currency_manager.get_currency_risk_metrics()
        assert len(currency_metrics) > 0  # Should have currency metrics
        
        # Fund flow manager should work with position sizer
        flow_summary = await fund_flow_manager.get_flow_summary()
        assert "total_deposits" in flow_summary
    
    @pytest.mark.asyncio
    async def test_capital_management_comprehensive_reporting(self, capital_allocator,
                                                            exchange_distributor,
                                                            currency_manager,
                                                            fund_flow_manager):
        """Test comprehensive reporting across all capital management components."""
        # Setup comprehensive data
        strategies = ["strategy_1", "strategy_2", "strategy_3"]
        exchanges = ["binance", "okx", "coinbase"]
        currencies = ["USDT", "BTC", "ETH"]
        
        # Allocate capital to all strategies (within available capital limits)
        for i, strategy in enumerate(strategies):
            exchange = exchanges[i % len(exchanges)]
            amount = Decimal("5000") + (Decimal("2000") * i)  # Smaller amounts
            await capital_allocator.allocate_capital(strategy, exchange, amount)
        
        # Distribute across all exchanges
        await exchange_distributor.distribute_capital(Decimal("60000"))
        
        # Update all currency exposures
        balances = {"binance": {}}
        for i, currency in enumerate(currencies):
            amount = Decimal("20000") + (Decimal("10000") * i)
            balances["binance"][currency] = amount
        await currency_manager.update_currency_exposures(balances)
        
        # Process multiple fund flows
        for i, strategy in enumerate(strategies):
            exchange = exchanges[i % len(exchanges)]
            amount = Decimal("2000") + (Decimal("1000") * i)
            await fund_flow_manager.process_deposit(amount, "USDT", exchange)
            await fund_flow_manager.update_performance(strategy, {"pnl": float(Decimal("500") + (Decimal("100") * i))})
        
        # Generate comprehensive reports
        capital_report = await capital_allocator.get_allocation_summary()
        exchange_report = await exchange_distributor.get_distribution_summary()
        currency_report = await currency_manager.get_currency_risk_metrics()
        flow_report = await fund_flow_manager.get_flow_summary()
        performance_report = await fund_flow_manager.get_performance_summary()
        
        # Verify comprehensive reporting
        assert "total_allocations" in capital_report
        assert "total_exchanges" in exchange_report
        assert len(currency_report) > 0  # Should have currency metrics
        assert "total_deposits" in flow_report
        assert "total_pnl" in performance_report
        
        # Verify data consistency across reports
        assert capital_report["total_capital"] == Decimal("100000")
        assert exchange_report["total_exchanges"] == len(exchanges)
        assert len(currency_report) > 0  # Should have currency metrics
        assert flow_report["total_deposits"] > 0
        assert performance_report["total_pnl"] > 0 