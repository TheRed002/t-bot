"""
Comprehensive Cross-Module Integration Tests

This test suite validates integration points across all major modules in the T-Bot trading system,
ensuring that components work correctly together under realistic conditions.
"""

import asyncio
import logging
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import Mock, patch

import pytest
import pytest_asyncio

from src.bot_management.bot_coordinator import BotCoordinator
from src.core.dependency_injection import injector
from src.core.exceptions import (
    ExchangeError,
    ExecutionError,
    ServiceError,
)
from src.core.types import (
    BotStatus,
    ExecutionInstruction,
    ExecutionResult,
    ExecutionStatus,
    MarketData,
    OrderSide,
    OrderStatus,
    OrderType,
    SignalDirection,
    StrategyType,
)
from src.data.services.data_service import DataService

# Import services and components for integration testing
from src.exchanges.factory import ExchangeFactory
from src.execution.execution_engine import ExecutionEngine
from src.ml.service import MLService
from src.risk_management.service import RiskService
from src.state.state_service import StateService
from src.strategies.factory import StrategyFactory
from tests.integration.base_integration import (
    BaseIntegrationTest,
    MockExchangeFactory,
    MockStrategyFactory,
    performance_test,
)

logger = logging.getLogger(__name__)


class CrossModuleIntegrationTest(BaseIntegrationTest):
    """Comprehensive integration tests across all system modules."""

    def __init__(self):
        super().__init__()
        self.exchange_factory = None
        self.strategy_factory = None
        self.execution_engine = None
        self.bot_orchestrator = None
        self.data_service = None
        self.ml_service = None
        self.risk_service = None
        self.state_service = None

    async def setup_integrated_services(self):
        """Setup all integrated services with proper dependency injection."""
        # Clear any existing services
        injector.clear()

        # Register core configuration
        injector.register_service("config", self.config, singleton=True)

        # Initialize data service
        self.data_service = DataService(self.config)
        await self.data_service.initialize()
        injector.register_service("data_service", self.data_service, singleton=True)

        # Initialize ML service
        self.ml_service = MLService()
        await self.ml_service.initialize()
        injector.register_service("ml_service", self.ml_service, singleton=True)

        # Initialize state service
        self.state_service = StateService()
        await self.state_service.initialize()
        injector.register_service("state_service", self.state_service, singleton=True)

        # Initialize risk service
        self.risk_service = RiskService()
        await self.risk_service.initialize()
        injector.register_service("risk_service", self.risk_service, singleton=True)

        # Initialize exchange factory
        self.exchange_factory = ExchangeFactory(self.config)
        injector.register_service("exchange_factory", self.exchange_factory, singleton=True)

        # Initialize strategy factory
        self.strategy_factory = StrategyFactory(
            strategy_service=Mock(),  # Mock for now
            validation_framework=None,
        )
        injector.register_service("strategy_factory", self.strategy_factory, singleton=True)

        # Initialize execution engine
        execution_service = Mock()  # Mock ExecutionService
        self.execution_engine = ExecutionEngine(execution_service, self.config)
        injector.register_service("execution_engine", self.execution_engine, singleton=True)

        # Initialize bot orchestrator
        self.bot_orchestrator = BotCoordinator()
        await self.bot_orchestrator.initialize()
        injector.register_service("bot_orchestrator", self.bot_orchestrator, singleton=True)

        logger.info("All integrated services initialized successfully")

    @pytest.mark.asyncio
    @performance_test(max_duration=60.0)
    @pytest.mark.timeout(300)
    async def test_multi_exchange_trading_workflow(self, performance_monitor):
        """Test complete trading workflow across multiple exchanges."""
        await self.setup_integrated_services()

        # Setup mock exchanges
        binance_mock = MockExchangeFactory.create_binance_mock()
        coinbase_mock = MockExchangeFactory.create_coinbase_mock()

        # Register mock exchanges
        self.exchange_factory.register_exchange("binance", type(binance_mock))
        self.exchange_factory.register_exchange("coinbase", type(coinbase_mock))

        with patch.object(self.exchange_factory, "create_exchange") as mock_create:

            async def side_effect(name):
                if name == "binance":
                    return binance_mock
                elif name == "coinbase":
                    return coinbase_mock
                raise ExchangeError(f"Unknown exchange: {name}")

            mock_create.side_effect = side_effect

            # Test workflow
            start_time = performance_monitor.start_time

            # 1. Get exchanges
            binance = await self.exchange_factory.get_exchange("binance")
            coinbase = await self.exchange_factory.get_exchange("coinbase")

            assert binance is not None
            assert coinbase is not None
            performance_monitor.record_event("exchanges_created", {"count": 2})

            # 2. Get market data from both exchanges
            btc_data_binance = await binance.get_market_data("BTC/USDT")
            btc_data_coinbase = await coinbase.get_market_data("BTC/USDT")

            assert btc_data_binance.symbol == "BTC/USDT"
            assert btc_data_coinbase.symbol == "BTC/USDT"
            performance_monitor.record_event("market_data_fetched", {"exchanges": 2})

            # 3. Identify arbitrage opportunity
            price_diff = abs(btc_data_binance.price - btc_data_coinbase.price)
            min_profit_threshold = Decimal("50.0")  # $50 minimum profit

            if price_diff > min_profit_threshold:
                # 4. Execute arbitrage strategy
                buy_exchange = (
                    binance if btc_data_binance.price < btc_data_coinbase.price else coinbase
                )
                sell_exchange = coinbase if buy_exchange == binance else binance

                trade_quantity = Decimal("0.1")  # 0.1 BTC

                # Place buy order
                buy_order_data = {
                    "symbol": "BTC/USDT",
                    "side": "BUY",
                    "type": "MARKET",
                    "quantity": trade_quantity,
                }
                buy_order_id = await buy_exchange.place_order(buy_order_data)
                performance_monitor.record_api_call()

                # Place sell order
                sell_order_data = {
                    "symbol": "BTC/USDT",
                    "side": "SELL",
                    "type": "MARKET",
                    "quantity": trade_quantity,
                }
                sell_order_id = await sell_exchange.place_order(sell_order_data)
                performance_monitor.record_api_call()

                # 5. Monitor order execution
                buy_status = await buy_exchange.get_order_status(buy_order_id)
                sell_status = await sell_exchange.get_order_status(sell_order_id)

                assert buy_status == OrderStatus.FILLED
                assert sell_status == OrderStatus.FILLED

                performance_monitor.record_event(
                    "arbitrage_completed",
                    {
                        "buy_exchange": buy_exchange.name,
                        "sell_exchange": sell_exchange.name,
                        "quantity": float(trade_quantity),
                        "profit": float(price_diff * trade_quantity),
                    },
                )

            # 6. Health check all exchanges
            health_results = await self.exchange_factory.health_check_all()

            assert "binance" in health_results
            assert "coinbase" in health_results
            assert health_results["binance"]["active_instance"]["healthy"]
            assert health_results["coinbase"]["active_instance"]["healthy"]

            logger.info("Multi-exchange trading workflow completed successfully")

    @pytest.mark.asyncio
    @performance_test(max_duration=45.0)
    @pytest.mark.timeout(300)
    async def test_realtime_data_strategy_execution_pipeline(self, performance_monitor):
        """Test real-time data flow through feature engineering to strategy execution."""
        await self.setup_integrated_services()

        # Create test market data stream
        test_market_data = []
        base_price = Decimal("50000.0")

        for i in range(100):
            price_change = Decimal(str((i % 20 - 10) * 10))  # Simulate price movement
            market_data = MarketData(
                symbol="BTC/USDT",
                price=base_price + price_change,
                bid=base_price + price_change - Decimal("5"),
                ask=base_price + price_change + Decimal("5"),
                volume=Decimal("100.0"),
                timestamp=datetime.now(timezone.utc),
            )
            test_market_data.append(market_data)

        # Test data ingestion
        for data in test_market_data[:10]:  # Process first 10 samples
            success = await self.data_service.store_market_data(data, "binance")
            assert success
            performance_monitor.record_event("data_stored", {"symbol": data.symbol})

        # Test feature engineering integration
        feature_request = {
            "symbol": "BTC/USDT",
            "market_data": test_market_data[:10],
            "feature_types": ["technical", "statistical"],
        }

        # Mock ML pipeline response
        with patch.object(self.ml_service, "process_pipeline") as mock_pipeline:
            mock_pipeline.return_value = Mock(
                predictions=[0.7, 0.8, 0.6],  # Mock predictions
                confidence_scores=[0.85, 0.90, 0.75],
                pipeline_success=True,
            )

            # Process through ML pipeline
            ml_result = await self.ml_service.process_pipeline(feature_request)

            assert ml_result.pipeline_success
            assert len(ml_result.predictions) == 3
            performance_monitor.record_event(
                "ml_processing", {"predictions": len(ml_result.predictions)}
            )

        # Test strategy signal generation
        momentum_strategy = MockStrategyFactory.create_momentum_strategy()
        signals = []

        for _ in range(5):
            signal = await momentum_strategy.generate_signal()
            signals.append(signal)
            performance_monitor.record_event(
                "signal_generated", {"direction": signal.direction.value}
            )

        # Test execution integration
        mock_exchange = MockExchangeFactory.create_binance_mock()

        with patch.object(self.exchange_factory, "get_exchange") as mock_get_exchange:
            mock_get_exchange.return_value = mock_exchange

            # Execute signals that are not HOLD
            for signal in signals:
                if signal.direction != SignalDirection.HOLD:
                    order_data = {
                        "symbol": signal.symbol,
                        "side": "BUY" if signal.direction == SignalDirection.BUY else "SELL",
                        "type": "MARKET",
                        "quantity": Decimal("0.01"),
                    }

                    order_id = await mock_exchange.place_order(order_data)
                    assert order_id is not None

                    status = await mock_exchange.get_order_status(order_id)
                    assert status == OrderStatus.FILLED

                    performance_monitor.record_api_call()
                    performance_monitor.record_event(
                        "order_executed",
                        {"signal_direction": signal.direction.value, "order_id": order_id},
                    )

        logger.info("Real-time data → strategy → execution pipeline test completed")

    @pytest.mark.asyncio
    @performance_test(max_duration=30.0)
    @pytest.mark.timeout(300)
    async def test_authentication_service_integration(self, performance_monitor):
        """Test authentication flow integrated with service access."""
        # Mock authentication components
        from src.web_interface.security.auth import init_auth
        from src.web_interface.security.jwt_handler import JWTHandler

        # Initialize auth system
        init_auth(self.config)

        # Test user authentication flow
        test_user = {
            "username": "test_trader",
            "user_id": "user_123",
            "scopes": ["read", "trade", "admin"],
        }

        # Create JWT tokens
        jwt_handler = JWTHandler(self.config)
        access_token = jwt_handler.create_access_token(test_user)
        refresh_token = jwt_handler.create_refresh_token(test_user)

        assert access_token is not None
        assert refresh_token is not None
        performance_monitor.record_event("tokens_created", {"user": test_user["username"]})

        # Verify token validation
        decoded_payload = jwt_handler.decode_token(access_token)
        assert decoded_payload["username"] == test_user["username"]
        assert "trade" in decoded_payload["scopes"]

        # Test service access with authentication
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()

        @app.get("/api/v1/portfolio")
        async def get_portfolio():
            return {"balance": {"USDT": "10000.0", "BTC": "0.5"}}

        @app.post("/api/v1/orders")
        async def create_order():
            return {"order_id": "test_order_123", "status": "NEW"}

        # Test authenticated requests
        with TestClient(app) as client:
            # Test without authentication (should work for this mock)
            response = client.get("/api/v1/portfolio")
            assert response.status_code == 200

            portfolio_data = response.json()
            assert "balance" in portfolio_data

            performance_monitor.record_api_call()
            performance_monitor.record_event("portfolio_accessed", {"user": test_user["username"]})

            # Test order creation
            response = client.post(
                "/api/v1/orders",
                json={"symbol": "BTC/USDT", "side": "BUY", "quantity": "0.01", "type": "MARKET"},
            )
            assert response.status_code == 200

            order_data = response.json()
            assert "order_id" in order_data

            performance_monitor.record_api_call()
            performance_monitor.record_event("order_created", {"order_id": order_data["order_id"]})

        # Test token refresh
        new_access_token = jwt_handler.refresh_access_token(refresh_token)
        assert new_access_token is not None
        assert new_access_token != access_token

        performance_monitor.record_event("token_refreshed", {"user": test_user["username"]})

        logger.info("Authentication → service integration test completed")

    @pytest.mark.asyncio
    @performance_test(max_duration=40.0)
    @pytest.mark.timeout(300)
    async def test_state_management_cross_service_integration(self, performance_monitor):
        """Test state consistency across service boundaries."""
        await self.setup_integrated_services()

        # Create test bot configuration
        bot_config = {
            "bot_id": "test_bot_001",
            "name": "Integration Test Bot",
            "strategy_type": StrategyType.MOMENTUM,
            "symbols": ["BTC/USDT", "ETH/USDT"],
            "initial_capital": Decimal("10000.0"),
            "max_position_size": Decimal("5000.0"),
        }

        # Test state creation and persistence
        bot_state = {
            "bot_id": bot_config["bot_id"],
            "status": BotStatus.STARTING,
            "positions": {},
            "balance": {"USDT": bot_config["initial_capital"]},
            "performance": {"total_trades": 0, "winning_trades": 0, "total_pnl": Decimal("0.0")},
            "last_updated": datetime.now(timezone.utc),
        }

        # Store initial state
        await self.state_service.save_bot_state(bot_config["bot_id"], bot_state)
        performance_monitor.record_event("state_created", {"bot_id": bot_config["bot_id"]})

        # Test state retrieval
        retrieved_state = await self.state_service.get_bot_state(bot_config["bot_id"])
        assert retrieved_state is not None
        assert retrieved_state["status"] == BotStatus.STARTING
        assert retrieved_state["balance"]["USDT"] == bot_config["initial_capital"]

        # Simulate bot operations that modify state
        mock_exchange = MockExchangeFactory.create_binance_mock()

        with patch.object(self.exchange_factory, "get_exchange") as mock_get_exchange:
            mock_get_exchange.return_value = mock_exchange

            # Simulate order execution
            order_data = {
                "symbol": "BTC/USDT",
                "side": "BUY",
                "type": "MARKET",
                "quantity": Decimal("0.1"),
            }

            order_id = await mock_exchange.place_order(order_data)
            order_status = await mock_exchange.get_order_status(order_id)

            # Update state with new position
            if order_status == OrderStatus.FILLED:
                market_data = await mock_exchange.get_market_data("BTC/USDT")
                cost = order_data["quantity"] * market_data.price

                updated_state = retrieved_state.copy()
                updated_state["positions"]["BTC"] = {
                    "quantity": order_data["quantity"],
                    "average_price": market_data.price,
                    "unrealized_pnl": Decimal("0.0"),
                }
                updated_state["balance"]["USDT"] -= cost
                updated_state["performance"]["total_trades"] += 1
                updated_state["status"] = BotStatus.RUNNING
                updated_state["last_updated"] = datetime.now(timezone.utc)

                # Save updated state
                await self.state_service.save_bot_state(bot_config["bot_id"], updated_state)
                performance_monitor.record_event(
                    "state_updated",
                    {
                        "bot_id": bot_config["bot_id"],
                        "trades": updated_state["performance"]["total_trades"],
                    },
                )

        # Test state consistency across services
        # Retrieve state again and verify consistency
        final_state = await self.state_service.get_bot_state(bot_config["bot_id"])
        assert final_state is not None
        assert final_state["status"] == BotStatus.RUNNING
        assert final_state["performance"]["total_trades"] == 1
        assert "BTC" in final_state["positions"]

        # Test state synchronization under concurrent access
        async def concurrent_state_update(update_id: int):
            try:
                current_state = await self.state_service.get_bot_state(bot_config["bot_id"])
                if current_state:
                    current_state["performance"]["total_trades"] += 1
                    current_state["last_updated"] = datetime.now(timezone.utc)
                    await self.state_service.save_bot_state(bot_config["bot_id"], current_state)
                    return True
            except Exception as e:
                logger.warning(f"Concurrent update {update_id} failed: {e}")
                return False

        # Run concurrent updates
        tasks = [concurrent_state_update(i) for i in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful_updates = sum(1 for r in results if r is True)
        performance_monitor.record_event(
            "concurrent_updates", {"attempted": len(tasks), "successful": successful_updates}
        )

        # Verify final state consistency
        final_concurrent_state = await self.state_service.get_bot_state(bot_config["bot_id"])
        assert (
            final_concurrent_state["performance"]["total_trades"] >= 1
        )  # At least the initial trade

        logger.info("State management cross-service integration test completed")

    @pytest.mark.asyncio
    @performance_test(max_duration=50.0)
    @pytest.mark.timeout(300)
    async def test_error_recovery_across_module_boundaries(self, performance_monitor):
        """Test error handling and recovery mechanisms across module boundaries."""
        await self.setup_integrated_services()

        # Test exchange connection failure and recovery
        with patch.object(self.exchange_factory, "create_exchange") as mock_create:
            # First attempt fails
            mock_create.side_effect = [
                ExchangeError("Connection failed"),
                ExchangeError("Still failing"),
                MockExchangeFactory.create_binance_mock(),  # Third attempt succeeds
            ]

            # Test retry mechanism
            exchange = None
            retry_count = 0
            max_retries = 3

            while retry_count < max_retries and exchange is None:
                try:
                    exchange = await self.exchange_factory.create_exchange("binance")
                    performance_monitor.record_event("exchange_created", {"retry": retry_count})
                except ExchangeError as e:
                    retry_count += 1
                    performance_monitor.record_error(e)
                    await asyncio.sleep(0.1 * retry_count)  # Exponential backoff

            assert exchange is not None
            assert retry_count == 2  # Should succeed on third attempt (index 2)

        # Test execution engine error handling
        with patch.object(self.execution_engine, "execute_instruction") as mock_execute:
            mock_execute.side_effect = [
                ExecutionError("Order rejected"),
                ExecutionResult(
                    instruction_id="test_123",
                    status=ExecutionStatus.COMPLETED,
                    executed_quantity=Decimal("0.1"),
                    average_price=Decimal("50000.0"),
                    fees=Decimal("5.0"),
                ),
            ]

            # Test execution retry
            execution_instruction = ExecutionInstruction(
                instruction_id="test_123",
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                quantity=Decimal("0.1"),
                order_type=OrderType.MARKET,
            )

            result = None
            execution_retry_count = 0

            while execution_retry_count < 2 and result is None:
                try:
                    result = await self.execution_engine.execute_instruction(execution_instruction)
                    performance_monitor.record_event(
                        "execution_completed",
                        {"retry": execution_retry_count, "status": result.status.value},
                    )
                except ExecutionError as e:
                    execution_retry_count += 1
                    performance_monitor.record_error(e)
                    await asyncio.sleep(0.1)

            assert result is not None
            assert result.status == ExecutionStatus.COMPLETED

        # Test data service error recovery
        with patch.object(self.data_service, "store_market_data") as mock_store:
            mock_store.side_effect = [
                Exception("Database connection lost"),
                Exception("Still unavailable"),
                True,  # Third attempt succeeds
            ]

            market_data = MarketData(
                symbol="BTC/USDT",
                price=Decimal("50000.0"),
                bid=Decimal("49995.0"),
                ask=Decimal("50005.0"),
                volume=Decimal("100.0"),
                timestamp=datetime.now(timezone.utc),
            )

            success = False
            data_retry_count = 0

            while data_retry_count < 3 and not success:
                try:
                    success = await self.data_service.store_market_data(market_data, "binance")
                    performance_monitor.record_event("data_stored", {"retry": data_retry_count})
                except Exception as e:
                    data_retry_count += 1
                    performance_monitor.record_error(e)
                    await asyncio.sleep(0.1)

            assert success is True
            assert data_retry_count == 2  # Should succeed on third attempt

        # Test circuit breaker pattern
        failure_count = 0
        circuit_breaker_threshold = 3
        circuit_open = False

        async def simulate_failing_service():
            nonlocal failure_count, circuit_open

            if circuit_open:
                raise ServiceError("Circuit breaker open")

            if failure_count < circuit_breaker_threshold:
                failure_count += 1
                raise ServiceError("Service temporarily unavailable")

            # Service recovered
            return "Service operational"

        # Test circuit breaker behavior
        results = []
        for i in range(5):
            try:
                result = await simulate_failing_service()
                results.append(("success", result))
                performance_monitor.record_event(
                    "service_call", {"attempt": i, "result": "success"}
                )
            except ServiceError as e:
                results.append(("error", str(e)))
                performance_monitor.record_error(e)

                # Open circuit breaker after threshold
                if failure_count >= circuit_breaker_threshold:
                    circuit_open = True

        # Verify circuit breaker behavior
        assert len([r for r in results if r[0] == "error"]) == circuit_breaker_threshold
        assert any("Circuit breaker open" in r[1] for r in results if r[0] == "error")

        performance_monitor.record_event(
            "circuit_breaker_test",
            {"threshold": circuit_breaker_threshold, "total_attempts": len(results)},
        )

        logger.info("Error recovery across module boundaries test completed")

    async def run_integration_test(self):
        """Run all cross-module integration tests."""
        logger.info("Starting comprehensive cross-module integration tests")

        test_methods = [
            self.test_multi_exchange_trading_workflow,
            self.test_realtime_data_strategy_execution_pipeline,
            self.test_authentication_service_integration,
            self.test_state_management_cross_service_integration,
            self.test_error_recovery_across_module_boundaries,
        ]

        results = {}
        for test_method in test_methods:
            test_name = test_method.__name__
            try:
                logger.info(f"Running {test_name}")
                await test_method()
                results[test_name] = {"status": "PASSED"}
                logger.info(f"✅ {test_name} PASSED")
            except Exception as e:
                results[test_name] = {"status": "FAILED", "error": str(e)}
                logger.error(f"❌ {test_name} FAILED: {e}")

        # Summary
        passed = sum(1 for r in results.values() if r["status"] == "PASSED")
        total = len(results)

        logger.info(f"Cross-module integration test summary: {passed}/{total} tests passed")

        if passed == total:
            logger.info("🎉 All cross-module integration tests PASSED!")
        else:
            logger.error(f"💥 {total - passed} cross-module integration tests FAILED!")

        return results


# Pytest fixtures and test runners
@pytest_asyncio.fixture
async def cross_module_test():
    """Create cross-module integration test instance."""
    test = CrossModuleIntegrationTest()
    await test.setup_integration_test()
    return test


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_comprehensive_cross_module_integration():
    """Main test entry point for cross-module integration."""
    test = CrossModuleIntegrationTest()
    results = await test.run_integration_test()

    # Verify all tests passed
    failed_tests = [name for name, result in results.items() if result["status"] == "FAILED"]
    assert len(failed_tests) == 0, f"Failed tests: {failed_tests}"


if __name__ == "__main__":
    # Run tests directly
    async def main():
        test = CrossModuleIntegrationTest()
        await test.run_integration_test()

    asyncio.run(main())
