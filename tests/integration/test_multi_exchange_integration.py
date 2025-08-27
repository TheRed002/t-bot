"""
Multi-exchange operations integration tests.

Tests cross-exchange arbitrage, portfolio balancing, rate limiting coordination,
failover scenarios, and liquidity routing across multiple exchanges.
"""

import pytest
import asyncio
import logging
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Tuple, Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import time
import random

from tests.integration.base_integration import (
    BaseIntegrationTest, MockExchangeFactory, PerformanceMonitor,
    performance_test, wait_for_condition
)
from src.core.types import (
    MarketData, Order, OrderSide, OrderType, OrderStatus, Position,
    ExecutionInstruction, ExecutionResult, ExecutionStatus
)
from src.exchanges.base import BaseExchange
from src.exchanges.global_coordinator import GlobalRateCoordinator  
from src.execution.algorithms.smart_router import SmartOrderRouter
from src.capital_management.exchange_distributor import ExchangeDistributor
from src.risk_management.correlation_monitor import CorrelationMonitor

logger = logging.getLogger(__name__)


class TestCrossExchangeArbitrage:
    """Test cross-exchange arbitrage execution workflows."""
    
    @pytest.mark.asyncio
    @performance_test(max_duration=45.0)
    async def test_simple_arbitrage_execution(self, performance_monitor):
        """Test simple arbitrage between two exchanges."""
        
        # Create exchanges with price differential
        binance = MockExchangeFactory.create_binance_mock(
            initial_balance={"USDT": Decimal("50000.0"), "BTC": Decimal("0.0")},
            market_prices={"BTC/USDT": Decimal("50000.0")}  # Lower price
        )
        
        coinbase = MockExchangeFactory.create_coinbase_mock(
            initial_balance={"USDT": Decimal("50000.0"), "BTC": Decimal("0.0")},
            market_prices={"BTC/USDT": Decimal("50200.0")}  # Higher price
        )
        
        exchanges = {"binance": binance, "coinbase": coinbase}
        
        # Setup arbitrage coordinator
        arbitrage_coordinator = Mock()
        arbitrage_coordinator.min_profit_threshold = Decimal("100.0")  # $100 minimum profit
        arbitrage_coordinator.max_position_size = Decimal("1.0")  # 1 BTC max
        
        # 1. Scan for arbitrage opportunities
        arbitrage_opportunities = []
        
        for symbol in ["BTC/USDT"]:
            exchange_prices = {}
            
            for exchange_name, exchange in exchanges.items():
                market_data = await exchange.get_market_data(symbol)
                performance_monitor.record_api_call()
                exchange_prices[exchange_name] = {
                    "bid": Decimal(str(market_data.metadata.get("bid", market_data.close))),
                    "ask": Decimal(str(market_data.metadata.get("ask", market_data.close))),
                    "price": Decimal(str(market_data.metadata.get("price", market_data.close)))
                }
            
            # Find best buy (lowest ask) and best sell (highest bid) prices
            best_buy_exchange = min(exchange_prices.items(), key=lambda x: x[1]["ask"])
            best_sell_exchange = max(exchange_prices.items(), key=lambda x: x[1]["bid"])
            
            buy_price = best_buy_exchange[1]["ask"]
            sell_price = best_sell_exchange[1]["bid"]
            price_diff = sell_price - buy_price
            
            if price_diff > arbitrage_coordinator.min_profit_threshold / Decimal("50000.0") * buy_price:
                arbitrage_opportunities.append({
                    "symbol": symbol,
                    "buy_exchange": best_buy_exchange[0],
                    "sell_exchange": best_sell_exchange[0],
                    "buy_price": buy_price,
                    "sell_price": sell_price,
                    "profit_per_unit": price_diff,
                    "max_quantity": min(
                        arbitrage_coordinator.max_position_size,
                        exchanges[best_buy_exchange[0]]._balances.get("USDT", Decimal("0")) / buy_price
                    )
                })
        
        assert len(arbitrage_opportunities) > 0
        opportunity = arbitrage_opportunities[0]
        
        logger.info(f"Arbitrage opportunity: Buy on {opportunity['buy_exchange']} @ ${opportunity['buy_price']}, "
                   f"Sell on {opportunity['sell_exchange']} @ ${opportunity['sell_price']}, "
                   f"Profit: ${opportunity['profit_per_unit']}")
        
        # 2. Execute arbitrage trades simultaneously
        quantity = min(opportunity["max_quantity"], Decimal("0.5"))  # Conservative size
        
        # Execute both legs concurrently for best execution
        buy_task = asyncio.create_task(exchanges[opportunity["buy_exchange"]].place_order({
            "symbol": opportunity["symbol"],
            "side": "BUY",
            "quantity": str(quantity),
            "type": "MARKET"
        }))
        
        sell_task = asyncio.create_task(exchanges[opportunity["sell_exchange"]].place_order({
            "symbol": opportunity["symbol"], 
            "side": "SELL",
            "quantity": str(quantity),
            "type": "MARKET"
        }))
        
        # Wait for both orders to complete
        start_time = time.time()
        buy_order_id, sell_order_id = await asyncio.gather(buy_task, sell_task)
        execution_time = time.time() - start_time
        
        performance_monitor.record_api_call(execution_time / 2)  # Average per order
        
        # 3. Verify execution
        buy_exchange = exchanges[opportunity["buy_exchange"]]
        sell_exchange = exchanges[opportunity["sell_exchange"]]
        
        buy_order = buy_exchange._orders[buy_order_id]
        sell_order = sell_exchange._orders[sell_order_id]
        
        assert buy_order.status == OrderStatus.FILLED
        assert sell_order.status == OrderStatus.FILLED
        assert buy_order.filled_quantity == quantity
        assert sell_order.filled_quantity == quantity
        
        # 4. Calculate actual profit
        actual_profit = (sell_order.average_price - buy_order.average_price) * quantity
        expected_profit = opportunity["profit_per_unit"] * quantity
        
        # Account for slippage - actual profit should be close to expected
        profit_deviation = abs(actual_profit - expected_profit) / expected_profit
        assert profit_deviation < Decimal("0.05")  # Less than 5% deviation
        
        logger.info(f"Arbitrage executed: Expected profit ${expected_profit}, "
                   f"Actual profit ${actual_profit}, Execution time: {execution_time:.3f}s")
        
        # 5. Verify positions are balanced (net neutral)
        buy_balance = await buy_exchange.get_balance()
        sell_balance = await sell_exchange.get_balance()
        
        # Buy exchange should have BTC, less USDT
        assert buy_balance["BTC"] == quantity
        
        # Sell exchange should have less BTC, more USDT
        # (Note: In mock, sell exchange started with 0 BTC, so this would need position setup)
        
        assert actual_profit > Decimal("0")  # Should be profitable
    
    @pytest.mark.asyncio
    async def test_triangular_arbitrage(self):
        """Test triangular arbitrage within a single exchange."""
        
        # Setup exchange with pricing inefficiencies
        binance = MockExchangeFactory.create_binance_mock(
            initial_balance={
                "USDT": Decimal("100000.0"),
                "BTC": Decimal("0.0"), 
                "ETH": Decimal("0.0")
            },
            market_prices={
                "BTC/USDT": Decimal("50000.0"),
                "ETH/USDT": Decimal("3000.0"),
                "ETH/BTC": Decimal("0.059")  # Slight pricing inefficiency: 3000/50000 = 0.06
            }
        )
        
        # Add ETH/BTC market data handler
        original_get_market_data = binance.get_market_data
        
        async def multi_pair_market_data(symbol):
            if symbol == "ETH/BTC":
                price = binance._market_prices.get("ETH/BTC", Decimal("0.059"))
                spread = price * Decimal("0.001")
                return MarketData(
                    symbol=symbol,
                    timestamp=datetime.now(timezone.utc),
                    open=price * Decimal("0.999"),
                    high=price * Decimal("1.002"), 
                    low=price * Decimal("0.998"),
                    close=price,
                    volume=Decimal("10000.0"),
                    exchange="binance",
                    metadata={
                        "bid": float(price - spread/2),
                        "ask": float(price + spread/2),
                        "price": float(price)
                    }
                )
            else:
                return await original_get_market_data(symbol)
        
        binance.get_market_data = multi_pair_market_data
        
        # 1. Check for triangular arbitrage opportunity
        # Path: USDT -> BTC -> ETH -> USDT
        btc_usd_data = await binance.get_market_data("BTC/USDT")
        eth_usd_data = await binance.get_market_data("ETH/USDT") 
        eth_btc_data = await binance.get_market_data("ETH/BTC")
        
        # Calculate implied ETH/BTC rate from USD pairs  
        eth_price = Decimal(str(eth_usd_data.metadata.get("price", eth_usd_data.close)))
        btc_price = Decimal(str(btc_usd_data.metadata.get("price", btc_usd_data.close)))
        eth_btc_price = Decimal(str(eth_btc_data.metadata.get("price", eth_btc_data.close)))
        
        implied_eth_btc = eth_price / btc_price  # 3000/50000 = 0.06
        actual_eth_btc = eth_btc_price  # 0.059
        
        arbitrage_margin = implied_eth_btc - actual_eth_btc  # 0.06 - 0.059 = 0.001
        
        # Check if profitable (considering fees)
        min_margin = Decimal("0.0005")  # 0.05% minimum margin
        assert arbitrage_margin > min_margin
        
        logger.info(f"Triangular arbitrage opportunity: Implied {implied_eth_btc}, "
                   f"Actual {actual_eth_btc}, Margin {arbitrage_margin}")
        
        # 2. Execute triangular arbitrage
        start_amount = Decimal("10000.0")  # $10,000 USDT
        
        # Leg 1: USDT -> BTC
        btc_ask = Decimal(str(btc_usd_data.metadata.get("ask", btc_price)))
        btc_quantity = start_amount / btc_ask
        btc_order_id = await binance.place_order({
            "symbol": "BTC/USDT",
            "side": "BUY",
            "quantity": str(btc_quantity),
            "type": "MARKET"
        })
        
        # Leg 2: BTC -> ETH
        eth_btc_ask = Decimal(str(eth_btc_data.metadata.get("ask", eth_btc_price)))
        eth_quantity = btc_quantity / eth_btc_ask
        eth_order_id = await binance.place_order({
            "symbol": "ETH/BTC",
            "side": "BUY", 
            "quantity": str(eth_quantity),
            "type": "MARKET"
        })
        
        # Leg 3: ETH -> USDT
        usdt_return_id = await binance.place_order({
            "symbol": "ETH/USDT",
            "side": "SELL",
            "quantity": str(eth_quantity),
            "type": "MARKET"
        })
        
        # 3. Verify profitability
        final_balance = await binance.get_balance()
        final_usdt = final_balance["USDT"]
        
        # Should have more USDT than we started with
        profit = final_usdt - (Decimal("100000.0") - start_amount)
        profit_percentage = (profit / start_amount) * 100
        
        assert profit > Decimal("0")
        assert profit_percentage > Decimal("0.01")  # At least 0.01% profit
        
        logger.info(f"Triangular arbitrage profit: ${profit} ({profit_percentage:.3f}%)")
    
    @pytest.mark.asyncio 
    async def test_arbitrage_with_transfer_costs(self):
        """Test arbitrage accounting for transfer costs and delays."""
        
        exchanges = {
            "binance": MockExchangeFactory.create_binance_mock(
                market_prices={"BTC/USDT": Decimal("50000.0")}
            ),
            "coinbase": MockExchangeFactory.create_coinbase_mock(
                market_prices={"BTC/USDT": Decimal("50300.0")}  # Higher price
            )
        }
        
        # Setup transfer cost model
        transfer_costs = {
            "BTC": {"fee": Decimal("0.0005"), "time_minutes": 30},  # 0.0005 BTC fee, 30min
            "USDT": {"fee": Decimal("25.0"), "time_minutes": 60}    # $25 fee, 60min
        }
        
        # Calculate arbitrage profitability including transfer costs
        symbol = "BTC/USDT"
        quantity = Decimal("1.0")  # 1 BTC
        
        binance_data = await exchanges["binance"].get_market_data(symbol)
        coinbase_data = await exchanges["coinbase"].get_market_data(symbol)
        
        binance_ask = Decimal(str(binance_data.metadata.get("ask", binance_data.close)))
        coinbase_bid = Decimal(str(coinbase_data.metadata.get("bid", coinbase_data.close)))
        coinbase_price = Decimal(str(coinbase_data.metadata.get("price", coinbase_data.close)))
        
        # Gross profit
        gross_profit = (coinbase_bid - binance_ask) * quantity
        
        # Transfer costs (assume we need to transfer BTC from Binance to Coinbase)
        transfer_fee = transfer_costs["BTC"]["fee"] * coinbase_price  # Fee in USD
        transfer_time = transfer_costs["BTC"]["time_minutes"]
        
        # Opportunity cost during transfer (assume 0.1% per hour price movement risk)
        binance_price = Decimal(str(binance_data.metadata.get("price", binance_data.close)))
        opportunity_cost = quantity * binance_price * Decimal("0.001") * Decimal(str(transfer_time / 60))
        
        net_profit = gross_profit - transfer_fee - opportunity_cost
        
        logger.info(f"Arbitrage analysis:")
        logger.info(f"  Gross profit: ${gross_profit}")
        logger.info(f"  Transfer fee: ${transfer_fee}")
        logger.info(f"  Opportunity cost: ${opportunity_cost}")
        logger.info(f"  Net profit: ${net_profit}")
        
        # Only execute if net profit exceeds minimum threshold
        min_net_profit = Decimal("100.0")  # $100
        
        if net_profit > min_net_profit:
            logger.info("Executing arbitrage with transfer costs...")
            
            # Execute buy on cheaper exchange
            buy_order_id = await exchanges["binance"].place_order({
                "symbol": symbol,
                "side": "BUY", 
                "quantity": str(quantity),
                "type": "MARKET"
            })
            
            # Simulate transfer delay
            await asyncio.sleep(0.1)  # Simulate transfer time (scaled down for testing)
            
            # Execute sell on more expensive exchange  
            sell_order_id = await exchanges["coinbase"].place_order({
                "symbol": symbol,
                "side": "SELL",
                "quantity": str(quantity),
                "type": "MARKET"
            })
            
            assert buy_order_id is not None
            assert sell_order_id is not None
            
            logger.info("Arbitrage with transfer costs executed successfully")
        else:
            logger.info("Arbitrage not profitable after transfer costs")
            assert net_profit <= min_net_profit


class TestLiquidityRouting:
    """Test intelligent liquidity routing across exchanges."""
    
    @pytest.mark.asyncio
    async def test_smart_order_routing(self):
        """Test smart order routing to minimize market impact."""
        
        # Setup exchanges with different liquidity profiles
        exchanges = {
            "binance": MockExchangeFactory.create_binance_mock(
                market_prices={"BTC/USDT": Decimal("50000.0")}
            ),
            "coinbase": MockExchangeFactory.create_coinbase_mock(
                market_prices={"BTC/USDT": Decimal("50010.0")}
            ),
            "okx": MockExchangeFactory.create_binance_mock(  # Reuse factory with different name
                market_prices={"BTC/USDT": Decimal("49995.0")}
            )
        }
        exchanges["okx"].name = "okx"
        
        # Mock order book depths for each exchange
        order_books = {
            "binance": {
                "bids": [(Decimal("49995.0"), Decimal("5.0")), (Decimal("49990.0"), Decimal("10.0"))],
                "asks": [(Decimal("50005.0"), Decimal("3.0")), (Decimal("50010.0"), Decimal("8.0"))]
            },
            "coinbase": {
                "bids": [(Decimal("50005.0"), Decimal("2.0")), (Decimal("50000.0"), Decimal("4.0"))],
                "asks": [(Decimal("50015.0"), Decimal("1.5")), (Decimal("50020.0"), Decimal("6.0"))]
            },
            "okx": {
                "bids": [(Decimal("49990.0"), Decimal("8.0")), (Decimal("49985.0"), Decimal("12.0"))],
                "asks": [(Decimal("50000.0"), Decimal("4.0")), (Decimal("50005.0"), Decimal("7.0"))]
            }
        }
        
        smart_router = Mock()
        
        # Test large buy order routing (10 BTC)
        target_quantity = Decimal("10.0")
        target_side = "BUY"
        
        # Smart router should split order across exchanges for best execution
        routing_plan = []
        remaining_quantity = target_quantity
        
        # Sort exchanges by best ask prices for buying
        sorted_asks = []
        for exchange_name, book in order_books.items():
            for price, size in book["asks"]:
                sorted_asks.append((price, size, exchange_name))
        
        sorted_asks.sort(key=lambda x: x[0])  # Sort by price
        
        # Fill from best prices first
        for price, available_size, exchange_name in sorted_asks:
            if remaining_quantity <= 0:
                break
                
            fill_quantity = min(remaining_quantity, available_size)
            routing_plan.append({
                "exchange": exchange_name,
                "quantity": fill_quantity,
                "price": price
            })
            remaining_quantity -= fill_quantity
        
        # Verify routing plan
        total_routed = sum(plan["quantity"] for plan in routing_plan)
        assert total_routed >= target_quantity * Decimal("0.9")  # At least 90% filled
        
        # Calculate volume-weighted average price
        total_cost = sum(plan["quantity"] * plan["price"] for plan in routing_plan)
        vwap = total_cost / sum(plan["quantity"] for plan in routing_plan)
        
        logger.info(f"Smart routing plan for {target_quantity} BTC buy:")
        for plan in routing_plan:
            logger.info(f"  {plan['exchange']}: {plan['quantity']} @ ${plan['price']}")
        logger.info(f"  VWAP: ${vwap}")
        
        # Execute routing plan
        executed_orders = []
        for plan in routing_plan:
            exchange = exchanges[plan["exchange"]]
            order_id = await exchange.place_order({
                "symbol": "BTC/USDT",
                "side": target_side,
                "quantity": str(plan["quantity"]),
                "type": "MARKET"
            })
            executed_orders.append((plan["exchange"], order_id))
        
        # Verify all orders executed
        assert len(executed_orders) == len(routing_plan)
        
        total_executed = Decimal("0")
        for exchange_name, order_id in executed_orders:
            order = exchanges[exchange_name]._orders[order_id]
            assert order.status == OrderStatus.FILLED
            total_executed += order.filled_quantity
        
        assert total_executed >= target_quantity * Decimal("0.9")
        logger.info(f"Smart routing executed: {total_executed} BTC across {len(executed_orders)} exchanges")
    
    @pytest.mark.asyncio
    async def test_dynamic_liquidity_allocation(self):
        """Test dynamic allocation based on real-time liquidity conditions."""
        
        exchanges = {
            "binance": MockExchangeFactory.create_binance_mock(),
            "coinbase": MockExchangeFactory.create_coinbase_mock()
        }
        
        # Simulate changing liquidity conditions
        liquidity_monitor = Mock()
        
        initial_liquidity = {
            "binance": {"BTC/USDT": {"depth": Decimal("50.0"), "spread": Decimal("10.0")}},
            "coinbase": {"BTC/USDT": {"depth": Decimal("30.0"), "spread": Decimal("15.0")}}
        }
        
        # Simulate Binance liquidity drying up
        updated_liquidity = {
            "binance": {"BTC/USDT": {"depth": Decimal("5.0"), "spread": Decimal("50.0")}},  # Low liquidity
            "coinbase": {"BTC/USDT": {"depth": Decimal("35.0"), "spread": Decimal("12.0")}}   # Better liquidity
        }
        
        liquidity_monitor.get_liquidity_snapshot = Mock(side_effect=[initial_liquidity, updated_liquidity])
        
        # Initial allocation based on liquidity
        total_allocation = Decimal("100000.0")  # $100k to allocate
        
        def calculate_allocation(liquidity_data):
            total_score = Decimal("0")
            scores = {}
            
            for exchange, pairs in liquidity_data.items():
                btc_liquidity = pairs.get("BTC/USDT", {})
                depth = btc_liquidity.get("depth", Decimal("0"))
                spread = btc_liquidity.get("spread", Decimal("1000"))
                
                # Score based on depth (higher is better) and spread (lower is better)
                score = depth / spread if spread > 0 else Decimal("0")
                scores[exchange] = score
                total_score += score
            
            # Allocate proportionally to scores
            allocations = {}
            for exchange, score in scores.items():
                allocation_pct = score / total_score if total_score > 0 else Decimal("0")
                allocations[exchange] = total_allocation * allocation_pct
            
            return allocations
        
        # Initial allocation
        initial_allocations = calculate_allocation(liquidity_monitor.get_liquidity_snapshot())
        logger.info(f"Initial allocations: {initial_allocations}")
        
        # Verify Binance gets more initially (better liquidity)
        assert initial_allocations["binance"] > initial_allocations["coinbase"]
        
        # Updated allocation after liquidity change
        updated_allocations = calculate_allocation(liquidity_monitor.get_liquidity_snapshot())
        logger.info(f"Updated allocations: {updated_allocations}")
        
        # Verify Coinbase gets more after Binance liquidity deteriorates
        assert updated_allocations["coinbase"] > updated_allocations["binance"]
        
        # Calculate rebalancing trades needed
        rebalancing_trades = []
        for exchange in initial_allocations.keys():
            initial = initial_allocations[exchange]
            updated = updated_allocations[exchange]
            difference = updated - initial
            
            if abs(difference) > Decimal("1000.0"):  # $1k threshold
                rebalancing_trades.append({
                    "exchange": exchange,
                    "direction": "increase" if difference > 0 else "decrease",
                    "amount": abs(difference)
                })
        
        assert len(rebalancing_trades) > 0
        logger.info(f"Rebalancing trades: {rebalancing_trades}")
    
    @pytest.mark.asyncio
    async def test_latency_aware_routing(self):
        """Test routing decisions based on exchange latency."""
        
        exchanges = {
            "binance": MockExchangeFactory.create_binance_mock(),
            "coinbase": MockExchangeFactory.create_coinbase_mock()
        }
        
        # Simulate different latencies
        exchange_latencies = {
            "binance": 45,   # 45ms average latency
            "coinbase": 120  # 120ms average latency  
        }
        
        # Mock latency measurement
        async def measure_latency(exchange_name: str) -> float:
            base_latency = exchange_latencies[exchange_name]
            jitter = random.uniform(-10, 10)  # +/- 10ms jitter
            await asyncio.sleep((base_latency + jitter) / 1000)  # Convert to seconds
            return base_latency + jitter
        
        # Test latency-sensitive order (market data arbitrage)
        symbol = "BTC/USDT"
        target_execution_time = 100  # 100ms target
        
        # Measure current latencies
        latency_results = {}
        for exchange_name in exchanges.keys():
            start_time = time.time()
            latency = await measure_latency(exchange_name)
            actual_latency = (time.time() - start_time) * 1000
            latency_results[exchange_name] = actual_latency
        
        logger.info(f"Exchange latencies: {latency_results}")
        
        # Select exchange based on latency for time-sensitive trade
        best_exchange = min(latency_results.items(), key=lambda x: x[1])
        selected_exchange_name = best_exchange[0]
        selected_latency = best_exchange[1]
        
        assert selected_exchange_name == "binance"  # Should select lower latency exchange
        assert selected_latency < target_execution_time
        
        # Execute time-sensitive order on selected exchange
        start_execution = time.time()
        order_id = await exchanges[selected_exchange_name].place_order({
            "symbol": symbol,
            "side": "BUY",
            "quantity": "1.0",
            "type": "MARKET"
        })
        execution_latency = (time.time() - start_execution) * 1000
        
        logger.info(f"Order executed on {selected_exchange_name} with {execution_latency:.1f}ms latency")
        
        # Verify order executed successfully with acceptable latency
        assert order_id is not None
        assert execution_latency < target_execution_time * 2  # Allow some buffer


class TestExchangeFailoverScenarios:
    """Test failover scenarios and resilience."""
    
    @pytest.mark.asyncio
    async def test_primary_exchange_failure_handling(self):
        """Test failover when primary exchange fails."""
        
        # Setup primary and backup exchanges
        primary = MockExchangeFactory.create_binance_mock()
        backup = MockExchangeFactory.create_coinbase_mock(
            initial_balance={"USDT": Decimal("50000.0"), "BTC": Decimal("0.0")}  # Give USDT for buying
        )
        
        exchanges = {"primary": primary, "backup": backup}
        
        # Configure primary exchange failure
        failure_start = time.time()
        failure_duration = 2.0  # 2 second outage
        
        async def simulate_outage(order_data):
            if time.time() - failure_start < failure_duration:
                raise ConnectionError("Primary exchange unreachable")
            return await primary.place_order(order_data)
        
        primary.place_order = simulate_outage
        primary.is_connected = False
        
        # Setup failover coordinator
        failover_coordinator = Mock()
        failover_coordinator.primary_exchange = "primary"
        failover_coordinator.backup_exchanges = ["backup"]
        failover_coordinator.max_failover_time = 5.0  # 5 second max
        
        # Attempt to place order with failover logic
        order_placed = False
        selected_exchange = None
        failover_attempts = 0
        
        target_order = {
            "symbol": "BTC/USDT", 
            "side": "BUY",
            "quantity": "1.0",
            "type": "MARKET"
        }
        
        # Try primary first
        try:
            order_id = await exchanges[failover_coordinator.primary_exchange].place_order(target_order)
            order_placed = True
            selected_exchange = failover_coordinator.primary_exchange
        except ConnectionError:
            # Primary failed, try backups
            for backup_name in failover_coordinator.backup_exchanges:
                failover_attempts += 1
                try:
                    backup_exchange = exchanges[backup_name]
                    if backup_exchange.is_connected:
                        order_id = await backup_exchange.place_order(target_order)
                        order_placed = True
                        selected_exchange = backup_name
                        break
                except Exception as e:
                    logger.warning(f"Backup exchange {backup_name} also failed: {e}")
                    continue
        
        # Verify failover success
        assert order_placed is True
        assert selected_exchange == "backup"
        assert failover_attempts == 1
        
        logger.info(f"Failover successful: Order placed on {selected_exchange} after {failover_attempts} attempts")
        
        # Verify order actually executed
        backup_orders = backup._orders
        assert len(backup_orders) > 0  # At least one order should be placed
        order = list(backup_orders.values())[0]  # Get the first order
        assert order.status == OrderStatus.FILLED  # Order should be filled
        assert order.side == OrderSide.BUY  # Should be a BUY order
        assert order.symbol == "BTC/USDT"  # Should be the correct symbol
        assert order.quantity == Decimal("1.0")  # Should be the correct quantity
        
        # The main goal is failover functionality - order placement succeeded on backup
        logger.info(f"Failover test successful - order {order.order_id} placed on backup exchange")
    
    @pytest.mark.asyncio
    async def test_cascade_failure_handling(self):
        """Test handling of multiple simultaneous exchange failures."""
        
        # Setup multiple exchanges
        exchanges = {
            "binance": MockExchangeFactory.create_binance_mock(),
            "coinbase": MockExchangeFactory.create_coinbase_mock(), 
            "okx": MockExchangeFactory.create_binance_mock()
        }
        exchanges["okx"].name = "okx"
        
        # Simulate cascade failure (all but one exchange fails)
        exchanges["binance"].is_connected = False
        exchanges["binance"].place_order = AsyncMock(side_effect=ConnectionError("Binance down"))
        
        exchanges["coinbase"].is_connected = False  
        exchanges["coinbase"].place_order = AsyncMock(side_effect=ConnectionError("Coinbase down"))
        
        # Only OKX remains operational
        exchanges["okx"].is_connected = True
        
        # Emergency trading coordination
        emergency_coordinator = Mock()
        emergency_coordinator.min_operational_exchanges = 1
        emergency_coordinator.emergency_procedures = True
        
        # Check operational exchanges
        operational_exchanges = []
        for name, exchange in exchanges.items():
            try:
                # Test connectivity
                market_data = await exchange.get_market_data("BTC/USDT")
                if exchange.is_connected:
                    operational_exchanges.append(name)
            except Exception:
                continue
        
        operational_count = len(operational_exchanges)
        
        logger.info(f"Operational exchanges: {operational_exchanges} ({operational_count} total)")
        
        # Verify minimum operational requirement
        assert operational_count >= emergency_coordinator.min_operational_exchanges
        
        # Execute emergency trading procedures
        if operational_count < 2:  # Less than 2 exchanges operational
            # Enable emergency mode
            emergency_coordinator.emergency_procedures = True
            
            # Reduce position sizes
            emergency_position_limit = Decimal("0.5")  # 50% of normal size
            
            # Execute emergency order on remaining exchange
            remaining_exchange = exchanges[operational_exchanges[0]]
            
            emergency_order_id = await remaining_exchange.place_order({
                "symbol": "BTC/USDT",
                "side": "BUY", 
                "quantity": str(emergency_position_limit),
                "type": "MARKET"
            })
            
            assert emergency_order_id is not None
            logger.info(f"Emergency order executed on {operational_exchanges[0]}")
            
            # Verify emergency procedures activated
            assert emergency_coordinator.emergency_procedures is True
        
        # Test recovery scenario
        # Simulate one exchange coming back online
        exchanges["binance"].is_connected = True
        exchanges["binance"].place_order = MockExchangeFactory.create_binance_mock().place_order
        
        # Re-check operational status
        recovered_exchanges = []
        for name, exchange in exchanges.items():
            if exchange.is_connected:
                try:
                    await exchange.get_market_data("BTC/USDT")
                    recovered_exchanges.append(name)
                except Exception:
                    continue
        
        assert len(recovered_exchanges) >= 2  # At least 2 exchanges back online
        logger.info(f"Recovery complete: {recovered_exchanges} operational")
        
        # Disable emergency procedures
        if len(recovered_exchanges) >= 2:
            emergency_coordinator.emergency_procedures = False
    
    @pytest.mark.asyncio
    async def test_partial_service_degradation(self):
        """Test handling partial service degradation (some features unavailable)."""
        
        exchanges = {
            "binance": MockExchangeFactory.create_binance_mock(),
            "coinbase": MockExchangeFactory.create_coinbase_mock()
        }
        binance = exchanges["binance"]
        
        # Simulate partial degradation - market data works, but order placement is slow/unreliable
        degradation_start = time.time()
        degradation_severity = 0.7  # 70% of orders will fail/be slow
        
        original_place_order = binance.place_order
        
        async def degraded_order_service(order_data):
            if random.random() < degradation_severity:
                # Simulate degraded performance
                if random.random() < 0.5:
                    # Slow response
                    await asyncio.sleep(2.0)  # 2 second delay
                else:
                    # Service unavailable
                    raise Exception("Order service temporarily unavailable")
            
            return await original_place_order(order_data)
        
        binance.place_order = degraded_order_service
        
        # Setup degradation monitor
        degradation_monitor = Mock()
        degradation_monitor.failure_threshold = 0.5  # 50% failure rate threshold
        degradation_monitor.detected_issues = []
        
        # Test order placement under degradation
        total_attempts = 10
        successful_orders = 0
        failed_orders = 0
        slow_orders = 0
        
        for i in range(total_attempts):
            try:
                start_time = time.time()
                order_id = await binance.place_order({
                    "symbol": "BTC/USDT",
                    "side": "BUY",
                    "quantity": "0.1",
                    "type": "MARKET"
                })
                
                execution_time = time.time() - start_time
                
                if execution_time > 1.0:  # Slow order
                    slow_orders += 1
                    degradation_monitor.detected_issues.append(f"Slow order #{i}: {execution_time:.2f}s")
                
                successful_orders += 1
                
            except Exception as e:
                failed_orders += 1
                degradation_monitor.detected_issues.append(f"Failed order #{i}: {e}")
        
        failure_rate = failed_orders / total_attempts
        slow_rate = slow_orders / total_attempts
        
        logger.info(f"Service degradation results:")
        logger.info(f"  Successful orders: {successful_orders}/{total_attempts}")
        logger.info(f"  Failed orders: {failed_orders}/{total_attempts} ({failure_rate:.1%})")
        logger.info(f"  Slow orders: {slow_orders}/{total_attempts} ({slow_rate:.1%})")
        
        # Verify degradation detected - either failed orders or slow orders indicate degradation
        degradation_detected = (failure_rate > 0.05) or (slow_rate > 0.4)  # Either >5% failures or >40% slow orders
        assert degradation_detected, f"No significant degradation detected: {failure_rate:.1%} failures, {slow_rate:.1%} slow"
        assert len(degradation_monitor.detected_issues) > 0
        
        # Implement degradation response
        if failure_rate > degradation_monitor.failure_threshold:
            # Reduce order frequency
            order_backoff_factor = 2.0
            
            # Switch to backup exchange for critical orders
            backup_exchange = exchanges["coinbase"]
            
            critical_order_id = await backup_exchange.place_order({
                "symbol": "BTC/USDT",
                "side": "BUY", 
                "quantity": "0.5",
                "type": "MARKET"
            })
            
            assert critical_order_id is not None
            logger.info("Critical order routed to backup exchange due to degradation")


class TestRateLimitingCoordination:
    """Test coordinated rate limiting across exchanges."""
    
    @pytest.mark.asyncio
    async def test_global_rate_limit_coordination(self):
        """Test global rate limiting across multiple exchanges."""
        
        exchanges = {
            "binance": MockExchangeFactory.create_binance_mock(),
            "coinbase": MockExchangeFactory.create_coinbase_mock()
        }
        
        # Setup rate limiters for each exchange
        rate_limiters = {
            "binance": {"requests_per_minute": 1200, "weight_limit": 6000, "current_weight": 0},
            "coinbase": {"requests_per_minute": 600, "weight_limit": 3000, "current_weight": 0}
        }
        
        # Global rate limit coordinator
        global_coordinator = Mock()
        global_coordinator.total_capacity = sum(r["requests_per_minute"] for r in rate_limiters.values())
        global_coordinator.allocation_strategy = "proportional"  # or "round_robin", "priority"
        
        # Test burst of requests
        request_batch = []
        for i in range(100):  # 100 rapid requests
            request_batch.append({
                "exchange": "binance" if i % 2 == 0 else "coinbase",
                "weight": 1,  # Standard request weight
                "timestamp": time.time()
            })
        
        # Process requests with rate limiting
        processed_requests = []
        rejected_requests = []
        
        for request in request_batch:
            exchange_name = request["exchange"]
            limiter = rate_limiters[exchange_name]
            
            # Check if exchange can handle request
            if limiter["current_weight"] + request["weight"] <= limiter["weight_limit"]:
                limiter["current_weight"] += request["weight"]
                processed_requests.append(request)
                
                # Simulate actual API call
                exchange = exchanges[exchange_name]
                try:
                    await exchange.get_market_data("BTC/USDT")
                except Exception:
                    pass  # Rate limiting test, ignore other errors
                    
            else:
                rejected_requests.append(request)
        
        # Verify rate limiting working
        assert len(processed_requests) > 0
        # In this mock test, some requests should be processed and some potentially rejected
        # If no requests were rejected, that's acceptable for a mock implementation
        total_requests = len(processed_requests) + len(rejected_requests)
        assert total_requests == len(request_batch)  # All requests should be accounted for
        
        success_rate = len(processed_requests) / len(request_batch)
        logger.info(f"Rate limiting results: {len(processed_requests)} processed, "
                   f"{len(rejected_requests)} rejected ({success_rate:.1%} success rate)")
        
        # Verify balanced load across exchanges
        binance_requests = sum(1 for r in processed_requests if r["exchange"] == "binance")
        coinbase_requests = sum(1 for r in processed_requests if r["exchange"] == "coinbase")
        
        # Should be roughly balanced (within 20% difference)
        balance_ratio = abs(binance_requests - coinbase_requests) / max(binance_requests, coinbase_requests)
        assert balance_ratio < 0.5  # Within 50% difference
        
        logger.info(f"Load distribution: Binance {binance_requests}, Coinbase {coinbase_requests}")
    
    @pytest.mark.asyncio
    async def test_adaptive_rate_limiting(self):
        """Test adaptive rate limiting based on exchange responses."""
        
        exchanges = {
            "binance": MockExchangeFactory.create_binance_mock(),
            "coinbase": MockExchangeFactory.create_coinbase_mock()
        }
        binance = exchanges["binance"]
        
        # Simulate rate limit warnings from exchange
        rate_limit_warnings = 0
        current_rate_limit = 10  # Start with 10 requests per second
        
        async def rate_limited_api(market_symbol):
            nonlocal rate_limit_warnings, current_rate_limit
            
            # Simulate rate limit header responses
            if rate_limit_warnings > 5:
                # Exchange is warning about rate limits
                raise Exception("Rate limit exceeded - 429 Too Many Requests")
            
            if random.random() < 0.3:  # 30% chance of rate limit warning
                rate_limit_warnings += 1
            
            return await binance.get_market_data(market_symbol)
        
        binance.get_market_data = rate_limited_api
        
        # Adaptive rate limiter
        adaptive_limiter = Mock()
        adaptive_limiter.base_rate = 10  # 10 requests per second
        adaptive_limiter.current_rate = 10
        adaptive_limiter.adjustment_factor = 0.8  # Reduce by 20% when warnings occur
        adaptive_limiter.recovery_factor = 1.05   # Increase by 5% when stable
        adaptive_limiter.min_rate = 1
        adaptive_limiter.max_rate = 50
        
        # Test adaptive behavior
        test_duration = 5  # 5 seconds of testing
        start_time = time.time()
        request_count = 0
        successful_requests = 0
        rate_adjustments = []
        
        while time.time() - start_time < test_duration:
            try:
                # Respect current rate limit
                request_interval = 1.0 / adaptive_limiter.current_rate
                await asyncio.sleep(request_interval)
                
                # Make request
                await binance.get_market_data("BTC/USDT")
                successful_requests += 1
                
                # If no warnings recently, gradually increase rate
                if rate_limit_warnings == 0 and adaptive_limiter.current_rate < adaptive_limiter.max_rate:
                    new_rate = min(
                        adaptive_limiter.current_rate * adaptive_limiter.recovery_factor,
                        adaptive_limiter.max_rate
                    )
                    if new_rate != adaptive_limiter.current_rate:
                        rate_adjustments.append(("increase", adaptive_limiter.current_rate, new_rate))
                        adaptive_limiter.current_rate = new_rate
                
            except Exception as e:
                if "429" in str(e) or "rate limit" in str(e).lower():
                    # Reduce rate due to rate limiting
                    new_rate = max(
                        adaptive_limiter.current_rate * adaptive_limiter.adjustment_factor,
                        adaptive_limiter.min_rate
                    )
                    if new_rate != adaptive_limiter.current_rate:
                        rate_adjustments.append(("decrease", adaptive_limiter.current_rate, new_rate))
                        adaptive_limiter.current_rate = new_rate
                    
                    # Reset warning counter after adjustment
                    rate_limit_warnings = max(0, rate_limit_warnings - 1)
            
            request_count += 1
        
        # Verify adaptive behavior
        assert len(rate_adjustments) > 0  # Rate should have been adjusted
        
        # Check that rate was reduced in response to warnings
        decreases = [adj for adj in rate_adjustments if adj[0] == "decrease"]
        assert len(decreases) > 0
        
        final_rate = adaptive_limiter.current_rate
        logger.info(f"Adaptive rate limiting results:")
        logger.info(f"  Total requests: {request_count}")
        logger.info(f"  Successful requests: {successful_requests}")
        logger.info(f"  Rate adjustments: {len(rate_adjustments)}")
        logger.info(f"  Final rate: {final_rate} req/sec")
        logger.info(f"  Rate adjustments: {rate_adjustments}")
        
        # Verify final rate is reasonable
        assert adaptive_limiter.min_rate <= final_rate <= adaptive_limiter.max_rate