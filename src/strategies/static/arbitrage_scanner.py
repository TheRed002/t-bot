"""
Arbitrage opportunity scanner implementation.

This module implements a comprehensive arbitrage opportunity scanner that
continuously monitors for cross-exchange and triangular arbitrage opportunities
across all supported exchanges and trading pairs.

CRITICAL: This scanner requires real-time data feeds and ultra-low latency
processing to detect opportunities before they disappear.
"""

import asyncio
from typing import List, Dict, Optional, Tuple, Any, Set
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timedelta

# MANDATORY: Import from P-011 - NEVER recreate the base strategy
from src.strategies.base import BaseStrategy

# From P-001 - Use existing types  
from src.core.types import (
    Signal, MarketData, Position, SignalDirection,
    StrategyConfig, StrategyType, OrderRequest, OrderResponse
)
from src.core.logging import get_logger
from src.core.exceptions import (
    ValidationError, ExecutionError, ArbitrageError, 
    ArbitrageOpportunityError, ArbitrageExecutionError, ArbitrageTimingError
)

# From P-007A - Use decorators and validators
from src.utils.decorators import time_execution, retry, circuit_breaker, log_errors
from src.utils.validators import validate_price, validate_quantity, validate_decimal, validate_percentage
from src.utils.helpers import round_to_precision, round_to_precision_decimal, calculate_volatility, calculate_correlation
from src.utils.formatters import format_currency, format_percentage, format_pnl
from src.utils.constants import FEE_STRUCTURES, GLOBAL_FEE_STRUCTURE, GLOBAL_MINIMUM_AMOUNTS, PRECISION_LEVELS

# From P-008+ - Use risk management
from src.risk_management.base import BaseRiskManager
from src.risk_management.position_sizing import PositionSizer
from src.risk_management.risk_metrics import RiskCalculator

# From P-003+ - Use exchange interfaces
from src.exchanges.base import BaseExchange

logger = get_logger(__name__)


class ArbitrageOpportunity(BaseStrategy):
    """
    Arbitrage opportunity scanner for detecting and prioritizing arbitrage opportunities.
    
    This scanner monitors all supported exchanges and trading pairs to detect
    both cross-exchange and triangular arbitrage opportunities, prioritizing
    them by profit potential and execution feasibility.
    """
    
    def __init__(self, config: dict):
        """Initialize arbitrage opportunity scanner.
        
        Args:
            config: Strategy configuration dictionary
        """
        super().__init__(config)
        self.name = "arbitrage_scanner"
        self.strategy_type = StrategyType.ARBITRAGE
        
        # Scanner-specific configuration
        self.scan_interval = config.get("scan_interval", 100)  # milliseconds
        self.max_execution_time = config.get("max_execution_time", 500)  # milliseconds
        self.min_profit_threshold = Decimal(str(config.get("min_profit_threshold", "0.001")))  # 0.1%
        self.max_opportunities = config.get("max_opportunities", 10)
        self.exchanges = config.get("exchanges", ["binance", "okx", "coinbase"])
        self.symbols = config.get("symbols", ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT"])
        self.triangular_paths = config.get("triangular_paths", [
            ["BTCUSDT", "ETHBTC", "ETHUSDT"],
            ["BTCUSDT", "BNBBTC", "BNBUSDT"]
        ])
        
        # State tracking
        self.active_opportunities: Dict[str, Dict] = {}
        self.exchange_prices: Dict[str, Dict[str, MarketData]] = {}
        self.opportunity_history: List[Dict] = []
        self.last_scan_time = datetime.now()
        
        # Performance tracking
        self.scan_count = 0
        self.opportunities_found = 0
        self.execution_success_rate = 0.0
        
        logger.info(
            "Arbitrage opportunity scanner initialized",
            strategy=self.name,
            exchanges=self.exchanges,
            symbols=self.symbols,
            scan_interval=self.scan_interval
        )
    
    @time_execution
    async def _generate_signals_impl(self, data: MarketData) -> List[Signal]:
        """
        Generate arbitrage opportunity signals based on comprehensive market scanning.
        
        Args:
            data: Market data from one exchange
            
        Returns:
            List of arbitrage opportunity signals
        """
        try:
            # Update price data for the exchange
            exchange_name = data.metadata.get("exchange", "unknown")
            if exchange_name not in self.exchange_prices:
                self.exchange_prices[exchange_name] = {}
            
            self.exchange_prices[exchange_name][data.symbol] = data
            
            # Perform comprehensive arbitrage scan
            signals = await self._scan_arbitrage_opportunities()
            
            # TODO: Remove in production - Debug logging
            if signals:
                logger.debug(
                    "Arbitrage opportunities found",
                    strategy=self.name,
                    opportunity_count=len(signals),
                    symbols=[s.symbol for s in signals]
                )
            
            return signals
            
        except Exception as e:
            logger.error(
                "Arbitrage opportunity scanning failed",
                strategy=self.name,
                symbol=data.symbol,
                error=str(e)
            )
            return []  # Graceful degradation
    
    async def _scan_arbitrage_opportunities(self) -> List[Signal]:
        """
        Perform comprehensive arbitrage opportunity scanning.
        
        Returns:
            List of arbitrage opportunity signals
        """
        signals = []
        
        try:
            # Scan cross-exchange opportunities
            cross_signals = await self._scan_cross_exchange_opportunities()
            signals.extend(cross_signals)
            
            # Scan triangular opportunities
            triangular_signals = await self._scan_triangular_opportunities()
            signals.extend(triangular_signals)
            
            # Prioritize and limit opportunities
            prioritized_signals = self._prioritize_opportunities(signals)
            
            # Update scanner metrics
            self.scan_count += 1
            self.opportunities_found += len(prioritized_signals)
            self.last_scan_time = datetime.now()
            
            return prioritized_signals
            
        except Exception as e:
            logger.error(
                "Arbitrage opportunity scanning failed",
                strategy=self.name,
                error=str(e)
            )
            return []
    
    async def _scan_cross_exchange_opportunities(self) -> List[Signal]:
        """
        Scan for cross-exchange arbitrage opportunities.
        
        Returns:
            List of cross-exchange arbitrage signals
        """
        signals = []
        
        try:
            for symbol in self.symbols:
                # Get all available prices for this symbol
                symbol_prices = {}
                for exchange in self.exchanges:
                    if exchange in self.exchange_prices and symbol in self.exchange_prices[exchange]:
                        symbol_prices[exchange] = self.exchange_prices[exchange][symbol]
                
                if len(symbol_prices) < 2:
                    continue  # Need at least 2 exchanges
                
                # Find best bid and ask across exchanges
                best_bid_exchange = None
                best_bid_price = Decimal("0")
                best_ask_exchange = None
                best_ask_price = Decimal("inf")
                
                for exchange, market_data in symbol_prices.items():
                    if market_data.bid and market_data.bid > best_bid_price:
                        best_bid_price = market_data.bid
                        best_bid_exchange = exchange
                    
                    if market_data.ask and market_data.ask < best_ask_price:
                        best_ask_price = market_data.ask
                        best_ask_exchange = exchange
                
                # Check if arbitrage opportunity exists
                if (best_bid_exchange and best_ask_exchange and 
                    best_bid_exchange != best_ask_exchange and
                    best_bid_price > best_ask_price):
                    
                    # Calculate potential profit
                    spread = best_bid_price - best_ask_price
                    spread_percentage = (spread / best_ask_price) * 100
                    
                    # Account for fees and slippage
                    estimated_fees = self._calculate_cross_exchange_fees(best_ask_price, best_bid_price)
                    net_profit = spread - estimated_fees
                    net_profit_percentage = (net_profit / best_ask_price) * 100
                    
                    # Check if profit meets threshold
                    if net_profit_percentage >= float(self.min_profit_threshold * 100):
                        
                        # Create cross-exchange arbitrage signal
                        signal = Signal(
                            direction=SignalDirection.BUY,
                            confidence=min(0.9, net_profit_percentage / 2),
                            timestamp=datetime.now(),
                            symbol=symbol,
                            strategy_name=self.name,
                            metadata={
                                "arbitrage_type": "cross_exchange",
                                "buy_exchange": best_ask_exchange,
                                "sell_exchange": best_bid_exchange,
                                "buy_price": float(best_ask_price),
                                "sell_price": float(best_bid_price),
                                "spread_percentage": float(spread_percentage),
                                "net_profit_percentage": float(net_profit_percentage),
                                "estimated_fees": float(estimated_fees),
                                "opportunity_priority": self._calculate_priority(net_profit_percentage, "cross_exchange")
                            }
                        )
                        
                        signals.append(signal)
        
        except Exception as e:
            logger.error(
                "Cross-exchange opportunity scanning failed",
                strategy=self.name,
                error=str(e)
            )
        
        return signals
    
    async def _scan_triangular_opportunities(self) -> List[Signal]:
        """
        Scan for triangular arbitrage opportunities.
        
        Returns:
            List of triangular arbitrage signals
        """
        signals = []
        
        try:
            for path in self.triangular_paths:
                # Ensure we have prices for all pairs in the path
                if not all(pair in self.pair_prices for pair in path):
                    continue
                
                # Extract prices for the three pairs
                pair1, pair2, pair3 = path
                price1 = self.pair_prices[pair1]
                price2 = self.pair_prices[pair2]
                price3 = self.pair_prices[pair3]
                
                # Calculate triangular arbitrage
                btc_usdt_rate = price1.price
                eth_btc_rate = price2.price
                eth_usdt_rate = price3.price
                
                # Calculate triangular conversion
                start_usdt = btc_usdt_rate
                eth_amount = 1 / eth_btc_rate
                final_usdt = eth_amount * eth_usdt_rate
                
                # Calculate profit
                profit = final_usdt - start_usdt
                profit_percentage = (profit / start_usdt) * 100
                
                # Account for fees and slippage
                total_fees = self._calculate_triangular_fees(btc_usdt_rate, eth_btc_rate, eth_usdt_rate)
                net_profit = profit - total_fees
                net_profit_percentage = (net_profit / start_usdt) * 100
                
                # Check if profit meets threshold
                if net_profit_percentage >= float(self.min_profit_threshold * 100):
                    
                    # Create triangular arbitrage signal
                    signal = Signal(
                        direction=SignalDirection.BUY,
                        confidence=min(0.9, net_profit_percentage / 2),
                        timestamp=datetime.now(),
                        symbol=pair1,
                        strategy_name=self.name,
                        metadata={
                            "arbitrage_type": "triangular",
                            "path": path,
                            "start_rate": float(btc_usdt_rate),
                            "intermediate_rate": float(eth_btc_rate),
                            "final_rate": float(eth_usdt_rate),
                            "profit_percentage": float(profit_percentage),
                            "net_profit_percentage": float(net_profit_percentage),
                            "estimated_fees": float(total_fees),
                            "opportunity_priority": self._calculate_priority(net_profit_percentage, "triangular")
                        }
                    )
                    
                    signals.append(signal)
        
        except Exception as e:
            logger.error(
                "Triangular opportunity scanning failed",
                strategy=self.name,
                error=str(e)
            )
        
        return signals
    
    @log_errors
    def _calculate_cross_exchange_fees(self, buy_price: Decimal, sell_price: Decimal) -> Decimal:
        """
        Calculate fees for cross-exchange arbitrage using proper validation and formatting.
        
        Args:
            buy_price: Price to buy at
            sell_price: Price to sell at
            
        Returns:
            Total estimated fees
            
        Raises:
            ValidationError: If prices are invalid
            ArbitrageError: If fee calculation fails
        """
        try:
            # Validate input prices using utils
            validate_decimal(buy_price)
            validate_decimal(sell_price)
            validate_price(buy_price, "buy_price")
            validate_price(sell_price, "sell_price")
            
            # Get fee structure from constants and convert to Decimal
            taker_fee_rate = Decimal(str(GLOBAL_FEE_STRUCTURE.get("taker_fee", 0.001)))  # 0.1%
            
            # Calculate fees using proper rounding
            buy_fees = round_to_precision_decimal(buy_price * taker_fee_rate, PRECISION_LEVELS["fee"])
            sell_fees = round_to_precision_decimal(sell_price * taker_fee_rate, PRECISION_LEVELS["fee"])
            
            # Calculate slippage cost
            slippage_cost = round_to_precision_decimal(
                (sell_price - buy_price) * Decimal("0.0005"), 
                PRECISION_LEVELS["price"]
            )
            
            # Calculate total fees
            total_fees = buy_fees + sell_fees + slippage_cost
            
            # Validate final result
            validate_decimal(total_fees)
            
            logger.debug(
                "Cross-exchange fee calculation completed",
                strategy=self.name,
                buy_price=format_currency(buy_price),
                sell_price=format_currency(sell_price),
                buy_fees=format_currency(buy_fees),
                sell_fees=format_currency(sell_fees),
                slippage_cost=format_currency(slippage_cost),
                total_fees=format_currency(total_fees)
            )
            
            return total_fees
            
        except Exception as e:
            logger.error(
                "Cross-exchange fee calculation failed",
                strategy=self.name,
                buy_price=float(buy_price),
                sell_price=float(sell_price),
                error=str(e)
            )
            raise ArbitrageError(f"Cross-exchange fee calculation failed: {str(e)}")
    
    @log_errors
    def _calculate_triangular_fees(self, rate1: Decimal, rate2: Decimal, rate3: Decimal) -> Decimal:
        """
        Calculate fees for triangular arbitrage using proper validation and formatting.
        
        Args:
            rate1: First pair rate
            rate2: Second pair rate
            rate3: Third pair rate
            
        Returns:
            Total estimated fees
            
        Raises:
            ValidationError: If rates are invalid
            ArbitrageError: If fee calculation fails
        """
        try:
            # Validate input rates using utils
            for rate, name in [(rate1, "rate1"), (rate2, "rate2"), (rate3, "rate3")]:
                validate_decimal(rate)
                validate_price(rate, name)
            
            # Get fee structure from constants and convert to Decimal
            taker_fee_rate = Decimal(str(GLOBAL_FEE_STRUCTURE.get("taker_fee", 0.001)))  # 0.1%
            
            # Calculate fees for each step using proper rounding
            step1_fees = round_to_precision_decimal(rate1 * taker_fee_rate, PRECISION_LEVELS["fee"])
            step2_fees = round_to_precision_decimal(rate2 * taker_fee_rate, PRECISION_LEVELS["fee"])
            step3_fees = round_to_precision_decimal(rate3 * taker_fee_rate, PRECISION_LEVELS["fee"])
            
            # Calculate slippage costs for each step
            slippage_cost1 = round_to_precision_decimal(rate1 * Decimal("0.0005"), PRECISION_LEVELS["price"])
            slippage_cost2 = round_to_precision_decimal(rate2 * Decimal("0.0005"), PRECISION_LEVELS["price"])
            slippage_cost3 = round_to_precision_decimal(rate3 * Decimal("0.0005"), PRECISION_LEVELS["price"])
            
            # Calculate total fees
            total_fees = step1_fees + step2_fees + step3_fees + slippage_cost1 + slippage_cost2 + slippage_cost3
            
            # Validate final result
            validate_decimal(total_fees)
            
            logger.debug(
                "Triangular fee calculation completed",
                strategy=self.name,
                rate1=format_currency(rate1),
                rate2=format_currency(rate2),
                rate3=format_currency(rate3),
                step1_fees=format_currency(step1_fees),
                step2_fees=format_currency(step2_fees),
                step3_fees=format_currency(step3_fees),
                slippage_costs=[format_currency(cost) for cost in [slippage_cost1, slippage_cost2, slippage_cost3]],
                total_fees=format_currency(total_fees)
            )
            
            return total_fees
            
        except Exception as e:
            logger.error(
                "Triangular fee calculation failed",
                strategy=self.name,
                rate1=float(rate1),
                rate2=float(rate2),
                rate3=float(rate3),
                error=str(e)
            )
            raise ArbitrageError(f"Triangular fee calculation failed: {str(e)}")
    
    @log_errors
    def _calculate_priority(self, profit_percentage: float, arbitrage_type: str) -> float:
        """
        Calculate opportunity priority based on profit and type using proper validation.
        
        Args:
            profit_percentage: Profit percentage
            arbitrage_type: Type of arbitrage (cross_exchange or triangular)
            
        Returns:
            Priority score (higher is better)
            
        Raises:
            ValidationError: If inputs are invalid
            ArbitrageError: If priority calculation fails
        """
        try:
            # Convert profit_percentage to float if it's a Decimal
            if hasattr(profit_percentage, '__float__'):
                profit_percentage = float(profit_percentage)
            
            # Validate inputs using utils
            validate_percentage(profit_percentage, "profit_percentage")
            
            if arbitrage_type not in ["cross_exchange", "triangular"]:
                raise ValidationError(f"Invalid arbitrage type: {arbitrage_type}")
            
            # Base priority on profit percentage
            base_priority = profit_percentage
            
            # Adjust for arbitrage type (cross-exchange typically more reliable)
            if arbitrage_type == "cross_exchange":
                type_multiplier = 1.2
            else:
                type_multiplier = 1.0
            
            # Adjust for market conditions (volatility, liquidity)
            # TODO: Implement market condition analysis using calculate_volatility
            market_multiplier = 1.0
            
            # Calculate final priority with proper validation
            priority = base_priority * type_multiplier * market_multiplier
            
            # Validate final result
            if priority < 0:
                priority = 0
            elif priority > 1000:  # Reasonable upper limit
                priority = 1000
            
            logger.debug(
                "Priority calculation completed",
                strategy=self.name,
                profit_percentage=format_percentage(profit_percentage),
                arbitrage_type=arbitrage_type,
                type_multiplier=type_multiplier,
                market_multiplier=market_multiplier,
                final_priority=priority
            )
            
            return priority
            
        except Exception as e:
            logger.error(
                "Priority calculation failed",
                strategy=self.name,
                profit_percentage=profit_percentage,
                arbitrage_type=arbitrage_type,
                error=str(e)
            )
            raise ArbitrageError(f"Priority calculation failed: {str(e)}")
    
    def _prioritize_opportunities(self, signals: List[Signal]) -> List[Signal]:
        """
        Prioritize arbitrage opportunities by profit potential and feasibility.
        
        Args:
            signals: List of arbitrage signals
            
        Returns:
            Prioritized list of signals
        """
        try:
            # Sort by priority (highest first)
            sorted_signals = sorted(
                signals,
                key=lambda s: s.metadata.get("opportunity_priority", 0),
                reverse=True
            )
            
            # Limit to maximum opportunities
            limited_signals = sorted_signals[:self.max_opportunities]
            
            # Log top opportunities
            if limited_signals:
                top_opportunity = limited_signals[0]
                logger.info(
                    "Top arbitrage opportunity",
                    strategy=self.name,
                    symbol=top_opportunity.symbol,
                    arbitrage_type=top_opportunity.metadata.get("arbitrage_type"),
                    profit_percentage=top_opportunity.metadata.get("net_profit_percentage"),
                    priority=top_opportunity.metadata.get("opportunity_priority")
                )
            
            return limited_signals
            
        except Exception as e:
            logger.error(
                "Opportunity prioritization failed",
                strategy=self.name,
                error=str(e)
            )
            return signals[:self.max_opportunities]  # Return first N on error
    
    async def validate_signal(self, signal: Signal) -> bool:
        """
        Validate arbitrage opportunity signal.
        
        Args:
            signal: Signal to validate
            
        Returns:
            True if signal is valid, False otherwise
        """
        try:
            # Basic signal validation
            if not signal or signal.confidence < self.config.min_confidence:
                return False
            
            # Validate arbitrage-specific metadata
            metadata = signal.metadata
            required_fields = ["arbitrage_type", "net_profit_percentage"]
            
            for field in required_fields:
                if field not in metadata:
                    return False
            
            # Validate profit threshold
            net_profit = metadata.get("net_profit_percentage", 0)
            if net_profit < float(self.min_profit_threshold * 100):
                return False
            
            # Validate arbitrage type
            arbitrage_type = metadata.get("arbitrage_type")
            if arbitrage_type not in ["cross_exchange", "triangular"]:
                return False
            
            # Additional validation based on type
            if arbitrage_type == "cross_exchange":
                required_cross_fields = ["buy_exchange", "sell_exchange", "buy_price", "sell_price"]
                for field in required_cross_fields:
                    if field not in metadata:
                        return False
                
                # Validate exchanges are different
                if metadata.get("buy_exchange") == metadata.get("sell_exchange"):
                    return False
            
            elif arbitrage_type == "triangular":
                required_triangular_fields = ["path", "start_rate", "intermediate_rate", "final_rate"]
                for field in required_triangular_fields:
                    if field not in metadata:
                        return False
                
                # Validate path structure
                path = metadata.get("path", [])
                if len(path) != 3:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(
                "Arbitrage signal validation failed",
                strategy=self.name,
                error=str(e)
            )
            return False
    
    @log_errors
    def get_position_size(self, signal: Signal) -> Decimal:
        """
        Calculate position size for arbitrage opportunity using risk management components.

        Args:
            signal: Arbitrage opportunity signal

        Returns:
            Position size in base currency

        Raises:
            ArbitrageError: If position size calculation fails
        """
        try:
            # Validate signal using utils
            if not signal:
                raise ArbitrageError("Invalid signal for arbitrage position sizing")

            validate_percentage(signal.confidence, "signal_confidence")

            # Get configuration parameters
            total_capital = Decimal(str(self.config.parameters.get("total_capital", 10000)))
            risk_per_trade = self.config.parameters.get("risk_per_trade", 0.02)
            max_position_size = self.config.parameters.get("max_position_size", 0.1)

            # Calculate base position size using simple percentage method
            base_size = total_capital * Decimal(str(risk_per_trade))
            
            # Apply maximum position size limit
            max_size = total_capital * Decimal(str(max_position_size))
            if base_size > max_size:
                base_size = max_size

            # Scale by arbitrage-specific factors
            metadata = signal.metadata
            profit_potential = Decimal(str(metadata.get("net_profit_percentage", 0))) / Decimal("100")
            validate_percentage(profit_potential * 100, "profit_potential")

            # Adjust for arbitrage type
            arbitrage_type = metadata.get("arbitrage_type", "cross_exchange")
            if arbitrage_type == "cross_exchange":
                type_multiplier = Decimal("1.0")
            else:  # triangular
                type_multiplier = Decimal("0.7")  # Smaller size for more complex arbitrage

            # Apply arbitrage-specific adjustments
            arbitrage_multiplier = min(Decimal("2.0"), profit_potential * Decimal("10"))  # Scale with profit
            confidence_multiplier = Decimal(str(signal.confidence))

            # Calculate final position size with proper validation
            position_size = round_to_precision_decimal(
                base_size * confidence_multiplier * arbitrage_multiplier * type_multiplier,
                PRECISION_LEVELS["position"]
            )

            # Apply minimum position size from constants
            min_size = Decimal(str(GLOBAL_MINIMUM_AMOUNTS.get("position", 0.001)))
            if position_size < min_size:
                position_size = min_size

            # Validate final result
            validate_decimal(position_size)
            validate_quantity(position_size, "position_size")

            logger.debug(
                "Arbitrage scanner position size calculated",
                strategy=self.name,
                base_size=format_currency(base_size),
                profit_potential=format_percentage(profit_potential * 100),
                confidence=format_percentage(signal.confidence * 100),
                arbitrage_type=arbitrage_type,
                type_multiplier=type_multiplier,
                final_size=format_currency(position_size)
            )

            return position_size

        except Exception as e:
            logger.error(
                "Arbitrage scanner position size calculation failed",
                strategy=self.name,
                signal_confidence=signal.confidence if signal else None,
                error=str(e)
            )
            raise ArbitrageError(f"Arbitrage scanner position size calculation failed: {str(e)}")
    
    async def should_exit(self, position: Position, data: MarketData) -> bool:
        """
        Determine if arbitrage opportunity position should be closed.
        
        Args:
            position: Current position
            data: Latest market data
            
        Returns:
            True if position should be closed, False otherwise
        """
        try:
            # Check if this is an arbitrage opportunity position
            if "arbitrage_type" not in position.metadata:
                return False  # Not an arbitrage position
            
            # Check execution timeout
            metadata = position.metadata
            execution_timeout = metadata.get("execution_timeout", self.max_execution_time)
            position_age = (datetime.now() - position.timestamp).total_seconds() * 1000
            
            if position_age > execution_timeout:
                logger.info(
                    "Arbitrage position timeout",
                    strategy=self.name,
                    symbol=position.symbol,
                    age_ms=position_age,
                    timeout_ms=execution_timeout
                )
                return True
            
            # Check if opportunity still exists
            arbitrage_type = metadata.get("arbitrage_type")
            
            if arbitrage_type == "cross_exchange":
                # Check current spread
                buy_exchange = metadata.get("buy_exchange")
                sell_exchange = metadata.get("sell_exchange")
                
                if buy_exchange and sell_exchange:
                    current_spread = await self._get_current_cross_exchange_spread(
                        position.symbol, buy_exchange, sell_exchange
                    )
                    
                    if current_spread <= 0:
                        return True
            
            elif arbitrage_type == "triangular":
                # Check if triangular opportunity still exists
                path = metadata.get("path", [])
                
                if path and len(path) == 3:
                    # Recalculate current triangular arbitrage
                    current_opportunity = await self._check_triangular_path(path)
                    
                    if not current_opportunity:
                        return True
            
            return False
            
        except Exception as e:
            logger.error(
                "Exit condition check failed",
                strategy=self.name,
                symbol=position.symbol,
                error=str(e)
            )
            return False
    
    async def _get_current_cross_exchange_spread(self, symbol: str, buy_exchange: str, sell_exchange: str) -> Decimal:
        """
        Get current cross-exchange spread.
        
        Args:
            symbol: Trading symbol
            buy_exchange: Exchange to buy from
            sell_exchange: Exchange to sell to
            
        Returns:
            Current spread (positive if profitable)
        """
        try:
            buy_price = None
            sell_price = None
            
            # Get current prices
            if (buy_exchange in self.exchange_prices and 
                symbol in self.exchange_prices[buy_exchange]):
                buy_price = self.exchange_prices[buy_exchange][symbol].ask
            
            if (sell_exchange in self.exchange_prices and 
                symbol in self.exchange_prices[sell_exchange]):
                sell_price = self.exchange_prices[sell_exchange][symbol].bid
            
            if buy_price and sell_price:
                return sell_price - buy_price
            
            return Decimal("0")
            
        except Exception as e:
            logger.error(
                "Cross-exchange spread calculation failed",
                strategy=self.name,
                symbol=symbol,
                error=str(e)
            )
            return Decimal("0")
    
    async def _check_triangular_path(self, path: List[str]) -> Optional[Signal]:
        """
        Check if triangular arbitrage opportunity still exists.
        
        Args:
            path: Triangular path to check
            
        Returns:
            Signal if opportunity exists, None otherwise
        """
        try:
            # Ensure we have prices for all pairs in the path
            if not all(pair in self.pair_prices for pair in path):
                return None
            
            # Extract prices for the three pairs
            pair1, pair2, pair3 = path
            price1 = self.pair_prices[pair1]
            price2 = self.pair_prices[pair2]
            price3 = self.pair_prices[pair3]
            
            # Calculate triangular arbitrage
            btc_usdt_rate = price1.price
            eth_btc_rate = price2.price
            eth_usdt_rate = price3.price
            
            # Calculate triangular conversion
            start_usdt = btc_usdt_rate
            eth_amount = 1 / eth_btc_rate
            final_usdt = eth_amount * eth_usdt_rate
            
            # Calculate profit
            profit = final_usdt - start_usdt
            profit_percentage = (profit / start_usdt) * 100
            
            # Account for fees and slippage
            total_fees = self._calculate_triangular_fees(btc_usdt_rate, eth_btc_rate, eth_usdt_rate)
            net_profit = profit - total_fees
            net_profit_percentage = (net_profit / start_usdt) * 100
            
            # Check if profit meets threshold
            if net_profit_percentage >= float(self.min_profit_threshold * 100):
                return Signal(
                    direction=SignalDirection.BUY,
                    confidence=min(0.9, net_profit_percentage / 2),
                    timestamp=datetime.now(),
                    symbol=pair1,
                    strategy_name=self.name,
                    metadata={
                        "arbitrage_type": "triangular",
                        "path": path,
                        "net_profit_percentage": float(net_profit_percentage)
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(
                "Triangular path check failed",
                strategy=self.name,
                path=path,
                error=str(e)
            )
            return None
    
    async def post_trade_processing(self, trade_result: Dict[str, Any]) -> None:
        """
        Process completed arbitrage opportunity trade.
        
        Args:
            trade_result: Trade execution result
        """
        try:
            # Update metrics
            self.metrics.total_trades += 1
            
            # Calculate P&L
            if "pnl" in trade_result:
                self.metrics.total_pnl += Decimal(str(trade_result["pnl"]))
                
                if trade_result["pnl"] > 0:
                    self.metrics.winning_trades += 1
                else:
                    self.metrics.losing_trades += 1
            
            # Update win rate
            if self.metrics.total_trades > 0:
                self.metrics.win_rate = self.metrics.winning_trades / self.metrics.total_trades
            
            # Update execution success rate
            if "execution_success" in trade_result:
                # TODO: Implement execution success rate tracking
                pass
            
            # Log trade result
            logger.info(
                "Arbitrage opportunity trade completed",
                strategy=self.name,
                symbol=trade_result.get("symbol"),
                arbitrage_type=trade_result.get("arbitrage_type"),
                pnl=float(trade_result.get("pnl", 0)),
                execution_time_ms=trade_result.get("execution_time_ms", 0)
            )
            
        except Exception as e:
            logger.error(
                "Post-trade processing failed",
                strategy=self.name,
                error=str(e)
            )
