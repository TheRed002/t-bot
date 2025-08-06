"""
Fund Flow Manager Implementation (P-010A)

This module implements deposit/withdrawal management with minimum capital
requirements, withdrawal rules enforcement, and auto-compounding features.

Key Features:
- Minimum capital requirements per strategy validation
- Withdrawal rules enforcement (profit-only, minimum maintenance)
- Auto-compounding of profits (weekly default)
- Performance-based withdrawal permissions
- Emergency capital preservation procedures
- Capital flow audit trail and reporting

Author: Trading Bot Framework
Version: 2.0.0
"""

import asyncio
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import statistics
import math

# MANDATORY: Import from P-001
from src.core.types import (
    FundFlow, CapitalAllocation, WithdrawalRule, CapitalProtection
)
from src.core.exceptions import (
    RiskManagementError, ValidationError
)
from src.core.config import Config
from src.core.logging import get_logger

# MANDATORY: Use structured logging from src.core.logging for all capital management operations
logger = get_logger(__name__)

# From P-002A - MANDATORY: Use error handling
from src.error_handling.error_handler import ErrorHandler

# From P-007A - MANDATORY: Use decorators and validators
from src.utils.decorators import time_execution, retry
from src.utils.validators import validate_quantity, validate_percentage
from src.utils.formatters import format_currency


class FundFlowManager:
    """
    Deposit/withdrawal management system.
    
    This class manages capital flows, enforces withdrawal rules,
    and implements auto-compounding features for optimal capital growth.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the fund flow manager.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.capital_config = config.capital_management
        
        # Flow tracking
        self.fund_flows: List[FundFlow] = []
        self.withdrawal_rules: Dict[str, WithdrawalRule] = {}
        self.capital_protection = CapitalProtection(
            emergency_reserve_pct=self.capital_config.emergency_reserve_pct,
            max_daily_loss_pct=self.capital_config.max_daily_loss_pct,
            max_weekly_loss_pct=self.capital_config.max_weekly_loss_pct,
            max_monthly_loss_pct=self.capital_config.max_monthly_loss_pct,
            profit_lock_pct=self.capital_config.profit_lock_pct,
            auto_compound_enabled=self.capital_config.auto_compound_enabled,
            auto_compound_frequency=self.capital_config.auto_compound_frequency,
            profit_threshold=Decimal(str(self.capital_config.profit_threshold))
        )
        
        # Performance tracking
        self.strategy_performance: Dict[str, Dict[str, float]] = {}
        self.total_profit = Decimal("0")
        self.locked_profit = Decimal("0")
        
        # Capital management integration
        self.capital_allocator = None  # TODO: Will be set by external integration
        self.total_capital = Decimal("0")  # TODO: Will be updated from actual balances
        
        # Auto-compounding tracking
        self.last_compound_date = datetime.now()
        self.compound_schedule = self._calculate_compound_schedule()
        
        # Error handler
        self.error_handler = ErrorHandler(config)
        
        # Initialize withdrawal rules
        self._initialize_withdrawal_rules()
        
        logger.info(
            "Fund flow manager initialized",
            auto_compound_enabled=self.capital_config.auto_compound_enabled,
            profit_threshold=format_currency(float(self.capital_config.profit_threshold))
        )
    
    @time_execution
    async def process_deposit(self, amount: Decimal, currency: str = "USDT", 
                            exchange: str = "binance") -> FundFlow:
        """
        Process a deposit request.
        
        Args:
            amount: Deposit amount
            currency: Currency of deposit
            exchange: Target exchange
            
        Returns:
            FundFlow: Deposit flow record
        """
        try:
            # Validate inputs
            validate_quantity(float(amount), "deposit_amount")
            
            if amount < Decimal(str(self.capital_config.min_deposit_amount)):
                raise ValidationError(
                    f"Deposit amount {amount} below minimum {self.capital_config.min_deposit_amount}"
                )
            
            # Create fund flow record
            flow = FundFlow(
                from_strategy=None,
                to_strategy=None,
                from_exchange=None,
                to_exchange=exchange,
                amount=amount,
                currency=currency,
                reason="deposit",
                timestamp=datetime.now()
            )
            
            # Add to flow history
            self.fund_flows.append(flow)
            
            logger.info(
                "Deposit processed successfully",
                amount=format_currency(float(amount), currency),
                exchange=exchange
            )
            
            return flow
            
        except Exception as e:
            logger.error(
                "Deposit processing failed",
                amount=format_currency(float(amount), currency),
                exchange=exchange,
                error=str(e)
            )
            raise
    
    @time_execution
    async def process_withdrawal(self, amount: Decimal, currency: str = "USDT",
                               exchange: str = "binance", reason: str = "withdrawal") -> FundFlow:
        """
        Process a withdrawal request with rule validation.
        
        Args:
            amount: Withdrawal amount
            currency: Currency of withdrawal
            exchange: Source exchange
            reason: Withdrawal reason
            
        Returns:
            FundFlow: Withdrawal flow record
            
        Raises:
            ValidationError: If withdrawal violates rules
        """
        try:
            # Validate inputs
            validate_quantity(float(amount), "withdrawal_amount")
            
            # Check minimum withdrawal amount
            if amount < Decimal(str(self.capital_config.min_withdrawal_amount)):
                raise ValidationError(
                    f"Withdrawal amount {amount} below minimum {self.capital_config.min_withdrawal_amount}"
                )
            
            # Validate withdrawal rules
            await self._validate_withdrawal_rules(amount, currency)
            
            # Check maximum withdrawal percentage
            if self.total_capital > Decimal("0"):
                max_withdrawal = self.total_capital * Decimal(str(self.capital_config.max_withdrawal_pct))
                if amount > max_withdrawal:
                    raise ValidationError(
                        f"Withdrawal amount {amount} exceeds maximum {max_withdrawal}"
                    )
            else:
                logger.warning("Total capital not set, skipping withdrawal percentage validation")
            
            # Check cooldown period
            await self._check_withdrawal_cooldown()
            
            # Create fund flow record
            flow = FundFlow(
                from_strategy=None,
                to_strategy=None,
                from_exchange=exchange,
                to_exchange=None,
                amount=amount,
                currency=currency,
                reason=reason,
                timestamp=datetime.now()
            )
            
            # Add to flow history
            self.fund_flows.append(flow)
            
            logger.info(
                "Withdrawal processed successfully",
                amount=format_currency(float(amount), currency),
                exchange=exchange,
                reason=reason
            )
            
            return flow
            
        except Exception as e:
            logger.error(
                "Withdrawal processing failed",
                amount=format_currency(float(amount), currency),
                exchange=exchange,
                error=str(e)
            )
            raise
    
    @time_execution
    async def process_strategy_reallocation(self, from_strategy: str, to_strategy: str,
                                         amount: Decimal, reason: str = "reallocation") -> FundFlow:
        """
        Process capital reallocation between strategies.
        
        Args:
            from_strategy: Source strategy
            to_strategy: Target strategy
            amount: Reallocation amount
            reason: Reallocation reason
            
        Returns:
            FundFlow: Reallocation flow record
        """
        try:
            # Validate inputs
            validate_quantity(float(amount), "reallocation_amount")
            
            # Check maximum daily reallocation
            if self.total_capital > Decimal("0"):
                max_daily_reallocation = self.total_capital * Decimal(str(self.capital_config.max_daily_reallocation_pct))
                daily_reallocation = await self._get_daily_reallocation_amount()
                
                if daily_reallocation + amount > max_daily_reallocation:
                    raise ValidationError(
                        f"Reallocation would exceed daily limit. Current: {daily_reallocation}, "
                        f"Requested: {amount}, Limit: {max_daily_reallocation}"
                    )
            else:
                logger.warning("Total capital not set, skipping reallocation limit validation")
            
            # Create fund flow record
            flow = FundFlow(
                from_strategy=from_strategy,
                to_strategy=to_strategy,
                from_exchange=None,
                to_exchange=None,
                amount=amount,
                reason=reason,
                timestamp=datetime.now()
            )
            
            # Add to flow history
            self.fund_flows.append(flow)
            
            logger.info(
                "Strategy reallocation processed",
                from_strategy=from_strategy,
                to_strategy=to_strategy,
                amount=format_currency(float(amount)),
                reason=reason
            )
            
            return flow
            
        except Exception as e:
            logger.error(
                "Strategy reallocation failed",
                from_strategy=from_strategy,
                to_strategy=to_strategy,
                amount=format_currency(float(amount)),
                error=str(e)
            )
            raise
    
    @time_execution
    async def process_auto_compound(self) -> Optional[FundFlow]:
        """
        Process auto-compounding of profits.
        
        Returns:
            Optional[FundFlow]: Compound flow record if applicable
        """
        try:
            if not self.capital_config.auto_compound_enabled:
                return None
            
            # Check if it's time to compound
            if not self._should_compound():
                return None
            
            # Calculate compound amount
            compound_amount = await self._calculate_compound_amount()
            
            if compound_amount <= Decimal("0"):
                return None
            
            # Create compound flow record
            flow = FundFlow(
                from_strategy=None,
                to_strategy=None,
                from_exchange=None,
                to_exchange=None,
                amount=compound_amount,
                reason="auto_compound",
                timestamp=datetime.now()
            )
            
            # Add to flow history
            self.fund_flows.append(flow)
            
            # Update tracking
            self.last_compound_date = datetime.now()
            self.locked_profit += compound_amount
            
            logger.info(
                "Auto-compound processed",
                compound_amount=format_currency(float(compound_amount)),
                total_locked_profit=format_currency(float(self.locked_profit))
            )
            
            return flow
            
        except Exception as e:
            logger.error("Auto-compound processing failed", error=str(e))
            raise
    
    @time_execution
    async def update_performance(self, strategy_id: str, performance_metrics: Dict[str, float]) -> None:
        """
        Update performance metrics for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            performance_metrics: Performance metrics
        """
        try:
            self.strategy_performance[strategy_id] = performance_metrics
            
            # Update total profit
            if 'total_pnl' in performance_metrics:
                self.total_profit = Decimal(str(performance_metrics['total_pnl']))
            
            logger.debug(
                "Performance updated",
                strategy_id=strategy_id,
                total_profit=format_currency(float(self.total_profit))
            )
            
        except Exception as e:
            logger.error(
                "Performance update failed",
                strategy_id=strategy_id,
                error=str(e)
            )
    
    @time_execution
    async def get_flow_history(self, days: int = 30) -> List[FundFlow]:
        """
        Get fund flow history for the specified period.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List[FundFlow]: Flow history
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            recent_flows = [
                flow for flow in self.fund_flows
                if flow.timestamp >= cutoff_date
            ]
            
            return recent_flows
            
        except Exception as e:
            logger.error("Failed to get flow history", error=str(e))
            raise
    
    @time_execution
    async def get_flow_summary(self, days: int = 30) -> Dict[str, Any]:
        """
        Get summary of fund flows for the specified period.
        
        Args:
            days: Number of days to look back
            
        Returns:
            Dict[str, Any]: Flow summary
        """
        try:
            flows = await self.get_flow_history(days)
            
            # Calculate summaries
            total_deposits = sum(flow.amount for flow in flows if flow.reason == "deposit")
            total_withdrawals = sum(flow.amount for flow in flows if flow.reason == "withdrawal")
            total_reallocations = sum(flow.amount for flow in flows if flow.reason == "reallocation")
            total_compounds = sum(flow.amount for flow in flows if flow.reason == "auto_compound")
            
            # Group by currency
            currency_flows = {}
            for flow in flows:
                currency = getattr(flow, 'currency', 'USDT')
                if currency not in currency_flows:
                    currency_flows[currency] = {
                        'deposits': Decimal("0"),
                        'withdrawals': Decimal("0"),
                        'reallocations': Decimal("0"),
                        'compounds': Decimal("0")
                    }
                
                if flow.reason == "deposit":
                    currency_flows[currency]['deposits'] += flow.amount
                elif flow.reason == "withdrawal":
                    currency_flows[currency]['withdrawals'] += flow.amount
                elif flow.reason == "reallocation":
                    currency_flows[currency]['reallocations'] += flow.amount
                elif flow.reason == "auto_compound":
                    currency_flows[currency]['compounds'] += flow.amount
            
            summary = {
                'period_days': days,
                'total_flows': len(flows),
                'total_deposits': float(total_deposits),
                'total_withdrawals': float(total_withdrawals),
                'total_reallocations': float(total_reallocations),
                'total_compounds': float(total_compounds),
                'net_flow': float(total_deposits - total_withdrawals),
                'currency_flows': {
                    curr: {k: float(v) for k, v in flows.items()}
                    for curr, flows in currency_flows.items()
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error("Failed to get flow summary", error=str(e))
            raise
    
    def _initialize_withdrawal_rules(self) -> None:
        """Initialize withdrawal rules from configuration."""
        for rule_name, rule_config in self.capital_config.withdrawal_rules.items():
            rule = WithdrawalRule(
                name=rule_name,
                description=rule_config.get('description', ''),
                enabled=rule_config.get('enabled', True),
                threshold=rule_config.get('threshold'),
                min_amount=Decimal(str(rule_config.get('min_amount', 0))) if rule_config.get('min_amount') else None,
                max_percentage=rule_config.get('max_percentage'),
                cooldown_hours=rule_config.get('cooldown_hours')
            )
            self.withdrawal_rules[rule_name] = rule
    
    async def _validate_withdrawal_rules(self, amount: Decimal, currency: str) -> None:
        """
        Validate withdrawal against all enabled rules.
        
        Args:
            amount: Withdrawal amount
            currency: Currency of withdrawal
            
        Raises:
            ValidationError: If any rule is violated
        """
        try:
            for rule_name, rule in self.withdrawal_rules.items():
                if not rule.enabled:
                    continue
                
                if rule_name == "profit_only":
                    if self.total_profit <= Decimal("0"):
                        raise ValidationError("Withdrawal not allowed: No profits available")
                
                elif rule_name == "maintain_minimum":
                    # Check if withdrawal would violate minimum capital requirements
                    remaining_capital = self.total_capital - amount
                    min_required = await self._calculate_minimum_capital_required()
                    if remaining_capital < min_required:
                        raise ValidationError(
                            f"Withdrawal would violate minimum capital requirement. "
                            f"Remaining: {remaining_capital}, Required: {min_required}"
                        )
                
                elif rule_name == "performance_based":
                    if rule.threshold:
                        # Check if performance meets threshold
                        performance_ok = await self._check_performance_threshold(rule.threshold)
                        if not performance_ok:
                            raise ValidationError(
                                f"Withdrawal not allowed: Performance below threshold {rule.threshold}"
                            )
                
                # Check minimum amount rule
                if rule.min_amount and amount < rule.min_amount:
                    raise ValidationError(
                        f"Withdrawal amount {amount} below minimum {rule.min_amount}"
                    )
                
                # Check maximum percentage rule
                if rule.max_percentage and self.total_capital > Decimal("0"):
                    max_amount = self.total_capital * Decimal(str(rule.max_percentage))
                    if amount > max_amount:
                        raise ValidationError(
                            f"Withdrawal amount {amount} exceeds maximum {max_amount} "
                            f"({rule.max_percentage:.1%} of total capital)"
                        )
            
        except Exception as e:
            logger.error("Withdrawal rule validation failed", error=str(e))
            raise
    
    async def _check_withdrawal_cooldown(self) -> None:
        """
        Check if withdrawal is allowed based on cooldown period.
        
        Raises:
            ValidationError: If cooldown period not met
        """
        try:
            # Find the most recent withdrawal
            recent_withdrawals = [
                flow for flow in self.fund_flows
                if flow.reason == "withdrawal"
            ]
            
            if recent_withdrawals:
                last_withdrawal = max(recent_withdrawals, key=lambda f: f.timestamp)
                cooldown_hours = self.capital_config.fund_flow_cooldown_minutes / 60
                
                if datetime.now() - last_withdrawal.timestamp < timedelta(hours=cooldown_hours):
                    raise ValidationError(
                        f"Withdrawal not allowed: Cooldown period not met. "
                        f"Last withdrawal: {last_withdrawal.timestamp}"
                    )
            
        except Exception as e:
            logger.error("Withdrawal cooldown check failed", error=str(e))
            raise
    
    async def _calculate_minimum_capital_required(self) -> Decimal:
        """Calculate minimum capital required for all strategies."""
        try:
            total_minimum = Decimal("0")
            
            for strategy_type, min_amount in self.capital_config.per_strategy_minimum.items():
                # Check if this strategy type is active
                active_strategies = [
                    strategy_id for strategy_id in self.strategy_performance.keys()
                    if strategy_type in strategy_id.lower()
                ]
                
                if active_strategies:
                    total_minimum += Decimal(str(min_amount))
            
            return total_minimum
            
        except Exception as e:
            logger.error("Failed to calculate minimum capital required", error=str(e))
            return Decimal("1000")  # Default minimum
    
    async def _check_performance_threshold(self, threshold: float) -> bool:
        """
        Check if overall performance meets threshold.
        
        Args:
            threshold: Performance threshold
            
        Returns:
            bool: True if performance meets threshold
        """
        try:
            if not self.strategy_performance:
                return False
            
            # Calculate overall performance
            total_return = Decimal("0")
            strategy_count = 0
            
            for strategy_id, metrics in self.strategy_performance.items():
                if 'total_pnl' in metrics and 'initial_capital' in metrics:
                    initial_capital = Decimal(str(metrics['initial_capital']))
                    if initial_capital > 0:
                        return_rate = Decimal(str(metrics['total_pnl'])) / initial_capital
                        total_return += return_rate
                        strategy_count += 1
            
            if strategy_count > 0:
                avg_return = total_return / strategy_count
                return float(avg_return) >= threshold
            
            return False
            
        except Exception as e:
            logger.error("Failed to check performance threshold", error=str(e))
            return False
    
    async def _get_daily_reallocation_amount(self) -> Decimal:
        """Get total reallocation amount for today."""
        try:
            today = datetime.now().date()
            today_flows = [
                flow for flow in self.fund_flows
                if flow.reason == "reallocation" and flow.timestamp.date() == today
            ]
            
            return sum(flow.amount for flow in today_flows)
            
        except Exception as e:
            logger.error("Failed to get daily reallocation amount", error=str(e))
            return Decimal("0")
    
    def _should_compound(self) -> bool:
        """Check if it's time to compound profits."""
        try:
            if not self.capital_config.auto_compound_enabled:
                return False
            
            # Check frequency
            if self.capital_config.auto_compound_frequency == "weekly":
                days_since_last = (datetime.now() - self.last_compound_date).days
                return days_since_last >= 7
            elif self.capital_config.auto_compound_frequency == "monthly":
                days_since_last = (datetime.now() - self.last_compound_date).days
                return days_since_last >= 30
            else:
                return False
            
        except Exception as e:
            logger.error("Failed to check compound timing", error=str(e))
            return False
    
    async def _calculate_compound_amount(self) -> Decimal:
        """Calculate amount to compound based on profits."""
        try:
            if self.total_profit <= self.capital_config.profit_threshold:
                return Decimal("0")
            
            # Calculate compound amount (profit above threshold)
            compound_amount = self.total_profit - self.capital_config.profit_threshold
            
            # Apply profit lock percentage
            locked_amount = compound_amount * Decimal(str(self.capital_config.profit_lock_pct))
            
            return locked_amount
            
        except Exception as e:
            logger.error("Failed to calculate compound amount", error=str(e))
            return Decimal("0")
    
    def _calculate_compound_schedule(self) -> Dict[str, Any]:
        """Calculate compound schedule based on frequency."""
        try:
            schedule = {
                'frequency': self.capital_config.auto_compound_frequency,
                'next_compound': self.last_compound_date + timedelta(
                    days=7 if self.capital_config.auto_compound_frequency == "weekly" else 30
                ),
                'enabled': self.capital_config.auto_compound_enabled
            }
            
            return schedule
            
        except Exception as e:
            logger.error("Failed to calculate compound schedule", error=str(e))
            return {}
    
    async def update_total_capital(self, total_capital: Decimal) -> None:
        """
        Update total capital from actual account balances.
        
        Args:
            total_capital: Current total capital across all exchanges
        """
        try:
            self.total_capital = total_capital
            logger.info(
                "Total capital updated",
                total_capital=format_currency(float(total_capital))
            )
        except Exception as e:
            logger.error("Failed to update total capital", error=str(e))
            raise
    
    async def get_total_capital(self) -> Decimal:
        """Get current total capital."""
        return self.total_capital
    
    async def get_capital_protection_status(self) -> Dict[str, Any]:
        """Get current capital protection status."""
        try:
            status = {
                'emergency_reserve_pct': self.capital_protection.emergency_reserve_pct,
                'max_daily_loss_pct': self.capital_protection.max_daily_loss_pct,
                'max_weekly_loss_pct': self.capital_protection.max_weekly_loss_pct,
                'max_monthly_loss_pct': self.capital_protection.max_monthly_loss_pct,
                'profit_lock_pct': self.capital_protection.profit_lock_pct,
                'total_profit': float(self.total_profit),
                'locked_profit': float(self.locked_profit),
                'auto_compound_enabled': self.capital_protection.auto_compound_enabled,
                'next_compound_date': self.compound_schedule.get('next_compound', datetime.now()),
                'protection_active': self.total_profit > 0 or self.locked_profit > 0  # Protection is active if there's profit
            }
            
            return status
            
        except Exception as e:
            logger.error("Failed to get capital protection status", error=str(e))
            raise
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all strategies."""
        try:
            total_pnl = 0.0
            
            # Calculate total PnL from strategy performance
            for strategy_id, metrics in self.strategy_performance.items():
                if isinstance(metrics, dict):
                    total_pnl += metrics.get('pnl', 0.0)
                else:
                    # Handle case where metrics might be a single value
                    total_pnl += float(metrics) if isinstance(metrics, (int, float, Decimal)) else 0.0
            
            summary = {
                'total_pnl': total_pnl,
                'total_profit': float(self.total_profit),
                'locked_profit': float(self.locked_profit),
                'strategy_count': len(self.strategy_performance),
                'strategies': {}
            }
            
            for strategy_id, metrics in self.strategy_performance.items():
                if isinstance(metrics, dict):
                    summary['strategies'][strategy_id] = {
                        'pnl': metrics.get('pnl', 0.0),
                        'performance_score': metrics.get('performance_score', 0.0),
                        'last_updated': metrics.get('last_updated', datetime.now())
                    }
                else:
                    # Handle case where metrics might be a single value
                    summary['strategies'][strategy_id] = {
                        'pnl': float(metrics) if isinstance(metrics, (int, float, Decimal)) else 0.0,
                        'performance_score': 0.0,
                        'last_updated': datetime.now()
                    }
            
            return summary
            
        except Exception as e:
            logger.error("Failed to get performance summary", error=str(e))
            raise 