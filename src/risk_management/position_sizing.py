"""
Position Sizing Module - Refactored to use centralized utilities.

This module now delegates to centralized position sizing utilities,
eliminating code duplication while maintaining backward compatibility.
"""

from decimal import Decimal
from typing import TYPE_CHECKING, Any

from src.core.base.component import BaseComponent
from src.core.config.main import Config
from src.core.exceptions import RiskManagementError, ValidationError
from src.core.types import PositionSizeMethod, Signal
from src.utils.decimal_utils import ZERO, format_decimal, to_decimal
from src.utils.decorators import time_execution

# Import centralized position sizing utilities
from src.utils.position_sizing import (
    calculate_position_size,
    get_signal_confidence,
    update_position_history,
    validate_position_size,
)

# Type checking imports to avoid circular dependencies
if TYPE_CHECKING:
    from src.database.service import DatabaseService


class PositionSizer(BaseComponent):
    """
    Position sizing calculator with multiple algorithms.

    This class provides position sizing calculations using centralized utilities.
    For enhanced functionality, consider using RiskService.calculate_position_size().
    """

    def __init__(self, config: Config, database_service: "DatabaseService | None" = None):
        """
        Initialize position sizer with configuration.

        Args:
            config: Application configuration containing risk settings
            database_service: Database service for data access (not used in this implementation)
        """
        super().__init__()  # Initialize BaseComponent
        self.config = config
        self.risk_config = config.risk
        self.database_service = database_service

        # Historical data for calculations - using centralized utilities
        self.price_history: dict[str, list[Decimal]] = {}
        self.return_history: dict[str, list[float]] = {}

        if database_service:
            self.logger.warning(
                "PositionSizer initialized with DatabaseService - "
                "consider migrating to RiskService for full integration"
            )
        else:
            self.logger.warning(
                "PositionSizer initialized - migrate to RiskService for enterprise features"
            )

    @time_execution
    async def calculate_position_size(
        self, signal: Signal, portfolio_value: Decimal, method: PositionSizeMethod | None = None
    ) -> Decimal:
        """
        Calculate position size using centralized utilities.

        Args:
            signal: Trading signal with direction and confidence
            portfolio_value: Current total portfolio value
            method: Position sizing method to use (defaults to config setting)

        Returns:
            Decimal: Calculated position size in base currency

        Raises:
            RiskManagementError: If position size calculation fails
        """
        try:
            # Validate inputs early
            if signal is None:
                raise ValidationError("Signal cannot be None")
            
            if portfolio_value <= ZERO:
                raise ValidationError(f"Portfolio value must be positive: {portfolio_value}")

            # Use default method if not specified or validate provided method
            if method is None:
                try:
                    method = PositionSizeMethod(self.risk_config.default_position_size_method)
                except (ValueError, AttributeError):
                    self.logger.warning(
                        "Invalid default position sizing method in config, using FIXED_PERCENTAGE"
                    )
                    method = PositionSizeMethod.FIXED_PERCENTAGE
            elif isinstance(method, str):
                # Convert string to enum, will raise ValueError if invalid
                try:
                    method = PositionSizeMethod(method)
                except ValueError:
                    raise ValidationError(f"Unsupported position sizing method: {method}")
            elif not isinstance(method, PositionSizeMethod):
                raise ValidationError(f"Invalid method type: {type(method)}")    

            # Get risk per trade from config
            risk_per_trade = to_decimal(self.risk_config.risk_per_trade)

            # Prepare method-specific kwargs
            kwargs = {}
            if method == PositionSizeMethod.KELLY_CRITERION:
                # Calculate Kelly parameters from return history
                kelly_params = self._get_kelly_parameters(signal.symbol)
                kwargs.update(kelly_params)
            elif method == PositionSizeMethod.VOLATILITY_ADJUSTED:
                # Get volatility data for the symbol
                kwargs.update(self._get_volatility_parameters(signal.symbol))

            # Use centralized position sizing utility
            position_size = calculate_position_size(
                method=method,
                signal=signal,
                portfolio_value=portfolio_value,
                risk_per_trade=risk_per_trade,
                **kwargs
            )

            # Get config values for validation bounds (needed early for Kelly)
            min_position_pct = to_decimal(getattr(self.risk_config, 'min_position_size_pct', '0.01'))
            max_position_pct = to_decimal(getattr(self.risk_config, 'max_position_size_pct', '0.1'))

            # Apply Kelly-specific logic for risk_per_trade cap
            if method == PositionSizeMethod.KELLY_CRITERION:
                max_kelly_size = portfolio_value * risk_per_trade
                min_kelly_size = portfolio_value * min_position_pct
                
                # Check if we have sufficient return history data
                has_sufficient_data = (
                    signal.symbol in self.return_history and 
                    len(self.return_history[signal.symbol]) >= 10
                )
                
                if has_sufficient_data:
                    # For Kelly with sufficient data, apply special handling
                    kelly_params = self._get_kelly_parameters(signal.symbol)
                    win_prob = kelly_params.get('win_probability', 0.5)
                    win_loss_ratio = kelly_params.get('win_loss_ratio', 1.0)
                    
                    # Check if we have real win/loss data (not defaults)
                    returns = self.return_history[signal.symbol]
                    wins = [r for r in returns if r > 0]
                    losses = [r for r in returns if r < 0]
                    has_real_data = len(wins) > 0 and len(losses) > 0
                    
                    # Only apply positive edge logic if we have real win/loss data
                    if has_real_data:
                        # Calculate if this should be a positive edge (profitable) strategy
                        is_positive_edge = (win_prob * win_loss_ratio) > (1.0 - win_prob)
                        
                        if is_positive_edge and position_size < max_kelly_size:
                            # Positive edge but below risk_per_trade cap -> use cap
                            position_size = max_kelly_size
                            self.logger.info(f"Kelly positive edge: using risk_per_trade cap: {max_kelly_size}")
                        elif position_size > max_kelly_size:
                            # Above cap -> cap it
                            position_size = max_kelly_size
                            self.logger.info(f"Kelly position size capped at risk_per_trade: {max_kelly_size}")
                        elif position_size < min_kelly_size:
                            # Below minimum -> use minimum (for negative edge cases)
                            position_size = min_kelly_size
                            self.logger.info(f"Kelly position size set to minimum: {min_kelly_size}")
                    else:
                        # All wins or all losses - use fallback behavior (Kelly already applied)
                        self.logger.info("Kelly fallback: insufficient win/loss data, using calculated Kelly result")
                # If insufficient data, let Kelly fallback to fixed percentage naturally

            # Validate position size using centralized utility with config bounds
            
            is_valid, validated_size = validate_position_size(
                position_size, 
                portfolio_value, 
                min_position_pct, 
                max_position_pct
            )
            if not is_valid:
                self.logger.warning("Position size validation failed, returning zero")
                return ZERO

            self.logger.info(
                "Position size calculated",
                method=method.value,
                signal_symbol=signal.symbol,
                portfolio_value=format_decimal(portfolio_value),
                position_size=format_decimal(validated_size),
            )

            return validated_size

        except ValidationError as e:
            self.logger.error("Position size validation error", error=str(e))
            raise RiskManagementError(f"Position size calculation failed: {e}") from e
        except Exception as e:
            self.logger.error(
                "Position size calculation failed",
                error=str(e),
                signal_symbol=signal.symbol if signal else None,
            )
            raise RiskManagementError(f"Position size calculation failed: {e}") from e

    @time_execution
    async def update_price_history(self, symbol: str, price: Decimal) -> None:
        """
        Update price history using centralized utility.

        Args:
            symbol: Trading symbol
            price: Current price as Decimal for financial precision

        Raises:
            ValidationError: If input parameters are invalid
        """
        try:
            # Validate inputs
            if not symbol or not isinstance(symbol, str) or symbol.strip() == "":
                raise ValidationError(f"Invalid symbol: {symbol} (must be non-empty string)")

            if price <= ZERO:
                raise ValidationError(f"Price must be positive: {price}")

            # Calculate max history based on volatility_window
            max_history = max(getattr(self.risk_config, "volatility_window", 20) * 2, 100)
            
            # Use centralized utility
            update_position_history(symbol, price, self.price_history, max_history)

            # Calculate returns if we have at least 2 prices
            if symbol in self.price_history and len(self.price_history[symbol]) >= 2:
                previous_price = self.price_history[symbol][-2]
                current_price = self.price_history[symbol][-1]
                ret = float((current_price - previous_price) / previous_price)
                
                # Initialize return history for symbol if needed
                if symbol not in self.return_history:
                    self.return_history[symbol] = []
                
                self.return_history[symbol].append(ret)
                
                # Maintain return history size
                if len(self.return_history[symbol]) > max_history:
                    self.return_history[symbol] = self.return_history[symbol][-max_history:]

            self.logger.info(
                f"Updated price history for {symbol}",
                price=str(price),
                history_length=len(self.price_history.get(symbol, [])),
            )

        except ValidationError:
            raise
        except Exception as e:
            self.logger.error(
                f"Unexpected error updating price history for {symbol}: {e}",
                symbol=symbol,
                price=str(price) if price else None,
                error_type=type(e).__name__,
            )
            raise ValidationError(f"Failed to update price history for {symbol}: {e}") from e

    @time_execution
    async def get_position_size_summary(
        self, signal: Signal, portfolio_value: Decimal
    ) -> dict[str, Any]:
        """
        Get comprehensive position size summary for all methods.

        Args:
            signal: Trading signal
            portfolio_value: Current portfolio value

        Returns:
            Dict containing position sizes for all methods
        """
        summary = {}

        for method in PositionSizeMethod:
            try:
                size = await self.calculate_position_size(signal, portfolio_value, method)
                summary[method.value] = {
                    "position_size": format_decimal(size),
                    "portfolio_percentage": format_decimal(
                        (size / portfolio_value) if portfolio_value > ZERO else ZERO
                    ),
                }
            except Exception as e:
                summary[method.value] = {
                    "error": str(e),
                    "position_size": "0",
                    "portfolio_percentage": "0",
                }

        return summary

    @time_execution
    async def validate_position_size(
        self, position_size: Decimal, portfolio_value: Decimal
    ) -> bool:
        """
        Validate calculated position size with strict bounds checking.

        Args:
            position_size: Calculated position size
            portfolio_value: Current portfolio value

        Returns:
            bool: True if position size is within bounds, False otherwise
        """
        try:
            if portfolio_value <= ZERO:
                return False

            # Get config values for validation bounds
            min_position_pct = to_decimal(getattr(self.risk_config, 'min_position_size_pct', '0.01'))
            max_position_pct = to_decimal(getattr(self.risk_config, 'max_position_size_pct', str(self.risk_config.risk_per_trade)))
            
            # Calculate position percentage
            from src.utils.decimal_utils import safe_divide
            position_pct = safe_divide(position_size, portfolio_value, ZERO)

            # Strict validation: reject if outside bounds
            if position_pct < min_position_pct:
                return False
            
            if position_pct > max_position_pct:
                return False

            return True
        except Exception as e:
            self.logger.error("Position size validation failed", error=str(e))
            return False

    def get_signal_confidence(self, signal: Signal) -> Decimal:
        """
        Get signal confidence using centralized utility.

        Args:
            signal: Trading signal

        Returns:
            Signal confidence as Decimal
        """
        return get_signal_confidence(signal)

    def _get_kelly_parameters(self, symbol: str) -> dict[str, float]:
        """
        Calculate Kelly Criterion parameters from return history.

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary with win_probability and win_loss_ratio
        """
        if symbol not in self.return_history or len(self.return_history[symbol]) < 10:
            # Not enough data, use defaults
            return {"win_probability": 0.55, "win_loss_ratio": 1.5}

        returns = self.return_history[symbol]
        
        # Separate wins and losses
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r < 0]

        if len(wins) == 0 or len(losses) == 0:
            # All wins or all losses, use defaults
            return {"win_probability": 0.55, "win_loss_ratio": 1.5}

        # Calculate win probability
        win_probability = len(wins) / len(returns)

        # Calculate average win and loss
        avg_win = sum(wins) / len(wins)
        avg_loss = abs(sum(losses) / len(losses))  # Make positive

        # Calculate win/loss ratio
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.5

        return {
            "win_probability": win_probability,
            "win_loss_ratio": win_loss_ratio
        }

    def _get_volatility_parameters(self, symbol: str) -> dict[str, float]:
        """
        Calculate volatility parameters from price history.

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary with volatility parameters
        """
        if symbol not in self.price_history or len(self.price_history[symbol]) < 10:
            return {
                "current_volatility": 0.02,
                "target_volatility": 0.015
            }

        prices = self.price_history[symbol]
        
        # Calculate daily returns
        returns = []
        for i in range(1, len(prices)):
            if prices[i - 1] > ZERO:
                ret = float((prices[i] - prices[i - 1]) / prices[i - 1])
                returns.append(ret)

        if len(returns) < 2:
            return {
                "current_volatility": 0.02,
                "target_volatility": 0.015
            }

        # Calculate volatility as standard deviation of returns
        import statistics
        current_volatility = statistics.stdev(returns) if len(returns) > 1 else 0.02

        return {
            "current_volatility": current_volatility,
            "target_volatility": 0.015  # Default target
        }
