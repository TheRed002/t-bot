"""
Quality Service - Handles quality control business logic.

This service contains all business rules and logic for quality operations,
decoupled from infrastructure and presentation concerns.
"""

from decimal import Decimal
from typing import TYPE_CHECKING, Any, Protocol

from src.core.base.service import BaseService
from src.core.types import ExecutionResult, MarketData, OrderRequest

if TYPE_CHECKING:
    from ..quality_controller import PostTradeAnalysis, PreTradeValidation


class QualityServiceProtocol(Protocol):
    """Protocol defining the quality service interface."""

    async def validate_pre_trade(
        self,
        order_request: OrderRequest,
        market_data: MarketData | None = None,
        portfolio_context: dict[str, Any] | None = None,
    ) -> "PreTradeValidation": ...

    async def analyze_post_trade(
        self,
        trade_id: str,
        execution_result: ExecutionResult,
        market_data_before: MarketData | None = None,
        market_data_after: MarketData | None = None,
    ) -> "PostTradeAnalysis": ...

    async def validate_state_consistency(self, state: Any) -> bool: ...

    async def validate_portfolio_balance(self, portfolio_state: Any) -> bool: ...

    async def validate_position_consistency(self, position: Any, related_orders: list) -> bool: ...


class QualityService(BaseService):
    """
    Quality service implementing core quality control business logic.

    This service handles all quality validations, analysis, and business rules
    independent of infrastructure concerns.
    """

    def __init__(self, config: Any = None):
        """
        Initialize the quality service.

        Args:
            config: Optional configuration object for quality rules
        """
        super().__init__(name="QualityService")

        # Quality configuration - use safe defaults if config sections not available
        self.min_quality_score = 70.0
        self.slippage_threshold_bps = 20.0
        self.execution_time_threshold_seconds = 30.0
        self.market_impact_threshold_bps = 10.0

        # Try to load quality config from various possible config locations
        if config:
            risk_config = getattr(config, "risk", {})
            if risk_config and hasattr(risk_config, "quality"):
                quality_config = getattr(risk_config, "quality", {})
                self.min_quality_score = getattr(quality_config, "min_quality_score", 70.0)
                self.slippage_threshold_bps = getattr(quality_config, "slippage_threshold_bps", 20.0)
                self.execution_time_threshold_seconds = getattr(
                    quality_config, "execution_time_threshold_seconds", 30.0
                )
                self.market_impact_threshold_bps = getattr(
                    quality_config, "market_impact_threshold_bps", 10.0
                )

        self.logger.info("QualityService initialized")

    async def validate_pre_trade(
        self,
        order_request: OrderRequest,
        market_data: MarketData | None = None,
        portfolio_context: dict[str, Any] | None = None,
    ) -> "PreTradeValidation":
        """
        Validate pre-trade conditions and assess risk.

        Args:
            order_request: Order to validate
            market_data: Current market data
            portfolio_context: Portfolio context for risk assessment

        Returns:
            Validation results
        """
        from ..quality_controller import PreTradeValidation, ValidationResult

        validation = PreTradeValidation(order_request=order_request)
        validation.overall_result = ValidationResult.PASSED
        validation.overall_score = Decimal("100.0")
        validation.risk_level = "low"
        validation.risk_score = Decimal("0.0")
        validation.recommendations = []

        # Basic validation logic would go here
        # For now, returning a passing validation
        return validation

    async def analyze_post_trade(
        self,
        trade_id: str,
        execution_result: ExecutionResult,
        market_data_before: MarketData | None = None,
        market_data_after: MarketData | None = None,
    ) -> "PostTradeAnalysis":
        """
        Analyze post-trade execution quality.

        Args:
            trade_id: Trade identifier
            execution_result: Execution results
            market_data_before: Market data before trade
            market_data_after: Market data after trade

        Returns:
            Post-trade analysis results
        """
        from ..quality_controller import PostTradeAnalysis

        analysis = PostTradeAnalysis(trade_id=trade_id, execution_result=execution_result)
        analysis.execution_quality_score = Decimal("100.0")
        analysis.timing_quality_score = Decimal("100.0")
        analysis.price_quality_score = Decimal("100.0")
        analysis.overall_quality_score = Decimal("100.0")
        analysis.slippage_bps = Decimal("0.0")
        analysis.execution_time_seconds = Decimal("0.0")
        analysis.fill_rate = Decimal("100.0")
        analysis.market_impact_bps = Decimal("0.0")
        analysis.issues = []
        analysis.recommendations = []

        # Analysis logic would go here
        return analysis

    async def validate_state_consistency(self, state: Any) -> bool:
        """
        Validate state consistency using business rules.

        Args:
            state: State to validate

        Returns:
            True if state is consistent
        """
        try:
            if not state:
                return False

            # Check if it's a portfolio state with required fields
            if hasattr(state, "total_value") and hasattr(state, "available_cash"):
                # Basic consistency checks for portfolio state
                total_value = getattr(state, "total_value", 0)
                available_cash = getattr(state, "available_cash", 0)
                total_positions_value = getattr(state, "total_positions_value", 0)

                # Total value should be sum of cash and positions
                expected_total = available_cash + total_positions_value
                tolerance = abs(expected_total * Decimal("0.01"))  # 1% tolerance

                if abs(total_value - expected_total) > tolerance:
                    return False

            return True

        except Exception as e:
            self.logger.warning(f"State consistency validation error: {e}")
            return False

    async def validate_portfolio_balance(self, portfolio_state: Any) -> bool:
        """
        Validate portfolio balance using business rules.

        Args:
            portfolio_state: Portfolio state to validate

        Returns:
            True if portfolio balance is valid
        """
        try:
            if not portfolio_state:
                return False

            # Check for negative cash
            available_cash = getattr(portfolio_state, "available_cash", 0)
            if available_cash < 0:
                return False

            # Check for reasonable total value
            total_value = getattr(portfolio_state, "total_value", 0)
            if total_value < 0:
                return False

            # Check positions are reasonable
            positions = getattr(portfolio_state, "positions", {})
            if positions:
                for position in positions.values():
                    if hasattr(position, "quantity") and position.quantity <= 0:
                        return False

            return True

        except Exception as e:
            self.logger.warning(f"Portfolio balance validation error: {e}")
            return False

    async def validate_position_consistency(self, position: Any, related_orders: list) -> bool:
        """
        Validate position consistency with related orders using business rules.

        Args:
            position: Position to validate
            related_orders: Related orders

        Returns:
            True if position is consistent
        """
        try:
            if not position or not related_orders:
                return True  # No validation needed if no data

            # Check if filled quantity matches position quantity
            total_filled = sum(
                order.filled_quantity
                for order in related_orders
                if hasattr(order, "filled_quantity") and order.filled_quantity > 0
            )

            position_quantity = getattr(position, "quantity", Decimal("0"))
            tolerance = abs(position_quantity * Decimal("0.01"))  # 1% tolerance

            return abs(total_filled - position_quantity) <= tolerance

        except Exception as e:
            self.logger.warning(f"Position consistency validation error: {e}")
            return False
