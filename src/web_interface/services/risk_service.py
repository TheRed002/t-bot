"""
Risk service for web interface business logic.

This service handles all risk management-related business logic that was previously
embedded in controllers, ensuring proper separation of concerns.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from src.core.base import BaseComponent
from src.core.exceptions import RiskManagementError, ServiceError, ValidationError
from src.risk_management.interfaces import RiskServiceInterface
from src.utils.decimal_utils import to_decimal
from src.web_interface.data_transformer import WebInterfaceDataTransformer
from src.web_interface.interfaces import WebRiskServiceInterface


class WebRiskService(BaseComponent):
    """
    Service handling risk management business logic for web interface.

    This service acts as a bridge between the web interface and the actual
    risk management service. It provides web-specific formatting and validation
    while delegating complex risk calculations to the risk management service.

    Note: This service should NOT duplicate risk calculations from src.risk_management.
    Complex risk calculations should be delegated to the risk management service
    through the risk_facade.
    """

    def __init__(self, risk_service: RiskServiceInterface = None):
        super().__init__()
        self.risk_service = risk_service

    async def initialize(self) -> None:
        """Initialize the service."""
        self.logger.info("Web risk service initialized")

    async def cleanup(self) -> None:
        """Cleanup the service."""
        self.logger.info("Web risk service cleaned up")

    async def get_risk_dashboard_data(self) -> dict[str, Any]:
        """Get risk dashboard data with web-specific formatting."""
        try:
            if self.risk_service:
                # Use actual risk management service
                risk_metrics = await self.risk_service.calculate_risk_metrics(positions=None, market_data=None)
                portfolio_metrics = await self.risk_service.get_portfolio_metrics()

                # Extract portfolio metrics from risk service
                portfolio_data = portfolio_metrics.model_dump()

                # Convert risk service data to expected format
                risk_data = {
                    "portfolio_value": portfolio_data.get("total_value", Decimal("0")),
                    "total_exposure": portfolio_data.get("total_exposure", Decimal("0")),
                    "leverage": portfolio_data.get("leverage", Decimal("1")),
                    "var_1d": risk_metrics.var_1d if risk_metrics else Decimal("0"),
                    "var_5d": risk_metrics.var_5d if risk_metrics else Decimal("0"),
                    "expected_shortfall": risk_metrics.expected_shortfall if risk_metrics else Decimal("0"),
                    "max_drawdown": risk_metrics.max_drawdown if risk_metrics else Decimal("0"),
                    "volatility": portfolio_data.get("volatility", Decimal("0")),
                    "correlation_btc": risk_metrics.correlation_risk if risk_metrics else None,
                    "active_positions": portfolio_data.get("position_count", 0),
                    "concentration_risk": {
                        "largest_position_pct": portfolio_data.get("largest_position_weight", 0),
                        "top_3_positions_pct": 0,  # Would need additional calculation
                        "sector_concentration": {
                            "crypto": 1.0,  # Assume all crypto for now
                            "stablecoins": 0.0,
                        },
                    },
                }
            else:
                # Mock data for development when facade is not available
                risk_data = {
                    "portfolio_value": Decimal("150000.00"),
                    "total_exposure": Decimal("75000.00"),
                    "leverage": Decimal("1.5"),
                    "var_1d": Decimal("2500.00"),
                    "var_5d": Decimal("5500.00"),
                    "expected_shortfall": Decimal("4200.00"),
                    "max_drawdown": Decimal("8750.00"),
                    "volatility": Decimal("0.18"),
                    "correlation_btc": Decimal("0.65"),
                    "active_positions": 8,
                    "concentration_risk": {
                        "largest_position_pct": 0.15,
                        "top_3_positions_pct": 0.35,
                        "sector_concentration": {
                            "crypto": 0.85,
                            "stablecoins": 0.15,
                        },
                    },
                }

            # Business logic: calculate risk scores and warnings
            risk_analysis = self._analyze_risk_levels(risk_data)

            # Business logic: format dashboard data with consistent transformation
            raw_dashboard_data = {
                "overview": {
                    "portfolio_value": risk_data.get("portfolio_value", Decimal("0")),
                    "total_exposure": risk_data.get("total_exposure", Decimal("0")),
                    "exposure_percentage": self._calculate_exposure_percentage(
                        risk_data.get("total_exposure", Decimal("0")),
                        risk_data.get("portfolio_value", Decimal("1")),
                    ),
                    "leverage": risk_data.get("leverage", Decimal("1")),
                    "risk_score": risk_analysis["overall_score"],
                    "risk_level": risk_analysis["risk_level"],
                },
                "var_metrics": {
                    "var_1d": risk_data.get("var_1d", Decimal("0")),
                    "var_5d": risk_data.get("var_5d", Decimal("0")),
                    "expected_shortfall": risk_data.get("expected_shortfall", Decimal("0")),
                    "confidence_level": "95%",
                },
                "portfolio_risk": {
                    "max_drawdown": risk_data.get("max_drawdown", Decimal("0")),
                    "max_drawdown_percentage": self._calculate_drawdown_percentage(
                        risk_data.get("max_drawdown", Decimal("0")),
                        risk_data.get("portfolio_value", Decimal("1")),
                    ),
                    "volatility": risk_data.get("volatility", Decimal("0")),
                    "beta": risk_data.get("beta", None),
                    "correlation_btc": risk_data.get("correlation_btc", None),
                },
                "concentration": risk_data.get("concentration_risk", {}),
                "warnings": risk_analysis["warnings"],
                "recommendations": risk_analysis["recommendations"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Transform data with boundary validation using WebInterfaceDataTransformer
            dashboard_data = WebInterfaceDataTransformer.transform_risk_data_to_event_data(
                raw_dashboard_data,
                metadata={"operation": "get_risk_dashboard_data", "source": "web_risk_service"},
            )

            # Ensure financial precision
            dashboard_data = WebInterfaceDataTransformer.validate_financial_precision(
                dashboard_data
            )

            return dashboard_data

        except RiskManagementError as e:
            # Propagate risk management errors from risk_management module
            self.logger.error(f"Risk management error getting dashboard data: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error getting risk dashboard data: {e}")
            raise ServiceError(f"Failed to get risk dashboard data: {e}")

    async def validate_risk_parameters(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Validate risk parameters with web-specific business logic."""
        try:
            validation_errors = []

            # Business logic: validate risk percentage
            if "max_portfolio_risk" in parameters:
                max_risk = parameters["max_portfolio_risk"]
                if isinstance(max_risk, (int, float, Decimal)):
                    if to_decimal(max_risk) < 0 or to_decimal(max_risk) > 1:
                        validation_errors.append("Max portfolio risk must be between 0 and 1")
                    elif to_decimal(max_risk) > Decimal("0.25"):
                        validation_errors.append("Max portfolio risk above 25% is not recommended")

            # Business logic: validate position size
            if "max_position_size" in parameters:
                max_pos = parameters["max_position_size"]
                if isinstance(max_pos, (int, float, Decimal)) and to_decimal(max_pos) <= 0:
                    validation_errors.append("Max position size must be greater than 0")

            # Business logic: validate leverage
            if "max_leverage" in parameters:
                max_lev = parameters["max_leverage"]
                if isinstance(max_lev, (int, float, Decimal)):
                    leverage_decimal = to_decimal(max_lev)
                    if leverage_decimal < 1:
                        validation_errors.append("Max leverage cannot be less than 1")
                    elif leverage_decimal > 10:
                        validation_errors.append("Max leverage above 10x is extremely risky")

            # Business logic: validate daily loss limit
            if "max_daily_loss" in parameters:
                daily_loss = parameters["max_daily_loss"]
                if isinstance(daily_loss, (int, float, Decimal)) and to_decimal(daily_loss) <= 0:
                    validation_errors.append("Max daily loss must be greater than 0")

            # Business logic: validate VaR limit
            if "var_limit" in parameters:
                var_limit = parameters["var_limit"]
                if isinstance(var_limit, (int, float, Decimal)) and to_decimal(var_limit) <= 0:
                    validation_errors.append("VaR limit must be greater than 0")

            # Business logic: validate risk per trade
            if "risk_per_trade" in parameters:
                risk_per_trade = parameters["risk_per_trade"]
                if isinstance(risk_per_trade, (int, float, Decimal)):
                    risk_decimal = to_decimal(risk_per_trade)
                    if risk_decimal < 0 or risk_decimal > Decimal("0.1"):
                        validation_errors.append("Risk per trade must be between 0 and 0.1 (10%)")

            # Note: Actual parameter updates would need to be implemented in risk_management service
            # This service only validates, the actual updates should go through the risk service

            return {
                "valid": len(validation_errors) == 0,
                "errors": validation_errors,
                "validated_parameters": parameters if len(validation_errors) == 0 else None,
                "recommendations": self._generate_risk_recommendations(parameters),
            }

        except RiskManagementError as e:
            # Propagate risk management errors from risk_management module
            self.logger.error(f"Risk management error validating parameters: {e}")
            raise
        except ValidationError:
            # Re-raise ValidationError as-is for consistent error handling
            raise
        except Exception as e:
            self.logger.error(f"Error validating risk parameters: {e}")
            raise ServiceError(f"Failed to validate risk parameters: {e}")

    async def calculate_position_risk(
        self, symbol: str, quantity: Decimal, price: Decimal
    ) -> dict[str, Any]:
        """Calculate position risk with web-specific metrics."""
        try:
            # Business logic: calculate position value and risk metrics
            position_value = quantity * price

            # Get current portfolio value from risk service if available
            if self.risk_service:
                portfolio_metrics = await self.risk_service.get_portfolio_metrics()
                portfolio_data = portfolio_metrics.model_dump()
                portfolio_value = portfolio_data.get("total_value", Decimal("100000"))
            else:
                portfolio_value = Decimal("100000")  # Mock portfolio value

            # Business logic: calculate position risk metrics
            position_percentage = (position_value / portfolio_value) * 100

            # Use risk service for position sizing if available
            if self.risk_service:
                try:
                    # Create a mock signal for position sizing calculation
                    from src.core.types import Signal, SignalDirection

                    # Convert symbol to proper format if needed
                    normalized_symbol = symbol
                    if "/" not in symbol and symbol.endswith("USDT"):
                        # Convert BTCUSDT to BTC/USDT format
                        base = symbol[:-4]  # Remove USDT
                        normalized_symbol = f"{base}/USDT"
                    elif "/" not in symbol:
                        # Default to USD quote
                        normalized_symbol = f"{symbol}/USD"

                    signal = Signal(
                        symbol=normalized_symbol,
                        direction=SignalDirection.BUY,
                        strength=Decimal("0.7"),
                        source="web_interface",
                        timestamp=datetime.now(timezone.utc),
                    )

                    # Calculate optimal position size using risk management
                    optimal_size = await self.risk_service.calculate_position_size(
                        signal=signal, available_capital=portfolio_value, current_price=price
                    )

                    # Calculate risk metrics based on optimal sizing
                    optimal_value = optimal_size * price
                    optimal_percentage = (optimal_value / portfolio_value) * 100

                except Exception as e:
                    self.logger.warning(f"Could not calculate optimal position size: {e}")
                    optimal_size = quantity
                    optimal_value = position_value
                    optimal_percentage = position_percentage
            else:
                optimal_size = quantity
                optimal_value = position_value
                optimal_percentage = position_percentage

            # Get volatility and VaR from risk service if available
            if self.risk_service:
                try:
                    # Use risk service to calculate metrics instead of duplicating calculations
                    risk_metrics = await self.risk_service.calculate_risk_metrics(positions=None, market_data=None)
                    portfolio_metrics = await self.risk_service.get_portfolio_metrics()
                    portfolio_data = portfolio_metrics.model_dump()

                    # Use portfolio volatility as approximation for individual position
                    volatility = portfolio_data.get("volatility", Decimal("0.25"))

                    # Calculate VaR proportionally based on position size
                    portfolio_var = risk_metrics.var_1d if risk_metrics else Decimal("0")
                    var_95 = (
                        portfolio_var * (position_value / portfolio_value)
                        if portfolio_value > 0
                        else Decimal("0")
                    )

                except Exception as e:
                    self.logger.warning(
                        f"Could not get risk metrics from service, using approximations: {e}"
                    )
                    # Fallback to simple approximations
                    if "BTC" in symbol.upper():
                        volatility = Decimal("0.25")
                    elif "ETH" in symbol.upper():
                        volatility = Decimal("0.30")
                    else:
                        volatility = Decimal("0.35")

                    daily_volatility = volatility / Decimal("252").sqrt()
                    var_95 = position_value * daily_volatility * Decimal("1.645")
            else:
                # Mock volatility calculation (in production, use historical data)
                if "BTC" in symbol.upper():
                    volatility = Decimal("0.25")  # 25% annual volatility
                elif "ETH" in symbol.upper():
                    volatility = Decimal("0.30")  # 30% annual volatility
                else:
                    volatility = Decimal("0.35")  # 35% for other crypto

                # Calculate daily VaR (95% confidence, normal distribution approximation)
                daily_volatility = (
                    volatility / Decimal("252").sqrt()
                )  # Approximate daily volatility
                var_95 = (
                    position_value * daily_volatility * Decimal("1.645")
                )  # 95% confidence z-score

            # Business logic: risk level assessment
            risk_level = self._assess_position_risk_level(position_percentage, volatility)

            # Business logic: calculate margin requirements (mock)
            margin_requirement = position_value * Decimal("0.1")  # 10% margin requirement

            return {
                "symbol": symbol,
                "position_size": {
                    "quantity": quantity,
                    "price": price,
                    "value": position_value,
                    "portfolio_percentage": position_percentage,
                },
                "optimal_sizing": {
                    "quantity": optimal_size,
                    "value": optimal_value,
                    "portfolio_percentage": optimal_percentage,
                },
                "risk_metrics": {
                    "daily_var_95": var_95,
                    "volatility": volatility,
                    "risk_level": risk_level,
                    "margin_requirement": margin_requirement,
                },
                "limits_check": {
                    "within_position_limit": position_percentage <= 10,  # 10% max position
                    "within_risk_budget": var_95
                    <= portfolio_value * Decimal("0.02"),  # 2% daily risk
                    "margin_available": True,  # Mock margin check
                },
                "warnings": self._generate_position_warnings(position_percentage, volatility),
                "recommendations": self._generate_position_recommendations(
                    symbol, position_percentage
                ),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except RiskManagementError as e:
            # Propagate risk management errors from risk_management module
            self.logger.error(f"Risk management error calculating position risk for {symbol}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error calculating position risk for {symbol}: {e}")
            raise ServiceError(f"Failed to calculate position risk: {e}")

    async def get_portfolio_risk_breakdown(self) -> dict[str, Any]:
        """Get portfolio risk breakdown with web-specific analysis."""
        try:
            if self.risk_service:
                # Get risk metrics from actual risk service
                risk_metrics = await self.risk_service.calculate_risk_metrics(positions=None, market_data=None)
                portfolio_metrics = await self.risk_service.get_portfolio_metrics()
                portfolio_data = portfolio_metrics.model_dump()

                # Convert risk service data to expected format
                # Note: Some data may not be available and will use mock values
                risk_data = {
                    "positions": [],  # Would need additional API to get individual positions
                    "correlations": {},  # Would need correlation data from risk service
                    "sector_breakdown": {
                        "crypto": 1.0,  # Assume all crypto for now
                        "cash": 0.0,
                    },
                    "portfolio_metrics": portfolio_metrics,
                }

                # If no position data available, provide mock data for UI
                if not risk_data["positions"]:
                    total_value = portfolio_data.get("total_value", Decimal("150000"))
                    portfolio_var = risk_metrics.var_1d if risk_metrics else Decimal("0")
                    risk_data["positions"] = [
                        {
                            "symbol": "BTCUSDT",
                            "value": total_value * Decimal("0.4"),  # 40% allocation
                            "weight": 0.4,
                            "var_contribution": portfolio_var * Decimal("0.6"),
                            "volatility": 0.25,
                        },
                        {
                            "symbol": "ETHUSDT",
                            "value": total_value * Decimal("0.3"),  # 30% allocation
                            "weight": 0.3,
                            "var_contribution": portfolio_var * Decimal("0.3"),
                            "volatility": 0.30,
                        },
                        {
                            "symbol": "ADAUSDT",
                            "value": total_value * Decimal("0.15"),  # 15% allocation
                            "weight": 0.15,
                            "var_contribution": portfolio_var * Decimal("0.1"),
                            "volatility": 0.40,
                        },
                    ]
            else:
                # Mock data for development when facade is not available
                risk_data = {
                    "positions": [
                        {
                            "symbol": "BTCUSDT",
                            "value": Decimal("25000"),
                            "weight": 0.167,
                            "var_contribution": Decimal("800"),
                            "volatility": 0.25,
                        },
                        {
                            "symbol": "ETHUSDT",
                            "value": Decimal("18000"),
                            "weight": 0.12,
                            "var_contribution": Decimal("650"),
                            "volatility": 0.30,
                        },
                        {
                            "symbol": "ADAUSDT",
                            "value": Decimal("8000"),
                            "weight": 0.053,
                            "var_contribution": Decimal("350"),
                            "volatility": 0.40,
                        },
                    ],
                    "correlations": {
                        ("BTCUSDT", "ETHUSDT"): 0.78,
                        ("BTCUSDT", "ADAUSDT"): 0.65,
                        ("ETHUSDT", "ADAUSDT"): 0.72,
                    },
                    "sector_breakdown": {
                        "large_cap": 0.287,
                        "mid_cap": 0.053,
                        "stablecoins": 0.12,
                        "defi": 0.08,
                        "cash": 0.46,
                    },
                }

            # Business logic: analyze risk concentration and diversification
            positions = risk_data.get("positions", [])
            risk_analysis = self._analyze_portfolio_risk_distribution(positions)

            return {
                "position_breakdown": [
                    {
                        **pos,
                        "risk_contribution_pct": (
                            pos["var_contribution"] / sum(p["var_contribution"] for p in positions)
                        )
                        * 100
                        if positions
                        else 0,
                        "risk_level": self._get_risk_level_by_volatility(pos["volatility"]),
                    }
                    for pos in positions
                ],
                "sector_allocation": risk_data.get("sector_breakdown", {}),
                "correlation_matrix": risk_data.get("correlations", {}),
                "risk_concentration": {
                    "herfindahl_index": risk_analysis["herfindahl_index"],
                    "concentration_ratio": risk_analysis["concentration_ratio"],
                    "diversification_score": risk_analysis["diversification_score"],
                },
                "analysis": {
                    "total_positions": len(positions),
                    "largest_position_weight": max((pos["weight"] for pos in positions), default=0),
                    "top_3_concentration": sum(
                        sorted([pos["weight"] for pos in positions], reverse=True)[:3]
                    ),
                    "average_correlation": risk_analysis["avg_correlation"],
                },
                "warnings": risk_analysis["warnings"],
                "recommendations": risk_analysis["recommendations"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except RiskManagementError as e:
            # Propagate risk management errors from risk_management module
            self.logger.error(f"Risk management error getting portfolio breakdown: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error getting portfolio risk breakdown: {e}")
            raise ServiceError(f"Failed to get portfolio risk breakdown: {e}")

    def _analyze_risk_levels(self, risk_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze overall risk levels and generate warnings (business logic)."""
        warnings = []
        recommendations = []
        score = 100  # Start with perfect score

        # Check leverage
        leverage = risk_data.get("leverage", Decimal("1"))
        if leverage > Decimal("3"):
            warnings.append("High leverage detected - consider reducing position sizes")
            score -= 20
        elif leverage > Decimal("2"):
            warnings.append("Moderate leverage - monitor positions closely")
            score -= 10

        # Check concentration
        concentration = risk_data.get("concentration_risk", {})
        largest_pos = concentration.get("largest_position_pct", 0)
        if largest_pos > 0.2:
            warnings.append("High concentration in single position")
            recommendations.append("Consider diversifying largest positions")
            score -= 15

        # Check VaR levels
        portfolio_value = risk_data.get("portfolio_value", Decimal("1"))
        var_1d = risk_data.get("var_1d", Decimal("0"))
        if portfolio_value > 0:
            var_pct = var_1d / portfolio_value
            if var_pct > Decimal("0.05"):
                warnings.append("High daily VaR - portfolio at risk of significant losses")
                score -= 25
            elif var_pct > Decimal("0.03"):
                warnings.append("Elevated daily VaR - consider risk reduction")
                score -= 15

        # Determine risk level
        if score >= 80:
            risk_level = "low"
        elif score >= 60:
            risk_level = "moderate"
        elif score >= 40:
            risk_level = "high"
        else:
            risk_level = "critical"

        return {
            "overall_score": max(0, score),
            "risk_level": risk_level,
            "warnings": warnings,
            "recommendations": recommendations,
        }

    def _calculate_exposure_percentage(
        self, exposure: Decimal, portfolio_value: Decimal
    ) -> Decimal:
        """Calculate exposure as percentage of portfolio (business logic)."""
        if portfolio_value <= 0:
            return Decimal("0")
        return (exposure / portfolio_value) * 100

    def _calculate_drawdown_percentage(
        self, drawdown: Decimal, portfolio_value: Decimal
    ) -> Decimal:
        """Calculate drawdown as percentage of portfolio (business logic)."""
        if portfolio_value <= 0:
            return Decimal("0")
        return (drawdown / portfolio_value) * 100

    def _assess_position_risk_level(self, position_percentage: Decimal, volatility: Decimal) -> str:
        """Assess position risk level based on size and volatility (business logic)."""
        if position_percentage > 15 or volatility > Decimal("0.4"):
            return "high"
        elif position_percentage > 10 or volatility > Decimal("0.3"):
            return "moderate"
        elif position_percentage > 5 or volatility > Decimal("0.2"):
            return "low"
        else:
            return "minimal"

    def _generate_risk_recommendations(self, parameters: dict[str, Any]) -> list[str]:
        """Generate risk parameter recommendations (business logic)."""
        recommendations = []

        max_risk = parameters.get("max_portfolio_risk")
        if isinstance(max_risk, (int, float, Decimal)) and Decimal(str(max_risk)) > Decimal("0.15"):
            recommendations.append("Consider reducing max portfolio risk to below 15%")

        max_leverage = parameters.get("max_leverage")
        if isinstance(max_leverage, (int, float, Decimal)) and Decimal(str(max_leverage)) > 3:
            recommendations.append("High leverage increases risk exponentially - consider reducing")

        return recommendations

    def _generate_position_warnings(
        self, position_percentage: Decimal, volatility: Decimal
    ) -> list[str]:
        """Generate position-specific warnings (business logic)."""
        warnings = []

        if position_percentage > 20:
            warnings.append("Position size exceeds 20% of portfolio - high concentration risk")
        elif position_percentage > 10:
            warnings.append("Large position size - monitor closely")

        if volatility > Decimal("0.5"):
            warnings.append("Extremely high volatility asset - expect large price swings")
        elif volatility > Decimal("0.3"):
            warnings.append("High volatility asset - use appropriate position sizing")

        return warnings

    def _generate_position_recommendations(
        self, symbol: str, position_percentage: Decimal
    ) -> list[str]:
        """Generate position-specific recommendations (business logic)."""
        recommendations = []

        if position_percentage > 15:
            recommendations.append("Consider scaling out of this position to reduce concentration")

        if "USDT" in symbol or "USDC" in symbol:
            recommendations.append(
                "Stablecoin positions are low risk but provide no growth potential"
            )
        else:
            recommendations.append("Consider setting stop-loss orders to limit downside risk")

        return recommendations

    def _analyze_portfolio_risk_distribution(
        self, positions: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Analyze portfolio risk distribution and diversification (business logic)."""
        if not positions:
            return {
                "herfindahl_index": 0,
                "concentration_ratio": 0,
                "diversification_score": 0,
                "avg_correlation": 0,
                "warnings": ["No positions in portfolio"],
                "recommendations": ["Consider adding diversified positions"],
            }

        # Note: In production, these calculations should be done in the risk management service
        # This is simplified calculation for web interface display

        # Calculate basic concentration metrics for web display
        weights = [pos["weight"] for pos in positions]
        herfindahl_index = sum(w**2 for w in weights)

        # Calculate concentration ratio (top 3 positions)
        sorted_weights = sorted(weights, reverse=True)
        concentration_ratio = sum(sorted_weights[:3])

        # Calculate diversification score (inverse of concentration)
        diversification_score = max(0, 1 - herfindahl_index)

        warnings = []
        recommendations = []

        if herfindahl_index > 0.25:
            warnings.append("High portfolio concentration - risk not well diversified")
            recommendations.append("Add more positions to improve diversification")

        if concentration_ratio > 0.6:
            warnings.append("Top 3 positions represent over 60% of portfolio")
            recommendations.append("Consider rebalancing to reduce concentration")

        return {
            "herfindahl_index": herfindahl_index,
            "concentration_ratio": concentration_ratio,
            "diversification_score": diversification_score,
            "avg_correlation": 0.7,  # Would need to get from risk service in production
            "warnings": warnings,
            "recommendations": recommendations,
        }

    def _get_risk_level_by_volatility(self, volatility: float) -> str:
        """Get risk level based on volatility (business logic)."""
        if volatility > 0.5:
            return "extreme"
        elif volatility > 0.4:
            return "high"
        elif volatility > 0.2:
            return "moderate"
        else:
            return "low"

    def health_check(self) -> dict[str, Any]:
        """Perform health check and return status."""
        return {
            "service": "WebRiskService",
            "status": "healthy",
            "risk_service_available": self.risk_service is not None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def generate_mock_risk_alerts(self, filters: dict[str, Any] = None) -> list[dict[str, Any]]:
        """Generate mock risk alerts for web interface (business logic for development)."""
        filters = filters or {}

        # Business logic: generate realistic alerts
        mock_alerts = [
            {
                "alert_id": "alert_001",
                "alert_type": "drawdown_limit",
                "severity": "high",
                "message": "Portfolio drawdown exceeded 12% threshold",
                "triggered_at": datetime.now(timezone.utc) - timedelta(hours=2),
                "current_value": Decimal("13.2"),
                "threshold_value": Decimal("12.0"),
                "affected_positions": ["BTCUSDT", "ETHUSDT"],
                "recommended_action": "Reduce position sizes or hedge portfolio",
                "is_resolved": False,
                "resolved_at": None,
            },
            {
                "alert_id": "alert_002",
                "alert_type": "var_limit",
                "severity": "medium",
                "message": "Daily VaR approaching limit",
                "triggered_at": datetime.now(timezone.utc) - timedelta(hours=4),
                "current_value": Decimal("7200.0"),
                "threshold_value": Decimal("7500.0"),
                "affected_positions": None,
                "recommended_action": "Monitor closely and consider reducing exposure",
                "is_resolved": False,
                "resolved_at": None,
            },
            {
                "alert_id": "alert_003",
                "alert_type": "concentration_risk",
                "severity": "medium",
                "message": "BTC concentration exceeds 35% of portfolio",
                "triggered_at": datetime.now(timezone.utc) - timedelta(hours=6),
                "current_value": Decimal("37.5"),
                "threshold_value": Decimal("35.0"),
                "affected_positions": ["BTCUSDT"],
                "recommended_action": "Diversify holdings or reduce BTC exposure",
                "is_resolved": True,
                "resolved_at": datetime.now(timezone.utc) - timedelta(hours=1),
            },
            {
                "alert_id": "alert_004",
                "alert_type": "leverage_limit",
                "severity": "low",
                "message": "Portfolio leverage increased to 2.8x",
                "triggered_at": datetime.now(timezone.utc) - timedelta(hours=8),
                "current_value": Decimal("2.8"),
                "threshold_value": Decimal("3.0"),
                "affected_positions": None,
                "recommended_action": "Monitor leverage and avoid additional exposure",
                "is_resolved": False,
                "resolved_at": None,
            },
        ]

        # Business logic: apply filters
        filtered_alerts = []
        for alert in mock_alerts:
            # Apply severity filter
            if filters.get("severity") and alert["severity"] != filters["severity"]:
                continue

            # Apply unresolved filter
            if filters.get("unresolved_only") and alert["is_resolved"]:
                continue

            filtered_alerts.append(alert)

        # Business logic: apply limit
        limit = filters.get("limit", 50)
        return filtered_alerts[:limit]

    def generate_mock_position_risks(self, filters: dict[str, Any] = None) -> list[dict[str, Any]]:
        """Generate mock position risk data for web interface (business logic for development)."""
        filters = filters or {}

        # Business logic: generate realistic position risk data
        mock_positions = [
            {
                "position_id": "pos_001",
                "symbol": "BTCUSDT",
                "exchange": "binance",
                "side": "long",
                "quantity": Decimal("2.0"),
                "market_value": Decimal("94000.00"),
                "unrealized_pnl": Decimal("4000.00"),
                "risk_amount": Decimal("1880.00"),  # 2% of market value
                "risk_percentage": Decimal("2.0"),
                "var_contribution": Decimal("2350.00"),
                "beta": Decimal("1.0"),  # BTC beta vs itself
                "correlation": Decimal("1.0"),  # BTC correlation vs itself
                "concentration_risk": Decimal("62.7"),  # High concentration
                "liquidity_risk": "low",
                "time_decay_risk": None,
            },
            {
                "position_id": "pos_002",
                "symbol": "ETHUSDT",
                "exchange": "binance",
                "side": "long",
                "quantity": Decimal("15.0"),
                "market_value": Decimal("46500.00"),
                "unrealized_pnl": Decimal("1500.00"),
                "risk_amount": Decimal("930.00"),
                "risk_percentage": Decimal("2.0"),
                "var_contribution": Decimal("1162.50"),
                "beta": Decimal("1.3"),  # ETH beta vs BTC
                "correlation": Decimal("0.85"),  # ETH correlation vs BTC
                "concentration_risk": Decimal("31.0"),
                "liquidity_risk": "low",
                "time_decay_risk": None,
            },
            {
                "position_id": "pos_003",
                "symbol": "ADAUSDT",
                "exchange": "coinbase",
                "side": "long",
                "quantity": Decimal("25000.0"),
                "market_value": Decimal("9500.00"),
                "unrealized_pnl": Decimal("-500.00"),
                "risk_amount": Decimal("190.00"),
                "risk_percentage": Decimal("2.0"),
                "var_contribution": Decimal("285.00"),
                "beta": Decimal("1.5"),  # ADA beta vs BTC
                "correlation": Decimal("0.72"),  # ADA correlation vs BTC
                "concentration_risk": Decimal("6.3"),
                "liquidity_risk": "medium",
                "time_decay_risk": None,
            },
        ]

        # Business logic: apply filters
        filtered_positions = []
        for position in mock_positions:
            # Apply exchange filter
            if filters.get("exchange") and position["exchange"] != filters["exchange"]:
                continue

            # Apply symbol filter
            if filters.get("symbol") and position["symbol"] != filters["symbol"]:
                continue

            filtered_positions.append(position)

        return filtered_positions

    def generate_mock_correlation_matrix(self, symbols: list[str], period: str) -> dict[str, Any]:
        """Generate mock correlation matrix for web interface (business logic for development)."""
        import random

        # Business logic: generate realistic correlation matrix
        correlation_matrix = {}
        for i, symbol1 in enumerate(symbols):
            correlation_matrix[symbol1] = {}
            for j, symbol2 in enumerate(symbols):
                if i == j:
                    correlation_matrix[symbol1][symbol2] = Decimal("1.0")
                else:
                    # Generate realistic correlation (higher for crypto pairs)
                    base_correlation = (
                        Decimal("0.6") if "USD" in symbol1 and "USD" in symbol2 else Decimal("0.3")
                    )
                    noise = Decimal(str(random.uniform(-0.2, 0.2)))
                    corr = max(Decimal("-1.0"), min(Decimal("1.0"), base_correlation + noise))
                    correlation_matrix[symbol1][symbol2] = corr.quantize(Decimal("0.001"))

        # Business logic: calculate average correlations
        avg_correlations = {}
        for symbol in symbols:
            other_correlations = [
                abs(correlation_matrix[symbol][other]) for other in symbols if other != symbol
            ]
            avg_correlations[symbol] = (
                (sum(other_correlations) / Decimal(str(len(other_correlations)))).quantize(
                    Decimal("0.001")
                )
                if other_correlations
                else Decimal("0.000")
            )

        # Business logic: find highest and lowest correlations
        all_correlations = [
            correlation_matrix[s1][s2] for s1 in symbols for s2 in symbols if s1 != s2
        ]

        return {
            "symbols": symbols,
            "period": period,
            "correlation_matrix": correlation_matrix,
            "average_correlations": avg_correlations,
            "highest_correlation": max(all_correlations) if all_correlations else 0.0,
            "lowest_correlation": min(all_correlations) if all_correlations else 0.0,
            "calculated_at": datetime.now(timezone.utc).isoformat(),
        }

    def generate_mock_stress_test_results(self, test_request: dict[str, Any]) -> dict[str, Any]:
        """Generate mock stress test results for web interface (business logic for development)."""
        import uuid

        # Business logic: generate realistic stress test results
        test_id = f"stress_{uuid.uuid4().hex[:8]}"

        # Mock scenarios and results based on request
        scenarios = test_request.get("scenarios", [])
        worst_case_loss = Decimal("-25000.00")  # 25k loss in worst case
        worst_case_scenario = "BTC -50%, Market Crash"

        # Mock confidence levels (VaR at different confidence levels)
        confidence_levels = {}
        for level in test_request.get("confidence_levels", [Decimal("0.95"), Decimal("0.99")]):
            if level == Decimal("0.95"):
                confidence_levels[float(level)] = Decimal("-7500.00")
            elif level == Decimal("0.99"):
                confidence_levels[float(level)] = Decimal("-15000.00")
            else:
                # Linear interpolation for other levels
                factor = float(level) * 2  # Simple factor for demonstration
                confidence_levels[float(level)] = Decimal(str(-3750 * factor))

        # Mock time horizons (losses over different time periods)
        time_horizons = {}
        for horizon in test_request.get("time_horizons", [1, 5, 10]):
            if horizon == 1:
                time_horizons[horizon] = Decimal("-3750.00")
            elif horizon == 5:
                time_horizons[horizon] = Decimal("-8400.00")
            elif horizon == 10:
                time_horizons[horizon] = Decimal("-11900.00")
            else:
                # Square root scaling for time
                factor = (horizon**0.5) * 1200
                time_horizons[horizon] = Decimal(str(-factor))

        # Business logic: calculate portfolio resilience score
        resilience_score = max(0, 100 - (len(scenarios) * 5))  # Mock calculation

        # Business logic: generate recommendations
        recommendations = [
            "Consider reducing BTC concentration below 50%",
            "Add hedging positions during high volatility periods",
            "Maintain higher cash reserves for extreme scenarios",
            "Implement dynamic position sizing based on volatility",
            "Consider correlation breakdown scenarios in risk models",
        ]

        return {
            "test_id": test_id,
            "test_name": test_request.get("test_name", "Unnamed Test"),
            "scenarios_tested": len(scenarios),
            "worst_case_loss": worst_case_loss,
            "worst_case_scenario": worst_case_scenario,
            "confidence_levels": confidence_levels,
            "time_horizons": time_horizons,
            "portfolio_resilience_score": Decimal(str(resilience_score)),
            "recommendations": recommendations,
            "completed_at": datetime.now(timezone.utc),
        }

    async def validate_risk_parameters_v2(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """
        Validate risk management parameters (second implementation for compatibility).

        Args:
            parameters: Risk parameters to validate

        Returns:
            Validation result with valid flag and errors
        """
        try:
            if self.risk_service and hasattr(self.risk_service, "validate_risk_parameters"):
                # Delegate to actual risk service if available
                return await self.risk_service.validate_risk_parameters(parameters)

            # Fallback validation if risk service not available
            validation_errors = []

            # Validate max_portfolio_risk
            if "max_portfolio_risk" in parameters:
                value = parameters["max_portfolio_risk"]
                if not isinstance(value, (int, float, Decimal)) or value < 0 or value > 1:
                    validation_errors.append("max_portfolio_risk must be between 0 and 1")

            # Validate max_position_size
            if "max_position_size" in parameters:
                value = parameters["max_position_size"]
                if not isinstance(value, (int, float, Decimal)) or value <= 0:
                    validation_errors.append("max_position_size must be greater than 0")

            # Validate max_leverage
            if "max_leverage" in parameters:
                value = parameters["max_leverage"]
                if not isinstance(value, (int, float, Decimal)) or value < 1 or value > 10:
                    validation_errors.append("max_leverage must be between 1 and 10")

            # Validate max_daily_loss
            if "max_daily_loss" in parameters:
                value = parameters["max_daily_loss"]
                if not isinstance(value, (int, float, Decimal)) or value <= 0:
                    validation_errors.append("max_daily_loss must be greater than 0")

            # Validate max_drawdown_limit
            if "max_drawdown_limit" in parameters:
                value = parameters["max_drawdown_limit"]
                if not isinstance(value, (int, float, Decimal)) or value < 0 or value > 1:
                    validation_errors.append("max_drawdown_limit must be between 0 and 1")

            # Validate concentration_limit
            if "concentration_limit" in parameters:
                value = parameters["concentration_limit"]
                if not isinstance(value, (int, float, Decimal)) or value < 0 or value > 1:
                    validation_errors.append("concentration_limit must be between 0 and 1")

            # Validate var_limit
            if "var_limit" in parameters:
                value = parameters["var_limit"]
                if not isinstance(value, (int, float, Decimal)) or value <= 0:
                    validation_errors.append("var_limit must be greater than 0")

            # Validate risk_per_trade
            if "risk_per_trade" in parameters:
                value = parameters["risk_per_trade"]
                if not isinstance(value, (int, float, Decimal)) or value < 0 or value > 0.1:
                    validation_errors.append("risk_per_trade must be between 0 and 0.1")

            is_valid = len(validation_errors) == 0

            return {
                "valid": is_valid,
                "errors": validation_errors,
                "validated_parameters": parameters if is_valid else {},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Error validating risk parameters: {e}")
            return {
                "valid": False,
                "errors": [f"Validation error: {e!s}"],
                "validated_parameters": {},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def get_current_risk_limits(self) -> dict[str, Any]:
        """Get current risk limits with web-specific business logic."""
        try:
            if self.risk_service and hasattr(self.risk_service, "get_current_risk_limits"):
                # Delegate to risk management service
                limits = await self.risk_service.get_current_risk_limits()
                return limits
            else:
                # Business logic: provide default risk limits for development
                return {
                    "max_portfolio_risk": Decimal("0.20"),  # 20% max portfolio risk
                    "max_position_size": Decimal("25000.00"),  # $25k max position
                    "max_leverage": Decimal("3.0"),  # 3x max leverage
                    "max_daily_loss": Decimal("5000.00"),  # $5k max daily loss
                    "max_drawdown_limit": Decimal("0.15"),  # 15% max drawdown
                    "concentration_limit": Decimal("0.30"),  # 30% max in single asset
                    "correlation_limit": Decimal("0.70"),  # 70% max correlation
                    "var_limit": Decimal("7500.00"),  # $7.5k VaR limit
                    "stop_loss_required": True,
                    "position_sizing_method": "kelly_criterion",
                    "risk_per_trade": Decimal("0.02"),  # 2% risk per trade
                }
        except Exception as e:
            self.logger.error(f"Error getting current risk limits: {e}")
            raise ServiceError(f"Failed to get current risk limits: {e}")

    def get_service_info(self) -> dict[str, Any]:
        """Get service information and capabilities."""
        return {
            "service": "WebRiskService",
            "description": "Web risk service handling risk management business logic",
            "capabilities": [
                "risk_dashboard_data",
                "risk_parameter_validation",
                "validate_risk_parameters",
                "position_risk_calculation",
                "portfolio_risk_analysis",
                "mock_risk_alerts",
                "mock_position_risks",
                "mock_correlation_matrix",
                "mock_stress_test_results",
            ],
            "version": "1.0.0",
        }
