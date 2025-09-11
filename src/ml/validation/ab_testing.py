"""
A/B Testing Framework for ML Model Deployment in Trading Systems.

This module provides comprehensive A/B testing capabilities for safely deploying
ML models in production trading environments with proper statistical analysis,
risk controls, and automated decision making.
"""

import hashlib
import uuid
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal, getcontext
from enum import Enum
from typing import Any

import numpy as np
from scipy import stats

from src.base import BaseComponent
from src.core.exceptions import ValidationError
from src.utils.decimal_utils import to_decimal
from src.utils.decorators import UnifiedDecorator

# Initialize decorator instance
dec = UnifiedDecorator()


class ABTestStatus(Enum):
    """A/B Test status enumeration."""

    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    TERMINATED = "terminated"


class ABTestVariant:
    """Represents a variant in an A/B test."""

    def __init__(
        self,
        variant_id: str,
        name: str,
        model_name: str,
        traffic_allocation: float,
        metadata: dict[str, Any] | None = None,
    ):
        self.variant_id = variant_id
        self.name = name
        self.model_name = model_name
        self.traffic_allocation = traffic_allocation
        self.metadata = metadata or {}
        self.created_at = datetime.now(timezone.utc)

        # Performance tracking
        self.predictions_count = 0
        self.performance_metrics: dict[str, Any] = {}
        self.trading_metrics: dict[str, Any] = {}


class ABTest:
    """Represents an A/B test for ML model evaluation."""

    def __init__(
        self,
        test_id: str,
        name: str,
        description: str,
        variants: list[ABTestVariant],
        minimum_sample_size: int = 1000,
        confidence_level: float = 0.95,
        minimum_effect_size: float = 0.05,
        max_duration_days: int = 30,
        traffic_split_method: str = "hash_based",
        primary_metric: str = "sharpe_ratio",
        metadata: dict[str, Any] | None = None,
    ):
        self.test_id = test_id
        self.name = name
        self.description = description
        self.variants = {v.variant_id: v for v in variants}
        self.minimum_sample_size = minimum_sample_size
        self.confidence_level = confidence_level
        self.minimum_effect_size = minimum_effect_size
        self.max_duration_days = max_duration_days
        self.traffic_split_method = traffic_split_method
        self.primary_metric = primary_metric
        self.metadata = metadata or {}

        # Test lifecycle
        self.status = ABTestStatus.DRAFT
        self.created_at = datetime.now(timezone.utc)
        self.started_at: datetime | None = None
        self.ended_at: datetime | None = None

        # Results tracking
        self.results_history: list[dict[str, Any]] = []
        self.statistical_power = 0.8  # 80% power
        self.early_stopping_enabled = True
        self.risk_controls = {
            "max_drawdown_threshold": 0.05,  # 5% max drawdown
            "min_sharpe_threshold": 0.5,  # Minimum Sharpe ratio
            "stop_on_significant_loss": True,
        }


class ABTestFramework(BaseComponent):
    """
    A/B Testing Framework for ML Model Deployment.

    This framework provides comprehensive A/B testing capabilities including:
    - Traffic splitting and allocation
    - Statistical analysis and significance testing
    - Trading-specific metrics evaluation
    - Risk controls and early stopping
    - Automated decision making
    - Performance monitoring and alerting
    """

    def __init__(
        self,
        default_confidence_level: float = 0.95,
        default_minimum_effect_size: float = 0.05,
        max_concurrent_tests: int = 5,
    ):
        """
        Initialize the A/B testing framework.

        Args:
            default_confidence_level: Default confidence level for statistical tests
            default_minimum_effect_size: Default minimum effect size to detect
            max_concurrent_tests: Maximum number of concurrent A/B tests
        """
        super().__init__()

        # Set precision context for financial calculations
        getcontext().prec = 28
        getcontext().rounding = ROUND_HALF_UP

        # A/B test management
        self.active_tests: dict[str, ABTest] = {}
        self.completed_tests: dict[str, ABTest] = {}
        self.test_assignments: dict[str, str] = {}  # user/session -> variant_id

        # Configuration
        self.default_confidence_level = default_confidence_level
        self.default_minimum_effect_size = default_minimum_effect_size
        self.max_concurrent_tests = max_concurrent_tests

        self.logger.info(
            "A/B Testing Framework initialized",
            default_confidence_level=self.default_confidence_level,
            default_minimum_effect_size=self.default_minimum_effect_size,
            max_concurrent_tests=self.max_concurrent_tests,
        )

    @dec.enhance(log=True, monitor=True, log_level="info")
    def create_ab_test(
        self,
        name: str,
        description: str,
        control_model: str | None = None,
        treatment_model: str | None = None,
        variants: list[ABTestVariant] | None = None,
        traffic_split: float = 0.5,
        minimum_sample_size: int = 1000,
        confidence_level: float | None = None,
        minimum_effect_size: float | None = None,
        max_duration_days: int = 30,
        primary_metric: str = "sharpe_ratio",
        risk_controls: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Create a new A/B test.

        Args:
            name: Test name
            description: Test description
            control_model: Control model name (baseline) - used if variants not provided
            treatment_model: Treatment model name (new model) - used if variants not provided
            variants: List of pre-configured variants - takes priority over control/treatment models
            traffic_split: Fraction of traffic for treatment (0.0 to 1.0)
            minimum_sample_size: Minimum samples per variant
            confidence_level: Statistical confidence level
            minimum_effect_size: Minimum detectable effect size
            max_duration_days: Maximum test duration in days
            primary_metric: Primary metric for comparison
            risk_controls: Risk control parameters
            metadata: Additional metadata

        Returns:
            Test ID of created test

        Raises:
            ValidationError: If test creation fails
        """
        try:
            # Check concurrent test limit
            if len(self.active_tests) >= self.max_concurrent_tests:
                raise ValidationError(
                    f"Maximum concurrent tests ({self.max_concurrent_tests}) reached"
                )

            # Handle variants parameter vs individual model parameters
            if variants is not None:
                # Use provided variants (test interface)
                test_variants = variants
            elif control_model is not None and treatment_model is not None:
                # Create variants from model names (original interface)
                test_variants = [
                    ABTestVariant(
                        variant_id="control",
                        name="Control Model",
                        model_name=control_model,
                        traffic_allocation=1.0 - traffic_split,
                        metadata={"description": "Baseline model"},
                    ),
                    ABTestVariant(
                        variant_id="treatment",
                        name="Treatment Model",
                        model_name=treatment_model,
                        traffic_allocation=traffic_split,
                        metadata={"description": "New model"},
                    ),
                ]
            else:
                raise ValidationError(
                    "Either variants or both control_model and treatment_model must be provided"
                )

            # Validate traffic splits
            total_traffic = sum(v.traffic_allocation for v in test_variants)
            if not (0.95 <= total_traffic <= 1.05):  # Allow small floating point errors
                raise ValidationError("Variant traffic allocations must sum to 1.0")

            if confidence_level is None:
                confidence_level = self.default_confidence_level
            if minimum_effect_size is None:
                minimum_effect_size = self.default_minimum_effect_size

            # Create test ID
            test_id = str(uuid.uuid4())

            # Create A/B test
            ab_test = ABTest(
                test_id=test_id,
                name=name,
                description=description,
                variants=test_variants,
                minimum_sample_size=minimum_sample_size,
                confidence_level=confidence_level,
                minimum_effect_size=minimum_effect_size,
                max_duration_days=max_duration_days,
                primary_metric=primary_metric,
                metadata=metadata,
            )

            # Apply custom risk controls
            if risk_controls:
                ab_test.risk_controls.update(risk_controls)

            # Store test
            self.active_tests[test_id] = ab_test

            self.logger.info(
                "A/B test created",
                test_id=test_id,
                name=name,
                control_model=control_model,
                treatment_model=treatment_model,
                traffic_split=traffic_split,
                minimum_sample_size=minimum_sample_size,
            )

            return test_id

        except Exception as e:
            self.logger.error(f"A/B test creation failed: {e}")
            raise ValidationError(f"A/B test creation failed: {e}")

    @dec.enhance(log=True, monitor=True, log_level="info")
    def start_ab_test(self, test_id: str) -> bool:
        """
        Start an A/B test.

        Args:
            test_id: Test ID to start

        Returns:
            True if successfully started

        Raises:
            ValidationError: If test start fails
        """
        try:
            if test_id not in self.active_tests:
                raise ValidationError(f"Test {test_id} not found")

            test = self.active_tests[test_id]

            if test.status != ABTestStatus.DRAFT:
                raise ValidationError(f"Test {test_id} is not in draft status")

            # Validate test configuration
            self._validate_test_configuration(test)

            # Start the test
            test.status = ABTestStatus.RUNNING
            test.started_at = datetime.now(timezone.utc)

            self.logger.info(
                "A/B test started",
                test_id=test_id,
                name=test.name,
                started_at=test.started_at,
            )

            return True

        except Exception as e:
            self.logger.error(f"A/B test start failed: {e}")
            raise ValidationError(f"A/B test start failed: {e}")

    @dec.enhance(log=True, monitor=True, log_level="info")
    def assign_variant(self, test_id: str, user_id: str) -> str:
        """
        Assign a variant to a user for an A/B test.

        Args:
            test_id: Test ID
            user_id: User identifier

        Returns:
            Variant ID assigned

        Raises:
            ValidationError: If assignment fails
        """
        try:
            if test_id not in self.active_tests:
                raise ValidationError(f"Test {test_id} not found")

            test = self.active_tests[test_id]

            if test.status != ABTestStatus.RUNNING:
                raise ValidationError(f"Test {test_id} is not running")

            # Check if user already assigned
            assignment_key = f"{test_id}_{user_id}"
            if assignment_key in self.test_assignments:
                return self.test_assignments[assignment_key]

            # Determine variant based on traffic splitting method
            if test.traffic_split_method == "hash_based":
                variant_id = self._hash_based_assignment(test, user_id)
            elif test.traffic_split_method == "random":
                variant_id = self._random_assignment(test)
            else:
                raise ValidationError(f"Unknown traffic split method: {test.traffic_split_method}")

            # Store assignment
            self.test_assignments[assignment_key] = variant_id

            self.logger.debug(
                "Variant assigned",
                test_id=test_id,
                user_id=user_id,
                variant_id=variant_id,
            )

            return variant_id

        except Exception as e:
            self.logger.error(f"Variant assignment failed: {e}")
            raise ValidationError(f"Variant assignment failed: {e}")

    @dec.enhance(log=True, monitor=True, log_level="info")
    async def record_result(
        self,
        test_id: str,
        variant_id: str,
        user_id: str,
        prediction: float,
        actual_return: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Record a result for an A/B test variant.

        Args:
            test_id: Test ID
            variant_id: Variant ID
            user_id: User identifier
            prediction: Model prediction
            actual_return: Actual return (if available)
            metadata: Additional metadata

        Returns:
            True if result recorded successfully

        Raises:
            ValidationError: If recording fails
        """
        try:
            if test_id not in self.active_tests:
                raise ValidationError(f"Test {test_id} not found")

            test = self.active_tests[test_id]

            if variant_id not in test.variants:
                raise ValidationError(f"Variant {variant_id} not found in test {test_id}")

            variant = test.variants[variant_id]

            # Update variant metrics
            variant.predictions_count += 1

            # Store prediction and return data
            if "predictions" not in variant.performance_metrics:
                variant.performance_metrics["predictions"] = []
                variant.performance_metrics["returns"] = []
                variant.performance_metrics["timestamps"] = []

            variant.performance_metrics["predictions"].append(prediction)
            variant.performance_metrics["returns"].append(actual_return)
            variant.performance_metrics["timestamps"].append(datetime.now(timezone.utc))

            # Calculate trading metrics if we have actual returns
            if actual_return is not None:
                self._update_trading_metrics(variant)

            # Check for early stopping conditions
            if test.early_stopping_enabled:
                await self._check_early_stopping(test)

            # Check risk controls
            self._check_risk_controls(test, variant_id)

            return True

        except Exception as e:
            self.logger.error(f"Result recording failed: {e}")
            raise ValidationError(f"Result recording failed: {e}")

    @dec.enhance(log=True, monitor=True, log_level="info")
    def analyze_ab_test(self, test_id: str) -> dict[str, Any]:
        """
        Analyze A/B test results with statistical testing.

        Args:
            test_id: Test ID to analyze

        Returns:
            Comprehensive analysis results

        Raises:
            ValidationError: If analysis fails
        """
        try:
            if test_id not in self.active_tests and test_id not in self.completed_tests:
                raise ValidationError(f"Test {test_id} not found")

            test = self.active_tests.get(test_id) or self.completed_tests[test_id]

            self.logger.info("Starting A/B test analysis", test_id=test_id)

            # Get variants
            variants = list(test.variants.values())
            if len(variants) != 2:
                raise ValidationError("Analysis currently supports only 2-variant tests")

            control = (
                variants[0]
                if variants[0].metadata.get("variant_type") == "control"
                else variants[1]
            )
            treatment = (
                variants[1]
                if variants[1].metadata.get("variant_type") == "treatment"
                else variants[0]
            )

            # Basic statistics
            analysis_result = {
                "test_id": test_id,
                "test_name": test.name,
                "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                "test_status": test.status.value,
                "test_duration_days": self._calculate_test_duration(test),
                "control_variant": {
                    "variant_id": control.variant_id,
                    "model_name": control.model_name,
                    "sample_size": control.predictions_count,
                    "traffic_allocation": control.traffic_allocation,
                },
                "treatment_variant": {
                    "variant_id": treatment.variant_id,
                    "model_name": treatment.model_name,
                    "sample_size": treatment.predictions_count,
                    "traffic_allocation": treatment.traffic_allocation,
                },
            }

            # Statistical power analysis
            power_analysis = self._calculate_statistical_power(test, control, treatment)
            analysis_result["power_analysis"] = power_analysis

            # Performance comparison
            if (
                control.predictions_count >= test.minimum_sample_size
                and treatment.predictions_count >= test.minimum_sample_size
            ):
                performance_comparison = self._compare_variant_performance(test, control, treatment)
                analysis_result["performance_comparison"] = performance_comparison

                # Trading metrics comparison
                trading_comparison = self._compare_trading_metrics(control, treatment)
                analysis_result["trading_comparison"] = trading_comparison

                # Statistical significance tests
                significance_tests = self._perform_significance_tests(test, control, treatment)
                analysis_result["significance_tests"] = significance_tests

                # Decision recommendation
                recommendation = self._generate_test_recommendation(test, analysis_result)
                analysis_result["recommendation"] = recommendation

            else:
                analysis_result["status"] = "insufficient_data"
                analysis_result["message"] = (
                    f"Need minimum {test.minimum_sample_size} samples per variant. "
                    f"Control: {control.predictions_count}, Treatment: {treatment.predictions_count}"
                )

            # Store analysis result
            test.results_history.append(analysis_result)

            self.logger.info(
                "A/B test analysis completed",
                test_id=test_id,
                control_samples=control.predictions_count,
                treatment_samples=treatment.predictions_count,
                has_sufficient_data=analysis_result.get("status") != "insufficient_data",
            )

            return analysis_result

        except Exception as e:
            self.logger.error(f"A/B test analysis failed: {e}")
            raise ValidationError(f"A/B test analysis failed: {e}")

    def _validate_test_configuration(self, test: ABTest) -> None:
        """Validate test configuration before starting."""
        # Check traffic allocation sums to 1.0
        total_allocation = sum(v.traffic_allocation for v in test.variants.values())
        if abs(total_allocation - 1.0) > 1e-6:
            raise ValidationError(f"Traffic allocations must sum to 1.0, got {total_allocation}")

        # Check minimum sample size is reasonable
        if test.minimum_sample_size < 100:
            raise ValidationError("Minimum sample size should be at least 100")

        # Check confidence level is reasonable
        if not (0.8 <= test.confidence_level <= 0.99):
            raise ValidationError("Confidence level should be between 0.8 and 0.99")

    def _hash_based_assignment(self, test: ABTest, user_id: str) -> str:
        """Assign variant using hash-based deterministic method."""
        # Create deterministic hash
        hash_input = f"{test.test_id}_{user_id}".encode()
        hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)

        # Normalize to [0, 1]
        normalized = (hash_value % 1000000) / 1000000.0

        # Assign based on traffic allocation
        cumulative = 0.0
        for variant in test.variants.values():
            cumulative += variant.traffic_allocation
            if normalized <= cumulative:
                return variant.variant_id

        # Fallback to first variant
        return next(iter(test.variants.keys()))

    def _random_assignment(self, test: ABTest) -> str:
        """Assign variant using random method."""
        rand_value = np.random.random()

        cumulative = 0.0
        for variant in test.variants.values():
            cumulative += variant.traffic_allocation
            if rand_value <= cumulative:
                return variant.variant_id

        # Fallback to first variant
        return next(iter(test.variants.keys()))

    def _update_trading_metrics(self, variant: ABTestVariant) -> None:
        """Update trading-specific metrics for a variant."""
        if "returns" not in variant.performance_metrics:
            return

        returns = [r for r in variant.performance_metrics["returns"] if r is not None]

        if len(returns) < 2:
            return

        returns_array = np.array(returns)

        # Calculate trading metrics with Decimal precision using consistent utility
        returns_decimal = [to_decimal(ret) for ret in returns_array]
        total_return_decimal = sum(returns_decimal)

        variant.trading_metrics = {
            "total_return": float(total_return_decimal),
            "mean_return": float(
                total_return_decimal / len(returns_decimal) if returns_decimal else Decimal("0")
            ),
            "return_volatility": float(np.std(returns_array)),
            "sharpe_ratio": self._calculate_sharpe_ratio(returns_array),
            "max_drawdown": self._calculate_max_drawdown(returns_array),
            "hit_ratio": float(np.mean(returns_array > 0)),
            "profit_factor": self._calculate_profit_factor(returns_array),
            "total_trades": len(returns),
        }

    def _calculate_sharpe_ratio(
        self, returns: np.ndarray, risk_free_rate: Decimal = Decimal("0.02")
    ) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        # Annualize assuming daily returns with Decimal precision
        mean_return_decimal = to_decimal(np.mean(returns)) * Decimal("252")
        std_return_decimal = to_decimal(np.std(returns)) * to_decimal(np.sqrt(252))

        if std_return_decimal == 0:
            return 0.0

        sharpe_decimal = (mean_return_decimal - risk_free_rate) / std_return_decimal
        return float(sharpe_decimal)

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0.0

        # Calculate drawdown with Decimal precision for initial calculations
        cumulative_decimal = [Decimal("1")]
        for ret in returns:
            cumulative_decimal.append(cumulative_decimal[-1] * (Decimal("1") + Decimal(str(ret))))

        # Convert to numpy for max calculations (precision maintained in key calculations)
        cumulative = np.array([float(c) for c in cumulative_decimal[1:]])
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max

        return float(abs(np.min(drawdown)))

    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """Calculate profit factor."""
        if len(returns) == 0:
            return 0.0

        # Calculate with Decimal precision
        gross_profit_decimal = sum(Decimal(str(ret)) for ret in returns if ret > 0)
        gross_loss_decimal = abs(sum(Decimal(str(ret)) for ret in returns if ret < 0))

        if gross_loss_decimal == 0:
            return float("inf")

        profit_factor_decimal = gross_profit_decimal / gross_loss_decimal
        return float(profit_factor_decimal)

    async def _check_early_stopping(self, test: ABTest) -> None:
        """Check if test should be stopped early."""
        try:
            # Run analysis to check for significance
            analysis = self.analyze_ab_test(test.test_id)

            if "significance_tests" in analysis:
                primary_test = analysis["significance_tests"].get(test.primary_metric)

                if primary_test and primary_test.get("is_significant", False):
                    # Check if effect size is meaningful
                    effect_size = primary_test.get("effect_size", 0)

                    if abs(effect_size) >= test.minimum_effect_size:
                        self.logger.info(
                            "Early stopping triggered - significant result detected",
                            test_id=test.test_id,
                            primary_metric=test.primary_metric,
                            effect_size=effect_size,
                        )

                        # Don't automatically stop, but flag for review
                        test.metadata["early_stopping_flagged"] = True
                        test.metadata["early_stopping_reason"] = "significant_result"

        except Exception as e:
            self.logger.error(f"Early stopping check failed: {e}")

    def _check_risk_controls(self, test: ABTest, variant_id: str) -> None:
        """Check risk control thresholds."""
        try:
            variant = test.variants[variant_id]

            if not variant.trading_metrics:
                return

            risk_triggered = False
            risk_reasons = []

            # Max drawdown check
            max_dd = variant.trading_metrics.get("max_drawdown", 0)
            if max_dd > test.risk_controls.get("max_drawdown_threshold", 0.05):
                risk_triggered = True
                risk_reasons.append(f"max_drawdown_exceeded_{max_dd:.3f}")

            # Minimum Sharpe ratio check
            sharpe = variant.trading_metrics.get("sharpe_ratio", 0)
            if sharpe < test.risk_controls.get("min_sharpe_threshold", 0.5):
                risk_triggered = True
                risk_reasons.append(f"sharpe_ratio_below_threshold_{sharpe:.3f}")

            # Significant loss check
            total_return = variant.trading_metrics.get("total_return", 0)
            if (
                test.risk_controls.get("stop_on_significant_loss", True) and total_return < -0.02
            ):  # 2% loss
                risk_triggered = True
                risk_reasons.append(f"significant_loss_{total_return:.3f}")

            if risk_triggered:
                self.logger.warning(
                    "Risk control triggered",
                    test_id=test.test_id,
                    variant_id=variant_id,
                    risk_reasons=risk_reasons,
                )

                # Pause the test
                test.status = ABTestStatus.PAUSED
                test.metadata["risk_control_triggered"] = True
                test.metadata["risk_reasons"] = risk_reasons

        except Exception as e:
            self.logger.error(f"Risk control check failed: {e}")

    def _calculate_test_duration(self, test: ABTest) -> float:
        """Calculate test duration in days."""
        if test.started_at is None:
            return 0.0

        end_time = test.ended_at or datetime.now(timezone.utc)
        return (end_time - test.started_at).total_seconds() / 86400  # Convert to days

    def _calculate_statistical_power(
        self, test: ABTest, control: ABTestVariant, treatment: ABTestVariant
    ) -> dict[str, Any]:
        """Calculate statistical power of the test."""
        try:
            control_size = control.predictions_count
            treatment_size = treatment.predictions_count

            # Estimate effect size from current data if available
            estimated_effect_size = test.minimum_effect_size

            if (
                control.trading_metrics
                and treatment.trading_metrics
                and test.primary_metric in control.trading_metrics
                and test.primary_metric in treatment.trading_metrics
            ):
                control_metric = control.trading_metrics[test.primary_metric]
                treatment_metric = treatment.trading_metrics[test.primary_metric]

                if control_metric != 0:
                    estimated_effect_size = abs(treatment_metric - control_metric) / abs(
                        control_metric
                    )

            # Rough power calculation (simplified)
            alpha = 1 - test.confidence_level

            # For trading metrics, assume higher variance
            assumed_std = 0.15  # 15% standard deviation

            # Calculate required sample size for desired power
            z_alpha = stats.norm.ppf(1 - alpha / 2)
            z_beta = stats.norm.ppf(test.statistical_power)

            # Handle zero effect size case
            if estimated_effect_size == 0:
                # Use minimum effect size as fallback
                estimated_effect_size = test.minimum_effect_size

            required_n = (2 * (z_alpha + z_beta) ** 2 * assumed_std**2) / (estimated_effect_size**2)

            # Handle infinity case
            if np.isinf(required_n) or np.isnan(required_n):
                required_n = test.minimum_sample_size * 2  # Use reasonable fallback
            else:
                required_n = int(np.ceil(required_n))

            return {
                "current_power_estimate": min(
                    1.0, (control_size + treatment_size) / (2 * required_n)
                ),
                "required_sample_size_per_variant": required_n,
                "current_sample_sizes": {
                    "control": control_size,
                    "treatment": treatment_size,
                },
                "estimated_effect_size": estimated_effect_size,
                "is_adequately_powered": control_size >= required_n
                and treatment_size >= required_n,
            }

        except Exception as e:
            self.logger.error(f"Statistical power calculation failed: {e}")
            return {"error": str(e)}

    def _compare_variant_performance(
        self, test: ABTest, control: ABTestVariant, treatment: ABTestVariant
    ) -> dict[str, Any]:
        """Compare performance between variants."""
        try:
            comparison = {
                "control_metrics": (
                    control.trading_metrics.copy() if control.trading_metrics else {}
                ),
                "treatment_metrics": (
                    treatment.trading_metrics.copy() if treatment.trading_metrics else {}
                ),
                "relative_changes": {},
            }

            # Calculate relative changes
            for metric in control.trading_metrics:
                if metric in treatment.trading_metrics:
                    control_value = control.trading_metrics[metric]
                    treatment_value = treatment.trading_metrics[metric]

                    if control_value != 0:
                        relative_change = (treatment_value - control_value) / control_value
                        comparison["relative_changes"][metric] = {
                            "absolute_change": treatment_value - control_value,
                            "relative_change": relative_change,
                            "relative_change_percent": relative_change * 100,
                        }

            return comparison

        except Exception as e:
            self.logger.error(f"Performance comparison failed: {e}")
            return {"error": str(e)}

    def _compare_trading_metrics(
        self, control: ABTestVariant, treatment: ABTestVariant
    ) -> dict[str, Any]:
        """Compare trading-specific metrics between variants."""
        try:
            if not (control.trading_metrics and treatment.trading_metrics):
                return {"error": "Insufficient trading metrics data"}

            trading_comparison = {}

            key_metrics = [
                "sharpe_ratio",
                "max_drawdown",
                "total_return",
                "hit_ratio",
                "profit_factor",
            ]

            for metric in key_metrics:
                if metric in control.trading_metrics and metric in treatment.trading_metrics:
                    control_value = control.trading_metrics[metric]
                    treatment_value = treatment.trading_metrics[metric]

                    # Determine if higher is better
                    higher_is_better = metric not in ["max_drawdown", "return_volatility"]

                    is_better = (treatment_value > control_value and higher_is_better) or (
                        treatment_value < control_value and not higher_is_better
                    )

                    trading_comparison[metric] = {
                        "control": control_value,
                        "treatment": treatment_value,
                        "difference": treatment_value - control_value,
                        "is_treatment_better": is_better,
                        "improvement_percent": (
                            ((treatment_value - control_value) / control_value * 100)
                            if control_value != 0
                            else 0
                        ),
                    }

            return trading_comparison

        except Exception as e:
            self.logger.error(f"Trading metrics comparison failed: {e}")
            return {"error": str(e)}

    def _perform_significance_tests(
        self, test: ABTest, control: ABTestVariant, treatment: ABTestVariant
    ) -> dict[str, Any]:
        """Perform statistical significance tests."""
        try:
            significance_tests = {}

            # Test primary metric
            if (
                control.trading_metrics
                and treatment.trading_metrics
                and test.primary_metric in control.trading_metrics
                and test.primary_metric in treatment.trading_metrics
            ):
                # Get returns data for statistical testing
                control_returns = [
                    r for r in control.performance_metrics.get("returns", []) if r is not None
                ]
                treatment_returns = [
                    r for r in treatment.performance_metrics.get("returns", []) if r is not None
                ]

                if len(control_returns) >= 30 and len(treatment_returns) >= 30:
                    # Perform t-test
                    t_stat, p_value = stats.ttest_ind(treatment_returns, control_returns)

                    # Calculate effect size (Cohen's d)
                    pooled_variance = (
                        (len(control_returns) - 1) * np.var(control_returns, ddof=1)
                        + (len(treatment_returns) - 1) * np.var(treatment_returns, ddof=1)
                    ) / (len(control_returns) + len(treatment_returns) - 2)
                    # Ensure non-negative variance before sqrt
                    pooled_variance = max(0, pooled_variance)
                    pooled_std = np.sqrt(pooled_variance)

                    cohens_d = (
                        (np.mean(treatment_returns) - np.mean(control_returns)) / pooled_std
                        if pooled_std != 0
                        else 0
                    )

                    significance_tests[test.primary_metric] = {
                        "test_type": "two_sample_ttest",
                        "t_statistic": t_stat,
                        "p_value": p_value,
                        "effect_size": cohens_d,
                        "is_significant": p_value < (1 - test.confidence_level),
                        "confidence_level": test.confidence_level,
                        "sample_sizes": {
                            "control": len(control_returns),
                            "treatment": len(treatment_returns),
                        },
                    }

                    # Mann-Whitney U test (non-parametric)
                    mw_stat, mw_p = stats.mannwhitneyu(
                        treatment_returns, control_returns, alternative="two-sided"
                    )

                    significance_tests[f"{test.primary_metric}_nonparametric"] = {
                        "test_type": "mann_whitney_u",
                        "statistic": mw_stat,
                        "p_value": mw_p,
                        "is_significant": mw_p < (1 - test.confidence_level),
                    }

            return significance_tests

        except Exception as e:
            self.logger.error(f"Significance tests failed: {e}")
            return {"error": str(e)}

    def _generate_test_recommendation(
        self, test: ABTest, analysis_result: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate recommendation based on test analysis."""
        try:
            recommendation: dict[str, Any] = {
                "action": "continue",  # continue, stop_launch_treatment, stop_keep_control
                "confidence": "low",  # low, medium, high
                "reasons": [],
                "risk_assessment": "low",  # low, medium, high
                "next_steps": [],
            }

            # Check if test has sufficient data
            if analysis_result.get("status") == "insufficient_data":
                recommendation.update(
                    {
                        "action": "continue",
                        "confidence": "low",
                        "reasons": ["Insufficient data for decision"],
                        "next_steps": [
                            f"Continue test until minimum {test.minimum_sample_size} samples per variant",
                            "Monitor for early stopping conditions",
                        ],
                    }
                )
                return recommendation

            # Get significance test results
            significance_tests = analysis_result.get("significance_tests", {})
            primary_test = significance_tests.get(test.primary_metric, {})

            # Check statistical significance
            is_significant = primary_test.get("is_significant", False)
            effect_size = abs(primary_test.get("effect_size", 0))

            # Get trading comparison
            trading_comparison = analysis_result.get("trading_comparison", {})
            primary_metric_comparison = trading_comparison.get(test.primary_metric, {})

            treatment_better = primary_metric_comparison.get("is_treatment_better", False)

            # Risk assessment
            risk_factors = []

            # Check treatment performance
            analysis_result["treatment_variant"]
            if "trading_comparison" in analysis_result:
                for metric, comparison in trading_comparison.items():
                    if metric == "max_drawdown" and comparison["treatment"] > 0.03:  # 3% drawdown
                        risk_factors.append(
                            f"High drawdown in treatment: {comparison['treatment']:.3f}"
                        )
                    elif metric == "sharpe_ratio" and comparison["treatment"] < 0.5:
                        risk_factors.append(
                            f"Low Sharpe ratio in treatment: {comparison['treatment']:.3f}"
                        )

            if len(risk_factors) > 2:
                recommendation["risk_assessment"] = "high"
            elif len(risk_factors) > 0:
                recommendation["risk_assessment"] = "medium"

            # Decision logic
            if is_significant and effect_size >= test.minimum_effect_size:
                if treatment_better and recommendation["risk_assessment"] != "high":
                    recommendation.update(
                        {
                            "action": "stop_launch_treatment",
                            "confidence": "high",
                            "reasons": [
                                "Treatment shows statistically significant improvement",
                                f"Effect size ({effect_size:.3f}) exceeds minimum threshold ({test.minimum_effect_size})",
                                "Risk assessment is acceptable",
                            ],
                            "next_steps": [
                                "Gradually roll out treatment model to full traffic",
                                "Continue monitoring performance in production",
                                "Plan follow-up A/B tests for further improvements",
                            ],
                        }
                    )
                else:
                    recommendation.update(
                        {
                            "action": "stop_keep_control",
                            "confidence": "high",
                            "reasons": [
                                "Treatment shows statistically significant performance but control performs better",
                                "Risk factors identified in treatment variant",
                            ],
                            "next_steps": [
                                "Continue with control model",
                                "Investigate treatment model issues",
                                "Design new treatment variant addressing identified problems",
                            ],
                        }
                    )
            elif is_significant:
                recommendation.update(
                    {
                        "action": "continue",
                        "confidence": "medium",
                        "reasons": [
                            "Statistically significant but effect size too small",
                            f"Effect size ({effect_size:.3f}) below minimum threshold ({test.minimum_effect_size})",
                        ],
                        "next_steps": [
                            "Continue test to gather more data",
                            "Consider if small improvement justifies change costs",
                        ],
                    }
                )
            else:
                # Check test duration
                test_duration = analysis_result.get("test_duration_days", 0)

                if test_duration >= test.max_duration_days:
                    recommendation.update(
                        {
                            "action": "stop_keep_control",
                            "confidence": "medium",
                            "reasons": [
                                "Maximum test duration reached without significant results",
                                "No clear evidence of treatment superiority",
                            ],
                            "next_steps": [
                                "Stop test and keep control model",
                                "Analyze results for insights",
                                "Design new treatment approach",
                            ],
                        }
                    )
                else:
                    recommendation.update(
                        {
                            "action": "continue",
                            "confidence": "low",
                            "reasons": [
                                "No significant difference detected yet",
                                "Test duration within acceptable limits",
                            ],
                            "next_steps": [
                                "Continue test to reach adequate sample size",
                                "Monitor for early stopping conditions",
                            ],
                        }
                    )

            # Add risk factors to reasons
            if risk_factors:
                recommendation["reasons"].extend([f"Risk factor: {rf}" for rf in risk_factors])

            return recommendation

        except Exception as e:
            self.logger.error(f"Recommendation generation failed: {e}")
            return {
                "action": "continue",
                "confidence": "low",
                "reasons": [f"Analysis error: {e!s}"],
                "risk_assessment": "unknown",
                "next_steps": ["Fix analysis issues before making decision"],
            }

    @dec.enhance(log=True, monitor=True, log_level="info")
    async def stop_ab_test(self, test_id: str, reason: str = "manual_stop") -> bool:
        """
        Stop an A/B test.

        Args:
            test_id: Test ID to stop
            reason: Reason for stopping

        Returns:
            True if successfully stopped

        Raises:
            ValidationError: If test stop fails
        """
        try:
            if test_id not in self.active_tests:
                raise ValidationError(f"Test {test_id} not found in active tests")

            test = self.active_tests[test_id]

            if test.status not in [ABTestStatus.RUNNING, ABTestStatus.PAUSED]:
                raise ValidationError(f"Test {test_id} is not running or paused")

            # Stop the test
            test.status = ABTestStatus.COMPLETED
            test.ended_at = datetime.now(timezone.utc)
            test.metadata["stop_reason"] = reason

            # Move to completed tests
            self.completed_tests[test_id] = test
            del self.active_tests[test_id]

            # Final analysis
            final_analysis = self.analyze_ab_test(test_id)

            self.logger.info(
                "A/B test stopped",
                test_id=test_id,
                name=test.name,
                reason=reason,
                duration_days=self._calculate_test_duration(test),
                final_recommendation=final_analysis.get("recommendation", {}).get(
                    "action", "unknown"
                ),
            )

            return True

        except Exception as e:
            self.logger.error(f"A/B test stop failed: {e}")
            raise ValidationError(f"A/B test stop failed: {e}")

    def get_active_tests(self) -> dict[str, dict[str, Any]]:
        """Get summary of all active tests."""
        summaries = {}

        for test_id, test in self.active_tests.items():
            summaries[test_id] = {
                "name": test.name,
                "description": test.description,
                "status": test.status.value,
                "started_at": test.started_at.isoformat() if test.started_at else None,
                "duration_days": self._calculate_test_duration(test),
                "variants": {
                    v_id: {
                        "name": variant.name,
                        "model_name": variant.model_name,
                        "sample_size": variant.predictions_count,
                        "traffic_allocation": variant.traffic_allocation,
                    }
                    for v_id, variant in test.variants.items()
                },
                "primary_metric": test.primary_metric,
            }

        return summaries

    async def get_test_results(self, test_id: str) -> dict[str, Any]:
        """Get detailed results for a specific test."""
        test = self.active_tests.get(test_id) or self.completed_tests.get(test_id)

        if not test:
            raise ValidationError(f"Test {test_id} not found")

        # Return latest analysis or perform new one
        if test.results_history:
            return test.results_history[-1]
        else:
            return self.analyze_ab_test(test_id)
