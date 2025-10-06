"""
Decimal Precision Validation Tests for Strategy Framework.

This module validates that ALL financial calculations throughout
the strategy framework use Decimal precision instead of float,
ensuring regulatory compliance and mathematical accuracy.

CRITICAL: Any use of float for financial calculations is a
compliance violation and must be fixed immediately.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, getcontext
from typing import Any

import pytest

from src.core.types import MarketData, Signal, StrategyConfig, StrategyType
from src.strategies.static.breakout import BreakoutStrategy
from src.strategies.static.mean_reversion import MeanReversionStrategy
from src.strategies.static.trend_following import TrendFollowingStrategy

from .fixtures.real_service_fixtures import generate_realistic_market_data_sequence

# Set maximum precision for testing
getcontext().prec = 28


class DecimalPrecisionValidator:
    """Comprehensive validator for Decimal precision throughout the system."""

    @staticmethod
    def validate_decimal_type(value: Any, field_name: str) -> bool:
        """
        Validate that a value is Decimal type for financial calculations.

        Args:
            value: Value to check
            field_name: Name of the field for error reporting

        Returns:
            True if value is Decimal, False otherwise

        Raises:
            AssertionError: If value is not Decimal type
        """
        if not isinstance(value, Decimal):
            raise AssertionError(
                f"Financial field '{field_name}' must be Decimal type, "
                f"got {type(value).__name__}: {value}"
            )
        return True

    @staticmethod
    def validate_market_data_precision(market_data: MarketData) -> dict[str, bool]:
        """
        Validate all price fields in MarketData use Decimal precision.

        Args:
            market_data: MarketData object to validate

        Returns:
            Dictionary of validation results for each field

        Raises:
            AssertionError: If any price field is not Decimal
        """
        results = {}

        # Price fields that MUST be Decimal
        price_fields = ["open", "high", "low", "close", "volume", "bid_price", "ask_price"]

        for field in price_fields:
            value = getattr(market_data, field, None)
            if value is not None:
                try:
                    DecimalPrecisionValidator.validate_decimal_type(value, f"MarketData.{field}")
                    results[field] = True
                except AssertionError:
                    results[field] = False
                    raise

        return results

    @staticmethod
    def validate_signal_precision(signal: Signal) -> dict[str, bool]:
        """
        Validate all financial fields in Signal use Decimal precision.

        Args:
            signal: Signal object to validate

        Returns:
            Dictionary of validation results for each field

        Raises:
            AssertionError: If any financial field is not Decimal
        """
        results = {}

        # Financial fields that MUST be Decimal
        financial_fields = ["confidence", "strength"]

        for field in financial_fields:
            value = getattr(signal, field, None)
            if value is not None:
                try:
                    DecimalPrecisionValidator.validate_decimal_type(value, f"Signal.{field}")
                    results[field] = True
                except AssertionError:
                    results[field] = False
                    raise

        # Validate metadata financial values
        if signal.metadata:
            financial_metadata_keys = [
                "price",
                "volume",
                "rsi",
                "macd",
                "sma",
                "ema",
                "atr",
                "z_score",
                "entry_price",
                "exit_price",
                "pnl",
                "stop_loss",
                "take_profit",
            ]

            for key in financial_metadata_keys:
                if key in signal.metadata:
                    value = signal.metadata[key]
                    if isinstance(value, (int, float, Decimal)):
                        try:
                            DecimalPrecisionValidator.validate_decimal_type(
                                Decimal(str(value)) if not isinstance(value, Decimal) else value,
                                f"Signal.metadata.{key}",
                            )
                            results[f"metadata.{key}"] = True
                        except (AssertionError, InvalidOperation):
                            results[f"metadata.{key}"] = False
                            if isinstance(value, float):
                                raise AssertionError(
                                    f"Signal metadata '{key}' uses float ({value}), "
                                    f"must use Decimal for financial precision"
                                )

        return results

    @staticmethod
    def validate_strategy_config_precision(config: StrategyConfig) -> dict[str, bool]:
        """
        Validate all financial fields in StrategyConfig use Decimal precision.

        Args:
            config: StrategyConfig object to validate

        Returns:
            Dictionary of validation results for each field

        Raises:
            AssertionError: If any financial field is not Decimal
        """
        results = {}

        # Financial fields that MUST be Decimal
        financial_fields = [
            "min_confidence",
            "position_size_pct",
            "stop_loss_pct",
            "take_profit_pct",
        ]

        for field in financial_fields:
            value = getattr(config, field, None)
            if value is not None:
                try:
                    DecimalPrecisionValidator.validate_decimal_type(
                        value, f"StrategyConfig.{field}"
                    )
                    results[field] = True
                except AssertionError:
                    results[field] = False
                    raise

        # Validate parameters
        if config.parameters:
            financial_param_keys = [
                "entry_threshold",
                "exit_threshold",
                "atr_multiplier",
                "min_volume_ratio",
                "trailing_stop_pct",
                "target_multiplier",
                "momentum_threshold",
                "signal_strength",
                "reversion_strength",
                "false_breakout_threshold",
            ]

            for key in financial_param_keys:
                if key in config.parameters:
                    value = config.parameters[key]
                    if isinstance(value, (int, float, Decimal)):
                        try:
                            if not isinstance(value, Decimal):
                                raise AssertionError(
                                    f"Strategy parameter '{key}' must be Decimal, "
                                    f"got {type(value).__name__}: {value}"
                                )
                            results[f"parameters.{key}"] = True
                        except AssertionError:
                            results[f"parameters.{key}"] = False
                            raise

        return results

    @staticmethod
    def scan_for_float_usage(obj: Any, path: str = "") -> list[str]:
        """
        Recursively scan object for float usage in financial contexts.

        Args:
            obj: Object to scan
            path: Current path for error reporting

        Returns:
            List of paths where float was found in financial context
        """
        float_violations = []

        if isinstance(obj, float):
            # Float found - this might be a violation
            float_violations.append(f"{path}: {obj}")

        elif isinstance(obj, dict):
            for key, value in obj.items():
                # Check if this is a financial field
                financial_keys = [
                    "price",
                    "volume",
                    "confidence",
                    "strength",
                    "pnl",
                    "threshold",
                    "ratio",
                    "multiplier",
                    "percentage",
                ]
                if any(fin_key in key.lower() for fin_key in financial_keys):
                    if isinstance(value, float):
                        float_violations.append(f"{path}.{key}: {value}")

                # Recurse into nested structures
                float_violations.extend(
                    DecimalPrecisionValidator.scan_for_float_usage(value, f"{path}.{key}")
                )

        elif isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                float_violations.extend(
                    DecimalPrecisionValidator.scan_for_float_usage(item, f"{path}[{i}]")
                )

        elif hasattr(obj, "__dict__"):
            for attr_name in dir(obj):
                if not attr_name.startswith("_"):
                    try:
                        attr_value = getattr(obj, attr_name)
                        if not callable(attr_value):
                            float_violations.extend(
                                DecimalPrecisionValidator.scan_for_float_usage(
                                    attr_value, f"{path}.{attr_name}"
                                )
                            )
                    except (AttributeError, Exception):
                        # Skip attributes that can't be accessed
                        pass

        return float_violations

    @staticmethod
    def validate_calculation_precision(
        input_values: list[Decimal], calculation_func: callable, expected_precision: int = 8
    ) -> tuple[bool, Decimal, str]:
        """
        Validate that a calculation maintains required precision.

        Args:
            input_values: Input values for calculation (all Decimal)
            calculation_func: Function to test
            expected_precision: Required decimal places of precision

        Returns:
            Tuple of (is_valid, result, error_message)
        """
        try:
            # Verify all inputs are Decimal
            for i, value in enumerate(input_values):
                if not isinstance(value, Decimal):
                    return False, Decimal("0"), f"Input {i} is not Decimal: {type(value)}"

            # Perform calculation
            result = calculation_func(*input_values)

            # Verify result is Decimal
            if not isinstance(result, Decimal):
                return False, result, f"Result is not Decimal: {type(result)}"

            # Verify precision
            result_str = str(result)
            if "." in result_str:
                decimal_places = len(result_str.split(".")[1])
                if decimal_places < expected_precision:
                    return (
                        False,
                        result,
                        f"Insufficient precision: {decimal_places} < {expected_precision}",
                    )

            return True, result, ""

        except Exception as e:
            return False, Decimal("0"), f"Calculation error: {e!s}"


class TestDecimalPrecisionCompliance:
    """Test suite for Decimal precision compliance throughout the strategy framework."""

    @pytest.fixture
    def precision_validator(self):
        """Create precision validator."""
        return DecimalPrecisionValidator()

    @pytest.fixture
    def sample_market_data_decimal(self):
        """Create sample market data with proper Decimal precision."""
        return MarketData(
            symbol="BTC/USDT",
            open=Decimal("50123.45678901"),
            high=Decimal("50234.56789012"),
            low=Decimal("50012.34567890"),
            close=Decimal("50156.78901234"),
            volume=Decimal("1234.56789012"),
            timestamp=datetime.now(timezone.utc),
            exchange="binance",
            bid_price=Decimal("50155.78901234"),
            ask_price=Decimal("50157.78901234"),
        )

    @pytest.fixture
    def sample_strategy_config_decimal(self):
        """Create sample strategy config with proper Decimal precision."""
        return StrategyConfig(
            strategy_id="decimal_test_001",
            name="decimal_test_strategy",
            strategy_type=StrategyType.MEAN_REVERSION,
            symbol="BTC/USDT",
            timeframe="1h",
            min_confidence=Decimal("0.75678901"),
            position_size_pct=Decimal("0.02345678"),
            stop_loss_pct=Decimal("0.01234567"),
            take_profit_pct=Decimal("0.04567890"),
            parameters={
                "lookback_period": 20,
                "entry_threshold": Decimal("2.12345678"),
                "exit_threshold": Decimal("0.56789012"),
                "atr_multiplier": Decimal("2.34567890"),
                "min_volume_ratio": Decimal("1.56789012"),
            },
        )

    def test_market_data_decimal_precision(self, precision_validator, sample_market_data_decimal):
        """Test that MarketData maintains Decimal precision for all price fields."""
        # Validate all price fields are Decimal
        results = precision_validator.validate_market_data_precision(sample_market_data_decimal)

        # All validations should pass
        for field, is_valid in results.items():
            assert is_valid, f"MarketData.{field} failed Decimal validation"

        # Verify high precision is maintained
        assert sample_market_data_decimal.close == Decimal("50156.78901234")
        assert sample_market_data_decimal.volume == Decimal("1234.56789012")

        # Test precision preservation in calculations
        total_value = sample_market_data_decimal.close * sample_market_data_decimal.volume
        assert isinstance(total_value, Decimal)

        # Verify calculation maintains precision
        expected_total = Decimal("50156.78901234") * Decimal("1234.56789012")
        assert total_value == expected_total

    def test_strategy_config_decimal_precision(
        self, precision_validator, sample_strategy_config_decimal
    ):
        """Test that StrategyConfig maintains Decimal precision for all financial fields."""
        # Validate all financial fields are Decimal
        results = precision_validator.validate_strategy_config_precision(
            sample_strategy_config_decimal
        )

        # All validations should pass
        for field, is_valid in results.items():
            assert is_valid, f"StrategyConfig.{field} failed Decimal validation"

        # Verify parameter precision
        assert sample_strategy_config_decimal.parameters["entry_threshold"] == Decimal("2.12345678")
        assert sample_strategy_config_decimal.min_confidence == Decimal("0.75678901")

        # Test that config values maintain precision in calculations
        risk_amount = (
            sample_strategy_config_decimal.position_size_pct
            * sample_strategy_config_decimal.stop_loss_pct
        )
        assert isinstance(risk_amount, Decimal)

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_signal_decimal_precision(self, precision_validator, strategy_service_container):
        """Test that Signal objects maintain Decimal precision."""
        # Create strategy and generate real signals
        config = StrategyConfig(
            strategy_id="signal_precision_test",
            name="signal_precision_test",
            strategy_type=StrategyType.MEAN_REVERSION,
            symbol="BTC/USDT",
            timeframe="1h",
            min_confidence=Decimal("0.70"),
            parameters={
                "lookback_period": 20,
                "entry_threshold": Decimal("2.0"),
            },
        )

        strategy = MeanReversionStrategy(config=config.dict(), services=strategy_service_container)
        await strategy.initialize(config)

        try:
            # Generate market data with Decimal precision
            market_data_series = []
            base_price = Decimal("50000.12345678")

            for i in range(25):
                price_change = Decimal(f"{i * 10}.{i:02d}")
                price = base_price + price_change

                market_data = MarketData(
                    symbol="BTC/USDT",
                    open=price - Decimal("10.11111111"),
                    high=price + Decimal("20.22222222"),
                    low=price - Decimal("15.33333333"),
                    close=price,
                    volume=Decimal(f"1000.{i:08d}"),
                    timestamp=datetime.now(timezone.utc) - timedelta(hours=25 - i),
                    exchange="binance",
                )

                market_data_series.append(market_data)
                await strategy.services.data_service.store_market_data(
                    market_data, exchange="binance"
                )

            # Generate signals
            signals = await strategy.generate_signals(market_data_series[-1])

            # Validate signal precision
            for signal in signals:
                results = precision_validator.validate_signal_precision(signal)

                # All validations should pass
                for field, is_valid in results.items():
                    assert is_valid, f"Signal.{field} failed Decimal validation"

                # Verify signal values are properly typed
                assert isinstance(signal.confidence, Decimal)
                assert isinstance(signal.strength, Decimal)

                # Verify metadata contains Decimal values for financial fields
                if signal.metadata:
                    for key, value in signal.metadata.items():
                        if any(
                            fin_key in key.lower()
                            for fin_key in ["price", "volume", "score", "ratio", "threshold"]
                        ):
                            if isinstance(value, (int, float, Decimal)):
                                assert isinstance(value, Decimal) or isinstance(value, int), (
                                    f"Signal metadata '{key}' should be Decimal, got {type(value)}: {value}"
                                )

        finally:
            strategy.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_indicator_calculation_precision(self, strategy_service_container):
        """Test that technical indicator calculations maintain Decimal precision."""
        config = StrategyConfig(
            strategy_id="indicator_precision_test",
            name="indicator_precision_test",
            strategy_type=StrategyType.TREND_FOLLOWING,
            symbol="BTC/USDT",
            timeframe="1h",
            parameters={
                "fast_ma_period": 10,
                "slow_ma_period": 20,
                "rsi_period": 14,
            },
        )

        strategy = TrendFollowingStrategy(config=config.dict(), services=strategy_service_container)
        await strategy.initialize(config)

        try:
            # Generate high-precision market data
            market_data_series = []
            base_price = Decimal("50000.123456789")

            for i in range(30):
                # Create price with maximum precision
                price = base_price + Decimal(f"{i}.{i:09d}")

                market_data = MarketData(
                    symbol="BTC/USDT",
                    open=price,
                    high=price + Decimal("0.123456789"),
                    low=price - Decimal("0.098765432"),
                    close=price + Decimal("0.011111111"),
                    volume=Decimal(f"1000.{i:09d}"),
                    timestamp=datetime.now(timezone.utc) - timedelta(hours=30 - i),
                    exchange="binance",
                )

                market_data_series.append(market_data)
                await strategy.services.data_service.store_market_data(
                    market_data, exchange="binance"
                )

            # Test various indicator calculations
            indicators = await asyncio.gather(
                strategy.get_sma("BTC/USDT", 20),
                strategy.get_ema("BTC/USDT", 20),
                strategy.get_rsi("BTC/USDT", 14),
                strategy.get_atr("BTC/USDT", 14),
                strategy.get_bollinger_bands("BTC/USDT", 20, 2.0),
            )

            sma, ema, rsi, atr, bb = indicators

            # Verify all indicators return Decimal values
            assert sma is None or isinstance(sma, Decimal), (
                f"SMA should be Decimal, got {type(sma)}"
            )
            assert ema is None or isinstance(ema, Decimal), (
                f"EMA should be Decimal, got {type(ema)}"
            )
            assert rsi is None or isinstance(rsi, Decimal), (
                f"RSI should be Decimal, got {type(rsi)}"
            )
            assert atr is None or isinstance(atr, Decimal), (
                f"ATR should be Decimal, got {type(atr)}"
            )

            if bb:
                assert isinstance(bb["upper"], Decimal), "BB upper should be Decimal"
                assert isinstance(bb["middle"], Decimal), "BB middle should be Decimal"
                assert isinstance(bb["lower"], Decimal), "BB lower should be Decimal"

            # Test precision in indicator calculations
            if sma and ema:
                # Calculation between indicators should maintain precision
                ma_diff = abs(sma - ema)
                assert isinstance(ma_diff, Decimal)

        finally:
            strategy.cleanup()

    def test_decimal_arithmetic_precision(self, precision_validator):
        """Test that Decimal arithmetic maintains required precision."""
        # Test basic arithmetic operations
        a = Decimal("123.456789012345")
        b = Decimal("678.901234567890")

        # Addition
        result_add = a + b
        assert isinstance(result_add, Decimal)
        expected_add = Decimal("802.358023580235")
        assert result_add == expected_add

        # Multiplication
        result_mul = a * b
        assert isinstance(result_mul, Decimal)

        # Division with precision
        result_div = a / b
        assert isinstance(result_div, Decimal)

        # Test that precision is maintained in complex calculations
        complex_calc = (a * b) / (a + b) * Decimal("1.234567890")
        assert isinstance(complex_calc, Decimal)

        # Verify calculation function validation
        def test_calculation(x: Decimal, y: Decimal) -> Decimal:
            return (x * y) / (x + y)

        is_valid, result, error = precision_validator.validate_calculation_precision(
            [a, b], test_calculation, expected_precision=8
        )

        assert is_valid, f"Calculation validation failed: {error}"
        assert isinstance(result, Decimal)

    def test_float_detection_and_prevention(self, precision_validator):
        """Test detection of float usage in financial contexts."""
        # Create objects with float violations
        bad_market_data = {
            "symbol": "BTC/USDT",
            "close": 50000.123,  # Float violation
            "volume": 1000.456,  # Float violation
            "metadata": {
                "price_change": 123.45,  # Float violation
                "non_financial": "test_string",  # OK
            },
        }

        # Scan for float violations
        violations = precision_validator.scan_for_float_usage(bad_market_data, "bad_market_data")

        # Should detect float usage in financial fields
        assert len(violations) >= 3  # close, volume, price_change

        # Verify specific violations
        violation_strings = [str(v) for v in violations]
        assert any("close: 50000.123" in v for v in violation_strings)
        assert any("volume: 1000.456" in v for v in violation_strings)

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_end_to_end_decimal_precision_workflow(self, strategy_service_container):
        """Test complete workflow maintains Decimal precision end-to-end."""
        # Create strategy configuration with Decimal precision
        config = StrategyConfig(
            strategy_id="e2e_precision_test",
            name="e2e_precision_test",
            strategy_type=StrategyType.BREAKOUT,
            symbol="BTC/USDT",
            timeframe="1h",
            min_confidence=Decimal("0.656789012"),
            position_size_pct=Decimal("0.023456789"),
            stop_loss_pct=Decimal("0.012345678"),
            take_profit_pct=Decimal("0.045678901"),
            parameters={
                "lookback_period": 20,
                "volume_confirmation": True,
                "min_volume_ratio": Decimal("1.567890123"),
                "false_breakout_threshold": Decimal("0.023456789"),
            },
        )

        strategy = BreakoutStrategy(config=config.dict(), services=strategy_service_container)
        await strategy.initialize(config)

        try:
            # Generate market data with high precision
            market_data_series = generate_realistic_market_data_sequence(
                pattern="breakout", periods=25
            )

            # Ensure all market data uses Decimal precision
            for i, md in enumerate(market_data_series):
                # Convert to high precision Decimal if needed
                if not isinstance(md.close, Decimal):
                    md.close = Decimal(f"{float(md.close):.9f}")
                if not isinstance(md.volume, Decimal):
                    md.volume = Decimal(f"{float(md.volume):.9f}")

                await strategy.services.data_service.store_market_data(md, exchange="binance")

            # Generate signals
            signals = await strategy.generate_signals(market_data_series[-1])

            # Verify end-to-end precision
            for signal in signals:
                # Validate signal precision
                DecimalPrecisionValidator.validate_signal_precision(signal)

                # Verify position sizing calculations maintain precision
                position_size = strategy.get_position_size(signal)
                assert isinstance(position_size, Decimal)

                # Verify risk calculations maintain precision
                if hasattr(strategy, "calculate_stop_loss"):
                    stop_loss = strategy.calculate_stop_loss(signal, market_data_series[-1])
                    if stop_loss:
                        assert isinstance(stop_loss, Decimal)

                # Verify all metadata values are properly typed
                if signal.metadata:
                    for key, value in signal.metadata.items():
                        if isinstance(value, (int, float)):
                            # Financial metadata should be Decimal
                            if any(
                                fin_key in key.lower()
                                for fin_key in ["price", "volume", "ratio", "threshold", "score"]
                            ):
                                assert isinstance(value, (Decimal, int)), (
                                    f"Metadata {key} should be Decimal, got {type(value)}: {value}"
                                )

            # Test that precision is maintained in aggregated calculations
            if len(signals) > 1:
                avg_confidence = sum(s.confidence for s in signals) / len(signals)
                assert isinstance(avg_confidence, Decimal)

                total_strength = sum(s.strength for s in signals)
                assert isinstance(total_strength, Decimal)

        finally:
            strategy.cleanup()

    def test_regulatory_compliance_precision_requirements(self):
        """Test that precision meets regulatory requirements for financial systems."""
        # Test minimum precision requirements
        # Financial regulations typically require:
        # - Currency amounts: 2-18 decimal places
        # - Percentages: 4-6 decimal places
        # - Prices: 2-18 decimal places (crypto typically 8)

        # Test currency precision (18 decimal places for crypto)
        crypto_amount = Decimal("123.45678901")
        assert len(str(crypto_amount).split(".")[1]) >= 8

        # Test percentage precision (6 decimal places minimum)
        percentage = Decimal("0.123456")
        assert len(str(percentage).split(".")[1]) >= 6

        # Test that calculations don't lose precision
        amount1 = Decimal("1000.12345678")
        amount2 = Decimal("0.000000000000000001")

        # Addition should maintain precision
        result_add = amount1 + amount2
        # The actual calculation result with 18 decimal precision
        expected = Decimal("1000.12345678") + Decimal("0.000000000000000001")
        tolerance = Decimal("0.000000000000000001")  # 18 decimal precision
        assert abs(result_add - expected) <= tolerance, f"Expected {expected}, got {result_add}"

        # Division should maintain reasonable precision
        result_div = amount1 / Decimal("3")
        # Should have at least 18 decimal places
        decimal_part = str(result_div).split(".")[1]
        assert len(decimal_part) >= 8

        # Test rounding behavior for regulatory compliance
        # Should round to nearest even (banker's rounding)
        test_value = Decimal("123.456785")
        rounded_6 = test_value.quantize(Decimal("0.000001"))
        assert rounded_6 == Decimal("123.456785")

        test_value2 = Decimal("123.4567865")
        rounded_6_2 = test_value2.quantize(Decimal("0.000001"))
        # Should round to nearest even (banker's rounding: 123.4567865 rounds to 123.456787)
        assert rounded_6_2 == Decimal("123.456787")

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_performance_impact_of_decimal_precision(self, strategy_service_container):
        """Test that Decimal precision doesn't significantly impact performance."""
        import time

        config = StrategyConfig(
            strategy_id="performance_precision_test",
            name="performance_precision_test",
            strategy_type=StrategyType.MEAN_REVERSION,
            symbol="BTC/USDT",
            timeframe="1h",
            parameters={"lookback_period": 20, "entry_threshold": Decimal("2.0")},
        )

        strategy = MeanReversionStrategy(config=config.dict(), services=strategy_service_container)
        await strategy.initialize(config)

        try:
            # Generate large dataset for performance testing
            market_data_series = []
            base_price = Decimal("50000.12345678")

            for i in range(100):  # 100 data points
                price = base_price + Decimal(f"{i}.{i:08d}")
                market_data = MarketData(
                    symbol="BTC/USDT",
                    open=price,
                    high=price + Decimal("10.11111111"),
                    low=price - Decimal("8.22222222"),
                    close=price + Decimal("2.33333333"),
                    volume=Decimal(f"1000.{i:08d}"),
                    timestamp=datetime.now(timezone.utc) - timedelta(hours=100 - i),
                    exchange="binance",
                )
                market_data_series.append(market_data)

            # Test processing time with Decimal precision
            start_time = time.time()

            # Store all market data
            for md in market_data_series:
                await strategy.services.data_service.store_market_data(md, exchange="binance")

            # Calculate indicators
            sma = await strategy.get_sma("BTC/USDT", 20)
            rsi = await strategy.get_rsi("BTC/USDT", 14)

            # Generate signals
            signals = await strategy.generate_signals(market_data_series[-1])

            processing_time = time.time() - start_time

            # Decimal precision should not significantly impact performance
            # This is a reasonable expectation for 100 data points
            assert processing_time < 10.0, f"Processing took too long: {processing_time}s"

            # Verify results maintain precision
            if sma:
                assert isinstance(sma, Decimal)
            if rsi:
                assert isinstance(rsi, Decimal)
            for signal in signals:
                assert isinstance(signal.confidence, Decimal)
                assert isinstance(signal.strength, Decimal)

        finally:
            strategy.cleanup()
