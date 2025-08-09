"""
Real-time Data Validation System

This module provides comprehensive validation for incoming market data, including:
- Schema validation for data structure
- Range checks and business rule validation
- Statistical outlier detection
- Data freshness monitoring
- Cross-source consistency checks

Dependencies:
- P-001: Core types, exceptions, logging
- P-002A: Error handling framework
- P-007A: Utility functions and decorators
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from decimal import Decimal
from datetime import datetime, timezone, timedelta
import statistics
import numpy as np
from dataclasses import dataclass
from enum import Enum

# Import from P-001 core components
from src.core.types import (
    MarketData, Signal, Position, Ticker, OrderBook, 
    ValidationLevel, ValidationResult
)
from src.core.exceptions import (
    DataError, DataValidationError, ValidationError
)
from src.core.config import Config
from src.core.logging import get_logger

# Import from P-002A error handling
from src.error_handling.error_handler import ErrorHandler

# Import from P-007A utilities
from src.utils.decorators import time_execution, retry
from src.utils.validators import validate_price, validate_quantity
from src.utils.helpers import calculate_percentage_change

logger = get_logger(__name__)


# ValidationLevel and ValidationResult are now imported from core.types


@dataclass
class ValidationIssue:
    """Data validation issue record"""
    field: str
    value: Any
    expected: Any
    message: str
    level: ValidationLevel
    timestamp: datetime
    source: str
    metadata: Dict[str, Any]


class DataValidator:
    """
    Comprehensive data validation system for market data quality assurance.
    
    This class provides real-time validation for all incoming market data,
    ensuring data quality for ML models and trading strategies.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the data validator with configuration.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.error_handler = ErrorHandler(config)
        
        # Validation thresholds
        self.price_change_threshold = getattr(config, 'price_change_threshold', 0.5)  # 50% max change
        self.volume_change_threshold = getattr(config, 'volume_change_threshold', 10.0)  # 1000% max change
        self.outlier_std_threshold = getattr(config, 'outlier_std_threshold', 3.0)  # 3 standard deviations
        self.max_data_age_seconds = getattr(config, 'max_data_age_seconds', 60)  # 1 minute max age
        
        # Statistical tracking for outlier detection
        self.price_history: Dict[str, List[float]] = {}
        self.volume_history: Dict[str, List[float]] = {}
        self.max_history_size = getattr(config, 'max_history_size', 1000)
        
        # Cross-source consistency tracking
        self.source_data: Dict[str, Dict[str, Any]] = {}
        self.consistency_threshold = getattr(config, 'consistency_threshold', 0.01)  # 1% max difference
        
        logger.info("DataValidator initialized", config=config)
    
    @time_execution
    async def validate_market_data(self, data: MarketData) -> Tuple[bool, List[ValidationIssue]]:
        """
        Validate market data for quality and consistency.
        
        Args:
            data: Market data to validate
            
        Returns:
            Tuple of (is_valid, validation_issues)
        """
        issues = []
        
        try:
            # Handle None data
            if data is None:
                issue = ValidationIssue(
                    field="data",
                    value=None,
                    expected="valid_market_data",
                    message="Market data is None",
                    level=ValidationLevel.CRITICAL,
                    timestamp=datetime.now(timezone.utc),
                    source="DataValidator",
                    metadata={"error_type": "NoneData"}
                )
                return False, [issue]
            
            # Schema validation
            schema_issues = await self._validate_schema(data)
            issues.extend(schema_issues)
            
            # Range checks
            range_issues = await self._validate_ranges(data)
            issues.extend(range_issues)
            
            # Business rule validation
            business_issues = await self._validate_business_rules(data)
            issues.extend(business_issues)
            
            # Statistical outlier detection
            outlier_issues = await self._detect_outliers(data)
            issues.extend(outlier_issues)
            
            # Data freshness check
            freshness_issues = await self._validate_freshness(data)
            issues.extend(freshness_issues)
            
            # Update statistical tracking
            await self._update_statistics(data)
            
            # Determine overall validation result
            critical_issues = [i for i in issues if i.level == ValidationLevel.CRITICAL]
            high_issues = [i for i in issues if i.level == ValidationLevel.HIGH]
            
            is_valid = len(critical_issues) == 0 and len(high_issues) == 0
            
            if issues:
                logger.warning(
                    "Market data validation issues detected",
                    symbol=data.symbol if data and data.symbol else "unknown",
                    issue_count=len(issues),
                    critical_count=len(critical_issues),
                    high_count=len(high_issues)
                )
            else:
                logger.debug("Market data validation passed", symbol=data.symbol if data else "unknown")
            
            return is_valid, issues
            
        except Exception as e:
            logger.error("Market data validation failed", symbol=data.symbol if data and data.symbol else "unknown", error=str(e))
            issue = ValidationIssue(
                field="validation_system",
                value="exception",
                expected="successful_validation",
                message=f"Validation system error: {str(e)}",
                level=ValidationLevel.CRITICAL,
                timestamp=datetime.now(timezone.utc),
                source="DataValidator",
                metadata={"error_type": type(e).__name__}
            )
            return False, [issue]
    
    @time_execution
    async def validate_signal(self, signal: Signal) -> Tuple[bool, List[ValidationIssue]]:
        """
        Validate trading signal for quality and consistency.
        
        Args:
            signal: Trading signal to validate
            
        Returns:
            Tuple of (is_valid, validation_issues)
        """
        issues = []
        
        try:
            # Basic signal validation
            if not signal.direction:
                issues.append(ValidationIssue(
                    field="direction",
                    value=signal.direction,
                    expected="valid_direction",
                    message="Signal direction is required",
                    level=ValidationLevel.CRITICAL,
                    timestamp=datetime.now(timezone.utc),
                    source="DataValidator",
                    metadata={}
                ))
            
            # Confidence validation
            if not (0.0 <= signal.confidence <= 1.0):
                issues.append(ValidationIssue(
                    field="confidence",
                    value=signal.confidence,
                    expected="0.0_to_1.0",
                    message="Signal confidence must be between 0 and 1",
                    level=ValidationLevel.HIGH,
                    timestamp=datetime.now(timezone.utc),
                    source="DataValidator",
                    metadata={}
                ))
            
            # Timestamp validation
            if signal.timestamp > datetime.now(timezone.utc) + timedelta(seconds=60):
                issues.append(ValidationIssue(
                    field="timestamp",
                    value=signal.timestamp,
                    expected="current_or_past_time",
                    message="Signal timestamp is in the future",
                    level=ValidationLevel.MEDIUM,
                    timestamp=datetime.now(timezone.utc),
                    source="DataValidator",
                    metadata={}
                ))
            
            # Symbol validation
            if not signal.symbol or len(signal.symbol) < 3:
                issues.append(ValidationIssue(
                    field="symbol",
                    value=signal.symbol,
                    expected="valid_symbol",
                    message="Signal symbol is invalid",
                    level=ValidationLevel.CRITICAL,
                    timestamp=datetime.now(timezone.utc),
                    source="DataValidator",
                    metadata={}
                ))
            
            is_valid = len([i for i in issues if i.level in [ValidationLevel.CRITICAL, ValidationLevel.HIGH]]) == 0
            
            if issues:
                logger.warning(
                    "Signal validation issues detected",
                    symbol=signal.symbol,
                    strategy=signal.strategy_name,
                    issue_count=len(issues)
                )
            else:
                logger.debug("Signal validation passed", symbol=signal.symbol, strategy=signal.strategy_name)
            
            return is_valid, issues
            
        except Exception as e:
            logger.error("Signal validation failed", symbol=signal.symbol, error=str(e))
            issue = ValidationIssue(
                field="validation_system",
                value="exception",
                expected="successful_validation",
                message=f"Signal validation system error: {str(e)}",
                level=ValidationLevel.CRITICAL,
                timestamp=datetime.now(timezone.utc),
                source="DataValidator",
                metadata={"error_type": type(e).__name__}
            )
            return False, [issue]
    
    @time_execution
    async def validate_cross_source_consistency(
        self, 
        primary_data: MarketData, 
        secondary_data: MarketData
    ) -> Tuple[bool, List[ValidationIssue]]:
        """
        Validate consistency between data from different sources.
        
        Args:
            primary_data: Primary market data source
            secondary_data: Secondary market data source
            
        Returns:
            Tuple of (is_consistent, validation_issues)
        """
        issues = []
        
        try:
            if primary_data.symbol != secondary_data.symbol:
                issues.append(ValidationIssue(
                    field="symbol_mismatch",
                    value=f"{primary_data.symbol} vs {secondary_data.symbol}",
                    expected="matching_symbols",
                    message="Symbol mismatch between data sources",
                    level=ValidationLevel.CRITICAL,
                    timestamp=datetime.now(timezone.utc),
                    source="DataValidator",
                    metadata={}
                ))
                return False, issues
            
            # Price consistency check
            if primary_data.price and secondary_data.price:
                price_diff = abs(primary_data.price - secondary_data.price) / primary_data.price
                
                if price_diff > self.consistency_threshold:
                    issues.append(ValidationIssue(
                        field="price_consistency",
                        value=f"diff={price_diff:.4f}",
                        expected=f"<{self.consistency_threshold}",
                        message=f"Price inconsistency between sources: {price_diff:.4f}",
                        level=ValidationLevel.HIGH,
                        timestamp=datetime.now(timezone.utc),
                        source="DataValidator",
                        metadata={
                            "primary_price": float(primary_data.price),
                            "secondary_price": float(secondary_data.price),
                            "difference": float(price_diff)
                        }
                    ))
            
            # Volume consistency check
            if primary_data.volume and secondary_data.volume:
                volume_diff = abs(primary_data.volume - secondary_data.volume) / primary_data.volume
                
                if volume_diff > self.consistency_threshold * 2:  # More lenient for volume
                    issues.append(ValidationIssue(
                        field="volume_consistency",
                        value=f"diff={volume_diff:.4f}",
                        expected=f"<{self.consistency_threshold * 2}",
                        message=f"Volume inconsistency between sources: {volume_diff:.4f}",
                        level=ValidationLevel.MEDIUM,
                        timestamp=datetime.now(timezone.utc),
                        source="DataValidator",
                        metadata={
                            "primary_volume": float(primary_data.volume),
                            "secondary_volume": float(secondary_data.volume),
                            "difference": float(volume_diff)
                        }
                    ))
            
            is_consistent = len([i for i in issues if i.level in [ValidationLevel.CRITICAL, ValidationLevel.HIGH]]) == 0
            
            if issues:
                logger.warning(
                    "Cross-source consistency issues detected",
                    symbol=primary_data.symbol,
                    issue_count=len(issues)
                )
            else:
                logger.debug("Cross-source consistency validated", symbol=primary_data.symbol)
            
            return is_consistent, issues
            
        except Exception as e:
            logger.error("Cross-source validation failed", error=str(e))
            issue = ValidationIssue(
                field="validation_system",
                value="exception",
                expected="successful_validation",
                message=f"Cross-source validation system error: {str(e)}",
                level=ValidationLevel.CRITICAL,
                timestamp=datetime.now(timezone.utc),
                source="DataValidator",
                metadata={"error_type": type(e).__name__}
            )
            return False, [issue]
    
    async def _validate_schema(self, data: MarketData) -> List[ValidationIssue]:
        """Validate data schema and structure"""
        issues = []
        
        # Required fields validation
        required_fields = ['symbol', 'price', 'volume', 'timestamp']
        for field in required_fields:
            if not hasattr(data, field) or getattr(data, field) is None:
                issues.append(ValidationIssue(
                    field=field,
                    value=getattr(data, field, None),
                    expected="non_null_value",
                    message=f"Required field {field} is missing or null",
                    level=ValidationLevel.CRITICAL,
                    timestamp=datetime.now(timezone.utc),
                    source="DataValidator",
                    metadata={}
                ))
        
        # Data type validation
        if not isinstance(data.symbol, str):
            issues.append(ValidationIssue(
                field="symbol",
                value=type(data.symbol),
                expected="str",
                message="Symbol must be a string",
                level=ValidationLevel.CRITICAL,
                timestamp=datetime.now(timezone.utc),
                source="DataValidator",
                metadata={}
            ))
        
        if not isinstance(data.price, Decimal):
            issues.append(ValidationIssue(
                field="price",
                value=type(data.price),
                expected="Decimal",
                message="Price must be a Decimal",
                level=ValidationLevel.CRITICAL,
                timestamp=datetime.now(timezone.utc),
                source="DataValidator",
                metadata={}
            ))
        
        if not isinstance(data.timestamp, datetime):
            issues.append(ValidationIssue(
                field="timestamp",
                value=type(data.timestamp),
                expected="datetime",
                message="Timestamp must be a datetime",
                level=ValidationLevel.CRITICAL,
                timestamp=datetime.now(timezone.utc),
                source="DataValidator",
                metadata={}
            ))
        
        return issues
    
    async def _validate_ranges(self, data: MarketData) -> List[ValidationIssue]:
        """Validate data ranges and bounds"""
        issues = []
        
        # Price range validation
        if data.price <= 0:
            issues.append(ValidationIssue(
                field="price",
                value=float(data.price),
                expected="positive_value",
                message="Price must be positive",
                level=ValidationLevel.CRITICAL,
                timestamp=datetime.now(timezone.utc),
                source="DataValidator",
                metadata={}
            ))
        
        # Volume range validation
        if data.volume < 0:
            issues.append(ValidationIssue(
                field="volume",
                value=float(data.volume),
                expected="non_negative_value",
                message="Volume must be non-negative",
                level=ValidationLevel.HIGH,
                timestamp=datetime.now(timezone.utc),
                source="DataValidator",
                metadata={}
            ))
        
        # Bid/Ask validation
        if data.bid and data.ask:
            if data.bid >= data.ask:
                issues.append(ValidationIssue(
                    field="bid_ask_spread",
                    value=f"bid={float(data.bid)}, ask={float(data.ask)}",
                    expected="bid < ask",
                    message="Bid must be less than ask",
                    level=ValidationLevel.HIGH,
                    timestamp=datetime.now(timezone.utc),
                    source="DataValidator",
                    metadata={}
                ))
        
        # OHLC validation
        if all([data.open_price, data.high_price, data.low_price]):
            if data.low_price > data.high_price:
                issues.append(ValidationIssue(
                    field="ohlc",
                    value=f"low={float(data.low_price)}, high={float(data.high_price)}",
                    expected="low <= high",
                    message="Low price must be less than or equal to high price",
                    level=ValidationLevel.HIGH,
                    timestamp=datetime.now(timezone.utc),
                    source="DataValidator",
                    metadata={}
                ))
            
            if data.open_price < data.low_price or data.open_price > data.high_price:
                issues.append(ValidationIssue(
                    field="open_price",
                    value=float(data.open_price),
                    expected=f"between {float(data.low_price)} and {float(data.high_price)}",
                    message="Open price must be between low and high",
                    level=ValidationLevel.MEDIUM,
                    timestamp=datetime.now(timezone.utc),
                    source="DataValidator",
                    metadata={}
                ))
        
        return issues
    
    async def _validate_business_rules(self, data: MarketData) -> List[ValidationIssue]:
        """Validate business rules and trading logic"""
        issues = []
        
        # Symbol format validation
        if data.symbol and not self._is_valid_symbol_format(data.symbol):
            issues.append(ValidationIssue(
                field="symbol_format",
                value=data.symbol,
                expected="valid_trading_symbol",
                message="Symbol format is invalid",
                level=ValidationLevel.MEDIUM,
                timestamp=datetime.now(timezone.utc),
                source="DataValidator",
                metadata={}
            ))
        
        # Price precision validation (assuming 8 decimal places for crypto)
        if data.price:
            price_str = str(data.price)
            if '.' in price_str:
                decimal_places = len(price_str.split('.')[1])
                if decimal_places > 8:
                    issues.append(ValidationIssue(
                        field="price_precision",
                        value=decimal_places,
                        expected="<=8",
                        message="Price precision exceeds 8 decimal places",
                        level=ValidationLevel.LOW,
                        timestamp=datetime.now(timezone.utc),
                        source="DataValidator",
                        metadata={}
                    ))
        
        return issues
    
    async def _detect_outliers(self, data: MarketData) -> List[ValidationIssue]:
        """Detect statistical outliers in price and volume data"""
        issues = []
        
        if not data.symbol or not data.price:
            return issues
        
        # Initialize history if needed
        if data.symbol not in self.price_history:
            self.price_history[data.symbol] = []
        
        price_history = self.price_history[data.symbol]
        current_price = float(data.price)
        
        # Add current price to history
        price_history.append(current_price)
        
        # Maintain history size
        if len(price_history) > self.max_history_size:
            price_history.pop(0)
        
        # Outlier detection (need at least 10 data points)
        if len(price_history) >= 10:
            mean_price = statistics.mean(price_history[:-1])  # Exclude current price
            std_price = statistics.stdev(price_history[:-1]) if len(price_history) > 1 else 0
            
            if std_price > 0:
                z_score = abs(current_price - mean_price) / std_price
                
                if z_score > self.outlier_std_threshold:
                    issues.append(ValidationIssue(
                        field="price_outlier",
                        value=f"z_score={z_score:.2f}",
                        expected=f"<{self.outlier_std_threshold}",
                        message=f"Price outlier detected: z-score {z_score:.2f}",
                        level=ValidationLevel.HIGH,
                        timestamp=datetime.now(timezone.utc),
                        source="DataValidator",
                        metadata={
                            "current_price": current_price,
                            "mean_price": mean_price,
                            "std_price": std_price,
                            "z_score": z_score
                        }
                    ))
        
        # Volume outlier detection
        if data.volume and data.symbol not in self.volume_history:
            self.volume_history[data.symbol] = []
        
        if data.volume and data.symbol in self.volume_history:
            volume_history = self.volume_history[data.symbol]
            current_volume = float(data.volume)
            
            volume_history.append(current_volume)
            
            if len(volume_history) > self.max_history_size:
                volume_history.pop(0)
            
            if len(volume_history) >= 10:
                mean_volume = statistics.mean(volume_history[:-1])
                std_volume = statistics.stdev(volume_history[:-1]) if len(volume_history) > 1 else 0
                
                if std_volume > 0:
                    z_score = abs(current_volume - mean_volume) / std_volume
                    
                    if z_score > self.outlier_std_threshold:
                        issues.append(ValidationIssue(
                            field="volume_outlier",
                            value=f"z_score={z_score:.2f}",
                            expected=f"<{self.outlier_std_threshold}",
                            message=f"Volume outlier detected: z-score {z_score:.2f}",
                            level=ValidationLevel.MEDIUM,
                            timestamp=datetime.now(timezone.utc),
                            source="DataValidator",
                            metadata={
                                "current_volume": current_volume,
                                "mean_volume": mean_volume,
                                "std_volume": std_volume,
                                "z_score": z_score
                            }
                        ))
        
        return issues
    
    async def _validate_freshness(self, data: MarketData) -> List[ValidationIssue]:
        """Validate data freshness and timeliness"""
        issues = []
        
        if not data.timestamp:
            return issues
        
        current_time = datetime.now(timezone.utc)
        data_age = (current_time - data.timestamp).total_seconds()
        
        if data_age > self.max_data_age_seconds:
            issues.append(ValidationIssue(
                field="data_freshness",
                value=f"{data_age:.1f}s",
                expected=f"<{self.max_data_age_seconds}s",
                message=f"Data is too old: {data_age:.1f} seconds",
                level=ValidationLevel.HIGH,
                timestamp=datetime.now(timezone.utc),
                source="DataValidator",
                metadata={
                    "data_age_seconds": data_age,
                    "max_age_seconds": self.max_data_age_seconds
                }
            ))
        
        # Future timestamp check
        if data.timestamp > current_time + timedelta(seconds=60):
            issues.append(ValidationIssue(
                field="future_timestamp",
                value=data.timestamp,
                expected="current_or_past_time",
                message="Data timestamp is in the future",
                level=ValidationLevel.CRITICAL,
                timestamp=datetime.now(timezone.utc),
                source="DataValidator",
                metadata={}
            ))
        
        return issues
    
    async def _update_statistics(self, data: MarketData) -> None:
        """Update statistical tracking for outlier detection"""
        if not data.symbol or not data.price:
            return
        
        # Update price statistics
        if data.symbol not in self.price_history:
            self.price_history[data.symbol] = []
        
        price_history = self.price_history[data.symbol]
        price_history.append(float(data.price))
        
        if len(price_history) > self.max_history_size:
            price_history.pop(0)
        
        # Update volume statistics
        if data.volume:
            if data.symbol not in self.volume_history:
                self.volume_history[data.symbol] = []
            
            volume_history = self.volume_history[data.symbol]
            volume_history.append(float(data.volume))
            
            if len(volume_history) > self.max_history_size:
                volume_history.pop(0)
    
    def _is_valid_symbol_format(self, symbol: str) -> bool:
        """Validate trading symbol format"""
        if not symbol or len(symbol) < 3:
            return False
        
        # Basic format check (e.g., BTCUSDT, ETH-BTC)
        valid_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-')
        symbol_upper = symbol.upper()
        
        # Check if all characters are valid
        if not all(c in valid_chars for c in symbol_upper):
            return False
        
        # Check for minimum length and common patterns
        if len(symbol_upper) < 6:  # Most trading pairs are at least 6 chars
            return False
        
        # Check for common quote currencies at the end
        quote_currencies = ['USDT', 'USDC', 'USD', 'BTC', 'ETH', 'BNB', 'ADA', 'DOT']
        has_valid_quote = any(symbol_upper.endswith(quote) for quote in quote_currencies)
        
        return has_valid_quote
    
    @time_execution
    async def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation statistics and summary"""
        return {
            "price_history_size": {symbol: len(history) for symbol, history in self.price_history.items()},
            "volume_history_size": {symbol: len(history) for symbol, history in self.volume_history.items()},
            "validation_config": {
                "price_change_threshold": self.price_change_threshold,
                "volume_change_threshold": self.volume_change_threshold,
                "outlier_std_threshold": self.outlier_std_threshold,
                "max_data_age_seconds": self.max_data_age_seconds,
                "consistency_threshold": self.consistency_threshold
            }
        }
