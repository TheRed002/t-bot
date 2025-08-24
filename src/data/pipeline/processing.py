"""
Data Processing Pipeline

This module provides data processing and transformation capabilities:
- Data normalization across sources
- Data enrichment and feature extraction
- Real-time data transformation
- Data aggregation and windowing
- Pipeline orchestration

Dependencies:
- P-001: Core types, exceptions, logging
- P-002A: Error handling framework
- P-007A: Utility functions and decorators
"""

import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from src.core.config import Config
from src.core.exceptions import DataError

# Import from P-001 core components
from src.core.types import MarketData, ProcessingStep

# Import from P-002A error handling
from src.error_handling.error_handler import ErrorHandler

# Import from P-007A utilities
from src.utils.decorators import time_execution
from src.utils.helpers import calculate_percentage_change
from src.utils.validators import ValidationFramework

# ProcessingStep is now imported from core.types


@dataclass
class ProcessingConfig:
    """Data processing configuration"""

    steps: list[ProcessingStep]
    window_size: int
    aggregation_interval: int
    normalization_method: str
    enrichment_enabled: bool
    validation_enabled: bool


@dataclass
class ProcessingResult:
    """Data processing result"""

    original_data: Any
    processed_data: Any
    steps_applied: list[str]
    processing_time: float
    metadata: dict[str, Any]
    success: bool
    error_message: str | None = None


class DataProcessor:
    """
    Comprehensive data processing pipeline for multi-source data transformation.

    This class handles data normalization, enrichment, aggregation, and
    transformation for various data types in the trading system.
    """

    def __init__(self, config: Config):
        """
        Initialize data processor.

        Args:
            config: Application configuration
        """
        super().__init__()  # Initialize BaseComponent
        self.config = config

        # Processing configuration
        processing_config = getattr(config, "data_processing", {})
        if hasattr(processing_config, "get"):
            self.processing_config = ProcessingConfig(
                steps=[
                    ProcessingStep.NORMALIZE,
                    ProcessingStep.ENRICH,
                    ProcessingStep.AGGREGATE,
                    ProcessingStep.VALIDATE,
                ],
                window_size=processing_config.get("window_size", 100),
                aggregation_interval=processing_config.get("aggregation_interval", 60),
                normalization_method=processing_config.get("normalization_method", "z_score"),
                enrichment_enabled=processing_config.get("enrichment_enabled", True),
                validation_enabled=processing_config.get("validation_enabled", True),
            )
        else:
            self.processing_config = ProcessingConfig(
                steps=[
                    ProcessingStep.NORMALIZE,
                    ProcessingStep.ENRICH,
                    ProcessingStep.AGGREGATE,
                    ProcessingStep.VALIDATE,
                ],
                window_size=100,
                aggregation_interval=60,
                normalization_method="z_score",
                enrichment_enabled=True,
                validation_enabled=True,
            )

        # Data windows for processing
        self.data_windows: dict[str, list[Any]] = {}
        self.aggregated_data: dict[str, dict[str, Any]] = {}

        # Processing statistics
        self.stats = {
            "total_processed": 0,
            "successful_processed": 0,
            "failed_processed": 0,
            "avg_processing_time": 0.0,
            "last_processing_time": None,
        }

        # Processing functions registry
        self.processors = {
            ProcessingStep.NORMALIZE: self._normalize_data,
            ProcessingStep.ENRICH: self._enrich_data,
            ProcessingStep.AGGREGATE: self._aggregate_data,
            ProcessingStep.TRANSFORM: self._transform_data,
            ProcessingStep.VALIDATE: self._validate_data,
            ProcessingStep.FILTER: self._filter_data,
        }

        self.logger.info("DataProcessor initialized")

    @time_execution
    async def process_market_data(
        self, data: MarketData, steps: list[ProcessingStep] | None = None
    ) -> ProcessingResult:
        """
        Process market data through the configured pipeline.

        Args:
            data: Market data to process
            steps: Optional list of processing steps (uses default if None)

        Returns:
            ProcessingResult: Processing result with transformed data
        """
        start_time = datetime.now()
        steps_applied = []
        processed_data = data

        try:
            processing_steps = steps or self.processing_config.steps

            for step in processing_steps:
                if step in self.processors:
                    processed_data = await self.processors[step](processed_data, "market_data")
                    steps_applied.append(step.value)
                else:
                    self.logger.warning(f"Unknown processing step: {step}")

            processing_time = (datetime.now() - start_time).total_seconds()

            # Update statistics
            self.stats["total_processed"] += 1
            old_avg = self.stats["avg_processing_time"]
            old_count = self.stats["successful_processed"]
            self.stats["successful_processed"] += 1

            # Calculate new average correctly
            if old_count == 0:
                self.stats["avg_processing_time"] = processing_time
            else:
                self.stats["avg_processing_time"] = (
                    old_avg * old_count + processing_time
                ) / self.stats["successful_processed"]
            self.stats["last_processing_time"] = datetime.now(timezone.utc)

            result = ProcessingResult(
                original_data=data,
                processed_data=processed_data,
                steps_applied=steps_applied,
                processing_time=processing_time,
                metadata={
                    "symbol": data.symbol if data else "unknown",
                    "processing_timestamp": datetime.now(timezone.utc).isoformat(),
                },
                success=True,
            )

            self.logger.debug(
                f"Market data processed successfully for {data.symbol if data else 'unknown'}"
            )
            return result

        except Exception as e:
            self.stats["failed_processed"] += 1
            self.logger.error(f"Market data processing failed: {e!s}")

            return ProcessingResult(
                original_data=data,
                processed_data=data,  # Return original on failure
                steps_applied=steps_applied,
                processing_time=(datetime.now() - start_time).total_seconds(),
                metadata={"error": str(e)},
                success=False,
                error_message=str(e),
            )

    @time_execution
    async def process_batch(
        self, data_list: list[Any], data_type: str, steps: list[ProcessingStep] | None = None
    ) -> list[ProcessingResult]:
        """
        Process a batch of data items.

        Args:
            data_list: List of data items to process
            data_type: Type of data being processed
            steps: Optional list of processing steps

        Returns:
            List[ProcessingResult]: Results for each data item
        """
        try:
            results = []

            for data_item in data_list:
                if data_type == "market_data":
                    result = await self.process_market_data(data_item, steps)
                else:
                    # Generic processing for other data types
                    result = await self._process_generic_data(data_item, data_type, steps)

                results.append(result)

            self.logger.info(f"Batch processing completed: {len(results)} items processed")
            return results

        except Exception as e:
            self.logger.error(f"Batch processing failed: {e!s}")
            raise DataError(f"Batch processing failed: {e!s}")

    async def _process_generic_data(
        self, data: Any, data_type: str, steps: list[ProcessingStep] | None = None
    ) -> ProcessingResult:
        """Process generic data through the pipeline."""
        start_time = datetime.now()
        steps_applied = []
        processed_data = data

        try:
            processing_steps = steps or self.processing_config.steps

            for step in processing_steps:
                if step in self.processors:
                    processed_data = await self.processors[step](processed_data, data_type)
                    steps_applied.append(step.value)

            processing_time = (datetime.now() - start_time).total_seconds()

            return ProcessingResult(
                original_data=data,
                processed_data=processed_data,
                steps_applied=steps_applied,
                processing_time=processing_time,
                metadata={
                    "data_type": data_type,
                    "processing_timestamp": datetime.now(timezone.utc).isoformat(),
                },
                success=True,
            )

        except Exception as e:
            return ProcessingResult(
                original_data=data,
                processed_data=data,
                steps_applied=steps_applied,
                processing_time=(datetime.now() - start_time).total_seconds(),
                metadata={"error": str(e), "data_type": data_type},
                success=False,
                error_message=str(e),
            )

    async def _normalize_data(self, data: Any, data_type: str) -> Any:
        """Normalize data based on type and configuration."""
        try:
            if data_type == "market_data" and isinstance(data, MarketData):
                # Normalize market data
                normalized_data = MarketData(
                    symbol=data.symbol.upper() if data.symbol else None,
                    price=self._normalize_price(data.price) if data.price else None,
                    volume=self._normalize_volume(data.volume) if data.volume else None,
                    timestamp=self._normalize_timestamp(data.timestamp) if data.timestamp else None,
                    exchange=data.exchange.lower() if data.exchange else None,
                    bid=self._normalize_price(data.bid) if data.bid else None,
                    ask=self._normalize_price(data.ask) if data.ask else None,
                    open_price=self._normalize_price(data.open_price) if data.open_price else None,
                    high_price=self._normalize_price(data.high_price) if data.high_price else None,
                    low_price=self._normalize_price(data.low_price) if data.low_price else None,
                )
                return normalized_data
            else:
                # Generic normalization for other data types
                return data

        except Exception as e:
            self.logger.error(f"Data normalization failed: {e!s}")
            return data

    def _normalize_price(self, price: Decimal) -> Decimal:
        """Normalize price values."""
        if not price:
            return price

        # Ensure price precision (8 decimal places for crypto)
        return Decimal(str(float(price))).quantize(Decimal("0.00000001"))

    def _normalize_volume(self, volume: Decimal) -> Decimal:
        """Normalize volume values."""
        if not volume:
            return volume

        # Ensure volume precision (8 decimal places)
        return Decimal(str(float(volume))).quantize(Decimal("0.00000001"))

    def _normalize_timestamp(self, timestamp: datetime) -> datetime:
        """Normalize timestamp to UTC timezone."""
        if not timestamp:
            return timestamp

        # Ensure timezone-aware timestamp in UTC
        if timestamp.tzinfo is None:
            return timestamp.replace(tzinfo=timezone.utc)
        elif timestamp.tzinfo != timezone.utc:
            return timestamp.astimezone(timezone.utc)

        return timestamp

    async def _enrich_data(self, data: Any, data_type: str) -> Any:
        """Enrich data with additional calculated fields."""
        try:
            if not self.processing_config.enrichment_enabled:
                return data

            if data_type == "market_data" and isinstance(data, MarketData):
                # Add enrichment fields
                enriched_data = data

                # Calculate spread if bid and ask are available
                if data.bid and data.ask:
                    spread = data.ask - data.bid
                    spread_percentage = (spread / data.price) * 100 if data.price else 0

                    # Add enrichment metadata
                    if not hasattr(enriched_data, "metadata"):
                        enriched_data.metadata = {}

                    enriched_data.metadata.update(
                        {
                            "spread": float(spread),
                            "spread_percentage": float(spread_percentage),
                            "enriched_at": datetime.now(timezone.utc).isoformat(),
                        }
                    )

                # Add price change calculation if we have historical data
                symbol_key = f"{data.exchange}_{data.symbol}"
                if symbol_key in self.data_windows:
                    window = self.data_windows[symbol_key]
                    if window:
                        last_price = window[-1].price if hasattr(window[-1], "price") else None
                        if last_price and data.price:
                            price_change = calculate_percentage_change(
                                float(last_price), float(data.price)
                            )

                            if not hasattr(enriched_data, "metadata"):
                                enriched_data.metadata = {}

                            enriched_data.metadata["price_change_percentage"] = price_change

                return enriched_data
            else:
                return data

        except Exception as e:
            self.logger.error(f"Data enrichment failed: {e!s}")
            return data

    async def _aggregate_data(self, data: Any, data_type: str) -> Any:
        """Aggregate data over time windows."""
        try:
            if data_type == "market_data" and isinstance(data, MarketData):
                # Add to data window
                window_key = f"{data.exchange}_{data.symbol}"

                if window_key not in self.data_windows:
                    self.data_windows[window_key] = []

                self.data_windows[window_key].append(data)

                # Maintain window size
                if len(self.data_windows[window_key]) > self.processing_config.window_size:
                    self.data_windows[window_key].pop(0)

                # Calculate aggregations if we have enough data
                window = self.data_windows[window_key]
                if len(window) >= 5:  # Minimum data points for aggregation
                    aggregations = self._calculate_aggregations(window)

                    # Store aggregated data
                    if window_key not in self.aggregated_data:
                        self.aggregated_data[window_key] = {}

                    self.aggregated_data[window_key].update(aggregations)

                    # Add aggregation metadata
                    if not hasattr(data, "metadata"):
                        data.metadata = {}

                    data.metadata.update({"aggregations": aggregations, "window_size": len(window)})

            return data

        except Exception as e:
            self.logger.error(f"Data aggregation failed: {e!s}")
            return data

    def _calculate_aggregations(self, window: list[MarketData]) -> dict[str, float]:
        """Calculate aggregations for a data window."""
        try:
            prices = [float(item.price) for item in window if item.price]
            volumes = [float(item.volume) for item in window if item.volume]

            aggregations = {}

            if prices:
                aggregations.update(
                    {
                        "price_mean": statistics.mean(prices),
                        "price_median": statistics.median(prices),
                        "price_std": statistics.stdev(prices) if len(prices) > 1 else 0.0,
                        "price_min": min(prices),
                        "price_max": max(prices),
                        "price_range": max(prices) - min(prices),
                    }
                )

            if volumes:
                aggregations.update(
                    {
                        "volume_mean": statistics.mean(volumes),
                        "volume_median": statistics.median(volumes),
                        "volume_total": sum(volumes),
                        "volume_std": statistics.stdev(volumes) if len(volumes) > 1 else 0.0,
                    }
                )

            return aggregations

        except Exception as e:
            self.logger.error(f"Aggregation calculation failed: {e!s}")
            return {}

    async def _transform_data(self, data: Any, data_type: str) -> Any:
        """Apply data transformations."""
        try:
            # Apply any necessary data transformations
            # This could include log transforms, scaling, etc.
            return data

        except Exception as e:
            self.logger.error(f"Data transformation failed: {e!s}")
            return data

    async def _validate_data(self, data: Any, data_type: str) -> Any:
        """Validate processed data."""
        try:
            if not self.processing_config.validation_enabled:
                return data

            if data_type == "market_data" and isinstance(data, MarketData):
                # Validate market data
                is_valid = True
                validation_errors = []

                # Price validation
                if data.price:
                    try:
                        ValidationFramework.validate_price(data.price)
                    except ValueError as e:
                        is_valid = False
                        validation_errors.append(f"Invalid price: {str(e)}")

                # Volume validation
                if data.volume:
                    try:
                        ValidationFramework.validate_quantity(data.volume)
                    except ValueError as e:
                        is_valid = False
                        validation_errors.append(f"Invalid volume: {str(e)}")

                # Symbol validation
                if not data.symbol or len(data.symbol) < 3:
                    is_valid = False
                    validation_errors.append("Invalid symbol")

                if not is_valid:
                    self.logger.warning(f"Data validation failed: {validation_errors}")
                    # Could raise exception or mark data as invalid
                    if not hasattr(data, "metadata"):
                        data.metadata = {}
                    data.metadata["validation_errors"] = validation_errors

            return data

        except Exception as e:
            self.logger.error(f"Data validation failed: {e!s}")
            return data

    async def _filter_data(self, data: Any, data_type: str) -> Any:
        """Filter data based on criteria."""
        try:
            # Apply data filtering if needed
            # This could filter out unwanted or low-quality data
            return data

        except Exception as e:
            self.logger.error(f"Data filtering failed: {e!s}")
            return data

    async def get_aggregated_data(self, symbol: str, exchange: str | None = None) -> dict[str, Any]:
        """
        Get aggregated data for a symbol.

        Args:
            symbol: Trading symbol
            exchange: Exchange name (optional)

        Returns:
            Dict with aggregated data
        """
        try:
            if exchange:
                window_key = f"{exchange}_{symbol}"
                return self.aggregated_data.get(window_key, {})
            else:
                # Return aggregations from all exchanges for the symbol
                results = {}
                for key, aggregations in self.aggregated_data.items():
                    if symbol in key:
                        exchange_name = key.split("_")[0]
                        results[exchange_name] = aggregations
                return results

        except Exception as e:
            self.logger.error(f"Failed to get aggregated data for {symbol}: {e!s}")
            return {}

    def get_processing_statistics(self) -> dict[str, Any]:
        """Get data processing statistics."""
        return {
            "total_processed": self.stats["total_processed"],
            "successful_processed": self.stats["successful_processed"],
            "failed_processed": self.stats["failed_processed"],
            "success_rate": (
                self.stats["successful_processed"] / self.stats["total_processed"]
                if self.stats["total_processed"] > 0
                else 0.0
            ),
            "avg_processing_time": self.stats["avg_processing_time"],
            "last_processing_time": self.stats["last_processing_time"],
            "data_windows": {key: len(window) for key, window in self.data_windows.items()},
            "aggregated_data_keys": list(self.aggregated_data.keys()),
            "configuration": {
                "steps": [step.value for step in self.processing_config.steps],
                "window_size": self.processing_config.window_size,
                "aggregation_interval": self.processing_config.aggregation_interval,
                "normalization_method": self.processing_config.normalization_method,
                "enrichment_enabled": self.processing_config.enrichment_enabled,
                "validation_enabled": self.processing_config.validation_enabled,
            },
        }

    async def reset_windows(self) -> None:
        """Reset all data windows and aggregations."""
        try:
            self.data_windows.clear()
            self.aggregated_data.clear()
            self.logger.info("Data windows and aggregations reset")

        except Exception as e:
            self.logger.error(f"Failed to reset data windows: {e!s}")
            raise DataError(f"Failed to reset data windows: {e!s}")

    async def cleanup(self) -> None:
        """Cleanup data processor resources."""
        try:
            await self.reset_windows()
            self.logger.info("DataProcessor cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during DataProcessor cleanup: {e!s}")
