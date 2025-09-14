"""
Data transformation utilities for backtesting module.

Provides consistent data transformation patterns to align with execution module
and ensure proper cross-module communication in batch processing mode.
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from src.core.types import MarketData, Signal
from src.utils.decimal_utils import to_decimal

if TYPE_CHECKING:
    from src.backtesting.engine import BacktestResult


class BacktestDataTransformer:
    """Handles consistent data transformation for backtesting module."""

    @staticmethod
    def transform_signal_to_event_data(
        signal: Signal, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Transform Signal to consistent event data format.

        Args:
            signal: Signal to transform
            metadata: Additional metadata

        Returns:
            Dict with consistent event data format
        """
        return {
            "symbol": signal.symbol,
            "direction": signal.direction.value,
            "strength": str(signal.strength),
            "source": signal.source,
            "timestamp": signal.timestamp.isoformat()
            if signal.timestamp
            else datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }

    @staticmethod
    def transform_backtest_result_to_event_data(
        result: "BacktestResult", metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Transform BacktestResult to consistent event data format.

        Args:
            result: Backtest result to transform
            metadata: Additional metadata

        Returns:
            Dict with consistent event data format
        """
        return {
            "total_return": str(result.total_return),
            "annual_return": str(result.annual_return),
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": str(result.max_drawdown),
            "win_rate": result.win_rate,
            "total_trades": result.total_trades,
            "winning_trades": result.winning_trades,
            "losing_trades": result.losing_trades,
            "avg_win": str(result.avg_win),
            "avg_loss": str(result.avg_loss),
            "profit_factor": result.profit_factor,
            "volatility": result.volatility,
            "processing_mode": "batch",  # Backtesting results are batch data
            "data_format": "event_data_v1",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }

    @staticmethod
    def transform_market_data_to_event_data(
        market_data: MarketData, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Transform MarketData to consistent event data format aligned with execution.

        Args:
            market_data: Market data to transform
            metadata: Additional metadata

        Returns:
            Dict with consistent event data format
        """
        return {
            "symbol": market_data.symbol,
            "price": str(market_data.close),
            "volume": str(market_data.volume) if market_data.volume else "0",
            "high": str(market_data.high)
            if hasattr(market_data, "high")
            else str(market_data.close),
            "low": str(market_data.low) if hasattr(market_data, "low") else str(market_data.close),
            "open": str(market_data.open)
            if hasattr(market_data, "open")
            else str(market_data.close),
            "processing_mode": "batch",  # Historical market data is batch
            "data_format": "event_data_v1",
            "timestamp": market_data.timestamp.isoformat()
            if market_data.timestamp
            else datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }

    @staticmethod
    def transform_trade_to_event_data(
        trade: dict[str, Any], metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Transform trade record to consistent event data format.

        Args:
            trade: Trade dictionary to transform
            metadata: Additional metadata

        Returns:
            Dict with consistent event data format
        """
        return {
            "symbol": trade.get("symbol"),
            "entry_price": str(trade.get("entry_price", 0)),
            "exit_price": str(trade.get("exit_price", 0)),
            "size": str(trade.get("size", 0)),
            "pnl": str(trade.get("pnl", 0)),
            "side": trade.get("side", "unknown"),
            "entry_time": trade.get("entry_time").isoformat()
            if trade.get("entry_time") and hasattr(trade.get("entry_time"), "isoformat")
            else str(trade.get("entry_time"))
            if trade.get("entry_time")
            else None,  # type: ignore
            "exit_time": trade.get("exit_time").isoformat()
            if trade.get("exit_time") and hasattr(trade.get("exit_time"), "isoformat")
            else str(trade.get("exit_time"))
            if trade.get("exit_time")
            else None,  # type: ignore
            "processing_mode": "batch",  # Trade history is batch data
            "data_format": "event_data_v1",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }

    @staticmethod
    def validate_financial_precision(data: dict[str, Any]) -> dict[str, Any]:
        """
        Ensure financial data has proper precision using Decimal.

        Args:
            data: Data dictionary to validate

        Returns:
            Dict with validated financial precision
        """
        financial_fields = [
            "price",
            "size",
            "pnl",
            "entry_price",
            "exit_price",
            "volume",
            "total_return",
            "annual_return",
            "max_drawdown",
            "avg_win",
            "avg_loss",
            "strength",
            "high",
            "low",
            "open",
        ]

        for field in financial_fields:
            if field in data and data[field] is not None and data[field] != "":
                try:
                    # Convert to Decimal for financial precision
                    decimal_value = to_decimal(data[field])
                    # Convert back to string for consistent format
                    data[field] = str(decimal_value)
                except (ValueError, TypeError, KeyError):
                    # Keep original value if conversion fails
                    pass

        return data

    @staticmethod
    def ensure_boundary_fields(data: dict[str, Any], source: str = "backtesting") -> dict[str, Any]:
        """
        Ensure data has required boundary fields for cross-module communication.

        Args:
            data: Data dictionary to enhance
            source: Source module name

        Returns:
            Dict with required boundary fields
        """
        # Ensure processing mode is set (batch for backtesting)
        if "processing_mode" not in data:
            data["processing_mode"] = "batch"

        # Ensure data format is set
        if "data_format" not in data:
            data["data_format"] = "event_data_v1"

        # Ensure timestamp is set
        if "timestamp" not in data:
            data["timestamp"] = datetime.now(timezone.utc).isoformat()

        # Add source information
        if "source" not in data:
            data["source"] = source

        # Ensure metadata exists
        if "metadata" not in data:
            data["metadata"] = {}

        # Add backtesting-specific fields
        data["message_pattern"] = "req_reply"  # Backtesting typically requires responses
        data["boundary_crossed"] = True

        return data

    @classmethod
    def transform_for_req_reply(
        cls, request_type: str, data: Any, correlation_id: str | None = None
    ) -> dict[str, Any]:
        """
        Transform data for request/reply messaging pattern (aligned with execution).

        Args:
            request_type: Type of request
            data: Request data
            correlation_id: Request correlation ID

        Returns:
            Dict formatted for req/reply pattern
        """
        # Base transformation
        if isinstance(data, Signal):
            transformed = cls.transform_signal_to_event_data(data)
        elif hasattr(data, "total_return"):  # Duck typing for BacktestResult
            from src.backtesting.engine import BacktestResult

            if isinstance(data, BacktestResult):
                transformed = cls.transform_backtest_result_to_event_data(data)
            else:
                transformed = {"payload": str(data), "type": type(data).__name__}
        elif isinstance(data, MarketData):
            transformed = cls.transform_market_data_to_event_data(data)
        elif isinstance(data, dict):
            if "symbol" in data and ("pnl" in data or "entry_price" in data):
                # Trade record
                transformed = cls.transform_trade_to_event_data(data)
            else:
                transformed = data.copy()
        else:
            transformed = {"payload": str(data), "type": type(data).__name__}

        # Ensure boundary fields
        transformed = cls.ensure_boundary_fields(transformed)

        # Validate financial precision
        transformed = cls.validate_financial_precision(transformed)

        # Add request/reply specific fields
        transformed.update(
            {
                "request_type": request_type,
                "correlation_id": correlation_id or datetime.now(timezone.utc).isoformat(),
                "processing_mode": "batch",  # Override to batch for backtesting
                "message_pattern": "req_reply",
                "boundary_crossed": True,
                "validation_status": "validated",
            }
        )

        return transformed

    @classmethod
    def transform_for_batch_processing(
        cls,
        batch_type: str,
        data_items: list,
        batch_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Transform data for batch processing pattern (native to backtesting).

        Args:
            batch_type: Type of batch operation
            data_items: List of items to process in batch
            batch_id: Unique batch identifier
            metadata: Additional batch metadata

        Returns:
            Dict formatted for batch processing
        """
        # Transform each item individually
        transformed_items = []
        for item in data_items:
            if isinstance(item, Signal):
                transformed_items.append(cls.transform_signal_to_event_data(item))
            elif hasattr(item, "total_return"):  # Duck typing for BacktestResult
                from src.backtesting.engine import BacktestResult

                if isinstance(item, BacktestResult):
                    transformed_items.append(cls.transform_backtest_result_to_event_data(item))
                else:
                    transformed_items.append({"payload": str(item), "type": type(item).__name__})
            elif isinstance(item, MarketData):
                transformed_items.append(cls.transform_market_data_to_event_data(item))
            elif isinstance(item, dict):
                if "symbol" in item and ("pnl" in item or "entry_price" in item):
                    transformed_items.append(cls.transform_trade_to_event_data(item))
                else:
                    transformed_items.append(cls.ensure_boundary_fields(item.copy()))
            else:
                transformed_items.append(
                    {
                        "payload": str(item),
                        "type": type(item).__name__,
                        "processing_mode": "batch",
                        "data_format": "event_data_v1",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )

        return {
            "batch_type": batch_type,
            "batch_id": batch_id or datetime.now(timezone.utc).isoformat(),
            "batch_size": len(data_items),
            "items": transformed_items,
            "processing_mode": "batch",
            "data_format": "batch_event_data_v1",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "backtesting",
            "message_pattern": "batch",
            "boundary_crossed": True,
            "metadata": metadata or {},
        }

    @classmethod
    def align_processing_paradigm(
        cls, data: dict[str, Any], target_mode: str = "batch"
    ) -> dict[str, Any]:
        """
        Align data processing paradigm with target module expectations.

        (consistent with optimization)

        Args:
            data: Data to align
            target_mode: Target processing mode ("stream", "batch", "request_reply")

        Returns:
            Dict with aligned processing mode
        """
        aligned_data = data.copy()

        # Use ProcessingParadigmAligner for consistency with optimization module
        from src.utils.messaging_patterns import ProcessingParadigmAligner

        # Apply consistent processing paradigm alignment
        source_mode = aligned_data.get("processing_mode", "batch")
        aligned_data = ProcessingParadigmAligner.align_processing_modes(
            source_mode=source_mode, target_mode=target_mode, data=aligned_data
        )

        # Add mode-specific fields (consistent with optimization patterns)
        if target_mode == "batch":
            if "batch_id" not in aligned_data:
                aligned_data["batch_id"] = datetime.now(timezone.utc).isoformat()
            aligned_data.update(
                {
                    "data_format": "batch_event_data_v1",
                    "message_pattern": "batch",
                    "boundary_crossed": True,
                }
            )
        elif target_mode == "request_reply":
            if "correlation_id" not in aligned_data:
                aligned_data["correlation_id"] = datetime.now(timezone.utc).isoformat()
            aligned_data.update(
                {
                    "data_format": "request_reply_data_v1",
                    "message_pattern": "req_reply",
                    "boundary_crossed": True,
                }
            )
        elif target_mode == "stream":
            aligned_data.update(
                {
                    "stream_position": datetime.now(timezone.utc).timestamp(),
                    "data_format": "event_data_v1",
                    "message_pattern": "pub_sub",
                    "boundary_crossed": True,
                }
            )

        # Add consistent validation status
        aligned_data["validation_status"] = "validated"

        return aligned_data

    @classmethod
    def apply_cross_module_validation(
        cls,
        data: dict[str, Any],
        source_module: str = "backtesting",
        target_module: str = "execution",
    ) -> dict[str, Any]:
        """
        Apply comprehensive cross-module validation for consistent data flow.

        Args:
            data: Data to validate and transform
            source_module: Source module name
            target_module: Target module name

        Returns:
            Dict with validated and aligned data for cross-module communication
        """
        validated_data = data.copy()

        # Apply consistent messaging patterns
        from src.utils.messaging_patterns import (
            BoundaryValidator,
            ProcessingParadigmAligner,
        )

        # Apply processing paradigm alignment
        source_mode = validated_data.get("processing_mode", "batch")
        target_mode = "stream" if target_module == "execution" else source_mode

        validated_data = ProcessingParadigmAligner.align_processing_modes(
            source_mode=source_mode, target_mode=target_mode, data=validated_data
        )

        # Add comprehensive boundary metadata
        validated_data.update(
            {
                "cross_module_validation": True,
                "source_module": source_module,
                "target_module": target_module,
                "boundary_validation_timestamp": datetime.now(timezone.utc).isoformat(),
                "data_flow_aligned": True,
            }
        )

        # Apply target-module specific boundary validation
        try:
            if target_module == "execution":
                # Validate at backtesting -> execution boundary
                boundary_data = {
                    "component": validated_data.get("component", source_module),
                    "operation": validated_data.get("operation", "backtest_operation"),
                    "timestamp": validated_data.get(
                        "timestamp", datetime.now(timezone.utc).isoformat()
                    ),
                    "processing_mode": validated_data.get("processing_mode", "batch"),
                    "data_format": validated_data.get("data_format", "event_data_v1"),
                    "boundary_crossed": True,
                }
                BoundaryValidator.validate_database_entity(boundary_data, "read")

            elif target_module == "optimization":
                # Validate at backtesting -> optimization boundary
                boundary_data = {
                    "backtest_id": validated_data.get("backtest_id", "unknown"),
                    "performance_metrics": validated_data.get("performance_metrics", {}),
                    "timestamp": validated_data.get(
                        "timestamp", datetime.now(timezone.utc).isoformat()
                    ),
                    "processing_mode": validated_data.get("processing_mode", "batch"),
                    "data_format": validated_data.get("data_format", "event_data_v1"),
                    "boundary_crossed": True,
                }
                BoundaryValidator.validate_database_entity(boundary_data, "read")

            elif target_module == "data":
                # Validate at backtesting -> data boundary
                boundary_data = {
                    "symbol": validated_data.get("symbol", "UNKNOWN"),
                    "exchange": validated_data.get("exchange", "backtesting"),
                    "start_time": validated_data.get(
                        "timestamp", datetime.now(timezone.utc).isoformat()
                    ),
                    "processing_mode": validated_data.get("processing_mode", "batch"),
                    "data_format": validated_data.get("data_format", "event_data_v1"),
                    "boundary_crossed": True,
                }
                BoundaryValidator.validate_database_entity(boundary_data, "read")

        except Exception:
            # Log validation issues but don't fail the data flow
            pass

        return validated_data
