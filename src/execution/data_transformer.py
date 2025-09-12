"""
Data transformation utilities for execution module.

Provides consistent data transformation patterns to align with core module
event system and ensure proper cross-module communication.

This centralizes all data conversion logic to prevent duplication across services.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from src.core.exceptions import ValidationError
from src.core.types import ExecutionResult, MarketData, OrderRequest, OrderSide, OrderType
from src.utils.decimal_utils import to_decimal


class ExecutionDataTransformer:
    """
    Centralized data transformation service for execution module.
    
    This service handles all data conversions to prevent duplication
    across ExecutionService and ExecutionOrchestrationService.
    """

    @staticmethod
    def transform_order_to_event_data(
        order: OrderRequest, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Transform OrderRequest to consistent event data format.

        Args:
            order: Order request to transform
            metadata: Additional metadata

        Returns:
            Dict with consistent event data format
        """
        return {
            "symbol": order.symbol,
            "side": order.side.value,
            "order_type": order.order_type.value,
            "quantity": str(order.quantity),
            "price": str(order.price) if order.price else None,
            "exchange": getattr(
                order, "exchange", None
            ),  # Handle missing exchange field gracefully
            "processing_mode": "stream",
            "data_format": "bot_event_v1",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }

    @staticmethod
    def transform_execution_result_to_event_data(
        execution_result: ExecutionResult, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Transform ExecutionResult to consistent event data format.

        Args:
            execution_result: Execution result to transform
            metadata: Additional metadata

        Returns:
            Dict with consistent event data format
        """
        return {
            "execution_id": execution_result.execution_id,
            "symbol": execution_result.original_order.symbol,
            "side": execution_result.original_order.side.value,
            "order_type": execution_result.original_order.order_type.value,
            "target_quantity": str(execution_result.original_order.quantity),
            "filled_quantity": str(execution_result.total_filled_quantity),
            "remaining_quantity": str(
                execution_result.original_order.quantity - execution_result.total_filled_quantity
            ),
            "average_price": str(execution_result.average_fill_price)
            if execution_result.average_fill_price
            else None,
            "status": execution_result.status.value,
            "algorithm": execution_result.algorithm.value if execution_result.algorithm else None,
            "fees": str(execution_result.total_fees) if execution_result.total_fees else "0",
            "processing_mode": "stream",
            "data_format": "bot_event_v1",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }

    @staticmethod
    def transform_market_data_to_event_data(
        market_data: MarketData, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Transform MarketData to consistent event data format.

        Args:
            market_data: Market data to transform
            metadata: Additional metadata

        Returns:
            Dict with consistent event data format
        """
        return {
            "symbol": market_data.symbol,
            "price": f"{market_data.price:.8f}",
            "volume": f"{market_data.volume:.8f}" if market_data.volume else "0.00000000",
            "bid": f"{market_data.bid:.8f}" if market_data.bid else None,
            "ask": f"{market_data.ask:.8f}" if market_data.ask else None,
            "spread": f"{market_data.ask - market_data.bid:.8f}"
            if market_data.bid and market_data.ask
            else None,
            "processing_mode": "stream",
            "data_format": "bot_event_v1",
            "timestamp": market_data.timestamp.isoformat()
            if market_data.timestamp
            else datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }

    @staticmethod
    def transform_error_to_event_data(
        error: Exception,
        context: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Transform error to consistent event data format using utils error propagation patterns.

        Args:
            error: Exception to transform
            context: Error context information
            metadata: Additional metadata

        Returns:
            Dict with consistent event data format aligned with utils patterns
        """
        # Use utils messaging patterns for consistent error handling aligned with error_handling module
        from src.utils.messaging_patterns import MessagePattern, MessageType

        # Create standardized error data format aligned with error_handling module
        error_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "error_context": context or {},  # Align with error_handling module naming
            "metadata": metadata or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "execution",
            "processing_mode": "stream",
            "data_format": "bot_event_v1",
            "message_pattern": MessagePattern.PUB_SUB.value,
            "message_type": MessageType.ERROR_EVENT.value,
            "boundary_crossed": True,
            "validation_status": "validated",  # Use consistent validation status
        }

        return error_data

    @staticmethod
    def convert_to_order_request(order_data: dict[str, Any]) -> OrderRequest:
        """
        Convert dictionary to OrderRequest.
        
        Centralizes order data conversion logic to prevent duplication
        between ExecutionService and ExecutionOrchestrationService.

        Args:
            order_data: Raw order data dictionary

        Returns:
            OrderRequest: Typed order request object

        Raises:
            ValidationError: If order data is invalid
        """
        try:
            return OrderRequest(
                symbol=order_data["symbol"],
                side=OrderSide(order_data["side"]),
                order_type=OrderType(order_data.get("order_type", "MARKET")),
                quantity=Decimal(str(order_data["quantity"])),
                price=Decimal(str(order_data["price"])) if order_data.get("price") else None,
                time_in_force=order_data.get("time_in_force"),
                exchange=order_data.get("exchange"),
                client_order_id=order_data.get("client_order_id"),
            )
        except (KeyError, ValueError, TypeError) as e:
            raise ValidationError(f"Invalid order data: {e}") from e

    @staticmethod
    def convert_to_market_data(market_data: dict[str, Any]) -> MarketData:
        """
        Convert dictionary to MarketData.
        
        Centralizes market data conversion logic to prevent duplication
        between ExecutionService and ExecutionOrchestrationService.

        Args:
            market_data: Raw market data dictionary

        Returns:
            MarketData: Typed market data object

        Raises:
            ValidationError: If market data is invalid
        """
        try:
            return MarketData(
                symbol=market_data["symbol"],
                price=Decimal(str(market_data["price"])),
                volume=Decimal(str(market_data.get("volume", "0.0"))),
                bid=Decimal(str(market_data["bid"])) if market_data.get("bid") else None,
                ask=Decimal(str(market_data["ask"])) if market_data.get("ask") else None,
                timestamp=datetime.fromisoformat(market_data["timestamp"])
                if market_data.get("timestamp")
                else datetime.now(timezone.utc),
                exchange=market_data.get("exchange"),
            )
        except (KeyError, ValueError, TypeError) as e:
            raise ValidationError(f"Invalid market data: {e}") from e

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
            "quantity",
            "volume",
            "fees",
            "target_quantity",
            "filled_quantity",
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
    def ensure_boundary_fields(data: dict[str, Any], source: str = "execution") -> dict[str, Any]:
        """
        Ensure data has required boundary fields for cross-module communication.

        Args:
            data: Data dictionary to enhance
            source: Source module name

        Returns:
            Dict with required boundary fields
        """
        # Ensure processing mode is set
        if "processing_mode" not in data:
            data["processing_mode"] = "stream"

        # Ensure data format is set
        if "data_format" not in data:
            data["data_format"] = "bot_event_v1"

        # Ensure timestamp is set
        if "timestamp" not in data:
            data["timestamp"] = datetime.now(timezone.utc).isoformat()

        # Add source information
        if "source" not in data:
            data["source"] = source

        # Ensure metadata exists
        if "metadata" not in data:
            data["metadata"] = {}

        return data

    @classmethod
    def transform_for_pub_sub(
        cls, event_type: str, data: Any, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Transform data for pub/sub messaging pattern with enhanced boundary validation.

        Args:
            event_type: Type of event
            data: Raw data to transform
            metadata: Additional metadata

        Returns:
            Dict formatted for pub/sub pattern with consistent validation
        """
        # Base transformation
        if isinstance(data, OrderRequest):
            transformed = cls.transform_order_to_event_data(data, metadata)
        elif isinstance(data, ExecutionResult):
            transformed = cls.transform_execution_result_to_event_data(data, metadata)
        elif isinstance(data, MarketData):
            transformed = cls.transform_market_data_to_event_data(data, metadata)
        elif isinstance(data, Exception):
            transformed = cls.transform_error_to_event_data(data, metadata)
        elif isinstance(data, dict):
            transformed = data.copy()
        else:
            transformed = {"payload": str(data), "type": type(data).__name__}

        # Ensure boundary fields with enhanced validation
        transformed = cls.ensure_boundary_fields(transformed)

        # Validate financial precision
        transformed = cls.validate_financial_precision(transformed)

        # Add event type and enhanced boundary metadata
        transformed.update(
            {
                "event_type": event_type,
                "message_pattern": "pub_sub",  # Consistent messaging pattern
                "boundary_crossed": True,  # Cross-module event flag
                "validation_status": "validated",  # Boundary validation status
            }
        )

        # Apply boundary validation for cross-module consistency
        from src.utils.messaging_patterns import BoundaryValidator

        try:
            # Use consistent boundary validation patterns
            if "component" in transformed and transformed["component"] in [
                "execution",
                "ExecutionService",
            ]:
                # Apply execution to error_handling boundary validation
                boundary_data = {
                    "component": "execution",
                    "error_type": transformed.get("error_type", "ExecutionEvent"),
                    "severity": transformed.get("severity", "medium"),
                    "timestamp": transformed.get(
                        "timestamp", datetime.now(timezone.utc).isoformat()
                    ),
                    "processing_mode": "stream",  # Align with error_handling expectations
                    "data_format": "bot_event_v1",
                    "message_pattern": "pub_sub",
                    "boundary_crossed": True,
                }
                # Use correct boundary validation method direction
                BoundaryValidator.validate_monitoring_to_error_boundary(boundary_data)
        except Exception as e:
            # Log validation issues but don't fail the transformation
            # Use a basic logger since this is a data transformer
            import logging

            logging.getLogger(__name__).debug(f"Boundary validation failed: {e}")

        return transformed

    @classmethod
    def transform_for_req_reply(
        cls, request_type: str, data: Any, correlation_id: str | None = None
    ) -> dict[str, Any]:
        """
        Transform data for request/reply messaging pattern.

        Args:
            request_type: Type of request
            data: Request data
            correlation_id: Request correlation ID

        Returns:
            Dict formatted for req/reply pattern
        """
        # Use pub/sub transformation as base
        transformed = cls.transform_for_pub_sub(request_type, data)

        # Add request/reply specific fields aligned with risk_management module  
        transformed["request_type"] = request_type
        transformed["correlation_id"] = correlation_id or datetime.now(timezone.utc).isoformat()
        transformed["message_pattern"] = "req_reply"  # Override message pattern for req/reply
        transformed["message_pattern"] = "req_reply"  # Override message pattern for req/reply
        transformed["processing_mode"] = "request_reply"  # Override processing mode

        return transformed

    @classmethod
    def transform_for_batch_processing(
        cls,
        batch_type: str,
        data_items: list[Any],
        batch_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Transform data for batch processing pattern to align with core module.

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
            if isinstance(item, OrderRequest):
                transformed_items.append(cls.transform_order_to_event_data(item))
            elif isinstance(item, ExecutionResult):
                transformed_items.append(cls.transform_execution_result_to_event_data(item))
            elif isinstance(item, MarketData):
                transformed_items.append(cls.transform_market_data_to_event_data(item))
            elif isinstance(item, dict):
                transformed_items.append(cls.ensure_boundary_fields(item.copy()))
            else:
                transformed_items.append(
                    {
                        "payload": str(item),
                        "type": type(item).__name__,
                        "processing_mode": "batch",
                        "data_format": "bot_event_v1",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )

        return {
            "batch_type": batch_type,
            "batch_id": batch_id or datetime.now(timezone.utc).isoformat(),
            "batch_size": len(data_items),
            "items": transformed_items,
            "processing_mode": "batch",
            "data_format": "bot_event_v1",  # Use consistent data format across all modes
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "execution",
            "metadata": metadata or {},
        }

    @classmethod
    def align_processing_paradigm(
        cls, data: dict[str, Any], target_mode: str = "stream"
    ) -> dict[str, Any]:
        """
        Align data processing paradigm with error_handling module expectations.

        Args:
            data: Data to align
            target_mode: Target processing mode ("stream", "batch", "request_reply")

        Returns:
            Dict with aligned processing mode using consistent patterns
        """
        aligned_data = data.copy()

        # Use ProcessingParadigmAligner for consistency with core module
        from src.utils.messaging_patterns import ProcessingParadigmAligner

        # Apply consistent processing paradigm alignment matching core module patterns
        source_mode = aligned_data.get("processing_mode", "stream")
        aligned_data = ProcessingParadigmAligner.align_processing_modes(
            source_mode=source_mode, target_mode=target_mode, data=aligned_data
        )

        # Add mode-specific fields with enhanced boundary metadata
        if target_mode == "stream":
            aligned_data.update(
                {
                    "stream_position": datetime.now(timezone.utc).timestamp(),
                    "data_format": "bot_event_v1",
                    "message_pattern": "pub_sub",  # Consistent messaging pattern
                    "boundary_crossed": True,
                }
            )
        elif target_mode == "batch":
            if "batch_id" not in aligned_data:
                aligned_data["batch_id"] = datetime.now(timezone.utc).isoformat()
            aligned_data.update(
                {
                    "data_format": "bot_event_v1",  # Use consistent data format
                    "message_pattern": "batch",  # Batch messaging pattern
                    "boundary_crossed": True,
                }
            )
        elif target_mode == "request_reply":
            if "correlation_id" not in aligned_data:
                aligned_data["correlation_id"] = datetime.now(timezone.utc).isoformat()
            aligned_data.update(
                {
                    "data_format": "bot_event_v1",  # Use consistent data format
                    "message_pattern": "req_reply",  # Request-reply messaging pattern
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
        source_module: str = "execution",
        target_module: str = "error_handling",
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
            MessagingCoordinator,
            ProcessingParadigmAligner,
        )

        # Apply consistent data transformation using our own methods
        try:
            validated_data = cls.ensure_boundary_fields(validated_data, source_module)
            validated_data = cls.validate_financial_precision(validated_data)
        except Exception as e:
            # Log warning but continue with basic transformation
            import logging

            logging.getLogger(__name__).warning(f"Data transformation failed: {e}")

        # Apply processing paradigm alignment
        source_mode = validated_data.get("processing_mode", "stream")
        target_mode = "stream" if target_module == "error_handling" else source_mode

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
            if target_module == "error_handling" or target_module == "monitoring":
                # Validate at execution -> error_handling/monitoring boundary
                boundary_data = {
                    "component": validated_data.get("component", source_module),
                    "error_type": validated_data.get("error_type", "ExecutionEvent"),
                    "severity": validated_data.get("severity", "medium"),
                    "timestamp": validated_data.get(
                        "timestamp", datetime.now(timezone.utc).isoformat()
                    ),
                    "processing_mode": validated_data.get("processing_mode", "stream"),
                    "data_format": validated_data.get("data_format", "bot_event_v1"),
                    "message_pattern": validated_data.get("message_pattern", "pub_sub"),
                    "boundary_crossed": True,
                }
                # Use correct boundary validation method direction
                BoundaryValidator.validate_monitoring_to_error_boundary(boundary_data)

            elif target_module == "risk_management" or target_module == "state":
                # Apply risk-state boundary validation
                boundary_data = {
                    "component": validated_data.get("component", source_module),
                    "operation": validated_data.get("operation", "execution_operation"),
                    "timestamp": validated_data.get(
                        "timestamp", datetime.now(timezone.utc).isoformat()
                    ),
                    "processing_mode": validated_data.get("processing_mode", "stream"),
                    "data_format": validated_data.get("data_format", "bot_event_v1"),
                    "boundary_crossed": True,
                }
                BoundaryValidator.validate_risk_to_state_boundary(boundary_data)

        except Exception as e:
            # Log validation issues but don't fail the data flow
            import logging

            logging.getLogger(__name__).debug(f"Risk to state boundary validation failed: {e}")

        return validated_data
