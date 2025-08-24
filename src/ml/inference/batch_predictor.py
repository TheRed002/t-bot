"""
Batch Prediction Infrastructure for ML Models.

This module provides efficient batch prediction capabilities for processing
large datasets with optimized memory usage and parallel processing.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.core.base.service import BaseService
from src.core.exceptions import ValidationError
from src.core.types.base import ConfigDict
from src.utils.decorators import UnifiedDecorator

# Initialize decorator instance
dec = UnifiedDecorator()


class BatchPredictorConfig(BaseModel):
    """Configuration for batch predictor service."""

    batch_size: int = Field(default=1000, description="Number of samples to process per batch")
    max_workers: int = Field(default=4, description="Maximum number of parallel workers")
    use_multiprocessing: bool = Field(
        default=False, description="Use multiprocessing for CPU-bound tasks"
    )
    chunk_size: int = Field(default=10000, description="Chunk size for large datasets")
    max_memory_mb: int = Field(default=1024, description="Maximum memory usage in MB")
    save_to_database: bool = Field(default=True, description="Save predictions to database")
    enable_confidence_scores: bool = Field(default=True, description="Calculate confidence scores")


class BatchPredictorService(BaseService):
    """
    Batch prediction system for efficient processing of large datasets.

    This service provides optimized batch prediction capabilities with memory
    management, parallel processing, and database integration for storing
    results using proper service patterns without direct database access.

    All data operations go through DataService dependency.
    """

    def __init__(
        self,
        config: ConfigDict | None = None,
        correlation_id: str | None = None,
    ):
        """
        Initialize the batch predictor service.

        Args:
            config: Service configuration
            correlation_id: Request correlation ID
        """
        super().__init__(
            name="BatchPredictorService",
            config=config,
            correlation_id=correlation_id,
        )

        # Parse batch predictor configuration
        bp_config_dict = (config or {}).get("batch_predictor", {})
        self.bp_config = BatchPredictorConfig(**bp_config_dict)

        # Service dependencies - resolved during startup
        self.data_service: Any = None
        self.model_registry: Any = None
        self.feature_engineering_service: Any = None

        # Memory management
        self.memory_threshold = 0.8  # Use 80% of available memory

        # Performance tracking
        self.prediction_count = 0
        self.total_processing_time = 0.0

        # Add required dependencies
        self.add_dependency("DataService")
        self.add_dependency("ModelRegistry")
        self.add_dependency("FeatureEngineeringService")

    async def _do_start(self) -> None:
        """Start the batch predictor service."""
        await super()._do_start()

        # Resolve dependencies
        self.data_service = self.resolve_dependency("DataService")
        self.model_registry = self.resolve_dependency("ModelRegistry")
        self.feature_engineering_service = self.resolve_dependency("FeatureEngineeringService")

        self._logger.info(
            "Batch predictor service started successfully",
            config=self.bp_config.dict(),
            dependencies_resolved=3,
        )

    async def _do_stop(self) -> None:
        """Stop the batch predictor service."""
        await super()._do_stop()

    @dec.enhance(log=True, monitor=True, log_level="info")
    async def predict_batch(
        self,
        model_name: str,
        data: pd.DataFrame,
        symbol: str,
        feature_types: list[str] | None = None,
        save_to_db: bool = True,
        output_file: str | None = None,
    ) -> pd.DataFrame:
        """
        Perform batch predictions on a large dataset.

        Args:
            model_name: Name of the model to use
            data: Input market data
            symbol: Trading symbol
            feature_types: Types of features to create
            save_to_db: Whether to save results to database
            output_file: Optional file to save results

        Returns:
            DataFrame with predictions

        Raises:
            ValidationError: If model or data is invalid
        """
        try:
            if data.empty:
                raise ValidationError("Input data cannot be empty")

            # Load model
            model = await self._load_model(model_name)

            # Calculate optimal batch size based on memory
            optimal_batch_size = self._calculate_optimal_batch_size(data)

            self._logger.info(
                "Starting batch prediction",
                model_name=model_name,
                data_points=len(data),
                batch_size=optimal_batch_size,
                symbol=symbol,
            )

            # Process data in batches
            predictions = []
            batch_count = 0

            for batch_data in self._create_batches(data, optimal_batch_size):
                batch_count += 1

                # Create features for batch using FeatureEngineeringService
                feature_request = {
                    "market_data": batch_data.to_dict("records"),
                    "symbol": symbol,
                    "feature_types": feature_types,
                    "enable_selection": False,
                    "enable_preprocessing": True,
                }
                feature_response = await self.feature_engineering_service.compute_features(
                    feature_request
                )
                features_df = pd.DataFrame(feature_response.feature_set.features)

                # Make predictions
                batch_predictions = await self._predict_batch_chunk(
                    model, features_df, batch_data.index
                )

                predictions.append(batch_predictions)

                # Log progress
                if batch_count % 10 == 0:
                    self._logger.info(
                        "Batch processing progress",
                        completed_batches=batch_count,
                        total_predictions=len(predictions) * optimal_batch_size,
                        model_name=model_name,
                    )

            # Combine all predictions
            result_df = pd.concat(predictions, ignore_index=False)
            result_df = result_df.sort_index()

            # Add metadata
            result_df["model_name"] = model_name
            result_df["symbol"] = symbol
            result_df["prediction_timestamp"] = datetime.utcnow()

            # Save to database if requested (using DataService)
            if save_to_db and self.bp_config.save_to_database:
                await self._save_predictions_to_db(result_df, model_name, symbol)

            # Save to file if requested
            if output_file:
                await self._save_predictions_to_file(result_df, output_file)

            # Update performance metrics
            self.prediction_count += len(result_df)

            self._logger.info(
                "Batch prediction completed",
                model_name=model_name,
                total_predictions=len(result_df),
                batches_processed=batch_count,
                symbol=symbol,
            )

            return result_df

        except Exception as e:
            self._logger.error(
                "Batch prediction failed", model_name=model_name, symbol=symbol, error=str(e)
            )
            raise ValidationError(f"Batch prediction failed: {e}")

    @dec.enhance(log=True, monitor=True, log_level="info")
    async def predict_multiple_symbols(
        self,
        model_name: str,
        data_dict: dict[str, pd.DataFrame],
        feature_types: list[str] | None = None,
        parallel: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """
        Perform batch predictions for multiple symbols.

        Args:
            model_name: Name of the model to use
            data_dict: Dictionary mapping symbols to their data
            feature_types: Types of features to create
            parallel: Whether to process symbols in parallel

        Returns:
            Dictionary mapping symbols to prediction results

        Raises:
            ValidationError: If any prediction fails
        """
        try:
            if not data_dict:
                raise ValidationError("Data dictionary cannot be empty")

            self._logger.info(
                "Starting multi-symbol batch prediction",
                model_name=model_name,
                symbol_count=len(data_dict),
                parallel=parallel,
            )

            if parallel and len(data_dict) > 1:
                # Process symbols in parallel
                tasks = []
                for symbol, data in data_dict.items():
                    task = self.predict_batch(
                        model_name=model_name,
                        data=data,
                        symbol=symbol,
                        feature_types=feature_types,
                        save_to_db=True,
                        output_file=None,
                    )
                    tasks.append((symbol, task))

                # Execute tasks concurrently
                results = {}
                for symbol, task in tasks:
                    try:
                        result = await task
                        results[symbol] = result
                    except Exception as e:
                        self._logger.error("Symbol prediction failed", symbol=symbol, error=str(e))
                        # Continue with other symbols
                        results[symbol] = pd.DataFrame()

            else:
                # Process symbols sequentially
                results = {}
                for symbol, data in data_dict.items():
                    try:
                        result = await self.predict_batch(
                            model_name=model_name,
                            data=data,
                            symbol=symbol,
                            feature_types=feature_types,
                            save_to_db=True,
                            output_file=None,
                        )
                        results[symbol] = result
                    except Exception as e:
                        self._logger.error("Symbol prediction failed", symbol=symbol, error=str(e))
                        results[symbol] = pd.DataFrame()

            successful_symbols = [s for s, r in results.items() if not r.empty]

            self._logger.info(
                "Multi-symbol batch prediction completed",
                model_name=model_name,
                total_symbols=len(data_dict),
                successful_symbols=len(successful_symbols),
                parallel=parallel,
            )

            return results

        except Exception as e:
            self._logger.error(
                "Multi-symbol batch prediction failed", model_name=model_name, error=str(e)
            )
            raise ValidationError(f"Multi-symbol prediction failed: {e}")

    @dec.enhance(log=True, monitor=True, log_level="info")
    async def backfill_predictions(
        self,
        model_name: str,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        data_source: str = "database",
    ) -> pd.DataFrame:
        """
        Backfill historical predictions for a given period.

        Args:
            model_name: Name of the model to use
            symbol: Trading symbol
            start_date: Start date for backfill
            end_date: End date for backfill
            data_source: Source of historical data

        Returns:
            DataFrame with backfilled predictions

        Raises:
            ValidationError: If backfill fails
        """
        try:
            self._logger.info(
                "Starting prediction backfill",
                model_name=model_name,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                data_source=data_source,
            )

            # Load historical data
            historical_data = await self._load_historical_data(
                symbol, start_date, end_date, data_source
            )

            if historical_data.empty:
                raise ValidationError(f"No historical data found for {symbol}")

            # Perform batch prediction
            predictions = await self.predict_batch(
                model_name=model_name,
                data=historical_data,
                symbol=symbol,
                save_to_db=True,
                output_file=None,
            )

            self._logger.info(
                "Prediction backfill completed",
                model_name=model_name,
                symbol=symbol,
                predictions_count=len(predictions),
            )

            return predictions

        except Exception as e:
            self._logger.error(
                "Prediction backfill failed", model_name=model_name, symbol=symbol, error=str(e)
            )
            raise ValidationError(f"Prediction backfill failed: {e}")

    async def _load_model(self, model_name: str) -> BaseModel:
        """Load model from registry."""
        try:
            model_info = self.model_registry.get_latest_model(model_name)
            if not model_info:
                raise ValidationError(f"Model {model_name} not found")

            # Load the actual model
            model = self.model_registry.load_model(model_info["id"])
            return model

        except Exception as e:
            self._logger.error(f"Failed to load model {model_name}: {e}")
            raise ValidationError(f"Model loading failed: {e}")

    def _calculate_optimal_batch_size(self, data: pd.DataFrame) -> int:
        """Calculate optimal batch size based on data size and memory."""
        try:
            # Estimate memory usage per row (rough approximation)
            sample_row = data.iloc[:1] if not data.empty else pd.DataFrame()
            memory_per_row = (
                sample_row.memory_usage(deep=True).sum() if not sample_row.empty else 1000
            )

            # Calculate maximum rows that fit in memory
            available_memory = self.bp_config.max_memory_mb * 1024 * 1024 * self.memory_threshold
            max_rows = int(available_memory / memory_per_row)

            # Use the smaller of configured batch size or memory-based limit
            optimal_size = min(self.bp_config.batch_size, max_rows, len(data))

            self._logger.debug(
                "Calculated optimal batch size",
                configured_batch_size=self.bp_config.batch_size,
                memory_based_limit=max_rows,
                data_size=len(data),
                optimal_size=optimal_size,
            )

            return max(1, optimal_size)  # Ensure at least 1

        except Exception as e:
            self._logger.warning(f"Failed to calculate optimal batch size: {e}")
            return min(self.bp_config.batch_size, len(data))

    def _create_batches(self, data: pd.DataFrame, batch_size: int):
        """Create data batches for processing."""
        for i in range(0, len(data), batch_size):
            yield data.iloc[i : i + batch_size]

    async def _predict_batch_chunk(
        self, model: BaseModel, features: pd.DataFrame, indices: pd.Index
    ) -> pd.DataFrame:
        """Make predictions for a single batch chunk."""
        try:
            # Make predictions
            predictions = model.predict(features)

            # Create result DataFrame
            result = pd.DataFrame(predictions, index=indices, columns=["prediction"])

            # Add confidence scores if available and enabled
            if self.bp_config.enable_confidence_scores and hasattr(model, "predict_proba"):
                try:
                    probabilities = model.predict_proba(features)
                    if probabilities.shape[1] >= 2:
                        result["confidence"] = np.max(probabilities, axis=1)
                except Exception:
                    result["confidence"] = 0.5  # Default confidence
            else:
                result["confidence"] = 0.5  # Default confidence

            return result

        except Exception as e:
            self._logger.error(f"Batch chunk prediction failed: {e}")
            # Return empty predictions with same index
            return pd.DataFrame({"prediction": np.nan, "confidence": 0.0}, index=indices)

    async def _save_predictions_to_db(
        self, predictions: pd.DataFrame, model_name: str, symbol: str
    ):
        """Save predictions to database using DataService."""
        try:
            prediction_data = []

            for idx, row in predictions.iterrows():
                prediction_record = {
                    "model_name": model_name,
                    "symbol": symbol,
                    "timestamp": idx.isoformat() if hasattr(idx, "isoformat") else str(idx),
                    "prediction_value": (
                        float(row["prediction"]) if not pd.isna(row["prediction"]) else None
                    ),
                    "confidence_score": (
                        float(row["confidence"]) if not pd.isna(row["confidence"]) else None
                    ),
                    "features_hash": hash(str(row.to_dict())),  # Simple hash of features
                    "prediction_timestamp": datetime.utcnow().isoformat(),
                }
                prediction_data.append(prediction_record)

            # Save through DataService
            await self.data_service.save_ml_predictions(prediction_data)

            self._logger.info(
                "Predictions saved to database",
                model_name=model_name,
                symbol=symbol,
                count=len(prediction_data),
            )

        except Exception as e:
            self._logger.error(
                "Failed to save predictions to database",
                model_name=model_name,
                symbol=symbol,
                error=str(e),
            )

    async def _save_predictions_to_file(self, predictions: pd.DataFrame, output_file: str):
        """Save predictions to file."""
        try:
            file_path = Path(output_file)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Determine file format based on extension
            if file_path.suffix.lower() == ".csv":
                predictions.to_csv(file_path)
            elif file_path.suffix.lower() in [".pkl", ".pickle"]:
                predictions.to_pickle(file_path)
            elif file_path.suffix.lower() in [".xlsx", ".xls"]:
                predictions.to_excel(file_path)
            else:
                # Default to CSV
                predictions.to_csv(file_path.with_suffix(".csv"))

            self._logger.info(
                "Predictions saved to file", output_file=str(file_path), count=len(predictions)
            )

        except Exception as e:
            self._logger.error(
                "Failed to save predictions to file", output_file=output_file, error=str(e)
            )

    async def _load_historical_data(
        self, symbol: str, start_date: datetime, end_date: datetime, data_source: str
    ) -> pd.DataFrame:
        """Load historical market data."""
        try:
            # This would integrate with your data loading system
            # For now, return empty DataFrame as placeholder
            self._logger.warning(
                "Historical data loading not implemented",
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                data_source=data_source,
            )

            return pd.DataFrame()

        except Exception as e:
            self._logger.error("Failed to load historical data", symbol=symbol, error=str(e))
            raise ValidationError(f"Historical data loading failed: {e}")

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics for the batch predictor."""
        avg_processing_time = (
            self.total_processing_time / self.prediction_count if self.prediction_count > 0 else 0
        )

        return {
            "total_predictions": self.prediction_count,
            "total_processing_time": self.total_processing_time,
            "average_processing_time_per_prediction": avg_processing_time,
            "configured_batch_size": self.bp_config.batch_size,
            "max_workers": self.bp_config.max_workers,
            "use_multiprocessing": self.bp_config.use_multiprocessing,
        }

    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.prediction_count = 0
        self.total_processing_time = 0.0

        self._logger.info("Performance statistics reset")

    # Service Health and Metrics
    async def _service_health_check(self) -> "HealthStatus":
        """Batch predictor service specific health check."""
        from src.core.base.interfaces import HealthStatus

        try:
            # Check dependencies
            if not all([self.data_service, self.model_registry, self.feature_engineering_service]):
                return HealthStatus.UNHEALTHY

            return HealthStatus.HEALTHY

        except Exception as e:
            self._logger.error("Batch predictor service health check failed", error=str(e))
            return HealthStatus.UNHEALTHY

    def get_batch_predictor_metrics(self) -> dict[str, Any]:
        """Get batch predictor service metrics."""
        return {
            "total_predictions": self.prediction_count,
            "total_processing_time": self.total_processing_time,
            "configured_batch_size": self.bp_config.batch_size,
            "max_workers": self.bp_config.max_workers,
            "max_memory_mb": self.bp_config.max_memory_mb,
            "use_multiprocessing": self.bp_config.use_multiprocessing,
            "save_to_database": self.bp_config.save_to_database,
            "enable_confidence_scores": self.bp_config.enable_confidence_scores,
        }

    # Configuration validation
    def _validate_service_config(self, config: ConfigDict) -> bool:
        """Validate batch predictor service configuration."""
        try:
            bp_config_dict = config.get("batch_predictor", {})
            BatchPredictorConfig(**bp_config_dict)
            return True
        except Exception as e:
            self._logger.error(
                "Batch predictor service configuration validation failed", error=str(e)
            )
            return False
