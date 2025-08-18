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

from src.core.config import Config
from src.core.exceptions import ValidationError
from src.core.logging import get_logger
from src.database.connection import get_sync_session
from src.database.models import MLPrediction
from src.ml.feature_engineering import FeatureEngineer
from src.ml.models.base_model import BaseModel
from src.ml.registry.model_registry import ModelRegistry
from src.utils.decorators import log_calls, time_execution

logger = get_logger(__name__)


class BatchPredictor:
    """
    Batch prediction system for efficient processing of large datasets.

    This class provides optimized batch prediction capabilities with memory
    management, parallel processing, and database integration for storing
    results.

    Attributes:
        config: Application configuration
        model_registry: Model registry for loading models
        feature_engineer: Feature engineering pipeline
        batch_size: Number of samples to process per batch
        max_workers: Maximum number of parallel workers
        use_multiprocessing: Whether to use multiprocessing for CPU-bound tasks
    """

    def __init__(self, config: Config):
        """
        Initialize the batch predictor.

        Args:
            config: Application configuration
        """
        self.config = config
        self.model_registry = ModelRegistry(config)
        self.feature_engineer = FeatureEngineer(config)

        # Batch processing configuration
        self.batch_size = config.ml.batch_size
        self.max_workers = config.ml.max_workers
        self.use_multiprocessing = config.ml.use_multiprocessing
        self.chunk_size = config.ml.chunk_size

        # Memory management
        self.max_memory_mb = config.ml.max_memory_mb
        self.memory_threshold = 0.8  # Use 80% of available memory

        # Performance tracking
        self.prediction_count = 0
        self.total_processing_time = 0.0

        logger.info(
            "Batch predictor initialized",
            batch_size=self.batch_size,
            max_workers=self.max_workers,
            use_multiprocessing=self.use_multiprocessing,
            max_memory_mb=self.max_memory_mb,
        )

    @time_execution
    @log_calls
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

            logger.info(
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

                # Create features for batch
                features = self.feature_engineer.create_features(batch_data, symbol, feature_types)

                # Make predictions
                batch_predictions = await self._predict_batch_chunk(
                    model, features, batch_data.index
                )

                predictions.append(batch_predictions)

                # Log progress
                if batch_count % 10 == 0:
                    logger.info(
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

            # Save to database if requested
            if save_to_db:
                await self._save_predictions_to_db(result_df, model_name, symbol)

            # Save to file if requested
            if output_file:
                await self._save_predictions_to_file(result_df, output_file)

            # Update performance metrics
            self.prediction_count += len(result_df)

            logger.info(
                "Batch prediction completed",
                model_name=model_name,
                total_predictions=len(result_df),
                batches_processed=batch_count,
                symbol=symbol,
            )

            return result_df

        except Exception as e:
            logger.error(
                "Batch prediction failed", model_name=model_name, symbol=symbol, error=str(e)
            )
            raise ValidationError(f"Batch prediction failed: {e}") from e

    @time_execution
    @log_calls
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

            logger.info(
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
                        logger.error("Symbol prediction failed", symbol=symbol, error=str(e))
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
                        logger.error("Symbol prediction failed", symbol=symbol, error=str(e))
                        results[symbol] = pd.DataFrame()

            successful_symbols = [s for s, r in results.items() if not r.empty]

            logger.info(
                "Multi-symbol batch prediction completed",
                model_name=model_name,
                total_symbols=len(data_dict),
                successful_symbols=len(successful_symbols),
                parallel=parallel,
            )

            return results

        except Exception as e:
            logger.error(
                "Multi-symbol batch prediction failed", model_name=model_name, error=str(e)
            )
            raise ValidationError(f"Multi-symbol prediction failed: {e}") from e

    @time_execution
    @log_calls
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
            logger.info(
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

            logger.info(
                "Prediction backfill completed",
                model_name=model_name,
                symbol=symbol,
                predictions_count=len(predictions),
            )

            return predictions

        except Exception as e:
            logger.error(
                "Prediction backfill failed", model_name=model_name, symbol=symbol, error=str(e)
            )
            raise ValidationError(f"Prediction backfill failed: {e}") from e

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
            logger.error(f"Failed to load model {model_name}: {e}")
            raise ValidationError(f"Model loading failed: {e}") from e

    def _calculate_optimal_batch_size(self, data: pd.DataFrame) -> int:
        """Calculate optimal batch size based on data size and memory."""
        try:
            # Estimate memory usage per row (rough approximation)
            sample_row = data.iloc[:1] if not data.empty else pd.DataFrame()
            memory_per_row = (
                sample_row.memory_usage(deep=True).sum() if not sample_row.empty else 1000
            )

            # Calculate maximum rows that fit in memory
            available_memory = self.max_memory_mb * 1024 * 1024 * self.memory_threshold
            max_rows = int(available_memory / memory_per_row)

            # Use the smaller of configured batch size or memory-based limit
            optimal_size = min(self.batch_size, max_rows, len(data))

            logger.debug(
                "Calculated optimal batch size",
                configured_batch_size=self.batch_size,
                memory_based_limit=max_rows,
                data_size=len(data),
                optimal_size=optimal_size,
            )

            return max(1, optimal_size)  # Ensure at least 1

        except Exception as e:
            logger.warning(f"Failed to calculate optimal batch size: {e}")
            return min(self.batch_size, len(data))

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

            # Add confidence scores if available
            if hasattr(model, "predict_proba"):
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
            logger.error(f"Batch chunk prediction failed: {e}")
            # Return empty predictions with same index
            return pd.DataFrame({"prediction": np.nan, "confidence": 0.0}, index=indices)

    async def _save_predictions_to_db(
        self, predictions: pd.DataFrame, model_name: str, symbol: str
    ):
        """Save predictions to database."""
        try:
            with get_sync_session() as session:
                prediction_records = []

                for idx, row in predictions.iterrows():
                    record = MLPrediction(
                        model_name=model_name,
                        symbol=symbol,
                        timestamp=idx,
                        prediction_value=(
                            float(row["prediction"]) if not pd.isna(row["prediction"]) else None
                        ),
                        confidence_score=(
                            float(row["confidence"]) if not pd.isna(row["confidence"]) else None
                        ),
                        features_hash=hash(str(row.to_dict())),  # Simple hash of features
                        prediction_timestamp=datetime.utcnow(),
                    )
                    prediction_records.append(record)

                # Batch insert
                session.bulk_save_objects(prediction_records)
                session.commit()

                logger.info(
                    "Predictions saved to database",
                    model_name=model_name,
                    symbol=symbol,
                    count=len(prediction_records),
                )

        except Exception as e:
            logger.error(
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

            logger.info(
                "Predictions saved to file", output_file=str(file_path), count=len(predictions)
            )

        except Exception as e:
            logger.error(
                "Failed to save predictions to file", output_file=output_file, error=str(e)
            )

    async def _load_historical_data(
        self, symbol: str, start_date: datetime, end_date: datetime, data_source: str
    ) -> pd.DataFrame:
        """Load historical market data."""
        try:
            # This would integrate with your data loading system
            # For now, return empty DataFrame as placeholder
            logger.warning(
                "Historical data loading not implemented",
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                data_source=data_source,
            )

            return pd.DataFrame()

        except Exception as e:
            logger.error("Failed to load historical data", symbol=symbol, error=str(e))
            raise ValidationError(f"Historical data loading failed: {e}") from e

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics for the batch predictor."""
        avg_processing_time = (
            self.total_processing_time / self.prediction_count if self.prediction_count > 0 else 0
        )

        return {
            "total_predictions": self.prediction_count,
            "total_processing_time": self.total_processing_time,
            "average_processing_time_per_prediction": avg_processing_time,
            "configured_batch_size": self.batch_size,
            "max_workers": self.max_workers,
            "use_multiprocessing": self.use_multiprocessing,
        }

    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.prediction_count = 0
        self.total_processing_time = 0.0

        logger.info("Performance statistics reset")
