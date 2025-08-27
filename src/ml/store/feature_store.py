"""
Feature Store Service for ML Feature Storage and Management.

This module provides centralized storage and retrieval of ML features with
caching, versioning, and performance optimization using proper service patterns
without direct database access.
"""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.core.base.service import BaseService
from src.core.exceptions import ModelError, ValidationError
from src.core.types.base import ConfigDict
from src.core.types.data import FeatureSet
from src.utils.decorators import UnifiedDecorator

# Initialize decorator instance
dec = UnifiedDecorator()


class FeatureStoreConfig(BaseModel):
    """Configuration for feature store service."""

    enable_caching: bool = Field(default=True, description="Enable feature caching")
    cache_ttl_hours: int = Field(default=12, description="Feature cache TTL in hours")
    enable_versioning: bool = Field(default=True, description="Enable feature versioning")
    max_versions_per_feature: int = Field(
        default=5, description="Maximum versions to keep per feature"
    )
    enable_compression: bool = Field(default=True, description="Enable feature compression")
    batch_size: int = Field(default=1000, description="Batch size for bulk operations")
    background_cleanup_interval: int = Field(
        default=7200, description="Background cleanup interval in seconds"
    )
    enable_statistics: bool = Field(
        default=True, description="Enable feature statistics computation"
    )
    feature_validation_enabled: bool = Field(default=True, description="Enable feature validation")
    max_concurrent_operations: int = Field(default=20, description="Maximum concurrent operations")


class FeatureStoreMetadata(BaseModel):
    """Feature store metadata structure."""

    feature_set_id: str
    symbol: str
    feature_names: list[str]
    feature_count: int
    data_points: int
    creation_timestamp: datetime
    last_accessed: datetime | None = None
    version: str = "1.0.0"
    tags: dict[str, str] = Field(default_factory=dict)
    statistics: dict[str, Any] = Field(default_factory=dict)
    data_hash: str = ""
    storage_format: str = "json"
    compressed: bool = False
    expires_at: datetime | None = None


class FeatureStoreRequest(BaseModel):
    """Request for feature store operations."""

    operation: str  # store, retrieve, list, delete
    symbol: str
    feature_set: FeatureSet | None = None
    feature_set_id: str | None = None
    version: str | None = None
    include_statistics: bool = False
    compress: bool = True
    tags: dict[str, str] = Field(default_factory=dict)


class FeatureStoreResponse(BaseModel):
    """Response from feature store operations."""

    success: bool
    feature_set: FeatureSet | None = None
    metadata: FeatureStoreMetadata | None = None
    feature_sets: list[FeatureStoreMetadata] | None = None
    processing_time_ms: float
    operation: str
    error: str | None = None
    cache_hit: bool = False


class FeatureStoreService(BaseService):
    """
    Feature store service for centralized ML feature management.

    This service provides centralized storage, retrieval, and management of ML features
    with caching, versioning, and optimization capabilities using proper service patterns
    without direct database access.

    All data operations go through DataService dependency.
    """

    def __init__(
        self,
        config: ConfigDict | None = None,
        correlation_id: str | None = None,
    ):
        """
        Initialize the feature store service.

        Args:
            config: Service configuration
            correlation_id: Request correlation ID
        """
        super().__init__(
            name="FeatureStoreService",
            config=config,
            correlation_id=correlation_id,
        )

        # Parse feature store configuration
        fs_config_dict = (config or {}).get("feature_store", {})
        self.fs_config = FeatureStoreConfig(**fs_config_dict)

        # Service dependencies - resolved during startup
        self.data_service: Any = None

        # Internal state
        self._feature_cache: dict[str, tuple[FeatureSet, FeatureStoreMetadata, datetime]] = {}
        self._metadata_cache: dict[str, tuple[FeatureStoreMetadata, datetime]] = {}

        # Thread pool for CPU-intensive operations
        self._executor = ThreadPoolExecutor(max_workers=4)

        # Background cleanup task
        self._cleanup_task: asyncio.Task | None = None

        # Operation semaphore
        self._operation_semaphore = asyncio.Semaphore(self.fs_config.max_concurrent_operations)

        # Add required dependencies
        self.add_dependency("DataService")

    async def _do_start(self) -> None:
        """Start the feature store service."""
        await super()._do_start()

        # Resolve dependencies
        self.data_service = self.resolve_dependency("DataService")

        # Start background cleanup task
        if self.fs_config.background_cleanup_interval > 0:
            self._cleanup_task = asyncio.create_task(self._background_cleanup())

        self._logger.info(
            "Feature store service started successfully",
            config=self.fs_config.dict(),
            cached_features=len(self._feature_cache),
        )

    async def _do_stop(self) -> None:
        """Stop the feature store service."""
        # Cancel background tasks
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Shutdown thread pool
        self._executor.shutdown(wait=True)

        await super()._do_stop()

    # Core Feature Store Operations
    @dec.enhance(log=True, monitor=True, log_level="info")
    async def store_features(
        self,
        symbol: str,
        feature_set: FeatureSet,
        compress: bool = True,
        tags: dict[str, str] | None = None,
        version: str | None = None,
    ) -> FeatureStoreResponse:
        """
        Store features in the feature store.

        Args:
            symbol: Trading symbol
            feature_set: Feature set to store
            compress: Whether to compress the features
            tags: Optional tags for the feature set
            version: Optional version string

        Returns:
            Feature store response

        Raises:
            ModelError: If feature storage fails
        """
        request = FeatureStoreRequest(
            operation="store",
            symbol=symbol,
            feature_set=feature_set,
            compress=compress,
            tags=tags or {},
        )

        return await self.execute_with_monitoring(
            "store_features",
            self._store_features_impl,
            request,
            version,
        )

    async def _store_features_impl(
        self, request: FeatureStoreRequest, version: str | None
    ) -> FeatureStoreResponse:
        """Internal feature storage implementation."""
        async with self._operation_semaphore:
            start_time = datetime.now(timezone.utc)

            try:
                if not request.feature_set:
                    raise ValidationError("Feature set is required for store operation")

                # Validate features
                if self.fs_config.feature_validation_enabled:
                    validation_result = await self._validate_feature_set(request.feature_set)
                    if not validation_result["valid"]:
                        raise ValidationError(
                            f"Feature validation failed: {validation_result['error']}"
                        )

                # Generate version if not provided
                if not version:
                    version = await self._generate_version(
                        request.symbol, request.feature_set.feature_set_id
                    )

                # Compute statistics if enabled
                statistics = {}
                if self.fs_config.enable_statistics:
                    statistics = await self._compute_feature_statistics(request.feature_set)

                # Create metadata
                metadata = FeatureStoreMetadata(
                    feature_set_id=request.feature_set.feature_set_id,
                    symbol=request.symbol,
                    feature_names=request.feature_set.feature_names,
                    feature_count=len(request.feature_set.feature_names),
                    data_points=len(request.feature_set.features),
                    creation_timestamp=datetime.now(timezone.utc),
                    version=version,
                    tags=request.tags,
                    statistics=statistics,
                    storage_format="json",
                    compressed=request.compress,
                    expires_at=datetime.now(timezone.utc)
                    + timedelta(hours=self.fs_config.cache_ttl_hours),
                )

                # Generate data hash
                metadata.data_hash = await self._generate_data_hash(request.feature_set)

                # Prepare feature data for storage
                feature_data = await self._prepare_feature_data(
                    request.feature_set, request.compress
                )

                # Store in data service
                await self.data_service.store_feature_set(
                    feature_set_id=request.feature_set.feature_set_id,
                    symbol=request.symbol,
                    version=version,
                    feature_data=feature_data,
                    metadata=metadata.dict(),
                )

                # Cache the features and metadata
                if self.fs_config.enable_caching:
                    await self._cache_features(request.feature_set, metadata)

                # Clean up old versions if enabled
                if self.fs_config.enable_versioning and self.fs_config.max_versions_per_feature > 0:
                    await self._cleanup_old_versions(
                        request.symbol, request.feature_set.feature_set_id
                    )

                processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

                self._logger.info(
                    "Features stored successfully",
                    symbol=request.symbol,
                    feature_set_id=request.feature_set.feature_set_id,
                    version=version,
                    feature_count=len(request.feature_set.feature_names),
                    data_points=len(request.feature_set.features),
                    compressed=request.compress,
                    processing_time_ms=processing_time,
                )

                return FeatureStoreResponse(
                    success=True,
                    metadata=metadata,
                    processing_time_ms=processing_time,
                    operation="store",
                )

            except Exception as e:
                processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

                self._logger.error(
                    "Feature storage failed",
                    symbol=request.symbol,
                    feature_set_id=(
                        request.feature_set.feature_set_id if request.feature_set else None
                    ),
                    error=str(e),
                )

                return FeatureStoreResponse(
                    success=False,
                    processing_time_ms=processing_time,
                    operation="store",
                    error=str(e),
                )

    @dec.enhance(log=True, monitor=True, log_level="info")
    async def retrieve_features(
        self,
        symbol: str,
        feature_set_id: str | None = None,
        version: str | None = None,
        include_statistics: bool = False,
    ) -> FeatureStoreResponse:
        """
        Retrieve features from the feature store.

        Args:
            symbol: Trading symbol
            feature_set_id: Feature set ID to retrieve
            version: Specific version to retrieve
            include_statistics: Whether to include feature statistics

        Returns:
            Feature store response with features

        Raises:
            ModelError: If feature retrieval fails
        """
        request = FeatureStoreRequest(
            operation="retrieve",
            symbol=symbol,
            feature_set_id=feature_set_id,
            version=version,
            include_statistics=include_statistics,
        )

        return await self.execute_with_monitoring(
            "retrieve_features",
            self._retrieve_features_impl,
            request,
        )

    async def _retrieve_features_impl(self, request: FeatureStoreRequest) -> FeatureStoreResponse:
        """Internal feature retrieval implementation."""
        async with self._operation_semaphore:
            start_time = datetime.now(timezone.utc)
            cache_hit = False

            try:
                # Check cache first
                if self.fs_config.enable_caching:
                    cache_key = self._generate_cache_key(
                        request.symbol, request.feature_set_id, request.version
                    )
                    cached_result = await self._get_cached_features(cache_key)
                    if cached_result:
                        feature_set, metadata = cached_result
                        cache_hit = True

                        processing_time = (
                            datetime.now(timezone.utc) - start_time
                        ).total_seconds() * 1000

                        return FeatureStoreResponse(
                            success=True,
                            feature_set=feature_set,
                            metadata=metadata,
                            processing_time_ms=processing_time,
                            operation="retrieve",
                            cache_hit=True,
                        )

                # Retrieve from data service
                stored_data = await self.data_service.get_feature_set(
                    symbol=request.symbol,
                    feature_set_id=request.feature_set_id,
                    version=request.version,
                )

                if not stored_data:
                    raise ModelError(
                        f"Feature set not found: symbol={request.symbol}, "
                        f"feature_set_id={request.feature_set_id}, version={request.version}"
                    )

                # Parse stored data
                metadata = FeatureStoreMetadata(**stored_data["metadata"])
                feature_data = stored_data["feature_data"]

                # Reconstruct feature set
                feature_set = await self._reconstruct_feature_set(feature_data, metadata)

                # Update last accessed timestamp
                metadata.last_accessed = datetime.now(timezone.utc)
                await self.data_service.update_feature_set_metadata(
                    feature_set_id=metadata.feature_set_id,
                    symbol=request.symbol,
                    version=metadata.version,
                    updates={"last_accessed": metadata.last_accessed.isoformat()},
                )

                # Cache the result
                if self.fs_config.enable_caching:
                    await self._cache_features(feature_set, metadata)

                processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

                self._logger.info(
                    "Features retrieved successfully",
                    symbol=request.symbol,
                    feature_set_id=metadata.feature_set_id,
                    version=metadata.version,
                    feature_count=metadata.feature_count,
                    data_points=metadata.data_points,
                    cache_hit=cache_hit,
                    processing_time_ms=processing_time,
                )

                return FeatureStoreResponse(
                    success=True,
                    feature_set=feature_set,
                    metadata=metadata,
                    processing_time_ms=processing_time,
                    operation="retrieve",
                    cache_hit=cache_hit,
                )

            except Exception as e:
                processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

                self._logger.error(
                    "Feature retrieval failed",
                    symbol=request.symbol,
                    feature_set_id=request.feature_set_id,
                    version=request.version,
                    error=str(e),
                )

                return FeatureStoreResponse(
                    success=False,
                    processing_time_ms=processing_time,
                    operation="retrieve",
                    error=str(e),
                )

    @dec.enhance(log=True, monitor=True, log_level="info")
    async def list_feature_sets(
        self,
        symbol: str | None = None,
        include_expired: bool = False,
        limit: int | None = None,
    ) -> FeatureStoreResponse:
        """
        List available feature sets.

        Args:
            symbol: Optional symbol filter
            include_expired: Whether to include expired feature sets
            limit: Optional limit on results

        Returns:
            Feature store response with feature set list
        """
        request = FeatureStoreRequest(
            operation="list",
            symbol=symbol or "",
        )

        return await self.execute_with_monitoring(
            "list_feature_sets",
            self._list_feature_sets_impl,
            request,
            include_expired,
            limit,
        )

    async def _list_feature_sets_impl(
        self, request: FeatureStoreRequest, include_expired: bool, limit: int | None
    ) -> FeatureStoreResponse:
        """Internal feature set listing implementation."""
        start_time = datetime.now(timezone.utc)

        try:
            # Get feature sets from data service
            feature_sets_data = await self.data_service.list_feature_sets(
                symbol=request.symbol if request.symbol else None,
                include_expired=include_expired,
                limit=limit,
            )

            # Convert to metadata objects
            feature_sets_metadata = []
            for fs_data in feature_sets_data:
                try:
                    metadata = FeatureStoreMetadata(**fs_data["metadata"])
                    feature_sets_metadata.append(metadata)
                except Exception as e:
                    self._logger.warning(f"Failed to parse feature set metadata: {e}")
                    continue

            # Sort by creation timestamp (newest first)
            feature_sets_metadata.sort(key=lambda x: x.creation_timestamp, reverse=True)

            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            self._logger.info(
                "Feature sets listed successfully",
                symbol=request.symbol,
                feature_sets_found=len(feature_sets_metadata),
                include_expired=include_expired,
                processing_time_ms=processing_time,
            )

            return FeatureStoreResponse(
                success=True,
                feature_sets=feature_sets_metadata,
                processing_time_ms=processing_time,
                operation="list",
            )

        except Exception as e:
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            self._logger.error("Feature set listing failed", symbol=request.symbol, error=str(e))

            return FeatureStoreResponse(
                success=False,
                processing_time_ms=processing_time,
                operation="list",
                error=str(e),
            )

    @dec.enhance(log=True, monitor=True, log_level="info")
    async def delete_features(
        self,
        symbol: str,
        feature_set_id: str,
        version: str | None = None,
        delete_all_versions: bool = False,
    ) -> FeatureStoreResponse:
        """
        Delete features from the feature store.

        Args:
            symbol: Trading symbol
            feature_set_id: Feature set ID to delete
            version: Specific version to delete
            delete_all_versions: Whether to delete all versions

        Returns:
            Feature store response
        """
        request = FeatureStoreRequest(
            operation="delete",
            symbol=symbol,
            feature_set_id=feature_set_id,
            version=version,
        )

        return await self.execute_with_monitoring(
            "delete_features",
            self._delete_features_impl,
            request,
            delete_all_versions,
        )

    async def _delete_features_impl(
        self, request: FeatureStoreRequest, delete_all_versions: bool
    ) -> FeatureStoreResponse:
        """Internal feature deletion implementation."""
        start_time = datetime.now(timezone.utc)

        try:
            deleted_count = await self.data_service.delete_feature_set(
                symbol=request.symbol,
                feature_set_id=request.feature_set_id,
                version=request.version,
                delete_all_versions=delete_all_versions,
            )

            # Remove from caches
            await self._remove_from_cache(request.symbol, request.feature_set_id, request.version)

            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            self._logger.info(
                "Features deleted successfully",
                symbol=request.symbol,
                feature_set_id=request.feature_set_id,
                version=request.version,
                delete_all_versions=delete_all_versions,
                deleted_count=deleted_count,
                processing_time_ms=processing_time,
            )

            return FeatureStoreResponse(
                success=True,
                processing_time_ms=processing_time,
                operation="delete",
            )

        except Exception as e:
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            self._logger.error(
                "Feature deletion failed",
                symbol=request.symbol,
                feature_set_id=request.feature_set_id,
                error=str(e),
            )

            return FeatureStoreResponse(
                success=False,
                processing_time_ms=processing_time,
                operation="delete",
                error=str(e),
            )

    # Helper Methods
    async def _validate_feature_set(self, feature_set: FeatureSet) -> dict[str, Any]:
        """Validate feature set."""
        try:
            # Basic validation
            if not feature_set.features:
                return {"valid": False, "error": "Feature set has no features"}

            if not feature_set.feature_names:
                return {"valid": False, "error": "Feature set has no feature names"}

            if len(feature_set.features) == 0:
                return {"valid": False, "error": "Feature set has no data points"}

            # Check feature consistency
            if isinstance(feature_set.features, list) and feature_set.features:
                first_row = feature_set.features[0]
                if isinstance(first_row, dict):
                    expected_features = set(first_row.keys())
                    actual_features = set(feature_set.feature_names)

                    if expected_features != actual_features:
                        return {
                            "valid": False,
                            "error": f"Feature names mismatch: expected {expected_features}, got {actual_features}",
                        }

            return {"valid": True}

        except Exception as e:
            return {"valid": False, "error": f"Validation error: {e}"}

    async def _compute_feature_statistics(self, feature_set: FeatureSet) -> dict[str, Any]:
        """Compute statistics for feature set."""
        try:
            # Convert to DataFrame for easier computation
            if isinstance(feature_set.features, list) and feature_set.features:
                df = pd.DataFrame(feature_set.features)
            else:
                return {}

            loop = asyncio.get_event_loop()
            stats = await loop.run_in_executor(self._executor, self._compute_stats_sync, df)

            return stats

        except Exception as e:
            self._logger.warning(f"Failed to compute feature statistics: {e}")
            return {}

    def _compute_stats_sync(self, df: pd.DataFrame) -> dict[str, Any]:
        """Synchronous statistics computation."""
        try:
            # Basic statistics
            numeric_columns = df.select_dtypes(include=[np.number]).columns

            stats = {
                "total_features": len(df.columns),
                "numeric_features": len(numeric_columns),
                "data_points": len(df),
                "missing_values": df.isnull().sum().to_dict(),
            }

            # Statistics for numeric features
            if len(numeric_columns) > 0:
                numeric_stats = df[numeric_columns].describe().to_dict()
                stats["numeric_statistics"] = numeric_stats

            # Memory usage
            stats["memory_usage_bytes"] = df.memory_usage(deep=True).sum()

            return stats

        except Exception as e:
            self._logger.warning(f"Statistics computation error: {e}")
            return {}

    async def _generate_version(self, symbol: str, feature_set_id: str) -> str:
        """Generate version string for feature set."""
        try:
            # Get existing versions
            existing_versions = await self.data_service.get_feature_set_versions(
                symbol, feature_set_id
            )

            if not existing_versions:
                return "1.0.0"

            # Parse latest version and increment
            versions = [v for v in existing_versions if not v.startswith("v")]
            if versions:
                latest = max(versions)
                parts = latest.split(".")
                major, minor, patch = map(int, parts)
                return f"{major}.{minor}.{patch + 1}"
            else:
                return "1.0.0"

        except Exception:
            # Fallback to timestamp-based version
            return f"1.0.{int(datetime.now(timezone.utc).timestamp())}"

    async def _generate_data_hash(self, feature_set: FeatureSet) -> str:
        """Generate hash for feature data."""
        import hashlib

        try:
            # Create a string representation of the data
            data_str = json.dumps(feature_set.features, sort_keys=True, default=str)
            return hashlib.md5(data_str.encode()).hexdigest()
        except Exception:
            return ""

    async def _prepare_feature_data(
        self, feature_set: FeatureSet, compress: bool
    ) -> dict[str, Any]:
        """Prepare feature data for storage."""
        try:
            feature_data = {
                "features": feature_set.features,
                "feature_names": feature_set.feature_names,
                "metadata": feature_set.metadata,
                "compressed": compress,
            }

            if compress and self.fs_config.enable_compression:
                # Simple compression by converting to more efficient formats
                if isinstance(feature_set.features, list):
                    df = pd.DataFrame(feature_set.features)

                    # Convert to more efficient data types
                    for col in df.columns:
                        if df[col].dtype == "object":
                            try:
                                df[col] = pd.to_numeric(df[col], downcast="float")
                            except (ValueError, TypeError):
                                pass

                    feature_data["features"] = df.to_dict("records")
                    feature_data["compression_applied"] = True

            return feature_data

        except Exception as e:
            self._logger.warning(f"Failed to prepare feature data: {e}")
            return {
                "features": feature_set.features,
                "feature_names": feature_set.feature_names,
                "metadata": feature_set.metadata,
                "compressed": False,
            }

    async def _reconstruct_feature_set(
        self, feature_data: dict[str, Any], metadata: FeatureStoreMetadata
    ) -> FeatureSet:
        """Reconstruct feature set from stored data."""
        try:
            return FeatureSet(
                feature_set_id=metadata.feature_set_id,
                symbol=metadata.symbol,
                features=feature_data["features"],
                feature_names=feature_data["feature_names"],
                computation_time_ms=0.0,  # Not applicable for stored features
                metadata=feature_data.get("metadata", {}),
            )

        except Exception as e:
            self._logger.error(f"Failed to reconstruct feature set: {e}")
            raise ModelError(f"Feature set reconstruction failed: {e}")

    # Caching Operations
    def _generate_cache_key(
        self, symbol: str, feature_set_id: str | None, version: str | None
    ) -> str:
        """Generate cache key."""
        return f"{symbol}_{feature_set_id or 'latest'}_{version or 'latest'}"

    async def _cache_features(
        self, feature_set: FeatureSet, metadata: FeatureStoreMetadata
    ) -> None:
        """Cache features and metadata."""
        if not self.fs_config.enable_caching:
            return

        cache_key = self._generate_cache_key(
            metadata.symbol, metadata.feature_set_id, metadata.version
        )
        self._feature_cache[cache_key] = (feature_set, metadata, datetime.now(timezone.utc))

        # Also cache metadata separately
        metadata_key = f"meta_{cache_key}"
        self._metadata_cache[metadata_key] = (metadata, datetime.now(timezone.utc))

    async def _get_cached_features(
        self, cache_key: str
    ) -> tuple[FeatureSet, FeatureStoreMetadata] | None:
        """Get cached features."""
        if not self.fs_config.enable_caching or cache_key not in self._feature_cache:
            return None

        feature_set, metadata, timestamp = self._feature_cache[cache_key]
        ttl_hours = self.fs_config.cache_ttl_hours

        if datetime.now(timezone.utc) - timestamp < timedelta(hours=ttl_hours):
            return feature_set, metadata
        else:
            # Remove expired entry
            del self._feature_cache[cache_key]
            return None

    async def _remove_from_cache(
        self, symbol: str, feature_set_id: str, version: str | None
    ) -> None:
        """Remove features from cache."""
        cache_key = self._generate_cache_key(symbol, feature_set_id, version)

        if cache_key in self._feature_cache:
            del self._feature_cache[cache_key]

        metadata_key = f"meta_{cache_key}"
        if metadata_key in self._metadata_cache:
            del self._metadata_cache[metadata_key]

    async def _cleanup_old_versions(self, symbol: str, feature_set_id: str) -> None:
        """Clean up old feature set versions."""
        try:
            if self.fs_config.max_versions_per_feature <= 0:
                return

            # Get all versions for this feature set
            versions = await self.data_service.get_feature_set_versions(symbol, feature_set_id)

            if len(versions) <= self.fs_config.max_versions_per_feature:
                return

            # Sort versions (assuming semantic versioning)
            try:
                sorted_versions = sorted(
                    versions, key=lambda x: tuple(map(int, x.split("."))), reverse=True
                )
            except ValueError:
                # Fallback to string sorting if not semantic versioning
                sorted_versions = sorted(versions, reverse=True)

            # Keep only the most recent versions
            versions_to_delete = sorted_versions[self.fs_config.max_versions_per_feature :]

            for version in versions_to_delete:
                try:
                    await self.data_service.delete_feature_set(
                        symbol=symbol,
                        feature_set_id=feature_set_id,
                        version=version,
                        delete_all_versions=False,
                    )

                    # Remove from cache
                    await self._remove_from_cache(symbol, feature_set_id, version)

                except Exception as e:
                    self._logger.warning(f"Failed to delete old feature set version {version}: {e}")

        except Exception as e:
            self._logger.error(f"Cleanup of old versions failed: {e}")

    # Background Tasks
    async def _background_cleanup(self) -> None:
        """Background task for cleanup and maintenance."""
        while True:
            try:
                await asyncio.sleep(self.fs_config.background_cleanup_interval)

                # Clean expired cache entries
                await self._clean_expired_cache()

                # Clean expired feature sets
                await self._clean_expired_feature_sets()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Background cleanup error: {e}")

    async def _clean_expired_cache(self) -> None:
        """Clean expired cache entries."""
        ttl_hours = self.fs_config.cache_ttl_hours
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=ttl_hours)

        # Clean feature cache
        expired_feature_keys = [
            key for key, (_, _, timestamp) in self._feature_cache.items() if timestamp < cutoff_time
        ]

        for key in expired_feature_keys:
            del self._feature_cache[key]

        # Clean metadata cache
        expired_metadata_keys = [
            key for key, (_, timestamp) in self._metadata_cache.items() if timestamp < cutoff_time
        ]

        for key in expired_metadata_keys:
            del self._metadata_cache[key]

        if expired_feature_keys or expired_metadata_keys:
            self._logger.debug(
                f"Cleaned expired cache entries: {len(expired_feature_keys)} features, {len(expired_metadata_keys)} metadata"
            )

    async def _clean_expired_feature_sets(self) -> None:
        """Clean expired feature sets from storage."""
        try:
            expired_count = await self.data_service.delete_expired_feature_sets()

            if expired_count > 0:
                self._logger.info(f"Cleaned {expired_count} expired feature sets from storage")

        except Exception as e:
            self._logger.error(f"Failed to clean expired feature sets: {e}")

    # Service Health and Metrics
    async def _service_health_check(self) -> Any:
        """Feature store service specific health check."""
        from src.core.base.interfaces import HealthStatus

        try:
            # Check dependencies
            if not self.data_service:
                return HealthStatus.UNHEALTHY

            # Check cache sizes
            if len(self._feature_cache) > 10000 or len(self._metadata_cache) > 20000:
                return HealthStatus.DEGRADED

            return HealthStatus.HEALTHY

        except Exception as e:
            self._logger.error("Feature store service health check failed", error=str(e))
            return HealthStatus.UNHEALTHY

    def get_feature_store_metrics(self) -> dict[str, Any]:
        """Get feature store service metrics."""
        return {
            "cached_features": len(self._feature_cache),
            "cached_metadata": len(self._metadata_cache),
            "caching_enabled": self.fs_config.enable_caching,
            "versioning_enabled": self.fs_config.enable_versioning,
            "compression_enabled": self.fs_config.enable_compression,
            "statistics_enabled": self.fs_config.enable_statistics,
            "max_versions_per_feature": self.fs_config.max_versions_per_feature,
            "cache_ttl_hours": self.fs_config.cache_ttl_hours,
        }

    async def clear_cache(self) -> dict[str, int]:
        """Clear feature store caches."""
        feature_cache_size = len(self._feature_cache)
        metadata_cache_size = len(self._metadata_cache)

        self._feature_cache.clear()
        self._metadata_cache.clear()

        self._logger.info(
            "Feature store caches cleared",
            features_removed=feature_cache_size,
            metadata_removed=metadata_cache_size,
        )

        return {
            "features_removed": feature_cache_size,
            "metadata_removed": metadata_cache_size,
        }

    # Configuration validation
    def _validate_service_config(self, config: ConfigDict) -> bool:
        """Validate feature store service configuration."""
        try:
            fs_config_dict = config.get("feature_store", {})
            FeatureStoreConfig(**fs_config_dict)
            return True
        except Exception as e:
            self._logger.error(
                "Feature store service configuration validation failed", error=str(e)
            )
            return False
