"""Data pipeline repositories implementation."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models.data import (
    DataPipelineRecord,
    DataQualityRecord,
    FeatureRecord,
)
from src.database.repository.base import DatabaseRepository


class FeatureRepository(DatabaseRepository):
    """Repository for FeatureRecord entities."""

    def __init__(self, session: AsyncSession):
        """Initialize feature repository."""

        super().__init__(
            session=session,
            model=FeatureRecord,
            entity_type=FeatureRecord,
            key_type=str,
            name="FeatureRepository",
        )

    async def get_by_symbol(self, symbol: str) -> list[FeatureRecord]:
        """Get features by symbol."""
        return await self.get_all(filters={"symbol": symbol}, order_by="-calculation_timestamp")

    async def get_by_feature_type(self, feature_type: str) -> list[FeatureRecord]:
        """Get features by type."""
        return await self.get_all(filters={"feature_type": feature_type}, order_by="-calculation_timestamp")

    async def get_by_symbol_and_type(self, symbol: str, feature_type: str) -> list[FeatureRecord]:
        """Get features by symbol and type."""
        return await self.get_all(
            filters={"symbol": symbol, "feature_type": feature_type},
            order_by="-calculation_timestamp",
        )

    async def get_latest_feature(self, symbol: str, feature_type: str, feature_name: str) -> FeatureRecord | None:
        """Get latest feature value."""
        features = await self.get_all(
            filters={"symbol": symbol, "feature_type": feature_type, "feature_name": feature_name},
            order_by="-calculation_timestamp",
            limit=1,
        )
        return features[0] if features else None

    async def get_features_by_date_range(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> list[FeatureRecord]:
        """Get features by date range."""
        return await self.get_all(
            filters={
                "symbol": symbol,
                "calculation_timestamp": {"gte": start_date, "lte": end_date},
            },
            order_by="-calculation_timestamp",
        )


class DataQualityRepository(DatabaseRepository):
    """Repository for DataQualityRecord entities."""

    def __init__(self, session: AsyncSession):
        """Initialize data quality repository."""

        super().__init__(
            session=session,
            model=DataQualityRecord,
            entity_type=DataQualityRecord,
            key_type=str,
            name="DataQualityRepository",
        )

    async def get_by_symbol(self, symbol: str) -> list[DataQualityRecord]:
        """Get quality records by symbol."""
        return await self.get_all(filters={"symbol": symbol}, order_by="-quality_check_timestamp")

    async def get_by_data_source(self, data_source: str) -> list[DataQualityRecord]:
        """Get quality records by data source."""
        return await self.get_all(filters={"data_source": data_source}, order_by="-quality_check_timestamp")

    async def get_poor_quality_records(self, threshold: Decimal = Decimal("0.8")) -> list[DataQualityRecord]:
        """Get records with poor quality scores."""
        records = await self.get_all()
        return [record for record in records if record.overall_score < threshold]

    async def get_latest_quality_check(self, symbol: str, data_source: str) -> DataQualityRecord | None:
        """Get latest quality check for symbol and source."""
        records = await self.get_all(
            filters={"symbol": symbol, "data_source": data_source},
            order_by="-quality_check_timestamp",
            limit=1,
        )
        return records[0] if records else None

    async def get_quality_trend(self, symbol: str, days: int = 30) -> list[DataQualityRecord]:
        """Get quality trend for symbol over specified days."""
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        return await self.get_all(
            filters={
                "symbol": symbol,
                "quality_check_timestamp": {"gte": start_date, "lte": end_date},
            },
            order_by="-quality_check_timestamp",
        )


class DataPipelineRepository(DatabaseRepository):
    """Repository for DataPipelineRecord entities."""

    def __init__(self, session: AsyncSession):
        """Initialize data pipeline repository."""

        super().__init__(
            session=session,
            model=DataPipelineRecord,
            entity_type=DataPipelineRecord,
            key_type=str,
            name="DataPipelineRepository",
        )

    async def get_by_pipeline_name(self, pipeline_name: str) -> list[DataPipelineRecord]:
        """Get records by pipeline name."""
        return await self.get_all(filters={"pipeline_name": pipeline_name}, order_by="-execution_timestamp")

    async def get_by_status(self, status: str) -> list[DataPipelineRecord]:
        """Get records by status."""
        return await self.get_all(filters={"status": status}, order_by="-execution_timestamp")

    async def get_running_pipelines(self) -> list[DataPipelineRecord]:
        """Get currently running pipelines."""
        return await self.get_all(filters={"status": "running"}, order_by="-execution_timestamp")

    async def get_failed_pipelines(self) -> list[DataPipelineRecord]:
        """Get failed pipeline executions."""
        return await self.get_all(filters={"status": "failed"}, order_by="-execution_timestamp")

    async def get_latest_execution(self, pipeline_name: str) -> DataPipelineRecord | None:
        """Get latest execution for a pipeline."""
        records = await self.get_all(filters={"pipeline_name": pipeline_name}, order_by="-execution_timestamp", limit=1)
        return records[0] if records else None

    async def get_pipeline_performance(self, pipeline_name: str, days: int = 30) -> list[DataPipelineRecord]:
        """Get pipeline performance over specified days."""
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        return await self.get_all(
            filters={
                "pipeline_name": pipeline_name,
                "execution_timestamp": {"gte": start_date, "lte": end_date},
            },
            order_by="-execution_timestamp",
        )
