"""
ML Repository - Data access layer for ML models.

This repository handles all database operations for ML-related models including
predictions, model metadata, and training jobs.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import and_, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models.ml import MLModelMetadata, MLPrediction, MLTrainingJob
from src.database.repository.base import BaseRepository


class MLPredictionRepository(BaseRepository):
    """Repository for ML predictions."""

    def __init__(self, session: AsyncSession):
        """
        Initialize ML prediction repository.

        Args:
            session: Database session
        """
        super().__init__(session, MLPrediction)

    async def get_by_model_and_symbol(
        self, model_name: str, symbol: str, limit: int = 100
    ) -> list[MLPrediction]:
        """
        Get predictions by model name and symbol.

        Args:
            model_name: Name of the ML model
            symbol: Trading symbol
            limit: Maximum number of predictions to return

        Returns:
            List of ML predictions
        """
        result = await self.session.execute(
            select(MLPrediction)
            .where(
                and_(
                    MLPrediction.model_name == model_name,
                    MLPrediction.symbol == symbol,
                )
            )
            .order_by(desc(MLPrediction.timestamp))
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_recent_predictions(
        self,
        model_name: str | None = None,
        hours: int = 24,
        min_confidence: float = 0.0,
    ) -> list[MLPrediction]:
        """
        Get recent predictions within specified hours.

        Args:
            model_name: Optional filter by model name
            hours: Number of hours to look back
            min_confidence: Minimum confidence score

        Returns:
            List of recent predictions
        """
        from datetime import timedelta

        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        query = select(MLPrediction).where(
            and_(
                MLPrediction.timestamp >= cutoff_time,
                MLPrediction.confidence_score >= min_confidence,
            )
        )

        if model_name:
            query = query.where(MLPrediction.model_name == model_name)

        query = query.order_by(desc(MLPrediction.timestamp))

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_prediction_accuracy(
        self, model_name: str, symbol: str | None = None, days: int = 30
    ) -> dict[str, Any]:
        """
        Calculate prediction accuracy metrics.

        Args:
            model_name: Name of the ML model
            symbol: Optional symbol filter
            days: Number of days to analyze

        Returns:
            Dictionary with accuracy metrics
        """
        from datetime import timedelta

        cutoff_time = datetime.utcnow() - timedelta(days=days)

        query = select(
            func.count(MLPrediction.id).label("total_predictions"),
            func.avg(MLPrediction.confidence_score).label("avg_confidence"),
            func.avg(MLPrediction.prediction_error).label("avg_error"),
            func.stddev(MLPrediction.prediction_error).label("error_stddev"),
        ).where(
            and_(
                MLPrediction.model_name == model_name,
                MLPrediction.timestamp >= cutoff_time,
                MLPrediction.actual_value.isnot(None),  # Only evaluated predictions
            )
        )

        if symbol:
            query = query.where(MLPrediction.symbol == symbol)

        result = await self.session.execute(query)
        row = result.first()

        if row:
            return {
                "total_predictions": row.total_predictions or 0,
                "avg_confidence": float(row.avg_confidence or 0),
                "avg_error": float(row.avg_error or 0),
                "error_stddev": float(row.error_stddev or 0),
            }

        return {
            "total_predictions": 0,
            "avg_confidence": 0.0,
            "avg_error": 0.0,
            "error_stddev": 0.0,
        }

    async def update_with_actual(
        self, prediction_id: int, actual_value: float
    ) -> MLPrediction | None:
        """
        Update prediction with actual value for evaluation.

        Args:
            prediction_id: ID of the prediction
            actual_value: Actual observed value

        Returns:
            Updated prediction or None if not found
        """
        prediction = await self.get_by_id(prediction_id)
        if prediction:
            prediction.actual_value = actual_value
            prediction.prediction_error = abs(prediction.prediction_value - actual_value)
            prediction.updated_at = datetime.utcnow()
            await self.session.commit()
            await self.session.refresh(prediction)
        return prediction


class MLModelMetadataRepository(BaseRepository):
    """Repository for ML model metadata."""

    def __init__(self, session: AsyncSession):
        """
        Initialize ML model metadata repository.

        Args:
            session: Database session
        """
        super().__init__(session, MLModelMetadata)

    async def get_latest_model(
        self, model_name: str, model_type: str
    ) -> MLModelMetadata | None:
        """
        Get the latest version of a model.

        Args:
            model_name: Name of the model
            model_type: Type of the model

        Returns:
            Latest model metadata or None
        """
        result = await self.session.execute(
            select(MLModelMetadata)
            .where(
                and_(
                    MLModelMetadata.model_name == model_name,
                    MLModelMetadata.model_type == model_type,
                    MLModelMetadata.is_active == True,
                )
            )
            .order_by(desc(MLModelMetadata.version))
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_active_models(self) -> list[MLModelMetadata]:
        """
        Get all active models.

        Returns:
            List of active model metadata
        """
        result = await self.session.execute(
            select(MLModelMetadata)
            .where(MLModelMetadata.is_active == True)
            .order_by(MLModelMetadata.model_name, desc(MLModelMetadata.version))
        )
        return list(result.scalars().all())

    async def deactivate_old_versions(self, model_name: str, keep_versions: int = 3) -> int:
        """
        Deactivate old versions of a model, keeping only recent ones.

        Args:
            model_name: Name of the model
            keep_versions: Number of recent versions to keep active

        Returns:
            Number of models deactivated
        """
        # Get versions to keep
        result = await self.session.execute(
            select(MLModelMetadata.id)
            .where(MLModelMetadata.model_name == model_name)
            .order_by(desc(MLModelMetadata.version))
            .limit(keep_versions)
        )
        keep_ids = [row[0] for row in result]

        # Deactivate older versions
        if keep_ids:
            result = await self.session.execute(
                select(MLModelMetadata).where(
                    and_(
                        MLModelMetadata.model_name == model_name,
                        MLModelMetadata.id.notin_(keep_ids),
                        MLModelMetadata.is_active == True,
                    )
                )
            )
            models_to_deactivate = result.scalars().all()

            for model in models_to_deactivate:
                model.is_active = False
                model.updated_at = datetime.utcnow()

            await self.session.commit()
            return len(models_to_deactivate)

        return 0


class MLTrainingJobRepository(BaseRepository):
    """Repository for ML training jobs."""

    def __init__(self, session: AsyncSession):
        """
        Initialize ML training job repository.

        Args:
            session: Database session
        """
        super().__init__(session, MLTrainingJob)

    async def get_running_jobs(self) -> list[MLTrainingJob]:
        """
        Get all currently running training jobs.

        Returns:
            List of running training jobs
        """
        result = await self.session.execute(
            select(MLTrainingJob)
            .where(MLTrainingJob.status == "running")
            .order_by(MLTrainingJob.started_at)
        )
        return list(result.scalars().all())

    async def get_job_by_model(
        self, model_name: str, status: str | None = None
    ) -> list[MLTrainingJob]:
        """
        Get training jobs for a specific model.

        Args:
            model_name: Name of the model
            status: Optional status filter

        Returns:
            List of training jobs
        """
        query = select(MLTrainingJob).where(MLTrainingJob.model_name == model_name)

        if status:
            query = query.where(MLTrainingJob.status == status)

        query = query.order_by(desc(MLTrainingJob.started_at))

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def update_job_status(
        self,
        job_id: int,
        status: str,
        metrics: dict[str, Any] | None = None,
        error_message: str | None = None,
    ) -> MLTrainingJob | None:
        """
        Update training job status and metrics.

        Args:
            job_id: ID of the training job
            status: New status
            metrics: Optional training metrics
            error_message: Optional error message

        Returns:
            Updated training job or None
        """
        job = await self.get_by_id(job_id)
        if job:
            job.status = status
            if metrics:
                job.training_metrics = metrics
            if error_message:
                job.error_message = error_message
            if status in ["completed", "failed"]:
                job.completed_at = datetime.utcnow()
            job.updated_at = datetime.utcnow()
            await self.session.commit()
            await self.session.refresh(job)
        return job

    async def get_successful_jobs(
        self, days: int = 30, limit: int = 100
    ) -> list[MLTrainingJob]:
        """
        Get successful training jobs within specified days.

        Args:
            days: Number of days to look back
            limit: Maximum number of jobs to return

        Returns:
            List of successful training jobs
        """
        from datetime import timedelta

        cutoff_time = datetime.utcnow() - timedelta(days=days)

        result = await self.session.execute(
            select(MLTrainingJob)
            .where(
                and_(
                    MLTrainingJob.status == "completed",
                    MLTrainingJob.completed_at >= cutoff_time,
                )
            )
            .order_by(desc(MLTrainingJob.completed_at))
            .limit(limit)
        )
        return list(result.scalars().all())


class MLRepository:
    """
    Unified ML repository combining all ML-related repositories.

    This provides a single interface for all ML database operations.
    """

    def __init__(self, session: AsyncSession):
        """
        Initialize unified ML repository.

        Args:
            session: Database session
        """
        self.session = session
        self.predictions = MLPredictionRepository(session)
        self.models = MLModelMetadataRepository(session)
        self.training_jobs = MLTrainingJobRepository(session)

    async def get_model_performance_summary(
        self, model_name: str, days: int = 30
    ) -> dict[str, Any]:
        """
        Get comprehensive performance summary for a model.

        Args:
            model_name: Name of the model
            days: Number of days to analyze

        Returns:
            Dictionary with performance summary
        """
        # Get prediction accuracy
        accuracy = await self.predictions.get_prediction_accuracy(model_name, days=days)

        # Get latest model metadata
        latest_model = await self.models.get_latest_model(model_name, "prediction")

        # Get recent training jobs
        recent_jobs = await self.training_jobs.get_job_by_model(model_name, status="completed")

        return {
            "model_name": model_name,
            "accuracy_metrics": accuracy,
            "latest_version": latest_model.version if latest_model else None,
            "model_parameters": latest_model.parameters if latest_model else {},
            "recent_training_jobs": len(recent_jobs),
            "last_training": recent_jobs[0].completed_at if recent_jobs else None,
        }
