"""
Data Validation Pipeline - Comprehensive Data Quality Assurance

This module implements a comprehensive data validation pipeline that integrates
all validation components into a unified workflow for ensuring data quality
across the entire trading system.

Dependencies:
- P-001: Core types, exceptions, logging
- P-002A: Error handling framework
- P-007A: Utility functions and decorators
- DataService components for validation orchestration
"""

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from src.base import BaseComponent
from src.core.config import Config
from src.core.types import MarketData
from src.data.pipeline.data_pipeline import DataPipeline, PipelineConfig
from src.data.validation.data_validator import (
    DataValidator,
    MarketDataValidationResult,
    ValidationSeverity,
)
from src.utils.decorators import time_execution


class ValidationStage(Enum):
    """Validation pipeline stage enumeration."""

    INTAKE = "intake"
    SCHEMA_VALIDATION = "schema_validation"
    BUSINESS_VALIDATION = "business_validation"
    STATISTICAL_VALIDATION = "statistical_validation"
    TEMPORAL_VALIDATION = "temporal_validation"
    REGULATORY_VALIDATION = "regulatory_validation"
    QUALITY_SCORING = "quality_scoring"
    DISPOSITION = "disposition"
    COMPLETED = "completed"


class ValidationAction(Enum):
    """Validation action enumeration."""

    ACCEPT = "accept"
    ACCEPT_WITH_WARNING = "accept_with_warning"
    QUARANTINE = "quarantine"
    REJECT = "reject"
    RETRY = "retry"


@dataclass
class ValidationMetrics:
    """Validation pipeline metrics."""

    total_records: int = 0
    accepted_records: int = 0
    quarantined_records: int = 0
    rejected_records: int = 0
    processing_time_ms: int = 0
    validation_stages_completed: int = 0
    critical_issues: int = 0
    error_issues: int = 0
    warning_issues: int = 0
    average_quality_score: float = 0.0


class ValidationPipelineConfig(BaseModel):
    """Validation pipeline configuration."""

    enable_schema_validation: bool = True
    enable_business_validation: bool = True
    enable_statistical_validation: bool = True
    enable_temporal_validation: bool = True
    enable_regulatory_validation: bool = True

    # Quality thresholds
    min_quality_score: float = Field(0.7, ge=0.0, le=1.0)
    quarantine_quality_threshold: float = Field(0.5, ge=0.0, le=1.0)

    # Error thresholds
    max_critical_issues: int = Field(0, ge=0)
    max_error_issues: int = Field(3, ge=0)
    max_warning_issues: int = Field(10, ge=0)

    # Processing settings
    batch_size: int = Field(100, ge=1, le=10000)
    max_workers: int = Field(4, ge=1, le=16)
    timeout_seconds: int = Field(60, ge=1, le=600)

    # Retry settings
    retry_quarantined: bool = True
    retry_attempts: int = Field(3, ge=1, le=10)
    retry_delay_seconds: int = Field(5, ge=1, le=60)


class ValidationDisposition(BaseModel):
    """Validation disposition result."""

    action: ValidationAction
    quality_score: float = Field(ge=0.0, le=1.0)
    critical_issues: int = Field(ge=0)
    error_issues: int = Field(ge=0)
    warning_issues: int = Field(ge=0)
    reasons: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ValidationPipelineResult(BaseModel):
    """Comprehensive validation pipeline result."""

    pipeline_id: str
    total_records: int
    dispositions: dict[str, ValidationDisposition]  # symbol -> disposition
    metrics: ValidationMetrics
    execution_time_ms: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Config:
        arbitrary_types_allowed = True


class DataValidationPipeline(BaseComponent):
    """
    Comprehensive data validation pipeline orchestrator.

    This pipeline integrates all validation components to provide:
    - Multi-stage validation workflow
    - Intelligent data disposition (accept/quarantine/reject)
    - Quality scoring and metrics
    - Automated retry and recovery
    - Comprehensive monitoring and alerting
    """

    def __init__(self, config: Config):
        """Initialize the validation pipeline."""
        super().__init__()
        self.config = config

        # Load configuration
        self._setup_configuration()

        # Initialize components
        self.validator = DataValidator(config)
        self.data_pipeline: DataPipeline | None = None

        # Pipeline state
        self._active_validations: dict[str, dict[str, Any]] = {}
        self._quarantine_store: dict[str, list[MarketData]] = {}

        # Metrics
        self._session_metrics = ValidationMetrics()

        self._initialized = False

    def _setup_configuration(self) -> None:
        """Setup validation pipeline configuration."""
        pipeline_config = getattr(self.config, "validation_pipeline", {})

        self.pipeline_config = ValidationPipelineConfig(
            **pipeline_config if isinstance(pipeline_config, dict) else {}
        )

        self.logger.info(
            f"Validation pipeline configured with quality threshold: {self.pipeline_config.min_quality_score}"
        )

    async def initialize(self) -> None:
        """Initialize the validation pipeline."""
        try:
            if self._initialized:
                return

            self.logger.info("Initializing DataValidationPipeline...")

            # Initialize validator
            # DataValidator doesn't have initialize method, so skip

            # Create data pipeline for processing
            pipeline_config = PipelineConfig(
                name="validation_pipeline",
                batch_size=self.pipeline_config.batch_size,
                max_workers=self.pipeline_config.max_workers,
                timeout_seconds=self.pipeline_config.timeout_seconds,
                retry_attempts=self.pipeline_config.retry_attempts,
                enable_metrics=True,
            )

            self.data_pipeline = DataPipeline(pipeline_config)

            self._initialized = True
            self.logger.info("DataValidationPipeline initialized successfully")

        except Exception as e:
            self.logger.error(f"DataValidationPipeline initialization failed: {e}")
            raise

    @time_execution
    async def validate_batch(
        self, data: list[MarketData], symbols: list[str] | None = None
    ) -> ValidationPipelineResult:
        """
        Execute comprehensive validation pipeline on batch of data.

        Args:
            data: List of market data to validate
            symbols: Optional list of symbols to filter (validate all if None)

        Returns:
            ValidationPipelineResult: Comprehensive validation results
        """
        try:
            if not self._initialized:
                await self.initialize()

            pipeline_id = str(uuid.uuid4())
            start_time = datetime.now(timezone.utc)

            self.logger.info(f"Starting validation pipeline {pipeline_id} for {len(data)} records")

            # Filter data by symbols if specified
            if symbols:
                data = [d for d in data if d.symbol in symbols]

            if not data:
                return ValidationPipelineResult(
                    pipeline_id=pipeline_id,
                    total_records=0,
                    dispositions={},
                    metrics=ValidationMetrics(),
                    execution_time_ms=0,
                )

            # Track pipeline execution
            self._active_validations[pipeline_id] = {
                "stage": ValidationStage.INTAKE,
                "start_time": start_time,
                "total_records": len(data),
                "processed_records": 0,
            }

            # Execute validation stages
            dispositions = await self._execute_validation_stages(pipeline_id, data)

            # Calculate final metrics
            metrics = self._calculate_pipeline_metrics(dispositions, data)

            # Calculate execution time
            end_time = datetime.now(timezone.utc)
            execution_time_ms = int((end_time - start_time).total_seconds() * 1000)

            # Update session metrics
            self._update_session_metrics(metrics)

            # Clean up tracking
            if pipeline_id in self._active_validations:
                self._active_validations[pipeline_id]["stage"] = ValidationStage.COMPLETED
                self._active_validations[pipeline_id]["end_time"] = end_time

            result = ValidationPipelineResult(
                pipeline_id=pipeline_id,
                total_records=len(data),
                dispositions=dispositions,
                metrics=metrics,
                execution_time_ms=execution_time_ms,
            )

            self.logger.info(
                f"Validation pipeline {pipeline_id} completed - "
                f"Accepted: {metrics.accepted_records}, "
                f"Quarantined: {metrics.quarantined_records}, "
                f"Rejected: {metrics.rejected_records}"
            )

            return result

        except Exception as e:
            self.logger.error(f"Validation pipeline execution failed: {e}")

            # Mark pipeline as failed
            if pipeline_id in self._active_validations:
                self._active_validations[pipeline_id]["stage"] = "failed"
                self._active_validations[pipeline_id]["error"] = str(e)

            raise

    async def _execute_validation_stages(
        self, pipeline_id: str, data: list[MarketData]
    ) -> dict[str, ValidationDisposition]:
        """Execute all validation stages."""
        dispositions = {}

        # Group data by symbol for efficient processing
        symbol_groups = self._group_data_by_symbol(data)

        for symbol, symbol_data in symbol_groups.items():
            try:
                # Stage 1: Schema validation
                await self._update_pipeline_stage(pipeline_id, ValidationStage.SCHEMA_VALIDATION)
                validation_results = await self.validator.validate_market_data(
                    symbol_data, include_statistical=False
                )

                # Process validation results for each record
                if isinstance(validation_results, list):
                    # Aggregate results for the symbol
                    disposition = await self._determine_symbol_disposition(
                        symbol, validation_results
                    )
                else:
                    # Single result
                    disposition = await self._determine_record_disposition(validation_results)

                dispositions[symbol] = disposition

                # Update processed count
                self._active_validations[pipeline_id]["processed_records"] += len(symbol_data)

            except Exception as e:
                self.logger.error(f"Validation failed for symbol {symbol}: {e}")

                # Create error disposition
                dispositions[symbol] = ValidationDisposition(
                    action=ValidationAction.REJECT,
                    quality_score=0.0,
                    critical_issues=1,
                    error_issues=0,
                    warning_issues=0,
                    reasons=[f"Validation error: {e}"],
                    metadata={"validation_error": str(e)},
                )

        return dispositions

    def _group_data_by_symbol(self, data: list[MarketData]) -> dict[str, list[MarketData]]:
        """Group market data by symbol."""
        groups = {}
        for item in data:
            if item.symbol not in groups:
                groups[item.symbol] = []
            groups[item.symbol].append(item)
        return groups

    async def _update_pipeline_stage(self, pipeline_id: str, stage: ValidationStage) -> None:
        """Update pipeline stage."""
        if pipeline_id in self._active_validations:
            self._active_validations[pipeline_id]["stage"] = stage
            self.logger.debug(f"Pipeline {pipeline_id} moved to stage {stage.value}")

    async def _determine_symbol_disposition(
        self, symbol: str, validation_results: list[MarketDataValidationResult]
    ) -> ValidationDisposition:
        """Determine disposition for a symbol based on all its validation results."""
        if not validation_results:
            return ValidationDisposition(
                action=ValidationAction.REJECT,
                quality_score=0.0,
                critical_issues=1,
                error_issues=0,
                warning_issues=0,
                reasons=["No validation results"],
            )

        # Aggregate validation metrics
        total_records = len(validation_results)
        valid_records = sum(1 for r in validation_results if r.is_valid)

        # Calculate aggregate quality score
        quality_scores = [r.quality_score.overall for r in validation_results]
        avg_quality_score = sum(quality_scores) / len(quality_scores)

        # Count issues by severity
        critical_issues = sum(
            len([i for i in r.issues if i.severity == ValidationSeverity.CRITICAL])
            for r in validation_results
        )
        error_issues = sum(
            len([i for i in r.issues if i.severity == ValidationSeverity.ERROR])
            for r in validation_results
        )
        warning_issues = sum(
            len([i for i in r.issues if i.severity == ValidationSeverity.WARNING])
            for r in validation_results
        )

        # Determine action based on thresholds
        action = self._determine_action(
            avg_quality_score,
            critical_issues,
            error_issues,
            warning_issues,
            valid_records,
            total_records,
        )

        # Build reasons
        reasons = []
        if critical_issues > 0:
            reasons.append(f"{critical_issues} critical issues found")
        if error_issues > 0:
            reasons.append(f"{error_issues} error issues found")
        if avg_quality_score < self.pipeline_config.min_quality_score:
            reasons.append(
                f"Quality score {avg_quality_score:.2f} below threshold {self.pipeline_config.min_quality_score}"
            )
        if valid_records < total_records:
            reasons.append(f"Only {valid_records}/{total_records} records passed validation")

        return ValidationDisposition(
            action=action,
            quality_score=avg_quality_score,
            critical_issues=critical_issues,
            error_issues=error_issues,
            warning_issues=warning_issues,
            reasons=reasons,
            metadata={
                "total_records": total_records,
                "valid_records": valid_records,
                "validation_results_count": len(validation_results),
            },
        )

    async def _determine_record_disposition(
        self, validation_result: MarketDataValidationResult
    ) -> ValidationDisposition:
        """Determine disposition for a single validation result."""
        # Count issues by severity
        critical_issues = len(
            [i for i in validation_result.issues if i.severity == ValidationSeverity.CRITICAL]
        )
        error_issues = len(
            [i for i in validation_result.issues if i.severity == ValidationSeverity.ERROR]
        )
        warning_issues = len(
            [i for i in validation_result.issues if i.severity == ValidationSeverity.WARNING]
        )

        # Determine action
        action = self._determine_action(
            validation_result.quality_score.overall,
            critical_issues,
            error_issues,
            warning_issues,
            1 if validation_result.is_valid else 0,
            1,
        )

        # Build reasons
        reasons = []
        if not validation_result.is_valid:
            reasons.append("Failed validation")
        for issue in validation_result.issues:
            if issue.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]:
                reasons.append(f"{issue.severity.value}: {issue.message}")

        return ValidationDisposition(
            action=action,
            quality_score=validation_result.quality_score.overall,
            critical_issues=critical_issues,
            error_issues=error_issues,
            warning_issues=warning_issues,
            reasons=reasons,
            metadata={
                "validation_timestamp": validation_result.validation_timestamp.isoformat(),
                "is_valid": validation_result.is_valid,
            },
        )

    def _determine_action(
        self,
        quality_score: float,
        critical_issues: int,
        error_issues: int,
        warning_issues: int,
        valid_records: int,
        total_records: int,
    ) -> ValidationAction:
        """Determine validation action based on thresholds."""
        # Reject if critical issues or too many errors
        if critical_issues > self.pipeline_config.max_critical_issues:
            return ValidationAction.REJECT

        if error_issues > self.pipeline_config.max_error_issues:
            return ValidationAction.REJECT

        # Reject if quality score too low
        if quality_score < self.pipeline_config.quarantine_quality_threshold:
            return ValidationAction.REJECT

        # Quarantine if quality score below minimum but above quarantine threshold
        if quality_score < self.pipeline_config.min_quality_score:
            return ValidationAction.QUARANTINE

        # Quarantine if too many warnings
        if warning_issues > self.pipeline_config.max_warning_issues:
            return ValidationAction.QUARANTINE

        # Accept with warning if some issues but above thresholds
        if error_issues > 0 or warning_issues > 0:
            return ValidationAction.ACCEPT_WITH_WARNING

        # Accept if all good
        return ValidationAction.ACCEPT

    def _calculate_pipeline_metrics(
        self, dispositions: dict[str, ValidationDisposition], data: list[MarketData]
    ) -> ValidationMetrics:
        """Calculate pipeline metrics from dispositions."""
        metrics = ValidationMetrics()
        metrics.total_records = len(data)

        for disposition in dispositions.values():
            if disposition.action == ValidationAction.ACCEPT:
                metrics.accepted_records += disposition.metadata.get("total_records", 1)
            elif disposition.action == ValidationAction.ACCEPT_WITH_WARNING:
                metrics.accepted_records += disposition.metadata.get("total_records", 1)
            elif disposition.action == ValidationAction.QUARANTINE:
                metrics.quarantined_records += disposition.metadata.get("total_records", 1)
            elif disposition.action == ValidationAction.REJECT:
                metrics.rejected_records += disposition.metadata.get("total_records", 1)

            metrics.critical_issues += disposition.critical_issues
            metrics.error_issues += disposition.error_issues
            metrics.warning_issues += disposition.warning_issues

        # Calculate average quality score
        if dispositions:
            total_quality_score = sum(d.quality_score for d in dispositions.values())
            metrics.average_quality_score = total_quality_score / len(dispositions)

        return metrics

    def _update_session_metrics(self, pipeline_metrics: ValidationMetrics) -> None:
        """Update session-level metrics."""
        self._session_metrics.total_records += pipeline_metrics.total_records
        self._session_metrics.accepted_records += pipeline_metrics.accepted_records
        self._session_metrics.quarantined_records += pipeline_metrics.quarantined_records
        self._session_metrics.rejected_records += pipeline_metrics.rejected_records
        self._session_metrics.critical_issues += pipeline_metrics.critical_issues
        self._session_metrics.error_issues += pipeline_metrics.error_issues
        self._session_metrics.warning_issues += pipeline_metrics.warning_issues

    async def get_quarantined_data(self, symbol: str | None = None) -> dict[str, list[MarketData]]:
        """Get quarantined data for review."""
        if symbol:
            return {symbol: self._quarantine_store.get(symbol, [])}
        return self._quarantine_store.copy()

    async def retry_quarantined_data(self, symbol: str) -> ValidationPipelineResult | None:
        """Retry validation for quarantined data."""
        if symbol not in self._quarantine_store:
            return None

        quarantined_data = self._quarantine_store[symbol]
        if not quarantined_data:
            return None

        self.logger.info(
            f"Retrying validation for {len(quarantined_data)} quarantined records for {symbol}"
        )

        # Re-validate the data
        result = await self.validate_batch(quarantined_data, [symbol])

        # If successful, remove from quarantine
        disposition = result.dispositions.get(symbol)
        if disposition and disposition.action in [
            ValidationAction.ACCEPT,
            ValidationAction.ACCEPT_WITH_WARNING,
        ]:
            del self._quarantine_store[symbol]
            self.logger.info(f"Successfully recovered quarantined data for {symbol}")

        return result

    async def get_pipeline_status(self) -> dict[str, Any]:
        """Get current pipeline status."""
        return {
            "active_validations": len(self._active_validations),
            "quarantined_symbols": len(self._quarantine_store),
            "quarantined_records": sum(len(records) for records in self._quarantine_store.values()),
            "session_metrics": self._session_metrics,
            "configuration": self.pipeline_config.dict(),
        }

    async def health_check(self) -> dict[str, Any]:
        """Perform validation pipeline health check."""
        health = {
            "status": "healthy",
            "initialized": self._initialized,
            "validator_available": self.validator is not None,
            "data_pipeline_available": self.data_pipeline is not None,
            "active_validations": len(self._active_validations),
            "quarantine_status": {
                "symbols_quarantined": len(self._quarantine_store),
                "total_quarantined_records": sum(
                    len(records) for records in self._quarantine_store.values()
                ),
            },
        }

        # Check validator health
        try:
            validator_health = await self.validator.health_check()
            health["validator_status"] = validator_health["status"]
        except Exception as e:
            health["validator_status"] = f"unhealthy: {e}"
            health["status"] = "degraded"

        return health

    async def cleanup(self) -> None:
        """Cleanup validation pipeline resources."""
        try:
            # Clean up active validations
            self._active_validations.clear()

            # Clean up quarantine store
            self._quarantine_store.clear()

            # Reset session metrics
            self._session_metrics = ValidationMetrics()

            # Cleanup validator
            if hasattr(self.validator, "cleanup"):
                await self.validator.cleanup()

            # Cleanup data pipeline
            if self.data_pipeline:
                await self.data_pipeline.cleanup()

            self._initialized = False
            self.logger.info("DataValidationPipeline cleanup completed")

        except Exception as e:
            self.logger.error(f"DataValidationPipeline cleanup error: {e}")
