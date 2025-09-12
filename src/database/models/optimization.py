
"""
Database models for optimization framework.

This module defines SQLAlchemy models for storing optimization results,
parameter sets, objectives, and related optimization data with proper
relationships and constraints for financial applications.
"""

import uuid
from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import (
    DECIMAL,
    JSON,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class OptimizationRun(Base):
    """
    Model for optimization run metadata.

    Tracks high-level information about optimization processes
    including configuration, status, and timing information.
    """

    __tablename__ = "optimization_runs"

    # Primary identification
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    algorithm_name = Column(String(100), nullable=False, index=True)
    strategy_name = Column(String(100), nullable=True, index=True)

    # Configuration
    parameter_space = Column(JSONB, nullable=False)
    objectives_config = Column(JSONB, nullable=False)
    constraints_config = Column(JSONB, nullable=True)
    algorithm_config = Column(JSONB, nullable=True)

    # Status tracking
    status = Column(String(20), nullable=False, default="pending", index=True)

    # Progress information
    current_iteration = Column(Integer, default=0, nullable=False)
    total_iterations = Column(Integer, default=0, nullable=False)
    completion_percentage: Mapped[Decimal] = mapped_column(
        DECIMAL(5, 2), default=Decimal("0.00"), nullable=False
    )
    evaluations_completed = Column(Integer, default=0, nullable=False)

    # Timing information
    start_time = Column(DateTime(timezone=True), nullable=True)
    end_time = Column(DateTime(timezone=True), nullable=True)
    total_duration_seconds: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)
    estimated_completion_time = Column(DateTime(timezone=True), nullable=True)

    # Results summary
    best_objective_value: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)
    convergence_achieved = Column(Boolean, default=False, nullable=False)

    # Metadata
    trading_mode = Column(String(20), nullable=True)
    data_start_date = Column(DateTime(timezone=True), nullable=True)
    data_end_date = Column(DateTime(timezone=True), nullable=True)
    initial_capital: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)

    # Error handling
    error_message = Column(Text, nullable=True)
    warnings = Column(JSON, nullable=True)

    # Audit fields
    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    created_by = Column(String(100), nullable=True)

    # Relationships
    results = relationship(
        "OptimizationResult",
        back_populates="optimization_run",
        cascade="all, delete-orphan",
        lazy="select",
    )

    parameter_sets = relationship(
        "ParameterSet",
        back_populates="optimization_run",
        cascade="all, delete-orphan",
        lazy="select",
    )

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "completion_percentage >= 0 AND completion_percentage <= 100",
            name="check_completion_percentage_range",
        ),
        CheckConstraint("current_iteration >= 0", name="check_current_iteration_positive"),
        CheckConstraint("total_iterations >= 0", name="check_total_iterations_positive"),
        CheckConstraint("evaluations_completed >= 0", name="check_evaluations_positive"),
        CheckConstraint("total_duration_seconds >= 0", name="check_duration_positive"),
        CheckConstraint("initial_capital > 0", name="check_initial_capital_positive"),
        CheckConstraint(
            "status IN ('pending', 'initializing', 'running', 'paused', 'completed', 'failed', 'cancelled')",
            name="check_valid_status",
        ),
        CheckConstraint(
            "end_time IS NULL OR start_time IS NULL OR end_time >= start_time",
            name="check_end_after_start",
        ),
        # Performance indexes
        Index("idx_optimization_run_strategy_status", "strategy_name", "status"),
        Index("idx_optimization_run_algorithm_created", "algorithm_name", "created_at"),
        Index("idx_optimization_run_status_updated", "status", "updated_at"),
    )


class OptimizationResult(Base):
    """
    Model for storing final optimization results.

    Contains the optimal parameters found, objective values achieved,
    and comprehensive performance metrics for financial analysis.
    """

    __tablename__ = "optimization_results"

    # Primary identification
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    optimization_run_id = Column(
        UUID(as_uuid=True),
        ForeignKey("optimization_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Optimal solution
    optimal_parameters = Column(JSONB, nullable=False)
    optimal_objective_value: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    objective_values = Column(JSONB, nullable=False)  # All objective values at optimal point

    # Validation results
    validation_score: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)
    overfitting_score: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)
    robustness_score: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)

    # Quality metrics
    parameter_stability = Column(JSONB, nullable=True)  # Parameter stability scores
    sensitivity_analysis = Column(JSONB, nullable=True)  # Parameter sensitivity analysis

    # Statistical significance
    statistical_significance: Mapped[Decimal | None] = mapped_column(DECIMAL(10, 8), nullable=True)
    confidence_interval_lower: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)
    confidence_interval_upper: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)

    # Performance metrics specific to trading
    sharpe_ratio: Mapped[Decimal | None] = mapped_column(DECIMAL(10, 6), nullable=True)
    max_drawdown: Mapped[Decimal | None] = mapped_column(DECIMAL(10, 6), nullable=True)
    total_return: Mapped[Decimal | None] = mapped_column(DECIMAL(10, 6), nullable=True)
    win_rate: Mapped[Decimal | None] = mapped_column(DECIMAL(5, 4), nullable=True)
    profit_factor: Mapped[Decimal | None] = mapped_column(DECIMAL(10, 6), nullable=True)
    volatility: Mapped[Decimal | None] = mapped_column(DECIMAL(10, 6), nullable=True)

    # Risk metrics
    value_at_risk: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)
    conditional_var: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)
    beta: Mapped[Decimal | None] = mapped_column(DECIMAL(10, 6), nullable=True)
    alpha: Mapped[Decimal | None] = mapped_column(DECIMAL(10, 6), nullable=True)

    # Additional metadata
    is_statistically_significant = Column(Boolean, default=False, nullable=False)
    confidence_level: Mapped[Decimal] = mapped_column(
        DECIMAL(4, 3), default=Decimal("0.95"), nullable=False
    )

    # Audit fields
    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    optimization_run = relationship("OptimizationRun", back_populates="results", lazy="select")

    # Constraints
    __table_args__ = (
        CheckConstraint("win_rate >= 0 AND win_rate <= 1", name="check_win_rate_range"),
        CheckConstraint("max_drawdown >= 0 AND max_drawdown <= 1", name="check_drawdown_range"),
        CheckConstraint(
            "confidence_level > 0 AND confidence_level < 1", name="check_confidence_level_range"
        ),
        CheckConstraint(
            "statistical_significance >= 0 AND statistical_significance <= 1",
            name="check_statistical_significance_range",
        ),
        CheckConstraint("volatility >= 0", name="check_volatility_positive"),
        CheckConstraint("profit_factor >= 0", name="check_profit_factor_positive"),
        CheckConstraint(
            "confidence_interval_lower IS NULL OR confidence_interval_upper IS NULL OR confidence_interval_lower <= confidence_interval_upper",
            name="check_confidence_interval_order",
        ),
        # Performance indexes
        Index(
            "idx_optimization_result_run_objective",
            "optimization_run_id",
            "optimal_objective_value",
        ),
        Index("idx_optimization_result_sharpe", "sharpe_ratio"),
        Index("idx_optimization_result_return", "total_return"),
        Index("idx_optimization_result_significant", "is_statistically_significant"),
    )


class ParameterSet(Base):
    """
    Model for storing individual parameter sets evaluated during optimization.

    Tracks all parameter combinations tested with their corresponding
    objective values for analysis and debugging purposes.
    """

    __tablename__ = "parameter_sets"

    # Primary identification
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    optimization_run_id = Column(
        UUID(as_uuid=True),
        ForeignKey("optimization_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Parameter data
    parameters = Column(JSONB, nullable=False)
    parameter_hash = Column(String(64), nullable=False, index=True)  # For deduplication

    # Evaluation results
    objective_value: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)
    objective_values = Column(JSONB, nullable=True)  # All objective values
    constraint_violations = Column(JSONB, nullable=True)  # Constraint violation details
    is_feasible = Column(Boolean, default=True, nullable=False)

    # Evaluation metadata
    iteration_number = Column(Integer, nullable=False, index=True)
    evaluation_time_seconds: Mapped[Decimal | None] = mapped_column(DECIMAL(10, 6), nullable=True)
    evaluation_status = Column(String(20), default="completed", nullable=False)
    evaluation_error = Column(Text, nullable=True)

    # Performance metrics (for quick analysis)
    sharpe_ratio: Mapped[Decimal | None] = mapped_column(DECIMAL(10, 6), nullable=True)
    total_return: Mapped[Decimal | None] = mapped_column(DECIMAL(10, 6), nullable=True)
    max_drawdown: Mapped[Decimal | None] = mapped_column(DECIMAL(10, 6), nullable=True)

    # Ranking information
    rank_by_objective = Column(Integer, nullable=True)
    percentile_rank: Mapped[Decimal | None] = mapped_column(DECIMAL(5, 2), nullable=True)

    # Audit fields
    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    # Relationships
    optimization_run = relationship(
        "OptimizationRun", back_populates="parameter_sets", lazy="select"
    )

    # Constraints
    __table_args__ = (
        UniqueConstraint("optimization_run_id", "parameter_hash", name="uq_parameter_set_run_hash"),
        CheckConstraint("iteration_number >= 0", name="check_iteration_positive"),
        CheckConstraint("evaluation_time_seconds >= 0", name="check_evaluation_time_positive"),
        CheckConstraint(
            "percentile_rank >= 0 AND percentile_rank <= 100", name="check_percentile_range"
        ),
        CheckConstraint("rank_by_objective > 0", name="check_rank_positive"),
        CheckConstraint(
            "evaluation_status IN ('pending', 'running', 'completed', 'failed', 'timeout')",
            name="check_valid_evaluation_status",
        ),
        CheckConstraint("max_drawdown >= 0 AND max_drawdown <= 1", name="check_drawdown_range"),
        # Performance indexes
        Index("idx_parameter_set_run_iteration", "optimization_run_id", "iteration_number"),
        Index("idx_parameter_set_objective_rank", "optimization_run_id", "rank_by_objective"),
        Index("idx_parameter_set_feasible_objective", "is_feasible", "objective_value"),
        Index("idx_parameter_set_status_created", "evaluation_status", "created_at"),
    )


class OptimizationObjectiveDB(Base):
    """
    Model for storing individual optimization objectives.

    Stores objective definitions including direction, weights,
    constraints, and targets for multi-objective optimization.
    """

    __tablename__ = "optimization_objectives"

    # Primary identification
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    optimization_run_id = Column(
        UUID(as_uuid=True),
        ForeignKey("optimization_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Objective definition
    name = Column(String(100), nullable=False, index=True)
    direction = Column(String(10), nullable=False)  # 'maximize' or 'minimize'
    weight: Mapped[Decimal] = mapped_column(DECIMAL(10, 6), default=Decimal("1.0"), nullable=False)
    is_primary = Column(Boolean, default=False, nullable=False)

    # Targets and constraints
    target_value: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)
    constraint_min: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)
    constraint_max: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)

    # Metadata
    description = Column(Text, nullable=True)
    units = Column(String(50), nullable=True)
    category = Column(
        String(50), nullable=True, index=True
    )  # e.g., 'performance', 'risk', 'efficiency'

    # Results tracking
    best_value_achieved: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)
    worst_value_achieved: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)
    average_value: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)
    standard_deviation: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)

    # Audit fields
    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    optimization_run = relationship("OptimizationRun", lazy="select")

    # Constraints
    __table_args__ = (
        UniqueConstraint("optimization_run_id", "name", name="uq_objective_run_name"),
        CheckConstraint("direction IN ('maximize', 'minimize')", name="check_valid_direction"),
        CheckConstraint("weight >= 0", name="check_weight_positive"),
        CheckConstraint(
            "constraint_min IS NULL OR constraint_max IS NULL OR constraint_min <= constraint_max",
            name="check_constraint_order",
        ),
        CheckConstraint("standard_deviation >= 0", name="check_std_dev_positive"),
        # Performance indexes
        Index("idx_objective_run_primary", "optimization_run_id", "is_primary"),
        Index("idx_objective_category_direction", "category", "direction"),
        Index("idx_objective_name_weight", "name", "weight"),
    )
