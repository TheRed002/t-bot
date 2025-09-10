"""Add optimization models

Revision ID: 004_optimization_models
Revises: 003_backtesting_models
Create Date: 2025-01-07 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy import text

# revision identifiers, used by Alembic.
revision = "004_optimization_models"
down_revision = "003_backtesting_models"
branch_labels = None
depends_on = None


def upgrade():
    """Create optimization models tables."""

    # Create optimization_runs table
    op.create_table(
        "optimization_runs",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("algorithm_name", sa.String(100), nullable=False),
        sa.Column("strategy_name", sa.String(100), nullable=True),
        sa.Column("parameter_space", sa.JSON(), nullable=False),
        sa.Column("objectives_config", sa.JSON(), nullable=False),
        sa.Column("constraints_config", sa.JSON(), nullable=True),
        sa.Column("algorithm_config", sa.JSON(), nullable=True),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("current_iteration", sa.Integer(), nullable=False),
        sa.Column("total_iterations", sa.Integer(), nullable=False),
        sa.Column("completion_percentage", sa.DECIMAL(5, 2), nullable=False),
        sa.Column("evaluations_completed", sa.Integer(), nullable=False),
        sa.Column("start_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("end_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("total_duration_seconds", sa.DECIMAL(20, 8), nullable=True),
        sa.Column("estimated_completion_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("best_objective_value", sa.DECIMAL(20, 8), nullable=True),
        sa.Column("convergence_achieved", sa.Boolean(), nullable=False),
        sa.Column("trading_mode", sa.String(20), nullable=True),
        sa.Column("data_start_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("data_end_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("initial_capital", sa.DECIMAL(20, 8), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("warnings", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_by", sa.String(100), nullable=True),
        sa.CheckConstraint(
            "completion_percentage >= 0 AND completion_percentage <= 100",
            name="check_completion_percentage_range",
        ),
        sa.CheckConstraint("current_iteration >= 0", name="check_current_iteration_positive"),
        sa.CheckConstraint("total_iterations >= 0", name="check_total_iterations_positive"),
        sa.CheckConstraint("evaluations_completed >= 0", name="check_evaluations_positive"),
        sa.CheckConstraint("total_duration_seconds >= 0", name="check_duration_positive"),
        sa.CheckConstraint("initial_capital > 0", name="check_initial_capital_positive"),
        sa.CheckConstraint(
            "status IN ('pending', 'initializing', 'running', 'paused', 'completed', 'failed', 'cancelled')",
            name="check_valid_status",
        ),
        sa.CheckConstraint(
            "end_time IS NULL OR start_time IS NULL OR end_time >= start_time",
            name="check_end_after_start",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.Index("idx_optimization_run_strategy_status", "strategy_name", "status"),
        sa.Index("idx_optimization_run_algorithm_created", "algorithm_name", "created_at"),
        sa.Index("idx_optimization_run_status_updated", "status", "updated_at"),
        sa.Index("ix_optimization_runs_id", "id"),
        sa.Index("ix_optimization_runs_algorithm_name", "algorithm_name"),
        sa.Index("ix_optimization_runs_strategy_name", "strategy_name"),
        sa.Index("ix_optimization_runs_status", "status"),
    )

    # Set default values
    op.execute(text("ALTER TABLE optimization_runs ALTER COLUMN status SET DEFAULT 'pending'"))
    op.execute(text("ALTER TABLE optimization_runs ALTER COLUMN current_iteration SET DEFAULT 0"))
    op.execute(text("ALTER TABLE optimization_runs ALTER COLUMN total_iterations SET DEFAULT 0"))
    op.execute(
        text("ALTER TABLE optimization_runs ALTER COLUMN completion_percentage SET DEFAULT 0.00")
    )
    op.execute(
        text("ALTER TABLE optimization_runs ALTER COLUMN evaluations_completed SET DEFAULT 0")
    )
    op.execute(
        text("ALTER TABLE optimization_runs ALTER COLUMN convergence_achieved SET DEFAULT false")
    )
    op.execute(
        text("ALTER TABLE optimization_runs ALTER COLUMN created_at SET DEFAULT CURRENT_TIMESTAMP")
    )
    op.execute(
        text("ALTER TABLE optimization_runs ALTER COLUMN updated_at SET DEFAULT CURRENT_TIMESTAMP")
    )

    # Create optimization_results table
    op.create_table(
        "optimization_results",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("optimization_run_id", sa.String(36), nullable=False),
        sa.Column("optimal_parameters", sa.JSON(), nullable=False),
        sa.Column("optimal_objective_value", sa.DECIMAL(20, 8), nullable=False),
        sa.Column("objective_values", sa.JSON(), nullable=False),
        sa.Column("validation_score", sa.DECIMAL(20, 8), nullable=True),
        sa.Column("overfitting_score", sa.DECIMAL(20, 8), nullable=True),
        sa.Column("robustness_score", sa.DECIMAL(20, 8), nullable=True),
        sa.Column("parameter_stability", sa.JSON(), nullable=True),
        sa.Column("sensitivity_analysis", sa.JSON(), nullable=True),
        sa.Column("statistical_significance", sa.DECIMAL(10, 8), nullable=True),
        sa.Column("confidence_interval_lower", sa.DECIMAL(20, 8), nullable=True),
        sa.Column("confidence_interval_upper", sa.DECIMAL(20, 8), nullable=True),
        sa.Column("sharpe_ratio", sa.DECIMAL(10, 6), nullable=True),
        sa.Column("max_drawdown", sa.DECIMAL(10, 6), nullable=True),
        sa.Column("total_return", sa.DECIMAL(10, 6), nullable=True),
        sa.Column("win_rate", sa.DECIMAL(5, 4), nullable=True),
        sa.Column("profit_factor", sa.DECIMAL(10, 6), nullable=True),
        sa.Column("volatility", sa.DECIMAL(10, 6), nullable=True),
        sa.Column("value_at_risk", sa.DECIMAL(20, 8), nullable=True),
        sa.Column("conditional_var", sa.DECIMAL(20, 8), nullable=True),
        sa.Column("beta", sa.DECIMAL(10, 6), nullable=True),
        sa.Column("alpha", sa.DECIMAL(10, 6), nullable=True),
        sa.Column("is_statistically_significant", sa.Boolean(), nullable=False),
        sa.Column("confidence_level", sa.DECIMAL(4, 3), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("win_rate >= 0 AND win_rate <= 1", name="check_win_rate_range"),
        sa.CheckConstraint("max_drawdown >= 0 AND max_drawdown <= 1", name="check_drawdown_range"),
        sa.CheckConstraint(
            "confidence_level > 0 AND confidence_level < 1", name="check_confidence_level_range"
        ),
        sa.CheckConstraint(
            "statistical_significance >= 0 AND statistical_significance <= 1",
            name="check_statistical_significance_range",
        ),
        sa.CheckConstraint("volatility >= 0", name="check_volatility_positive"),
        sa.CheckConstraint("profit_factor >= 0", name="check_profit_factor_positive"),
        sa.CheckConstraint(
            "confidence_interval_lower IS NULL OR confidence_interval_upper IS NULL OR confidence_interval_lower <= confidence_interval_upper",
            name="check_confidence_interval_order",
        ),
        sa.ForeignKeyConstraint(
            ["optimization_run_id"], ["optimization_runs.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.Index(
            "idx_optimization_result_run_objective",
            "optimization_run_id",
            "optimal_objective_value",
        ),
        sa.Index("idx_optimization_result_sharpe", "sharpe_ratio"),
        sa.Index("idx_optimization_result_return", "total_return"),
        sa.Index("idx_optimization_result_significant", "is_statistically_significant"),
        sa.Index("ix_optimization_results_id", "id"),
        sa.Index("ix_optimization_results_optimization_run_id", "optimization_run_id"),
    )

    # Set default values
    op.execute(
        text(
            "ALTER TABLE optimization_results ALTER COLUMN is_statistically_significant SET DEFAULT false"
        )
    )
    op.execute(
        text("ALTER TABLE optimization_results ALTER COLUMN confidence_level SET DEFAULT 0.95")
    )
    op.execute(
        text(
            "ALTER TABLE optimization_results ALTER COLUMN created_at SET DEFAULT CURRENT_TIMESTAMP"
        )
    )
    op.execute(
        text(
            "ALTER TABLE optimization_results ALTER COLUMN updated_at SET DEFAULT CURRENT_TIMESTAMP"
        )
    )

    # Create parameter_sets table
    op.create_table(
        "parameter_sets",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("optimization_run_id", sa.String(36), nullable=False),
        sa.Column("parameters", sa.JSON(), nullable=False),
        sa.Column("parameter_hash", sa.String(64), nullable=False),
        sa.Column("objective_value", sa.DECIMAL(20, 8), nullable=True),
        sa.Column("objective_values", sa.JSON(), nullable=True),
        sa.Column("constraint_violations", sa.JSON(), nullable=True),
        sa.Column("is_feasible", sa.Boolean(), nullable=False),
        sa.Column("iteration_number", sa.Integer(), nullable=False),
        sa.Column("evaluation_time_seconds", sa.DECIMAL(10, 6), nullable=True),
        sa.Column("evaluation_status", sa.String(20), nullable=False),
        sa.Column("evaluation_error", sa.Text(), nullable=True),
        sa.Column("sharpe_ratio", sa.DECIMAL(10, 6), nullable=True),
        sa.Column("total_return", sa.DECIMAL(10, 6), nullable=True),
        sa.Column("max_drawdown", sa.DECIMAL(10, 6), nullable=True),
        sa.Column("rank_by_objective", sa.Integer(), nullable=True),
        sa.Column("percentile_rank", sa.DECIMAL(5, 2), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("iteration_number >= 0", name="check_iteration_positive"),
        sa.CheckConstraint("evaluation_time_seconds >= 0", name="check_evaluation_time_positive"),
        sa.CheckConstraint(
            "percentile_rank >= 0 AND percentile_rank <= 100", name="check_percentile_range"
        ),
        sa.CheckConstraint("rank_by_objective > 0", name="check_rank_positive"),
        sa.CheckConstraint(
            "evaluation_status IN ('pending', 'running', 'completed', 'failed', 'timeout')",
            name="check_valid_evaluation_status",
        ),
        sa.CheckConstraint("max_drawdown >= 0 AND max_drawdown <= 1", name="check_drawdown_range"),
        sa.ForeignKeyConstraint(
            ["optimization_run_id"], ["optimization_runs.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "optimization_run_id", "parameter_hash", name="uq_parameter_set_run_hash"
        ),
        sa.Index("idx_parameter_set_run_iteration", "optimization_run_id", "iteration_number"),
        sa.Index("idx_parameter_set_objective_rank", "optimization_run_id", "rank_by_objective"),
        sa.Index("idx_parameter_set_feasible_objective", "is_feasible", "objective_value"),
        sa.Index("idx_parameter_set_status_created", "evaluation_status", "created_at"),
        sa.Index("ix_parameter_sets_id", "id"),
        sa.Index("ix_parameter_sets_optimization_run_id", "optimization_run_id"),
        sa.Index("ix_parameter_sets_parameter_hash", "parameter_hash"),
        sa.Index("ix_parameter_sets_iteration_number", "iteration_number"),
    )

    # Set default values
    op.execute(text("ALTER TABLE parameter_sets ALTER COLUMN is_feasible SET DEFAULT true"))
    op.execute(
        text("ALTER TABLE parameter_sets ALTER COLUMN evaluation_status SET DEFAULT 'completed'")
    )
    op.execute(
        text("ALTER TABLE parameter_sets ALTER COLUMN created_at SET DEFAULT CURRENT_TIMESTAMP")
    )

    # Create optimization_objectives table
    op.create_table(
        "optimization_objectives",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("optimization_run_id", sa.String(36), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("direction", sa.String(10), nullable=False),
        sa.Column("weight", sa.DECIMAL(10, 6), nullable=False),
        sa.Column("is_primary", sa.Boolean(), nullable=False),
        sa.Column("target_value", sa.DECIMAL(20, 8), nullable=True),
        sa.Column("constraint_min", sa.DECIMAL(20, 8), nullable=True),
        sa.Column("constraint_max", sa.DECIMAL(20, 8), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("units", sa.String(50), nullable=True),
        sa.Column("category", sa.String(50), nullable=True),
        sa.Column("best_value_achieved", sa.DECIMAL(20, 8), nullable=True),
        sa.Column("worst_value_achieved", sa.DECIMAL(20, 8), nullable=True),
        sa.Column("average_value", sa.DECIMAL(20, 8), nullable=True),
        sa.Column("standard_deviation", sa.DECIMAL(20, 8), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("direction IN ('maximize', 'minimize')", name="check_valid_direction"),
        sa.CheckConstraint("weight >= 0", name="check_weight_positive"),
        sa.CheckConstraint(
            "constraint_min IS NULL OR constraint_max IS NULL OR constraint_min <= constraint_max",
            name="check_constraint_order",
        ),
        sa.CheckConstraint("standard_deviation >= 0", name="check_std_dev_positive"),
        sa.ForeignKeyConstraint(
            ["optimization_run_id"], ["optimization_runs.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("optimization_run_id", "name", name="uq_objective_run_name"),
        sa.Index("idx_objective_run_primary", "optimization_run_id", "is_primary"),
        sa.Index("idx_objective_category_direction", "category", "direction"),
        sa.Index("idx_objective_name_weight", "name", "weight"),
        sa.Index("ix_optimization_objectives_id", "id"),
        sa.Index("ix_optimization_objectives_optimization_run_id", "optimization_run_id"),
        sa.Index("ix_optimization_objectives_name", "name"),
        sa.Index("ix_optimization_objectives_category", "category"),
    )

    # Set default values
    op.execute(text("ALTER TABLE optimization_objectives ALTER COLUMN weight SET DEFAULT 1.0"))
    op.execute(
        text("ALTER TABLE optimization_objectives ALTER COLUMN is_primary SET DEFAULT false")
    )
    op.execute(
        text(
            "ALTER TABLE optimization_objectives ALTER COLUMN created_at SET DEFAULT CURRENT_TIMESTAMP"
        )
    )
    op.execute(
        text(
            "ALTER TABLE optimization_objectives ALTER COLUMN updated_at SET DEFAULT CURRENT_TIMESTAMP"
        )
    )


def downgrade():
    """Drop optimization models tables."""
    op.drop_table("optimization_objectives")
    op.drop_table("parameter_sets")
    op.drop_table("optimization_results")
    op.drop_table("optimization_runs")
