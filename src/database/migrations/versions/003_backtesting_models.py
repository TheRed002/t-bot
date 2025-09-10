"""Add backtesting database models

Revision ID: 003_backtesting_models
Revises: 002_data_models
Create Date: 2024-01-20 10:00:00.000000

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "003_backtesting_models"
down_revision = "002_data_models"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create backtest_runs table
    op.create_table(
        "backtest_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("strategy_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("symbols", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("exchange", sa.String(length=50), nullable=False),
        sa.Column("timeframe", sa.String(length=20), nullable=False),
        sa.Column("start_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("end_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("initial_capital", sa.DECIMAL(precision=20, scale=8), nullable=False),
        sa.Column(
            "commission_rate", sa.DECIMAL(precision=10, scale=6), nullable=False, server_default="0"
        ),
        sa.Column(
            "slippage_rate", sa.DECIMAL(precision=10, scale=6), nullable=False, server_default="0"
        ),
        sa.Column("enable_shorting", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("max_open_positions", sa.Integer(), nullable=False, server_default="5"),
        sa.Column("strategy_config", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("risk_config", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("execution_start_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("execution_end_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("execution_duration_seconds", sa.Integer(), nullable=True),
        sa.Column("progress_percentage", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("current_stage", sa.String(length=50), nullable=True),
        sa.Column("total_trades", sa.Integer(), nullable=True),
        sa.Column("winning_trades", sa.Integer(), nullable=True),
        sa.Column("losing_trades", sa.Integer(), nullable=True),
        sa.Column("total_return_pct", sa.DECIMAL(precision=10, scale=4), nullable=True),
        sa.Column("max_drawdown_pct", sa.DECIMAL(precision=10, scale=4), nullable=True),
        sa.Column("sharpe_ratio", sa.DECIMAL(precision=10, scale=4), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("additional_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["strategy_id"], ["strategies.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes for backtest_runs
    op.create_index("idx_backtest_runs_user_id", "backtest_runs", ["user_id"], unique=False)
    op.create_index("idx_backtest_runs_strategy_id", "backtest_runs", ["strategy_id"], unique=False)
    op.create_index("idx_backtest_runs_status", "backtest_runs", ["status"], unique=False)
    op.create_index("idx_backtest_runs_exchange", "backtest_runs", ["exchange"], unique=False)
    op.create_index("idx_backtest_runs_timeframe", "backtest_runs", ["timeframe"], unique=False)
    op.create_index("idx_backtest_runs_start_date", "backtest_runs", ["start_date"], unique=False)
    op.create_index("idx_backtest_runs_end_date", "backtest_runs", ["end_date"], unique=False)
    op.create_index(
        "idx_backtest_runs_user_status", "backtest_runs", ["user_id", "status"], unique=False
    )
    op.create_index(
        "idx_backtest_runs_execution_times",
        "backtest_runs",
        ["execution_start_time", "execution_end_time"],
        unique=False,
    )

    # Create constraints for backtest_runs
    op.create_check_constraint(
        "check_backtest_date_range", "backtest_runs", "end_date > start_date"
    )
    op.create_check_constraint(
        "check_backtest_initial_capital_positive", "backtest_runs", "initial_capital > 0"
    )
    op.create_check_constraint(
        "check_backtest_commission_rate_range",
        "backtest_runs",
        "commission_rate >= 0 AND commission_rate <= 1",
    )
    op.create_check_constraint(
        "check_backtest_slippage_rate_range",
        "backtest_runs",
        "slippage_rate >= 0 AND slippage_rate <= 1",
    )
    op.create_check_constraint(
        "check_backtest_max_positions_range",
        "backtest_runs",
        "max_open_positions > 0 AND max_open_positions <= 100",
    )
    op.create_check_constraint(
        "check_backtest_progress_range",
        "backtest_runs",
        "progress_percentage >= 0 AND progress_percentage <= 100",
    )
    op.create_check_constraint(
        "check_backtest_status",
        "backtest_runs",
        "status IN ('pending', 'running', 'completed', 'failed', 'cancelled')",
    )
    op.create_check_constraint(
        "check_backtest_exchange",
        "backtest_runs",
        "exchange IN ('binance', 'coinbase', 'okx', 'mock')",
    )
    op.create_check_constraint(
        "check_backtest_total_trades_non_negative",
        "backtest_runs",
        "total_trades IS NULL OR total_trades >= 0",
    )
    op.create_check_constraint(
        "check_backtest_winning_trades_non_negative",
        "backtest_runs",
        "winning_trades IS NULL OR winning_trades >= 0",
    )
    op.create_check_constraint(
        "check_backtest_losing_trades_non_negative",
        "backtest_runs",
        "losing_trades IS NULL OR losing_trades >= 0",
    )
    op.create_check_constraint(
        "check_backtest_trades_consistency",
        "backtest_runs",
        "(winning_trades IS NULL AND losing_trades IS NULL AND total_trades IS NULL) OR (winning_trades + losing_trades <= total_trades)",
    )

    # Create unique constraint for backtest_runs
    op.create_unique_constraint("uq_backtest_runs_user_name", "backtest_runs", ["user_id", "name"])

    # Create backtest_results table
    op.create_table(
        "backtest_results",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("backtest_run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("total_return_pct", sa.DECIMAL(precision=10, scale=4), nullable=False),
        sa.Column("annual_return_pct", sa.DECIMAL(precision=10, scale=4), nullable=False),
        sa.Column("sharpe_ratio", sa.DECIMAL(precision=10, scale=4), nullable=True),
        sa.Column("sortino_ratio", sa.DECIMAL(precision=10, scale=4), nullable=True),
        sa.Column("max_drawdown_pct", sa.DECIMAL(precision=10, scale=4), nullable=False),
        sa.Column("win_rate_pct", sa.DECIMAL(precision=5, scale=4), nullable=False),
        sa.Column("total_trades", sa.Integer(), nullable=False),
        sa.Column("winning_trades", sa.Integer(), nullable=False),
        sa.Column("losing_trades", sa.Integer(), nullable=False),
        sa.Column("avg_win_amount", sa.DECIMAL(precision=20, scale=8), nullable=True),
        sa.Column("avg_loss_amount", sa.DECIMAL(precision=20, scale=8), nullable=True),
        sa.Column("profit_factor", sa.DECIMAL(precision=10, scale=4), nullable=True),
        sa.Column("volatility_pct", sa.DECIMAL(precision=10, scale=4), nullable=True),
        sa.Column("value_at_risk_95_pct", sa.DECIMAL(precision=10, scale=4), nullable=True),
        sa.Column("conditional_var_95_pct", sa.DECIMAL(precision=10, scale=4), nullable=True),
        sa.Column("initial_capital", sa.DECIMAL(precision=20, scale=8), nullable=False),
        sa.Column("final_capital", sa.DECIMAL(precision=20, scale=8), nullable=False),
        sa.Column("peak_capital", sa.DECIMAL(precision=20, scale=8), nullable=False),
        sa.Column("lowest_capital", sa.DECIMAL(precision=20, scale=8), nullable=False),
        sa.Column("total_time_in_market_hours", sa.DECIMAL(precision=10, scale=2), nullable=True),
        sa.Column("avg_trade_duration_hours", sa.DECIMAL(precision=10, scale=2), nullable=True),
        sa.Column("longest_winning_streak", sa.Integer(), nullable=True),
        sa.Column("longest_losing_streak", sa.Integer(), nullable=True),
        sa.Column("equity_curve", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("daily_returns", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("monte_carlo_results", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("walk_forward_results", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "performance_attribution", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("analysis_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["backtest_run_id"], ["backtest_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes for backtest_results
    op.create_index(
        "idx_backtest_results_run_id", "backtest_results", ["backtest_run_id"], unique=False
    )
    op.create_index(
        "idx_backtest_results_total_return", "backtest_results", ["total_return_pct"], unique=False
    )
    op.create_index(
        "idx_backtest_results_sharpe_ratio", "backtest_results", ["sharpe_ratio"], unique=False
    )
    op.create_index(
        "idx_backtest_results_max_drawdown", "backtest_results", ["max_drawdown_pct"], unique=False
    )
    op.create_index(
        "idx_backtest_results_win_rate", "backtest_results", ["win_rate_pct"], unique=False
    )
    op.create_index(
        "idx_backtest_results_profit_factor", "backtest_results", ["profit_factor"], unique=False
    )

    # Create constraints for backtest_results
    op.create_check_constraint(
        "check_backtest_result_total_trades_non_negative", "backtest_results", "total_trades >= 0"
    )
    op.create_check_constraint(
        "check_backtest_result_winning_trades_non_negative",
        "backtest_results",
        "winning_trades >= 0",
    )
    op.create_check_constraint(
        "check_backtest_result_losing_trades_non_negative", "backtest_results", "losing_trades >= 0"
    )
    op.create_check_constraint(
        "check_backtest_result_trades_consistency",
        "backtest_results",
        "winning_trades + losing_trades <= total_trades",
    )
    op.create_check_constraint(
        "check_backtest_result_win_rate_range",
        "backtest_results",
        "win_rate_pct >= 0 AND win_rate_pct <= 1",
    )
    op.create_check_constraint(
        "check_backtest_result_drawdown_range",
        "backtest_results",
        "max_drawdown_pct >= 0 AND max_drawdown_pct <= 1",
    )
    op.create_check_constraint(
        "check_backtest_result_initial_capital_positive", "backtest_results", "initial_capital > 0"
    )
    op.create_check_constraint(
        "check_backtest_result_final_capital_non_negative", "backtest_results", "final_capital >= 0"
    )
    op.create_check_constraint(
        "check_backtest_result_peak_capital_valid",
        "backtest_results",
        "peak_capital >= initial_capital",
    )
    op.create_check_constraint(
        "check_backtest_result_lowest_capital_non_negative",
        "backtest_results",
        "lowest_capital >= 0",
    )
    op.create_check_constraint(
        "check_backtest_result_lowest_capital_valid",
        "backtest_results",
        "lowest_capital <= initial_capital",
    )
    op.create_check_constraint(
        "check_backtest_result_profit_factor_non_negative",
        "backtest_results",
        "profit_factor IS NULL OR profit_factor >= 0",
    )
    op.create_check_constraint(
        "check_backtest_result_avg_win_positive",
        "backtest_results",
        "avg_win_amount IS NULL OR avg_win_amount > 0",
    )
    op.create_check_constraint(
        "check_backtest_result_avg_loss_negative",
        "backtest_results",
        "avg_loss_amount IS NULL OR avg_loss_amount < 0",
    )
    op.create_check_constraint(
        "check_backtest_result_market_time_non_negative",
        "backtest_results",
        "total_time_in_market_hours IS NULL OR total_time_in_market_hours >= 0",
    )
    op.create_check_constraint(
        "check_backtest_result_avg_duration_positive",
        "backtest_results",
        "avg_trade_duration_hours IS NULL OR avg_trade_duration_hours > 0",
    )
    op.create_check_constraint(
        "check_backtest_result_win_streak_non_negative",
        "backtest_results",
        "longest_winning_streak IS NULL OR longest_winning_streak >= 0",
    )
    op.create_check_constraint(
        "check_backtest_result_lose_streak_non_negative",
        "backtest_results",
        "longest_losing_streak IS NULL OR longest_losing_streak >= 0",
    )

    # Create unique constraint for backtest_results
    op.create_unique_constraint(
        "uq_backtest_results_run_id", "backtest_results", ["backtest_run_id"]
    )

    # Create backtest_trades table
    op.create_table(
        "backtest_trades",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("backtest_run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("trade_sequence", sa.Integer(), nullable=False),
        sa.Column("symbol", sa.String(length=20), nullable=False),
        sa.Column("entry_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("entry_price", sa.DECIMAL(precision=20, scale=8), nullable=False),
        sa.Column("entry_signal_strength", sa.DECIMAL(precision=5, scale=4), nullable=True),
        sa.Column("exit_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("exit_price", sa.DECIMAL(precision=20, scale=8), nullable=False),
        sa.Column("exit_reason", sa.String(length=50), nullable=False),
        sa.Column("side", sa.String(length=10), nullable=False),
        sa.Column("quantity", sa.DECIMAL(precision=20, scale=8), nullable=False),
        sa.Column("position_size_usd", sa.DECIMAL(precision=20, scale=8), nullable=False),
        sa.Column("pnl_usd", sa.DECIMAL(precision=20, scale=8), nullable=False),
        sa.Column("pnl_percentage", sa.DECIMAL(precision=10, scale=6), nullable=False),
        sa.Column(
            "commission_paid", sa.DECIMAL(precision=20, scale=8), nullable=False, server_default="0"
        ),
        sa.Column(
            "slippage_cost", sa.DECIMAL(precision=20, scale=8), nullable=False, server_default="0"
        ),
        sa.Column("duration_hours", sa.DECIMAL(precision=10, scale=2), nullable=False),
        sa.Column("bars_held", sa.Integer(), nullable=True),
        sa.Column("max_adverse_excursion_pct", sa.DECIMAL(precision=10, scale=6), nullable=True),
        sa.Column("max_favorable_excursion_pct", sa.DECIMAL(precision=10, scale=6), nullable=True),
        sa.Column("strategy_signal_data", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("risk_metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("execution_algorithm", sa.String(length=50), nullable=True),
        sa.Column("execution_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["backtest_run_id"], ["backtest_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes for backtest_trades
    op.create_index(
        "idx_backtest_trades_run_id", "backtest_trades", ["backtest_run_id"], unique=False
    )
    op.create_index("idx_backtest_trades_symbol", "backtest_trades", ["symbol"], unique=False)
    op.create_index("idx_backtest_trades_side", "backtest_trades", ["side"], unique=False)
    op.create_index(
        "idx_backtest_trades_entry_time", "backtest_trades", ["entry_timestamp"], unique=False
    )
    op.create_index(
        "idx_backtest_trades_exit_time", "backtest_trades", ["exit_timestamp"], unique=False
    )
    op.create_index("idx_backtest_trades_pnl", "backtest_trades", ["pnl_usd"], unique=False)
    op.create_index(
        "idx_backtest_trades_duration", "backtest_trades", ["duration_hours"], unique=False
    )
    op.create_index(
        "idx_backtest_trades_run_sequence",
        "backtest_trades",
        ["backtest_run_id", "trade_sequence"],
        unique=False,
    )
    op.create_index(
        "idx_backtest_trades_run_symbol",
        "backtest_trades",
        ["backtest_run_id", "symbol"],
        unique=False,
    )
    op.create_index(
        "idx_backtest_trades_performance",
        "backtest_trades",
        ["backtest_run_id", "pnl_usd"],
        unique=False,
    )

    # Create constraints for backtest_trades
    op.create_check_constraint(
        "check_backtest_trade_time_order", "backtest_trades", "exit_timestamp > entry_timestamp"
    )
    op.create_check_constraint(
        "check_backtest_trade_entry_price_positive", "backtest_trades", "entry_price > 0"
    )
    op.create_check_constraint(
        "check_backtest_trade_exit_price_positive", "backtest_trades", "exit_price > 0"
    )
    op.create_check_constraint(
        "check_backtest_trade_quantity_positive", "backtest_trades", "quantity > 0"
    )
    op.create_check_constraint(
        "check_backtest_trade_position_size_positive", "backtest_trades", "position_size_usd > 0"
    )
    op.create_check_constraint(
        "check_backtest_trade_commission_non_negative", "backtest_trades", "commission_paid >= 0"
    )
    op.create_check_constraint(
        "check_backtest_trade_slippage_non_negative", "backtest_trades", "slippage_cost >= 0"
    )
    op.create_check_constraint(
        "check_backtest_trade_duration_positive", "backtest_trades", "duration_hours > 0"
    )
    op.create_check_constraint(
        "check_backtest_trade_sequence_positive", "backtest_trades", "trade_sequence > 0"
    )
    op.create_check_constraint(
        "check_backtest_trade_side", "backtest_trades", "side IN ('LONG', 'SHORT')"
    )
    op.create_check_constraint(
        "check_backtest_trade_exit_reason",
        "backtest_trades",
        "exit_reason IN ('signal', 'stop_loss', 'take_profit', 'max_duration', 'backtest_end', 'risk_limit')",
    )
    op.create_check_constraint(
        "check_backtest_trade_signal_strength_range",
        "backtest_trades",
        "entry_signal_strength IS NULL OR (entry_signal_strength >= 0 AND entry_signal_strength <= 1)",
    )
    op.create_check_constraint(
        "check_backtest_trade_bars_held_positive",
        "backtest_trades",
        "bars_held IS NULL OR bars_held > 0",
    )
    op.create_check_constraint(
        "check_backtest_trade_mae_non_positive",
        "backtest_trades",
        "max_adverse_excursion_pct IS NULL OR max_adverse_excursion_pct <= 0",
    )
    op.create_check_constraint(
        "check_backtest_trade_mfe_non_negative",
        "backtest_trades",
        "max_favorable_excursion_pct IS NULL OR max_favorable_excursion_pct >= 0",
    )

    # Create unique constraint for backtest_trades
    op.create_unique_constraint(
        "uq_backtest_trades_run_sequence", "backtest_trades", ["backtest_run_id", "trade_sequence"]
    )


def downgrade() -> None:
    # Drop backtest_trades table
    op.drop_constraint("uq_backtest_trades_run_sequence", "backtest_trades", type_="unique")
    op.drop_index("idx_backtest_trades_performance", table_name="backtest_trades")
    op.drop_index("idx_backtest_trades_run_symbol", table_name="backtest_trades")
    op.drop_index("idx_backtest_trades_run_sequence", table_name="backtest_trades")
    op.drop_index("idx_backtest_trades_duration", table_name="backtest_trades")
    op.drop_index("idx_backtest_trades_pnl", table_name="backtest_trades")
    op.drop_index("idx_backtest_trades_exit_time", table_name="backtest_trades")
    op.drop_index("idx_backtest_trades_entry_time", table_name="backtest_trades")
    op.drop_index("idx_backtest_trades_side", table_name="backtest_trades")
    op.drop_index("idx_backtest_trades_symbol", table_name="backtest_trades")
    op.drop_index("idx_backtest_trades_run_id", table_name="backtest_trades")
    op.drop_table("backtest_trades")

    # Drop backtest_results table
    op.drop_constraint("uq_backtest_results_run_id", "backtest_results", type_="unique")
    op.drop_index("idx_backtest_results_profit_factor", table_name="backtest_results")
    op.drop_index("idx_backtest_results_win_rate", table_name="backtest_results")
    op.drop_index("idx_backtest_results_max_drawdown", table_name="backtest_results")
    op.drop_index("idx_backtest_results_sharpe_ratio", table_name="backtest_results")
    op.drop_index("idx_backtest_results_total_return", table_name="backtest_results")
    op.drop_index("idx_backtest_results_run_id", table_name="backtest_results")
    op.drop_table("backtest_results")

    # Drop backtest_runs table
    op.drop_constraint("uq_backtest_runs_user_name", "backtest_runs", type_="unique")
    op.drop_index("idx_backtest_runs_execution_times", table_name="backtest_runs")
    op.drop_index("idx_backtest_runs_user_status", table_name="backtest_runs")
    op.drop_index("idx_backtest_runs_end_date", table_name="backtest_runs")
    op.drop_index("idx_backtest_runs_start_date", table_name="backtest_runs")
    op.drop_index("idx_backtest_runs_timeframe", table_name="backtest_runs")
    op.drop_index("idx_backtest_runs_exchange", table_name="backtest_runs")
    op.drop_index("idx_backtest_runs_status", table_name="backtest_runs")
    op.drop_index("idx_backtest_runs_strategy_id", table_name="backtest_runs")
    op.drop_index("idx_backtest_runs_user_id", table_name="backtest_runs")
    op.drop_table("backtest_runs")
