"""Initial database schema

Revision ID: 001_initial_schema
Revises:
Create Date: 2024-01-01 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create users table
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("username", sa.String(length=50), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("is_verified", sa.Boolean(), nullable=False),
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
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email"),
        sa.UniqueConstraint("username"),
    )
    op.create_index("idx_users_username", "users", ["username"], unique=False)
    op.create_index("idx_users_email", "users", ["email"], unique=False)
    op.create_index("idx_users_active", "users", ["is_active"], unique=False)
    op.create_check_constraint("username_min_length", "users", "length(username) >= 3")
    op.create_check_constraint("email_min_length", "users", "length(email) >= 5")

    # Create bot_instances table
    op.create_table(
        "bot_instances",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("strategy_type", sa.String(length=50), nullable=False),
        sa.Column("exchange", sa.String(length=50), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("config", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
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
        sa.Column("last_active", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_bot_instances_user_id", "bot_instances", ["user_id"], unique=False)
    op.create_index("idx_bot_instances_strategy_type", "bot_instances", ["strategy_type"], unique=False)
    op.create_index("idx_bot_instances_exchange", "bot_instances", ["exchange"], unique=False)
    op.create_index("idx_bot_instances_status", "bot_instances", ["status"], unique=False)
    op.create_unique_constraint("unique_user_bot_name", "bot_instances", ["user_id", "name"])
    op.create_check_constraint(
        "valid_bot_status", "bot_instances", "status IN ('stopped', 'running', 'paused', 'error')"
    )

    # Create trades table
    op.create_table(
        "trades",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("bot_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("exchange_order_id", sa.String(length=100), nullable=False),
        sa.Column("exchange", sa.String(length=50), nullable=False),
        sa.Column("symbol", sa.String(length=20), nullable=False),
        sa.Column("side", sa.String(length=10), nullable=False),
        sa.Column("order_type", sa.String(length=20), nullable=False),
        sa.Column("quantity", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("price", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("executed_price", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("fee", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("fee_currency", sa.String(length=10), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("pnl", sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("executed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["bot_id"],
            ["bot_instances.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_trades_bot_id", "trades", ["bot_id"], unique=False)
    op.create_index("idx_trades_exchange", "trades", ["exchange"], unique=False)
    op.create_index("idx_trades_symbol", "trades", ["symbol"], unique=False)
    op.create_index("idx_trades_timestamp", "trades", ["timestamp"], unique=False)
    op.create_index("idx_trades_status", "trades", ["status"], unique=False)
    op.create_check_constraint("valid_trade_side", "trades", "side IN ('buy', 'sell')")
    op.create_check_constraint(
        "valid_order_type",
        "trades",
        "order_type IN ('market', 'limit', 'stop_loss', 'take_profit')",
    )
    op.create_check_constraint(
        "valid_trade_status", "trades", "status IN ('pending', 'filled', 'cancelled', 'rejected')"
    )
    op.create_check_constraint("positive_quantity", "trades", "quantity > 0")
    op.create_check_constraint("positive_price", "trades", "price > 0")

    # Create positions table
    op.create_table(
        "positions",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("bot_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("exchange", sa.String(length=50), nullable=False),
        sa.Column("symbol", sa.String(length=20), nullable=False),
        sa.Column("side", sa.String(length=10), nullable=False),
        sa.Column("quantity", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("entry_price", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("current_price", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("unrealized_pnl", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("realized_pnl", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("stop_loss_price", sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column("take_profit_price", sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column("opened_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["bot_id"],
            ["bot_instances.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_positions_bot_id", "positions", ["bot_id"], unique=False)
    op.create_index("idx_positions_exchange", "positions", ["exchange"], unique=False)
    op.create_index("idx_positions_symbol", "positions", ["symbol"], unique=False)
    op.create_index("idx_positions_side", "positions", ["side"], unique=False)
    op.create_unique_constraint("unique_position", "positions", ["bot_id", "exchange", "symbol", "side"])
    op.create_check_constraint("valid_position_side", "positions", "side IN ('long', 'short')")
    op.create_check_constraint("positive_position_quantity", "positions", "quantity > 0")
    op.create_check_constraint("positive_entry_price", "positions", "entry_price > 0")
    op.create_check_constraint("positive_current_price", "positions", "current_price > 0")

    # Create balance_snapshots table
    op.create_table(
        "balance_snapshots",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("exchange", sa.String(length=50), nullable=False),
        sa.Column("currency", sa.String(length=10), nullable=False),
        sa.Column("free_balance", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("locked_balance", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("total_balance", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("btc_value", sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column("usd_value", sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_balance_snapshots_user_id", "balance_snapshots", ["user_id"], unique=False)
    op.create_index("idx_balance_snapshots_exchange", "balance_snapshots", ["exchange"], unique=False)
    op.create_index("idx_balance_snapshots_currency", "balance_snapshots", ["currency"], unique=False)
    op.create_index("idx_balance_snapshots_timestamp", "balance_snapshots", ["timestamp"], unique=False)
    op.create_check_constraint("non_negative_free_balance", "balance_snapshots", "free_balance >= 0")
    op.create_check_constraint("non_negative_locked_balance", "balance_snapshots", "locked_balance >= 0")
    op.create_check_constraint("non_negative_total_balance", "balance_snapshots", "total_balance >= 0")

    # Create strategy_configs table
    op.create_table(
        "strategy_configs",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("strategy_type", sa.String(length=50), nullable=False),
        sa.Column("parameters", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("risk_parameters", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("version", sa.String(length=20), nullable=False),
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
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_strategy_configs_name", "strategy_configs", ["name"], unique=False)
    op.create_index("idx_strategy_configs_type", "strategy_configs", ["strategy_type"], unique=False)
    op.create_index("idx_strategy_configs_active", "strategy_configs", ["is_active"], unique=False)
    op.create_unique_constraint("unique_strategy_config", "strategy_configs", ["name", "version"])

    # Create ml_models table
    op.create_table(
        "ml_models",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("model_type", sa.String(length=50), nullable=False),
        sa.Column("version", sa.String(length=20), nullable=False),
        sa.Column("file_path", sa.String(length=500), nullable=False),
        sa.Column("metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("parameters", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("training_data_range", sa.String(length=100), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
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
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_ml_models_name", "ml_models", ["name"], unique=False)
    op.create_index("idx_ml_models_type", "ml_models", ["model_type"], unique=False)
    op.create_index("idx_ml_models_active", "ml_models", ["is_active"], unique=False)
    op.create_unique_constraint("unique_ml_model", "ml_models", ["name", "version"])

    # Create performance_metrics table
    op.create_table(
        "performance_metrics",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("bot_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("metric_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("total_trades", sa.Integer(), nullable=False),
        sa.Column("winning_trades", sa.Integer(), nullable=False),
        sa.Column("losing_trades", sa.Integer(), nullable=False),
        sa.Column("total_pnl", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("realized_pnl", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("unrealized_pnl", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("win_rate", sa.Numeric(precision=5, scale=4), nullable=False),
        sa.Column("profit_factor", sa.Numeric(precision=10, scale=4), nullable=False),
        sa.Column("sharpe_ratio", sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column("max_drawdown", sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["bot_id"],
            ["bot_instances.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_performance_metrics_bot_id", "performance_metrics", ["bot_id"], unique=False)
    op.create_index("idx_performance_metrics_date", "performance_metrics", ["metric_date"], unique=False)
    op.create_unique_constraint("unique_daily_metrics", "performance_metrics", ["bot_id", "metric_date"])
    op.create_check_constraint("non_negative_total_trades", "performance_metrics", "total_trades >= 0")
    op.create_check_constraint("non_negative_winning_trades", "performance_metrics", "winning_trades >= 0")
    op.create_check_constraint("non_negative_losing_trades", "performance_metrics", "losing_trades >= 0")
    op.create_check_constraint("valid_win_rate", "performance_metrics", "win_rate >= 0 AND win_rate <= 1")
    op.create_check_constraint("non_negative_profit_factor", "performance_metrics", "profit_factor >= 0")

    # Create alerts table
    op.create_table(
        "alerts",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("bot_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("alert_type", sa.String(length=50), nullable=False),
        sa.Column("severity", sa.String(length=20), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("is_read", sa.Boolean(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(
            ["bot_id"],
            ["bot_instances.id"],
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_alerts_user_id", "alerts", ["user_id"], unique=False)
    op.create_index("idx_alerts_bot_id", "alerts", ["bot_id"], unique=False)
    op.create_index("idx_alerts_type", "alerts", ["alert_type"], unique=False)
    op.create_index("idx_alerts_severity", "alerts", ["severity"], unique=False)
    op.create_index("idx_alerts_read", "alerts", ["is_read"], unique=False)
    op.create_index("idx_alerts_timestamp", "alerts", ["timestamp"], unique=False)
    op.create_check_constraint("valid_alert_severity", "alerts", "severity IN ('low', 'medium', 'high', 'critical')")

    # Create audit_logs table
    op.create_table(
        "audit_logs",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("action", sa.String(length=100), nullable=False),
        sa.Column("resource_type", sa.String(length=50), nullable=False),
        sa.Column("resource_id", sa.String(length=100), nullable=True),
        sa.Column("old_value", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("new_value", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("ip_address", sa.String(length=45), nullable=True),
        sa.Column("user_agent", sa.String(length=500), nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_audit_logs_user_id", "audit_logs", ["user_id"], unique=False)
    op.create_index("idx_audit_logs_action", "audit_logs", ["action"], unique=False)
    op.create_index("idx_audit_logs_resource_type", "audit_logs", ["resource_type"], unique=False)
    op.create_index("idx_audit_logs_resource_id", "audit_logs", ["resource_id"], unique=False)
    op.create_index("idx_audit_logs_timestamp", "audit_logs", ["timestamp"], unique=False)


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_table("audit_logs")
    op.drop_table("alerts")
    op.drop_table("performance_metrics")
    op.drop_table("ml_models")
    op.drop_table("strategy_configs")
    op.drop_table("balance_snapshots")
    op.drop_table("positions")
    op.drop_table("trades")
    op.drop_table("bot_instances")
    op.drop_table("users")
