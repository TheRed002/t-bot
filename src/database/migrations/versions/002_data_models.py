"""Add data management models

Revision ID: 002
Revises: 001
Create Date: 2024-01-15 10:00:00.000000

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "002_data_models"
down_revision = "001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create market_data_records table
    op.create_table(
        "market_data_records",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("symbol", sa.String(length=20), nullable=False),
        sa.Column("exchange", sa.String(length=50), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("open_price", sa.Float(), nullable=True),
        sa.Column("high_price", sa.Float(), nullable=True),
        sa.Column("low_price", sa.Float(), nullable=True),
        sa.Column("close_price", sa.Float(), nullable=True),
        sa.Column("price", sa.Float(), nullable=True),
        sa.Column("volume", sa.Float(), nullable=True),
        sa.Column("quote_volume", sa.Float(), nullable=True),
        sa.Column("trades_count", sa.Integer(), nullable=True),
        sa.Column("bid", sa.Float(), nullable=True),
        sa.Column("ask", sa.Float(), nullable=True),
        sa.Column("bid_volume", sa.Float(), nullable=True),
        sa.Column("ask_volume", sa.Float(), nullable=True),
        sa.Column("data_source", sa.String(length=100), nullable=False),
        sa.Column("quality_score", sa.Float(), nullable=True),
        sa.Column("validation_status", sa.String(length=20), nullable=False),
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

    # Create indexes for market_data_records
    op.create_index(
        "idx_market_data_symbol_timestamp",
        "market_data_records",
        ["symbol", "timestamp"],
        unique=False,
    )
    op.create_index(
        "idx_market_data_exchange_timestamp",
        "market_data_records",
        ["exchange", "timestamp"],
        unique=False,
    )
    op.create_index(
        "idx_market_data_quality", "market_data_records", ["quality_score"], unique=False
    )
    op.create_index(
        "idx_market_data_validation", "market_data_records", ["validation_status"], unique=False
    )
    op.create_unique_constraint(
        "uq_market_data_unique", "market_data_records", ["symbol", "exchange", "timestamp"]
    )

    # Create feature_records table
    op.create_table(
        "feature_records",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("symbol", sa.String(length=20), nullable=False),
        sa.Column("feature_type", sa.String(length=100), nullable=False),
        sa.Column("feature_name", sa.String(length=100), nullable=False),
        sa.Column("calculation_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("feature_value", sa.Float(), nullable=False),
        sa.Column("confidence_score", sa.Float(), nullable=True),
        sa.Column("lookback_period", sa.Integer(), nullable=True),
        sa.Column("parameters", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("calculation_method", sa.String(length=100), nullable=False),
        sa.Column("source_data_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("source_data_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes for feature_records
    op.create_index(
        "idx_feature_symbol_type", "feature_records", ["symbol", "feature_type"], unique=False
    )
    op.create_index(
        "idx_feature_timestamp", "feature_records", ["calculation_timestamp"], unique=False
    )
    op.create_index("idx_feature_name", "feature_records", ["feature_name"], unique=False)
    op.create_unique_constraint(
        "uq_feature_unique",
        "feature_records",
        ["symbol", "feature_type", "feature_name", "calculation_timestamp"],
    )

    # Create data_quality_records table
    op.create_table(
        "data_quality_records",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("symbol", sa.String(length=20), nullable=False),
        sa.Column("data_source", sa.String(length=100), nullable=False),
        sa.Column("quality_check_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completeness_score", sa.Float(), nullable=False),
        sa.Column("accuracy_score", sa.Float(), nullable=False),
        sa.Column("consistency_score", sa.Float(), nullable=False),
        sa.Column("timeliness_score", sa.Float(), nullable=False),
        sa.Column("overall_score", sa.Float(), nullable=False),
        sa.Column("missing_data_count", sa.Integer(), nullable=False),
        sa.Column("outlier_count", sa.Integer(), nullable=False),
        sa.Column("duplicate_count", sa.Integer(), nullable=False),
        sa.Column("validation_errors", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("check_type", sa.String(length=50), nullable=False),
        sa.Column("data_period_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("data_period_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes for data_quality_records
    op.create_index(
        "idx_quality_symbol_timestamp",
        "data_quality_records",
        ["symbol", "quality_check_timestamp"],
        unique=False,
    )
    op.create_index(
        "idx_quality_source_timestamp",
        "data_quality_records",
        ["data_source", "quality_check_timestamp"],
        unique=False,
    )
    op.create_index(
        "idx_quality_overall_score", "data_quality_records", ["overall_score"], unique=False
    )
    op.create_index("idx_quality_check_type", "data_quality_records", ["check_type"], unique=False)

    # Create data_pipeline_records table
    op.create_table(
        "data_pipeline_records",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("pipeline_name", sa.String(length=100), nullable=False),
        sa.Column("execution_id", sa.String(length=100), nullable=False),
        sa.Column("execution_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("stage", sa.String(length=50), nullable=False),
        sa.Column("records_processed", sa.Integer(), nullable=False),
        sa.Column("records_successful", sa.Integer(), nullable=False),
        sa.Column("records_failed", sa.Integer(), nullable=False),
        sa.Column("processing_time_ms", sa.Integer(), nullable=True),
        sa.Column("error_count", sa.Integer(), nullable=False),
        sa.Column("error_messages", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("configuration", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("dependencies", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
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

    # Create indexes for data_pipeline_records
    op.create_index(
        "idx_pipeline_name_timestamp",
        "data_pipeline_records",
        ["pipeline_name", "execution_timestamp"],
        unique=False,
    )
    op.create_index("idx_pipeline_status", "data_pipeline_records", ["status"], unique=False)
    op.create_index("idx_pipeline_stage", "data_pipeline_records", ["stage"], unique=False)
    op.create_index(
        "idx_pipeline_execution_id", "data_pipeline_records", ["execution_id"], unique=False
    )


def downgrade() -> None:
    # Drop data_pipeline_records table
    op.drop_index("idx_pipeline_execution_id", table_name="data_pipeline_records")
    op.drop_index("idx_pipeline_stage", table_name="data_pipeline_records")
    op.drop_index("idx_pipeline_status", table_name="data_pipeline_records")
    op.drop_index("idx_pipeline_name_timestamp", table_name="data_pipeline_records")
    op.drop_table("data_pipeline_records")

    # Drop data_quality_records table
    op.drop_index("idx_quality_check_type", table_name="data_quality_records")
    op.drop_index("idx_quality_overall_score", table_name="data_quality_records")
    op.drop_index("idx_quality_source_timestamp", table_name="data_quality_records")
    op.drop_index("idx_quality_symbol_timestamp", table_name="data_quality_records")
    op.drop_table("data_quality_records")

    # Drop feature_records table
    op.drop_unique_constraint("uq_feature_unique", table_name="feature_records")
    op.drop_index("idx_feature_name", table_name="feature_records")
    op.drop_index("idx_feature_timestamp", table_name="feature_records")
    op.drop_index("idx_feature_symbol_type", table_name="feature_records")
    op.drop_table("feature_records")

    # Drop market_data_records table
    op.drop_unique_constraint("uq_market_data_unique", table_name="market_data_records")
    op.drop_index("idx_market_data_validation", table_name="market_data_records")
    op.drop_index("idx_market_data_quality", table_name="market_data_records")
    op.drop_index("idx_market_data_exchange_timestamp", table_name="market_data_records")
    op.drop_index("idx_market_data_symbol_timestamp", table_name="market_data_records")
    op.drop_table("market_data_records")
