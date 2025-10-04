"""add_default_status_to_position

Revision ID: 249c01341fc2
Revises: 9449369979fc
Create Date: 2025-09-30 20:03:04.694244

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "249c01341fc2"
down_revision = "9449369979fc"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add default value to status column for positions table
    op.execute("ALTER TABLE positions ALTER COLUMN status SET DEFAULT 'OPEN'")

    # Update any existing NULL status values to 'OPEN' (if any exist)
    op.execute("UPDATE positions SET status = 'OPEN' WHERE status IS NULL")


def downgrade() -> None:
    # Remove default value from status column
    op.execute("ALTER TABLE positions ALTER COLUMN status DROP DEFAULT")
