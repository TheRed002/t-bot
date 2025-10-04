"""merge_heads

Revision ID: 9449369979fc
Revises: 004_optimization_models, 9dbcb25ea329
Create Date: 2025-09-30 20:02:54.983837

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "9449369979fc"
down_revision = ("004_optimization_models", "9dbcb25ea329")
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
