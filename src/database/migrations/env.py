"""
Alembic environment configuration for database migrations.

This module configures Alembic for database migrations with proper
environment handling and model discovery.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Import core components from P-001
from src.core.config import Config
from src.core.logging import get_logger

# Import database models
from src.database.models import Base

logger = get_logger(__name__)

# Initialize Alembic configuration for logging if available
# NOTE: We use try-except to handle cases where context might not be available
try:
    # Interpret the config file for Python logging.
    # This line sets up loggers basically.
    if context.config.config_file_name is not None:
        fileConfig(context.config.config_file_name)
except Exception as e:
    logger.warning(f"Could not initialize Alembic config: {e}")

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def get_url() -> str:
    """Get database URL from configuration."""
    import os

    # Check if we're in testing mode
    if os.getenv("TESTING") == "true":
        # Use the DATABASE_URL environment variable directly for testing
        database_url = os.getenv("DATABASE_URL")
        if database_url:
            logger.info(f"Using test database URL: {database_url}")
            return database_url

    try:
        # Load application config
        app_config = Config()
        return app_config.get_database_url()
    except Exception as e:
        logger.error("Failed to get database URL", error=str(e))
        # Fallback to environment variable
        return os.getenv("DATABASE_URL", "postgresql://user:pass@host:5432/db")


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.
    """
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.
    """
    configuration = context.config.get_section(context.config.config_ini_section)
    configuration["sqlalchemy.url"] = get_url()

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


# Only run migrations when in Alembic context
try:
    if context.is_offline_mode():
        run_migrations_offline()
    else:
        run_migrations_online()
except Exception as e:
    # Not in Alembic context, probably being imported for testing
    import logging

    logger = logging.getLogger(__name__)
    logger.debug(f"Migration not in Alembic context: {e}")
