"""
Database migrations for the trading bot framework.

This module provides Alembic-based database migrations for schema management
and version control of the database structure.

CRITICAL: This module integrates with P-001 core framework and will be
used by all subsequent prompts for database schema changes.
"""

from .env import get_url, run_migrations_offline, run_migrations_online
from .script import template

__all__ = ["get_url", "run_migrations_offline", "run_migrations_online", "template"]
