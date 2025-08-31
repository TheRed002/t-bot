"""
Configuration Migration Helper for T-Bot Trading System.

This module provides utilities to help migrate from legacy get_config()
pattern to modern ConfigService dependency injection pattern.

Usage:
    ```python
    # In application startup
    from src.core.config.migration import setup_config_service


    async def main():
        await setup_config_service("config/main.yaml")

        # Now all modules can use ConfigService via dependency injection
        app = create_app()
        await app.start()
    ```
"""

import logging
from pathlib import Path

from .service import ConfigService, register_config_service_in_container

logger = logging.getLogger(__name__)


async def setup_config_service(
    config_file: str | Path | None = None,
    enable_hot_reload: bool = False,
    hot_reload_interval: int = 30,
) -> ConfigService:
    """
    Setup ConfigService for the entire application.

    This function should be called during application startup to initialize
    and register ConfigService in the dependency injection container.

    Args:
        config_file: Path to configuration file
        enable_hot_reload: Enable hot reloading of configuration
        hot_reload_interval: Interval in seconds for hot reload checks

    Returns:
        Initialized ConfigService instance

    Example:
        ```python
        # In main.py or app initialization
        config_service = await setup_config_service(
            config_file="config/production.yaml", enable_hot_reload=True
        )
        ```
    """
    logger.info("Setting up ConfigService for application")

    # Create and initialize ConfigService
    service = ConfigService(
        enable_hot_reload=enable_hot_reload, hot_reload_interval=hot_reload_interval
    )

    await service.initialize(config_file=config_file)

    # Register in dependency injection container
    register_config_service_in_container(
        config_file=str(config_file) if config_file else None, enable_hot_reload=enable_hot_reload
    )

    logger.info("ConfigService setup complete - available for dependency injection")
    return service


def migrate_legacy_config_usage() -> None:
    """
    Print migration guide for updating legacy config usage.

    This function helps developers understand how to migrate from
    legacy get_config() pattern to modern ConfigService pattern.
    """
    migration_guide = """
    Configuration Migration Guide
    ============================

    LEGACY PATTERN (to be replaced):
    ```python
    from src.core.config import get_config

    def __init__(self):
        config = get_config()
        self.db_url = config.database.postgresql_url
    ```

    MODERN PATTERN (recommended):
    ```python
    from src.core.dependency_injection import get_container

    def __init__(self, config_service = None):
        if config_service is None:
            config_service = get_container().get("ConfigService")

        db_config = config_service.get_database_config()
        self.db_url = db_config.postgresql_url
    ```

    OR with dependency injection:
    ```python
    from src.core.config.service import ConfigService

    def __init__(self, config_service: ConfigService):
        db_config = config_service.get_database_config()
        self.db_url = db_config.postgresql_url
    ```

    CONFIGURATION ACCESS PATTERNS:
    - Database: config_service.get_database_config()
    - Exchange: config_service.get_exchange_config()
    - Risk: config_service.get_risk_config()
    - Strategy: config_service.get_strategy_config()
    - App: config_service.get_app_config()
    - Specific value: config_service.get_config_value("path.to.value", default)

    BENEFITS:
    - ✅ No global state dependencies
    - ✅ Better testability with dependency injection
    - ✅ Configuration caching and validation
    - ✅ Hot-reload capability
    - ✅ Type-safe configuration access
    - ✅ Configuration change notifications
    """

    from src.core.logging import get_logger

    migration_logger = get_logger(__name__)
    migration_logger.info("Configuration migration guide", guide=migration_guide)


def validate_migration_status() -> dict[str, bool]:
    """
    Validate the migration status of the configuration system.

    Returns:
        Dictionary with validation results
    """
    results = {
        "config_service_available": False,
        "dependency_injection_setup": False,
        "legacy_fallbacks_working": False,
    }

    try:
        # Check if ConfigService can be imported
        results["config_service_available"] = True

        # Check if dependency injection is setup
        from ..dependency_injection import get_container

        container = get_container()
        if container.has("ConfigService"):
            results["dependency_injection_setup"] = True

        # Check if legacy fallbacks work
        from .main import get_config

        config = get_config()
        if config is not None:
            results["legacy_fallbacks_working"] = True

    except Exception as e:
        logger.error(f"Migration validation failed: {e}")

    return results


async def test_migration() -> None:
    """
    Test the migration by exercising both legacy and modern patterns.

    This function can be used to verify that the migration was successful
    and both patterns work correctly.
    """
    logger.info("Testing configuration migration...")

    # Test legacy pattern
    try:
        from .main import get_config

        legacy_config = get_config()
        assert legacy_config is not None
        logger.info("✅ Legacy get_config() pattern working")
    except Exception as e:
        logger.error(f"❌ Legacy pattern failed: {e}")

    # Test modern pattern
    try:
        service = ConfigService()
        await service.initialize()

        db_config = service.get_database_config()
        assert db_config is not None

        await service.shutdown()
        logger.info("✅ Modern ConfigService pattern working")
    except Exception as e:
        logger.error(f"❌ Modern pattern failed: {e}")

    # Test dependency injection
    try:
        from ..dependency_injection import get_container

        register_config_service_in_container()

        container = get_container()
        service = await container.get("ConfigService")
        assert service is not None

        await service.shutdown()
        logger.info("✅ Dependency injection pattern working")
    except Exception as e:
        logger.error(f"❌ Dependency injection failed: {e}")

    logger.info("Migration test completed")


# Migration helper functions for specific patterns
def create_service_with_config_injection():
    """
    Example of how to create a service with ConfigService dependency injection.

    This shows the recommended pattern for new services.
    """
    example_code = """
from src.core.config.service import ConfigService

class MyService:
    def __init__(self, config_service: ConfigService):
        self.config_service = config_service

    async def initialize(self):
        # Access configuration through service
        db_config = self.config_service.get_database_config()
        exchange_config = self.config_service.get_exchange_config()

        # Use configurations...
"""
    return example_code


def create_service_with_fallback_injection():
    """
    Example of how to create a service with fallback config injection.

    This shows the migration pattern that supports both legacy and modern usage.
    """
    example_code = """
from src.core.dependency_injection import get_container
from src.core.config.service import ConfigService

class MyService:
    def __init__(self, config_service: ConfigService = None):
        if config_service is None:
            # Try to get from dependency injection
            try:
                config_service = get_container().get("ConfigService")
            except (KeyError, AttributeError):
                # Fallback to legacy pattern
                from src.core.config import get_config
                legacy_config = get_config()
                # Create ConfigService from legacy config or use legacy directly

        self.config_service = config_service

    async def get_database_config(self):
        if hasattr(self.config_service, 'get_database_config'):
            return self.config_service.get_database_config()
        else:
            # Handle legacy config object
            return self.config_service.database
"""
    return example_code
