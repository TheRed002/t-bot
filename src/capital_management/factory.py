"""
Capital Management Factory Implementation.

This module provides a factory pattern for creating capital management service instances
with proper dependency injection, configuration management, and lifecycle support.
"""

from typing import TYPE_CHECKING, Any

from src.core.base.factory import BaseFactory
from src.core.exceptions import CreationError

if TYPE_CHECKING:
    from src.capital_management.capital_allocator import CapitalAllocator
    from src.capital_management.currency_manager import CurrencyManager
    from src.capital_management.exchange_distributor import ExchangeDistributor
    from src.capital_management.fund_flow_manager import FundFlowManager
    from src.capital_management.service import CapitalService


class CapitalServiceFactory(BaseFactory["CapitalService"]):
    """Factory for creating CapitalService instances."""

    def __init__(self, dependency_container: Any = None, correlation_id: str | None = None):
        """
        Initialize capital service factory.

        Args:
            dependency_container: Dependency injection container
            correlation_id: Request correlation ID
        """
        # Import at runtime to avoid circular imports
        from src.capital_management.service import CapitalService

        super().__init__(
            product_type=CapitalService,
            name="CapitalServiceFactory",
            correlation_id=correlation_id,
        )

        if dependency_container:
            self.configure_dependencies(dependency_container)

        # Register default creator
        self.register(
            "default",
            self._create_capital_service,
            singleton=True,
        )

    def _create_capital_service(
        self,
        capital_repository: Any = None,
        audit_repository: Any = None,
        state_service: Any = None,
        correlation_id: str | None = None,
    ) -> "CapitalService":
        """
        Create CapitalService instance with dependency injection.

        Args:
            capital_repository: Capital repository instance (optional)
            audit_repository: Audit repository instance (optional)
            state_service: State service instance (optional)
            correlation_id: Request correlation ID (optional)

        Returns:
            CapitalService instance
        """
        # Get optional dependencies from DI container if not provided
        if self._dependency_container:
            if capital_repository is None:
                try:
                    capital_repository = self._dependency_container.get("CapitalRepository")
                except Exception:
                    pass  # Optional dependency

            if audit_repository is None:
                try:
                    audit_repository = self._dependency_container.get("AuditRepository")
                except Exception:
                    pass  # Optional dependency

            if state_service is None:
                try:
                    state_service = self._dependency_container.get("StateService")
                except Exception:
                    pass  # Optional dependency

        # Import at runtime to avoid circular imports
        from src.capital_management.service import CapitalService

        return CapitalService(
            capital_repository=capital_repository,
            audit_repository=audit_repository,
            state_service=state_service,
            correlation_id=correlation_id or self._correlation_id,
        )


class CapitalAllocatorFactory(BaseFactory["CapitalAllocator"]):
    """Factory for creating CapitalAllocator instances."""

    def __init__(self, dependency_container: Any = None, correlation_id: str | None = None):
        """
        Initialize capital allocator factory.

        Args:
            dependency_container: Dependency injection container
            correlation_id: Request correlation ID
        """
        # Import at runtime to avoid circular imports
        from src.capital_management.capital_allocator import CapitalAllocator

        super().__init__(
            product_type=CapitalAllocator,
            name="CapitalAllocatorFactory",
            correlation_id=correlation_id,
        )

        if dependency_container:
            self.configure_dependencies(dependency_container)

        # Register default creator
        self.register(
            "default",
            self._create_capital_allocator,
            singleton=True,
        )

    def _create_capital_allocator(
        self,
        capital_service: "CapitalService | None" = None,
        config_service: Any = None,
        risk_manager: Any = None,
        trade_lifecycle_manager: Any = None,
        validation_service: Any = None,
    ) -> "CapitalAllocator":
        """
        Create CapitalAllocator instance with dependency injection.

        Args:
            capital_service: Capital service instance (required)
            config_service: Config service instance (optional)
            risk_manager: Risk manager instance (optional)
            trade_lifecycle_manager: Trade lifecycle manager instance (optional)
            validation_service: Validation service instance (optional)

        Returns:
            CapitalAllocator instance

        Raises:
            CreationError: If capital_service is not provided or available
        """
        # Get dependencies from DI container if not provided
        if self._dependency_container:
            if capital_service is None:
                try:
                    capital_service = self._dependency_container.get("CapitalService")
                except Exception as e:
                    raise CreationError(f"CapitalService is required but not available: {e}") from e

            if config_service is None:
                try:
                    config_service = self._dependency_container.get("ConfigService")
                except Exception:
                    pass  # Optional dependency

            if risk_manager is None:
                try:
                    risk_manager = self._dependency_container.get("RiskService")
                except Exception:
                    try:
                        risk_manager = self._dependency_container.get("RiskManager")
                    except Exception:
                        pass  # Optional dependency

            if trade_lifecycle_manager is None:
                try:
                    trade_lifecycle_manager = self._dependency_container.get(
                        "TradeLifecycleManager"
                    )
                except Exception:
                    pass  # Optional dependency

            if validation_service is None:
                try:
                    validation_service = self._dependency_container.get("ValidationService")
                except Exception:
                    pass  # Optional dependency

        if capital_service is None:
            raise CreationError("CapitalService is required for CapitalAllocator")

        # Import at runtime to avoid circular imports
        from src.capital_management.capital_allocator import CapitalAllocator

        return CapitalAllocator(
            capital_service=capital_service,
            config_service=config_service,
            risk_manager=risk_manager,
            trade_lifecycle_manager=trade_lifecycle_manager,
            validation_service=validation_service,
        )


class CurrencyManagerFactory(BaseFactory["CurrencyManager"]):
    """Factory for creating CurrencyManager instances."""

    def __init__(self, dependency_container: Any = None, correlation_id: str | None = None):
        """
        Initialize currency manager factory.

        Args:
            dependency_container: Dependency injection container
            correlation_id: Request correlation ID
        """
        # Import at runtime to avoid circular imports
        from src.capital_management.currency_manager import CurrencyManager

        super().__init__(
            product_type=CurrencyManager,
            name="CurrencyManagerFactory",
            correlation_id=correlation_id,
        )

        if dependency_container:
            self.configure_dependencies(dependency_container)

        # Register default creator
        self.register(
            "default",
            self._create_currency_manager,
            singleton=True,
        )

    def _create_currency_manager(
        self,
        exchange_data_service: Any = None,
        validation_service: Any = None,
        correlation_id: str | None = None,
    ) -> "CurrencyManager":
        """
        Create CurrencyManager instance with dependency injection.

        Args:
            exchange_data_service: Exchange data service instance (optional)
            validation_service: Validation service instance (optional)
            correlation_id: Request correlation ID (optional)

        Returns:
            CurrencyManager instance
        """
        # Get optional dependencies from DI container if not provided
        if self._dependency_container:
            if exchange_data_service is None:
                try:
                    exchange_data_service = self._dependency_container.get("ExchangeDataService")
                except Exception:
                    pass  # Optional dependency

            if validation_service is None:
                try:
                    validation_service = self._dependency_container.get("ValidationService")
                except Exception:
                    pass  # Optional dependency

        # Import at runtime to avoid circular imports
        from src.capital_management.currency_manager import CurrencyManager

        return CurrencyManager(
            exchange_data_service=exchange_data_service,
            validation_service=validation_service,
            correlation_id=correlation_id or self._correlation_id,
        )


class ExchangeDistributorFactory(BaseFactory["ExchangeDistributor"]):
    """Factory for creating ExchangeDistributor instances."""

    def __init__(self, dependency_container: Any = None, correlation_id: str | None = None):
        """
        Initialize exchange distributor factory.

        Args:
            dependency_container: Dependency injection container
            correlation_id: Request correlation ID
        """
        # Import at runtime to avoid circular imports
        from src.capital_management.exchange_distributor import ExchangeDistributor

        super().__init__(
            product_type=ExchangeDistributor,
            name="ExchangeDistributorFactory",
            correlation_id=correlation_id,
        )

        if dependency_container:
            self.configure_dependencies(dependency_container)

        # Register default creator
        self.register(
            "default",
            self._create_exchange_distributor,
            singleton=True,
        )

    def _create_exchange_distributor(
        self,
        exchanges: dict[str, Any] | None = None,
        validation_service: Any = None,
        correlation_id: str | None = None,
    ) -> "ExchangeDistributor":
        """
        Create ExchangeDistributor instance with dependency injection.

        Args:
            exchanges: Dictionary of exchange instances (optional)
            validation_service: Validation service instance (optional)
            correlation_id: Request correlation ID (optional)

        Returns:
            ExchangeDistributor instance
        """
        # Get optional dependencies from DI container if not provided
        if self._dependency_container:
            if exchanges is None:
                try:
                    exchanges = self._dependency_container.get("ExchangeRegistry")
                except Exception:
                    exchanges = {}  # Use empty dict as fallback

            if validation_service is None:
                try:
                    validation_service = self._dependency_container.get("ValidationService")
                except Exception:
                    pass  # Optional dependency

        # Import at runtime to avoid circular imports
        from src.capital_management.exchange_distributor import ExchangeDistributor

        return ExchangeDistributor(
            exchanges=exchanges,
            validation_service=validation_service,
            correlation_id=correlation_id or self._correlation_id,
        )


class FundFlowManagerFactory(BaseFactory["FundFlowManager"]):
    """Factory for creating FundFlowManager instances."""

    def __init__(self, dependency_container: Any = None, correlation_id: str | None = None):
        """
        Initialize fund flow manager factory.

        Args:
            dependency_container: Dependency injection container
            correlation_id: Request correlation ID
        """
        # Import at runtime to avoid circular imports
        from src.capital_management.fund_flow_manager import FundFlowManager

        super().__init__(
            product_type=FundFlowManager,
            name="FundFlowManagerFactory",
            correlation_id=correlation_id,
        )

        if dependency_container:
            self.configure_dependencies(dependency_container)

        # Register default creator
        self.register(
            "default",
            self._create_fund_flow_manager,
            singleton=True,
        )

    def _create_fund_flow_manager(
        self,
        cache_service: Any = None,
        time_series_service: Any = None,
        validation_service: Any = None,
        correlation_id: str | None = None,
    ) -> "FundFlowManager":
        """
        Create FundFlowManager instance with dependency injection.

        Args:
            cache_service: Cache service instance (optional)
            time_series_service: Time series service instance (optional)
            validation_service: Validation service instance (optional)
            correlation_id: Request correlation ID (optional)

        Returns:
            FundFlowManager instance
        """
        # Get optional dependencies from DI container if not provided
        if self._dependency_container:
            if cache_service is None:
                try:
                    cache_service = self._dependency_container.get("CacheService")
                except Exception:
                    pass  # Optional dependency

            if time_series_service is None:
                try:
                    time_series_service = self._dependency_container.get("TimeSeriesService")
                except Exception:
                    pass  # Optional dependency

            if validation_service is None:
                try:
                    validation_service = self._dependency_container.get("ValidationService")
                except Exception:
                    pass  # Optional dependency

        # Import at runtime to avoid circular imports
        from src.capital_management.fund_flow_manager import FundFlowManager

        return FundFlowManager(
            cache_service=cache_service,
            time_series_service=time_series_service,
            validation_service=validation_service,
            correlation_id=correlation_id or self._correlation_id,
        )


class CapitalManagementFactory:
    """
    Composite factory for all capital management services.

    Provides a single entry point for creating all capital management
    service instances with proper dependency resolution.
    """

    def __init__(self, dependency_container: Any = None, correlation_id: str | None = None):
        """
        Initialize capital management factory.

        Args:
            dependency_container: Dependency injection container
            correlation_id: Request correlation ID
        """
        self.dependency_container = dependency_container
        self.correlation_id = correlation_id

        # Initialize sub-factories
        self.capital_service_factory = CapitalServiceFactory(dependency_container, correlation_id)
        self.capital_allocator_factory = CapitalAllocatorFactory(
            dependency_container, correlation_id
        )
        self.currency_manager_factory = CurrencyManagerFactory(dependency_container, correlation_id)
        self.exchange_distributor_factory = ExchangeDistributorFactory(
            dependency_container, correlation_id
        )
        self.fund_flow_manager_factory = FundFlowManagerFactory(
            dependency_container, correlation_id
        )

    def create_capital_service(self, **kwargs) -> "CapitalService":
        """Create CapitalService instance."""
        return self.capital_service_factory.create("default", **kwargs)

    def create_capital_allocator(self, **kwargs) -> "CapitalAllocator":
        """Create CapitalAllocator instance."""
        return self.capital_allocator_factory.create("default", **kwargs)

    def create_currency_manager(self, **kwargs) -> "CurrencyManager":
        """Create CurrencyManager instance."""
        return self.currency_manager_factory.create("default", **kwargs)

    def create_exchange_distributor(self, **kwargs) -> "ExchangeDistributor":
        """Create ExchangeDistributor instance."""
        return self.exchange_distributor_factory.create("default", **kwargs)

    def create_fund_flow_manager(self, **kwargs) -> "FundFlowManager":
        """Create FundFlowManager instance."""
        return self.fund_flow_manager_factory.create("default", **kwargs)

    def register_factories(self, container: Any) -> None:
        """
        Register all factories with the dependency injection container.

        Args:
            container: Dependency injection container
        """
        container.register(
            "CapitalServiceFactory", lambda: self.capital_service_factory, singleton=True
        )
        container.register(
            "CapitalAllocatorFactory", lambda: self.capital_allocator_factory, singleton=True
        )
        container.register(
            "CurrencyManagerFactory", lambda: self.currency_manager_factory, singleton=True
        )
        container.register(
            "ExchangeDistributorFactory", lambda: self.exchange_distributor_factory, singleton=True
        )
        container.register(
            "FundFlowManagerFactory", lambda: self.fund_flow_manager_factory, singleton=True
        )
        container.register("CapitalManagementFactory", lambda: self, singleton=True)
