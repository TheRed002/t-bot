"""
Capital Management Factory Implementation.

Simple factory for creating capital management service instances.
"""

from typing import TYPE_CHECKING, Any

from src.core.base.factory import BaseFactory
from src.core.exceptions import CreationError, ServiceError
from src.core.logging import get_logger

if TYPE_CHECKING:
    from src.capital_management.capital_allocator import CapitalAllocator
    from src.capital_management.currency_manager import CurrencyManager
    from src.capital_management.exchange_distributor import ExchangeDistributor
    from src.capital_management.fund_flow_manager import FundFlowManager
    from src.capital_management.service import CapitalService


class CapitalServiceFactory(BaseFactory["CapitalService"]):
    """Factory for creating CapitalService instances."""

    def __init__(self, dependency_container: Any = None, correlation_id: str | None = None):
        from src.capital_management.service import CapitalService

        super().__init__(product_type=CapitalService, correlation_id=correlation_id)
        self._dependency_container = dependency_container

    def create(
        self, name: str = "", *args, config: dict[str, Any] | None = None, **kwargs
    ) -> "CapitalService":
        """Create CapitalService instance."""
        factory = CapitalManagementFactory(self._dependency_container, self._correlation_id)
        if "correlation_id" not in kwargs:
            kwargs["correlation_id"] = self._correlation_id
        return factory.create_capital_service(**kwargs)


class CapitalAllocatorFactory(BaseFactory["CapitalAllocator"]):
    """Factory for creating CapitalAllocator instances."""

    def __init__(self, dependency_container: Any = None, correlation_id: str | None = None):
        from src.capital_management.capital_allocator import CapitalAllocator

        super().__init__(product_type=CapitalAllocator, correlation_id=correlation_id)
        self._dependency_container = dependency_container

    def create(
        self, name: str = "", *args, config: dict[str, Any] | None = None, **kwargs
    ) -> "CapitalAllocator":
        """Create CapitalAllocator instance."""
        factory = CapitalManagementFactory(self._dependency_container, self._correlation_id)
        if "correlation_id" not in kwargs:
            kwargs["correlation_id"] = self._correlation_id
        return factory.create_capital_allocator(**kwargs)


class CurrencyManagerFactory(BaseFactory["CurrencyManager"]):
    """Factory for creating CurrencyManager instances."""

    def __init__(self, dependency_container: Any = None, correlation_id: str | None = None):
        from src.capital_management.currency_manager import CurrencyManager

        super().__init__(product_type=CurrencyManager, correlation_id=correlation_id)
        self._dependency_container = dependency_container

    def create(
        self, name: str = "", *args, config: dict[str, Any] | None = None, **kwargs
    ) -> "CurrencyManager":
        """Create CurrencyManager instance."""
        factory = CapitalManagementFactory(self._dependency_container, self._correlation_id)
        if "correlation_id" not in kwargs:
            kwargs["correlation_id"] = self._correlation_id
        return factory.create_currency_manager(**kwargs)


class ExchangeDistributorFactory(BaseFactory["ExchangeDistributor"]):
    """Factory for creating ExchangeDistributor instances."""

    def __init__(self, dependency_container: Any = None, correlation_id: str | None = None):
        from src.capital_management.exchange_distributor import ExchangeDistributor

        super().__init__(product_type=ExchangeDistributor, correlation_id=correlation_id)
        self._dependency_container = dependency_container

    def create(
        self, name: str = "", *args, config: dict[str, Any] | None = None, **kwargs
    ) -> "ExchangeDistributor":
        """Create ExchangeDistributor instance."""
        factory = CapitalManagementFactory(self._dependency_container, self._correlation_id)
        if "correlation_id" not in kwargs:
            kwargs["correlation_id"] = self._correlation_id
        return factory.create_exchange_distributor(**kwargs)


class FundFlowManagerFactory(BaseFactory["FundFlowManager"]):
    """Factory for creating FundFlowManager instances."""

    def __init__(self, dependency_container: Any = None, correlation_id: str | None = None):
        from src.capital_management.fund_flow_manager import FundFlowManager

        super().__init__(product_type=FundFlowManager, correlation_id=correlation_id)
        self._dependency_container = dependency_container

    def create(
        self, name: str = "", *args, config: dict[str, Any] | None = None, **kwargs
    ) -> "FundFlowManager":
        """Create FundFlowManager instance."""
        factory = CapitalManagementFactory(self._dependency_container, self._correlation_id)
        if "correlation_id" not in kwargs:
            kwargs["correlation_id"] = self._correlation_id
        return factory.create_fund_flow_manager(**kwargs)


class CapitalManagementFactory:
    """Simple factory for all capital management services."""

    def __init__(self, dependency_container: Any = None, correlation_id: str | None = None):
        """
        Initialize factory.

        Args:
            dependency_container: Dependency injection container (optional)
            correlation_id: Correlation ID for tracking (optional)
        """
        self._dependency_container = dependency_container
        self._correlation_id = correlation_id
        self._logger = get_logger(self.__class__.__name__)

    @property
    def dependency_container(self) -> Any:
        """Get the dependency container."""
        return self._dependency_container

    @property
    def correlation_id(self) -> str | None:
        """Get the correlation ID."""
        return self._correlation_id

    @property
    def capital_service_factory(self) -> "CapitalServiceFactory":
        """Get CapitalService factory."""
        return CapitalServiceFactory(self._dependency_container, self._correlation_id)

    @property
    def capital_allocator_factory(self) -> "CapitalAllocatorFactory":
        """Get CapitalAllocator factory."""
        return CapitalAllocatorFactory(self._dependency_container, self._correlation_id)

    @property
    def currency_manager_factory(self) -> "CurrencyManagerFactory":
        """Get CurrencyManager factory."""
        return CurrencyManagerFactory(self._dependency_container, self._correlation_id)

    @property
    def exchange_distributor_factory(self) -> "ExchangeDistributorFactory":
        """Get ExchangeDistributor factory."""
        return ExchangeDistributorFactory(self._dependency_container, self._correlation_id)

    @property
    def fund_flow_manager_factory(self) -> "FundFlowManagerFactory":
        """Get FundFlowManager factory."""
        return FundFlowManagerFactory(self._dependency_container, self._correlation_id)

    def create_capital_service(self, **kwargs) -> "CapitalService":
        """
        Create CapitalService instance.

        Args:
            **kwargs: Constructor arguments

        Returns:
            CapitalService instance
        """
        try:
            from src.capital_management.service import CapitalService

            capital_repository = None
            audit_repository = None

            if self._dependency_container:
                try:
                    capital_repository = self._dependency_container.get("CapitalRepository")
                except (KeyError, AttributeError, ServiceError):
                    pass
                try:
                    audit_repository = self._dependency_container.get("AuditRepository")
                except (KeyError, AttributeError, ServiceError):
                    pass

            capital_repository = kwargs.get("capital_repository", capital_repository)
            audit_repository = kwargs.get("audit_repository", audit_repository)

            return CapitalService(
                capital_repository=capital_repository,
                audit_repository=audit_repository,
                correlation_id=kwargs.get("correlation_id", self._correlation_id),
            )

        except Exception as e:
            self._logger.error(f"Failed to create CapitalService: {e}")
            raise CreationError(f"Failed to create CapitalService: {e}") from e

    def create_capital_allocator(self, **kwargs) -> "CapitalAllocator":
        """
        Create CapitalAllocator instance.

        Args:
            **kwargs: Constructor arguments

        Returns:
            CapitalAllocator instance
        """
        try:
            from src.capital_management.capital_allocator import CapitalAllocator

            capital_service = None
            config_service = None
            risk_service = None
            trade_lifecycle_manager = None
            validation_service = None

            if self._dependency_container:
                try:
                    capital_service = self._dependency_container.get("CapitalService")
                except (KeyError, AttributeError, ServiceError):
                    pass
                try:
                    config_service = self._dependency_container.get("ConfigService")
                except (KeyError, AttributeError, ServiceError):
                    pass
                try:
                    risk_service = self._dependency_container.get("RiskService")
                except (KeyError, AttributeError, ServiceError):
                    pass
                try:
                    trade_lifecycle_manager = self._dependency_container.get(
                        "TradeLifecycleManager"
                    )
                except (KeyError, AttributeError, ServiceError):
                    pass
                try:
                    validation_service = self._dependency_container.get("ValidationService")
                except (KeyError, AttributeError, ServiceError):
                    pass

            capital_service = kwargs.get("capital_service", capital_service)
            config_service = kwargs.get("config_service", config_service)
            risk_service = kwargs.get("risk_service", risk_service)
            trade_lifecycle_manager = kwargs.get("trade_lifecycle_manager", trade_lifecycle_manager)
            validation_service = kwargs.get("validation_service", validation_service)

            if capital_service is None:
                raise CreationError("CapitalService is required for CapitalAllocator")

            return CapitalAllocator(
                capital_service=capital_service,
                config_service=config_service,
                risk_service=risk_service,
                trade_lifecycle_manager=trade_lifecycle_manager,
                validation_service=validation_service,
                correlation_id=kwargs.get("correlation_id", self._correlation_id),
            )

        except Exception as e:
            self._logger.error(f"Failed to create CapitalAllocator: {e}")
            raise CreationError(f"Failed to create CapitalAllocator: {e}") from e

    def create_currency_manager(self, **kwargs) -> "CurrencyManager":
        """
        Create CurrencyManager instance.

        Args:
            **kwargs: Constructor arguments

        Returns:
            CurrencyManager instance
        """
        try:
            from src.capital_management.currency_manager import CurrencyManager

            exchange_data_service = None
            validation_service = None
            risk_service = None

            if self._dependency_container:
                try:
                    exchange_data_service = self._dependency_container.get("ExchangeDataService")
                except (KeyError, AttributeError, ServiceError):
                    pass
                try:
                    validation_service = self._dependency_container.get("ValidationService")
                except (KeyError, AttributeError, ServiceError):
                    pass
                try:
                    risk_service = self._dependency_container.get("RiskService")
                except (KeyError, AttributeError, ServiceError):
                    pass

            exchange_data_service = kwargs.get("exchange_data_service", exchange_data_service)
            validation_service = kwargs.get("validation_service", validation_service)
            risk_service = kwargs.get("risk_service", risk_service)

            return CurrencyManager(
                exchange_data_service=exchange_data_service,
                validation_service=validation_service,
                risk_service=risk_service,
                correlation_id=kwargs.get("correlation_id", self._correlation_id),
            )

        except Exception as e:
            self._logger.error(f"Failed to create CurrencyManager: {e}")
            raise CreationError(f"Failed to create CurrencyManager: {e}") from e

    def create_exchange_distributor(self, **kwargs) -> "ExchangeDistributor":
        """
        Create ExchangeDistributor instance.

        Args:
            **kwargs: Constructor arguments

        Returns:
            ExchangeDistributor instance
        """
        try:
            from src.capital_management.exchange_distributor import ExchangeDistributor

            exchanges = None
            validation_service = None
            exchange_info_service = None

            if self._dependency_container:
                try:
                    exchanges = self._dependency_container.get("Exchanges")
                except (KeyError, AttributeError, ServiceError):
                    pass
                try:
                    validation_service = self._dependency_container.get("ValidationService")
                except (KeyError, AttributeError, ServiceError):
                    pass
                try:
                    exchange_info_service = self._dependency_container.get("ExchangeInfoService")
                except (KeyError, AttributeError, ServiceError):
                    pass

            exchanges = kwargs.get("exchanges", exchanges)
            validation_service = kwargs.get("validation_service", validation_service)
            exchange_info_service = kwargs.get("exchange_info_service", exchange_info_service)

            return ExchangeDistributor(
                exchanges=exchanges,
                validation_service=validation_service,
                exchange_info_service=exchange_info_service,
                correlation_id=kwargs.get("correlation_id", self._correlation_id),
            )

        except Exception as e:
            self._logger.error(f"Failed to create ExchangeDistributor: {e}")
            raise CreationError(f"Failed to create ExchangeDistributor: {e}") from e

    def create_fund_flow_manager(self, **kwargs) -> "FundFlowManager":
        """
        Create FundFlowManager instance.

        Args:
            **kwargs: Constructor arguments

        Returns:
            FundFlowManager instance
        """
        try:
            from src.capital_management.fund_flow_manager import FundFlowManager

            cache_service = None
            time_series_service = None
            validation_service = None
            capital_allocator = None

            if self._dependency_container:
                try:
                    cache_service = self._dependency_container.get("CacheService")
                except (KeyError, AttributeError, ServiceError):
                    pass
                try:
                    time_series_service = self._dependency_container.get("TimeSeriesService")
                except (KeyError, AttributeError, ServiceError):
                    pass
                try:
                    validation_service = self._dependency_container.get("ValidationService")
                except (KeyError, AttributeError, ServiceError):
                    pass
                try:
                    capital_allocator = self._dependency_container.get("CapitalAllocator")
                except (KeyError, AttributeError, ServiceError):
                    pass

            cache_service = kwargs.get("cache_service", cache_service)
            time_series_service = kwargs.get("time_series_service", time_series_service)
            validation_service = kwargs.get("validation_service", validation_service)
            capital_allocator = kwargs.get("capital_allocator", capital_allocator)

            return FundFlowManager(
                cache_service=cache_service,
                time_series_service=time_series_service,
                validation_service=validation_service,
                capital_allocator=capital_allocator,
                correlation_id=kwargs.get("correlation_id", self._correlation_id),
            )

        except Exception as e:
            self._logger.error(f"Failed to create FundFlowManager: {e}")
            raise CreationError(f"Failed to create FundFlowManager: {e}") from e

    def register_factories(self, container: Any) -> None:
        """
        Register all factories in the dependency injection container.

        Args:
            container: Dependency injection container
        """
        if not container:
            return

        try:
            container.register(
                "CapitalServiceFactory", lambda: self.create_capital_service, singleton=True
            )
            container.register(
                "CapitalAllocatorFactory", lambda: self.create_capital_allocator, singleton=True
            )
            container.register(
                "CurrencyManagerFactory", lambda: self.create_currency_manager, singleton=True
            )
            container.register(
                "ExchangeDistributorFactory",
                lambda: self.create_exchange_distributor,
                singleton=True,
            )
            container.register(
                "FundFlowManagerFactory", lambda: self.create_fund_flow_manager, singleton=True
            )

            self._logger.info("Capital management factories registered successfully")

        except Exception as e:
            self._logger.error(f"Failed to register factories: {e}")
            raise CreationError(f"Failed to register factories: {e}") from e


def create_capital_service(**kwargs) -> "CapitalService":
    """Create CapitalService instance using factory."""
    factory = CapitalManagementFactory()
    return factory.create_capital_service(**kwargs)


def create_capital_allocator(**kwargs) -> "CapitalAllocator":
    """Create CapitalAllocator instance using factory."""
    factory = CapitalManagementFactory()
    return factory.create_capital_allocator(**kwargs)


def create_currency_manager(**kwargs) -> "CurrencyManager":
    """Create CurrencyManager instance using factory."""
    factory = CapitalManagementFactory()
    return factory.create_currency_manager(**kwargs)


def create_exchange_distributor(**kwargs) -> "ExchangeDistributor":
    """Create ExchangeDistributor instance using factory."""
    factory = CapitalManagementFactory()
    return factory.create_exchange_distributor(**kwargs)


def create_fund_flow_manager(**kwargs) -> "FundFlowManager":
    """Create FundFlowManager instance using factory."""
    factory = CapitalManagementFactory()
    return factory.create_fund_flow_manager(**kwargs)
