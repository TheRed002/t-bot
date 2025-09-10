"""Repository factory for creating repository instances with dependency injection."""

from typing import TYPE_CHECKING, Any, TypeVar

from src.core.logging import get_logger
from src.database.interfaces import RepositoryFactoryInterface

if TYPE_CHECKING:
    from src.database.models import Base

logger = get_logger(__name__)

T = TypeVar("T", bound="Base")
R = TypeVar("R")  # Repository type


class RepositoryFactory(RepositoryFactoryInterface):
    """Factory for creating repository instances using dependency injection."""

    def __init__(self, dependency_injector: Any | None = None) -> None:
        """
        Initialize repository factory.

        Args:
            dependency_injector: Optional dependency injector for repository creation
        """
        self._dependency_injector = dependency_injector
        self._registered_repositories: dict[str, type] = {}
        self._logger = logger

    def create_repository(self, repository_class: type[Any], session: Any) -> Any:
        """
        Create repository instance using dependency injection.

        Args:
            repository_class: Repository class to instantiate
            session: Database session to inject

        Returns:
            Repository instance

        Raises:
            RuntimeError: If repository creation fails
        """
        try:
            # Use dependency injector if available
            if self._dependency_injector:
                repo_name = repository_class.__name__
                try:
                    # Try to resolve from dependency injector first
                    return self._dependency_injector.resolve(repo_name)
                except (ImportError, AttributeError, KeyError, TypeError):
                    # Fallback to direct instantiation
                    self._logger.debug(
                        f"DI resolution failed for {repo_name}, using direct instantiation"
                    )

            # Direct instantiation with session
            return repository_class(session)

        except Exception as e:
            self._logger.error(f"Failed to create repository {repository_class.__name__}: {e}")
            raise RuntimeError(
                f"Repository creation failed for {repository_class.__name__}: {e}"
            ) from e

    def register_repository(self, name: str, repository_class: type[R]) -> None:
        """
        Register repository class for factory creation.

        Args:
            name: Repository name for registration
            repository_class: Repository class to register
        """
        self._registered_repositories[name] = repository_class
        self._logger.debug(f"Registered repository: {name}")

    def is_repository_registered(self, name: str) -> bool:
        """
        Check if repository is registered.

        Args:
            name: Repository name to check

        Returns:
            True if repository is registered
        """
        return name in self._registered_repositories

    def get_registered_repository(self, name: str) -> type[R] | None:
        """
        Get registered repository class by name.

        Args:
            name: Repository name

        Returns:
            Repository class or None if not found
        """
        return self._registered_repositories.get(name)

    def configure_dependencies(self, dependency_injector) -> None:
        """
        Configure dependency injection for repository creation.

        Args:
            dependency_injector: Dependency injector to use
        """
        self._dependency_injector = dependency_injector
        self._logger.info("Dependencies configured for RepositoryFactory")

    def list_registered_repositories(self) -> list[str]:
        """
        List all registered repository names.

        Returns:
            List of registered repository names
        """
        return list(self._registered_repositories.keys())

    def clear_registrations(self) -> None:
        """Clear all repository registrations."""
        self._registered_repositories.clear()
        self._logger.info("All repository registrations cleared")
