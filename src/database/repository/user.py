"""User repository implementation."""

from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models.user import User
from src.database.repository.base import DatabaseRepository
from src.database.repository.utils import RepositoryUtils


class UserRepository(DatabaseRepository):
    """Repository for User entities."""

    def __init__(self, session: AsyncSession):
        """Initialize user repository."""

        super().__init__(
            session=session, model=User, entity_type=User, key_type=str, name="UserRepository"
        )

    async def get_by_username(self, username: str) -> User | None:
        """Get user by username."""
        return await self.get_by(username=username)

    async def get_by_email(self, email: str) -> User | None:
        """Get user by email."""
        return await self.get_by(email=email)

    async def get_active_users(self) -> list[User]:
        """Get all active users."""
        return await RepositoryUtils.get_entities_by_field(self, "is_active", True)

    async def get_verified_users(self) -> list[User]:
        """Get all verified users."""
        return await RepositoryUtils.get_entities_by_field(self, "is_verified", True)

    async def get_admin_users(self) -> list[User]:
        """Get all admin users."""
        return await RepositoryUtils.get_entities_by_field(self, "is_admin", True)

    async def activate_user(self, user_id: str) -> bool:
        """Activate a user."""
        return await RepositoryUtils.mark_entity_field(self, user_id, "is_active", True, "User")

    async def deactivate_user(self, user_id: str) -> bool:
        """Deactivate a user."""
        return await RepositoryUtils.mark_entity_field(self, user_id, "is_active", False, "User")

    async def verify_user(self, user_id: str) -> bool:
        """Verify a user."""
        return await RepositoryUtils.mark_entity_field(self, user_id, "is_verified", True, "User")
