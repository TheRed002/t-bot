"""User repository implementation."""

from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models.user import User
from src.database.repository.core_compliant_base import DatabaseRepository


class UserRepository(DatabaseRepository[User, str]):
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
        return await self.get_all(filters={"is_active": True})

    async def get_verified_users(self) -> list[User]:
        """Get all verified users."""
        return await self.get_all(filters={"is_verified": True})

    async def get_admin_users(self) -> list[User]:
        """Get all admin users."""
        return await self.get_all(filters={"is_admin": True})

    async def activate_user(self, user_id: str) -> bool:
        """Activate a user."""
        user = await self.get(user_id)
        if user:
            user.is_active = True
            await self.update(user)
            return True
        return False

    async def deactivate_user(self, user_id: str) -> bool:
        """Deactivate a user."""
        user = await self.get(user_id)
        if user:
            user.is_active = False
            await self.update(user)
            return True
        return False

    async def verify_user(self, user_id: str) -> bool:
        """Verify a user."""
        user = await self.get(user_id)
        if user:
            user.is_verified = True
            await self.update(user)
            return True
        return False
