"""
Authentication models for T-Bot Trading System.

This module defines the core authentication and authorization models
including users, roles, permissions, and tokens.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any


class PermissionType(Enum):
    """Permission types for role-based access control."""

    # Basic permissions
    READ = "read"
    WRITE = "write"
    DELETE = "delete"

    # Trading permissions
    PLACE_ORDER = "place_order"
    CANCEL_ORDER = "cancel_order"
    MODIFY_ORDER = "modify_order"
    VIEW_PORTFOLIO = "view_portfolio"

    # Bot management permissions
    CREATE_BOT = "create_bot"
    START_BOT = "start_bot"
    STOP_BOT = "stop_bot"
    DELETE_BOT = "delete_bot"
    CONFIGURE_BOT = "configure_bot"

    # Risk management permissions
    SET_RISK_LIMITS = "set_risk_limits"
    OVERRIDE_RISK = "override_risk"
    VIEW_RISK_METRICS = "view_risk_metrics"

    # Strategy permissions
    CREATE_STRATEGY = "create_strategy"
    MODIFY_STRATEGY = "modify_strategy"
    DEPLOY_STRATEGY = "deploy_strategy"

    # Administrative permissions
    MANAGE_USERS = "manage_users"
    VIEW_SYSTEM_LOGS = "view_system_logs"
    SYSTEM_CONFIG = "system_config"

    # WebSocket permissions
    WEBSOCKET_CONNECT = "websocket_connect"
    WEBSOCKET_MARKET_DATA = "websocket_market_data"
    WEBSOCKET_BOT_STATUS = "websocket_bot_status"
    WEBSOCKET_PORTFOLIO = "websocket_portfolio"


@dataclass
class Permission:
    """Represents a system permission."""

    name: str
    permission_type: PermissionType
    description: str
    resource: str | None = None  # Specific resource this applies to

    def __str__(self) -> str:
        return (
            f"{self.permission_type.value}:{self.resource}"
            if self.resource
            else self.permission_type.value
        )

    def __hash__(self) -> int:
        return hash((self.permission_type, self.resource))


@dataclass
class Role:
    """Represents a user role with associated permissions."""

    name: str
    description: str
    permissions: set[Permission] = field(default_factory=set)
    is_system_role: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __hash__(self) -> int:
        """Make Role hashable so it can be stored in sets."""
        return hash(self.name)

    def __eq__(self, other) -> bool:
        """Role equality based on name."""
        if not isinstance(other, Role):
            return False
        return self.name == other.name

    def add_permission(self, permission: Permission) -> None:
        """Add a permission to this role."""
        self.permissions.add(permission)

    def remove_permission(self, permission: Permission) -> None:
        """Remove a permission from this role."""
        self.permissions.discard(permission)

    def has_permission(self, permission_type: PermissionType, resource: str | None = None) -> bool:
        """Check if this role has a specific permission."""
        for perm in self.permissions:
            if perm.permission_type == permission_type:
                if resource is None or perm.resource is None or perm.resource == resource:
                    return True
        return False

    def get_permissions_by_type(self, permission_type: PermissionType) -> list[Permission]:
        """Get all permissions of a specific type."""
        return [p for p in self.permissions if p.permission_type == permission_type]


class UserStatus(Enum):
    """User account status."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    LOCKED = "locked"
    PENDING_VERIFICATION = "pending_verification"


@dataclass
class User:
    """Represents a system user."""

    user_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    username: str = ""
    email: str = ""
    full_name: str = ""
    status: UserStatus = UserStatus.PENDING_VERIFICATION
    roles: set[Role] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_login: datetime | None = None
    login_attempts: int = 0
    max_login_attempts: int = 5
    lockout_until: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Trading specific attributes
    allocated_capital: float = 0.0
    max_daily_loss: float = 0.0
    risk_level: str = "medium"

    def add_role(self, role: Role) -> None:
        """Add a role to this user."""
        self.roles.add(role)

    def remove_role(self, role: Role) -> None:
        """Remove a role from this user."""
        self.roles.discard(role)

    def has_role(self, role_name: str) -> bool:
        """Check if user has a specific role."""
        return any(role.name == role_name for role in self.roles)

    def has_permission(self, permission_type: PermissionType, resource: str | None = None) -> bool:
        """Check if user has a specific permission through their roles."""
        for role in self.roles:
            if role.has_permission(permission_type, resource):
                return True
        return False

    def get_all_permissions(self) -> set[Permission]:
        """Get all permissions from all roles."""
        all_permissions = set()
        for role in self.roles:
            all_permissions.update(role.permissions)
        return all_permissions

    def is_locked(self) -> bool:
        """Check if the user account is locked."""
        if self.status == UserStatus.LOCKED:
            return True
        if self.lockout_until and datetime.utcnow() < self.lockout_until:
            return True
        return False

    def is_active(self) -> bool:
        """Check if the user account is active and can be used."""
        return self.status == UserStatus.ACTIVE and not self.is_locked()

    def lock_account(self, duration_minutes: int = 30) -> None:
        """Lock the user account for a specified duration."""
        self.status = UserStatus.LOCKED
        self.lockout_until = datetime.utcnow() + timedelta(minutes=duration_minutes)

    def unlock_account(self) -> None:
        """Unlock the user account."""
        if self.status == UserStatus.LOCKED:
            self.status = UserStatus.ACTIVE
        self.lockout_until = None
        self.login_attempts = 0

    def increment_login_attempts(self) -> None:
        """Increment login attempts and lock if necessary."""
        self.login_attempts += 1
        if self.login_attempts >= self.max_login_attempts:
            self.lock_account()

    def reset_login_attempts(self) -> None:
        """Reset login attempts after successful login."""
        self.login_attempts = 0
        self.last_login = datetime.utcnow()


class TokenType(Enum):
    """Authentication token types."""

    ACCESS = "access"
    REFRESH = "refresh"
    API_KEY = "api_key"
    WEBSOCKET = "websocket"


@dataclass
class AuthToken:
    """Represents an authentication token."""

    token_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    token_type: TokenType = TokenType.ACCESS
    token_value: str = ""
    expires_at: datetime | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used: datetime | None = None
    is_revoked: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    # Security attributes
    client_ip: str | None = None
    user_agent: str | None = None

    def is_expired(self) -> bool:
        """Check if the token is expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def is_valid(self) -> bool:
        """Check if the token is valid (not expired or revoked)."""
        return not (self.is_expired() or self.is_revoked)

    def revoke(self) -> None:
        """Revoke the token."""
        self.is_revoked = True

    def touch(self) -> None:
        """Update the last used timestamp."""
        self.last_used = datetime.utcnow()


# Predefined system roles
def create_system_roles() -> dict[str, Role]:
    """Create predefined system roles."""

    # Guest role - minimal permissions
    guest_permissions = {
        Permission("read_public", PermissionType.READ, "Read public information"),
        Permission("websocket_connect", PermissionType.WEBSOCKET_CONNECT, "Connect to WebSocket"),
        Permission(
            "websocket_market_data",
            PermissionType.WEBSOCKET_MARKET_DATA,
            "Access market data via WebSocket",
        ),
    }

    guest_role = Role(
        name="guest",
        description="Guest user with minimal read-only access",
        permissions=guest_permissions,
        is_system_role=True,
    )

    # User role - basic trading permissions
    user_permissions = guest_permissions | {
        Permission("view_portfolio", PermissionType.VIEW_PORTFOLIO, "View portfolio information"),
        Permission(
            "websocket_portfolio",
            PermissionType.WEBSOCKET_PORTFOLIO,
            "Access portfolio data via WebSocket",
        ),
        Permission(
            "websocket_bot_status",
            PermissionType.WEBSOCKET_BOT_STATUS,
            "Access bot status via WebSocket",
        ),
        Permission("view_risk_metrics", PermissionType.VIEW_RISK_METRICS, "View risk metrics"),
    }

    user_role = Role(
        name="user",
        description="Standard user with portfolio viewing access",
        permissions=user_permissions,
        is_system_role=True,
    )

    # Trader role - full trading permissions
    trader_permissions = user_permissions | {
        Permission("place_order", PermissionType.PLACE_ORDER, "Place trading orders"),
        Permission("cancel_order", PermissionType.CANCEL_ORDER, "Cancel trading orders"),
        Permission("modify_order", PermissionType.MODIFY_ORDER, "Modify trading orders"),
        Permission("create_bot", PermissionType.CREATE_BOT, "Create trading bots"),
        Permission("start_bot", PermissionType.START_BOT, "Start trading bots"),
        Permission("stop_bot", PermissionType.STOP_BOT, "Stop trading bots"),
        Permission("configure_bot", PermissionType.CONFIGURE_BOT, "Configure trading bots"),
    }

    trader_role = Role(
        name="trader",
        description="Active trader with bot management capabilities",
        permissions=trader_permissions,
        is_system_role=True,
    )

    # Admin role - all permissions
    admin_permissions = trader_permissions | {
        Permission("delete_bot", PermissionType.DELETE_BOT, "Delete trading bots"),
        Permission("set_risk_limits", PermissionType.SET_RISK_LIMITS, "Set risk limits"),
        Permission("override_risk", PermissionType.OVERRIDE_RISK, "Override risk controls"),
        Permission("create_strategy", PermissionType.CREATE_STRATEGY, "Create trading strategies"),
        Permission("modify_strategy", PermissionType.MODIFY_STRATEGY, "Modify trading strategies"),
        Permission("deploy_strategy", PermissionType.DEPLOY_STRATEGY, "Deploy trading strategies"),
        Permission("manage_users", PermissionType.MANAGE_USERS, "Manage system users"),
        Permission("view_system_logs", PermissionType.VIEW_SYSTEM_LOGS, "View system logs"),
        Permission("system_config", PermissionType.SYSTEM_CONFIG, "Configure system settings"),
    }

    admin_role = Role(
        name="admin",
        description="Administrator with full system access",
        permissions=admin_permissions,
        is_system_role=True,
    )

    return {"guest": guest_role, "user": user_role, "trader": trader_role, "admin": admin_role}
