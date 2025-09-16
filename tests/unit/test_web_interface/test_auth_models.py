"""
Tests for web_interface.auth.models module.
"""

import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from src.web_interface.auth.models import (
    AuthToken,
    Permission,
    PermissionType,
    Role,
    TokenType,
    User,
    UserStatus,
    create_system_roles,
)


class TestPermissionType:
    """Tests for PermissionType enum."""

    def test_permission_type_values(self):
        """Test that permission types have correct values."""
        assert PermissionType.READ.value == "read"
        assert PermissionType.WRITE.value == "write"
        assert PermissionType.DELETE.value == "delete"
        assert PermissionType.PLACE_ORDER.value == "place_order"
        assert PermissionType.MANAGE_USERS.value == "manage_users"

    def test_permission_type_count(self):
        """Test that all expected permission types are defined."""
        expected_permissions = {
            "READ", "WRITE", "DELETE", "PLACE_ORDER", "CANCEL_ORDER", "MODIFY_ORDER",
            "VIEW_PORTFOLIO", "CREATE_BOT", "START_BOT", "STOP_BOT", "DELETE_BOT",
            "CONFIGURE_BOT", "SET_RISK_LIMITS", "OVERRIDE_RISK", "VIEW_RISK_METRICS",
            "CREATE_STRATEGY", "MODIFY_STRATEGY", "DEPLOY_STRATEGY", "MANAGE_USERS",
            "VIEW_SYSTEM_LOGS", "SYSTEM_CONFIG", "WEBSOCKET_CONNECT",
            "WEBSOCKET_MARKET_DATA", "WEBSOCKET_BOT_STATUS", "WEBSOCKET_PORTFOLIO"
        }
        actual_permissions = {perm.name for perm in PermissionType}
        assert actual_permissions == expected_permissions


class TestPermission:
    """Tests for Permission class."""

    def test_permission_creation(self):
        """Test basic permission creation."""
        perm = Permission(
            name="test_permission",
            permission_type=PermissionType.READ,
            description="Test permission"
        )
        assert perm.name == "test_permission"
        assert perm.permission_type == PermissionType.READ
        assert perm.description == "Test permission"
        assert perm.resource is None

    def test_permission_with_resource(self):
        """Test permission creation with resource."""
        perm = Permission(
            name="read_portfolio",
            permission_type=PermissionType.READ,
            description="Read portfolio",
            resource="portfolio"
        )
        assert perm.resource == "portfolio"

    def test_permission_str_without_resource(self):
        """Test string representation without resource."""
        perm = Permission(
            name="test_permission",
            permission_type=PermissionType.READ,
            description="Test permission"
        )
        assert str(perm) == "read"

    def test_permission_str_with_resource(self):
        """Test string representation with resource."""
        perm = Permission(
            name="read_portfolio",
            permission_type=PermissionType.READ,
            description="Read portfolio",
            resource="portfolio"
        )
        assert str(perm) == "read:portfolio"

    def test_permission_hash(self):
        """Test permission hashing."""
        perm1 = Permission("test", PermissionType.READ, "Test")
        perm2 = Permission("test", PermissionType.READ, "Test")
        perm3 = Permission("test", PermissionType.READ, "Test", resource="resource")

        assert hash(perm1) == hash(perm2)
        assert hash(perm1) != hash(perm3)

    def test_permission_equality_in_set(self):
        """Test that permissions can be used in sets correctly."""
        perm1 = Permission("test", PermissionType.READ, "Test")
        perm2 = Permission("test", PermissionType.READ, "Test")

        permission_set = {perm1}
        permission_set.add(perm2)

        # Should have only one permission since they have the same hash
        assert len(permission_set) == 1


class TestRole:
    """Tests for Role class."""

    def test_role_creation(self):
        """Test basic role creation."""
        role = Role(name="test_role", description="Test role")
        assert role.name == "test_role"
        assert role.description == "Test role"
        assert len(role.permissions) == 0
        assert not role.is_system_role
        assert isinstance(role.created_at, datetime)

    def test_role_hash(self):
        """Test role hashing based on name."""
        role1 = Role("admin", "Admin role")
        role2 = Role("admin", "Different description")
        role3 = Role("user", "User role")

        assert hash(role1) == hash(role2)
        assert hash(role1) != hash(role3)

    def test_role_equality(self):
        """Test role equality based on name."""
        role1 = Role("admin", "Admin role")
        role2 = Role("admin", "Different description")
        role3 = Role("user", "User role")

        assert role1 == role2
        assert role1 != role3
        assert role1 != "not_a_role"

    def test_add_permission(self):
        """Test adding permission to role."""
        role = Role("test", "Test role")
        perm = Permission("test", PermissionType.READ, "Test")

        role.add_permission(perm)

        assert perm in role.permissions
        assert len(role.permissions) == 1

    def test_remove_permission(self):
        """Test removing permission from role."""
        role = Role("test", "Test role")
        perm = Permission("test", PermissionType.READ, "Test")

        role.add_permission(perm)
        role.remove_permission(perm)

        assert perm not in role.permissions
        assert len(role.permissions) == 0

    def test_has_permission_exact_match(self):
        """Test has_permission with exact match."""
        role = Role("test", "Test role")
        perm = Permission("test", PermissionType.READ, "Test", resource="portfolio")
        role.add_permission(perm)

        assert role.has_permission(PermissionType.READ, "portfolio")
        assert not role.has_permission(PermissionType.WRITE, "portfolio")
        assert not role.has_permission(PermissionType.READ, "trading")

    def test_has_permission_wildcard(self):
        """Test has_permission with wildcard resource."""
        role = Role("test", "Test role")
        perm = Permission("test", PermissionType.READ, "Test")  # No resource = wildcard
        role.add_permission(perm)

        assert role.has_permission(PermissionType.READ)
        assert role.has_permission(PermissionType.READ, "any_resource")

    def test_has_permission_no_resource_specified(self):
        """Test has_permission when no resource is specified in check."""
        role = Role("test", "Test role")
        perm = Permission("test", PermissionType.READ, "Test", resource="portfolio")
        role.add_permission(perm)

        assert role.has_permission(PermissionType.READ)  # Should match any resource

    def test_get_permissions_by_type(self):
        """Test getting permissions by type."""
        role = Role("test", "Test role")
        read_perm = Permission("read1", PermissionType.READ, "Read 1")
        write_perm = Permission("write1", PermissionType.WRITE, "Write 1")
        read_perm2 = Permission("read2", PermissionType.READ, "Read 2")

        role.add_permission(read_perm)
        role.add_permission(write_perm)
        role.add_permission(read_perm2)

        read_permissions = role.get_permissions_by_type(PermissionType.READ)
        write_permissions = role.get_permissions_by_type(PermissionType.WRITE)

        assert len(read_permissions) == 2
        assert len(write_permissions) == 1
        assert read_perm in read_permissions
        assert read_perm2 in read_permissions
        assert write_perm in write_permissions


class TestUserStatus:
    """Tests for UserStatus enum."""

    def test_user_status_values(self):
        """Test user status enum values."""
        assert UserStatus.ACTIVE.value == "active"
        assert UserStatus.INACTIVE.value == "inactive"
        assert UserStatus.SUSPENDED.value == "suspended"
        assert UserStatus.LOCKED.value == "locked"
        assert UserStatus.PENDING_VERIFICATION.value == "pending_verification"


class TestUser:
    """Tests for User class."""

    def test_user_creation_defaults(self):
        """Test user creation with default values."""
        user = User()

        assert user.user_id  # Should be set to a UUID
        assert user.username == ""
        assert user.email == ""
        assert user.full_name == ""
        assert user.status == UserStatus.PENDING_VERIFICATION
        assert len(user.roles) == 0
        assert user.login_attempts == 0
        assert user.max_login_attempts == 5
        assert user.lockout_until is None
        assert user.allocated_capital == Decimal("0.0")
        assert user.max_daily_loss == Decimal("0.0")
        assert user.risk_level == "medium"

    def test_user_creation_with_values(self):
        """Test user creation with specified values."""
        user = User(
            username="testuser",
            email="test@example.com",
            full_name="Test User",
            status=UserStatus.ACTIVE
        )

        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.full_name == "Test User"
        assert user.status == UserStatus.ACTIVE

    def test_add_remove_role(self):
        """Test adding and removing roles."""
        user = User()
        role = Role("test", "Test role")

        user.add_role(role)
        assert role in user.roles
        assert len(user.roles) == 1

        user.remove_role(role)
        assert role not in user.roles
        assert len(user.roles) == 0

    def test_has_role(self):
        """Test role checking."""
        user = User()
        role = Role("admin", "Admin role")

        user.add_role(role)

        assert user.has_role("admin")
        assert not user.has_role("user")

    def test_has_permission(self):
        """Test permission checking through roles."""
        user = User()
        role = Role("test", "Test role")
        perm = Permission("test", PermissionType.READ, "Test")

        role.add_permission(perm)
        user.add_role(role)

        assert user.has_permission(PermissionType.READ)
        assert not user.has_permission(PermissionType.WRITE)

    def test_get_all_permissions(self):
        """Test getting all permissions from all roles."""
        user = User()
        role1 = Role("role1", "Role 1")
        role2 = Role("role2", "Role 2")

        perm1 = Permission("perm1", PermissionType.READ, "Permission 1")
        perm2 = Permission("perm2", PermissionType.WRITE, "Permission 2")
        perm3 = Permission("perm3", PermissionType.DELETE, "Permission 3")

        role1.add_permission(perm1)
        role1.add_permission(perm2)
        role2.add_permission(perm2)  # Duplicate should not appear twice
        role2.add_permission(perm3)

        user.add_role(role1)
        user.add_role(role2)

        all_permissions = user.get_all_permissions()

        assert len(all_permissions) == 3  # perm2 should not be duplicated
        assert perm1 in all_permissions
        assert perm2 in all_permissions
        assert perm3 in all_permissions

    def test_is_locked_status(self):
        """Test is_locked with LOCKED status."""
        user = User(status=UserStatus.LOCKED)
        assert user.is_locked()

    def test_is_locked_timeout(self):
        """Test is_locked with timeout."""
        user = User()
        user.lockout_until = datetime.now(timezone.utc) + timedelta(minutes=30)
        assert user.is_locked()

    def test_is_locked_timeout_expired(self):
        """Test is_locked with expired timeout."""
        user = User()
        user.lockout_until = datetime.now(timezone.utc) - timedelta(minutes=30)
        assert not user.is_locked()

    def test_is_active(self):
        """Test is_active method."""
        user = User(status=UserStatus.ACTIVE)
        assert user.is_active()

        user.status = UserStatus.INACTIVE
        assert not user.is_active()

        user.status = UserStatus.ACTIVE
        user.lockout_until = datetime.now(timezone.utc) + timedelta(minutes=30)
        assert not user.is_active()

    def test_lock_account(self):
        """Test account locking."""
        user = User()

        user.lock_account(60)

        assert user.status == UserStatus.LOCKED
        assert user.lockout_until is not None
        assert user.lockout_until > datetime.now(timezone.utc)

    def test_unlock_account(self):
        """Test account unlocking."""
        user = User(status=UserStatus.LOCKED)
        user.lockout_until = datetime.now(timezone.utc) + timedelta(minutes=30)
        user.login_attempts = 5

        user.unlock_account()

        assert user.status == UserStatus.ACTIVE
        assert user.lockout_until is None
        assert user.login_attempts == 0

    def test_unlock_account_non_locked_status(self):
        """Test unlocking when status is not LOCKED."""
        user = User(status=UserStatus.INACTIVE)
        user.lockout_until = datetime.now(timezone.utc) + timedelta(minutes=30)
        user.login_attempts = 3

        user.unlock_account()

        assert user.status == UserStatus.INACTIVE  # Should not change
        assert user.lockout_until is None
        assert user.login_attempts == 0

    def test_increment_login_attempts(self):
        """Test login attempt incrementing."""
        user = User()

        user.increment_login_attempts()
        assert user.login_attempts == 1
        assert user.status != UserStatus.LOCKED

    def test_increment_login_attempts_lock(self):
        """Test account locks after max attempts."""
        user = User(max_login_attempts=3)

        # First two attempts should not lock
        user.increment_login_attempts()
        user.increment_login_attempts()
        assert user.status != UserStatus.LOCKED

        # Third attempt should lock
        user.increment_login_attempts()
        assert user.status == UserStatus.LOCKED
        assert user.lockout_until is not None

    def test_reset_login_attempts(self):
        """Test resetting login attempts."""
        user = User()
        user.login_attempts = 3

        user.reset_login_attempts()

        assert user.login_attempts == 0
        assert user.last_login is not None
        assert user.last_login <= datetime.now(timezone.utc)


class TestTokenType:
    """Tests for TokenType enum."""

    def test_token_type_values(self):
        """Test token type enum values."""
        assert TokenType.ACCESS.value == "access"
        assert TokenType.REFRESH.value == "refresh"
        assert TokenType.API_KEY.value == "api_key"
        assert TokenType.WEBSOCKET.value == "websocket"


class TestAuthToken:
    """Tests for AuthToken class."""

    def test_token_creation_defaults(self):
        """Test token creation with defaults."""
        token = AuthToken()

        assert token.token_id
        assert token.user_id == ""
        assert token.token_type == TokenType.ACCESS
        assert token.token_value == ""
        assert token.expires_at is None
        assert isinstance(token.created_at, datetime)
        assert token.last_used is None
        assert not token.is_revoked
        assert len(token.metadata) == 0

    def test_token_creation_with_values(self):
        """Test token creation with specified values."""
        expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
        token = AuthToken(
            user_id="user123",
            token_type=TokenType.REFRESH,
            token_value="token_value",
            expires_at=expires_at
        )

        assert token.user_id == "user123"
        assert token.token_type == TokenType.REFRESH
        assert token.token_value == "token_value"
        assert token.expires_at == expires_at

    def test_is_expired_no_expiry(self):
        """Test is_expired with no expiry set."""
        token = AuthToken()
        assert not token.is_expired()

    def test_is_expired_future_expiry(self):
        """Test is_expired with future expiry."""
        token = AuthToken()
        token.expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
        assert not token.is_expired()

    def test_is_expired_past_expiry(self):
        """Test is_expired with past expiry."""
        token = AuthToken()
        token.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
        assert token.is_expired()

    def test_is_valid_good_token(self):
        """Test is_valid with good token."""
        token = AuthToken()
        token.expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
        assert token.is_valid()

    def test_is_valid_expired_token(self):
        """Test is_valid with expired token."""
        token = AuthToken()
        token.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
        assert not token.is_valid()

    def test_is_valid_revoked_token(self):
        """Test is_valid with revoked token."""
        token = AuthToken()
        token.is_revoked = True
        assert not token.is_valid()

    def test_revoke(self):
        """Test token revocation."""
        token = AuthToken()
        assert not token.is_revoked

        token.revoke()
        assert token.is_revoked

    def test_touch(self):
        """Test updating last used timestamp."""
        token = AuthToken()
        assert token.last_used is None

        token.touch()
        assert token.last_used is not None
        assert token.last_used <= datetime.now(timezone.utc)


class TestCreateSystemRoles:
    """Tests for create_system_roles function."""

    def test_create_system_roles_structure(self):
        """Test that system roles are created with correct structure."""
        roles = create_system_roles()

        assert "guest" in roles
        assert "user" in roles
        assert "trader" in roles
        assert "admin" in roles

        for role in roles.values():
            assert isinstance(role, Role)
            assert role.is_system_role

    def test_guest_role_permissions(self):
        """Test guest role has minimal permissions."""
        roles = create_system_roles()
        guest = roles["guest"]

        assert guest.has_permission(PermissionType.READ)
        assert guest.has_permission(PermissionType.WEBSOCKET_CONNECT)
        assert guest.has_permission(PermissionType.WEBSOCKET_MARKET_DATA)
        assert not guest.has_permission(PermissionType.PLACE_ORDER)

    def test_user_role_permissions(self):
        """Test user role has portfolio access."""
        roles = create_system_roles()
        user = roles["user"]

        # Should have all guest permissions
        assert user.has_permission(PermissionType.READ)
        assert user.has_permission(PermissionType.WEBSOCKET_CONNECT)

        # Plus additional user permissions
        assert user.has_permission(PermissionType.VIEW_PORTFOLIO)
        assert user.has_permission(PermissionType.WEBSOCKET_PORTFOLIO)
        assert user.has_permission(PermissionType.VIEW_RISK_METRICS)

        # But not trading permissions
        assert not user.has_permission(PermissionType.PLACE_ORDER)

    def test_trader_role_permissions(self):
        """Test trader role has trading permissions."""
        roles = create_system_roles()
        trader = roles["trader"]

        # Should have all user permissions
        assert trader.has_permission(PermissionType.VIEW_PORTFOLIO)
        assert trader.has_permission(PermissionType.WEBSOCKET_PORTFOLIO)

        # Plus trading permissions
        assert trader.has_permission(PermissionType.PLACE_ORDER)
        assert trader.has_permission(PermissionType.CANCEL_ORDER)
        assert trader.has_permission(PermissionType.CREATE_BOT)
        assert trader.has_permission(PermissionType.START_BOT)

        # But not admin permissions
        assert not trader.has_permission(PermissionType.MANAGE_USERS)

    def test_admin_role_permissions(self):
        """Test admin role has all permissions."""
        roles = create_system_roles()
        admin = roles["admin"]

        # Should have all trader permissions
        assert admin.has_permission(PermissionType.PLACE_ORDER)
        assert admin.has_permission(PermissionType.CREATE_BOT)

        # Plus admin permissions
        assert admin.has_permission(PermissionType.DELETE_BOT)
        assert admin.has_permission(PermissionType.MANAGE_USERS)
        assert admin.has_permission(PermissionType.SYSTEM_CONFIG)
        assert admin.has_permission(PermissionType.OVERRIDE_RISK)

    def test_role_permission_hierarchy(self):
        """Test that roles build upon each other."""
        roles = create_system_roles()

        guest_perms = len(roles["guest"].permissions)
        user_perms = len(roles["user"].permissions)
        trader_perms = len(roles["trader"].permissions)
        admin_perms = len(roles["admin"].permissions)

        # Each role should have more permissions than the previous
        assert user_perms > guest_perms
        assert trader_perms > user_perms
        assert admin_perms > trader_perms

    def test_system_role_flag(self):
        """Test that all created roles are marked as system roles."""
        roles = create_system_roles()

        for role in roles.values():
            assert role.is_system_role

    def test_role_descriptions(self):
        """Test that roles have meaningful descriptions."""
        roles = create_system_roles()

        assert "guest" in roles["guest"].description.lower()
        assert "user" in roles["user"].description.lower()
        assert "trader" in roles["trader"].description.lower()
        assert "admin" in roles["admin"].description.lower()