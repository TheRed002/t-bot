"""
Version Manager for API versioning in T-Bot Trading System.

This module handles API version management, compatibility checking,
and deprecation warnings for trading system endpoints.
"""

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from src.base import BaseComponent


class VersionStatus(Enum):
    """API version status."""

    ACTIVE = "active"
    DEPRECATED = "deprecated"
    SUNSET = "sunset"


@dataclass
class APIVersion:
    """API version configuration."""

    version: str  # e.g., "v1", "v2.1"
    major: int
    minor: int
    patch: int = 0
    status: VersionStatus = VersionStatus.ACTIVE
    release_date: datetime | None = None
    deprecation_date: datetime | None = None
    sunset_date: datetime | None = None
    features: set[str] | None = None
    breaking_changes: list[str] | None = None

    def __post_init__(self):
        if self.features is None:
            self.features = set()
        if self.breaking_changes is None:
            self.breaking_changes = []

    def __str__(self) -> str:
        return self.version

    def __lt__(self, other: "APIVersion") -> bool:
        """Compare versions for sorting."""
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    def __eq__(self, other: object) -> bool:
        """Check version equality."""
        if not isinstance(other, APIVersion):
            return NotImplemented
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)

    def is_compatible_with(self, other: "APIVersion") -> bool:
        """Check if this version is compatible with another version."""
        # Same major version is generally compatible
        if self.major == other.major:
            return True

        # Check if there are no breaking changes between versions
        if self.major > other.major:
            # Check if any version between other and self has breaking changes
            return len(self.breaking_changes) == 0

        return False

    def is_deprecated(self) -> bool:
        """Check if this version is deprecated."""
        return self.status in (VersionStatus.DEPRECATED, VersionStatus.SUNSET)

    def is_sunset(self) -> bool:
        """Check if this version is sunset (no longer supported)."""
        return self.status == VersionStatus.SUNSET


class VersionManager(BaseComponent):
    """Manager for API versioning and compatibility."""

    def __init__(self):
        super().__init__()
        self._versions: dict[str, APIVersion] = {}
        self._default_version: str | None = None
        self._latest_version: str | None = None
        self._version_pattern = re.compile(r"^v(\d+)(?:\.(\d+))?(?:\.(\d+))?$")

        # Initialize with default versions
        self._initialize_default_versions()

    def _initialize_default_versions(self) -> None:
        """Initialize default API versions."""
        # Version 1.0
        v1 = APIVersion(
            version="v1",
            major=1,
            minor=0,
            patch=0,
            status=VersionStatus.ACTIVE,
            release_date=datetime.now(timezone.utc) - timedelta(days=365),
            features={"basic_trading", "bot_management", "portfolio_view", "market_data"},
        )

        # Version 1.1 - with enhanced features
        v1_1 = APIVersion(
            version="v1.1",
            major=1,
            minor=1,
            patch=0,
            status=VersionStatus.ACTIVE,
            release_date=datetime.now(timezone.utc) - timedelta(days=180),
            features={
                "basic_trading",
                "bot_management",
                "portfolio_view",
                "market_data",
                "advanced_orders",
                "risk_metrics",
            },
        )

        # Version 2.0 - with breaking changes
        v2 = APIVersion(
            version="v2",
            major=2,
            minor=0,
            patch=0,
            status=VersionStatus.ACTIVE,
            release_date=datetime.now(timezone.utc) - timedelta(days=90),
            features={
                "enhanced_trading",
                "advanced_bot_management",
                "real_time_portfolio",
                "streaming_market_data",
                "ml_strategies",
                "comprehensive_risk_management",
            },
            breaking_changes=[
                "Order structure changed",
                "Authentication flow updated",
                "WebSocket event format modified",
            ],
        )

        self.register_version(v1)
        self.register_version(v1_1)
        self.register_version(v2)

        self._default_version = "v2"
        self._latest_version = "v2"

    def register_version(self, version: APIVersion) -> None:
        """Register a new API version."""
        self._versions[version.version] = version
        self.logger.info(f"Registered API version: {version.version}")

    def parse_version(self, version_string: str) -> APIVersion | None:
        """Parse a version string and return APIVersion object."""
        match = self._version_pattern.match(version_string)
        if not match:
            return None

        major = int(match.group(1))
        minor = int(match.group(2)) if match.group(2) else 0
        patch = int(match.group(3)) if match.group(3) else 0

        return APIVersion(version=version_string, major=major, minor=minor, patch=patch)

    def get_version(self, version: str) -> APIVersion | None:
        """Get a specific version."""
        return self._versions.get(version)

    def get_latest_version(self) -> APIVersion | None:
        """Get the latest API version."""
        if self._latest_version:
            return self._versions.get(self._latest_version)
        return None

    def get_default_version(self) -> APIVersion | None:
        """Get the default API version."""
        if self._default_version:
            return self._versions.get(self._default_version)
        return None

    def list_versions(self, include_deprecated: bool = True) -> list[APIVersion]:
        """List all API versions."""
        versions = list(self._versions.values())
        if not include_deprecated:
            versions = [v for v in versions if not v.is_deprecated()]
        return sorted(versions)

    def resolve_version(self, requested_version: str | None = None) -> APIVersion:
        """
        Resolve the version to use for a request.

        Args:
            requested_version: The version requested by the client

        Returns:
            The resolved API version

        Raises:
            ValueError: If the requested version is invalid or sunset
        """
        if requested_version is None:
            return self.get_default_version()

        # Check if the exact version exists
        version = self.get_version(requested_version)
        if version:
            if version.is_sunset():
                raise ValueError(f"API version {requested_version} is no longer supported")
            return version

        # Try to parse and find compatible version
        parsed = self.parse_version(requested_version)
        if not parsed:
            raise ValueError(f"Invalid version format: {requested_version}")

        # Find the best compatible version
        compatible_versions = []
        for existing_version in self._versions.values():
            if existing_version.is_compatible_with(parsed) and not existing_version.is_sunset():
                compatible_versions.append(existing_version)

        if not compatible_versions:
            raise ValueError(f"No compatible version found for: {requested_version}")

        # Return the highest compatible version
        return max(compatible_versions)

    def check_feature_availability(self, version: str, feature: str) -> bool:
        """Check if a feature is available in a specific version."""
        api_version = self.get_version(version)
        if not api_version:
            return False
        return feature in api_version.features

    def get_deprecation_info(self, version: str) -> dict[str, Any] | None:
        """Get deprecation information for a version."""
        api_version = self.get_version(version)
        if not api_version or not api_version.is_deprecated():
            return None

        return {
            "version": version,
            "status": api_version.status.value,
            "deprecation_date": (
                api_version.deprecation_date.isoformat() if api_version.deprecation_date else None
            ),
            "sunset_date": api_version.sunset_date.isoformat() if api_version.sunset_date else None,
            "recommended_version": self._latest_version,
            "breaking_changes": api_version.breaking_changes,
        }

    def deprecate_version(self, version: str, sunset_date: datetime | None = None) -> bool:
        """Mark a version as deprecated."""
        api_version = self.get_version(version)
        if not api_version:
            return False

        api_version.status = VersionStatus.DEPRECATED
        api_version.deprecation_date = datetime.now(timezone.utc)
        if sunset_date:
            api_version.sunset_date = sunset_date

        self.logger.warning(f"API version {version} has been deprecated")
        return True

    def sunset_version(self, version: str) -> bool:
        """Mark a version as sunset (no longer supported)."""
        api_version = self.get_version(version)
        if not api_version:
            return False

        api_version.status = VersionStatus.SUNSET
        api_version.sunset_date = datetime.now(timezone.utc)

        self.logger.warning(f"API version {version} has been sunset")
        return True

    def get_version_migration_guide(self, from_version: str, to_version: str) -> dict[str, Any]:
        """Get migration guide between versions."""
        from_api = self.get_version(from_version)
        to_api = self.get_version(to_version)

        if not from_api or not to_api:
            return {"error": "Invalid version specified"}

        if from_api == to_api:
            return {"message": "No migration needed - same version"}

        # Generate migration information
        migration_info = {
            "from_version": from_version,
            "to_version": to_version,
            "compatibility": from_api.is_compatible_with(to_api),
            "breaking_changes": (to_api.breaking_changes or []) if from_api < to_api else [],
            "new_features": (to_api.features or set()) - (from_api.features or set())
            if from_api < to_api
            else set(),
            "deprecated_features": (
                (from_api.features or set()) - (to_api.features or set())
                if from_api > to_api
                else set()
            ),
        }

        # Convert sets to lists for JSON serialization
        migration_info["new_features"] = list(migration_info["new_features"])
        migration_info["deprecated_features"] = list(migration_info["deprecated_features"])

        return migration_info


# Global version manager instance
_global_version_manager: VersionManager | None = None


def get_version_manager() -> VersionManager:
    """Get or create the global version manager."""
    global _global_version_manager
    if _global_version_manager is None:
        _global_version_manager = VersionManager()
    return _global_version_manager
