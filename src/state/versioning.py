"""
State versioning and migration system for T-Bot trading platform.

This module provides comprehensive state schema versioning, migration support,
and backward compatibility for state persistence operations.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from packaging import version

try:
    from packaging.version import LegacyVersion
except (ImportError, AttributeError):
    # Fallback for older packaging versions or when LegacyVersion doesn't exist
    LegacyVersion = None

from src.core.exceptions import StateError


class MigrationType(Enum):
    """Types of migrations."""

    SCHEMA_UPGRADE = "schema_upgrade"
    SCHEMA_DOWNGRADE = "schema_downgrade"
    DATA_TRANSFORM = "data_transform"
    INDEX_UPDATE = "index_update"
    CLEANUP = "cleanup"


class MigrationStatus(Enum):
    """Migration execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class StateVersion:
    """State version information."""

    version_string: str
    major: int = 0
    minor: int = 0
    patch: int = 0
    build: str | None = None

    def __post_init__(self) -> None:
        """Parse version string into components."""
        try:
            parsed = version.parse(self.version_string)
            if hasattr(parsed, "major"):
                self.major = parsed.major
                self.minor = parsed.minor
                self.patch = parsed.micro
            elif LegacyVersion and isinstance(parsed, LegacyVersion):
                # Handle legacy version strings
                parts = str(parsed).split(".")
                self.major = int(parts[0]) if len(parts) > 0 else 0
                self.minor = int(parts[1]) if len(parts) > 1 else 0
                self.patch = int(parts[2]) if len(parts) > 2 else 0
        except Exception as e:
            logging.warning(f"Failed to parse version {self.version_string}: {e}")

    def __str__(self) -> str:
        return self.version_string

    def __lt__(self, other: "StateVersion") -> bool:
        return version.parse(self.version_string) < version.parse(other.version_string)

    def __le__(self, other: "StateVersion") -> bool:
        return version.parse(self.version_string) <= version.parse(other.version_string)

    def __gt__(self, other: "StateVersion") -> bool:
        return version.parse(self.version_string) > version.parse(other.version_string)

    def __ge__(self, other: "StateVersion") -> bool:
        return version.parse(self.version_string) >= version.parse(other.version_string)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StateVersion):
            return NotImplemented
        return version.parse(self.version_string) == version.parse(other.version_string)

    def is_compatible_with(self, other: "StateVersion") -> bool:
        """Check if this version is backward compatible with another."""
        return self.major == other.major and self.minor >= other.minor


@dataclass
class MigrationRecord:
    """Record of a migration operation."""

    migration_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    migration_type: MigrationType = MigrationType.SCHEMA_UPGRADE

    # Version information
    from_version: StateVersion | None = None
    to_version: StateVersion | None = None

    # Execution details
    status: MigrationStatus = MigrationStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: int = 0

    # Migration specifics
    affected_state_types: list[str] = field(default_factory=list)
    states_migrated: int = 0
    states_failed: int = 0

    # Error information
    error_message: str = ""
    rollback_available: bool = True
    rollback_script: str = ""

    # Dependencies
    depends_on: list[str] = field(default_factory=list)  # Other migration IDs
    blocks: list[str] = field(default_factory=list)  # Migrations blocked by this one

    # Validation
    pre_migration_checksum: str = ""
    post_migration_checksum: str = ""
    validation_passed: bool = False


class StateMigration(ABC):
    """Abstract base class for state migrations."""

    def __init__(self, migration_id: str, name: str, description: str = ""):
        """Initialize migration."""
        self.migration_id = migration_id
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @property
    @abstractmethod
    def from_version(self) -> StateVersion:
        """Source version for this migration."""
        pass

    @property
    @abstractmethod
    def to_version(self) -> StateVersion:
        """Target version for this migration."""
        pass

    @property
    @abstractmethod
    def migration_type(self) -> MigrationType:
        """Type of migration."""
        pass

    @property
    def depends_on(self) -> list[str]:
        """List of migration IDs this migration depends on."""
        return []

    @property
    def affected_state_types(self) -> list[str]:
        """List of state types affected by this migration."""
        return []

    @abstractmethod
    async def migrate(self, state_data: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
        """
        Perform the migration on state data.

        Args:
            state_data: Current state data
            metadata: State metadata

        Returns:
            Migrated state data
        """
        pass

    @abstractmethod
    async def rollback(
        self, state_data: dict[str, Any], metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Rollback the migration.

        Args:
            state_data: Current state data (post-migration)
            metadata: State metadata

        Returns:
            Rolled back state data
        """
        pass

    async def validate_pre_migration(
        self, state_data: dict[str, Any], metadata: dict[str, Any]
    ) -> bool:
        """Validate state before migration."""
        return True

    async def validate_post_migration(
        self, state_data: dict[str, Any], metadata: dict[str, Any]
    ) -> bool:
        """Validate state after migration."""
        return True


class StateVersioningSystem:
    """
    State versioning and migration management system.

    Handles state schema evolution, migration execution, and version compatibility.

    Repository Interface:
        The metadata_repository parameter should implement:
        - get_all() -> AsyncIterable: Returns all state metadata records

        Each metadata record should support either:
        - Attribute access: record.schema_version, record.state_type
        - Dict access: record['schema_version'], record['state_type']

    This design allows integration with any repository implementation that
    provides the required interface without tight coupling to specific types.
    """

    def __init__(
        self,
        current_version: str = "1.0.0",
        metadata_repository=None,
    ):
        """
        Initialize versioning system.

        Args:
            current_version: Current system version
            metadata_repository: Optional repository for state metadata operations.
                                Must implement get_all() async method if provided.
                                Can be any object with compatible interface.
        """
        self.current_version = StateVersion(current_version)
        self.logger = logging.getLogger(__name__)
        self._metadata_repository = metadata_repository

        # Migration registry
        self._migrations: dict[str, StateMigration] = {}
        self._migration_records: dict[str, MigrationRecord] = {}

        # Version registry
        self._version_schemas: dict[str, dict[str, Any]] = {}
        self._compatibility_matrix: dict[str, list[str]] = {}

        # Execution tracking
        self._active_migrations: dict[str, MigrationRecord] = {}
        self._migration_order_cache: list[str] = []

        # Configuration
        self.auto_migrate = True
        self.validate_migrations = True
        self.create_backups = True
        self.max_concurrent_migrations = 3

    def register_migration(self, migration: StateMigration) -> None:
        """Register a migration."""
        if migration.migration_id in self._migrations:
            raise StateError(f"Migration {migration.migration_id} already registered")

        self._migrations[migration.migration_id] = migration
        self._migration_order_cache = []  # Clear cache

        self.logger.info(f"Registered migration: {migration.migration_id}")

    def register_version_schema(self, version: str, schema: dict[str, Any]) -> None:
        """Register a version schema."""
        self._version_schemas[version] = schema.copy()
        self.logger.debug(f"Registered schema for version: {version}")

    def set_version_compatibility(self, version: str, compatible_versions: list[str]) -> None:
        """Set version compatibility information."""
        self._compatibility_matrix[version] = compatible_versions.copy()

    def get_migration_path(self, from_version: StateVersion, to_version: StateVersion) -> list[str]:
        """
        Get ordered list of migrations needed to go from one version to another.

        Args:
            from_version: Starting version
            to_version: Target version

        Returns:
            List of migration IDs in execution order
        """
        if from_version == to_version:
            return []

        # Build migration graph
        migration_graph: dict[StateVersion, list[tuple[StateVersion, str]]] = {}
        for migration_id, migration in self._migrations.items():
            from_ver = migration.from_version
            to_ver = migration.to_version

            if from_ver not in migration_graph:
                migration_graph[from_ver] = []
            migration_graph[from_ver].append((to_ver, migration_id))

        # Find path using BFS
        from collections import deque

        queue: deque[tuple[StateVersion, list[str]]] = deque([(from_version, [])])
        visited = {from_version}

        while queue:
            current_version, path = queue.popleft()

            if current_version == to_version:
                return path

            if current_version in migration_graph:
                for next_version, migration_id in migration_graph[current_version]:
                    if next_version not in visited:
                        visited.add(next_version)
                        queue.append((next_version, [*path, migration_id]))

        # No path found
        raise StateError(f"No migration path from {from_version} to {to_version}")

    async def migrate_state(
        self,
        state_type: str,
        state_id: str,
        state_data: dict[str, Any],
        current_version: str,
        target_version: str | None = None,
    ) -> dict[str, Any]:
        """
        Migrate state from current version to target version.

        Args:
            state_type: Type of state
            state_id: State identifier
            state_data: Current state data
            current_version: Current state version
            target_version: Target version (defaults to current system version)

        Returns:
            Migrated state data
        """
        if target_version is None:
            target_version = self.current_version.version_string

        from_version = StateVersion(current_version)
        to_version = StateVersion(target_version)

        if from_version == to_version:
            return state_data  # No migration needed

        # Check if versions are compatible
        if not self.is_version_compatible(current_version, target_version):
            raise StateError(f"Version {current_version} is not compatible with {target_version}")

        # Get migration path
        migration_path = self.get_migration_path(from_version, to_version)

        if not migration_path:
            return state_data  # No migrations needed

        # Execute migrations in order
        migrated_data = state_data.copy()
        metadata = {
            "state_type": state_type,
            "state_id": state_id,
            "original_version": current_version,
            "target_version": target_version,
        }

        for migration_id in migration_path:
            migration = self._migrations[migration_id]

            # Create migration record
            record = MigrationRecord(
                migration_id=migration_id,
                name=migration.name,
                description=migration.description,
                migration_type=migration.migration_type,
                from_version=migration.from_version,
                to_version=migration.to_version,
                affected_state_types=[state_type],
                started_at=datetime.now(timezone.utc),
            )

            self._migration_records[record.migration_id] = record
            self._active_migrations[migration_id] = record

            try:
                record.status = MigrationStatus.RUNNING

                # Pre-migration validation
                if self.validate_migrations:
                    if not await migration.validate_pre_migration(migrated_data, metadata):
                        raise StateError(f"Pre-migration validation failed for {migration_id}")

                # Execute migration
                self.logger.info(f"Executing migration: {migration_id}")
                start_time = datetime.now(timezone.utc)

                migrated_data = await migration.migrate(migrated_data, metadata)

                end_time = datetime.now(timezone.utc)
                record.duration_ms = int((end_time - start_time).total_seconds() * 1000)

                # Post-migration validation
                if self.validate_migrations:
                    if not await migration.validate_post_migration(migrated_data, metadata):
                        raise StateError(f"Post-migration validation failed for {migration_id}")

                record.status = MigrationStatus.COMPLETED
                record.completed_at = end_time
                record.states_migrated = 1
                record.validation_passed = True

                self.logger.info(f"Migration completed: {migration_id}")

            except Exception as e:
                record.status = MigrationStatus.FAILED
                record.completed_at = datetime.now(timezone.utc)
                record.error_message = str(e)
                record.states_failed = 1

                self.logger.error(f"Migration failed: {migration_id}: {e}")

                # Rollback previous migrations if needed
                if hasattr(migration, "rollback") and callable(migration.rollback):
                    try:
                        migrated_data = await migration.rollback(migrated_data, metadata)
                        record.status = MigrationStatus.ROLLED_BACK
                    except Exception as rollback_error:
                        self.logger.error(f"Rollback failed: {migration_id}: {rollback_error}")
                        raise StateError(
                            f"Migration {migration_id} failed with rollback error: {e}"
                        ) from rollback_error

                raise StateError(f"Migration {migration_id} failed: {e}") from e

            finally:
                if migration_id in self._active_migrations:
                    del self._active_migrations[migration_id]

        return migrated_data

    async def batch_migrate_states(
        self, states: list[dict[str, Any]], target_version: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Migrate multiple states in batch.

        Args:
            states: List of state dictionaries with version info
            target_version: Target version for all states

        Returns:
            List of migrated states
        """
        if target_version is None:
            target_version = self.current_version.version_string

        migrated_states = []

        for state_info in states:
            try:
                state_type = state_info.get("state_type", "")
                state_id = state_info.get("state_id", "")
                state_data = state_info.get("state_data", {})
                current_version = state_info.get("version", "1.0.0")

                migrated_data = await self.migrate_state(
                    state_type, state_id, state_data, current_version, target_version
                )

                migrated_states.append(
                    {
                        "state_type": state_type,
                        "state_id": state_id,
                        "state_data": migrated_data,
                        "version": target_version,
                        "migrated": True,
                    }
                )

            except Exception as e:
                self.logger.error(f"Failed to migrate state {state_info}: {e}")
                migrated_states.append(
                    {
                        **state_info,
                        "migrated": False,
                        "error": str(e),
                    }
                )

        return migrated_states

    def is_version_compatible(self, version1: str, version2: str) -> bool:
        """Check if two versions are compatible."""
        v1 = StateVersion(version1)
        v2 = StateVersion(version2)

        # Check explicit compatibility matrix
        if version1 in self._compatibility_matrix:
            return version2 in self._compatibility_matrix[version1]

        if version2 in self._compatibility_matrix:
            return version1 in self._compatibility_matrix[version2]

        # Default compatibility rules
        return v1.is_compatible_with(v2) or v2.is_compatible_with(v1)

    def get_schema_for_version(self, version: str) -> dict[str, Any]:
        """Get schema definition for a version."""
        return self._version_schemas.get(version, {})

    async def validate_state_schema(
        self, state_data: dict[str, Any], state_type: str, version: str
    ) -> bool:
        """Validate state data against version schema."""
        schema = self.get_schema_for_version(version)
        if not schema:
            return True  # No schema defined, assume valid

        try:
            # Could implement JSON schema validation here
            # For now, basic validation
            required_fields = schema.get("required", [])
            for field in required_fields:
                if field not in state_data:
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Schema validation failed: {e}")
            return False

    async def get_version_statistics(self) -> dict[str, Any]:
        """Get version and migration statistics."""
        # Get version distribution from database
        version_counts: dict[str, dict[str, int]] = {}

        try:
            if not self._metadata_repository:
                self.logger.warning("No metadata repository available for version statistics")
                return {
                    "error": "Repository not configured",
                    "version_distribution": {},
                    "total_states": 0,
                }

            # Check if repository has required method
            if not hasattr(self._metadata_repository, "get_all"):
                self.logger.error("Metadata repository does not implement get_all() method")
                return {
                    "error": "Repository method not available",
                    "version_distribution": {},
                    "total_states": 0,
                }

            # Use repository to get all metadata
            all_metadata = await self._metadata_repository.get_all()

            for metadata in all_metadata:
                # Handle both attribute and dict-like access patterns
                if hasattr(metadata, "schema_version"):
                    version = metadata.schema_version or "unknown"
                    state_type = metadata.state_type
                elif isinstance(metadata, dict):
                    version = metadata.get("schema_version", "unknown")
                    state_type = metadata.get("state_type", "unknown")
                else:
                    self.logger.warning(f"Unknown metadata format: {type(metadata)}")
                    continue

                if version not in version_counts:
                    version_counts[version] = {}

                version_counts[version][state_type] = version_counts[version].get(state_type, 0) + 1

        except Exception as e:
            self.logger.error(f"Failed to get version statistics: {e}")
            return {
                "error": f"Statistics retrieval failed: {e!s}",
                "version_distribution": {},
                "total_states": 0,
            }

        return {
            "current_version": self.current_version.version_string,
            "registered_migrations": len(self._migrations),
            "completed_migrations": len(
                [
                    r
                    for r in self._migration_records.values()
                    if r.status == MigrationStatus.COMPLETED
                ]
            ),
            "failed_migrations": len(
                [r for r in self._migration_records.values() if r.status == MigrationStatus.FAILED]
            ),
            "version_distribution": version_counts,
            "active_migrations": len(self._active_migrations),
            "registered_schemas": len(self._version_schemas),
        }

    def get_migration_history(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get migration execution history."""
        records = list(self._migration_records.values())
        records.sort(key=lambda x: x.started_at or datetime.min, reverse=True)

        return [
            {
                "migration_id": record.migration_id,
                "name": record.name,
                "description": record.description,
                "migration_type": record.migration_type.value,
                "from_version": str(record.from_version) if record.from_version else None,
                "to_version": str(record.to_version) if record.to_version else None,
                "status": record.status.value,
                "started_at": record.started_at.isoformat() if record.started_at else None,
                "completed_at": record.completed_at.isoformat() if record.completed_at else None,
                "duration_ms": record.duration_ms,
                "states_migrated": record.states_migrated,
                "states_failed": record.states_failed,
                "error_message": record.error_message,
                "validation_passed": record.validation_passed,
            }
            for record in records[:limit]
        ]


# Example migration implementations


class AddTimestampMigration(StateMigration):
    """Example migration to add timestamp field."""

    def __init__(self):
        super().__init__(
            migration_id="add_timestamp_001",
            name="Add Timestamp Field",
            description="Add created_at and updated_at timestamps to all state types",
        )

    @property
    def from_version(self) -> StateVersion:
        return StateVersion("1.0.0")

    @property
    def to_version(self) -> StateVersion:
        return StateVersion("1.1.0")

    @property
    def migration_type(self) -> MigrationType:
        return MigrationType.SCHEMA_UPGRADE

    async def migrate(self, state_data: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
        """Add timestamp fields if missing."""
        migrated = state_data.copy()
        now = datetime.now(timezone.utc).isoformat()

        if "created_at" not in migrated:
            migrated["created_at"] = now

        if "updated_at" not in migrated:
            migrated["updated_at"] = now

        return migrated

    async def rollback(
        self, state_data: dict[str, Any], metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Remove timestamp fields."""
        rolled_back = state_data.copy()
        rolled_back.pop("created_at", None)
        rolled_back.pop("updated_at", None)
        return rolled_back


class RenameFieldMigration(StateMigration):
    """Example migration to rename a field."""

    def __init__(self, old_field: str, new_field: str, affected_types: list[str]):
        super().__init__(
            migration_id=f"rename_{old_field}_to_{new_field}",
            name=f"Rename {old_field} to {new_field}",
            description=f"Rename field {old_field} to {new_field} in {', '.join(affected_types)}",
        )
        self.old_field = old_field
        self.new_field = new_field
        self._affected_types = affected_types

    @property
    def from_version(self) -> StateVersion:
        return StateVersion("1.1.0")

    @property
    def to_version(self) -> StateVersion:
        return StateVersion("1.2.0")

    @property
    def migration_type(self) -> MigrationType:
        return MigrationType.DATA_TRANSFORM

    @property
    def affected_state_types(self) -> list[str]:
        return self._affected_types

    async def migrate(self, state_data: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
        """Rename field in state data."""
        migrated = state_data.copy()

        # Only apply to affected state types
        if metadata.get("state_type") in self._affected_types:
            if self.old_field in migrated:
                migrated[self.new_field] = migrated.pop(self.old_field)

        return migrated

    async def rollback(
        self, state_data: dict[str, Any], metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Restore original field name."""
        rolled_back = state_data.copy()

        if metadata.get("state_type") in self._affected_types:
            if self.new_field in rolled_back:
                rolled_back[self.old_field] = rolled_back.pop(self.new_field)

        return rolled_back
