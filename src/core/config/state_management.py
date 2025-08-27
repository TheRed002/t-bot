"""State Management Configuration Module.

This module provides configuration settings for state management services
including synchronization intervals, cleanup policies, and checkpoint management.
"""

from pydantic import BaseModel, Field


class StateManagementConfig(BaseModel):
    """Configuration for state management system.

    This configuration covers state synchronization, persistence, validation,
    and cleanup operations for the trading bot system.
    """

    # Core synchronization settings
    sync_interval_seconds: int = Field(
        default=60, ge=10, le=3600, description="State synchronization interval in seconds"
    )

    cleanup_interval_seconds: int = Field(
        default=3600,  # 1 hour
        ge=300,
        le=86400,
        description="State cleanup interval in seconds",
    )

    state_ttl_seconds: int = Field(
        default=86400,  # 24 hours
        ge=3600,
        le=604800,  # 7 days
        description="State time-to-live in seconds",
    )

    validation_interval_seconds: int = Field(
        default=300,  # 5 minutes
        ge=60,
        le=3600,
        description="State validation interval in seconds",
    )

    snapshot_interval_seconds: int = Field(
        default=1800,  # 30 minutes
        ge=300,
        le=7200,
        description="State snapshot interval in seconds",
    )

    max_state_versions: int = Field(
        default=10, ge=3, le=100, description="Maximum number of state versions to keep"
    )

    # Snapshot and persistence settings
    max_snapshots_per_bot: int = Field(
        default=10, ge=3, le=50, description="Maximum number of snapshots to keep per bot"
    )

    snapshot_interval_minutes: int = Field(
        default=5, ge=1, le=60, description="Snapshot creation interval in minutes"
    )

    # Checkpoint management
    checkpoints: dict = Field(
        default_factory=lambda: {
            "directory": "data/checkpoints",
            "max_per_bot": 50,
            "compression_enabled": True,
            "encryption_enabled": False,
        },
        description="Checkpoint management configuration",
    )

    # State validation settings
    validation_enabled: bool = Field(default=True, description="Enable state validation")

    strict_validation: bool = Field(
        default=False, description="Enable strict state validation (may impact performance)"
    )

    validation_timeout_seconds: int = Field(
        default=30, ge=5, le=300, description="Timeout for state validation operations"
    )

    # Recovery and rollback settings
    enable_auto_recovery: bool = Field(
        default=True, description="Enable automatic state recovery on corruption"
    )

    recovery_timeout_seconds: int = Field(
        default=120, ge=30, le=600, description="Timeout for state recovery operations"
    )

    max_recovery_attempts: int = Field(
        default=3, ge=1, le=10, description="Maximum recovery attempts before manual intervention"
    )

    # Performance and optimization
    batch_size: int = Field(
        default=100, ge=10, le=1000, description="Batch size for state operations"
    )

    async_operations_enabled: bool = Field(
        default=True, description="Enable asynchronous state operations"
    )

    compression_enabled: bool = Field(
        default=True, description="Enable state compression for storage"
    )

    encryption_enabled: bool = Field(
        default=False, description="Enable state encryption (requires additional setup)"
    )

    # Memory management
    memory_cache_size_mb: int = Field(
        default=256, ge=64, le=2048, description="Memory cache size in MB for state data"
    )

    cache_ttl_seconds: int = Field(
        default=300,  # 5 minutes
        ge=60,
        le=3600,
        description="Cache TTL for state data in seconds",
    )

    # Monitoring and alerting
    enable_metrics: bool = Field(
        default=True, description="Enable state management metrics collection"
    )

    metrics_interval_seconds: int = Field(
        default=60, ge=30, le=300, description="Metrics collection interval in seconds"
    )

    alert_on_corruption: bool = Field(
        default=True, description="Send alerts when state corruption is detected"
    )

    alert_on_recovery_failure: bool = Field(
        default=True, description="Send alerts when state recovery fails"
    )

    def get_checkpoint_config(self) -> dict:
        """Get checkpoint configuration."""
        return self.checkpoints.copy()

    def get_validation_config(self) -> dict:
        """Get validation configuration."""
        return {
            "validation_enabled": self.validation_enabled,
            "strict_validation": self.strict_validation,
            "validation_interval_seconds": self.validation_interval_seconds,
            "validation_timeout_seconds": self.validation_timeout_seconds,
        }

    def get_recovery_config(self) -> dict:
        """Get recovery configuration."""
        return {
            "enable_auto_recovery": self.enable_auto_recovery,
            "recovery_timeout_seconds": self.recovery_timeout_seconds,
            "max_recovery_attempts": self.max_recovery_attempts,
        }

    def get_performance_config(self) -> dict:
        """Get performance configuration."""
        return {
            "batch_size": self.batch_size,
            "async_operations_enabled": self.async_operations_enabled,
            "compression_enabled": self.compression_enabled,
            "encryption_enabled": self.encryption_enabled,
            "memory_cache_size_mb": self.memory_cache_size_mb,
            "cache_ttl_seconds": self.cache_ttl_seconds,
        }

    def get_monitoring_config(self) -> dict:
        """Get monitoring configuration."""
        return {
            "enable_metrics": self.enable_metrics,
            "metrics_interval_seconds": self.metrics_interval_seconds,
            "alert_on_corruption": self.alert_on_corruption,
            "alert_on_recovery_failure": self.alert_on_recovery_failure,
        }

    def get_sync_config(self) -> dict:
        """Get synchronization configuration."""
        return {
            "sync_interval_seconds": self.sync_interval_seconds,
            "snapshot_interval_seconds": self.snapshot_interval_seconds,
            "max_state_versions": self.max_state_versions,
            "state_ttl_seconds": self.state_ttl_seconds,
        }

    model_config = {"validate_assignment": True, "extra": "forbid"}
