"""Bot Management Configuration Module.

This module provides configuration settings for bot management services
including resource limits, monitoring intervals, and operational parameters.
"""

from decimal import Decimal

from pydantic import BaseModel, Field, field_serializer


class BotManagementConfig(BaseModel):
    """Configuration for bot management system.

    This configuration covers bot lifecycle management, resource allocation,
    coordination, and monitoring parameters.
    """

    # Core operational limits
    max_concurrent_bots: int = Field(
        default=50, ge=1, le=1000, description="Maximum number of concurrent bots allowed"
    )

    max_capital_allocation: Decimal = Field(
        default=Decimal("1000000"), gt=0, description="Maximum capital allocation per bot in USD"
    )

    heartbeat_timeout_seconds: int = Field(
        default=300,
        ge=30,
        le=3600,
        description="Bot heartbeat timeout in seconds (5 minutes default)",
    )

    health_check_interval_seconds: int = Field(
        default=60, ge=10, le=600, description="Health check interval for bots in seconds"
    )

    bot_startup_timeout_seconds: int = Field(
        default=120, ge=30, le=600, description="Maximum time to wait for bot startup"
    )

    bot_shutdown_timeout_seconds: int = Field(
        default=60, ge=15, le=300, description="Maximum time to wait for graceful bot shutdown"
    )

    # Resource management
    resource_monitoring_interval: int = Field(
        default=30, ge=10, le=300, description="Resource monitoring interval in seconds"
    )

    resource_cleanup_interval: int = Field(
        default=300, ge=60, le=3600, description="Resource cleanup interval in seconds"
    )

    # Coordination and signaling
    signal_retention_hours: int = Field(
        default=24, ge=1, le=168, description="Signal retention period in hours"
    )

    coordination_check_interval: int = Field(
        default=30, ge=5, le=300, description="Coordination check interval in seconds"
    )

    coordination_interval: int = Field(
        default=10, ge=1, le=60, description="Coordination interval between bots in seconds"
    )

    max_signal_recipients: int = Field(
        default=50, ge=1, le=1000, description="Maximum number of signal recipients"
    )

    signal_retention_minutes: int = Field(
        default=60, ge=5, le=1440, description="Signal retention period in minutes"
    )

    # Risk and exposure management
    max_symbol_exposure: Decimal = Field(
        default=Decimal("100000"), gt=0, description="Maximum exposure per symbol across all bots"
    )

    arbitrage_detection_enabled: bool = Field(
        default=True, description="Enable arbitrage opportunity detection"
    )

    # Performance and monitoring
    performance_window_minutes: int = Field(
        default=60, ge=5, le=1440, description="Performance monitoring window in minutes"
    )

    # Event and lifecycle management
    event_retention_hours: int = Field(
        default=168,  # 7 days
        ge=24,
        le=720,
        description="Event retention period in hours",
    )

    graceful_shutdown_timeout: int = Field(
        default=300, ge=60, le=600, description="Graceful shutdown timeout in seconds"
    )

    restart_max_attempts: int = Field(
        default=3, ge=1, le=10, description="Maximum restart attempts for failed bots"
    )

    restart_delay_seconds: int = Field(
        default=60, ge=10, le=300, description="Delay between restart attempts in seconds"
    )

    heartbeat_interval: int = Field(
        default=30, ge=10, le=120, description="Bot heartbeat interval in seconds"
    )

    position_timeout_minutes: int = Field(
        default=60, ge=5, le=1440, description="Position timeout in minutes"
    )

    max_restart_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum restart attempts (alias for restart_max_attempts)",
    )

    # Resource limits (nested configuration)
    resource_limits: dict = Field(
        default_factory=lambda: {
            "total_capital": 1000000,
            "max_api_requests_per_minute": 1000,
            "max_database_connections": 50,
            "max_memory_usage_mb": 2048,
            "max_cpu_usage_percent": 80,
        },
        description="Resource limits for bot operations",
    )

    # Alert thresholds
    alert_thresholds: dict = Field(
        default_factory=lambda: {
            "error_rate_percent": 5,
            "response_time_ms": 1000,
            "memory_usage_percent": 85,
            "cpu_usage_percent": 85,
        },
        description="Alert thresholds for monitoring",
    )

    def get_resource_limits(self) -> dict:
        """Get resource limits configuration."""
        return self.resource_limits.copy()

    def get_alert_thresholds(self) -> dict:
        """Get alert thresholds configuration."""
        return self.alert_thresholds.copy()

    def get_coordination_config(self) -> dict:
        """Get coordination-specific configuration."""
        return {
            "coordination_interval": self.coordination_interval,
            "signal_retention_minutes": self.signal_retention_minutes,
            "max_signal_recipients": self.max_signal_recipients,
            "arbitrage_detection_enabled": self.arbitrage_detection_enabled,
        }

    def get_lifecycle_config(self) -> dict:
        """Get lifecycle management configuration."""
        return {
            "event_retention_hours": self.event_retention_hours,
            "graceful_shutdown_timeout": self.graceful_shutdown_timeout,
            "restart_max_attempts": self.restart_max_attempts,
            "restart_delay_seconds": self.restart_delay_seconds,
        }

    def get_monitoring_config(self) -> dict:
        """Get monitoring configuration."""
        return {
            "health_check_interval_seconds": self.health_check_interval_seconds,
            "resource_monitoring_interval": self.resource_monitoring_interval,
            "performance_window_minutes": self.performance_window_minutes,
            "alert_thresholds": self.alert_thresholds,
        }

    model_config = {"validate_assignment": True, "extra": "forbid"}

    @field_serializer("max_capital_allocation", "max_symbol_exposure")
    def serialize_decimal(self, value):
        """Serialize Decimal fields to string for JSON compatibility."""
        return str(value)
