"""Bot management types for the T-Bot trading system."""

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

from pydantic import BaseModel, Field


class BotStatus(Enum):
    """Bot operational status."""

    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class BotType(Enum):
    """Bot type classification."""

    TRADING = "trading"
    ARBITRAGE = "arbitrage"
    MARKET_MAKING = "market_making"
    LIQUIDATION = "liquidation"
    REBALANCING = "rebalancing"
    DATA_COLLECTION = "data_collection"
    MONITORING = "monitoring"
    TESTING = "testing"


class BotPriority(Enum):
    """Bot execution priority."""

    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    IDLE = "idle"


class ResourceType(Enum):
    """System resource types."""

    CPU = "cpu"
    MEMORY = "memory"
    NETWORK = "network"
    DISK = "disk"
    API_CALLS = "api_calls"
    WEBSOCKET_CONNECTIONS = "websocket_connections"
    DATABASE_CONNECTIONS = "database_connections"
    CAPITAL = "capital"


class BotConfiguration(BaseModel):
    """Bot configuration parameters."""

    bot_id: str
    bot_type: BotType
    name: str
    version: str

    # Operational settings
    enabled: bool = True
    auto_start: bool = False
    auto_restart: bool = True
    restart_delay: int = 60  # seconds
    max_restarts: int = 3

    # Strategy assignment
    strategy_id: str | None = None
    strategy_name: str | None = None  # Legacy attribute for compatibility
    strategy_config: dict[str, Any] = Field(default_factory=dict)
    strategy_parameters: dict[str, Any] = Field(default_factory=dict)  # Legacy attribute

    # Trading parameters
    symbols: list[str] = Field(default_factory=list)
    exchanges: list[str] = Field(default_factory=list)

    # Resource limits
    max_cpu_percent: float = 50.0
    max_memory_mb: int = 512
    max_api_calls_per_minute: int = 100
    max_websocket_connections: int = 5

    # Risk limits
    max_capital: Decimal | None = None
    allocated_capital: Decimal = Field(default=Decimal("0"))  # Legacy attribute for compatibility
    max_position_size: Decimal | None = None
    max_daily_loss: Decimal | None = None
    risk_percentage: float = Field(default=0.02, ge=0.0, le=1.0)  # Legacy attribute

    # Scheduling
    schedule: dict[str, Any] | None = None  # cron-like schedule
    active_hours: dict[str, Any] | None = None  # trading hours

    # Monitoring
    health_check_interval: int = 60  # seconds
    heartbeat_interval: int = 60  # Legacy attribute
    metrics_interval: int = 300  # seconds
    alert_settings: dict[str, Any] = Field(default_factory=dict)

    # Priority setting
    priority: BotPriority = BotPriority.NORMAL

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def bot_name(self) -> str:
        """Legacy property alias for name."""
        return self.name


class BotMetrics(BaseModel):
    """Bot performance and resource metrics."""

    bot_id: str

    # Performance metrics
    uptime_seconds: int = 0
    total_trades: int = 0
    successful_trades: int = 0
    failed_trades: int = 0
    total_pnl: Decimal = Decimal("0")

    # Legacy attributes for compatibility
    profitable_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    average_trade_pnl: Decimal = Decimal("0")
    start_time: datetime | None = None
    uptime_percentage: float = 0.0
    last_trade_time: datetime | None = None
    last_heartbeat: datetime | None = None
    error_count: int = 0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0  # MB
    metrics_updated_at: datetime | None = None

    # Resource usage
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_received: int = 0
    disk_usage_mb: float = 0.0

    # API usage
    api_calls_made: int = 0
    api_calls_remaining: int = 0
    api_errors: int = 0

    # WebSocket metrics
    ws_connections_active: int = 0
    ws_messages_received: int = 0
    ws_messages_sent: int = 0
    ws_reconnections: int = 0

    # Error metrics
    total_errors: int = 0
    critical_errors: int = 0
    last_error: str | None = None
    last_error_at: datetime | None = None

    # Timing metrics
    avg_response_time_ms: float = 0.0
    max_response_time_ms: float = 0.0
    avg_processing_time_ms: float = 0.0

    # Health score
    health_score: float = Field(default=100.0, ge=0.0, le=100.0)

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class BotState(BaseModel):
    """Bot runtime state."""

    bot_id: str
    status: BotStatus = BotStatus.INITIALIZING
    priority: BotPriority = BotPriority.NORMAL
    configuration: "BotConfiguration | None" = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Runtime info
    pid: int | None = None
    started_at: datetime | None = None
    stopped_at: datetime | None = None
    last_heartbeat: datetime | None = None

    # Current activity
    current_action: str | None = None
    current_symbol: str | None = None
    active_orders: list[str] = Field(default_factory=list)
    open_positions: list[str] = Field(default_factory=list)

    # Resource allocation
    allocated_capital: Decimal = Decimal("0")
    used_capital: Decimal = Decimal("0")

    # State persistence
    checkpoint: dict[str, Any] | None = None
    checkpoint_at: datetime | None = None
    checkpoint_created: datetime | None = None  # Legacy attribute
    state_version: int = 1  # Legacy attribute

    # Error state
    error_count: int = 0
    last_error: str | None = None
    error_recovery_attempts: int = 0

    metadata: dict[str, Any] = Field(default_factory=dict)


class ResourceAllocation(BaseModel):
    """Resource allocation for bots."""

    bot_id: str
    resource_type: ResourceType

    # Allocation limits
    allocated_amount: Decimal
    used_amount: Decimal
    available_amount: Decimal
    utilization_percent: Decimal

    # Quotas
    soft_limit: Decimal
    hard_limit: Decimal
    burst_limit: Decimal | None = None

    # Usage tracking
    peak_usage: Decimal
    avg_usage: Decimal
    total_consumed: Decimal

    # Time windows
    measurement_window: int  # seconds
    reset_at: datetime | None = None

    # Throttling
    is_throttled: bool = False
    throttle_until: datetime | None = None

    updated_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class BotEvent(BaseModel):
    """Bot lifecycle and operational events."""

    event_id: str
    bot_id: str
    event_type: str  # "started", "stopped", "error", "trade", etc.
    event_severity: str  # "info", "warning", "error", "critical"

    # Event details
    message: str
    details: dict[str, Any] = Field(default_factory=dict)

    # Context
    triggered_by: str | None = None  # user, system, scheduler
    related_order: str | None = None
    related_position: str | None = None

    # Timing
    occurred_at: datetime
    processed_at: datetime | None = None

    # Response
    action_taken: str | None = None
    action_result: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)
