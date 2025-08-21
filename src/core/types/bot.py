"""Bot management types for the T-Bot trading system."""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

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
    strategy_config: dict[str, Any] = Field(default_factory=dict)

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
    max_position_size: Decimal | None = None
    max_daily_loss: Decimal | None = None

    # Scheduling
    schedule: dict[str, Any] | None = None  # cron-like schedule
    active_hours: dict[str, Any] | None = None  # trading hours

    # Monitoring
    health_check_interval: int = 60  # seconds
    metrics_interval: int = 300  # seconds
    alert_settings: dict[str, Any] = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)


class BotMetrics(BaseModel):
    """Bot performance and resource metrics."""

    bot_id: str

    # Performance metrics
    uptime_seconds: int
    total_trades: int
    successful_trades: int
    failed_trades: int
    total_pnl: Decimal

    # Resource usage
    cpu_usage_percent: float
    memory_usage_mb: float
    network_bytes_sent: int
    network_bytes_received: int
    disk_usage_mb: float

    # API usage
    api_calls_made: int
    api_calls_remaining: int
    api_errors: int

    # WebSocket metrics
    ws_connections_active: int
    ws_messages_received: int
    ws_messages_sent: int
    ws_reconnections: int

    # Error metrics
    total_errors: int
    critical_errors: int
    last_error: str | None = None
    last_error_at: datetime | None = None

    # Timing metrics
    avg_response_time_ms: float
    max_response_time_ms: float
    avg_processing_time_ms: float

    # Health score
    health_score: float = Field(ge=0.0, le=100.0)

    timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class BotState(BaseModel):
    """Bot runtime state."""

    bot_id: str
    status: BotStatus
    priority: BotPriority

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
    allocated_amount: float
    used_amount: float
    available_amount: float
    utilization_percent: float

    # Quotas
    soft_limit: float
    hard_limit: float
    burst_limit: float | None = None

    # Usage tracking
    peak_usage: float
    avg_usage: float
    total_consumed: float

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
