"""Bot management types for the T-Bot trading system."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

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
    strategy_id: Optional[str] = None
    strategy_config: Dict[str, Any] = Field(default_factory=dict)
    
    # Trading parameters
    symbols: List[str] = Field(default_factory=list)
    exchanges: List[str] = Field(default_factory=list)
    
    # Resource limits
    max_cpu_percent: float = 50.0
    max_memory_mb: int = 512
    max_api_calls_per_minute: int = 100
    max_websocket_connections: int = 5
    
    # Risk limits
    max_capital: Optional[Decimal] = None
    max_position_size: Optional[Decimal] = None
    max_daily_loss: Optional[Decimal] = None
    
    # Scheduling
    schedule: Optional[Dict[str, Any]] = None  # cron-like schedule
    active_hours: Optional[Dict[str, Any]] = None  # trading hours
    
    # Monitoring
    health_check_interval: int = 60  # seconds
    metrics_interval: int = 300  # seconds
    alert_settings: Dict[str, Any] = Field(default_factory=dict)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


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
    last_error: Optional[str] = None
    last_error_at: Optional[datetime] = None
    
    # Timing metrics
    avg_response_time_ms: float
    max_response_time_ms: float
    avg_processing_time_ms: float
    
    # Health score
    health_score: float = Field(ge=0.0, le=100.0)
    
    timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BotState(BaseModel):
    """Bot runtime state."""

    bot_id: str
    status: BotStatus
    priority: BotPriority
    
    # Runtime info
    pid: Optional[int] = None
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    last_heartbeat: Optional[datetime] = None
    
    # Current activity
    current_action: Optional[str] = None
    current_symbol: Optional[str] = None
    active_orders: List[str] = Field(default_factory=list)
    open_positions: List[str] = Field(default_factory=list)
    
    # Resource allocation
    allocated_capital: Decimal = Decimal("0")
    used_capital: Decimal = Decimal("0")
    
    # State persistence
    checkpoint: Optional[Dict[str, Any]] = None
    checkpoint_at: Optional[datetime] = None
    
    # Error state
    error_count: int = 0
    last_error: Optional[str] = None
    error_recovery_attempts: int = 0
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


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
    burst_limit: Optional[float] = None
    
    # Usage tracking
    peak_usage: float
    avg_usage: float
    total_consumed: float
    
    # Time windows
    measurement_window: int  # seconds
    reset_at: Optional[datetime] = None
    
    # Throttling
    is_throttled: bool = False
    throttle_until: Optional[datetime] = None
    
    updated_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BotEvent(BaseModel):
    """Bot lifecycle and operational events."""

    event_id: str
    bot_id: str
    event_type: str  # "started", "stopped", "error", "trade", etc.
    event_severity: str  # "info", "warning", "error", "critical"
    
    # Event details
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    
    # Context
    triggered_by: Optional[str] = None  # user, system, scheduler
    related_order: Optional[str] = None
    related_position: Optional[str] = None
    
    # Timing
    occurred_at: datetime
    processed_at: Optional[datetime] = None
    
    # Response
    action_taken: Optional[str] = None
    action_result: Optional[str] = None
    
    metadata: Dict[str, Any] = Field(default_factory=dict)