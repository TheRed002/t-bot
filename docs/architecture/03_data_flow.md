# T-Bot Data Flow Architecture

## Market Data Flow

```mermaid
graph LR
    subgraph "External Sources"
        BINANCE[Binance API]
        COINBASE[Coinbase API]
        OKX[OKX API]
    end

    subgraph "Data Ingestion"
        WS[WebSocket Streams]
        REST[REST Polling]
        VALIDATOR[Market Data Validator]
    end

    subgraph "Processing"
        STREAM[StreamingDataService]
        TRANSFORM[Data Transformer]
        CACHE[Redis Cache]
    end

    subgraph "Storage"
        TS[(TimescaleDB)]
        PG[(PostgreSQL)]
    end

    subgraph "Consumers"
        STRAT[Strategy Engine]
        RISK[Risk Manager]
        ANALYTICS[Analytics]
        WEB[Web Interface]
    end

    BINANCE --> WS
    COINBASE --> WS
    OKX --> REST

    WS --> VALIDATOR
    REST --> VALIDATOR

    VALIDATOR --> STREAM
    STREAM --> TRANSFORM
    TRANSFORM --> CACHE
    TRANSFORM --> TS

    CACHE --> STRAT
    CACHE --> RISK
    TS --> ANALYTICS
    CACHE --> WEB
```

## Data Processing Pipeline

### 1. Real-Time Market Data Flow

```mermaid
sequenceDiagram
    participant Exchange
    participant WebSocket
    participant StreamingService
    participant Validator
    participant Cache
    participant Strategy
    participant UI

    Exchange->>WebSocket: Price Update
    WebSocket->>StreamingService: Raw Data
    StreamingService->>Validator: Validate Data

    alt Valid Data
        Validator->>Cache: Store in Redis
        Validator->>StreamingService: Confirmed
        StreamingService-->>Strategy: Notify Update
        StreamingService-->>UI: WebSocket Broadcast
        Strategy->>Cache: Get Latest Data
        Cache-->>Strategy: Return Data
    else Invalid Data
        Validator->>StreamingService: Reject
        StreamingService->>WebSocket: Reconnect
    end
```

### 2. Historical Data Flow

```mermaid
graph TD
    subgraph "Data Collection"
        FETCH[Fetch Historical Data]
        VALIDATE[Validate & Clean]
        STORE[Store in TimescaleDB]
    end

    subgraph "Data Processing"
        AGG[Create Aggregates]
        COMP[Compress Old Data]
        INDEX[Update Indexes]
    end

    subgraph "Data Access"
        BACK[Backtesting Engine]
        OPT[Optimization Engine]
        ANALYTICS[Analytics Service]
    end

    FETCH --> VALIDATE
    VALIDATE --> STORE
    STORE --> AGG
    AGG --> COMP
    COMP --> INDEX

    INDEX --> BACK
    INDEX --> OPT
    INDEX --> ANALYTICS
```

## Data Models & Structures

### Market Data Structure (Actual from Code)
```python
# From src/core/types/market.py
class MarketData:
    symbol: str
    exchange: str
    timestamp: datetime
    bid: Decimal
    ask: Decimal
    last_price: Decimal
    volume_24h: Decimal
    bid_size: Decimal
    ask_size: Decimal
```

### Order Flow
```python
# From src/core/types/trading.py
class Order:
    order_id: UUID
    bot_id: UUID
    symbol: str
    side: OrderSide  # BUY/SELL
    order_type: OrderType  # MARKET/LIMIT
    quantity: Decimal
    price: Decimal | None
    status: OrderStatus
    timestamp: datetime
```

## Data Storage Layers

### 1. Hot Storage (Redis Cache)
```mermaid
graph LR
    subgraph "Redis Cache Structure"
        PRICE[Price Cache<br/>TTL: 1s]
        ORDER[Order Book<br/>TTL: 5s]
        POS[Positions<br/>TTL: 30s]
        METRICS[Metrics<br/>TTL: 60s]
    end

    subgraph "Access Patterns"
        HIGH[High Frequency<br/>Price Updates]
        MED[Medium Frequency<br/>Order Updates]
        LOW[Low Frequency<br/>Position Queries]
    end

    HIGH --> PRICE
    MED --> ORDER
    LOW --> POS
    LOW --> METRICS
```

### 2. Warm Storage (PostgreSQL)
```
Tables:
├── trades            # Recent trade executions
├── orders            # Order history
├── positions         # Current positions
├── bot_instances     # Bot configurations
└── strategies        # Strategy parameters
```

### 3. Cold Storage (TimescaleDB)
```
Hypertables:
├── market_data       # Price history (compressed)
├── order_history     # Historical orders
├── trade_history     # Historical trades
└── performance_metrics # Bot performance over time
```

## Event-Driven Data Flow

```mermaid
graph TD
    subgraph "Event Publishers"
        BOT[Bot Events]
        MARKET[Market Events]
        ORDER[Order Events]
        RISK[Risk Events]
    end

    subgraph "Event Bus"
        BUS[AsyncIO Event Bus]
    end

    subgraph "Event Subscribers"
        LOG[Logging Service]
        MONITOR[Monitoring Service]
        ALERT[Alert Service]
        WS[WebSocket Manager]
    end

    BOT --> BUS
    MARKET --> BUS
    ORDER --> BUS
    RISK --> BUS

    BUS --> LOG
    BUS --> MONITOR
    BUS --> ALERT
    BUS --> WS
```

## Data Validation Pipeline

```mermaid
flowchart LR
    INPUT[Raw Data Input]

    subgraph "Validation Stages"
        V1[Schema Validation<br/>Pydantic Models]
        V2[Business Rules<br/>Price Bounds]
        V3[Consistency Check<br/>Timestamps]
        V4[Duplicate Detection<br/>Deduplication]
    end

    OUTPUT[Clean Data]
    ERROR[Error Handler]

    INPUT --> V1
    V1 -->|Valid| V2
    V1 -->|Invalid| ERROR
    V2 -->|Valid| V3
    V2 -->|Invalid| ERROR
    V3 -->|Valid| V4
    V3 -->|Invalid| ERROR
    V4 -->|Valid| OUTPUT
    V4 -->|Duplicate| ERROR
```

## Real-Time Data Synchronization

```mermaid
sequenceDiagram
    participant Bot1 as Bot Instance 1
    participant Bot2 as Bot Instance 2
    participant Redis
    participant EventBus
    participant DB

    Bot1->>Redis: Update Position
    Redis->>EventBus: Publish Event
    EventBus-->>Bot2: Notify Update
    Bot2->>Redis: Get Latest Position

    Bot1->>DB: Log Trade
    DB->>EventBus: Publish Event
    EventBus-->>Bot2: Sync Trade History
```

## Performance Metrics Flow

```mermaid
graph TB
    subgraph "Metrics Collection"
        CPU[CPU Usage]
        MEM[Memory Usage]
        NET[Network I/O]
        TRADES[Trade Metrics]
    end

    subgraph "Aggregation"
        AGG[Metrics Aggregator<br/>1min intervals]
        CALC[Calculate Averages]
    end

    subgraph "Storage & Display"
        INFLUX[(InfluxDB)]
        GRAF[Grafana]
        API[Metrics API]
    end

    CPU --> AGG
    MEM --> AGG
    NET --> AGG
    TRADES --> AGG

    AGG --> CALC
    CALC --> INFLUX
    INFLUX --> GRAF
    CALC --> API
```

## Data Consistency Guarantees

| Layer | Consistency Model | Implementation |
|-------|------------------|----------------|
| **Cache** | Eventually Consistent | TTL-based expiration |
| **Database** | Strong Consistency | ACID transactions |
| **Event Bus** | At-least-once delivery | Retry with idempotency |
| **WebSocket** | Best-effort | Reconnect on failure |
| **TimescaleDB** | Strong Consistency | Hypertable constraints |

---

## Next Steps

**Continue exploring:**

1. **[Trading Workflow](04_trading_workflow.md)** - Complete trade execution flow
2. **[Technology Stack](05_technology_stack.md)** - Technologies and frameworks
3. **[Module Structure](02_module_structure.md)** - Back to module details
4. **[Back to Overview](00_overview.md)** - Return to index

What would you like to explore next? (Choose 1-4)