# T-Bot System Architecture

## High-Level Architecture

```mermaid
graph TB
    subgraph "External Layer"
        EX1[Binance API]
        EX2[Coinbase API]
        EX3[Other Exchanges]
        MD[Market Data Providers]
    end

    subgraph "API Gateway"
        WS[WebSocket Server]
        REST[REST API<br/>FastAPI]
    end

    subgraph "Application Core"
        BM[Bot Manager]
        SM[Strategy Manager]
        EM[Execution Manager]
        RM[Risk Manager]
    end

    subgraph "Data Layer"
        DS[Data Service]
        CS[Cache Service]
        PS[Persistence Service]
    end

    subgraph "Storage"
        PG[(PostgreSQL<br/>TimescaleDB)]
        RD[(Redis Cache)]
    end

    subgraph "Monitoring"
        MON[Health Monitor]
        LOG[Logging Service]
        ALERT[Alert Manager]
    end

    EX1 --> DS
    EX2 --> DS
    EX3 --> DS
    MD --> DS

    WS --> BM
    REST --> BM

    BM --> SM
    BM --> EM
    BM --> RM

    SM --> DS
    EM --> DS
    RM --> DS

    DS --> CS
    DS --> PS

    CS --> RD
    PS --> PG

    BM --> MON
    SM --> LOG
    EM --> ALERT
```

## System Components

### 1. External Integration Layer
- **Exchange Adapters**: Standardized interface for multiple exchanges
- **Market Data Feed**: Real-time price and volume data
- **Order Routing**: Smart order routing across exchanges

### 2. Application Core
- **Bot Manager**: Lifecycle management of trading bots
- **Strategy Manager**: Strategy selection and parameter optimization
- **Execution Manager**: Order placement and management
- **Risk Manager**: Position sizing, stop-loss, exposure control

### 3. Data Management
- **Data Service**: Centralized data access and transformation
- **Cache Service**: Redis-based caching for performance
- **Persistence Service**: Database operations via repositories

### 4. Infrastructure
- **PostgreSQL + TimescaleDB**: Time-series data storage
- **Redis**: Caching and real-time data
- **Event Bus**: Asynchronous message passing

## Communication Patterns

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant BotMgr as Bot Manager
    participant Strategy
    participant Execution
    participant Exchange
    participant DB

    Client->>API: Create Bot Request
    API->>BotMgr: Initialize Bot
    BotMgr->>Strategy: Load Strategy
    BotMgr->>DB: Save Bot Config

    loop Trading Loop
        Strategy->>Strategy: Analyze Market
        Strategy->>BotMgr: Generate Signal
        BotMgr->>Execution: Execute Trade
        Execution->>Exchange: Place Order
        Exchange-->>Execution: Order Status
        Execution->>DB: Log Trade
        Execution-->>BotMgr: Update Status
        BotMgr-->>API: Status Update
        API-->>Client: WebSocket Update
    end
```

## Deployment Architecture

```mermaid
graph LR
    subgraph "Production Environment"
        LB[Load Balancer]

        subgraph "Application Servers"
            APP1[T-Bot Instance 1]
            APP2[T-Bot Instance 2]
        end

        subgraph "Data Tier"
            PG_M[(PostgreSQL Master)]
            PG_R[(PostgreSQL Replica)]
            REDIS_C[(Redis Cluster)]
        end

        subgraph "Monitoring"
            PROM[Prometheus]
            GRAF[Grafana]
        end
    end

    LB --> APP1
    LB --> APP2
    APP1 --> PG_M
    APP2 --> PG_M
    PG_M --> PG_R
    APP1 --> REDIS_C
    APP2 --> REDIS_C
    APP1 --> PROM
    APP2 --> PROM
    PROM --> GRAF
```

## Security Architecture

```mermaid
graph TB
    subgraph "Security Layers"
        FW[Firewall]
        WAF[Web Application Firewall]

        subgraph "Authentication"
            JWT[JWT Tokens]
            OAuth[OAuth2]
            API_KEY[API Key Management]
        end

        subgraph "Encryption"
            TLS[TLS/SSL]
            ENC_DB[Database Encryption]
            ENC_CACHE[Cache Encryption]
        end

        subgraph "Access Control"
            RBAC[Role-Based Access]
            IP_WL[IP Whitelist]
            RATE[Rate Limiting]
        end
    end

    FW --> WAF
    WAF --> JWT
    JWT --> RBAC
    RBAC --> TLS
    TLS --> ENC_DB
```

---

## Next Steps

**Continue exploring:**

1. **[Module Structure](02_module_structure.md)** - Detailed code organization
2. **[Data Flow](03_data_flow.md)** - How data moves through the system
3. **[Trading Workflow](04_trading_workflow.md)** - Complete trade execution
4. **[Technology Stack](05_technology_stack.md)** - Tools and frameworks
5. **[Back to Overview](00_overview.md)** - Return to index

What would you like to explore next? (Choose 1-5)