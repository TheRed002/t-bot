/**
 * Core type definitions for T-Bot Trading System
 * Shared interfaces and types used throughout the application
 */

// Authentication types
export interface User {
  user_id: string;
  username: string;
  email: string;
  full_name?: string;
  is_active: boolean;
  status: UserStatus;
  roles: string[];
  scopes: string[];
  preferences?: UserPreferences;
  created_at: string;
  last_login?: string;
  allocated_capital?: number;
  max_daily_loss?: number;
  risk_level?: string;
}

export enum UserStatus {
  ACTIVE = 'active',
  INACTIVE = 'inactive',
  SUSPENDED = 'suspended',
  LOCKED = 'locked',
  PENDING_VERIFICATION = 'pending_verification'
}

export interface UserPreferences {
  theme: 'dark' | 'light';
  currency: string;
  timezone: string;
  notifications: NotificationSettings;
}

export interface NotificationSettings {
  email: boolean;
  push: boolean;
  trades: boolean;
  alerts: boolean;
  systemUpdates: boolean;
}

export interface AuthTokens {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

export interface LoginCredentials {
  username: string;
  password: string;
  remember_me?: boolean;
}

export interface LoginResponse {
  success: boolean;
  message: string;
  user: User;
  tokens: AuthTokens;
}

export interface RefreshTokenResponse {
  success: boolean;
  message: string;
  user: User;
  tokens: AuthTokens;
}

export interface AuthState {
  user: User | null;
  tokens: AuthTokens | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  isRefreshing: boolean;
  error: string | null;
  rememberMe: boolean;
  sessionExpiresAt: number | null;
}

// Bot Management types - Backend Compatible

/**
 * Backend BotStatus enum values.
 * Maps directly to Python BotStatus enum in core.types.bot
 */
export enum BotStatus {
  INITIALIZING = 'initializing',
  READY = 'ready',
  RUNNING = 'running',
  PAUSED = 'paused',
  STOPPING = 'stopping',
  STOPPED = 'stopped',
  ERROR = 'error',
  MAINTENANCE = 'maintenance'
}

/**
 * Backend BotType enum values.
 * Maps directly to Python BotType enum in core.types.bot
 */
export enum BotType {
  TRADING = 'trading',
  ARBITRAGE = 'arbitrage',
  MARKET_MAKING = 'market_making',
  LIQUIDATION = 'liquidation',
  REBALANCING = 'rebalancing',
  DATA_COLLECTION = 'data_collection',
  MONITORING = 'monitoring',
  TESTING = 'testing'
}

/**
 * Backend BotPriority enum values.
 * Maps directly to Python BotPriority enum in core.types.bot
 */
export enum BotPriority {
  CRITICAL = 'critical',
  HIGH = 'high',
  NORMAL = 'normal',
  LOW = 'low',
  IDLE = 'idle'
}

/**
 * Bot configuration matching backend BotConfiguration model.
 * Maps directly to Python BotConfiguration in core.types.bot
 */
export interface BotConfiguration {
  bot_id: string;
  bot_name: string;  // Backend uses bot_name, not name
  bot_type: BotType;
  strategy_name: StrategyType;  // Backend uses strategy_name with StrategyType enum
  exchanges: string[];
  symbols: string[];
  allocated_capital: number;
  risk_percentage: number;
  priority: BotPriority;
  auto_start: boolean;
  strategy_config: Record<string, any>;  // Backend uses strategy_config
  created_at: string;
  updated_at?: string;
}

/**
 * Bot metrics matching backend BotMetrics model.
 * Maps directly to Python BotMetrics in core.types.bot
 */
export interface BotMetrics {
  bot_id: string;
  uptime_seconds: number;
  total_trades: number;
  successful_trades: number;  // Backend uses successful_trades, not winningTrades
  failed_trades: number;     // Backend uses failed_trades, not losingTrades
  total_pnl: number;
  cpu_usage_percent: number;
  memory_usage_mb: number;
  win_rate: number;         // Calculated field: successful_trades / total_trades
  health_score: number;
  timestamp: string;
}

/**
 * Bot instance interface matching backend structure.
 * Uses backend field names for consistency
 */
export interface BotInstance {
  bot_id: string;           // Backend uses bot_id, not id
  bot_name: string;         // Backend uses bot_name, not name
  userId: string;
  strategy_name: StrategyType;    // Backend uses strategy_name with StrategyType enum
  exchange: string;
  status: BotStatus;
  config: BotConfiguration;
  createdAt: string;
  updatedAt: string;
  metrics?: BotMetrics;     // Changed from performance to metrics
}

// Legacy type aliases for backward compatibility
/** @deprecated Use BotConfiguration instead */
export interface BotConfig extends Omit<BotConfiguration, 'bot_id' | 'bot_name' | 'strategy_name' | 'exchanges' | 'symbols' | 'allocated_capital' | 'risk_percentage' | 'priority' | 'auto_start' | 'strategy_config' | 'created_at' | 'updated_at'> {
  symbol: string;          // Legacy single symbol
  baseAmount: number;      // Legacy field name
  maxPosition: number;
  riskPercentage: number;  // Legacy camelCase
  stopLoss?: number;
  takeProfit?: number;
  strategyParams: Record<string, any>;  // Legacy field name
}

/** @deprecated Use BotMetrics instead */
export interface BotPerformance {
  totalTrades: number;
  winningTrades: number;   // Legacy field name
  losingTrades: number;    // Legacy field name
  totalPnl: number;
  winRate: number;
  maxDrawdown: number;
  sharpeRatio?: number;
  startDate: string;
  lastTradeAt?: string;
}

/**
 * Bot summary response from backend API.
 * Maps directly to BotSummaryResponse in web_interface.api.bot_management
 */
export interface BotSummaryResponse {
  bot_id: string;
  bot_name: string;
  status: string;
  allocated_capital: number;
  current_pnl?: number;
  total_trades?: number;
  win_rate?: number;
  last_trade?: string;
  uptime?: string;
}

/**
 * Bot list response from backend API.
 * Maps directly to BotListResponse in web_interface.api.bot_management
 */
export interface BotListResponse {
  bots: BotSummaryResponse[];
  total: number;
  running: number;
  stopped: number;
  error: number;
}

/**
 * Bot creation request matching backend API.
 * Maps directly to CreateBotRequest in web_interface.api.bot_management
 */
export interface CreateBotRequest {
  bot_name: string;
  bot_type: BotType;
  strategy_name: StrategyType;
  exchanges: string[];
  symbols: string[];
  allocated_capital: number;
  risk_percentage: number;
  priority?: BotPriority;
  auto_start?: boolean;
  configuration?: Record<string, any>;
}

/**
 * Bot update request matching backend API.
 * Maps directly to UpdateBotRequest in web_interface.api.bot_management
 */
export interface UpdateBotRequest {
  bot_name?: string;
  allocated_capital?: number;
  risk_percentage?: number;
  priority?: BotPriority;
  configuration?: Record<string, any>;
}

/**
 * Frontend bot state management
 */
export interface BotState {
  bots: BotInstance[];
  selectedBot: BotInstance | null;
  botList?: BotListResponse;  // Backend response structure
  isLoading: boolean;
  error: string | null;
  filters: BotFilters;
}

export interface BotFilters {
  status?: BotStatus[];
  exchange?: string[];
  strategy?: string[];
  searchTerm?: string;
}

// Portfolio types
export interface Position {
  id: string;
  botId: string;
  exchange: string;
  symbol: string;
  side: 'long' | 'short';
  quantity: number;
  entryPrice: number;
  currentPrice: number;
  unrealizedPnl: number;
  realizedPnl: number;
  createdAt: string;
  updatedAt: string;
}

export interface Balance {
  currency: string;
  exchange: string;
  free: number;
  locked: number;
  total: number;
  usdValue: number;
}

export interface PortfolioSummary {
  totalValue: number;
  totalPnl: number;
  dailyPnl: number;
  dailyPnlPercentage: number;
  weeklyPnl: number;
  weeklyPnlPercentage: number;
  monthlyPnl: number;
  monthlyPnlPercentage: number;
  positions: Position[];
  balances: Balance[];
  openOrders: number;
  winRate: number;
  sharpeRatio: number;
  maxDrawdown: number;
  lastUpdated: string;
}

export interface PortfolioState {
  summary: PortfolioSummary | null;
  positions: Position[];
  balances: Balance[];
  isLoading: boolean;
  error: string | null;
  filters: PortfolioFilters;
}

export interface PortfolioFilters {
  exchange?: string[];
  currency?: string[];
  positionType?: ('long' | 'short')[];
  showZeroBalances: boolean;
}

// Trading types
export interface Trade {
  id: string;
  botId: string;
  exchange: string;
  symbol: string;
  side: 'buy' | 'sell';
  orderType: string;
  quantity: number;
  price: number;
  fee: number;
  feeCurrency: string;
  pnl?: number;
  executedAt: string;
  orderId: string;
}

export interface Order {
  id: string;
  botId: string;
  exchange: string;
  symbol: string;
  side: 'buy' | 'sell';
  orderType: string;
  quantity: number;
  price?: number;
  status: OrderStatus;
  filled: number;
  remaining: number;
  createdAt: string;
  updatedAt: string;
}

export type OrderStatus = 'pending' | 'open' | 'filled' | 'partially_filled' | 'cancelled' | 'rejected';

// Complete Strategy Type System - Backend Compatible

/**
 * Complete strategy type enumeration matching backend StrategyType.
 * Maps directly to Python StrategyType enum in core.types.strategy
 */
export enum StrategyType {
  // Static Strategies
  MEAN_REVERSION = 'mean_reversion',
  MOMENTUM = 'momentum', 
  ARBITRAGE = 'arbitrage',
  MARKET_MAKING = 'market_making',
  TREND_FOLLOWING = 'trend_following',
  PAIRS_TRADING = 'pairs_trading',
  STATISTICAL_ARBITRAGE = 'statistical_arbitrage',
  BREAKOUT = 'breakout',
  CROSS_EXCHANGE_ARBITRAGE = 'cross_exchange_arbitrage',
  TRIANGULAR_ARBITRAGE = 'triangular_arbitrage',
  
  // Dynamic Strategies
  VOLATILITY_BREAKOUT = 'volatility_breakout',
  
  // Hybrid Strategies
  ENSEMBLE = 'ensemble',
  FALLBACK = 'fallback',
  RULE_BASED_AI = 'rule_based_ai',
  
  // Custom
  CUSTOM = 'custom'
}

/**
 * Strategy categories for organization and filtering.
 */
export enum StrategyCategory {
  STATIC = 'static',
  DYNAMIC = 'dynamic', 
  HYBRID = 'hybrid',
  EVOLUTIONARY = 'evolutionary',
  CUSTOM = 'custom'
}

/**
 * Strategy operational status matching backend.
 */
export enum StrategyStatus {
  INACTIVE = 'inactive',
  STARTING = 'starting',
  ACTIVE = 'active',
  PAUSED = 'paused',
  STOPPING = 'stopping',
  STOPPED = 'stopped',
  ERROR = 'error'
}

/**
 * Risk levels for strategies.
 */
export enum RiskLevel {
  CONSERVATIVE = 'conservative',
  MODERATE = 'moderate',
  AGGRESSIVE = 'aggressive',
  EXPERIMENTAL = 'experimental'
}

/**
 * Position sizing methods.
 */
export enum PositionSizeMethod {
  FIXED = 'fixed',
  PERCENTAGE = 'percentage',
  KELLY_CRITERION = 'kelly_criterion',
  RISK_PARITY = 'risk_parity',
  VOLATILITY_ADJUSTED = 'volatility_adjusted'
}

/**
 * Capital allocation strategies.
 */
export enum AllocationStrategy {
  EQUAL_WEIGHT = 'equal_weight',
  RISK_PARITY = 'risk_parity',
  MARKET_CAP = 'market_cap',
  MOMENTUM = 'momentum',
  MEAN_REVERSION = 'mean_reversion',
  CUSTOM = 'custom'
}

/**
 * Parameter validation types.
 */
export type ParameterType = 'number' | 'string' | 'boolean' | 'select' | 'multi_select' | 'range' | 'percentage' | 'currency' | 'timeframe';

/**
 * Strategy parameter definition with comprehensive validation.
 */
export interface StrategyParameterDefinition {
  name: string;
  displayName: string;
  type: ParameterType;
  value: any;
  defaultValue: any;
  description: string;
  required: boolean;
  
  // Validation rules
  validation?: {
    min?: number;
    max?: number;
    step?: number;
    pattern?: string;
    customValidator?: (value: any) => string | null;
  };
  
  // For select/multi_select types
  options?: Array<{
    label: string;
    value: any;
    description?: string;
    disabled?: boolean;
  }>;
  
  // Conditional display
  dependsOn?: {
    parameter: string;
    value: any;
    condition?: 'equals' | 'not_equals' | 'greater_than' | 'less_than' | 'in' | 'not_in';
  };
  
  // Parameter grouping
  group?: string;
  order?: number;
  
  // Help and documentation
  helpText?: string;
  tooltipText?: string;
  docUrl?: string;
}

/**
 * Risk management configuration.
 */
export interface RiskConfiguration {
  maxDrawdownPercentage: number;
  maxRiskPerTrade: number;
  stopLossPercentage?: number;
  takeProfitPercentage?: number;
  positionSizeMethod: PositionSizeMethod;
  maxPositions: number;
  correlationLimit: number;
  volatilityLimit?: number;
  
  // Circuit breakers
  enableCircuitBreaker: boolean;
  dailyLossLimit?: number;
  weeklyLossLimit?: number;
  consecutiveLossLimit?: number;
  
  // Position sizing parameters
  positionSizeConfig: {
    fixedAmount?: number;
    percentageOfCapital?: number;
    kellyFraction?: number;
    maxKellyFraction?: number;
    riskParityConfig?: {
      lookbackPeriod: number;
      rebalanceFrequency: string;
    };
  };
}

/**
 * Strategy template from backend configuration templates.
 */
export interface StrategyTemplate {
  id: string;
  name: string;
  displayName: string;
  strategyType: StrategyType;
  category: StrategyCategory;
  description: string;
  riskLevel: RiskLevel;
  
  // Requirements
  minimumCapital: number;
  supportedExchanges: string[];
  supportedSymbols: string[];
  recommendedTimeframes: string[];
  
  // Parameters
  parameters: StrategyParameterDefinition[];
  riskConfiguration: RiskConfiguration;
  
  // Backtesting configuration
  backtestingConfig?: {
    enabled: boolean;
    startDate: string;
    endDate: string;
    initialCapital: number;
    commission: number;
    slippage: number;
  };
  
  // Performance monitoring
  monitoringConfig?: {
    enabled: boolean;
    alertThresholds: {
      drawdownPercentage: number;
      consecutiveLosses: number;
      dailyPnlThreshold: number;
    };
    reportingFrequency: string;
  };
  
  // Template metadata
  version: string;
  author: string;
  createdAt: string;
  updatedAt?: string;
  tags: string[];
  isProduction: boolean;
}

/**
 * Strategy configuration for deployment.
 */
export interface StrategyConfiguration {
  id?: string;
  templateId: string;
  name: string;
  strategyType: StrategyType;
  status: StrategyStatus;
  
  // Deployment settings
  exchanges: string[];
  symbols: string[];
  timeframes: string[];
  
  // Capital allocation
  allocatedCapital: number;
  allocationStrategy: AllocationStrategy;
  
  // Parameters (with resolved values)
  parameters: Record<string, any>;
  riskConfiguration: RiskConfiguration;
  
  // Execution settings
  executionConfig: {
    maxSlippagePercentage: number;
    orderTimeoutSeconds: number;
    retryAttempts: number;
    fillStrategy: 'market' | 'limit' | 'smart';
  };
  
  // Monitoring
  monitoringEnabled: boolean;
  
  // Metadata
  createdAt: string;
  updatedAt?: string;
  createdBy: string;
  lastModifiedBy?: string;
}

/**
 * Updated Strategy interface with enhanced typing.
 */
export interface Strategy {
  id: string;
  name: string;
  strategyType: StrategyType;
  category: StrategyCategory;
  status: StrategyStatus;
  description: string;
  
  // Template reference
  templateId?: string;
  template?: StrategyTemplate;
  
  // Configuration
  configuration: StrategyConfiguration;
  
  // Performance data
  performance?: StrategyPerformance;
  
  // Metadata
  isActive: boolean;
  createdAt: string;
  updatedAt?: string;
}

/**
 * Legacy parameter interface for backward compatibility.
 * @deprecated Use StrategyParameterDefinition instead
 */
export interface StrategyParameter {
  name: string;
  type: 'number' | 'string' | 'boolean' | 'select';
  value: any;
  defaultValue: any;
  min?: number;
  max?: number;
  step?: number;
  options?: Array<{ label: string; value: any }>;
  description: string;
  required: boolean;
}

export interface StrategyPerformance {
  backtestResults?: BacktestResult;
  livePerformance?: LivePerformance;
}

export interface BacktestResult {
  id: string;
  strategyId: string;
  startDate: string;
  endDate: string;
  totalReturn: number;
  annualizedReturn: number;
  volatility: number;
  sharpeRatio: number;
  maxDrawdown: number;
  totalTrades: number;
  winRate: number;
  profitFactor: number;
  createdAt: string;
}

export interface LivePerformance {
  totalPnl: number;
  dailyPnl: number;
  winRate: number;
  totalTrades: number;
  avgTradeSize: number;
  lastTradeAt?: string;
}

/**
 * Enhanced strategy state management.
 */
export interface StrategyState {
  // Templates
  templates: StrategyTemplate[];
  templatesLoading: boolean;
  templatesError: string | null;
  
  // Strategies
  strategies: Strategy[];
  selectedStrategy: Strategy | null;
  
  // Configuration
  activeConfiguration: StrategyConfiguration | null;
  configurationHistory: StrategyConfiguration[];
  
  // Backtesting
  backtestResults: BacktestResult[];
  isBacktesting: boolean;
  backtestProgress?: number;
  
  // UI State
  isLoading: boolean;
  error: string | null;
  
  // Filters
  filters: StrategyFilters;
}

/**
 * Strategy filtering options.
 */
export interface StrategyFilters {
  category?: StrategyCategory[];
  strategyType?: StrategyType[];
  status?: StrategyStatus[];
  riskLevel?: RiskLevel[];
  exchange?: string[];
  searchTerm?: string;
  sortBy?: 'name' | 'performance' | 'created_at' | 'risk_level';
  sortOrder?: 'asc' | 'desc';
}

// Enhanced Risk Management types
export interface RiskMetrics {
  portfolioValue: number;
  totalExposure: number;
  maxDrawdown: number;
  currentDrawdown: number;
  var95: number; // Value at Risk 95%
  var99: number; // Value at Risk 99%
  expectedShortfall: number; // Conditional VaR
  sharpeRatio: number;
  sortinoRatio: number;
  calmarRatio: number;
  correlation: Record<string, number>;
  beta: number;
  
  // Position-level metrics
  positionMetrics: {
    totalPositions: number;
    longPositions: number;
    shortPositions: number;
    netExposure: number;
    grossExposure: number;
  };
  
  // Risk attribution
  riskAttribution: {
    byStrategy: Record<string, number>;
    byExchange: Record<string, number>;
    byAsset: Record<string, number>;
  };
  
  lastUpdated: string;
}

export interface CircuitBreaker {
  id: string;
  name: string;
  type: string;
  threshold: number;
  isActive: boolean;
  isTriggered: boolean;
  cooldownMinutes: number;
  lastTriggeredAt?: string;
  config: Record<string, any>;
}

export interface RiskAlert {
  id: string;
  type: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  title: string;
  message: string;
  data: Record<string, any>;
  isAcknowledged: boolean;
  createdAt: string;
  acknowledgedAt?: string;
}

export interface RiskState {
  metrics: RiskMetrics | null;
  circuitBreakers: CircuitBreaker[];
  alerts: RiskAlert[];
  isLoading: boolean;
  error: string | null;
}

// Market Data types
export interface MarketData {
  symbol: string;
  price: number;
  change24h: number;
  volume24h: number;
  high24h: number;
  low24h: number;
  timestamp: string;
}

export interface CandlestickData {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface OrderBookEntry {
  price: number;
  quantity: number;
  total: number;
}

export interface OrderBook {
  symbol: string;
  bids: OrderBookEntry[];
  asks: OrderBookEntry[];
  timestamp: string;
}

export interface MarketState {
  marketData: Record<string, MarketData>;
  candlestickData: Record<string, CandlestickData[]>;
  orderBooks: Record<string, OrderBook>;
  watchlist: string[];
  isLoading: boolean;
  error: string | null;
}

// UI State types
export interface UIState {
  sidebar: {
    isOpen: boolean;
    isCollapsed: boolean;
  };
  modals: {
    createBot: boolean;
    editBot: boolean;
    strategyConfig: boolean;
    riskSettings: boolean;
  };
  notifications: Notification[];
  theme: 'dark' | 'light';
  isLoading: boolean;
}

export interface Notification {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message: string;
  autoHide: boolean;
  duration?: number;
  createdAt: string;
}

// WebSocket types
export interface WebSocketState {
  isConnected: boolean;
  connectionStatus: 'connecting' | 'connected' | 'disconnected' | 'error';
  lastHeartbeat?: string;
  error: string | null;
}

export interface WebSocketMessage {
  type: string;
  data: any;
  timestamp: string;
}

// API Response types
export interface ApiResponse<T = any> {
  data: T;
  message: string;
  success: boolean;
  timestamp: string;
}

export interface PaginatedResponse<T = any> {
  data: T[];
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
  };
  message: string;
  success: boolean;
}

// Error types
export interface ErrorDetails {
  code: string;
  message: string;
  details?: Record<string, any>;
  timestamp: string;
}

export interface AuthError {
  type: 'INVALID_CREDENTIALS' | 'ACCOUNT_LOCKED' | 'TOKEN_EXPIRED' | 'NETWORK_ERROR' | 'UNKNOWN_ERROR';
  message: string;
  details?: any;
}

// Form types
export interface FormField {
  name: string;
  type: 'text' | 'number' | 'select' | 'checkbox' | 'switch' | 'slider';
  label: string;
  placeholder?: string;
  required: boolean;
  validation?: {
    min?: number;
    max?: number;
    pattern?: string;
    custom?: (value: any) => string | null;
  };
  options?: Array<{ label: string; value: any }>;
  helpText?: string;
}

// Chart types
export interface ChartConfig {
  type: 'line' | 'candlestick' | 'bar' | 'area';
  timeframe: '1m' | '5m' | '15m' | '1h' | '4h' | '1d';
  indicators: string[];
  overlays: string[];
  height: number;
}

// Playground types for testing strategies and configurations
/**
 * Enhanced playground configuration with strategy template integration.
 */
export interface PlaygroundConfiguration {
  id?: string;
  name: string;
  description?: string;
  
  // Strategy configuration
  strategyTemplate: {
    templateId: string;
    strategyType: StrategyType;
    parameters: Record<string, any>;
  };
  
  // Market configuration
  symbols: string[];
  timeframe: TimeInterval;
  
  // Position sizing (updated to match PositionSizeMethod)
  positionSizing: {
    method: PositionSizeMethod;
    value: number;
    maxPositions: number;
    // Kelly Criterion specific
    kellyFraction?: number;
    maxKellyFraction?: number;
    // Risk parity specific
    lookbackPeriod?: number;
  };
  
  tradingSide: 'long' | 'short' | 'both';
  
  // Risk settings (updated to match RiskConfiguration)
  riskSettings: {
    stopLossPercentage: number;
    takeProfitPercentage: number;
    maxDrawdownPercentage: number;
    maxRiskPerTrade: number;
    correlationLimit: number;
    enableCircuitBreaker: boolean;
    dailyLossLimit?: number;
  };
  
  // Portfolio settings (updated to match AllocationStrategy)
  portfolioSettings: {
    maxPositions: number;
    allocationStrategy: AllocationStrategy;
    rebalanceFrequency: 'daily' | 'weekly' | 'monthly' | 'never';
    // Custom allocation weights
    customWeights?: Record<string, number>;
  };
  
  // Legacy strategy field for backward compatibility
  /** @deprecated Use strategyTemplate instead */
  strategy?: {
    type: string;
    parameters: Record<string, any>;
  };
  
  // ML model configuration
  model?: {
    type: string;
    version: string;
    parameters: Record<string, any>;
  };
  
  // Metadata
  createdAt?: string;
  updatedAt?: string;
}

export interface PlaygroundExecution {
  id: string;
  configurationId: string;
  mode: 'historical' | 'live' | 'sandbox' | 'production';
  status: 'idle' | 'running' | 'paused' | 'completed' | 'error' | 'stopped';
  progress: number;
  startTime?: string;
  endTime?: string;
  duration?: number;
  settings: {
    startDate?: string;
    endDate?: string;
    speed: number; // 1x, 5x, 10x, etc.
    initialBalance: number;
    commission: number;
  };
  metrics?: PlaygroundMetrics;
  trades?: PlaygroundTrade[];
  logs: PlaygroundLog[];
  error?: string;
}

export interface PlaygroundMetrics {
  totalReturn: number;
  annualizedReturn: number;
  sharpeRatio: number;
  sortinoRatio: number;
  maxDrawdown: number;
  volatility: number;
  winRate: number;
  profitFactor: number;
  totalTrades: number;
  avgTradeSize: number;
  avgHoldingPeriod: number;
  finalBalance: number;
  peakBalance: number;
  benchmark?: {
    symbol: string;
    return: number;
    volatility: number;
    correlation: number;
  };
}

export interface PlaygroundTrade {
  id: string;
  executionId: string;
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  price: number;
  timestamp: string;
  pnl?: number;
  commission: number;
  reason: string; // strategy signal that triggered the trade
  confidence?: number; // if using ML models
}

export interface PlaygroundLog {
  id: string;
  timestamp: string;
  level: 'debug' | 'info' | 'warning' | 'error';
  category: 'strategy' | 'risk' | 'execution' | 'data' | 'system';
  message: string;
  data?: Record<string, any>;
}

/**
 * Enhanced playground state with strategy template integration.
 */
export interface PlaygroundState {
  // Strategy templates
  availableTemplates: StrategyTemplate[];
  templatesLoading: boolean;
  
  // Configurations
  configurations: PlaygroundConfiguration[];
  activeConfiguration: PlaygroundConfiguration | null;
  configurationHistory: PlaygroundConfiguration[];
  
  // Executions
  executions: PlaygroundExecution[];
  activeExecution: PlaygroundExecution | null;
  comparisonExecutions: PlaygroundExecution[];
  
  // A/B Testing
  abTests: StrategyABTest[];
  activeABTest: StrategyABTest | null;
  
  // Presets
  presets: PlaygroundPreset[];
  presetsLoading: boolean;
  
  // UI State
  isLoading: boolean;
  error: string | null;
  
  // Filters
  filters: {
    status?: PlaygroundExecution['status'][];
    mode?: PlaygroundExecution['mode'][];
    strategyType?: StrategyType[];
    riskLevel?: RiskLevel[];
    dateRange?: DateRange;
    performanceRange?: {
      minReturn?: number;
      maxReturn?: number;
      minSharpe?: number;
      maxSharpe?: number;
    };
  };
  
  // Optimization state
  optimization: {
    isOptimizing: boolean;
    optimizationProgress: number;
    currentGeneration?: number;
    bestConfiguration?: PlaygroundConfiguration;
    optimizationHistory: Array<{
      generation: number;
      bestFitness: number;
      averageFitness: number;
      configuration: PlaygroundConfiguration;
    }>;
  };
}

/**
 * Enhanced playground preset with strategy template integration.
 */
export interface PlaygroundPreset {
  id: string;
  name: string;
  description: string;
  category: RiskLevel; // Use RiskLevel instead of string literals
  
  // Strategy template reference
  strategyTemplateId: string;
  strategyType: StrategyType;
  
  // Configuration
  configuration: PlaygroundConfiguration;
  
  // Performance data
  performanceMetrics?: {
    backtestPeriod: string;
    totalReturn: number;
    sharpeRatio: number;
    maxDrawdown: number;
    winRate: number;
    volatility: number;
  };
  
  // Metadata
  author: string;
  rating?: number;
  downloads: number;
  tags: string[];
  isVerified: boolean;
  
  // Timestamps
  createdAt: string;
  updatedAt?: string;
}

// Advanced playground features
export interface PlaygroundBatch {
  id: string;
  name: string;
  description?: string;
  configurations: PlaygroundConfiguration[];
  status: 'pending' | 'running' | 'completed' | 'error';
  startTime?: string;
  endTime?: string;
  results?: {
    bestConfiguration: PlaygroundConfiguration;
    results: Array<{
      configurationId: string;
      metrics: PlaygroundMetrics;
      rank: number;
    }>;
  };
  settings: {
    crossValidationFolds: number;
    optimizationMetric: 'sharpe_ratio' | 'return' | 'calmar_ratio' | 'sortino_ratio';
    overfittingProtection: {
      enabled: boolean;
      walkForwardWindows: number;
      outOfSamplePercentage: number;
    };
  };
}

export interface ParameterOptimization {
  parameter: string;
  type: 'range' | 'discrete' | 'categorical';
  min?: number;
  max?: number;
  step?: number;
  values?: any[];
  current?: any;
  optimal?: any;
  sensitivity?: number;
}

/**
 * Enhanced A/B testing with strategy template comparison.
 */
export interface StrategyABTest {
  id: string;
  name: string;
  description?: string;
  
  // Strategy configurations
  configurations: {
    control: {
      templateId: string;
      configuration: PlaygroundConfiguration;
    };
    treatment: {
      templateId: string;
      configuration: PlaygroundConfiguration;
    };
  };
  
  // Test executions
  executions: {
    control: PlaygroundExecution;
    treatment: PlaygroundExecution;
  };
  
  // Statistical results
  results?: {
    significanceLevel: number;
    pValue: number;
    confidenceInterval: [number, number];
    winner: 'control' | 'treatment' | 'inconclusive';
    effect: {
      magnitude: number;
      direction: 'positive' | 'negative';
      metric: string;
    };
    detailedMetrics: {
      control: PlaygroundMetrics;
      treatment: PlaygroundMetrics;
      comparison: {
        returnDifference: number;
        sharpeDifference: number;
        drawdownDifference: number;
        winRateDifference: number;
      };
    };
  };
  
  // Test configuration
  testConfig: {
    primaryMetric: 'total_return' | 'sharpe_ratio' | 'calmar_ratio' | 'sortino_ratio';
    minSampleSize: number;
    maxDurationDays: number;
    significanceThreshold: number;
    stopEarlyOnSignificance: boolean;
  };
  
  status: 'setup' | 'running' | 'completed' | 'failed' | 'stopped';
  
  // Metadata
  createdBy: string;
  createdAt: string;
  startedAt?: string;
  completedAt?: string;
}

/**
 * Legacy ABTest interface for backward compatibility.
 * @deprecated Use StrategyABTest instead
 */
export interface ABTest {
  id: string;
  name: string;
  configurations: {
    control: PlaygroundConfiguration;
    treatment: PlaygroundConfiguration;
  };
  executions: {
    control: PlaygroundExecution;
    treatment: PlaygroundExecution;
  };
  results?: {
    significanceLevel: number;
    pValue: number;
    confidenceInterval: [number, number];
    winner: 'control' | 'treatment' | 'inconclusive';
    effect: {
      magnitude: number;
      direction: 'positive' | 'negative';
      metric: string;
    };
  };
  status: 'setup' | 'running' | 'completed' | 'failed';
  createdAt: string;
}

// Strategy Template API Response Types

/**
 * API response for strategy template list.
 */
export interface StrategyTemplateListResponse {
  templates: StrategyTemplate[];
  categories: StrategyCategory[];
  total: number;
  filters: {
    categories: Array<{ value: StrategyCategory; label: string; count: number }>;
    riskLevels: Array<{ value: RiskLevel; label: string; count: number }>;
    exchanges: Array<{ value: string; label: string; count: number }>;
  };
}

/**
 * API response for individual strategy template.
 */
export interface StrategyTemplateResponse {
  template: StrategyTemplate;
  relatedTemplates?: StrategyTemplate[];
  performanceMetrics?: {
    backtestResults: BacktestResult[];
    livePerformance?: {
      totalReturn: number;
      sharpeRatio: number;
      maxDrawdown: number;
      winRate: number;
    };
  };
}

/**
 * Strategy template validation response.
 */
export interface TemplateValidationResponse {
  isValid: boolean;
  errors: Array<{
    parameter: string;
    message: string;
    severity: 'error' | 'warning' | 'info';
  }>;
  warnings: Array<{
    parameter: string;
    message: string;
    recommendation?: string;
  }>;
  estimatedCapitalRequirement: number;
  riskAssessment: {
    riskScore: number;
    riskLevel: RiskLevel;
    riskFactors: string[];
  };
}

/**
 * Strategy deployment request.
 */
export interface StrategyDeploymentRequest {
  templateId: string;
  name: string;
  configuration: {
    exchanges: string[];
    symbols: string[];
    allocatedCapital: number;
    parameters: Record<string, any>;
    riskConfiguration: RiskConfiguration;
  };
  autoStart: boolean;
}

/**
 * Strategy deployment response.
 */
export interface StrategyDeploymentResponse {
  strategyId: string;
  deploymentId: string;
  status: 'deployed' | 'starting' | 'error';
  configuration: StrategyConfiguration;
  estimatedStartTime?: string;
  validationResults: TemplateValidationResponse;
}

// Export commonly used utility types
export type LoadingState = 'idle' | 'loading' | 'succeeded' | 'failed';
export type SortDirection = 'asc' | 'desc';
export type DateRange = { start: string; end: string };
export type TimeInterval = '1m' | '5m' | '15m' | '30m' | '1h' | '4h' | '1d' | '1w';

// Strategy-specific utility types
export type StrategyParameterValue = string | number | boolean | string[] | number[];
export type ValidationRule = (value: any, allValues?: Record<string, any>) => string | null;
export type ParameterDependency = {
  parameter: string;
  condition: (value: any) => boolean;
};