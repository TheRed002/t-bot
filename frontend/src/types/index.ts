/**
 * Core type definitions for T-Bot Trading System
 * Shared interfaces and types used throughout the application
 */

// Authentication types
export interface User {
  id: string;
  username: string;
  email: string;
  isActive: boolean;
  createdAt: string;
  preferences?: UserPreferences;
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

export interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
}

// Bot Management types
export interface BotInstance {
  id: string;
  name: string;
  userId: string;
  strategyType: string;
  exchange: string;
  status: BotStatus;
  config: BotConfig;
  createdAt: string;
  updatedAt: string;
  performance?: BotPerformance;
}

export type BotStatus = 'stopped' | 'running' | 'paused' | 'error' | 'pending';

export interface BotConfig {
  symbol: string;
  baseAmount: number;
  maxPosition: number;
  riskPercentage: number;
  stopLoss?: number;
  takeProfit?: number;
  strategyParams: Record<string, any>;
}

export interface BotPerformance {
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  totalPnl: number;
  winRate: number;
  maxDrawdown: number;
  sharpeRatio?: number;
  startDate: string;
  lastTradeAt?: string;
}

export interface BotState {
  bots: BotInstance[];
  selectedBot: BotInstance | null;
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
  positions: Position[];
  balances: Balance[];
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

// Strategy types
export interface Strategy {
  id: string;
  name: string;
  type: string;
  description: string;
  parameters: StrategyParameter[];
  isActive: boolean;
  createdAt: string;
  performance?: StrategyPerformance;
}

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

export interface StrategyState {
  strategies: Strategy[];
  selectedStrategy: Strategy | null;
  backtestResults: BacktestResult[];
  isLoading: boolean;
  isBacktesting: boolean;
  error: string | null;
}

// Risk Management types
export interface RiskMetrics {
  portfolioValue: number;
  totalExposure: number;
  maxDrawdown: number;
  var95: number; // Value at Risk 95%
  var99: number; // Value at Risk 99%
  sharpeRatio: number;
  correlation: Record<string, number>;
  beta: number;
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

// Export commonly used utility types
export type LoadingState = 'idle' | 'loading' | 'succeeded' | 'failed';
export type SortDirection = 'asc' | 'desc';
export type DateRange = { start: string; end: string };
export type TimeInterval = '1m' | '5m' | '15m' | '30m' | '1h' | '4h' | '1d' | '1w';