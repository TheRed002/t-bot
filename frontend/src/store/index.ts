/**
 * Redux store configuration for T-Bot Trading System
 * Combines all slices with middleware for optimal performance
 */

import { configureStore } from '@reduxjs/toolkit';
import { TypedUseSelectorHook, useDispatch, useSelector } from 'react-redux';

// Import slice reducers
import authReducer from './slices/authSlice';
import botReducer from './slices/botSlice';
import portfolioReducer from './slices/portfolioSlice';
import strategyReducer from './slices/strategySlice';
import riskReducer from './slices/riskSlice';
import marketReducer from './slices/marketSlice';
import uiReducer from './slices/uiSlice';
import websocketReducer from './slices/websocketSlice';
import playgroundReducer from './slices/playgroundSlice';

// Import middleware
import { authMiddleware } from './middleware/authMiddleware';
import { errorMiddleware } from './middleware/errorMiddleware';

// Configure the store
export const store = configureStore({
  reducer: {
    auth: authReducer,
    bots: botReducer,
    portfolio: portfolioReducer,
    strategies: strategyReducer,
    risk: riskReducer,
    market: marketReducer,
    ui: uiReducer,
    websocket: websocketReducer,
    playground: playgroundReducer,
  },
  
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        // Ignore these action types for serializable check
        ignoredActions: [
          'websocket/connect',
          'websocket/disconnect',
          'websocket/messageReceived',
        ],
        // Ignore these field paths in all actions
        ignoredActionsPaths: ['meta.arg', 'payload.timestamp'],
        // Ignore these paths in the state
        ignoredPaths: ['websocket.connection'],
      },
      immutableCheck: {
        // Ignore these paths for immutability check
        ignoredPaths: ['websocket.connection'],
      },
    })
      .concat(authMiddleware)
      .concat(errorMiddleware),
  
  devTools: process.env.NODE_ENV !== 'production',
  
  // Preloaded state for SSR or persisted state
  preloadedState: undefined,
});

// Export store types
export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

// Export typed hooks for components
export const useAppDispatch = () => useDispatch<AppDispatch>();
export const useAppSelector: TypedUseSelectorHook<RootState> = useSelector;

// Export action creators and selectors with namespacing to avoid conflicts
export {
  // Auth slice
  loginUser,
  registerUser,
  logoutUser,
  refreshToken,
  fetchUserProfile,
  updateUserProfile,
  clearError as clearAuthError
} from './slices/authSlice';

export {
  // Bot slice
  fetchBots,
  fetchBotById,
  createBot,
  updateBot,
  deleteBot,
  startBot,
  stopBot,
  pauseBot,
  clearError as clearBotError,
  setSelectedBot,
  updateFilters as updateBotFilters,
  clearFilters as clearBotFilters
} from './slices/botSlice';

export {
  // Portfolio slice
  fetchPortfolioSummary,
  fetchPositions,
  fetchBalances,
  updateFilters as updatePortfolioFilters,
  clearError as clearPortfolioError,
  selectPortfolioState
} from './slices/portfolioSlice';

export {
  // Strategy slice
  fetchStrategies,
  runBacktest,
  deployStrategy,
  updateStrategy,
  clearError as clearStrategyError,
  setSelectedStrategy,
  selectStrategies,
  selectSelectedStrategy,
  selectStrategyLoading,
  selectStrategyError
} from './slices/strategySlice';

export {
  // Risk slice
  fetchRiskMetrics,
  updateRiskMetrics,
  addAlert,
  clearError as clearRiskError
} from './slices/riskSlice';

export {
  // Market slice
  updateMarketData,
  updateCandlestickData,
  updateOrderBook,
  addToWatchlist,
  removeFromWatchlist
} from './slices/marketSlice';

export {
  // UI slice
  toggleSidebar,
  setSidebarCollapsed,
  openModal,
  closeModal,
  addNotification,
  removeNotification,
  clearNotifications,
  setTheme
} from './slices/uiSlice';

export {
  // WebSocket slice
  connect as connectWebSocket,
  connected as connectedWebSocket,
  disconnect as disconnectWebSocket,
  error as websocketError,
  heartbeat as websocketHeartbeat,
  messageReceived as websocketMessageReceived
} from './slices/websocketSlice';

export {
  // Playground slice actions
  fetchConfigurations,
  saveConfiguration,
  deleteConfiguration,
  startExecution,
  controlExecution,
  fetchExecutions,
  fetchExecutionLogs,
  fetchExecutionTrades,
  createABTest,
  runABTest,
  startBatchOptimization,
  fetchPresets,
  setActiveConfiguration,
  updateConfigurationField,
  resetActiveConfiguration,
  setActiveExecution,
  updateExecutionProgress,
  updateExecutionStatus,
  addExecutionLog,
  updateExecutionMetrics,
  addExecutionTrade,
  addToComparison,
  removeFromComparison,
  clearComparison,
  setFilters as setPlaygroundFilters,
  updateFilter as updatePlaygroundFilter,
  clearFilters as clearPlaygroundFilters,
  clearError as clearPlaygroundError,
  setError as setPlaygroundError,
  resetPlaygroundState,
  // Playground slice selectors
  selectPlaygroundState,
  selectConfigurations as selectPlaygroundConfigurations,
  selectActiveConfiguration,
  selectExecutions,
  selectActiveExecution,
  selectComparisonExecutions,
  selectPlaygroundIsLoading as selectPlaygroundLoading,
  selectPlaygroundError,
  selectPlaygroundFilters,
  selectCompletedExecutions,
  selectRunningExecutions,
  selectFilteredExecutions,
  selectBestPerformingExecution
} from './slices/playgroundSlice';

// Export store instance
export default store;