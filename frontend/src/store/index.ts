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
import { websocketMiddleware } from './middleware/websocketMiddleware';
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
      .concat(websocketMiddleware)
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

// Export action creators and selectors
export * from './slices/authSlice';
export * from './slices/botSlice';
export * from './slices/portfolioSlice';
export * from './slices/strategySlice';
export * from './slices/riskSlice';
export * from './slices/marketSlice';
export * from './slices/uiSlice';
export * from './slices/websocketSlice';
export * from './slices/playgroundSlice';

// Export store instance
export default store;