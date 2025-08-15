/**
 * Strategy API service
 */

import { api } from './client';
import { Strategy, BacktestResult } from '@/types';

export const strategyAPI = {
  // Get all strategies
  getStrategies: async () => {
    return api.get<Strategy[]>('/strategies');
  },

  // Get strategy by ID
  getStrategyById: async (strategyId: string) => {
    return api.get<Strategy>(`/strategies/${strategyId}`);
  },

  // Run backtest
  runBacktest: async (strategyId: string, params: any) => {
    return api.post<BacktestResult>(`/strategies/${strategyId}/backtest`, params);
  },

  // Get backtest results
  getBacktestResults: async (strategyId: string) => {
    return api.get<BacktestResult[]>(`/strategies/${strategyId}/backtests`);
  },

  // Create custom strategy
  createStrategy: async (strategyData: Partial<Strategy>) => {
    return api.post<Strategy>('/strategies', strategyData);
  },

  // Update strategy
  updateStrategy: async (strategyId: string, updates: Partial<Strategy>) => {
    return api.patch<Strategy>(`/strategies/${strategyId}`, updates);
  },
};