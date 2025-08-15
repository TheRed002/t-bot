/**
 * Portfolio API service
 */

import { api } from './client';
import { PortfolioSummary, Position, Balance, PortfolioFilters } from '@/types';

export const portfolioAPI = {
  // Get portfolio summary
  getSummary: async () => {
    return api.get<PortfolioSummary>('/portfolio/summary');
  },

  // Get positions
  getPositions: async (filters?: PortfolioFilters) => {
    const params = filters ? { ...filters } : {};
    return api.get<Position[]>('/portfolio/positions', { params });
  },

  // Get balances
  getBalances: async (filters?: PortfolioFilters) => {
    const params = filters ? { ...filters } : {};
    return api.get<Balance[]>('/portfolio/balances', { params });
  },

  // Get portfolio history
  getPortfolioHistory: async (startDate: string, endDate: string, interval: string = '1h') => {
    return api.get('/portfolio/history', {
      params: { startDate, endDate, interval },
    });
  },

  // Get P&L breakdown
  getPnLBreakdown: async (period: string = '1d') => {
    return api.get('/portfolio/pnl', { params: { period } });
  },
};