/**
 * Risk management API service
 */

import { api } from './client';
import { RiskMetrics, CircuitBreaker, RiskAlert } from '@/types';

export const riskAPI = {
  // Get risk metrics
  getRiskMetrics: async () => {
    return api.get<RiskMetrics>('/risk/metrics');
  },

  // Get circuit breakers
  getCircuitBreakers: async () => {
    return api.get<CircuitBreaker[]>('/risk/circuit-breakers');
  },

  // Update circuit breaker
  updateCircuitBreaker: async (breakerId: string, updates: Partial<CircuitBreaker>) => {
    return api.patch<CircuitBreaker>(`/risk/circuit-breakers/${breakerId}`, updates);
  },

  // Get risk alerts
  getRiskAlerts: async (limit?: number) => {
    return api.get<RiskAlert[]>('/risk/alerts', { params: { limit } });
  },

  // Acknowledge alert
  acknowledgeAlert: async (alertId: string) => {
    return api.post(`/risk/alerts/${alertId}/acknowledge`);
  },

  // Get risk settings
  getRiskSettings: async () => {
    return api.get('/risk/settings');
  },

  // Update risk settings
  updateRiskSettings: async (settings: any) => {
    return api.patch('/risk/settings', settings);
  },
};