/**
 * Bot management API service
 */

import { api } from './client';
import { BotInstance, BotFilters, BotConfig, BotPerformance } from '@/types';

export const botAPI = {
  // Get all bots
  getBots: async (filters?: BotFilters) => {
    const params = filters ? { ...filters } : {};
    return api.get<BotInstance[]>('/bots', { params });
  },

  // Get bot by ID
  getBotById: async (botId: string) => {
    return api.get<BotInstance>(`/bots/${botId}`);
  },

  // Create new bot
  createBot: async (botData: {
    name: string;
    strategyType: string;
    exchange: string;
    config: BotConfig;
  }) => {
    return api.post<BotInstance>('/bots', botData);
  },

  // Update bot
  updateBot: async (botId: string, updates: Partial<BotInstance>) => {
    return api.patch<BotInstance>(`/bots/${botId}`, updates);
  },

  // Delete bot
  deleteBot: async (botId: string) => {
    return api.delete(`/bots/${botId}`);
  },

  // Start bot
  startBot: async (botId: string) => {
    return api.post(`/bots/${botId}/start`);
  },

  // Stop bot
  stopBot: async (botId: string) => {
    return api.post(`/bots/${botId}/stop`);
  },

  // Pause bot
  pauseBot: async (botId: string) => {
    return api.post(`/bots/${botId}/pause`);
  },

  // Get bot performance
  getBotPerformance: async (botId: string) => {
    return api.get<BotPerformance>(`/bots/${botId}/performance`);
  },

  // Get bot logs
  getBotLogs: async (botId: string, limit?: number) => {
    return api.get(`/bots/${botId}/logs`, { params: { limit } });
  },

  // Update bot configuration
  updateBotConfig: async (botId: string, config: BotConfig) => {
    return api.patch<BotInstance>(`/bots/${botId}/config`, { config });
  },
};