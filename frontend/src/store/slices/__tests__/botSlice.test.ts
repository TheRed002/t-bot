/**
 * Test suite for botSlice Redux store
 * Tests the bot data pipeline and transformations
 */

import { configureStore } from '@reduxjs/toolkit';
import botSlice, {
  fetchBots,
  fetchBotById,
  createBot,
  updateBot,
  deleteBot,
  startBot,
  stopBot,
  pauseBot,
  resumeBot,
  fetchBotPerformance,
  updateBotStatus,
  updateBotPerformance,
  clearError,
  setSelectedBot,
  updateFilters,
  clearFilters,
  selectBots,
  selectSelectedBot,
  selectBotLoading,
  selectBotError,
  selectBotFilters,
  selectFilteredBots,
  selectBotById,
  selectRunningBots,
  selectBotsByExchange,
  selectBotCount,
  selectBotsByStatus,
  selectBotStatusCounts,
} from '../botSlice';
import { BotStatus, BotInstance, BotConfiguration, BotMetrics } from '@/types';

// Mock botAPI
jest.mock('@/services/api/botAPI', () => ({
  botAPI: {
    getBots: jest.fn(),
    getBotById: jest.fn(),
    createBot: jest.fn(),
    updateBot: jest.fn(),
    deleteBot: jest.fn(),
    startBot: jest.fn(),
    stopBot: jest.fn(),
    pauseBot: jest.fn(),
    resumeBot: jest.fn(),
    getBotPerformance: jest.fn(),
  },
}));

import { botAPI } from '@/services/api/botAPI';

const mockBotAPI = botAPI as jest.Mocked<typeof botAPI>;

describe('botSlice', () => {
  let store: ReturnType<typeof configureStore>;

  const mockBot: BotInstance = {
    bot_id: 'test-bot-1',
    bot_name: 'Test Bot',
    userId: 'user-123',
    strategy_name: 'mean_reversion',
    exchange: 'binance',
    status: BotStatus.STOPPED,
    config: {
      bot_id: 'test-bot-1',
      bot_name: 'Test Bot',
      bot_type: 'trading' as any,
      strategy_name: 'mean_reversion',
      exchanges: ['binance'],
      symbols: ['BTC/USDT'],
      allocated_capital: 1000,
      risk_percentage: 0.02,
      priority: 'normal' as any,
      auto_start: false,
      strategy_config: {},
      created_at: '2024-01-01T00:00:00Z',
      updated_at: '2024-01-01T00:00:00Z',
    },
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-01-01T00:00:00Z',
    metrics: {
      bot_id: 'test-bot-1',
      uptime_seconds: 3600,
      total_trades: 10,
      successful_trades: 6,
      failed_trades: 4,
      total_pnl: 50.25,
      cpu_usage_percent: 2.5,
      memory_usage_mb: 128,
      win_rate: 60,
      health_score: 95,
      timestamp: '2024-01-01T01:00:00Z',
    },
  };

  beforeEach(() => {
    store = configureStore({
      reducer: {
        bots: botSlice,
      },
    });
    jest.clearAllMocks();
  });

  describe('initial state', () => {
    it('should have correct initial state', () => {
      const state = store.getState();
      expect(selectBots(state)).toEqual([]);
      expect(selectSelectedBot(state)).toBeNull();
      expect(selectBotLoading(state)).toBe(false);
      expect(selectBotError(state)).toBeNull();
      expect(selectBotFilters(state)).toEqual({
        status: undefined,
        exchange: undefined,
        strategy: undefined,
        searchTerm: '',
      });
    });
  });

  describe('synchronous actions', () => {
    it('should clear error', () => {
      // Set error first
      store.dispatch({ type: 'bots/fetchBots/rejected', payload: 'Test error' });
      expect(selectBotError(store.getState())).toBe('Test error');

      // Clear error
      store.dispatch(clearError());
      expect(selectBotError(store.getState())).toBeNull();
    });

    it('should set selected bot', () => {
      store.dispatch(setSelectedBot(mockBot));
      expect(selectSelectedBot(store.getState())).toEqual(mockBot);
    });

    it('should update filters', () => {
      store.dispatch(updateFilters({ status: [BotStatus.RUNNING] }));
      const filters = selectBotFilters(store.getState());
      expect(filters.status).toEqual([BotStatus.RUNNING]);
    });

    it('should clear filters', () => {
      // Set filters first
      store.dispatch(updateFilters({ status: [BotStatus.RUNNING], searchTerm: 'test' }));
      
      // Clear filters
      store.dispatch(clearFilters());
      const filters = selectBotFilters(store.getState());
      expect(filters).toEqual({
        status: undefined,
        exchange: undefined,
        strategy: undefined,
        searchTerm: '',
      });
    });

    it('should update bot status', () => {
      // Add bot to store first
      store.dispatch({
        type: 'bots/fetchBots/fulfilled',
        payload: [mockBot],
      });

      // Update status
      store.dispatch(updateBotStatus({
        botId: mockBot.bot_id,
        status: BotStatus.RUNNING,
        timestamp: '2024-01-01T02:00:00Z',
      }));

      const bots = selectBots(store.getState());
      expect(bots[0].status).toBe(BotStatus.RUNNING);
      expect(bots[0].updatedAt).toBe('2024-01-01T02:00:00Z');
    });

    it('should handle invalid bot status update', () => {
      const consoleSpy = jest.spyOn(console, 'warn').mockImplementation();
      
      // Try to update non-existent bot
      store.dispatch(updateBotStatus({
        botId: 'non-existent',
        status: BotStatus.RUNNING,
      }));

      expect(consoleSpy).toHaveBeenCalledWith('Bot not found for status update: non-existent');
      consoleSpy.mockRestore();
    });
  });

  describe('async actions', () => {
    describe('fetchBots', () => {
      it('should fetch bots successfully', async () => {
        const mockResponse = {
          data: [mockBot],
          success: true,
          message: 'Bots retrieved successfully',
          timestamp: '2024-01-01T00:00:00Z',
        };

        mockBotAPI.getBots.mockResolvedValue(mockResponse);

        await store.dispatch(fetchBots({}));

        const state = store.getState();
        expect(selectBots(state)).toEqual([mockBot]);
        expect(selectBotLoading(state)).toBe(false);
        expect(selectBotError(state)).toBeNull();
      });

      it('should handle fetch bots error', async () => {
        mockBotAPI.getBots.mockRejectedValue(new Error('API Error'));

        await store.dispatch(fetchBots({}));

        const state = store.getState();
        expect(selectBots(state)).toEqual([]);
        expect(selectBotLoading(state)).toBe(false);
        expect(selectBotError(state)).toBe('API Error');
      });

      it('should handle invalid response format', async () => {
        const mockResponse = {
          data: 'invalid data', // Should be array
          success: true,
          message: 'Bots retrieved successfully',
          timestamp: '2024-01-01T00:00:00Z',
        };

        mockBotAPI.getBots.mockResolvedValue(mockResponse as any);

        await store.dispatch(fetchBots({}));

        const state = store.getState();
        expect(selectBots(state)).toEqual([]);
        expect(selectBotError(state)).toBe('Invalid response format: expected array of bots');
      });
    });

    describe('startBot', () => {
      it('should start bot successfully', async () => {
        const mockResponse = {
          success: true,
          message: 'Bot started successfully',
          timestamp: '2024-01-01T00:00:00Z',
        };

        mockBotAPI.startBot.mockResolvedValue(mockResponse);

        // Add bot to store first
        store.dispatch({
          type: 'bots/fetchBots/fulfilled',
          payload: [mockBot],
        });

        await store.dispatch(startBot(mockBot.bot_id));

        const bots = selectBots(store.getState());
        expect(bots[0].status).toBe(BotStatus.RUNNING);
      });

      it('should handle start bot error', async () => {
        mockBotAPI.startBot.mockRejectedValue(new Error('Cannot start bot'));

        await store.dispatch(startBot('test-bot-1'));

        const state = store.getState();
        expect(selectBotError(state)).toBe('Cannot start bot');
      });
    });
  });

  describe('selectors', () => {
    beforeEach(() => {
      const bots = [
        { ...mockBot, bot_id: 'bot-1', bot_name: 'Trading Bot 1', status: BotStatus.RUNNING, exchange: 'binance' },
        { ...mockBot, bot_id: 'bot-2', bot_name: 'Trading Bot 2', status: BotStatus.STOPPED, exchange: 'coinbase' },
        { ...mockBot, bot_id: 'bot-3', bot_name: 'Trading Bot 3', status: BotStatus.ERROR, exchange: 'binance' },
      ];

      store.dispatch({
        type: 'bots/fetchBots/fulfilled',
        payload: bots,
      });
    });

    it('should select filtered bots by status', () => {
      store.dispatch(updateFilters({ status: [BotStatus.RUNNING] }));
      const filteredBots = selectFilteredBots(store.getState());
      expect(filteredBots).toHaveLength(1);
      expect(filteredBots[0].status).toBe(BotStatus.RUNNING);
    });

    it('should select filtered bots by exchange', () => {
      store.dispatch(updateFilters({ exchange: ['binance'] }));
      const filteredBots = selectFilteredBots(store.getState());
      expect(filteredBots).toHaveLength(2);
      expect(filteredBots.every(bot => bot.exchange === 'binance')).toBe(true);
    });

    it('should select filtered bots by search term', () => {
      store.dispatch(updateFilters({ searchTerm: 'Bot 1' }));
      const filteredBots = selectFilteredBots(store.getState());
      expect(filteredBots).toHaveLength(1);
      expect(filteredBots[0].bot_id).toBe('bot-1');
    });

    it('should select bot by id', () => {
      const bot = selectBotById(store.getState(), 'bot-2');
      expect(bot?.bot_id).toBe('bot-2');
    });

    it('should select running bots', () => {
      const runningBots = selectRunningBots(store.getState());
      expect(runningBots).toHaveLength(1);
      expect(runningBots[0].status).toBe(BotStatus.RUNNING);
    });

    it('should select bots by exchange', () => {
      const binanceBots = selectBotsByExchange(store.getState(), 'binance');
      expect(binanceBots).toHaveLength(2);
      expect(binanceBots.every(bot => bot.exchange === 'binance')).toBe(true);
    });

    it('should select bot count', () => {
      const count = selectBotCount(store.getState());
      expect(count).toBe(3);
    });

    it('should select bots by status', () => {
      const stoppedBots = selectBotsByStatus(store.getState(), BotStatus.STOPPED);
      expect(stoppedBots).toHaveLength(1);
      expect(stoppedBots[0].status).toBe(BotStatus.STOPPED);
    });

    it('should select bot status counts', () => {
      const counts = selectBotStatusCounts(store.getState());
      expect(counts).toEqual({
        running: 1,
        stopped: 1,
        error: 1,
        paused: 0,
        total: 3,
      });
    });

    it('should handle empty bot list in selectors', () => {
      // Reset to empty state
      store.dispatch({
        type: 'bots/fetchBots/fulfilled',
        payload: [],
      });

      expect(selectBots(store.getState())).toEqual([]);
      expect(selectRunningBots(store.getState())).toEqual([]);
      expect(selectBotsByExchange(store.getState(), 'binance')).toEqual([]);
      expect(selectBotCount(store.getState())).toBe(0);
      expect(selectBotStatusCounts(store.getState())).toEqual({
        running: 0,
        stopped: 0,
        error: 0,
        paused: 0,
        total: 0,
      });
    });
  });

  describe('error handling', () => {
    it('should handle null/undefined bot data gracefully', () => {
      const consoleSpy = jest.spyOn(console, 'warn').mockImplementation();
      
      // Try to filter with invalid bot data
      store.dispatch({
        type: 'bots/fetchBots/fulfilled',
        payload: [null, undefined, { bot_id: null }, mockBot],
      });

      const filteredBots = selectFilteredBots(store.getState());
      expect(filteredBots).toHaveLength(1); // Only valid bot should remain
      expect(filteredBots[0]).toEqual(mockBot);

      consoleSpy.mockRestore();
    });

    it('should handle invalid selector inputs', () => {
      expect(selectBotById(store.getState(), '')).toBeUndefined();
      expect(selectBotsByExchange(store.getState(), '')).toEqual([]);
      expect(selectBotsByStatus(store.getState(), null as any)).toEqual([]);
    });
  });
});