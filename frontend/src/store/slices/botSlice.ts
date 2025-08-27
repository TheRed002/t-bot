/**
 * Bot management slice for Redux store
 * Manages trading bot instances and their states
 * Handles proper data transformation between frontend and backend API
 */

import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { BotState, BotInstance, BotStatus, BotFilters, BotConfiguration, BotMetrics, BotListResponse } from '@/types';
// Legacy imports for backward compatibility
import type { BotConfig, BotPerformance } from '@/types';
import { botAPI } from '@/services/api/botAPI';

// Initial state
const initialState: BotState = {
  bots: [],
  selectedBot: null,
  botList: undefined,
  isLoading: false,
  error: null,
  filters: {
    status: undefined,
    exchange: undefined,
    strategy: undefined,
    searchTerm: '',
  },
};

// Async thunks
export const fetchBots = createAsyncThunk(
  'bots/fetchBots',
  async (filters: BotFilters = {}, { rejectWithValue }) => {
    try {
      const response = await botAPI.getBots(filters);
      
      // Log for debugging
      console.log('Bot API Response:', {
        success: response.success,
        dataLength: response.data?.length,
        timestamp: response.timestamp
      });
      
      if (!response.success) {
        throw new Error(response.message || 'API returned unsuccessful response');
      }
      
      // Ensure response data is valid
      if (!Array.isArray(response.data)) {
        console.error('Invalid bot data format:', response.data);
        throw new Error('Invalid response format: expected array of bots');
      }
      
      return response.data;
    } catch (error: any) {
      console.error('fetchBots error:', error);
      return rejectWithValue(
        error.message || 
        error.response?.data?.message || 
        'Failed to fetch bots'
      );
    }
  }
);

export const fetchBotById = createAsyncThunk(
  'bots/fetchBotById',
  async (botId: string, { rejectWithValue }) => {
    try {
      if (!botId || typeof botId !== 'string') {
        throw new Error('Invalid bot ID provided');
      }
      
      const response = await botAPI.getBotById(botId);
      
      // Log for debugging
      console.log('fetchBotById response:', {
        botId,
        success: response.success,
        hasData: !!response.data,
        timestamp: response.timestamp
      });
      
      if (!response.success || !response.data) {
        throw new Error(response.message || 'Bot not found');
      }
      
      return response.data;
    } catch (error: any) {
      console.error('fetchBotById error:', { botId, error });
      return rejectWithValue(
        error.message || 
        error.response?.data?.message || 
        `Failed to fetch bot: ${botId}`
      );
    }
  }
);

export const createBot = createAsyncThunk(
  'bots/createBot',
  async (
    botData: {
      bot_name: string;  // Use backend field name
      strategy_name: string;  // Use backend field name
      exchange: string;
      config: BotConfiguration;  // Use new config type
    },
    { rejectWithValue }
  ) => {
    try {
      // Validate input data
      if (!botData.bot_name || !botData.strategy_name || !botData.exchange) {
        throw new Error('Missing required bot creation parameters');
      }
      
      if (!botData.config || typeof botData.config !== 'object') {
        throw new Error('Invalid bot configuration provided');
      }
      
      const response = await botAPI.createBot(botData);
      
      // Log for debugging
      console.log('createBot response:', {
        botName: botData.bot_name,
        success: response.success,
        hasData: !!response.data,
        timestamp: response.timestamp
      });
      
      if (!response.success || !response.data) {
        throw new Error(response.message || 'Failed to create bot');
      }
      
      return response.data;
    } catch (error: any) {
      console.error('createBot error:', { botData, error });
      return rejectWithValue(
        error.message || 
        error.response?.data?.message || 
        'Failed to create bot'
      );
    }
  }
);

export const updateBot = createAsyncThunk(
  'bots/updateBot',
  async (
    { botId, updates }: { botId: string; updates: Partial<BotInstance> },
    { rejectWithValue }
  ) => {
    try {
      if (!botId || typeof botId !== 'string') {
        throw new Error('Invalid bot ID provided');
      }
      
      if (!updates || typeof updates !== 'object') {
        throw new Error('Invalid update data provided');
      }
      
      const response = await botAPI.updateBot(botId, updates);
      
      // Log for debugging
      console.log('updateBot response:', {
        botId,
        success: response.success,
        hasData: !!response.data,
        timestamp: response.timestamp
      });
      
      if (!response.success || !response.data) {
        throw new Error(response.message || 'Failed to update bot');
      }
      
      return response.data;
    } catch (error: any) {
      console.error('updateBot error:', { botId, updates, error });
      return rejectWithValue(
        error.message || 
        error.response?.data?.message || 
        `Failed to update bot: ${botId}`
      );
    }
  }
);

export const deleteBot = createAsyncThunk(
  'bots/deleteBot',
  async (botId: string, { rejectWithValue }) => {
    try {
      if (!botId || typeof botId !== 'string') {
        throw new Error('Invalid bot ID provided');
      }
      
      const response = await botAPI.deleteBot(botId);
      
      // Log for debugging
      console.log('deleteBot response:', {
        botId,
        success: response.success,
        timestamp: response.timestamp
      });
      
      if (!response.success) {
        throw new Error(response.message || 'Failed to delete bot');
      }
      
      return botId;
    } catch (error: any) {
      console.error('deleteBot error:', { botId, error });
      return rejectWithValue(
        error.message || 
        error.response?.data?.message || 
        `Failed to delete bot: ${botId}`
      );
    }
  }
);

export const startBot = createAsyncThunk(
  'bots/startBot',
  async (botId: string, { rejectWithValue }) => {
    try {
      if (!botId || typeof botId !== 'string') {
        throw new Error('Invalid bot ID provided');
      }
      
      const response = await botAPI.startBot(botId);
      
      // Log for debugging
      console.log('startBot response:', {
        botId,
        success: response.success,
        message: response.message,
        timestamp: response.timestamp
      });
      
      if (!response.success) {
        throw new Error(response.message || 'Failed to start bot');
      }
      
      // Return data for reducer to update bot status
      return { 
        botId, 
        status: BotStatus.RUNNING,
        message: response.message,
        timestamp: response.timestamp
      };
    } catch (error: any) {
      console.error('startBot error:', { botId, error });
      return rejectWithValue(
        error.message || 
        error.response?.data?.message || 
        `Failed to start bot: ${botId}`
      );
    }
  }
);

export const stopBot = createAsyncThunk(
  'bots/stopBot',
  async (botId: string, { rejectWithValue }) => {
    try {
      if (!botId || typeof botId !== 'string') {
        throw new Error('Invalid bot ID provided');
      }
      
      const response = await botAPI.stopBot(botId);
      
      // Log for debugging
      console.log('stopBot response:', {
        botId,
        success: response.success,
        message: response.message,
        timestamp: response.timestamp
      });
      
      if (!response.success) {
        throw new Error(response.message || 'Failed to stop bot');
      }
      
      // Return data for reducer to update bot status
      return { 
        botId, 
        status: BotStatus.STOPPED,
        message: response.message,
        timestamp: response.timestamp
      };
    } catch (error: any) {
      console.error('stopBot error:', { botId, error });
      return rejectWithValue(
        error.message || 
        error.response?.data?.message || 
        `Failed to stop bot: ${botId}`
      );
    }
  }
);

export const pauseBot = createAsyncThunk(
  'bots/pauseBot',
  async (botId: string, { rejectWithValue }) => {
    try {
      if (!botId || typeof botId !== 'string') {
        throw new Error('Invalid bot ID provided');
      }
      
      const response = await botAPI.pauseBot(botId);
      
      // Log for debugging
      console.log('pauseBot response:', {
        botId,
        success: response.success,
        message: response.message,
        timestamp: response.timestamp
      });
      
      if (!response.success) {
        throw new Error(response.message || 'Failed to pause bot');
      }
      
      // Return data for reducer to update bot status
      return { 
        botId, 
        status: BotStatus.PAUSED,
        message: response.message,
        timestamp: response.timestamp
      };
    } catch (error: any) {
      console.error('pauseBot error:', { botId, error });
      return rejectWithValue(
        error.message || 
        error.response?.data?.message || 
        `Failed to pause bot: ${botId}`
      );
    }
  }
);

export const resumeBot = createAsyncThunk(
  'bots/resumeBot',
  async (botId: string, { rejectWithValue }) => {
    try {
      if (!botId || typeof botId !== 'string') {
        throw new Error('Invalid bot ID provided');
      }
      
      const response = await botAPI.resumeBot(botId);
      
      // Log for debugging
      console.log('resumeBot response:', {
        botId,
        success: response.success,
        message: response.message,
        timestamp: response.timestamp
      });
      
      if (!response.success) {
        throw new Error(response.message || 'Failed to resume bot');
      }
      
      // Return data for reducer to update bot status
      return { 
        botId, 
        status: BotStatus.RUNNING,
        message: response.message,
        timestamp: response.timestamp
      };
    } catch (error: any) {
      console.error('resumeBot error:', { botId, error });
      return rejectWithValue(
        error.message || 
        error.response?.data?.message || 
        `Failed to resume bot: ${botId}`
      );
    }
  }
);

export const fetchBotPerformance = createAsyncThunk(
  'bots/fetchBotPerformance',
  async (botId: string, { rejectWithValue }) => {
    try {
      if (!botId || typeof botId !== 'string') {
        throw new Error('Invalid bot ID provided');
      }
      
      const response = await botAPI.getBotPerformance(botId);
      
      // Log for debugging
      console.log('fetchBotPerformance response:', {
        botId,
        success: response.success,
        hasData: !!response.data,
        timestamp: response.timestamp
      });
      
      if (!response.success) {
        console.warn('Bot performance API returned unsuccessful response:', response.message);
        // Return empty performance data instead of failing
        return { 
          botId, 
          performance: {
            totalTrades: 0,
            winningTrades: 0,
            losingTrades: 0,
            totalPnl: 0,
            winRate: 0,
            maxDrawdown: 0,
            startDate: new Date().toISOString()
          } as BotPerformance
        };
      }
      
      return { botId, performance: response.data };
    } catch (error: any) {
      console.error('fetchBotPerformance error:', { botId, error });
      // Return empty performance instead of rejecting to avoid breaking UI
      return { 
        botId, 
        performance: {
          totalTrades: 0,
          winningTrades: 0,
          losingTrades: 0,
          totalPnl: 0,
          winRate: 0,
          maxDrawdown: 0,
          startDate: new Date().toISOString()
        } as BotPerformance
      };
    }
  }
);

// Bot slice
const botSlice = createSlice({
  name: 'bots',
  initialState,
  reducers: {
    // Clear error state
    clearError: (state) => {
      state.error = null;
    },
    
    // Set selected bot
    setSelectedBot: (state, action: PayloadAction<BotInstance | null>) => {
      state.selectedBot = action.payload;
    },
    
    // Update filters
    updateFilters: (state, action: PayloadAction<Partial<BotFilters>>) => {
      state.filters = { ...state.filters, ...action.payload };
    },
    
    // Clear filters
    clearFilters: (state) => {
      state.filters = {
        status: undefined,
        exchange: undefined,
        strategy: undefined,
        searchTerm: '',
      };
    },
    
    // Update bot status (for real-time updates)
    updateBotStatus: (state, action: PayloadAction<{ botId: string; status: BotStatus; timestamp?: string }>) => {
      const { botId, status, timestamp } = action.payload;
      
      if (!botId || !status) {
        console.warn('updateBotStatus: Invalid payload', action.payload);
        return;
      }
      
      const updateTime = timestamp || new Date().toISOString();
      const botIndex = state.bots.findIndex(bot => bot.bot_id === botId);
      
      if (botIndex !== -1) {
        state.bots[botIndex].status = status;
        state.bots[botIndex].updatedAt = updateTime;
        console.log(`Updated bot ${botId} status to ${status}`);
      } else {
        console.warn(`Bot not found for status update: ${botId}`);
      }
      
      // Update selected bot if it matches
      if (state.selectedBot?.bot_id === botId) {
        state.selectedBot.status = status;
        state.selectedBot.updatedAt = updateTime;
      }
    },
    
    // Update bot performance (for real-time updates)
    updateBotPerformance: (state, action: PayloadAction<{ botId: string; performance: BotPerformance | BotMetrics; timestamp?: string }>) => {
      const { botId, performance, timestamp } = action.payload;
      
      if (!botId || !performance) {
        console.warn('updateBotPerformance: Invalid payload', action.payload);
        return;
      }
      
      const updateTime = timestamp || new Date().toISOString();
      const botIndex = state.bots.findIndex(bot => bot.bot_id === botId);
      
      // Helper to convert legacy BotPerformance to BotMetrics
      const convertToMetrics = (perf: BotPerformance | BotMetrics): BotMetrics => {
        // If it's already BotMetrics format, use it directly
        if ('bot_id' in perf && 'total_trades' in perf) {
          return perf as BotMetrics;
        }
        
        // Convert from legacy BotPerformance format
        const legacyPerf = perf as BotPerformance;
        return {
          bot_id: botId,
          uptime_seconds: 0,
          total_trades: legacyPerf.totalTrades || 0,
          successful_trades: legacyPerf.winningTrades || 0,
          failed_trades: legacyPerf.losingTrades || 0,
          total_pnl: legacyPerf.totalPnl || 0,
          cpu_usage_percent: 0,
          memory_usage_mb: 0,
          win_rate: legacyPerf.winRate || 0,
          health_score: 100,
          timestamp: updateTime
        };
      };
      
      const metrics = convertToMetrics(performance);
      
      if (botIndex !== -1) {
        state.bots[botIndex].metrics = metrics;
        state.bots[botIndex].updatedAt = updateTime;
        console.log(`Updated bot ${botId} performance metrics`);
      } else {
        console.warn(`Bot not found for performance update: ${botId}`);
      }
      
      // Update selected bot if it matches
      if (state.selectedBot?.bot_id === botId) {
        state.selectedBot.metrics = metrics;
        state.selectedBot.updatedAt = updateTime;
      }
    },
  },
  extraReducers: (builder) => {
    // Fetch bots
    builder
      .addCase(fetchBots.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(fetchBots.fulfilled, (state, action) => {
        state.isLoading = false;
        
        // Validate payload is an array
        if (Array.isArray(action.payload)) {
          state.bots = action.payload;
          console.log(`Successfully loaded ${action.payload.length} bots`);
        } else {
          console.error('fetchBots.fulfilled: Invalid payload format', action.payload);
          state.bots = [];
          state.error = 'Invalid data format received from server';
          return;
        }
        
        state.error = null;
      })
      .addCase(fetchBots.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
      });

    // Fetch bot by ID
    builder
      .addCase(fetchBotById.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(fetchBotById.fulfilled, (state, action) => {
        state.isLoading = false;
        
        // Validate payload has required fields
        if (!action.payload || !action.payload.bot_id) {
          console.error('fetchBotById.fulfilled: Invalid bot data', action.payload);
          state.error = 'Invalid bot data received from server';
          return;
        }
        
        state.selectedBot = action.payload;
        state.error = null;
        
        // Update bot in list if it exists
        const botIndex = state.bots.findIndex(bot => bot.bot_id === action.payload.bot_id);
        if (botIndex !== -1) {
          state.bots[botIndex] = action.payload;
          console.log(`Updated bot ${action.payload.bot_id} in list from detailed fetch`);
        } else {
          console.log(`Bot ${action.payload.bot_id} fetched but not in current list`);
        }
      })
      .addCase(fetchBotById.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
      });

    // Create bot
    builder
      .addCase(createBot.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(createBot.fulfilled, (state, action) => {
        state.isLoading = false;
        
        // Validate payload has required fields
        if (!action.payload || !action.payload.bot_id) {
          console.error('createBot.fulfilled: Invalid bot data', action.payload);
          state.error = 'Invalid bot data received after creation';
          return;
        }
        
        // Check if bot already exists (shouldn't happen, but safety check)
        const existingIndex = state.bots.findIndex(bot => bot.bot_id === action.payload.bot_id);
        if (existingIndex !== -1) {
          console.warn(`Bot ${action.payload.bot_id} already exists, updating instead of adding`);
          state.bots[existingIndex] = action.payload;
        } else {
          state.bots.push(action.payload);
          console.log(`Created new bot: ${action.payload.bot_name} (${action.payload.bot_id})`);
        }
        
        state.selectedBot = action.payload;
        state.error = null;
      })
      .addCase(createBot.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
      });

    // Update bot
    builder
      .addCase(updateBot.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(updateBot.fulfilled, (state, action) => {
        state.isLoading = false;
        
        // Validate payload has required fields
        if (!action.payload || !action.payload.bot_id) {
          console.error('updateBot.fulfilled: Invalid bot data', action.payload);
          state.error = 'Invalid bot data received after update';
          return;
        }
        
        const botIndex = state.bots.findIndex(bot => bot.bot_id === action.payload.bot_id);
        if (botIndex !== -1) {
          state.bots[botIndex] = action.payload;
          console.log(`Updated bot: ${action.payload.bot_name} (${action.payload.bot_id})`);
        } else {
          console.warn(`Bot ${action.payload.bot_id} not found in list for update`);
        }
        
        // Update selected bot if it matches
        if (state.selectedBot?.bot_id === action.payload.bot_id) {
          state.selectedBot = action.payload;
        }
        state.error = null;
      })
      .addCase(updateBot.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
      });

    // Delete bot
    builder
      .addCase(deleteBot.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(deleteBot.fulfilled, (state, action) => {
        state.isLoading = false;
        
        // Validate payload is a valid bot ID
        if (!action.payload || typeof action.payload !== 'string') {
          console.error('deleteBot.fulfilled: Invalid bot ID', action.payload);
          state.error = 'Invalid bot ID received after deletion';
          return;
        }
        
        const deletedBotId = action.payload;
        const botIndex = state.bots.findIndex(bot => bot.bot_id === deletedBotId);
        
        if (botIndex !== -1) {
          const deletedBot = state.bots[botIndex];
          state.bots = state.bots.filter(bot => bot.bot_id !== deletedBotId);
          console.log(`Deleted bot: ${deletedBot.bot_name} (${deletedBotId})`);
        } else {
          console.warn(`Bot ${deletedBotId} not found in list for deletion`);
        }
        
        // Clear selected bot if it was deleted
        if (state.selectedBot?.bot_id === deletedBotId) {
          state.selectedBot = null;
          console.log(`Cleared selected bot after deletion: ${deletedBotId}`);
        }
        
        state.error = null;
      })
      .addCase(deleteBot.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
      });

    // Start bot
    builder
      .addCase(startBot.pending, (state) => {
        // Don't set global loading for bot lifecycle operations
        state.error = null;
      })
      .addCase(startBot.fulfilled, (state, action) => {
        const { botId, status, timestamp } = action.payload;
        
        if (!botId || !status) {
          console.error('startBot.fulfilled: Invalid payload', action.payload);
          return;
        }
        
        const updateTime = timestamp || new Date().toISOString();
        const botIndex = state.bots.findIndex(bot => bot.bot_id === botId);
        
        if (botIndex !== -1) {
          state.bots[botIndex].status = status;
          state.bots[botIndex].updatedAt = updateTime;
          console.log(`Started bot: ${state.bots[botIndex].bot_name} (${botId})`);
        } else {
          console.warn(`Bot ${botId} not found for start operation`);
        }
        
        if (state.selectedBot?.bot_id === botId) {
          state.selectedBot.status = status;
          state.selectedBot.updatedAt = updateTime;
        }
      })
      .addCase(startBot.rejected, (state, action) => {
        state.error = action.payload as string;
        console.error('Failed to start bot:', action.payload);
      });

    // Stop bot
    builder
      .addCase(stopBot.pending, (state) => {
        state.error = null;
      })
      .addCase(stopBot.fulfilled, (state, action) => {
        const { botId, status, timestamp } = action.payload;
        
        if (!botId || !status) {
          console.error('stopBot.fulfilled: Invalid payload', action.payload);
          return;
        }
        
        const updateTime = timestamp || new Date().toISOString();
        const botIndex = state.bots.findIndex(bot => bot.bot_id === botId);
        
        if (botIndex !== -1) {
          state.bots[botIndex].status = status;
          state.bots[botIndex].updatedAt = updateTime;
          console.log(`Stopped bot: ${state.bots[botIndex].bot_name} (${botId})`);
        } else {
          console.warn(`Bot ${botId} not found for stop operation`);
        }
        
        if (state.selectedBot?.bot_id === botId) {
          state.selectedBot.status = status;
          state.selectedBot.updatedAt = updateTime;
        }
      })
      .addCase(stopBot.rejected, (state, action) => {
        state.error = action.payload as string;
        console.error('Failed to stop bot:', action.payload);
      });

    // Pause bot
    builder
      .addCase(pauseBot.pending, (state) => {
        state.error = null;
      })
      .addCase(pauseBot.fulfilled, (state, action) => {
        const { botId, status, timestamp } = action.payload;
        
        if (!botId || !status) {
          console.error('pauseBot.fulfilled: Invalid payload', action.payload);
          return;
        }
        
        const updateTime = timestamp || new Date().toISOString();
        const botIndex = state.bots.findIndex(bot => bot.bot_id === botId);
        
        if (botIndex !== -1) {
          state.bots[botIndex].status = status;
          state.bots[botIndex].updatedAt = updateTime;
          console.log(`Paused bot: ${state.bots[botIndex].bot_name} (${botId})`);
        } else {
          console.warn(`Bot ${botId} not found for pause operation`);
        }
        
        if (state.selectedBot?.bot_id === botId) {
          state.selectedBot.status = status;
          state.selectedBot.updatedAt = updateTime;
        }
      })
      .addCase(pauseBot.rejected, (state, action) => {
        state.error = action.payload as string;
        console.error('Failed to pause bot:', action.payload);
      });

    // Resume bot
    builder
      .addCase(resumeBot.pending, (state) => {
        state.error = null;
      })
      .addCase(resumeBot.fulfilled, (state, action) => {
        const { botId, status, timestamp } = action.payload;
        
        if (!botId || !status) {
          console.error('resumeBot.fulfilled: Invalid payload', action.payload);
          return;
        }
        
        const updateTime = timestamp || new Date().toISOString();
        const botIndex = state.bots.findIndex(bot => bot.bot_id === botId);
        
        if (botIndex !== -1) {
          state.bots[botIndex].status = status;
          state.bots[botIndex].updatedAt = updateTime;
          console.log(`Resumed bot: ${state.bots[botIndex].bot_name} (${botId})`);
        } else {
          console.warn(`Bot ${botId} not found for resume operation`);
        }
        
        if (state.selectedBot?.bot_id === botId) {
          state.selectedBot.status = status;
          state.selectedBot.updatedAt = updateTime;
        }
      })
      .addCase(resumeBot.rejected, (state, action) => {
        state.error = action.payload as string;
        console.error('Failed to resume bot:', action.payload);
      });

    // Fetch bot performance
    builder
      .addCase(fetchBotPerformance.pending, (state) => {
        // Don't set global loading for performance fetches
        state.error = null;
      })
      .addCase(fetchBotPerformance.fulfilled, (state, action) => {
        const { botId, performance } = action.payload;
        
        if (!botId) {
          console.error('fetchBotPerformance.fulfilled: Invalid botId', action.payload);
          return;
        }
        
        // Convert legacy BotPerformance to BotMetrics if needed
        const convertToMetrics = (perf: BotPerformance): BotMetrics => ({
          bot_id: botId,
          uptime_seconds: 0,
          total_trades: perf.totalTrades || 0,
          successful_trades: perf.winningTrades || 0,
          failed_trades: perf.losingTrades || 0,
          total_pnl: perf.totalPnl || 0,
          cpu_usage_percent: 0,
          memory_usage_mb: 0,
          win_rate: perf.winRate || 0,
          health_score: 100,
          timestamp: new Date().toISOString()
        });
        
        const metrics = performance ? convertToMetrics(performance) : undefined;
        const botIndex = state.bots.findIndex(bot => bot.bot_id === botId);
        
        if (botIndex !== -1) {
          state.bots[botIndex].metrics = metrics;
          console.log(`Updated performance metrics for bot: ${botId}`);
        } else {
          console.warn(`Bot ${botId} not found for performance update`);
        }
        
        if (state.selectedBot?.bot_id === botId) {
          state.selectedBot.metrics = metrics;
        }
      })
      .addCase(fetchBotPerformance.rejected, (state, action) => {
        // Don't set error for performance fetch failures as they're non-critical
        console.warn('Failed to fetch bot performance:', action.payload);
      });
  },
});

// Export actions
export const {
  clearError,
  setSelectedBot,
  updateFilters,
  clearFilters,
  updateBotStatus,
  updateBotPerformance,
} = botSlice.actions;

// Export selectors
export const selectBots = (state: { bots: BotState }) => state.bots.bots;
export const selectSelectedBot = (state: { bots: BotState }) => state.bots.selectedBot;
export const selectBotLoading = (state: { bots: BotState }) => state.bots.isLoading;
export const selectBotError = (state: { bots: BotState }) => state.bots.error;
export const selectBotFilters = (state: { bots: BotState }) => state.bots.filters;

export const selectFilteredBots = (state: { bots: BotState }) => {
  const { bots, filters } = state.bots;
  
  if (!Array.isArray(bots)) {
    console.warn('selectFilteredBots: bots is not an array', bots);
    return [];
  }
  
  return bots.filter(bot => {
    // Ensure bot has required fields
    if (!bot || !bot.bot_id) {
      console.warn('selectFilteredBots: Invalid bot object', bot);
      return false;
    }
    
    // Filter by status
    if (filters.status && filters.status.length > 0) {
      if (!filters.status.includes(bot.status)) return false;
    }
    
    // Filter by exchange (check if bot.exchange exists)
    if (filters.exchange && filters.exchange.length > 0) {
      if (!bot.exchange || !filters.exchange.includes(bot.exchange)) return false;
    }
    
    // Filter by strategy (check if bot.strategy_name exists)
    if (filters.strategy && filters.strategy.length > 0) {
      if (!bot.strategy_name || !filters.strategy.includes(bot.strategy_name)) return false;
    }
    
    // Filter by search term (with null/undefined protection)
    if (filters.searchTerm && filters.searchTerm.trim()) {
      const searchTerm = filters.searchTerm.toLowerCase().trim();
      const matchesName = (bot.bot_name || '').toLowerCase().includes(searchTerm);
      const matchesStrategy = (bot.strategy_name || '').toLowerCase().includes(searchTerm);
      const matchesExchange = (bot.exchange || '').toLowerCase().includes(searchTerm);
      
      if (!matchesName && !matchesStrategy && !matchesExchange) return false;
    }
    
    return true;
  });
};

export const selectBotById = (state: { bots: BotState }, botId: string) => {
  if (!botId || !Array.isArray(state.bots.bots)) {
    return undefined;
  }
  return state.bots.bots.find(bot => bot?.bot_id === botId);
};

export const selectRunningBots = (state: { bots: BotState }) => {
  if (!Array.isArray(state.bots.bots)) {
    return [];
  }
  return state.bots.bots.filter(bot => bot?.status === BotStatus.RUNNING);
};

export const selectBotsByExchange = (state: { bots: BotState }, exchange: string) => {
  if (!exchange || !Array.isArray(state.bots.bots)) {
    return [];
  }
  return state.bots.bots.filter(bot => bot?.exchange === exchange);
};

// Additional utility selectors
export const selectBotCount = (state: { bots: BotState }) => {
  return Array.isArray(state.bots.bots) ? state.bots.bots.length : 0;
};

export const selectBotsByStatus = (state: { bots: BotState }, status: BotStatus) => {
  if (!status || !Array.isArray(state.bots.bots)) {
    return [];
  }
  return state.bots.bots.filter(bot => bot?.status === status);
};

export const selectBotStatusCounts = (state: { bots: BotState }) => {
  if (!Array.isArray(state.bots.bots)) {
    return { running: 0, stopped: 0, error: 0, paused: 0, total: 0 };
  }
  
  const counts = {
    running: 0,
    stopped: 0,
    error: 0,
    paused: 0,
    total: state.bots.bots.length
  };
  
  state.bots.bots.forEach(bot => {
    if (!bot?.status) return;
    
    switch (bot.status) {
      case BotStatus.RUNNING:
        counts.running++;
        break;
      case BotStatus.STOPPED:
        counts.stopped++;
        break;
      case BotStatus.ERROR:
        counts.error++;
        break;
      case BotStatus.PAUSED:
        counts.paused++;
        break;
    }
  });
  
  return counts;
};

// Export reducer
export default botSlice.reducer;