/**
 * Bot management slice for Redux store
 * Manages trading bot instances and their states
 */

import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { BotState, BotInstance, BotStatus, BotFilters, BotConfig } from '@/types';
import { botAPI } from '@/services/api/botAPI';

// Initial state
const initialState: BotState = {
  bots: [],
  selectedBot: null,
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
  async (filters?: BotFilters, { rejectWithValue }) => {
    try {
      const response = await botAPI.getBots(filters);
      return response.data;
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Failed to fetch bots');
    }
  }
);

export const fetchBotById = createAsyncThunk(
  'bots/fetchBotById',
  async (botId: string, { rejectWithValue }) => {
    try {
      const response = await botAPI.getBotById(botId);
      return response.data;
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Failed to fetch bot');
    }
  }
);

export const createBot = createAsyncThunk(
  'bots/createBot',
  async (
    botData: {
      name: string;
      strategyType: string;
      exchange: string;
      config: BotConfig;
    },
    { rejectWithValue }
  ) => {
    try {
      const response = await botAPI.createBot(botData);
      return response.data;
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Failed to create bot');
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
      const response = await botAPI.updateBot(botId, updates);
      return response.data;
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Failed to update bot');
    }
  }
);

export const deleteBot = createAsyncThunk(
  'bots/deleteBot',
  async (botId: string, { rejectWithValue }) => {
    try {
      await botAPI.deleteBot(botId);
      return botId;
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Failed to delete bot');
    }
  }
);

export const startBot = createAsyncThunk(
  'bots/startBot',
  async (botId: string, { rejectWithValue }) => {
    try {
      const response = await botAPI.startBot(botId);
      return { botId, status: 'running' as BotStatus };
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Failed to start bot');
    }
  }
);

export const stopBot = createAsyncThunk(
  'bots/stopBot',
  async (botId: string, { rejectWithValue }) => {
    try {
      const response = await botAPI.stopBot(botId);
      return { botId, status: 'stopped' as BotStatus };
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Failed to stop bot');
    }
  }
);

export const pauseBot = createAsyncThunk(
  'bots/pauseBot',
  async (botId: string, { rejectWithValue }) => {
    try {
      const response = await botAPI.pauseBot(botId);
      return { botId, status: 'paused' as BotStatus };
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Failed to pause bot');
    }
  }
);

export const fetchBotPerformance = createAsyncThunk(
  'bots/fetchBotPerformance',
  async (botId: string, { rejectWithValue }) => {
    try {
      const response = await botAPI.getBotPerformance(botId);
      return { botId, performance: response.data };
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Failed to fetch bot performance');
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
    updateBotStatus: (state, action: PayloadAction<{ botId: string; status: BotStatus }>) => {
      const { botId, status } = action.payload;
      const botIndex = state.bots.findIndex(bot => bot.id === botId);
      if (botIndex !== -1) {
        state.bots[botIndex].status = status;
        state.bots[botIndex].updatedAt = new Date().toISOString();
      }
      
      // Update selected bot if it matches
      if (state.selectedBot?.id === botId) {
        state.selectedBot.status = status;
        state.selectedBot.updatedAt = new Date().toISOString();
      }
    },
    
    // Update bot performance (for real-time updates)
    updateBotPerformance: (state, action: PayloadAction<{ botId: string; performance: any }>) => {
      const { botId, performance } = action.payload;
      const botIndex = state.bots.findIndex(bot => bot.id === botId);
      if (botIndex !== -1) {
        state.bots[botIndex].performance = performance;
        state.bots[botIndex].updatedAt = new Date().toISOString();
      }
      
      // Update selected bot if it matches
      if (state.selectedBot?.id === botId) {
        state.selectedBot.performance = performance;
        state.selectedBot.updatedAt = new Date().toISOString();
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
        state.bots = action.payload;
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
        state.selectedBot = action.payload;
        state.error = null;
        
        // Update bot in list if it exists
        const botIndex = state.bots.findIndex(bot => bot.id === action.payload.id);
        if (botIndex !== -1) {
          state.bots[botIndex] = action.payload;
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
        state.bots.push(action.payload);
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
        const botIndex = state.bots.findIndex(bot => bot.id === action.payload.id);
        if (botIndex !== -1) {
          state.bots[botIndex] = action.payload;
        }
        
        // Update selected bot if it matches
        if (state.selectedBot?.id === action.payload.id) {
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
        state.bots = state.bots.filter(bot => bot.id !== action.payload);
        
        // Clear selected bot if it was deleted
        if (state.selectedBot?.id === action.payload) {
          state.selectedBot = null;
        }
        state.error = null;
      })
      .addCase(deleteBot.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
      });

    // Start bot
    builder
      .addCase(startBot.fulfilled, (state, action) => {
        const { botId, status } = action.payload;
        const botIndex = state.bots.findIndex(bot => bot.id === botId);
        if (botIndex !== -1) {
          state.bots[botIndex].status = status;
        }
        
        if (state.selectedBot?.id === botId) {
          state.selectedBot.status = status;
        }
      });

    // Stop bot
    builder
      .addCase(stopBot.fulfilled, (state, action) => {
        const { botId, status } = action.payload;
        const botIndex = state.bots.findIndex(bot => bot.id === botId);
        if (botIndex !== -1) {
          state.bots[botIndex].status = status;
        }
        
        if (state.selectedBot?.id === botId) {
          state.selectedBot.status = status;
        }
      });

    // Pause bot
    builder
      .addCase(pauseBot.fulfilled, (state, action) => {
        const { botId, status } = action.payload;
        const botIndex = state.bots.findIndex(bot => bot.id === botId);
        if (botIndex !== -1) {
          state.bots[botIndex].status = status;
        }
        
        if (state.selectedBot?.id === botId) {
          state.selectedBot.status = status;
        }
      });

    // Fetch bot performance
    builder
      .addCase(fetchBotPerformance.fulfilled, (state, action) => {
        const { botId, performance } = action.payload;
        const botIndex = state.bots.findIndex(bot => bot.id === botId);
        if (botIndex !== -1) {
          state.bots[botIndex].performance = performance;
        }
        
        if (state.selectedBot?.id === botId) {
          state.selectedBot.performance = performance;
        }
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
  
  return bots.filter(bot => {
    // Filter by status
    if (filters.status && filters.status.length > 0) {
      if (!filters.status.includes(bot.status)) return false;
    }
    
    // Filter by exchange
    if (filters.exchange && filters.exchange.length > 0) {
      if (!filters.exchange.includes(bot.exchange)) return false;
    }
    
    // Filter by strategy
    if (filters.strategy && filters.strategy.length > 0) {
      if (!filters.strategy.includes(bot.strategyType)) return false;
    }
    
    // Filter by search term
    if (filters.searchTerm) {
      const searchTerm = filters.searchTerm.toLowerCase();
      const matchesName = bot.name.toLowerCase().includes(searchTerm);
      const matchesStrategy = bot.strategyType.toLowerCase().includes(searchTerm);
      const matchesExchange = bot.exchange.toLowerCase().includes(searchTerm);
      
      if (!matchesName && !matchesStrategy && !matchesExchange) return false;
    }
    
    return true;
  });
};

export const selectBotById = (state: { bots: BotState }, botId: string) => 
  state.bots.bots.find(bot => bot.id === botId);

export const selectRunningBots = (state: { bots: BotState }) =>
  state.bots.bots.filter(bot => bot.status === 'running');

export const selectBotsByExchange = (state: { bots: BotState }, exchange: string) =>
  state.bots.bots.filter(bot => bot.exchange === exchange);

// Export reducer
export default botSlice.reducer;