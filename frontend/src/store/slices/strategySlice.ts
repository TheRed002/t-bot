/**
 * Strategy slice for Redux store
 * Manages trading strategies and backtesting
 */

import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import { StrategyState, Strategy, BacktestResult } from '@/types';
import { strategyAPI } from '@/services/api/strategyAPI';

const initialState: StrategyState = {
  // Templates
  templates: [],
  templatesLoading: false,
  templatesError: null,
  
  // Strategies
  strategies: [],
  selectedStrategy: null,
  
  // Configuration
  activeConfiguration: null,
  configurationHistory: [],
  
  // Backtesting
  backtestResults: [],
  isBacktesting: false,
  backtestProgress: 0,
  
  // UI State
  isLoading: false,
  error: null,
  
  // Filters
  filters: {
    sortBy: 'name',
    sortOrder: 'asc'
  }
};

export const fetchStrategies = createAsyncThunk(
  'strategies/fetchStrategies',
  async (_, { rejectWithValue }) => {
    try {
      const response = await strategyAPI.getStrategies();
      return response.data;
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Failed to fetch strategies');
    }
  }
);

export const runBacktest = createAsyncThunk(
  'strategies/runBacktest',
  async (
    { strategyId, params }: { strategyId: string; params: any },
    { rejectWithValue }
  ) => {
    try {
      const response = await strategyAPI.runBacktest(strategyId, params);
      return response.data;
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Backtest failed');
    }
  }
);

export const deployStrategy = createAsyncThunk(
  'strategies/deployStrategy',
  async (strategy: Strategy, { rejectWithValue }) => {
    try {
      // Use updateStrategy API method for now
      const response = await strategyAPI.updateStrategy(strategy.id, strategy);
      return response.data;
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Failed to deploy strategy');
    }
  }
);

export const updateStrategy = createAsyncThunk(
  'strategies/updateStrategy',
  async (strategy: Strategy, { rejectWithValue }) => {
    try {
      const response = await strategyAPI.updateStrategy(strategy.id, strategy);
      return response.data;
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Failed to update strategy');
    }
  }
);

const strategySlice = createSlice({
  name: 'strategies',
  initialState,
  reducers: {
    clearError: (state) => {
      state.error = null;
    },
    setSelectedStrategy: (state, action) => {
      state.selectedStrategy = action.payload;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchStrategies.fulfilled, (state, action) => {
        state.strategies = action.payload;
      })
      .addCase(runBacktest.pending, (state) => {
        state.isBacktesting = true;
      })
      .addCase(runBacktest.fulfilled, (state, action) => {
        state.isBacktesting = false;
        state.backtestResults.push(action.payload);
      })
      .addCase(runBacktest.rejected, (state, action) => {
        state.isBacktesting = false;
        state.error = action.payload as string;
      })
      .addCase(deployStrategy.pending, (state) => {
        state.isLoading = true;
      })
      .addCase(deployStrategy.fulfilled, (state, action) => {
        state.isLoading = false;
        const index = state.strategies.findIndex(s => s.id === action.payload.id);
        if (index !== -1) {
          state.strategies[index] = action.payload;
        } else {
          state.strategies.push(action.payload);
        }
      })
      .addCase(deployStrategy.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
      })
      .addCase(updateStrategy.pending, (state) => {
        state.isLoading = true;
      })
      .addCase(updateStrategy.fulfilled, (state, action) => {
        state.isLoading = false;
        const index = state.strategies.findIndex(s => s.id === action.payload.id);
        if (index !== -1) {
          state.strategies[index] = action.payload;
        }
      })
      .addCase(updateStrategy.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
      });
  },
});

export const { clearError, setSelectedStrategy } = strategySlice.actions;

// Selectors
export const selectStrategies = (state: { strategies: StrategyState }) => state.strategies.strategies;
export const selectSelectedStrategy = (state: { strategies: StrategyState }) => state.strategies.selectedStrategy;
export const selectStrategyLoading = (state: { strategies: StrategyState }) => state.strategies.isLoading;
export const selectStrategyError = (state: { strategies: StrategyState }) => state.strategies.error;

export default strategySlice.reducer;