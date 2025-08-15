/**
 * Strategy slice for Redux store
 * Manages trading strategies and backtesting
 */

import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import { StrategyState, Strategy, BacktestResult } from '@/types';
import { strategyAPI } from '@/services/api/strategyAPI';

const initialState: StrategyState = {
  strategies: [],
  selectedStrategy: null,
  backtestResults: [],
  isLoading: false,
  isBacktesting: false,
  error: null,
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
      });
  },
});

export const { clearError, setSelectedStrategy } = strategySlice.actions;
export default strategySlice.reducer;