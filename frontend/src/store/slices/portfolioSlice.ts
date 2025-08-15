/**
 * Portfolio slice for Redux store
 * Manages portfolio positions, balances, and P&L tracking
 */

import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { PortfolioState, Position, Balance, PortfolioSummary, PortfolioFilters } from '@/types';
import { portfolioAPI } from '@/services/api/portfolioAPI';

const initialState: PortfolioState = {
  summary: null,
  positions: [],
  balances: [],
  isLoading: false,
  error: null,
  filters: {
    exchange: undefined,
    currency: undefined,
    positionType: undefined,
    showZeroBalances: false,
  },
};

// Async thunks
export const fetchPortfolioSummary = createAsyncThunk(
  'portfolio/fetchSummary',
  async (_, { rejectWithValue }) => {
    try {
      const response = await portfolioAPI.getSummary();
      return response.data;
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Failed to fetch portfolio summary');
    }
  }
);

export const fetchPositions = createAsyncThunk(
  'portfolio/fetchPositions',
  async (filters?: PortfolioFilters, { rejectWithValue }) => {
    try {
      const response = await portfolioAPI.getPositions(filters);
      return response.data;
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Failed to fetch positions');
    }
  }
);

export const fetchBalances = createAsyncThunk(
  'portfolio/fetchBalances',
  async (filters?: PortfolioFilters, { rejectWithValue }) => {
    try {
      const response = await portfolioAPI.getBalances(filters);
      return response.data;
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Failed to fetch balances');
    }
  }
);

const portfolioSlice = createSlice({
  name: 'portfolio',
  initialState,
  reducers: {
    clearError: (state) => {
      state.error = null;
    },
    updateFilters: (state, action: PayloadAction<Partial<PortfolioFilters>>) => {
      state.filters = { ...state.filters, ...action.payload };
    },
    updatePositionPrice: (state, action: PayloadAction<{ symbol: string; price: number }>) => {
      const { symbol, price } = action.payload;
      state.positions.forEach(position => {
        if (position.symbol === symbol) {
          position.currentPrice = price;
          position.unrealizedPnl = (price - position.entryPrice) * position.quantity;
        }
      });
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchPortfolioSummary.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(fetchPortfolioSummary.fulfilled, (state, action) => {
        state.isLoading = false;
        state.summary = action.payload;
      })
      .addCase(fetchPortfolioSummary.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
      })
      .addCase(fetchPositions.fulfilled, (state, action) => {
        state.positions = action.payload;
      })
      .addCase(fetchBalances.fulfilled, (state, action) => {
        state.balances = action.payload;
      });
  },
});

export const { clearError, updateFilters, updatePositionPrice } = portfolioSlice.actions;

export const selectPortfolioSummary = (state: { portfolio: PortfolioState }) => state.portfolio.summary;
export const selectPositions = (state: { portfolio: PortfolioState }) => state.portfolio.positions;
export const selectBalances = (state: { portfolio: PortfolioState }) => state.portfolio.balances;
export const selectPortfolioLoading = (state: { portfolio: PortfolioState }) => state.portfolio.isLoading;

export default portfolioSlice.reducer;