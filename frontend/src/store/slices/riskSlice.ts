/**
 * Risk management slice for Redux store
 */

import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import { RiskState, RiskMetrics, CircuitBreaker, RiskAlert } from '@/types';
import { riskAPI } from '@/services/api/riskAPI';

const initialState: RiskState = {
  metrics: null,
  circuitBreakers: [],
  alerts: [],
  isLoading: false,
  error: null,
};

export const fetchRiskMetrics = createAsyncThunk(
  'risk/fetchMetrics',
  async (_, { rejectWithValue }) => {
    try {
      const response = await riskAPI.getRiskMetrics();
      return response.data;
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Failed to fetch risk metrics');
    }
  }
);

export const fetchCircuitBreakers = createAsyncThunk(
  'risk/fetchCircuitBreakers',
  async (_, { rejectWithValue }) => {
    try {
      const response = await riskAPI.getCircuitBreakers();
      return response.data;
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Failed to fetch circuit breakers');
    }
  }
);

const riskSlice = createSlice({
  name: 'risk',
  initialState,
  reducers: {
    clearError: (state) => {
      state.error = null;
    },
    updateRiskMetrics: (state, action) => {
      state.metrics = action.payload;
    },
    addAlert: (state, action) => {
      state.alerts.unshift(action.payload);
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchRiskMetrics.fulfilled, (state, action) => {
        state.metrics = action.payload;
      })
      .addCase(fetchCircuitBreakers.fulfilled, (state, action) => {
        state.circuitBreakers = action.payload;
      });
  },
});

export const { clearError, updateRiskMetrics, addAlert } = riskSlice.actions;
export default riskSlice.reducer;