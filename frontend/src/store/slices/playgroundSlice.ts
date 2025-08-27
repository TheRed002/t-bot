/**
 * Playground Redux Slice
 * State management for playground functionality including configurations, executions, and results
 */

import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { RootState } from '../index';
import {
  PlaygroundState,
  PlaygroundConfiguration,
  PlaygroundExecution,
  PlaygroundBatch,
  ABTest,
  PlaygroundPreset,
  PlaygroundMetrics,
  PlaygroundTrade,
  PlaygroundLog,
  DateRange
} from '@/types';
import { playgroundAPI } from '@/services/api/playgroundAPI';

// Initial state
const initialState: PlaygroundState = {
  // Strategy templates
  availableTemplates: [],
  templatesLoading: false,
  
  // Configurations
  configurations: [],
  activeConfiguration: null,
  configurationHistory: [],
  
  // Executions
  executions: [],
  activeExecution: null,
  comparisonExecutions: [],
  
  // A/B Testing
  abTests: [],
  activeABTest: null,
  
  // Presets
  presets: [],
  presetsLoading: false,
  
  // UI State
  isLoading: false,
  error: null,
  
  // Filters
  filters: {},
  
  // Optimization state
  optimization: {
    isOptimizing: false,
    optimizationProgress: 0,
    optimizationHistory: []
  }
};

// Async thunks for API calls
export const fetchConfigurations = createAsyncThunk(
  'playground/fetchConfigurations',
  async (_, { rejectWithValue }) => {
    try {
      const response = await playgroundAPI.getConfigurations();
      return response.data;
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Failed to fetch configurations');
    }
  }
);

export const saveConfiguration = createAsyncThunk(
  'playground/saveConfiguration',
  async (configuration: PlaygroundConfiguration, { rejectWithValue }) => {
    try {
      const response = await playgroundAPI.saveConfiguration(configuration);
      return response.data;
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Failed to save configuration');
    }
  }
);

export const deleteConfiguration = createAsyncThunk(
  'playground/deleteConfiguration',
  async (configurationId: string, { rejectWithValue }) => {
    try {
      await playgroundAPI.deleteConfiguration(configurationId);
      return configurationId;
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Failed to delete configuration');
    }
  }
);

export const startExecution = createAsyncThunk(
  'playground/startExecution',
  async (execution: Omit<PlaygroundExecution, 'id'>, { rejectWithValue }) => {
    try {
      const response = await playgroundAPI.startExecution(execution);
      return response.data;
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Failed to start execution');
    }
  }
);

export const controlExecution = createAsyncThunk(
  'playground/controlExecution',
  async ({ executionId, action }: { executionId: string; action: 'pause' | 'resume' | 'stop' }, { rejectWithValue }) => {
    try {
      const response = await playgroundAPI.controlExecution(executionId, action);
      return response.data;
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Failed to control execution');
    }
  }
);

export const fetchExecutions = createAsyncThunk(
  'playground/fetchExecutions',
  async (filters: { status?: string[]; mode?: string[]; dateRange?: DateRange } = {}, { rejectWithValue }) => {
    try {
      const response = await playgroundAPI.getExecutions(filters);
      return response.data;
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Failed to fetch executions');
    }
  }
);

export const fetchExecutionLogs = createAsyncThunk(
  'playground/fetchExecutionLogs',
  async (executionId: string, { rejectWithValue }) => {
    try {
      const response = await playgroundAPI.getExecutionLogs(executionId);
      return { executionId, logs: response.data };
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Failed to fetch execution logs');
    }
  }
);

export const fetchExecutionTrades = createAsyncThunk(
  'playground/fetchExecutionTrades',
  async (executionId: string, { rejectWithValue }) => {
    try {
      const response = await playgroundAPI.getExecutionTrades(executionId);
      return { executionId, trades: response.data };
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Failed to fetch execution trades');
    }
  }
);

export const createABTest = createAsyncThunk(
  'playground/createABTest',
  async (abTest: Omit<ABTest, 'id' | 'createdAt'>, { rejectWithValue }) => {
    try {
      const response = await playgroundAPI.createABTest(abTest);
      return response.data;
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Failed to create A/B test');
    }
  }
);

export const runABTest = createAsyncThunk(
  'playground/runABTest',
  async (abTestId: string, { rejectWithValue }) => {
    try {
      const response = await playgroundAPI.runABTest(abTestId);
      return response.data;
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Failed to run A/B test');
    }
  }
);

export const startBatchOptimization = createAsyncThunk(
  'playground/startBatchOptimization',
  async (batch: Omit<PlaygroundBatch, 'id' | 'startTime'>, { rejectWithValue }) => {
    try {
      const response = await playgroundAPI.startBatchOptimization(batch);
      return response.data;
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Failed to start batch optimization');
    }
  }
);

export const fetchPresets = createAsyncThunk(
  'playground/fetchPresets',
  async (category: string | undefined, { rejectWithValue }) => {
    try {
      const response = await playgroundAPI.getPresets(category);
      return response.data;
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Failed to fetch presets');
    }
  }
);

// Playground slice
const playgroundSlice = createSlice({
  name: 'playground',
  initialState,
  reducers: {
    // Configuration management
    setActiveConfiguration: (state, action: PayloadAction<PlaygroundConfiguration | null>) => {
      state.activeConfiguration = action.payload;
    },
    updateConfigurationField: (state, action: PayloadAction<{ path: string[]; value: any }>) => {
      if (state.activeConfiguration) {
        const { path, value } = action.payload;
        let current: any = state.activeConfiguration;
        
        for (let i = 0; i < path.length - 1; i++) {
          if (!current[path[i]]) current[path[i]] = {};
          current = current[path[i]];
        }
        
        current[path[path.length - 1]] = value;
      }
    },
    resetActiveConfiguration: (state) => {
      state.activeConfiguration = null;
    },

    // Execution management
    setActiveExecution: (state, action: PayloadAction<PlaygroundExecution | null>) => {
      state.activeExecution = action.payload;
      
      // Update execution in executions array
      if (action.payload) {
        const index = state.executions.findIndex(e => e.id === action.payload!.id);
        if (index >= 0) {
          state.executions[index] = action.payload;
        } else {
          state.executions.push(action.payload);
        }
      }
    },
    updateExecutionProgress: (state, action: PayloadAction<{ executionId: string; progress: number }>) => {
      const { executionId, progress } = action.payload;
      
      // Update active execution
      if (state.activeExecution && state.activeExecution.id === executionId) {
        state.activeExecution.progress = progress;
      }
      
      // Update in executions array
      const execution = state.executions.find(e => e.id === executionId);
      if (execution) {
        execution.progress = progress;
      }
    },
    updateExecutionStatus: (state, action: PayloadAction<{ executionId: string; status: PlaygroundExecution['status'] }>) => {
      const { executionId, status } = action.payload;
      
      // Update active execution
      if (state.activeExecution && state.activeExecution.id === executionId) {
        state.activeExecution.status = status;
      }
      
      // Update in executions array
      const execution = state.executions.find(e => e.id === executionId);
      if (execution) {
        execution.status = status;
      }
    },
    addExecutionLog: (state, action: PayloadAction<{ executionId: string; log: PlaygroundLog }>) => {
      const { executionId, log } = action.payload;
      
      // Update active execution
      if (state.activeExecution && state.activeExecution.id === executionId) {
        state.activeExecution.logs.push(log);
      }
      
      // Update in executions array
      const execution = state.executions.find(e => e.id === executionId);
      if (execution) {
        execution.logs.push(log);
      }
    },
    updateExecutionMetrics: (state, action: PayloadAction<{ executionId: string; metrics: PlaygroundMetrics }>) => {
      const { executionId, metrics } = action.payload;
      
      // Update active execution
      if (state.activeExecution && state.activeExecution.id === executionId) {
        state.activeExecution.metrics = metrics;
      }
      
      // Update in executions array
      const execution = state.executions.find(e => e.id === executionId);
      if (execution) {
        execution.metrics = metrics;
      }
    },
    addExecutionTrade: (state, action: PayloadAction<{ executionId: string; trade: PlaygroundTrade }>) => {
      const { executionId, trade } = action.payload;
      
      // Update active execution
      if (state.activeExecution && state.activeExecution.id === executionId) {
        if (!state.activeExecution.trades) state.activeExecution.trades = [];
        state.activeExecution.trades.push(trade);
      }
      
      // Update in executions array
      const execution = state.executions.find(e => e.id === executionId);
      if (execution) {
        if (!execution.trades) execution.trades = [];
        execution.trades.push(trade);
      }
    },

    // Comparison management
    addToComparison: (state, action: PayloadAction<PlaygroundExecution>) => {
      const existingIndex = state.comparisonExecutions.findIndex(e => e.id === action.payload.id);
      if (existingIndex === -1) {
        state.comparisonExecutions.push(action.payload);
      }
    },
    removeFromComparison: (state, action: PayloadAction<string>) => {
      state.comparisonExecutions = state.comparisonExecutions.filter(e => e.id !== action.payload);
    },
    clearComparison: (state) => {
      state.comparisonExecutions = [];
    },

    // Filter management
    setFilters: (state, action: PayloadAction<PlaygroundState['filters']>) => {
      state.filters = action.payload;
    },
    updateFilter: (state, action: PayloadAction<{ key: string; value: any }>) => {
      const { key, value } = action.payload;
      state.filters = { ...state.filters, [key]: value };
    },
    clearFilters: (state) => {
      state.filters = {};
    },

    // Error management
    clearError: (state) => {
      state.error = null;
    },
    setError: (state, action: PayloadAction<string>) => {
      state.error = action.payload;
    },

    // Reset playground state
    resetPlaygroundState: () => initialState
  },
  extraReducers: (builder) => {
    // Fetch configurations
    builder
      .addCase(fetchConfigurations.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(fetchConfigurations.fulfilled, (state, action) => {
        state.isLoading = false;
        state.configurations = action.payload;
      })
      .addCase(fetchConfigurations.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
      });

    // Save configuration
    builder
      .addCase(saveConfiguration.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(saveConfiguration.fulfilled, (state, action) => {
        state.isLoading = false;
        const savedConfig = action.payload;
        
        // Update or add configuration
        const existingIndex = state.configurations.findIndex(c => c.id === savedConfig.id);
        if (existingIndex >= 0) {
          state.configurations[existingIndex] = savedConfig;
        } else {
          state.configurations.push(savedConfig);
        }
        
        // Update active configuration if it's the same
        if (state.activeConfiguration && state.activeConfiguration.id === savedConfig.id) {
          state.activeConfiguration = savedConfig;
        }
      })
      .addCase(saveConfiguration.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
      });

    // Delete configuration
    builder
      .addCase(deleteConfiguration.fulfilled, (state, action) => {
        const configId = action.payload;
        state.configurations = state.configurations.filter(c => c.id !== configId);
        
        // Clear active configuration if it was deleted
        if (state.activeConfiguration && state.activeConfiguration.id === configId) {
          state.activeConfiguration = null;
        }
      });

    // Start execution
    builder
      .addCase(startExecution.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(startExecution.fulfilled, (state, action) => {
        state.isLoading = false;
        const execution = action.payload;
        state.executions.push(execution);
        state.activeExecution = execution;
      })
      .addCase(startExecution.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
      });

    // Control execution
    builder
      .addCase(controlExecution.fulfilled, (state, action) => {
        const updatedExecution = action.payload;
        
        // Update active execution
        if (state.activeExecution && state.activeExecution.id === updatedExecution.id) {
          state.activeExecution = updatedExecution;
        }
        
        // Update in executions array
        const index = state.executions.findIndex(e => e.id === updatedExecution.id);
        if (index >= 0) {
          state.executions[index] = updatedExecution;
        }
      });

    // Fetch executions
    builder
      .addCase(fetchExecutions.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(fetchExecutions.fulfilled, (state, action) => {
        state.isLoading = false;
        state.executions = action.payload;
      })
      .addCase(fetchExecutions.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
      });

    // Fetch execution logs
    builder
      .addCase(fetchExecutionLogs.fulfilled, (state, action) => {
        const { executionId, logs } = action.payload;
        
        // Update active execution
        if (state.activeExecution && state.activeExecution.id === executionId) {
          state.activeExecution.logs = logs;
        }
        
        // Update in executions array
        const execution = state.executions.find(e => e.id === executionId);
        if (execution) {
          execution.logs = logs;
        }
      });

    // Fetch execution trades
    builder
      .addCase(fetchExecutionTrades.fulfilled, (state, action) => {
        const { executionId, trades } = action.payload;
        
        // Update active execution
        if (state.activeExecution && state.activeExecution.id === executionId) {
          state.activeExecution.trades = trades;
        }
        
        // Update in executions array
        const execution = state.executions.find(e => e.id === executionId);
        if (execution) {
          execution.trades = trades;
        }
      });

    // Handle other async thunks...
    builder
      .addCase(createABTest.fulfilled, (state) => {
        // A/B test created successfully
        state.isLoading = false;
      })
      .addCase(runABTest.fulfilled, (state) => {
        // A/B test started successfully
        state.isLoading = false;
      })
      .addCase(startBatchOptimization.fulfilled, (state) => {
        // Batch optimization started successfully
        state.isLoading = false;
      })
      .addCase(fetchPresets.fulfilled, (state) => {
        // Presets fetched successfully
        state.isLoading = false;
      });

    // Handle rejected states for other thunks
    builder
      .addMatcher(
        (action) => action.type.endsWith('/rejected'),
        (state, action) => {
          state.isLoading = false;
          state.error = action.payload as string || 'An error occurred';
        }
      );
  }
});

// Action creators
export const {
  setActiveConfiguration,
  updateConfigurationField,
  resetActiveConfiguration,
  setActiveExecution,
  updateExecutionProgress,
  updateExecutionStatus,
  addExecutionLog,
  updateExecutionMetrics,
  addExecutionTrade,
  addToComparison,
  removeFromComparison,
  clearComparison,
  setFilters,
  updateFilter,
  clearFilters,
  clearError,
  setError,
  resetPlaygroundState
} = playgroundSlice.actions;

// Selectors
export const selectPlaygroundState = (state: RootState) => state.playground;
export const selectConfigurations = (state: RootState) => state.playground.configurations;
export const selectActiveConfiguration = (state: RootState) => state.playground.activeConfiguration;
export const selectExecutions = (state: RootState) => state.playground.executions;
export const selectActiveExecution = (state: RootState) => state.playground.activeExecution;
export const selectComparisonExecutions = (state: RootState) => state.playground.comparisonExecutions;
export const selectPlaygroundIsLoading = (state: RootState) => state.playground.isLoading;
export const selectPlaygroundError = (state: RootState) => state.playground.error;
export const selectPlaygroundFilters = (state: RootState) => state.playground.filters;

// Complex selectors
export const selectCompletedExecutions = (state: RootState) =>
  state.playground.executions.filter(e => e.status === 'completed');

export const selectRunningExecutions = (state: RootState) =>
  state.playground.executions.filter(e => e.status === 'running');

export const selectFilteredExecutions = (state: RootState) => {
  const { executions, filters } = state.playground;
  
  return executions.filter(execution => {
    if (filters.status && filters.status.length > 0 && !filters.status.includes(execution.status)) {
      return false;
    }
    
    if (filters.mode && filters.mode.length > 0 && !filters.mode.includes(execution.mode)) {
      return false;
    }
    
    if (filters.dateRange) {
      const executionDate = new Date(execution.startTime || 0);
      const startDate = new Date(filters.dateRange.start);
      const endDate = new Date(filters.dateRange.end);
      
      if (executionDate < startDate || executionDate > endDate) {
        return false;
      }
    }
    
    return true;
  });
};

export const selectBestPerformingExecution = (state: RootState) => {
  const completedExecutions = selectCompletedExecutions(state);
  
  if (completedExecutions.length === 0) return null;
  
  return completedExecutions.reduce((best, current) => {
    if (!best.metrics || !current.metrics) return best;
    
    // Compare by Sharpe ratio as primary metric
    if (current.metrics.sharpeRatio > best.metrics.sharpeRatio) {
      return current;
    }
    
    return best;
  });
};

export default playgroundSlice.reducer;