/**
 * Tests for playground Redux slice
 */

import { configureStore } from '@reduxjs/toolkit';
import authReducer from '../authSlice';
import botReducer from '../botSlice';
import portfolioReducer from '../portfolioSlice';
import strategyReducer from '../strategySlice';
import riskReducer from '../riskSlice';
import marketReducer from '../marketSlice';
import uiReducer from '../uiSlice';
import websocketReducer from '../websocketSlice';
import playgroundReducer, {
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
  resetPlaygroundState,
  selectPlaygroundState,
  selectActiveConfiguration,
  selectActiveExecution,
  selectCompletedExecutions,
  selectRunningExecutions,
  selectBestPerformingExecution
} from '../playgroundSlice';
import { PlaygroundConfiguration, PlaygroundExecution, PlaygroundMetrics } from '@/types';

// Mock store setup with all required reducers
const createMockStore = (preloadedState = {}) => {
  return configureStore({
    reducer: {
      auth: authReducer,
      bots: botReducer,
      portfolio: portfolioReducer,
      strategies: strategyReducer,
      risk: riskReducer,
      market: marketReducer,
      ui: uiReducer,
      websocket: websocketReducer,
      playground: playgroundReducer
    },
    preloadedState
  });
};

const mockConfiguration: PlaygroundConfiguration = {
  id: 'config-1',
  name: 'Test Configuration',
  description: 'Test description',
  symbols: ['BTC/USDT', 'ETH/USDT'],
  positionSizing: {
    type: 'percentage',
    value: 2,
    maxPositions: 5
  },
  tradingSide: 'both',
  riskSettings: {
    stopLossPercentage: 2,
    takeProfitPercentage: 4,
    maxDrawdownPercentage: 10,
    maxRiskPerTrade: 2
  },
  portfolioSettings: {
    maxPositions: 5,
    allocationStrategy: 'equal_weight',
    rebalanceFrequency: 'daily'
  },
  strategy: {
    type: 'trend_following',
    parameters: {}
  },
  timeframe: '1h'
};

const mockExecution: PlaygroundExecution = {
  id: 'exec-1',
  configurationId: 'config-1',
  mode: 'historical',
  status: 'running',
  progress: 50,
  startTime: '2023-01-01T00:00:00Z',
  settings: {
    speed: 1,
    initialBalance: 10000,
    commission: 0.1
  },
  logs: [],
  metrics: {
    totalReturn: 5.2,
    annualizedReturn: 15.6,
    sharpeRatio: 1.5,
    sortinoRatio: 1.8,
    maxDrawdown: 8.5,
    volatility: 18.3,
    winRate: 65.4,
    profitFactor: 1.8,
    totalTrades: 45,
    avgTradeSize: 250,
    avgHoldingPeriod: 6.5,
    finalBalance: 10520,
    peakBalance: 11200
  }
};

describe('playgroundSlice', () => {
  describe('initial state', () => {
    it('has correct initial state', () => {
      const store = createMockStore();
      const state = selectPlaygroundState(store.getState());

      expect(state).toEqual({
        configurations: [],
        activeConfiguration: null,
        executions: [],
        activeExecution: null,
        comparisonExecutions: [],
        isLoading: false,
        error: null,
        filters: {}
      });
    });
  });

  describe('configuration management', () => {
    it('sets active configuration', () => {
      const store = createMockStore();
      
      store.dispatch(setActiveConfiguration(mockConfiguration));
      
      const activeConfig = selectActiveConfiguration(store.getState());
      expect(activeConfig).toEqual(mockConfiguration);
    });

    it('updates configuration field', () => {
      const store = createMockStore({
        playground: {
          configurations: [],
          activeConfiguration: mockConfiguration,
          executions: [],
          activeExecution: null,
          comparisonExecutions: [],
          isLoading: false,
          error: null,
          filters: {}
        }
      });

      store.dispatch(updateConfigurationField({
        path: ['name'],
        value: 'Updated Name'
      }));

      const activeConfig = selectActiveConfiguration(store.getState());
      expect(activeConfig?.name).toBe('Updated Name');
    });

    it('updates nested configuration field', () => {
      const store = createMockStore({
        playground: {
          configurations: [],
          activeConfiguration: mockConfiguration,
          executions: [],
          activeExecution: null,
          comparisonExecutions: [],
          isLoading: false,
          error: null,
          filters: {}
        }
      });

      store.dispatch(updateConfigurationField({
        path: ['riskSettings', 'stopLossPercentage'],
        value: 3
      }));

      const activeConfig = selectActiveConfiguration(store.getState());
      expect(activeConfig?.riskSettings.stopLossPercentage).toBe(3);
    });

    it('resets active configuration', () => {
      const store = createMockStore({
        playground: {
          configurations: [],
          activeConfiguration: mockConfiguration,
          executions: [],
          activeExecution: null,
          comparisonExecutions: [],
          isLoading: false,
          error: null,
          filters: {}
        }
      });

      store.dispatch(resetActiveConfiguration());

      const activeConfig = selectActiveConfiguration(store.getState());
      expect(activeConfig).toBeNull();
    });
  });

  describe('execution management', () => {
    it('sets active execution', () => {
      const store = createMockStore();
      
      store.dispatch(setActiveExecution(mockExecution));
      
      const activeExecution = selectActiveExecution(store.getState());
      expect(activeExecution).toEqual(mockExecution);
      
      const state = selectPlaygroundState(store.getState());
      expect(state.executions).toContain(mockExecution);
    });

    it('updates execution progress', () => {
      const store = createMockStore({
        playground: {
          configurations: [],
          activeConfiguration: null,
          executions: [mockExecution],
          activeExecution: mockExecution,
          comparisonExecutions: [],
          isLoading: false,
          error: null,
          filters: {}
        }
      });

      store.dispatch(updateExecutionProgress({
        executionId: 'exec-1',
        progress: 75
      }));

      const activeExecution = selectActiveExecution(store.getState());
      expect(activeExecution?.progress).toBe(75);
    });

    it('updates execution status', () => {
      const store = createMockStore({
        playground: {
          configurations: [],
          activeConfiguration: null,
          executions: [mockExecution],
          activeExecution: mockExecution,
          comparisonExecutions: [],
          isLoading: false,
          error: null,
          filters: {}
        }
      });

      store.dispatch(updateExecutionStatus({
        executionId: 'exec-1',
        status: 'completed'
      }));

      const activeExecution = selectActiveExecution(store.getState());
      expect(activeExecution?.status).toBe('completed');
    });

    it('adds execution log', () => {
      const store = createMockStore({
        playground: {
          configurations: [],
          activeConfiguration: null,
          executions: [mockExecution],
          activeExecution: mockExecution,
          comparisonExecutions: [],
          isLoading: false,
          error: null,
          filters: {}
        }
      });

      const newLog = {
        id: 'log-1',
        timestamp: '2023-01-01T01:00:00Z',
        level: 'info' as const,
        category: 'strategy' as const,
        message: 'Test log message'
      };

      store.dispatch(addExecutionLog({
        executionId: 'exec-1',
        log: newLog
      }));

      const activeExecution = selectActiveExecution(store.getState());
      expect(activeExecution?.logs).toContain(newLog);
    });

    it('updates execution metrics', () => {
      const store = createMockStore({
        playground: {
          configurations: [],
          activeConfiguration: null,
          executions: [mockExecution],
          activeExecution: mockExecution,
          comparisonExecutions: [],
          isLoading: false,
          error: null,
          filters: {}
        }
      });

      const newMetrics: PlaygroundMetrics = {
        ...mockExecution.metrics!,
        totalReturn: 10.5
      };

      store.dispatch(updateExecutionMetrics({
        executionId: 'exec-1',
        metrics: newMetrics
      }));

      const activeExecution = selectActiveExecution(store.getState());
      expect(activeExecution?.metrics?.totalReturn).toBe(10.5);
    });

    it('adds execution trade', () => {
      const store = createMockStore({
        playground: {
          configurations: [],
          activeConfiguration: null,
          executions: [mockExecution],
          activeExecution: mockExecution,
          comparisonExecutions: [],
          isLoading: false,
          error: null,
          filters: {}
        }
      });

      const newTrade = {
        id: 'trade-1',
        executionId: 'exec-1',
        symbol: 'BTC/USDT',
        side: 'buy' as const,
        quantity: 0.1,
        price: 45000,
        timestamp: '2023-01-01T02:00:00Z',
        commission: 4.5,
        reason: 'Strategy signal',
        pnl: 150
      };

      store.dispatch(addExecutionTrade({
        executionId: 'exec-1',
        trade: newTrade
      }));

      const activeExecution = selectActiveExecution(store.getState());
      expect(activeExecution?.trades).toContain(newTrade);
    });
  });

  describe('comparison management', () => {
    it('adds execution to comparison', () => {
      const store = createMockStore();
      
      store.dispatch(addToComparison(mockExecution));
      
      const state = selectPlaygroundState(store.getState());
      expect(state.comparisonExecutions).toContain(mockExecution);
    });

    it('does not add duplicate executions to comparison', () => {
      const store = createMockStore({
        playground: {
          configurations: [],
          activeConfiguration: null,
          executions: [],
          activeExecution: null,
          comparisonExecutions: [mockExecution],
          isLoading: false,
          error: null,
          filters: {}
        }
      });

      store.dispatch(addToComparison(mockExecution));

      const state = selectPlaygroundState(store.getState());
      expect(state.comparisonExecutions).toHaveLength(1);
    });

    it('removes execution from comparison', () => {
      const store = createMockStore({
        playground: {
          configurations: [],
          activeConfiguration: null,
          executions: [],
          activeExecution: null,
          comparisonExecutions: [mockExecution],
          isLoading: false,
          error: null,
          filters: {}
        }
      });

      store.dispatch(removeFromComparison('exec-1'));

      const state = selectPlaygroundState(store.getState());
      expect(state.comparisonExecutions).toHaveLength(0);
    });

    it('clears all comparisons', () => {
      const execution2 = { ...mockExecution, id: 'exec-2' };
      const store = createMockStore({
        playground: {
          configurations: [],
          activeConfiguration: null,
          executions: [],
          activeExecution: null,
          comparisonExecutions: [mockExecution, execution2],
          isLoading: false,
          error: null,
          filters: {}
        }
      });

      store.dispatch(clearComparison());

      const state = selectPlaygroundState(store.getState());
      expect(state.comparisonExecutions).toHaveLength(0);
    });
  });

  describe('filter management', () => {
    it('sets filters', () => {
      const store = createMockStore();
      const filters = {
        status: ['running', 'completed'] as ('running' | 'completed')[],
        mode: ['historical'] as ('historical')[]
      };

      store.dispatch(setFilters(filters));

      const state = selectPlaygroundState(store.getState());
      expect(state.filters).toEqual(filters);
    });

    it('updates single filter', () => {
      const store = createMockStore({
        playground: {
          configurations: [],
          activeConfiguration: null,
          executions: [],
          activeExecution: null,
          comparisonExecutions: [],
          isLoading: false,
          error: null,
          filters: { status: ['running'] }
        }
      });

      store.dispatch(updateFilter({
        key: 'mode',
        value: ['historical', 'live']
      }));

      const state = selectPlaygroundState(store.getState());
      expect(state.filters).toEqual({
        status: ['running'],
        mode: ['historical', 'live']
      });
    });

    it('clears filters', () => {
      const store = createMockStore({
        playground: {
          configurations: [],
          activeConfiguration: null,
          executions: [],
          activeExecution: null,
          comparisonExecutions: [],
          isLoading: false,
          error: null,
          filters: { status: ['running'], mode: ['historical'] }
        }
      });

      store.dispatch(clearFilters());

      const state = selectPlaygroundState(store.getState());
      expect(state.filters).toEqual({});
    });
  });

  describe('error management', () => {
    it('sets error', () => {
      const store = createMockStore();
      const errorMessage = 'Test error message';

      store.dispatch(setError(errorMessage));

      const state = selectPlaygroundState(store.getState());
      expect(state.error).toBe(errorMessage);
    });

    it('clears error', () => {
      const store = createMockStore({
        playground: {
          configurations: [],
          activeConfiguration: null,
          executions: [],
          activeExecution: null,
          comparisonExecutions: [],
          isLoading: false,
          error: 'Previous error',
          filters: {}
        }
      });

      store.dispatch(clearError());

      const state = selectPlaygroundState(store.getState());
      expect(state.error).toBeNull();
    });
  });

  describe('state reset', () => {
    it('resets playground state', () => {
      const store = createMockStore({
        playground: {
          configurations: [mockConfiguration],
          activeConfiguration: mockConfiguration,
          executions: [mockExecution],
          activeExecution: mockExecution,
          comparisonExecutions: [mockExecution],
          isLoading: true,
          error: 'Some error',
          filters: { status: ['running'] }
        }
      });

      store.dispatch(resetPlaygroundState());

      const state = selectPlaygroundState(store.getState());
      expect(state).toEqual({
        configurations: [],
        activeConfiguration: null,
        executions: [],
        activeExecution: null,
        comparisonExecutions: [],
        isLoading: false,
        error: null,
        filters: {}
      });
    });
  });

  describe('selectors', () => {
    const completedExecution = {
      ...mockExecution,
      id: 'exec-completed',
      status: 'completed' as const
    };

    const runningExecution = {
      ...mockExecution,
      id: 'exec-running',
      status: 'running' as const
    };

    const stateWithExecutions = {
      playground: {
        configurations: [],
        activeConfiguration: null,
        executions: [completedExecution, runningExecution, mockExecution],
        activeExecution: null,
        comparisonExecutions: [],
        isLoading: false,
        error: null,
        filters: {}
      }
    };

    it('selects completed executions', () => {
      const store = createMockStore(stateWithExecutions);
      const completed = selectCompletedExecutions(store.getState());
      
      expect(completed).toHaveLength(1);
      expect(completed[0].id).toBe('exec-completed');
    });

    it('selects running executions', () => {
      const store = createMockStore(stateWithExecutions);
      const running = selectRunningExecutions(store.getState());
      
      expect(running).toHaveLength(2); // 'running' and 'running' (mockExecution)
      expect(running.map(e => e.id)).toContain('exec-running');
    });

    it('selects best performing execution', () => {
      const betterExecution = {
        ...completedExecution,
        id: 'exec-better',
        metrics: {
          ...completedExecution.metrics!,
          sharpeRatio: 2.5
        }
      };

      const stateWithBetterExecution = {
        playground: {
          ...stateWithExecutions.playground,
          executions: [completedExecution, betterExecution]
        }
      };

      const store = createMockStore(stateWithBetterExecution);
      const best = selectBestPerformingExecution(store.getState());
      
      expect(best?.id).toBe('exec-better');
      expect(best?.metrics?.sharpeRatio).toBe(2.5);
    });

    it('returns null for best performing execution when no completed executions', () => {
      const store = createMockStore({
        playground: {
          configurations: [],
          activeConfiguration: null,
          executions: [runningExecution],
          activeExecution: null,
          comparisonExecutions: [],
          isLoading: false,
          error: null,
          filters: {}
        }
      });

      const best = selectBestPerformingExecution(store.getState());
      expect(best).toBeNull();
    });
  });

  describe('async thunks', () => {
    // Note: These would require mocking the API calls
    // For now, we'll just test that the slice structure supports async actions
    
    it('handles async thunk pending states', () => {
      const store = createMockStore();
      
      // Simulate a pending async action
      store.dispatch({
        type: 'playground/fetchConfigurations/pending'
      });

      // The reducer should handle pending states for loading indicators
      // This would be implemented in the actual async thunk handlers
    });
  });
});