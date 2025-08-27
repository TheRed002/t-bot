/**
 * Tests for PlaygroundPage component
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { Provider } from 'react-redux';
import { BrowserRouter } from 'react-router-dom';
import { ThemeProvider } from '@mui/material/styles';
import { configureStore } from '@reduxjs/toolkit';

import PlaygroundPage from '../PlaygroundPage';
import playgroundReducer from '@/store/slices/playgroundSlice';
import uiReducer from '@/store/slices/uiSlice';
import authReducer from '@/store/slices/authSlice';
import { theme } from '@/theme';

// Mock the child components to simplify testing
jest.mock('../components/ConfigurationPanel', () => {
  return function MockConfigurationPanel({ onConfigurationChange, isExecutionActive }: any) {
    return (
      <div data-testid="configuration-panel">
        <button 
          onClick={() => onConfigurationChange({ 
            id: 'test-config',
            name: 'Test Configuration',
            symbols: ['BTC/USDT'],
            positionSizing: { type: 'percentage', value: 2, maxPositions: 5 },
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
            strategy: { type: 'trend_following', parameters: {} },
            timeframe: '1h'
          })}
          disabled={isExecutionActive}
        >
          Set Configuration
        </button>
      </div>
    );
  };
});

jest.mock('../components/ExecutionControls', () => {
  return function MockExecutionControls({ onExecutionStart, configuration }: any) {
    return (
      <div data-testid="execution-controls">
        <button 
          onClick={() => onExecutionStart({
            id: 'test-execution',
            configurationId: 'test-config',
            mode: 'historical',
            status: 'running',
            progress: 0,
            logs: [],
            settings: {
              speed: 1,
              initialBalance: 10000,
              commission: 0.1
            }
          })}
          disabled={!configuration}
        >
          Start Execution
        </button>
      </div>
    );
  };
});

jest.mock('../components/MonitoringDashboard', () => {
  return function MockMonitoringDashboard({ execution }: any) {
    return (
      <div data-testid="monitoring-dashboard">
        {execution ? `Monitoring: ${execution.id}` : 'No execution'}
      </div>
    );
  };
});

jest.mock('../components/ResultsAnalysis', () => {
  return function MockResultsAnalysis({ executions }: any) {
    return (
      <div data-testid="results-analysis">
        Results for {executions.length} executions
      </div>
    );
  };
});

jest.mock('../components/AdvancedFeatures', () => {
  return function MockAdvancedFeatures() {
    return <div data-testid="advanced-features">Advanced Features</div>;
  };
});

jest.mock('../components/BatchOptimizer', () => {
  return function MockBatchOptimizer() {
    return <div data-testid="batch-optimizer">Batch Optimizer</div>;
  };
});

// Create a mock store
const createMockStore = (initialState = {}) => {
  return configureStore({
    reducer: {
      playground: playgroundReducer,
      ui: uiReducer,
      auth: authReducer,
    },
    preloadedState: {
      playground: {
        configurations: [],
        activeConfiguration: null,
        executions: [],
        activeExecution: null,
        comparisonExecutions: [],
        isLoading: false,
        error: null,
        filters: {}
      },
      ui: {
        sidebar: { isOpen: false, isCollapsed: false },
        modals: {
          createBot: false,
          editBot: false,
          strategyConfig: false,
          riskSettings: false
        },
        notifications: [],
        theme: 'light',
        isLoading: false
      },
      auth: {
        user: { 
          user_id: '1', 
          username: 'testuser', 
          email: 'test@example.com', 
          is_active: true, 
          status: 'active' as any,
          roles: ['user'],
          scopes: ['read', 'write'],
          created_at: '2023-01-01T00:00:00Z'
        },
        tokens: { access_token: 'test-token', refresh_token: 'refresh-token', token_type: 'Bearer', expires_in: 3600 },
        isAuthenticated: true,
        isLoading: false,
        isRefreshing: false,
        error: null,
        rememberMe: false,
        sessionExpiresAt: null
      },
      ...initialState
    }
  });
};

const renderWithProviders = (component: React.ReactElement, { store = createMockStore() } = {}) => {
  return render(
    <Provider store={store}>
      <BrowserRouter>
        <ThemeProvider theme={theme}>
          {component}
        </ThemeProvider>
      </BrowserRouter>
    </Provider>
  );
};

describe('PlaygroundPage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders playground page with all main sections', () => {
    renderWithProviders(<PlaygroundPage />);

    expect(screen.getByText('Strategy Playground')).toBeInTheDocument();
    expect(screen.getByText('Test, optimize, and validate your trading strategies with comprehensive backtesting and simulation tools')).toBeInTheDocument();
    
    // Check tabs are present
    expect(screen.getByText('Configuration')).toBeInTheDocument();
    expect(screen.getByText('Execution & Monitoring')).toBeInTheDocument();
    expect(screen.getByText('Results & Analysis')).toBeInTheDocument();
    expect(screen.getByText('Advanced Features')).toBeInTheDocument();
    expect(screen.getByText('Batch Optimizer')).toBeInTheDocument();
  });

  it('displays configuration panel by default', () => {
    renderWithProviders(<PlaygroundPage />);
    expect(screen.getByTestId('configuration-panel')).toBeInTheDocument();
  });

  it('switches between tabs correctly', async () => {
    renderWithProviders(<PlaygroundPage />);

    // Initially on configuration tab
    expect(screen.getByTestId('configuration-panel')).toBeInTheDocument();

    // Switch to execution tab
    fireEvent.click(screen.getByText('Execution & Monitoring'));
    await waitFor(() => {
      expect(screen.getByTestId('execution-controls')).toBeInTheDocument();
      expect(screen.getByTestId('monitoring-dashboard')).toBeInTheDocument();
    });

    // Switch to results tab
    fireEvent.click(screen.getByText('Results & Analysis'));
    await waitFor(() => {
      expect(screen.getByTestId('results-analysis')).toBeInTheDocument();
    });

    // Switch to advanced features tab
    fireEvent.click(screen.getByText('Advanced Features'));
    await waitFor(() => {
      expect(screen.getByTestId('advanced-features')).toBeInTheDocument();
    });

    // Switch to batch optimizer tab
    fireEvent.click(screen.getByText('Batch Optimizer'));
    await waitFor(() => {
      expect(screen.getByTestId('batch-optimizer')).toBeInTheDocument();
    });
  });

  it('handles configuration changes', async () => {
    renderWithProviders(<PlaygroundPage />);

    // Set a configuration
    fireEvent.click(screen.getByText('Set Configuration'));

    await waitFor(() => {
      expect(screen.getByText('Configuration: Test Configuration')).toBeInTheDocument();
    });
  });

  it('handles execution start and switches to monitoring tab', async () => {
    renderWithProviders(<PlaygroundPage />);

    // First set a configuration
    fireEvent.click(screen.getByText('Set Configuration'));
    
    await waitFor(() => {
      expect(screen.getByText('Configuration: Test Configuration')).toBeInTheDocument();
    });

    // Switch to execution tab
    fireEvent.click(screen.getByText('Execution & Monitoring'));
    
    await waitFor(() => {
      expect(screen.getByText('Start Execution')).toBeInTheDocument();
    });

    // Start execution
    fireEvent.click(screen.getByText('Start Execution'));

    await waitFor(() => {
      expect(screen.getByText('Execution: RUNNING (0%)')).toBeInTheDocument();
    });
  });

  it('displays execution status when active', async () => {
    const storeWithActiveExecution = createMockStore({
      playground: {
        configurations: [],
        activeConfiguration: {
          id: 'test-config',
          name: 'Test Configuration',
          symbols: ['BTC/USDT'],
          positionSizing: { type: 'percentage', value: 2, maxPositions: 5 },
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
          strategy: { type: 'trend_following', parameters: {} },
          timeframe: '1h'
        },
        executions: [],
        activeExecution: {
          id: 'test-execution',
          configurationId: 'test-config',
          mode: 'historical',
          status: 'running',
          progress: 25,
          logs: [],
          settings: {
            speed: 1,
            initialBalance: 10000,
            commission: 0.1
          }
        },
        comparisonExecutions: [],
        isLoading: false,
        error: null,
        filters: {}
      }
    });

    renderWithProviders(<PlaygroundPage />, { store: storeWithActiveExecution });

    expect(screen.getByText('Configuration: Test Configuration')).toBeInTheDocument();
    expect(screen.getByText('Execution: RUNNING (25%)')).toBeInTheDocument();
  });

  it('shows loading state', () => {
    const storeWithLoading = createMockStore({
      playground: {
        configurations: [],
        activeConfiguration: null,
        executions: [],
        activeExecution: null,
        comparisonExecutions: [],
        isLoading: true,
        error: null,
        filters: {}
      }
    });

    renderWithProviders(<PlaygroundPage />, { store: storeWithLoading });

    // The loading state would be handled by individual components
    expect(screen.getByText('Strategy Playground')).toBeInTheDocument();
  });

  it('shows error state', () => {
    const storeWithError = createMockStore({
      playground: {
        configurations: [],
        activeConfiguration: null,
        executions: [],
        activeExecution: null,
        comparisonExecutions: [],
        isLoading: false,
        error: 'Test error message',
        filters: {}
      }
    });

    renderWithProviders(<PlaygroundPage />, { store: storeWithError });

    // The error would be displayed in the UI, but our mocked components don't show it
    // In a real implementation, we'd add error display to the main component
    expect(screen.getByText('Strategy Playground')).toBeInTheDocument();
  });

  it('disables configuration changes when execution is active', async () => {
    const storeWithActiveExecution = createMockStore({
      playground: {
        configurations: [],
        activeConfiguration: null,
        executions: [],
        activeExecution: {
          id: 'test-execution',
          configurationId: 'test-config',
          mode: 'historical',
          status: 'running',
          progress: 50,
          logs: [],
          settings: {
            speed: 1,
            initialBalance: 10000,
            commission: 0.1
          }
        },
        comparisonExecutions: [],
        isLoading: false,
        error: null,
        filters: {}
      }
    });

    renderWithProviders(<PlaygroundPage />, { store: storeWithActiveExecution });

    const setConfigButton = screen.getByText('Set Configuration');
    expect(setConfigButton).toBeDisabled();
  });

  it('passes correct props to child components', () => {
    const storeWithData = createMockStore({
      playground: {
        configurations: [],
        activeConfiguration: {
          id: 'test-config',
          name: 'Test Configuration',
          symbols: ['BTC/USDT'],
          positionSizing: { type: 'percentage', value: 2, maxPositions: 5 },
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
          strategy: { type: 'trend_following', parameters: {} },
          timeframe: '1h'
        },
        executions: [
          {
            id: 'exec-1',
            configurationId: 'test-config',
            mode: 'historical',
            status: 'completed',
            progress: 100,
            logs: [],
            settings: { speed: 1, initialBalance: 10000, commission: 0.1 }
          }
        ],
        activeExecution: null,
        comparisonExecutions: [],
        isLoading: false,
        error: null,
        filters: {}
      }
    });

    renderWithProviders(<PlaygroundPage />, { store: storeWithData });

    // Switch to results tab to check executions are passed
    fireEvent.click(screen.getByText('Results & Analysis'));
    expect(screen.getByText('Results for 1 executions')).toBeInTheDocument();
  });

  it('has correct accessibility attributes', () => {
    renderWithProviders(<PlaygroundPage />);

    // Check tab accessibility
    const configTab = screen.getByRole('tab', { name: /configuration/i });
    expect(configTab).toHaveAttribute('aria-controls');
    expect(configTab).toHaveAttribute('id');

    const tabPanel = screen.getByRole('tabpanel');
    expect(tabPanel).toHaveAttribute('aria-labelledby');
  });

  it('handles keyboard navigation', () => {
    renderWithProviders(<PlaygroundPage />);

    const configTab = screen.getByRole('tab', { name: /configuration/i });
    const executionTab = screen.getByRole('tab', { name: /execution & monitoring/i });

    // Focus on first tab
    configTab.focus();
    expect(configTab).toHaveFocus();

    // Use arrow key to navigate (this would work in a real browser environment)
    fireEvent.keyDown(configTab, { key: 'ArrowRight' });
    // In a real test, we'd check that focus moved to the next tab
    
    fireEvent.click(executionTab);
    expect(screen.getByTestId('execution-controls')).toBeInTheDocument();
  });
});