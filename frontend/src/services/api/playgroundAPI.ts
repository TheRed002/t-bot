/**
 * Playground API Service
 * Handles all API calls related to playground functionality
 */

import { apiClient } from './client';
import {
  PlaygroundConfiguration,
  PlaygroundExecution,
  PlaygroundBatch,
  ABTest,
  PlaygroundPreset,
  PlaygroundTrade,
  PlaygroundLog,
  ApiResponse,
  PaginatedResponse,
  DateRange
} from '@/types';

export const playgroundAPI = {
  // Configuration management
  getConfigurations: async (params?: {
    page?: number;
    limit?: number;
    search?: string;
  }): Promise<PaginatedResponse<PlaygroundConfiguration>> => {
    const response = await apiClient.get('/playground/configurations', { params });
    return response.data;
  },

  getConfiguration: async (configurationId: string): Promise<ApiResponse<PlaygroundConfiguration>> => {
    const response = await apiClient.get(`/playground/configurations/${configurationId}`);
    return response.data;
  },

  saveConfiguration: async (configuration: PlaygroundConfiguration): Promise<ApiResponse<PlaygroundConfiguration>> => {
    if (configuration.id) {
      const response = await apiClient.put(`/playground/configurations/${configuration.id}`, configuration);
      return response.data;
    } else {
      const response = await apiClient.post('/playground/configurations', configuration);
      return response.data;
    }
  },

  deleteConfiguration: async (configurationId: string): Promise<ApiResponse<void>> => {
    const response = await apiClient.delete(`/playground/configurations/${configurationId}`);
    return response.data;
  },

  cloneConfiguration: async (configurationId: string, newName: string): Promise<ApiResponse<PlaygroundConfiguration>> => {
    const response = await apiClient.post(`/playground/configurations/${configurationId}/clone`, { name: newName });
    return response.data;
  },

  // Execution management
  getExecutions: async (filters?: {
    status?: string[];
    mode?: string[];
    dateRange?: DateRange;
    page?: number;
    limit?: number;
  }): Promise<PaginatedResponse<PlaygroundExecution>> => {
    const response = await apiClient.get('/playground/executions', { params: filters });
    return response.data;
  },

  getExecution: async (executionId: string): Promise<ApiResponse<PlaygroundExecution>> => {
    const response = await apiClient.get(`/playground/executions/${executionId}`);
    return response.data;
  },

  startExecution: async (execution: Omit<PlaygroundExecution, 'id'>): Promise<ApiResponse<PlaygroundExecution>> => {
    const response = await apiClient.post('/playground/executions', execution);
    return response.data;
  },

  controlExecution: async (
    executionId: string,
    action: 'pause' | 'resume' | 'stop' | 'restart'
  ): Promise<ApiResponse<PlaygroundExecution>> => {
    const response = await apiClient.post(`/playground/executions/${executionId}/control`, { action });
    return response.data;
  },

  deleteExecution: async (executionId: string): Promise<ApiResponse<void>> => {
    const response = await apiClient.delete(`/playground/executions/${executionId}`);
    return response.data;
  },

  // Execution monitoring
  getExecutionLogs: async (
    executionId: string,
    params?: {
      level?: string;
      category?: string;
      search?: string;
      limit?: number;
      offset?: number;
    }
  ): Promise<ApiResponse<PlaygroundLog[]>> => {
    const response = await apiClient.get(`/playground/executions/${executionId}/logs`, { params });
    return response.data;
  },

  getExecutionTrades: async (
    executionId: string,
    params?: {
      limit?: number;
      offset?: number;
    }
  ): Promise<ApiResponse<PlaygroundTrade[]>> => {
    const response = await apiClient.get(`/playground/executions/${executionId}/trades`, { params });
    return response.data;
  },

  getExecutionMetrics: async (executionId: string): Promise<ApiResponse<any>> => {
    const response = await apiClient.get(`/playground/executions/${executionId}/metrics`);
    return response.data;
  },

  downloadExecutionResults: async (executionId: string, format: 'csv' | 'json' | 'excel'): Promise<Blob> => {
    const response = await apiClient.get(`/playground/executions/${executionId}/download`, {
      params: { format },
      responseType: 'blob'
    });
    return response.data;
  },

  // Backtesting specific
  runBacktest: async (config: {
    configurationId: string;
    startDate: string;
    endDate: string;
    initialBalance: number;
    commission: number;
    speed?: number;
  }): Promise<ApiResponse<PlaygroundExecution>> => {
    const response = await apiClient.post('/playground/backtest', config);
    return response.data;
  },

  getBacktestProgress: async (executionId: string): Promise<ApiResponse<{
    progress: number;
    stage: string;
    estimatedTimeRemaining?: number;
  }>> => {
    const response = await apiClient.get(`/playground/backtest/${executionId}/progress`);
    return response.data;
  },

  // A/B Testing
  getABTests: async (params?: {
    status?: string;
    page?: number;
    limit?: number;
  }): Promise<PaginatedResponse<ABTest>> => {
    const response = await apiClient.get('/playground/ab-tests', { params });
    return response.data;
  },

  getABTest: async (abTestId: string): Promise<ApiResponse<ABTest>> => {
    const response = await apiClient.get(`/playground/ab-tests/${abTestId}`);
    return response.data;
  },

  createABTest: async (abTest: Omit<ABTest, 'id' | 'createdAt'>): Promise<ApiResponse<ABTest>> => {
    const response = await apiClient.post('/playground/ab-tests', abTest);
    return response.data;
  },

  runABTest: async (abTestId: string): Promise<ApiResponse<ABTest>> => {
    const response = await apiClient.post(`/playground/ab-tests/${abTestId}/run`);
    return response.data;
  },

  deleteABTest: async (abTestId: string): Promise<ApiResponse<void>> => {
    const response = await apiClient.delete(`/playground/ab-tests/${abTestId}`);
    return response.data;
  },

  getABTestResults: async (abTestId: string): Promise<ApiResponse<any>> => {
    const response = await apiClient.get(`/playground/ab-tests/${abTestId}/results`);
    return response.data;
  },

  // Batch optimization
  getBatches: async (params?: {
    status?: string;
    page?: number;
    limit?: number;
  }): Promise<PaginatedResponse<PlaygroundBatch>> => {
    const response = await apiClient.get('/playground/batches', { params });
    return response.data;
  },

  getBatch: async (batchId: string): Promise<ApiResponse<PlaygroundBatch>> => {
    const response = await apiClient.get(`/playground/batches/${batchId}`);
    return response.data;
  },

  startBatchOptimization: async (batch: Omit<PlaygroundBatch, 'id' | 'startTime'>): Promise<ApiResponse<PlaygroundBatch>> => {
    const response = await apiClient.post('/playground/batches', batch);
    return response.data;
  },

  getBatchProgress: async (batchId: string): Promise<ApiResponse<{
    completedConfigurations: number;
    totalConfigurations: number;
    progress: number;
    currentStage: string;
    estimatedTimeRemaining?: number;
  }>> => {
    const response = await apiClient.get(`/playground/batches/${batchId}/progress`);
    return response.data;
  },

  getBatchResults: async (batchId: string): Promise<ApiResponse<{
    results: Array<{
      configurationId: string;
      metrics: any;
      rank: number;
      parameters: Record<string, any>;
      overfittingRisk: string;
    }>;
    bestConfiguration: PlaygroundConfiguration;
    summary: {
      totalConfigurations: number;
      successfulRuns: number;
      failedRuns: number;
      avgReturn: number;
      avgSharpe: number;
      bestSharpe: number;
    };
  }>> => {
    const response = await apiClient.get(`/playground/batches/${batchId}/results`);
    return response.data;
  },

  stopBatch: async (batchId: string): Promise<ApiResponse<PlaygroundBatch>> => {
    const response = await apiClient.post(`/playground/batches/${batchId}/stop`);
    return response.data;
  },

  deleteBatch: async (batchId: string): Promise<ApiResponse<void>> => {
    const response = await apiClient.delete(`/playground/batches/${batchId}`);
    return response.data;
  },

  // Parameter optimization
  optimizeParameters: async (config: {
    configurationId: string;
    parameters: Array<{
      name: string;
      type: 'range' | 'discrete' | 'categorical';
      min?: number;
      max?: number;
      step?: number;
      values?: any[];
    }>;
    optimizationMetric: 'sharpe_ratio' | 'return' | 'calmar_ratio' | 'sortino_ratio';
    maxIterations: number;
    crossValidationFolds?: number;
  }): Promise<ApiResponse<{
    optimizationId: string;
    status: string;
  }>> => {
    const response = await apiClient.post('/playground/optimize', config);
    return response.data;
  },

  getOptimizationProgress: async (optimizationId: string): Promise<ApiResponse<{
    progress: number;
    currentIteration: number;
    totalIterations: number;
    bestResult?: any;
    estimatedTimeRemaining?: number;
  }>> => {
    const response = await apiClient.get(`/playground/optimize/${optimizationId}/progress`);
    return response.data;
  },

  getOptimizationResults: async (optimizationId: string): Promise<ApiResponse<{
    optimalParameters: Record<string, any>;
    performance: any;
    sensitivity: Record<string, number>;
    iterations: Array<{
      parameters: Record<string, any>;
      metrics: any;
      rank: number;
    }>;
  }>> => {
    const response = await apiClient.get(`/playground/optimize/${optimizationId}/results`);
    return response.data;
  },

  // Presets and templates
  getPresets: async (category?: string): Promise<ApiResponse<PlaygroundPreset[]>> => {
    const response = await apiClient.get('/playground/presets', { params: { category } });
    return response.data;
  },

  getPreset: async (presetId: string): Promise<ApiResponse<PlaygroundPreset>> => {
    const response = await apiClient.get(`/playground/presets/${presetId}`);
    return response.data;
  },

  saveAsPreset: async (configurationId: string, presetData: {
    name: string;
    description: string;
    category: string;
  }): Promise<ApiResponse<PlaygroundPreset>> => {
    const response = await apiClient.post(`/playground/configurations/${configurationId}/save-as-preset`, presetData);
    return response.data;
  },

  // Multi-instance execution
  startMultiInstance: async (config: {
    configurations: PlaygroundConfiguration[];
    parallelExecution: boolean;
    resourceLimit: number;
    maxConcurrentInstances: number;
  }): Promise<ApiResponse<{
    batchId: string;
    instances: PlaygroundExecution[];
  }>> => {
    const response = await apiClient.post('/playground/multi-instance', config);
    return response.data;
  },

  getMultiInstanceStatus: async (batchId: string): Promise<ApiResponse<{
    instances: PlaygroundExecution[];
    overallProgress: number;
    resourceUsage: {
      cpu: number;
      memory: number;
      activeInstances: number;
    };
  }>> => {
    const response = await apiClient.get(`/playground/multi-instance/${batchId}/status`);
    return response.data;
  },

  // Walk-forward analysis
  runWalkForwardAnalysis: async (config: {
    configurationId: string;
    windowSize: number; // in days
    stepSize: number; // in days
    minTrainSize: number; // minimum training window size
    maxLookback: number; // maximum lookback period
  }): Promise<ApiResponse<{
    analysisId: string;
    status: string;
  }>> => {
    const response = await apiClient.post('/playground/walk-forward', config);
    return response.data;
  },

  getWalkForwardResults: async (analysisId: string): Promise<ApiResponse<{
    windows: Array<{
      trainStart: string;
      trainEnd: string;
      testStart: string;
      testEnd: string;
      metrics: any;
    }>;
    summary: {
      avgReturn: number;
      avgSharpe: number;
      consistency: number;
      degradation: number;
    };
    recommendations: string[];
  }>> => {
    const response = await apiClient.get(`/playground/walk-forward/${analysisId}/results`);
    return response.data;
  },

  // Risk analysis
  analyzeRisk: async (executionId: string): Promise<ApiResponse<{
    var95: number;
    var99: number;
    cvar95: number;
    cvar99: number;
    maxDrawdown: number;
    calmarRatio: number;
    sterlingRatio: number;
    ulcerIndex: number;
    painIndex: number;
    stressTest: {
      scenarios: Array<{
        name: string;
        description: string;
        impact: number;
        probability: number;
      }>;
    };
  }>> => {
    const response = await apiClient.post(`/playground/executions/${executionId}/risk-analysis`);
    return response.data;
  },

  // Performance attribution
  getPerformanceAttribution: async (executionId: string): Promise<ApiResponse<{
    bySymbol: Record<string, {
      contribution: number;
      trades: number;
      winRate: number;
      avgReturn: number;
    }>;
    byStrategy: Record<string, {
      contribution: number;
      allocation: number;
      performance: number;
    }>;
    byTimeframe: Array<{
      period: string;
      return: number;
      contribution: number;
      benchmark: number;
    }>;
  }>> => {
    const response = await apiClient.get(`/playground/executions/${executionId}/attribution`);
    return response.data;
  },

  // Market regime analysis
  analyzeMarketRegimes: async (config: {
    symbols: string[];
    startDate: string;
    endDate: string;
    regimeIndicators: string[];
  }): Promise<ApiResponse<{
    regimes: Array<{
      start: string;
      end: string;
      type: string;
      characteristics: Record<string, number>;
    }>;
    performance: Record<string, {
      regime: string;
      return: number;
      volatility: number;
      sharpe: number;
    }>;
  }>> => {
    const response = await apiClient.post('/playground/market-regimes', config);
    return response.data;
  },

  // Real-time monitoring
  subscribeToExecution: async (executionId: string): Promise<EventSource> => {
    // Return EventSource for real-time updates
    return new EventSource(`/api/playground/executions/${executionId}/stream`);
  },

  // Export and reporting
  generateReport: async (executionId: string, format: 'pdf' | 'html' | 'excel'): Promise<Blob> => {
    const response = await apiClient.get(`/playground/executions/${executionId}/report`, {
      params: { format },
      responseType: 'blob'
    });
    return response.data;
  },

  compareExecutions: async (executionIds: string[]): Promise<ApiResponse<{
    comparison: Array<{
      executionId: string;
      metrics: any;
      configuration: PlaygroundConfiguration;
    }>;
    analysis: {
      bestPerformer: string;
      keyDifferences: string[];
      recommendations: string[];
    };
  }>> => {
    const response = await apiClient.post('/playground/compare', { executionIds });
    return response.data;
  }
};

export default playgroundAPI;