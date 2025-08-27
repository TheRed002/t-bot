/**
 * Bot Management API service
 * Handles all bot-related API calls including CRUD operations and bot lifecycle management
 */

import { api, apiClient } from './client';
import { 
  BotInstance, 
  BotStatus, 
  BotFilters, 
  BotConfiguration, 
  BotMetrics,
  BotSummaryResponse,
  BotListResponse,
  CreateBotRequest as FrontendCreateBotRequest,
  UpdateBotRequest as FrontendUpdateBotRequest,
  ApiResponse,
  StrategyType
} from '@/types';

// Legacy imports for backward compatibility
import type { BotConfig, BotPerformance } from '@/types';

// Backend-specific types (matching backend response models)
interface BackendBotSummaryResponse {
  bot_id: string;
  bot_name: string;
  status: string;
  allocated_capital: number;
  current_pnl?: number;
  total_trades?: number;
  win_rate?: number;
  last_trade?: string;
  uptime?: string;
}

interface BackendBotListResponse {
  bots: BackendBotSummaryResponse[];
  total: number;
  running: number;
  stopped: number;
  error: number;
}

interface BackendBotDetailResponse {
  success: boolean;
  bot: {
    bot_id: string;
    state: {
      status: string;
      configuration: {
        bot_name: string;
        bot_type: string;
        strategy_name: string;
        exchanges: string[];
        symbols: string[];
        allocated_capital: number;
        risk_percentage: number;
        priority: string;
        auto_start: boolean;
        configuration: Record<string, any>;
      };
    };
    metrics?: {
      total_pnl?: number;
      total_trades?: number;
      win_rate?: number;
      last_trade_time?: string;
      max_drawdown?: number;
      sharpe_ratio?: number;
    };
    uptime?: string;
  };
}

interface CreateBotRequest {
  bot_name: string;
  bot_type: string;
  strategy_name: string;
  exchanges: string[];
  symbols: string[];
  allocated_capital: number;
  risk_percentage: number;
  priority?: 'low' | 'normal' | 'high';
  auto_start?: boolean;
  configuration?: Record<string, any>;
}

interface UpdateBotRequest {
  bot_name?: string;
  allocated_capital?: number;
  risk_percentage?: number;
  priority?: 'low' | 'normal' | 'high';
  configuration?: Record<string, any>;
}

// Status mapping from backend to frontend - now using exact backend values
const mapBotStatus = (backendStatus: string): BotStatus => {
  // Backend enum values match frontend enum values exactly
  const statusMap: Record<string, BotStatus> = {
    'initializing': BotStatus.INITIALIZING,
    'ready': BotStatus.READY,
    'running': BotStatus.RUNNING,
    'paused': BotStatus.PAUSED,
    'stopping': BotStatus.STOPPING,
    'stopped': BotStatus.STOPPED,
    'error': BotStatus.ERROR,
    'maintenance': BotStatus.MAINTENANCE
  };
  
  return statusMap[backendStatus.toLowerCase()] || BotStatus.STOPPED;
};

// Transform backend bot summary to frontend format
const transformBotSummary = (backendBot: BackendBotSummaryResponse): BotInstance => ({
  bot_id: backendBot.bot_id,  // Use backend field name
  bot_name: backendBot.bot_name,  // Use backend field name
  userId: '', // Will be set by the caller or from auth context
  strategy_name: StrategyType.CUSTOM, // Will be populated from detail call if needed
  exchange: 'unknown', // Will be populated from detail call if needed
  status: mapBotStatus(backendBot.status),
  config: {
    bot_id: backendBot.bot_id,
    bot_name: backendBot.bot_name,
    bot_type: 'trading' as any, // Default, will be updated from detail
    strategy_name: StrategyType.CUSTOM,
    exchanges: ['unknown'],
    symbols: ['unknown'],
    allocated_capital: backendBot.allocated_capital,
    risk_percentage: 0.01, // Default value
    priority: 'normal' as any,
    auto_start: false,
    strategy_config: {},
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString()
  } as BotConfiguration,
  createdAt: new Date().toISOString(),
  updatedAt: new Date().toISOString(),
  metrics: backendBot.current_pnl !== undefined ? {
    bot_id: backendBot.bot_id,
    uptime_seconds: 0, // Not available in summary
    total_trades: backendBot.total_trades || 0,
    successful_trades: Math.round((backendBot.total_trades || 0) * (backendBot.win_rate || 0) / 100),
    failed_trades: Math.round((backendBot.total_trades || 0) * (1 - (backendBot.win_rate || 0) / 100)),
    total_pnl: backendBot.current_pnl,
    cpu_usage_percent: 0, // Not available in summary
    memory_usage_mb: 0, // Not available in summary
    win_rate: backendBot.win_rate || 0,
    health_score: 100, // Default value
    timestamp: new Date().toISOString()
  } as BotMetrics : undefined
});

// Transform backend bot detail to frontend format
const transformBotDetail = (backendBot: BackendBotDetailResponse['bot']): BotInstance => {
  const config = backendBot.state.configuration;
  const metrics = backendBot.metrics;
  
  return {
    bot_id: backendBot.bot_id,  // Use backend field name
    bot_name: config.bot_name,  // Use backend field name
    userId: '', // Will be set from auth context
    strategy_name: (config.strategy_name as StrategyType) || StrategyType.CUSTOM,  // Use backend field name
    exchange: config.exchanges?.[0] || 'unknown',
    status: mapBotStatus(backendBot.state.status),
    config: {
      bot_id: backendBot.bot_id,
      bot_name: config.bot_name,
      bot_type: config.bot_type as any,
      strategy_name: config.strategy_name,
      exchanges: config.exchanges,
      symbols: config.symbols,
      allocated_capital: config.allocated_capital,
      risk_percentage: config.risk_percentage,
      priority: config.priority as any,
      auto_start: config.auto_start,
      strategy_config: config.configuration || {},
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString()
    } as BotConfiguration,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    metrics: metrics ? {
      bot_id: backendBot.bot_id,
      uptime_seconds: 0, // Calculate if needed
      total_trades: metrics.total_trades || 0,
      successful_trades: Math.round((metrics.total_trades || 0) * (metrics.win_rate || 0) / 100),
      failed_trades: Math.round((metrics.total_trades || 0) * (1 - (metrics.win_rate || 0) / 100)),
      total_pnl: metrics.total_pnl || 0,
      cpu_usage_percent: 0, // Not available in current metrics
      memory_usage_mb: 0, // Not available in current metrics
      win_rate: metrics.win_rate || 0,
      health_score: 100, // Calculate if needed
      timestamp: new Date().toISOString()
    } as BotMetrics : undefined
  };
};

export const botAPI = {
  /**
   * Get list of bots with optional filtering
   * @param filters - Optional filters for bot listing
   * @returns Promise<ApiResponse<BotInstance[]>> - List of bots
   */
  getBots: async (filters?: BotFilters): Promise<ApiResponse<BotInstance[]>> => {
    try {
      // Build query parameters
      const params = new URLSearchParams();
      
      if (filters?.status && filters.status.length > 0) {
        // Backend expects single status filter
        params.append('status_filter', filters.status[0]);
      }
      
      const queryString = params.toString();
      const url = queryString ? `/api/trading/bots/?${queryString}` : '/api/trading/bots/';
      
      const response = await apiClient.get<BotListResponse>(url);
      const backendData = response.data;
      
      // Transform backend response to frontend format
      const transformedBots = backendData.bots.map(transformBotSummary);
      
      return {
        data: transformedBots,
        message: 'Bots retrieved successfully',
        success: true,
        timestamp: new Date().toISOString()
      };
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to fetch bots');
    }
  },

  /**
   * Get detailed information about a specific bot
   * @param botId - Bot identifier
   * @returns Promise<ApiResponse<BotInstance>> - Bot details
   */
  getBotById: async (botId: string): Promise<ApiResponse<BotInstance>> => {
    try {
      const response = await apiClient.get<BackendBotDetailResponse>(`/api/trading/bots/${botId}`);
      
      if (!response.data.success || !response.data.bot) {
        throw new Error('Bot not found');
      }
      
      const transformedBot = transformBotDetail(response.data.bot);
      
      return {
        data: transformedBot,
        message: 'Bot retrieved successfully',
        success: true,
        timestamp: new Date().toISOString()
      };
    } catch (error: any) {
      if (error.response?.status === 404) {
        throw new Error(`Bot not found: ${botId}`);
      }
      throw new Error(error.response?.data?.message || 'Failed to fetch bot details');
    }
  },

  /**
   * Create a new trading bot
   * @param botData - Bot configuration data
   * @returns Promise<ApiResponse<BotInstance>> - Created bot details
   */
  createBot: async (botData: {
    bot_name: string;  // Use backend field name
    strategy_name: string;  // Use backend field name
    exchange: string;
    config: BotConfiguration;  // Use new config type
  }): Promise<ApiResponse<BotInstance>> => {
    try {
      // Transform frontend data to backend format
      const createRequest: CreateBotRequest = {
        bot_name: botData.bot_name,
        bot_type: botData.config.bot_type,
        strategy_name: botData.strategy_name,
        exchanges: botData.config.exchanges,
        symbols: botData.config.symbols,
        allocated_capital: botData.config.allocated_capital,
        risk_percentage: botData.config.risk_percentage,
        priority: botData.config.priority.toLowerCase() as 'low' | 'normal' | 'high',
        auto_start: botData.config.auto_start,
        configuration: botData.config.strategy_config || {}
      };
      
      const response = await apiClient.post(`/api/trading/bots/`, createRequest);
      
      if (!response.data.success) {
        throw new Error(response.data.message || 'Failed to create bot');
      }
      
      // Get the created bot details
      const createdBotId = response.data.bot_id;
      const botDetails = await botAPI.getBotById(createdBotId);
      
      return {
        data: botDetails.data,
        message: 'Bot created successfully',
        success: true,
        timestamp: new Date().toISOString()
      };
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to create bot');
    }
  },

  /**
   * Update bot configuration
   * @param botId - Bot identifier
   * @param updates - Partial bot updates
   * @returns Promise<ApiResponse<BotInstance>> - Updated bot details
   */
  updateBot: async (botId: string, updates: Partial<BotInstance>): Promise<ApiResponse<BotInstance>> => {
    try {
      // Transform frontend updates to backend format
      const updateRequest: UpdateBotRequest = {};
      
      if (updates.bot_name !== undefined) {
        updateRequest.bot_name = updates.bot_name;
      }
      
      if (updates.config?.allocated_capital !== undefined) {
        updateRequest.allocated_capital = updates.config.allocated_capital;
      }
      
      if (updates.config?.risk_percentage !== undefined) {
        updateRequest.risk_percentage = updates.config.risk_percentage;
      }
      
      if (updates.config?.strategy_config !== undefined) {
        updateRequest.configuration = updates.config.strategy_config;
      }
      
      const response = await apiClient.put(`/api/trading/bots/${botId}`, updateRequest);
      
      if (!response.data.success) {
        throw new Error(response.data.message || 'Failed to update bot');
      }
      
      // Get updated bot details
      const botDetails = await botAPI.getBotById(botId);
      
      return {
        data: botDetails.data,
        message: 'Bot updated successfully',
        success: true,
        timestamp: new Date().toISOString()
      };
    } catch (error: any) {
      if (error.response?.status === 404) {
        throw new Error(`Bot not found: ${botId}`);
      }
      throw new Error(error.response?.data?.message || 'Failed to update bot');
    }
  },

  /**
   * Delete a trading bot
   * @param botId - Bot identifier
   * @returns Promise<ApiResponse<void>> - Deletion result
   */
  deleteBot: async (botId: string): Promise<ApiResponse<void>> => {
    try {
      const response = await apiClient.delete(`/api/trading/bots/${botId}?force=false`);
      
      if (!response.data.success) {
        throw new Error(response.data.message || 'Failed to delete bot');
      }
      
      return {
        data: undefined,
        message: 'Bot deleted successfully',
        success: true,
        timestamp: new Date().toISOString()
      };
    } catch (error: any) {
      if (error.response?.status === 404) {
        throw new Error(`Bot not found: ${botId}`);
      }
      if (error.response?.status === 409) {
        throw new Error('Cannot delete running bot. Stop the bot first.');
      }
      throw new Error(error.response?.data?.message || 'Failed to delete bot');
    }
  },

  /**
   * Start a trading bot
   * @param botId - Bot identifier
   * @returns Promise<ApiResponse<void>> - Start operation result
   */
  startBot: async (botId: string): Promise<ApiResponse<void>> => {
    try {
      const response = await apiClient.post(`/api/trading/bots/${botId}/start`);
      
      if (!response.data.success) {
        throw new Error(response.data.message || 'Failed to start bot');
      }
      
      return {
        data: undefined,
        message: 'Bot started successfully',
        success: true,
        timestamp: new Date().toISOString()
      };
    } catch (error: any) {
      if (error.response?.status === 404) {
        throw new Error(`Bot not found: ${botId}`);
      }
      if (error.response?.status === 409) {
        throw new Error('Bot is already running');
      }
      throw new Error(error.response?.data?.message || 'Failed to start bot');
    }
  },

  /**
   * Stop a trading bot
   * @param botId - Bot identifier
   * @returns Promise<ApiResponse<void>> - Stop operation result
   */
  stopBot: async (botId: string): Promise<ApiResponse<void>> => {
    try {
      const response = await apiClient.post(`/api/trading/bots/${botId}/stop`);
      
      if (!response.data.success) {
        throw new Error(response.data.message || 'Failed to stop bot');
      }
      
      return {
        data: undefined,
        message: 'Bot stopped successfully',
        success: true,
        timestamp: new Date().toISOString()
      };
    } catch (error: any) {
      if (error.response?.status === 404) {
        throw new Error(`Bot not found: ${botId}`);
      }
      if (error.response?.status === 409) {
        throw new Error('Bot is not running');
      }
      throw new Error(error.response?.data?.message || 'Failed to stop bot');
    }
  },

  /**
   * Pause a trading bot
   * @param botId - Bot identifier
   * @returns Promise<ApiResponse<void>> - Pause operation result
   */
  pauseBot: async (botId: string): Promise<ApiResponse<void>> => {
    try {
      const response = await apiClient.post(`/api/trading/bots/${botId}/pause`);
      
      if (!response.data.success) {
        throw new Error(response.data.message || 'Failed to pause bot');
      }
      
      return {
        data: undefined,
        message: 'Bot paused successfully',
        success: true,
        timestamp: new Date().toISOString()
      };
    } catch (error: any) {
      if (error.response?.status === 404) {
        throw new Error(`Bot not found: ${botId}`);
      }
      if (error.response?.status === 409) {
        throw new Error('Cannot pause - bot is not running');
      }
      throw new Error(error.response?.data?.message || 'Failed to pause bot');
    }
  },

  /**
   * Resume a paused trading bot
   * @param botId - Bot identifier
   * @returns Promise<ApiResponse<void>> - Resume operation result
   */
  resumeBot: async (botId: string): Promise<ApiResponse<void>> => {
    try {
      const response = await apiClient.post(`/api/trading/bots/${botId}/resume`);
      
      if (!response.data.success) {
        throw new Error(response.data.message || 'Failed to resume bot');
      }
      
      return {
        data: undefined,
        message: 'Bot resumed successfully',
        success: true,
        timestamp: new Date().toISOString()
      };
    } catch (error: any) {
      if (error.response?.status === 404) {
        throw new Error(`Bot not found: ${botId}`);
      }
      if (error.response?.status === 409) {
        throw new Error('Cannot resume - bot is not paused');
      }
      throw new Error(error.response?.data?.message || 'Failed to resume bot');
    }
  },

  /**
   * Get bot performance metrics
   * @param botId - Bot identifier
   * @returns Promise<ApiResponse<BotPerformance>> - Bot performance data
   */
  getBotPerformance: async (botId: string): Promise<ApiResponse<BotPerformance>> => {
    try {
      // Get bot details which includes performance metrics
      const botDetails = await botAPI.getBotById(botId);
      
      if (!botDetails.data.metrics) {
        // Return empty performance metrics if none available
        const emptyPerformance: BotPerformance = {
          totalTrades: 0,
          winningTrades: 0,
          losingTrades: 0,
          totalPnl: 0,
          winRate: 0,
          maxDrawdown: 0,
          startDate: new Date().toISOString()
        };
        
        return {
          data: emptyPerformance,
          message: 'No performance data available',
          success: true,
          timestamp: new Date().toISOString()
        };
      }
      
      // Convert BotMetrics to legacy BotPerformance format for backward compatibility
      const performanceData: BotPerformance = {
        totalTrades: botDetails.data.metrics.total_trades,
        winningTrades: botDetails.data.metrics.successful_trades,
        losingTrades: botDetails.data.metrics.failed_trades,
        totalPnl: botDetails.data.metrics.total_pnl,
        winRate: botDetails.data.metrics.win_rate,
        maxDrawdown: 0, // Not available in current metrics
        sharpeRatio: undefined,
        startDate: new Date().toISOString(),
        lastTradeAt: undefined
      };
      
      return {
        data: performanceData,
        message: 'Bot performance retrieved successfully',
        success: true,
        timestamp: new Date().toISOString()
      };
    } catch (error: any) {
      if (error.response?.status === 404) {
        throw new Error(`Bot not found: ${botId}`);
      }
      throw new Error(error.response?.data?.message || 'Failed to fetch bot performance');
    }
  },

  /**
   * Get orchestrator status and bot statistics
   * @returns Promise<ApiResponse<any>> - Orchestrator status
   */
  getOrchestratorStatus: async (): Promise<ApiResponse<any>> => {
    try {
      const response = await apiClient.get('/api/trading/bots/orchestrator/status');
      
      return {
        data: response.data.status,
        message: 'Orchestrator status retrieved successfully',
        success: true,
        timestamp: new Date().toISOString()
      };
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to fetch orchestrator status');
    }
  }
};

export default botAPI;