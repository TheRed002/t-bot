/**
 * Strategy Template API Service
 * Handles all strategy template operations including fetching templates,
 * validation, configuration, and deployment
 */

import { api } from './client';
import { 
  Strategy, 
  BacktestResult,
  StrategyTemplate,
  StrategyTemplateListResponse,
  StrategyTemplateResponse,
  TemplateValidationResponse,
  StrategyDeploymentRequest,
  StrategyDeploymentResponse,
  StrategyType,
  StrategyCategory,
  RiskLevel,
  StrategyConfiguration
} from '@/types';

/**
 * Strategy Template API endpoints
 */
const TEMPLATE_ENDPOINTS = {
  LIST_TEMPLATES: '/strategies/templates',
  GET_TEMPLATE: (templateId: string) => `/strategies/templates/${templateId}`,
  VALIDATE_CONFIG: '/strategies/templates/validate',
  DEPLOY_STRATEGY: '/strategies/deploy',
  LIST_CATEGORIES: '/strategies/categories',
  SEARCH_TEMPLATES: '/strategies/templates/search'
} as const;

/**
 * Template list filter parameters
 */
export interface TemplateFilters {
  category?: StrategyCategory[];
  strategyType?: StrategyType[];
  riskLevel?: RiskLevel[];
  exchanges?: string[];
  minCapital?: number;
  maxCapital?: number;
  tags?: string[];
  search?: string;
  sortBy?: 'name' | 'risk_level' | 'created_at' | 'performance';
  sortOrder?: 'asc' | 'desc';
  limit?: number;
  offset?: number;
}

/**
 * Template search parameters
 */
export interface TemplateSearchRequest {
  query: string;
  filters?: TemplateFilters;
  includeRelated?: boolean;
  maxResults?: number;
}

export const strategyAPI = {
  // Legacy strategy endpoints
  // Get all strategies
  getStrategies: async () => {
    return api.get<Strategy[]>('/strategies');
  },

  // Get strategy by ID
  getStrategyById: async (strategyId: string) => {
    return api.get<Strategy>(`/strategies/${strategyId}`);
  },

  // Run backtest
  runBacktest: async (strategyId: string, params: any) => {
    return api.post<BacktestResult>(`/strategies/${strategyId}/backtest`, params);
  },

  // Get backtest results
  getBacktestResults: async (strategyId: string) => {
    return api.get<BacktestResult[]>(`/strategies/${strategyId}/backtests`);
  },

  // Create custom strategy
  createStrategy: async (strategyData: Partial<Strategy>) => {
    return api.post<Strategy>('/strategies', strategyData);
  },

  // Update strategy
  updateStrategy: async (strategyId: string, updates: Partial<Strategy>) => {
    return api.patch<Strategy>(`/strategies/${strategyId}`, updates);
  },

  // New Template-based API endpoints
  
  /**
   * Get list of all strategy templates with optional filtering
   */
  getTemplates: async (filters?: TemplateFilters): Promise<StrategyTemplateListResponse> => {
    const params = new URLSearchParams();
    
    if (filters) {
      // Handle array filters
      if (filters.category?.length) {
        filters.category.forEach(cat => params.append('category', cat));
      }
      if (filters.strategyType?.length) {
        filters.strategyType.forEach(type => params.append('strategy_type', type));
      }
      if (filters.riskLevel?.length) {
        filters.riskLevel.forEach(level => params.append('risk_level', level));
      }
      if (filters.exchanges?.length) {
        filters.exchanges.forEach(exchange => params.append('exchanges', exchange));
      }
      if (filters.tags?.length) {
        filters.tags.forEach(tag => params.append('tags', tag));
      }
      
      // Handle scalar filters
      if (filters.minCapital) params.set('min_capital', filters.minCapital.toString());
      if (filters.maxCapital) params.set('max_capital', filters.maxCapital.toString());
      if (filters.search) params.set('search', filters.search);
      if (filters.sortBy) params.set('sort_by', filters.sortBy);
      if (filters.sortOrder) params.set('sort_order', filters.sortOrder);
      if (filters.limit) params.set('limit', filters.limit.toString());
      if (filters.offset) params.set('offset', filters.offset.toString());
    }
    
    const url = `${TEMPLATE_ENDPOINTS.LIST_TEMPLATES}${params.toString() ? '?' + params.toString() : ''}`;
    const response = await api.get<StrategyTemplateListResponse>(url);
    return response.data;
  },

  /**
   * Get a specific strategy template by ID
   */
  getTemplate: async (templateId: string, includeRelated = false): Promise<StrategyTemplateResponse> => {
    const params = includeRelated ? '?include_related=true' : '';
    const response = await api.get<StrategyTemplateResponse>(
      `${TEMPLATE_ENDPOINTS.GET_TEMPLATE(templateId)}${params}`
    );
    return response.data;
  },

  /**
   * Get strategy categories with metadata
   */
  getCategories: async (): Promise<{
    categories: Array<{
      category: StrategyCategory;
      name: string;
      description: string;
      templateCount: number;
      riskLevels: RiskLevel[];
    }>;
  }> => {
    const response = await api.get<{
      categories: Array<{
        category: StrategyCategory;
        name: string;
        description: string;
        templateCount: number;
        riskLevels: RiskLevel[];
      }>;
    }>(TEMPLATE_ENDPOINTS.LIST_CATEGORIES);
    return response.data;
  },

  /**
   * Search strategy templates with fuzzy matching
   */
  searchTemplates: async (searchRequest: TemplateSearchRequest): Promise<{
    templates: StrategyTemplate[];
    suggestions: string[];
    totalResults: number;
  }> => {
    const response = await api.post<{
      templates: StrategyTemplate[];
      suggestions: string[];
      totalResults: number;
    }>(TEMPLATE_ENDPOINTS.SEARCH_TEMPLATES, searchRequest);
    return response.data;
  },

  /**
   * Validate strategy template configuration
   */
  validateConfiguration: async (
    templateId: string,
    configuration: {
      exchanges: string[];
      symbols: string[];
      allocatedCapital: number;
      parameters: Record<string, any>;
    }
  ): Promise<TemplateValidationResponse> => {
    const response = await api.post<TemplateValidationResponse>(
      TEMPLATE_ENDPOINTS.VALIDATE_CONFIG,
      {
        templateId,
        configuration
      }
    );
    return response.data;
  },

  /**
   * Deploy a strategy from template
   */
  deployStrategy: async (request: StrategyDeploymentRequest): Promise<StrategyDeploymentResponse> => {
    const response = await api.post<StrategyDeploymentResponse>(
      TEMPLATE_ENDPOINTS.DEPLOY_STRATEGY,
      request
    );
    return response.data;
  },

  /**
   * Get template parameter schema for dynamic form generation
   */
  getParameterSchema: async (templateId: string): Promise<{
    schema: {
      parameters: Array<{
        name: string;
        type: string;
        validation: Record<string, any>;
        uiHints: Record<string, any>;
      }>;
      groups: Array<{
        name: string;
        displayName: string;
        parameters: string[];
      }>;
    };
  }> => {
    const response = await api.get<{
      schema: {
        parameters: Array<{
          name: string;
          type: string;
          validation: Record<string, any>;
          uiHints: Record<string, any>;
        }>;
        groups: Array<{
          name: string;
          displayName: string;
          parameters: string[];
        }>;
      };
    }>(`${TEMPLATE_ENDPOINTS.GET_TEMPLATE(templateId)}/schema`);
    return response.data;
  },

  /**
   * Get template performance benchmarks
   */
  getTemplateBenchmarks: async (templateId: string, period?: string): Promise<{
    benchmarks: {
      backtestResults: Array<{
        period: string;
        totalReturn: number;
        sharpeRatio: number;
        maxDrawdown: number;
        winRate: number;
        volatility: number;
      }>;
      livePerformance?: {
        totalReturn: number;
        sharpeRatio: number;
        maxDrawdown: number;
        winRate: number;
        avgDailyReturn: number;
      };
      comparison: {
        benchmark: string;
        alpha: number;
        beta: number;
        correlation: number;
      };
    };
  }> => {
    const params = period ? `?period=${period}` : '';
    const response = await api.get<{
      benchmarks: {
        backtestResults: Array<{
          period: string;
          totalReturn: number;
          sharpeRatio: number;
          maxDrawdown: number;
          winRate: number;
          volatility: number;
        }>;
        livePerformance?: {
          totalReturn: number;
          sharpeRatio: number;
          maxDrawdown: number;
          winRate: number;
          avgDailyReturn: number;
        };
        comparison: {
          benchmark: string;
          alpha: number;
          beta: number;
          correlation: number;
        };
      };
    }>(`${TEMPLATE_ENDPOINTS.GET_TEMPLATE(templateId)}/benchmarks${params}`);
    return response.data;
  },

  /**
   * Clone and customize a strategy template
   */
  cloneTemplate: async (
    templateId: string,
    customization: {
      name: string;
      description?: string;
      parameters?: Record<string, any>;
      riskConfiguration?: Partial<{
        maxDrawdownPercentage: number;
        maxRiskPerTrade: number;
        positionSizeMethod: string;
      }>;
    }
  ): Promise<{ templateId: string; template: StrategyTemplate }> => {
    const response = await api.post<{ templateId: string; template: StrategyTemplate }>(
      `${TEMPLATE_ENDPOINTS.GET_TEMPLATE(templateId)}/clone`,
      customization
    );
    return response.data;
  },

  /**
   * Get template usage statistics
   */
  getTemplateStats: async (templateId: string): Promise<{
    statistics: {
      totalDeployments: number;
      activeStrategies: number;
      averagePerformance: {
        totalReturn: number;
        sharpeRatio: number;
        maxDrawdown: number;
        winRate: number;
      };
      usageByExchange: Record<string, number>;
      usageByTimeframe: Record<string, number>;
      ratingDistribution: Record<string, number>;
    };
  }> => {
    const response = await api.get<{
      statistics: {
        totalDeployments: number;
        activeStrategies: number;
        averagePerformance: {
          totalReturn: number;
          sharpeRatio: number;
          maxDrawdown: number;
          winRate: number;
        };
        usageByExchange: Record<string, number>;
        usageByTimeframe: Record<string, number>;
        ratingDistribution: Record<string, number>;
      };
    }>(`${TEMPLATE_ENDPOINTS.GET_TEMPLATE(templateId)}/stats`);
    return response.data;
  }
};

/**
 * Strategy template cache for offline usage
 */
class StrategyTemplateCache {
  private cache = new Map<string, { data: any; timestamp: number; ttl: number }>();
  private readonly DEFAULT_TTL = 5 * 60 * 1000; // 5 minutes

  set<T>(key: string, data: T, ttl = this.DEFAULT_TTL): void {
    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      ttl
    });
  }

  get<T>(key: string): T | null {
    const entry = this.cache.get(key);
    if (!entry) return null;

    if (Date.now() - entry.timestamp > entry.ttl) {
      this.cache.delete(key);
      return null;
    }

    return entry.data as T;
  }

  clear(): void {
    this.cache.clear();
  }

  has(key: string): boolean {
    return this.cache.has(key) && Date.now() - this.cache.get(key)!.timestamp <= this.cache.get(key)!.ttl;
  }
}

export const templateCache = new StrategyTemplateCache();

/**
 * Cached strategy API with automatic cache management
 */
export const cachedStrategyAPI = {
  /**
   * Get templates with caching
   */
  getTemplates: async (filters?: TemplateFilters, useCache = true): Promise<StrategyTemplateListResponse> => {
    const cacheKey = `templates:${JSON.stringify(filters || {})}`;
    
    if (useCache) {
      const cached = templateCache.get<StrategyTemplateListResponse>(cacheKey);
      if (cached) return cached;
    }

    const data = await strategyAPI.getTemplates(filters);
    templateCache.set(cacheKey, data);
    return data;
  },

  /**
   * Get single template with caching
   */
  getTemplate: async (templateId: string, includeRelated = false, useCache = true): Promise<StrategyTemplateResponse> => {
    const cacheKey = `template:${templateId}:${includeRelated}`;
    
    if (useCache) {
      const cached = templateCache.get<StrategyTemplateResponse>(cacheKey);
      if (cached) return cached;
    }

    const data = await strategyAPI.getTemplate(templateId, includeRelated);
    templateCache.set(cacheKey, data);
    return data;
  },

  /**
   * Clear all cached data
   */
  clearCache: (): void => {
    templateCache.clear();
  },

  /**
   * Pre-warm cache with popular templates
   */
  warmCache: async (): Promise<void> => {
    try {
      // Get popular templates (conservative and moderate risk levels)
      await Promise.all([
        cachedStrategyAPI.getTemplates({ 
          riskLevel: [RiskLevel.CONSERVATIVE, RiskLevel.MODERATE],
          limit: 20 
        }, false),
        cachedStrategyAPI.getTemplates({ 
          category: [StrategyCategory.STATIC, StrategyCategory.DYNAMIC],
          limit: 15 
        }, false)
      ]);
    } catch (error) {
      console.warn('Failed to warm strategy template cache:', error);
    }
  }
};

// Export both cached and uncached versions
export default cachedStrategyAPI;