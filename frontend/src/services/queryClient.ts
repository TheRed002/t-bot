/**
 * React Query client configuration
 * Optimized for financial data with appropriate cache settings
 */

import { QueryClient, QueryCache, MutationCache } from '@tanstack/react-query';

// Custom error handler
const handleError = (error: any) => {
  console.error('Query error:', error);
  
  // Handle specific error types
  if (error?.response?.status === 401) {
    // Redirect to login on auth errors
    window.location.href = '/login';
  }
};

// Create query client with optimized settings for trading data
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Stale time for different data types
      staleTime: 5 * 1000, // 5 seconds default
      gcTime: 10 * 60 * 1000, // 10 minutes cache
      
      // Retry configuration
      retry: (failureCount, error: any) => {
        // Don't retry on auth errors
        if (error?.response?.status === 401 || error?.response?.status === 403) {
          return false;
        }
        
        // Retry up to 3 times for other errors
        return failureCount < 3;
      },
      
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
      
      // Refetch configuration
      refetchOnMount: true,
      refetchOnWindowFocus: false,
      refetchOnReconnect: true,
    },
    mutations: {
      // Retry configuration for mutations
      retry: false, // Don't retry mutations by default
    },
  },
  
  queryCache: new QueryCache({
    onError: handleError,
  }),
  
  mutationCache: new MutationCache({
    onError: handleError,
  }),
});

// Query keys for consistent caching
export const queryKeys = {
  // Authentication
  auth: ['auth'] as const,
  profile: () => [...queryKeys.auth, 'profile'] as const,
  
  // Bots
  bots: ['bots'] as const,
  botsList: (filters?: any) => [...queryKeys.bots, 'list', filters] as const,
  bot: (id: string) => [...queryKeys.bots, 'detail', id] as const,
  botPerformance: (id: string) => [...queryKeys.bots, 'performance', id] as const,
  botLogs: (id: string) => [...queryKeys.bots, 'logs', id] as const,
  
  // Portfolio
  portfolio: ['portfolio'] as const,
  portfolioSummary: () => [...queryKeys.portfolio, 'summary'] as const,
  positions: (filters?: any) => [...queryKeys.portfolio, 'positions', filters] as const,
  balances: (filters?: any) => [...queryKeys.portfolio, 'balances', filters] as const,
  portfolioHistory: (params: any) => [...queryKeys.portfolio, 'history', params] as const,
  
  // Strategies
  strategies: ['strategies'] as const,
  strategiesList: () => [...queryKeys.strategies, 'list'] as const,
  strategy: (id: string) => [...queryKeys.strategies, 'detail', id] as const,
  backtestResults: (id: string) => [...queryKeys.strategies, 'backtest', id] as const,
  
  // Risk
  risk: ['risk'] as const,
  riskMetrics: () => [...queryKeys.risk, 'metrics'] as const,
  circuitBreakers: () => [...queryKeys.risk, 'circuit-breakers'] as const,
  riskAlerts: () => [...queryKeys.risk, 'alerts'] as const,
  riskSettings: () => [...queryKeys.risk, 'settings'] as const,
  
  // Market data
  market: ['market'] as const,
  marketData: (symbol: string) => [...queryKeys.market, 'data', symbol] as const,
  orderBook: (symbol: string) => [...queryKeys.market, 'orderbook', symbol] as const,
  candlestick: (symbol: string, timeframe: string) => [
    ...queryKeys.market,
    'candlestick',
    symbol,
    timeframe,
  ] as const,
} as const;

// Predefined query options for different data types
export const queryOptions = {
  // Real-time data (very short stale time)
  realtime: {
    staleTime: 1000, // 1 second
    gcTime: 5 * 60 * 1000, // 5 minutes
    refetchInterval: 1000, // Refetch every second
  },
  
  // Frequently updated data
  frequent: {
    staleTime: 5 * 1000, // 5 seconds
    gcTime: 10 * 60 * 1000, // 10 minutes
    refetchInterval: 5000, // Refetch every 5 seconds
  },
  
  // Moderately updated data
  moderate: {
    staleTime: 30 * 1000, // 30 seconds
    gcTime: 30 * 60 * 1000, // 30 minutes
    refetchInterval: 30000, // Refetch every 30 seconds
  },
  
  // Rarely updated data
  static: {
    staleTime: 5 * 60 * 1000, // 5 minutes
    gcTime: 60 * 60 * 1000, // 1 hour
    refetchInterval: false, // No automatic refetch
  },
  
  // One-time data (like strategy definitions)
  once: {
    staleTime: Infinity,
    gcTime: Infinity,
    refetchInterval: false,
    refetchOnMount: false,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
  },
};

// Utility functions for cache management
export const cacheUtils = {
  // Invalidate all queries for a specific key
  invalidateQueries: (key: readonly unknown[]) => {
    return queryClient.invalidateQueries({ queryKey: key });
  },
  
  // Refetch specific queries
  refetchQueries: (key: readonly unknown[]) => {
    return queryClient.refetchQueries({ queryKey: key });
  },
  
  // Clear cache for specific queries
  removeQueries: (key: readonly unknown[]) => {
    return queryClient.removeQueries({ queryKey: key });
  },
  
  // Update cache data optimistically
  setQueryData: <T>(key: readonly unknown[], data: T) => {
    return queryClient.setQueryData(key, data);
  },
  
  // Get cached data
  getQueryData: <T>(key: readonly unknown[]): T | undefined => {
    return queryClient.getQueryData<T>(key);
  },
  
  // Prefetch data
  prefetchQuery: (key: readonly unknown[], fetcher: () => Promise<any>, options?: any) => {
    return queryClient.prefetchQuery({
      queryKey: key,
      queryFn: fetcher,
      ...options,
    });
  },
};

export default queryClient;