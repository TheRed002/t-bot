/**
 * API client configuration with axios
 * Centralized HTTP client with interceptors for auth and error handling
 */

import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse, InternalAxiosRequestConfig } from 'axios';
import { ApiResponse, PaginatedResponse, AuthTokens, RefreshTokenResponse } from '@/types';

// Prevent circular imports by declaring the function here
let refreshTokenFunction: (() => Promise<RefreshTokenResponse>) | null = null;
let tokenStorage: any = null;

// Function to set dependencies (called from authAPI)
export const setAuthDependencies = (refreshFn: () => Promise<RefreshTokenResponse>, storage: any) => {
  refreshTokenFunction = refreshFn;
  tokenStorage = storage;
};

// API base configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const API_TIMEOUT = 30000; // 30 seconds

// Create axios instance
export const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL, // Backend routes are at root level (/auth, /trading, etc.)
  timeout: API_TIMEOUT,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Extend InternalAxiosRequestConfig to include metadata
interface CustomAxiosRequestConfig extends InternalAxiosRequestConfig {
  metadata?: {
    startTime: Date;
  };
}

// Request interceptor for adding auth token
apiClient.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    // Get token from secure storage (will be set by authAPI)
    let token: string | null = null;
    
    if (tokenStorage) {
      token = tokenStorage.getAccessToken();
    } else {
      // Fallback to localStorage for backward compatibility
      token = localStorage.getItem('token') || localStorage.getItem('tbot_access_token') || sessionStorage.getItem('tbot_access_token');
    }
    
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    
    // Add request timestamp for debugging
    (config as any).metadata = { startTime: new Date() };
    
    return config;
  },
  (error) => {
    console.error('Request interceptor error:', error);
    return Promise.reject(error);
  }
);

// Track requests being refreshed to prevent multiple refresh attempts
let isRefreshing = false;
let failedQueue: Array<{ resolve: Function; reject: Function }> = [];

const processQueue = (error: any, token: string | null = null) => {
  failedQueue.forEach(({ resolve, reject }) => {
    if (error) {
      reject(error);
    } else {
      resolve(token);
    }
  });
  
  failedQueue = [];
};

// Response interceptor for error handling and automatic token refresh
apiClient.interceptors.response.use(
  (response: AxiosResponse) => {
    // Log response time for performance monitoring
    const metadata = (response.config as any).metadata;
    if (metadata?.startTime) {
      const responseTime = new Date().getTime() - metadata.startTime.getTime();
      if (responseTime > 1000) {
        console.warn(`Slow API response: ${response.config.url} took ${responseTime}ms`);
      }
    }
    
    return response;
  },
  async (error) => {
    const originalRequest = error.config;
    
    // Handle common error scenarios
    if (error.response) {
      const { status, data } = error.response;
      
      // Handle 401 Unauthorized with automatic token refresh
      if (status === 401 && !originalRequest._retry) {
        if (isRefreshing) {
          // If already refreshing, queue this request
          return new Promise((resolve, reject) => {
            failedQueue.push({ resolve, reject });
          }).then(token => {
            originalRequest.headers.Authorization = `Bearer ${token}`;
            return apiClient(originalRequest);
          }).catch(err => {
            return Promise.reject(err);
          });
        }
        
        originalRequest._retry = true;
        isRefreshing = true;
        
        // Attempt to refresh token
        if (refreshTokenFunction && tokenStorage) {
          try {
            const refreshResponse = await refreshTokenFunction();
            const newToken = refreshResponse.tokens.access_token;
            
            // Update authorization header
            apiClient.defaults.headers.common['Authorization'] = `Bearer ${newToken}`;
            originalRequest.headers.Authorization = `Bearer ${newToken}`;
            
            processQueue(null, newToken);
            
            return apiClient(originalRequest);
          } catch (refreshError) {
            processQueue(refreshError, null);
            
            // Clear tokens and redirect to login
            if (tokenStorage) {
              tokenStorage.clearTokens();
            }
            
            // Only redirect if we're not already on the login page
            if (!window.location.pathname.includes('/login')) {
              window.location.href = '/login';
            }
            
            return Promise.reject(refreshError);
          } finally {
            isRefreshing = false;
          }
        } else {
          // No refresh function available, clear tokens and redirect
          if (tokenStorage) {
            tokenStorage.clearTokens();
          } else {
            // Fallback cleanup
            localStorage.removeItem('token');
            localStorage.removeItem('tbot_access_token');
            sessionStorage.removeItem('tbot_access_token');
          }
          
          if (!window.location.pathname.includes('/login')) {
            window.location.href = '/login';
          }
        }
      }
      
      // Handle other error status codes
      switch (status) {
        case 403:
          console.error('Forbidden access:', data?.message || 'Access denied');
          break;
          
        case 404:
          console.error('Resource not found:', error.config.url);
          break;
          
        case 429:
          console.error('Rate limit exceeded');
          break;
          
        case 500:
          console.error('Server error:', data?.message || 'Internal server error');
          break;
          
        default:
          console.error('API error:', status, data?.message || 'Unknown error');
      }
    } else if (error.request) {
      // Network error
      console.error('Network error - no response received');
    } else {
      // Request setup error
      console.error('Request setup error:', error.message);
    }
    
    return Promise.reject(error);
  }
);

// API response types are imported from @/types

// Common API methods
export const api = {
  // GET request
  get: async <T = any>(url: string, config?: AxiosRequestConfig): Promise<ApiResponse<T>> => {
    const response = await apiClient.get<ApiResponse<T>>(url, config);
    return response.data;
  },

  // POST request
  post: async <T = any>(
    url: string,
    data?: any,
    config?: AxiosRequestConfig
  ): Promise<ApiResponse<T>> => {
    const response = await apiClient.post<ApiResponse<T>>(url, data, config);
    return response.data;
  },

  // PUT request
  put: async <T = any>(
    url: string,
    data?: any,
    config?: AxiosRequestConfig
  ): Promise<ApiResponse<T>> => {
    const response = await apiClient.put<ApiResponse<T>>(url, data, config);
    return response.data;
  },

  // PATCH request
  patch: async <T = any>(
    url: string,
    data?: any,
    config?: AxiosRequestConfig
  ): Promise<ApiResponse<T>> => {
    const response = await apiClient.patch<ApiResponse<T>>(url, data, config);
    return response.data;
  },

  // DELETE request
  delete: async <T = any>(url: string, config?: AxiosRequestConfig): Promise<ApiResponse<T>> => {
    const response = await apiClient.delete<ApiResponse<T>>(url, config);
    return response.data;
  },

  // GET with pagination
  getPaginated: async <T = any>(
    url: string,
    params?: { page?: number; limit?: number; [key: string]: any },
    config?: AxiosRequestConfig
  ): Promise<PaginatedResponse<T>> => {
    const response = await apiClient.get<PaginatedResponse<T>>(url, {
      ...config,
      params: {
        page: 1,
        limit: 20,
        ...params,
      },
    });
    return response.data;
  },
};

// Utility functions for API
export const buildQueryString = (params: Record<string, any>): string => {
  const searchParams = new URLSearchParams();
  
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null) {
      if (Array.isArray(value)) {
        value.forEach(v => searchParams.append(key, v.toString()));
      } else {
        searchParams.append(key, value.toString());
      }
    }
  });
  
  return searchParams.toString();
};

export const createAbortController = (): AbortController => {
  return new AbortController();
};

// Export configured client
export default apiClient;

// Export utility for setting up token refresh
export const setupTokenRefresh = () => {
  // This will be called from the auth slice to set up dependencies
  console.log('Token refresh interceptor configured');
};