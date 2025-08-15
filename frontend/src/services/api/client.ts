/**
 * API client configuration with axios
 * Centralized HTTP client with interceptors for auth and error handling
 */

import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';

// API base configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const API_TIMEOUT = 30000; // 30 seconds

// Create axios instance
export const apiClient: AxiosInstance = axios.create({
  baseURL: `${API_BASE_URL}/api`,
  timeout: API_TIMEOUT,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for adding auth token
apiClient.interceptors.request.use(
  (config: AxiosRequestConfig) => {
    const token = localStorage.getItem('token');
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    
    // Add request timestamp for debugging
    config.metadata = { startTime: new Date() };
    
    return config;
  },
  (error) => {
    console.error('Request interceptor error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling and logging
apiClient.interceptors.response.use(
  (response: AxiosResponse) => {
    // Log response time for performance monitoring
    const responseTime = new Date().getTime() - response.config.metadata?.startTime?.getTime();
    if (responseTime > 1000) {
      console.warn(`Slow API response: ${response.config.url} took ${responseTime}ms`);
    }
    
    return response;
  },
  (error) => {
    // Handle common error scenarios
    if (error.response) {
      const { status, data } = error.response;
      
      switch (status) {
        case 401:
          // Unauthorized - clear token and redirect to login
          localStorage.removeItem('token');
          window.location.href = '/login';
          break;
          
        case 403:
          console.error('Forbidden access:', data.message);
          break;
          
        case 404:
          console.error('Resource not found:', error.config.url);
          break;
          
        case 429:
          console.error('Rate limit exceeded');
          break;
          
        case 500:
          console.error('Server error:', data.message);
          break;
          
        default:
          console.error('API error:', status, data.message);
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

// API response types
export interface ApiResponse<T = any> {
  data: T;
  message: string;
  success: boolean;
  timestamp: string;
}

export interface PaginatedResponse<T = any> {
  data: T[];
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
  };
  message: string;
  success: boolean;
}

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