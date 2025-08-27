/**
 * Authentication API service
 * Handles all authentication-related API calls including login, refresh, logout
 */

import { api, apiClient, setAuthDependencies } from './client';
import { 
  User, 
  LoginCredentials, 
  LoginResponse, 
  RefreshTokenResponse,
  AuthTokens 
} from '@/types';

// Token storage utilities
class TokenStorage {
  private static readonly ACCESS_TOKEN_KEY = 'tbot_access_token';
  private static readonly REFRESH_TOKEN_KEY = 'tbot_refresh_token';
  private static readonly REMEMBER_ME_KEY = 'tbot_remember_me';
  
  static setTokens(tokens: AuthTokens, rememberMe: boolean = false): void {
    if (rememberMe) {
      // Store in localStorage for persistent sessions
      localStorage.setItem(this.ACCESS_TOKEN_KEY, tokens.access_token);
      localStorage.setItem(this.REFRESH_TOKEN_KEY, tokens.refresh_token);
      localStorage.setItem(this.REMEMBER_ME_KEY, 'true');
    } else {
      // Store in sessionStorage for session-only
      sessionStorage.setItem(this.ACCESS_TOKEN_KEY, tokens.access_token);
      sessionStorage.setItem(this.REFRESH_TOKEN_KEY, tokens.refresh_token);
      localStorage.removeItem(this.REMEMBER_ME_KEY);
    }
  }
  
  static getAccessToken(): string | null {
    return localStorage.getItem(this.ACCESS_TOKEN_KEY) || 
           sessionStorage.getItem(this.ACCESS_TOKEN_KEY);
  }
  
  static getRefreshToken(): string | null {
    return localStorage.getItem(this.REFRESH_TOKEN_KEY) || 
           sessionStorage.getItem(this.REFRESH_TOKEN_KEY);
  }
  
  static getRememberMe(): boolean {
    return localStorage.getItem(this.REMEMBER_ME_KEY) === 'true';
  }
  
  static clearTokens(): void {
    localStorage.removeItem(this.ACCESS_TOKEN_KEY);
    localStorage.removeItem(this.REFRESH_TOKEN_KEY);
    localStorage.removeItem(this.REMEMBER_ME_KEY);
    sessionStorage.removeItem(this.ACCESS_TOKEN_KEY);
    sessionStorage.removeItem(this.REFRESH_TOKEN_KEY);
  }
}

export const authAPI = {
  // Login user with credentials
  login: async (credentials: LoginCredentials): Promise<LoginResponse> => {
    try {
      const response = await apiClient.post<LoginResponse>('/auth/login', {
        username: credentials.username,
        password: credentials.password,
      });
      
      const loginResponse = response.data;
      
      if (loginResponse.success && loginResponse.tokens) {
        // Store tokens based on remember me preference
        TokenStorage.setTokens(loginResponse.tokens, credentials.remember_me || false);
        
        return loginResponse;
      }
      
      throw new Error(loginResponse.message || 'Login failed');
    } catch (error: any) {
      // Enhanced error handling for different authentication failures
      if (error.response) {
        const { status, data } = error.response;
        
        switch (status) {
          case 401:
            throw new Error(data.message || 'Invalid username or password');
          case 423:
            throw new Error('Account is locked. Please try again later.');
          case 429:
            throw new Error('Too many login attempts. Please try again later.');
          case 403:
            throw new Error('Account is disabled. Please contact support.');
          default:
            throw new Error(data.message || 'Login failed');
        }
      }
      
      if (error.code === 'ERR_NETWORK') {
        throw new Error('Network error. Please check your connection and try again.');
      }
      
      throw new Error(error.message || 'An unexpected error occurred');
    }
  },

  // Register new user
  register: async (username: string, email: string, password: string): Promise<LoginResponse> => {
    try {
      const response = await apiClient.post<LoginResponse>('/auth/register', {
        username,
        email,
        password,
      });
      
      const registerResponse = response.data;
      
      if (registerResponse.success && registerResponse.tokens) {
        // Store tokens after successful registration
        TokenStorage.setTokens(registerResponse.tokens);
        return registerResponse;
      }
      
      throw new Error(registerResponse.message || 'Registration failed');
    } catch (error: any) {
      if (error.response) {
        const { status, data } = error.response;
        
        switch (status) {
          case 409:
            throw new Error('Username or email already exists');
          case 422:
            throw new Error(data.message || 'Invalid registration data');
          default:
            throw new Error(data.message || 'Registration failed');
        }
      }
      
      throw new Error(error.message || 'Registration failed');
    }
  },

  // Logout user
  logout: async (): Promise<void> => {
    try {
      const refreshToken = TokenStorage.getRefreshToken();
      
      if (refreshToken) {
        // Notify backend to invalidate tokens
        await apiClient.post('/auth/logout', {
          refresh_token: refreshToken
        });
      }
    } catch (error) {
      // Log error but don't throw - always clear local tokens
      console.warn('Logout API call failed:', error);
    } finally {
      // Always clear local tokens
      TokenStorage.clearTokens();
    }
  },

  // Refresh access token
  refreshToken: async (): Promise<RefreshTokenResponse> => {
    try {
      const refreshToken = TokenStorage.getRefreshToken();
      
      if (!refreshToken) {
        throw new Error('No refresh token available');
      }
      
      const response = await apiClient.post<RefreshTokenResponse>('/auth/refresh', {
        refresh_token: refreshToken
      });
      
      const refreshResponse = response.data;
      
      if (refreshResponse.success && refreshResponse.tokens) {
        // Update stored tokens
        TokenStorage.setTokens(refreshResponse.tokens, TokenStorage.getRememberMe());
        return refreshResponse;
      }
      
      throw new Error(refreshResponse.message || 'Token refresh failed');
    } catch (error: any) {
      // Clear tokens on refresh failure
      TokenStorage.clearTokens();
      
      if (error.response?.status === 401) {
        throw new Error('Session expired. Please log in again.');
      }
      
      throw new Error(error.message || 'Failed to refresh session');
    }
  },

  // Get current user profile
  getProfile: async (): Promise<{ data: User }> => {
    try {
      const response = await api.get<User>('/auth/me');
      return { data: response.data };
    } catch (error: any) {
      if (error.response?.status === 401) {
        // Token is invalid, clear stored tokens
        TokenStorage.clearTokens();
        throw new Error('Authentication expired. Please log in again.');
      }
      throw error;
    }
  },

  // Update user profile
  updateProfile: async (updates: Partial<User>): Promise<{ data: User }> => {
    const response = await api.patch<User>('/auth/me', updates);
    return { data: response.data };
  },

  // Change password
  changePassword: async (currentPassword: string, newPassword: string) => {
    return api.post('/auth/me/password', {
      current_password: currentPassword,
      new_password: newPassword,
    });
  },

  // Request password reset
  requestPasswordReset: async (email: string) => {
    return api.post('/auth/password-reset', { email });
  },

  // Reset password with token
  resetPassword: async (token: string, newPassword: string) => {
    return api.post('/auth/password-reset/confirm', {
      token,
      new_password: newPassword,
    });
  },

  // Validate current session
  validateSession: async (): Promise<boolean> => {
    try {
      const accessToken = TokenStorage.getAccessToken();
      if (!accessToken) {
        return false;
      }
      
      // Try to get user profile to validate token
      await authAPI.getProfile();
      return true;
    } catch {
      return false;
    }
  },

  // Check if user has valid stored tokens
  hasValidTokens: (): boolean => {
    const accessToken = TokenStorage.getAccessToken();
    const refreshToken = TokenStorage.getRefreshToken();
    return !!(accessToken && refreshToken);
  },

  // Get stored tokens
  getStoredTokens: (): AuthTokens | null => {
    const accessToken = TokenStorage.getAccessToken();
    const refreshToken = TokenStorage.getRefreshToken();
    
    if (!accessToken || !refreshToken) {
      return null;
    }
    
    return {
      access_token: accessToken,
      refresh_token: refreshToken,
      token_type: 'Bearer',
      expires_in: 3600, // Will be updated on refresh
    };
  },
};

// Set up dependencies for automatic token refresh
setAuthDependencies(authAPI.refreshToken, TokenStorage);

// Export token storage utilities for use in other parts of the app
export { TokenStorage };