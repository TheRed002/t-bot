/**
 * Authentication slice for Redux store
 * Manages user authentication state and actions
 */

import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { AuthState, User, LoginCredentials, AuthTokens } from '@/types';
import { authAPI, TokenStorage } from '@/services/api/authAPI';

// Initial state
const initialState: AuthState = {
  user: null,
  tokens: authAPI.getStoredTokens(),
  isAuthenticated: authAPI.hasValidTokens(),
  isLoading: false,
  isRefreshing: false,
  error: null,
  rememberMe: TokenStorage.getRememberMe(),
  sessionExpiresAt: null,
};

// Async thunks
export const loginUser = createAsyncThunk(
  'auth/login',
  async (credentials: LoginCredentials, { rejectWithValue }) => {
    try {
      const response = await authAPI.login(credentials);
      
      if (response.success) {
        return {
          user: response.user,
          tokens: response.tokens,
          rememberMe: credentials.remember_me || false,
        };
      }
      
      throw new Error(response.message || 'Login failed');
    } catch (error: any) {
      return rejectWithValue(error.message || 'Login failed');
    }
  }
);

export const registerUser = createAsyncThunk(
  'auth/register',
  async (
    { username, email, password }: { username: string; email: string; password: string },
    { rejectWithValue }
  ) => {
    try {
      const response = await authAPI.register(username, email, password);
      
      if (response.success) {
        return {
          user: response.user,
          tokens: response.tokens,
        };
      }
      
      throw new Error(response.message || 'Registration failed');
    } catch (error: any) {
      return rejectWithValue(error.message || 'Registration failed');
    }
  }
);

export const logoutUser = createAsyncThunk(
  'auth/logout',
  async (_, { rejectWithValue }) => {
    try {
      await authAPI.logout();
      return null;
    } catch (error: any) {
      // Always succeed logout locally even if API call fails
      return null;
    }
  }
);

export const refreshToken = createAsyncThunk(
  'auth/refreshToken',
  async (_, { rejectWithValue }) => {
    try {
      const response = await authAPI.refreshToken();
      
      if (response.success) {
        return {
          user: response.user,
          tokens: response.tokens,
        };
      }
      
      throw new Error(response.message || 'Token refresh failed');
    } catch (error: any) {
      return rejectWithValue(error.message || 'Token refresh failed');
    }
  }
);

export const fetchUserProfile = createAsyncThunk(
  'auth/fetchProfile',
  async (_, { rejectWithValue }) => {
    try {
      const response = await authAPI.getProfile();
      return response.data;
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Failed to fetch profile');
    }
  }
);

export const updateUserProfile = createAsyncThunk(
  'auth/updateProfile',
  async (updates: Partial<User>, { rejectWithValue }) => {
    try {
      const response = await authAPI.updateProfile(updates);
      return response.data;
    } catch (error: any) {
      return rejectWithValue(error.message || 'Failed to update profile');
    }
  }
);

// Mock login for testing without backend
export const mockLogin = createAsyncThunk(
  'auth/mockLogin',
  async (credentials: LoginCredentials, { rejectWithValue }) => {
    try {
      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Create mock user and tokens
      const mockUser: User = {
        user_id: 'mock-user-123',
        username: credentials.username,
        email: `${credentials.username}@example.com`,
        full_name: 'Mock User',
        is_active: true,
        status: 'active' as any,
        roles: ['trader'],
        scopes: ['read', 'write', 'place_order', 'view_portfolio'],
        created_at: new Date().toISOString(),
        last_login: new Date().toISOString(),
        allocated_capital: 10000,
        max_daily_loss: 500,
        risk_level: 'medium',
      };
      
      const mockTokens = {
        access_token: 'mock-access-token-' + Date.now(),
        refresh_token: 'mock-refresh-token-' + Date.now(),
        token_type: 'Bearer',
        expires_in: 3600,
      };
      
      // Store tokens in mock storage
      if (credentials.remember_me) {
        localStorage.setItem('mock_access_token', mockTokens.access_token);
        localStorage.setItem('mock_refresh_token', mockTokens.refresh_token);
        localStorage.setItem('mock_remember_me', 'true');
      } else {
        sessionStorage.setItem('mock_access_token', mockTokens.access_token);
        sessionStorage.setItem('mock_refresh_token', mockTokens.refresh_token);
      }
      
      return {
        user: mockUser,
        tokens: mockTokens,
        rememberMe: credentials.remember_me || false,
      };
    } catch (error: any) {
      return rejectWithValue('Mock login failed');
    }
  }
);

// Auth slice
const authSlice = createSlice({
  name: 'auth',
  initialState,
  reducers: {
    // Clear error state
    clearError: (state) => {
      state.error = null;
    },
    
    // Set loading state
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.isLoading = action.payload;
    },
    
    // Initialize auth from stored token
    initializeAuth: (state) => {
      const tokens = authAPI.getStoredTokens();
      console.log('[authSlice] initializeAuth - tokens found:', !!tokens);
      if (tokens) {
        state.tokens = tokens;
        state.isAuthenticated = true;
        state.rememberMe = TokenStorage.getRememberMe();
        console.log('[authSlice] initializeAuth - setting isAuthenticated to true');
      } else {
        // Clear any stale auth state
        state.tokens = null;
        state.isAuthenticated = false;
        state.user = null;
        console.log('[authSlice] initializeAuth - no tokens, clearing auth state');
      }
    },
    
    // Clear auth state (for forced logout)
    clearAuth: (state) => {
      state.user = null;
      state.tokens = null;
      state.isAuthenticated = false;
      state.error = null;
      state.rememberMe = false;
      state.sessionExpiresAt = null;
      TokenStorage.clearTokens();
    },
    
    // Set refreshing state
    setRefreshing: (state, action: PayloadAction<boolean>) => {
      state.isRefreshing = action.payload;
    },
  },
  extraReducers: (builder) => {
    // Login user
    builder
      .addCase(loginUser.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(loginUser.fulfilled, (state, action) => {
        console.log('[authSlice] loginUser.fulfilled - setting isAuthenticated to true');
        state.isLoading = false;
        state.user = action.payload.user;
        state.tokens = action.payload.tokens;
        state.isAuthenticated = true;
        state.rememberMe = action.payload.rememberMe;
        state.error = null;
        // Calculate session expiry from token expires_in
        if (action.payload.tokens.expires_in) {
          state.sessionExpiresAt = Date.now() + (action.payload.tokens.expires_in * 1000);
        }
        console.log('[authSlice] loginUser.fulfilled - auth state updated:', {
          hasUser: !!action.payload.user,
          hasTokens: !!action.payload.tokens,
          isAuthenticated: true,
          rememberMe: action.payload.rememberMe
        });
      })
      .addCase(loginUser.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
        state.isAuthenticated = false;
      });

    // Register user
    builder
      .addCase(registerUser.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(registerUser.fulfilled, (state, action) => {
        state.isLoading = false;
        state.user = action.payload.user;
        state.tokens = action.payload.tokens;
        state.isAuthenticated = true;
        state.error = null;
        // Calculate session expiry from token expires_in
        if (action.payload.tokens.expires_in) {
          state.sessionExpiresAt = Date.now() + (action.payload.tokens.expires_in * 1000);
        }
      })
      .addCase(registerUser.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
        state.isAuthenticated = false;
      });

    // Logout user
    builder
      .addCase(logoutUser.pending, (state) => {
        state.isLoading = true;
      })
      .addCase(logoutUser.fulfilled, (state) => {
        state.isLoading = false;
        state.user = null;
        state.tokens = null;
        state.isAuthenticated = false;
        state.error = null;
        state.rememberMe = false;
        state.sessionExpiresAt = null;
      })
      .addCase(logoutUser.rejected, (state) => {
        state.isLoading = false;
        // Still clear auth even if logout API call failed
        state.user = null;
        state.tokens = null;
        state.isAuthenticated = false;
        state.rememberMe = false;
        state.sessionExpiresAt = null;
      });

    // Refresh token
    builder
      .addCase(refreshToken.pending, (state) => {
        state.isRefreshing = true;
      })
      .addCase(refreshToken.fulfilled, (state, action) => {
        state.isRefreshing = false;
        state.user = action.payload.user;
        state.tokens = action.payload.tokens;
        state.isAuthenticated = true;
        state.error = null;
        // Calculate session expiry from token expires_in
        if (action.payload.tokens.expires_in) {
          state.sessionExpiresAt = Date.now() + (action.payload.tokens.expires_in * 1000);
        }
      })
      .addCase(refreshToken.rejected, (state, action) => {
        state.isRefreshing = false;
        state.user = null;
        state.tokens = null;
        state.isAuthenticated = false;
        state.error = action.payload as string;
        state.rememberMe = false;
        state.sessionExpiresAt = null;
      });

    // Fetch user profile
    builder
      .addCase(fetchUserProfile.pending, (state) => {
        state.isLoading = true;
      })
      .addCase(fetchUserProfile.fulfilled, (state, action) => {
        state.isLoading = false;
        state.user = action.payload;
        state.error = null;
      })
      .addCase(fetchUserProfile.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
      });

    // Update user profile
    builder
      .addCase(updateUserProfile.pending, (state) => {
        state.isLoading = true;
      })
      .addCase(updateUserProfile.fulfilled, (state, action) => {
        state.isLoading = false;
        state.user = action.payload;
        state.error = null;
      })
      .addCase(updateUserProfile.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
      });

    // Mock login for development
    builder
      .addCase(mockLogin.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(mockLogin.fulfilled, (state, action) => {
        console.log('[authSlice] mockLogin.fulfilled - setting isAuthenticated to true');
        state.isLoading = false;
        state.user = action.payload.user;
        state.tokens = action.payload.tokens;
        state.isAuthenticated = true;
        state.rememberMe = action.payload.rememberMe;
        state.error = null;
        // Calculate session expiry from token expires_in
        if (action.payload.tokens.expires_in) {
          state.sessionExpiresAt = Date.now() + (action.payload.tokens.expires_in * 1000);
        }
        console.log('[authSlice] mockLogin.fulfilled - auth state updated:', {
          hasUser: !!action.payload.user,
          hasTokens: !!action.payload.tokens,
          isAuthenticated: true,
          rememberMe: action.payload.rememberMe
        });
      })
      .addCase(mockLogin.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
        state.isAuthenticated = false;
      });
  },
});

// Export actions
export const { clearError, setLoading, initializeAuth, clearAuth, setRefreshing } = authSlice.actions;

// Export selectors
export const selectAuth = (state: { auth: AuthState }) => state.auth;
export const selectUser = (state: { auth: AuthState }) => state.auth.user;
export const selectIsAuthenticated = (state: { auth: AuthState }) => state.auth.isAuthenticated;
export const selectAuthTokens = (state: { auth: AuthState }) => state.auth.tokens;
export const selectAuthLoading = (state: { auth: AuthState }) => state.auth.isLoading;
export const selectAuthRefreshing = (state: { auth: AuthState }) => state.auth.isRefreshing;
export const selectAuthError = (state: { auth: AuthState }) => state.auth.error;
export const selectRememberMe = (state: { auth: AuthState }) => state.auth.rememberMe;
export const selectSessionExpiresAt = (state: { auth: AuthState }) => state.auth.sessionExpiresAt;

// Export reducer
export default authSlice.reducer;