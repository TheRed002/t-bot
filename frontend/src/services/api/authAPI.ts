/**
 * Authentication API service
 */

import { api } from './client';
import { User } from '@/types';

export const authAPI = {
  // Login user
  login: async (email: string, password: string) => {
    return api.post<{ user: User; token: string }>('/auth/login', {
      email,
      password,
    });
  },

  // Register user
  register: async (username: string, email: string, password: string) => {
    return api.post<{ user: User; token: string }>('/auth/register', {
      username,
      email,
      password,
    });
  },

  // Logout user
  logout: async () => {
    return api.post('/auth/logout');
  },

  // Refresh token
  refreshToken: async () => {
    return api.post<{ user: User; token: string }>('/auth/refresh');
  },

  // Get user profile
  getProfile: async () => {
    return api.get<User>('/auth/profile');
  },

  // Update user profile
  updateProfile: async (updates: Partial<User>) => {
    return api.patch<User>('/auth/profile', updates);
  },

  // Change password
  changePassword: async (currentPassword: string, newPassword: string) => {
    return api.post('/auth/change-password', {
      currentPassword,
      newPassword,
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
      newPassword,
    });
  },
};