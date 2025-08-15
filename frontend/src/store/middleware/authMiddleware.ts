/**
 * Authentication middleware for automatic token management
 */

import { Middleware } from '@reduxjs/toolkit';
import { clearAuth } from '../slices/authSlice';

export const authMiddleware: Middleware = (store) => (next) => (action) => {
  // Check for 401 errors and automatically logout
  if (action.type.endsWith('/rejected') && action.payload?.includes('401')) {
    store.dispatch(clearAuth());
  }
  
  return next(action);
};