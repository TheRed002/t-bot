/**
 * Error handling middleware
 */

import { Middleware } from '@reduxjs/toolkit';
import { addNotification } from '../slices/uiSlice';

export const errorMiddleware: Middleware = (store) => (next) => (action) => {
  const result = next(action);
  
  // Handle rejected actions and show notifications
  if (action.type.endsWith('/rejected') && action.payload) {
    store.dispatch(addNotification({
      type: 'error',
      title: 'Error',
      message: action.payload,
      autoHide: true,
      duration: 5000,
    }));
  }
  
  return result;
};