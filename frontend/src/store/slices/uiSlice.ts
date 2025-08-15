/**
 * UI state slice for Redux store
 */

import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { UIState, Notification } from '@/types';

const initialState: UIState = {
  sidebar: {
    isOpen: true,
    isCollapsed: false,
  },
  modals: {
    createBot: false,
    editBot: false,
    strategyConfig: false,
    riskSettings: false,
  },
  notifications: [],
  theme: 'dark',
  isLoading: false,
};

const uiSlice = createSlice({
  name: 'ui',
  initialState,
  reducers: {
    toggleSidebar: (state) => {
      state.sidebar.isOpen = !state.sidebar.isOpen;
    },
    setSidebarCollapsed: (state, action: PayloadAction<boolean>) => {
      state.sidebar.isCollapsed = action.payload;
    },
    openModal: (state, action: PayloadAction<keyof typeof initialState.modals>) => {
      state.modals[action.payload] = true;
    },
    closeModal: (state, action: PayloadAction<keyof typeof initialState.modals>) => {
      state.modals[action.payload] = false;
    },
    addNotification: (state, action: PayloadAction<Omit<Notification, 'id' | 'createdAt'>>) => {
      const notification: Notification = {
        ...action.payload,
        id: Date.now().toString(),
        createdAt: new Date().toISOString(),
      };
      state.notifications.push(notification);
    },
    removeNotification: (state, action: PayloadAction<string>) => {
      state.notifications = state.notifications.filter(n => n.id !== action.payload);
    },
    clearNotifications: (state) => {
      state.notifications = [];
    },
    setTheme: (state, action: PayloadAction<'dark' | 'light'>) => {
      state.theme = action.payload;
    },
  },
});

export const {
  toggleSidebar,
  setSidebarCollapsed,
  openModal,
  closeModal,
  addNotification,
  removeNotification,
  clearNotifications,
  setTheme,
} = uiSlice.actions;

export default uiSlice.reducer;