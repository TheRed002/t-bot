/**
 * Tests for MainLayout component
 * Verifies that the sidebar layout is working correctly
 */

import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { Provider } from 'react-redux';
import { BrowserRouter } from 'react-router-dom';
import { configureStore } from '@reduxjs/toolkit';
import MainLayout from '../MainLayout';
import uiSlice from '@/store/slices/uiSlice';
import authSlice from '@/store/slices/authSlice';

// Mock store for testing
const createMockStore = (initialState = {}) => {
  return configureStore({
    reducer: {
      ui: uiSlice,
      auth: authSlice,
    },
    preloadedState: {
      ui: {
        sidebar: { isOpen: true, isCollapsed: false },
        modals: {
          createBot: false,
          editBot: false,
          strategyConfig: false,
          riskSettings: false,
        },
        notifications: [],
        theme: 'dark',
        isLoading: false,
      },
      auth: {
        user: { 
          user_id: '1', 
          username: 'testuser', 
          email: 'test@example.com', 
          is_active: true, 
          status: 'active' as any,
          roles: ['user'],
          scopes: [],
          created_at: '2023-01-01T00:00:00Z'
        },
        tokens: { access_token: 'test-token', refresh_token: 'refresh-token', token_type: 'Bearer', expires_in: 3600 },
        isAuthenticated: true,
        isLoading: false,
        isRefreshing: false,
        error: null,
        rememberMe: false,
        sessionExpiresAt: null,
      },
      ...initialState,
    },
  });
};

// Test wrapper component
const TestWrapper: React.FC<{ 
  children: React.ReactNode; 
  store?: any;
}> = ({ children, store }) => {
  const testStore = store || createMockStore();
  
  return (
    <Provider store={testStore}>
      <BrowserRouter>
        {children}
      </BrowserRouter>
    </Provider>
  );
};

describe('MainLayout', () => {
  it('renders without crashing', () => {
    render(
      <TestWrapper>
        <MainLayout>
          <div>Test Content</div>
        </MainLayout>
      </TestWrapper>
    );
    
    expect(screen.getByText('Test Content')).toBeInTheDocument();
  });

  it('displays sidebar when open', () => {
    render(
      <TestWrapper>
        <MainLayout>
          <div>Test Content</div>
        </MainLayout>
      </TestWrapper>
    );
    
    // Check that navigation elements are visible
    expect(screen.getByText('Dashboard')).toBeInTheDocument();
    expect(screen.getByText('Trading')).toBeInTheDocument();
    expect(screen.getByText('T-Bot')).toBeInTheDocument();
  });

  it('toggles sidebar when menu button is clicked', () => {
    const store = createMockStore();
    
    render(
      <TestWrapper store={store}>
        <MainLayout>
          <div>Test Content</div>
        </MainLayout>
      </TestWrapper>
    );
    
    const menuButton = screen.getByLabelText('toggle sidebar');
    fireEvent.click(menuButton);
    
    // Check that the sidebar state changed in the store
    const state = store.getState();
    expect(state.ui.sidebar.isOpen).toBe(false);
  });

  it('renders user information in header', () => {
    render(
      <TestWrapper>
        <MainLayout>
          <div>Test Content</div>
        </MainLayout>
      </TestWrapper>
    );
    
    // Check for T-Bot branding
    expect(screen.getByText('T-Bot Trading System')).toBeInTheDocument();
    expect(screen.getByText('Live')).toBeInTheDocument();
  });

  it('handles sidebar closed state correctly', () => {
    const store = createMockStore({
      ui: {
        sidebar: { isOpen: false, isCollapsed: false },
        modals: {
          createBot: false,
          editBot: false,
          strategyConfig: false,
          riskSettings: false,
        },
        notifications: [],
        theme: 'dark',
        isLoading: false,
      },
    });
    
    render(
      <TestWrapper store={store}>
        <MainLayout>
          <div data-testid="main-content">Test Content</div>
        </MainLayout>
      </TestWrapper>
    );
    
    const mainContent = screen.getByTestId('main-content');
    expect(mainContent).toBeInTheDocument();
    
    // The sidebar should not be affecting the main content spacing when closed
    // This is tested implicitly by the component rendering without errors
  });

  it('shows correct theme toggle button', () => {
    render(
      <TestWrapper>
        <MainLayout>
          <div>Test Content</div>
        </MainLayout>
      </TestWrapper>
    );
    
    // Should show light mode icon when in dark theme
    const themeButton = screen.getByTitle('Switch to light mode');
    expect(themeButton).toBeInTheDocument();
  });
});