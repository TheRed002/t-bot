/**
 * Main entry point for T-Bot Trading System Frontend
 * Bootstraps React app with all necessary providers
 */

import React from 'react';
import { createRoot } from 'react-dom/client';
import { Provider } from 'react-redux';
import { QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import { ThemeProvider } from '@mui/material/styles';
import { CssBaseline } from '@mui/material';
import { BrowserRouter } from 'react-router-dom';

import App from './App';
import { store } from '@/store';
import { queryClient } from '@/services/queryClient';
import { extendedTheme } from '@/theme';

// Error boundary for the entire app
class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean; error?: Error }
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('App Error Boundary caught an error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100vh',
          backgroundColor: '#0a0e1a',
          color: '#ffffff',
          fontFamily: 'Inter, sans-serif',
          textAlign: 'center',
          padding: '20px',
        }}>
          <h1>T-Bot System Error</h1>
          <p>Something went wrong with the trading system.</p>
          <p style={{ fontSize: '14px', color: '#8892a0', marginTop: '10px' }}>
            {this.state.error?.message}
          </p>
          <button
            onClick={() => window.location.reload()}
            style={{
              marginTop: '20px',
              padding: '10px 20px',
              backgroundColor: '#1c96ff',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
            }}
          >
            Reload Application
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

// Get the root element
const container = document.getElementById('root');
if (!container) {
  throw new Error('Root element not found');
}

const root = createRoot(container);

// Render the app with all providers
root.render(
  <React.StrictMode>
    <ErrorBoundary>
      <Provider store={store}>
        <QueryClientProvider client={queryClient}>
          <ThemeProvider theme={extendedTheme}>
            <CssBaseline />
            <BrowserRouter>
              <App />
            </BrowserRouter>
          </ThemeProvider>
          {process.env.NODE_ENV === 'development' && (
            <ReactQueryDevtools initialIsOpen={false} />
          )}
        </QueryClientProvider>
      </Provider>
    </ErrorBoundary>
  </React.StrictMode>
);