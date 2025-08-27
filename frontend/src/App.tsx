/**
 * Main App component for T-Bot Trading System
 * Handles routing, authentication, and global state management
 */

import React, { useEffect, Suspense, lazy } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';

// Import global styles
import './styles/globals.css';

import { useAppDispatch, useAppSelector } from '@/store';
import { initializeAuth, selectIsAuthenticated, selectUser, selectAuthTokens, fetchUserProfile } from '@/store/slices/authSlice';
import { connect } from '@/store/slices/websocketSlice';
import { websocketService } from '@/services/websocket';

// Lazy load pages for better performance
const LoginPage = lazy(() => import('@/pages/LoginPage'));
const DashboardPage = lazy(() => import('@/pages/DashboardPage'));
const TradingPage = lazy(() => import('@/pages/TradingPage'));
const BotManagementPage = lazy(() => import('@/pages/BotManagementPage'));
const PortfolioPage = lazy(() => import('@/pages/PortfolioPage'));
const StrategyCenterPage = lazy(() => import('@/pages/StrategyCenterPage'));
const RiskDashboardPage = lazy(() => import('@/pages/RiskDashboardPage'));
const PlaygroundPage = lazy(() => import('@/pages/Playground/PlaygroundPage'));
const HelpPage = lazy(() => import('@/pages/HelpPage'));

// Layout components
import MainLayout from '@/components/Layout/MainLayout';
import AuthLayout from '@/components/Layout/AuthLayout';
import LoadingScreen from '@/components/Common/LoadingScreen';
import NotificationSystem from '@/components/Common/NotificationSystem';

// Protected Route component
interface ProtectedRouteProps {
  children: React.ReactNode;
}

const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ children }) => {
  const isAuthenticated = useAppSelector(selectIsAuthenticated);
  
  console.log('[ProtectedRoute] isAuthenticated:', isAuthenticated);
  
  if (!isAuthenticated) {
    console.log('[ProtectedRoute] User not authenticated, redirecting to login...');
    return <Navigate to="/login" replace />;
  }
  
  console.log('[ProtectedRoute] User authenticated, rendering protected content');
  return <>{children}</>;
};

// Loading fallback component
const PageLoadingFallback: React.FC = () => (
  <div className="flex items-center justify-center min-h-[400px]">
    <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-primary"></div>
  </div>
);

const App: React.FC = () => {
  const dispatch = useAppDispatch();
  const isAuthenticated = useAppSelector(selectIsAuthenticated);
  const user = useAppSelector(selectUser);
  const tokens = useAppSelector(selectAuthTokens);

  // Initialize authentication state on app start
  useEffect(() => {
    // Only run once on mount
    console.log('[App] Initializing authentication...');
    dispatch(initializeAuth());
  }, [dispatch]);

  // Fetch user profile if we have a token but no user data
  useEffect(() => {
    console.log('[App] Auth state:', { isAuthenticated, hasTokens: !!tokens, hasUser: !!user });
    if (isAuthenticated && tokens && !user) {
      console.log('[App] Fetching user profile...');
      dispatch(fetchUserProfile());
    }
  }, [dispatch, isAuthenticated, tokens, user]);

  // Connect to WebSocket when authenticated
  useEffect(() => {
    // Only connect if we have both authentication and user data
    if (isAuthenticated && user) {
      const token = localStorage.getItem('token');
      if (token) {
        try {
          websocketService.connect(token);
          dispatch(connect());
        } catch (error) {
          console.error('WebSocket connection failed:', error);
        }
      }
    } else {
      // Ensure WebSocket is disconnected when not authenticated
      websocketService.disconnect();
    }
  }, [isAuthenticated, user, dispatch]);

  // Cleanup WebSocket on unmount
  useEffect(() => {
    return () => {
      websocketService.disconnect();
    };
  }, []);

  return (
    <div className="flex flex-col min-h-screen bg-background text-foreground">
      <Suspense fallback={<LoadingScreen />}>
        <Routes>
          {/* Authentication routes */}
          <Route
            path="/login"
            element={
              <AuthLayout>
                <LoginPage />
              </AuthLayout>
            }
          />
          
          {/* Protected application routes */}
          <Route
            path="/*"
            element={
              <ProtectedRoute>
                <MainLayout>
                  <Suspense fallback={<PageLoadingFallback />}>
                    <Routes>
                      {/* Dashboard - Default route */}
                      <Route path="/" element={<Navigate to="/dashboard" replace />} />
                      <Route path="/dashboard" element={<DashboardPage />} />
                      
                      {/* Trading */}
                      <Route path="/trading" element={<TradingPage />} />
                      
                      {/* Bot Management */}
                      <Route path="/bots" element={<BotManagementPage />} />
                      <Route path="/bots/:botId" element={<BotManagementPage />} />
                      
                      {/* Portfolio */}
                      <Route path="/portfolio" element={<PortfolioPage />} />
                      
                      {/* Strategy Center */}
                      <Route path="/strategies" element={<StrategyCenterPage />} />
                      <Route path="/strategies/:strategyId" element={<StrategyCenterPage />} />
                      
                      {/* Risk Dashboard */}
                      <Route path="/risk" element={<RiskDashboardPage />} />
                      
                      {/* Playground */}
                      <Route path="/playground" element={<PlaygroundPage />} />
                      
                      {/* Help & Documentation */}
                      <Route path="/help" element={<HelpPage />} />
                      
                      {/* Catch-all route */}
                      <Route path="*" element={<Navigate to="/dashboard" replace />} />
                    </Routes>
                  </Suspense>
                </MainLayout>
              </ProtectedRoute>
            }
          />
        </Routes>
      </Suspense>
      
      {/* Global notification system */}
      <NotificationSystem />
    </div>
  );
};

export default App;