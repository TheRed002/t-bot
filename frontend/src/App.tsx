/**
 * Main App component for T-Bot Trading System
 * Handles routing, authentication, and global state management
 */

import React, { useEffect, Suspense, lazy } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { Box, CircularProgress } from '@mui/material';

import { useAppDispatch, useAppSelector } from '@/store';
import { initializeAuth, selectIsAuthenticated, selectUser } from '@/store/slices/authSlice';
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
  
  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }
  
  return <>{children}</>;
};

// Loading fallback component
const PageLoadingFallback: React.FC = () => (
  <Box
    display="flex"
    justifyContent="center"
    alignItems="center"
    minHeight="400px"
  >
    <CircularProgress size={40} />
  </Box>
);

const App: React.FC = () => {
  const dispatch = useAppDispatch();
  const isAuthenticated = useAppSelector(selectIsAuthenticated);
  const user = useAppSelector(selectUser);

  // Initialize authentication state on app start
  useEffect(() => {
    dispatch(initializeAuth());
  }, [dispatch]);

  // Connect to WebSocket when authenticated
  useEffect(() => {
    if (isAuthenticated && user) {
      const token = localStorage.getItem('token');
      if (token) {
        websocketService.connect(token);
        dispatch(connect());
      }
    }
  }, [isAuthenticated, user, dispatch]);

  // Cleanup WebSocket on unmount
  useEffect(() => {
    return () => {
      websocketService.disconnect();
    };
  }, []);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
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
    </Box>
  );
};

export default App;