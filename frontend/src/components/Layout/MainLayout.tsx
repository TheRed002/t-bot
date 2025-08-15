/**
 * Main layout component for authenticated users
 * Includes sidebar navigation and header
 */

import React from 'react';
import { Box, AppBar, Toolbar, Typography, IconButton } from '@mui/material';
import { Menu as MenuIcon, Notifications as NotificationsIcon } from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '@/store';
import { toggleSidebar } from '@/store/slices/uiSlice';
import { colors } from '@/theme/colors';

interface MainLayoutProps {
  children: React.ReactNode;
}

const MainLayout: React.FC<MainLayoutProps> = ({ children }) => {
  const dispatch = useAppDispatch();
  const { sidebar } = useAppSelector((state) => state.ui);

  const handleToggleSidebar = () => {
    dispatch(toggleSidebar());
  };

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      {/* App Bar */}
      <AppBar
        position="fixed"
        sx={{
          zIndex: (theme) => theme.zIndex.drawer + 1,
          backgroundColor: colors.background.elevated,
          borderBottom: `1px solid ${colors.border.primary}`,
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="toggle sidebar"
            onClick={handleToggleSidebar}
            edge="start"
            sx={{ mr: 2 }}
          >
            <MenuIcon />
          </IconButton>
          
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            T-Bot Trading System
          </Typography>
          
          <IconButton color="inherit">
            <NotificationsIcon />
          </IconButton>
        </Toolbar>
      </AppBar>

      {/* Sidebar placeholder */}
      {sidebar.isOpen && (
        <Box
          sx={{
            width: 280,
            flexShrink: 0,
            backgroundColor: colors.background.elevated,
            borderRight: `1px solid ${colors.border.primary}`,
            position: 'fixed',
            height: '100vh',
            zIndex: (theme) => theme.zIndex.drawer,
            mt: 8, // Account for AppBar height
          }}
        >
          {/* Sidebar content will go here */}
          <Typography sx={{ p: 2, color: colors.text.secondary }}>
            Navigation placeholder
          </Typography>
        </Box>
      )}

      {/* Main content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          mt: 8, // Account for AppBar height
          ml: sidebar.isOpen ? '280px' : 0,
          transition: 'margin-left 0.3s ease',
          backgroundColor: colors.background.primary,
          minHeight: 'calc(100vh - 64px)',
        }}
      >
        {children}
      </Box>
    </Box>
  );
};

export default MainLayout;