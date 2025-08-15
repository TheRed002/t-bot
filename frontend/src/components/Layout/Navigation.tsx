/**
 * Navigation component for the main layout
 * Provides sidebar navigation with all major sections
 */

import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Divider,
  Typography,
  Box,
  useTheme,
  alpha
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  TrendingUp as TradingIcon,
  SmartToy as BotIcon,
  AccountBalance as PortfolioIcon,
  Psychology as StrategyIcon,
  Security as RiskIcon,
  Science as PlaygroundIcon,
  Assessment as AnalyticsIcon,
  Settings as SettingsIcon,
  Help as HelpIcon
} from '@mui/icons-material';

interface NavigationItem {
  path: string;
  label: string;
  icon: React.ReactNode;
  description: string;
}

const navigationItems: NavigationItem[] = [
  {
    path: '/dashboard',
    label: 'Dashboard',
    icon: <DashboardIcon />,
    description: 'System overview and key metrics'
  },
  {
    path: '/trading',
    label: 'Trading',
    icon: <TradingIcon />,
    description: 'Live trading interface'
  },
  {
    path: '/bots',
    label: 'Bot Management',
    icon: <BotIcon />,
    description: 'Manage trading bots'
  },
  {
    path: '/portfolio',
    label: 'Portfolio',
    icon: <PortfolioIcon />,
    description: 'Portfolio tracking and analysis'
  },
  {
    path: '/strategies',
    label: 'Strategy Center',
    icon: <StrategyIcon />,
    description: 'Strategy development and testing'
  },
  {
    path: '/risk',
    label: 'Risk Dashboard',
    icon: <RiskIcon />,
    description: 'Risk monitoring and controls'
  },
  {
    path: '/playground',
    label: 'Playground',
    icon: <PlaygroundIcon />,
    description: 'Strategy testing and optimization'
  },
  {
    path: '/help',
    label: 'Help & Documentation',
    icon: <HelpIcon />,
    description: 'Documentation and user guides'
  }
];

const Navigation: React.FC = () => {
  const theme = useTheme();
  const location = useLocation();
  const navigate = useNavigate();

  const handleNavigate = (path: string) => {
    navigate(path);
  };

  const isActive = (path: string) => {
    if (path === '/dashboard') {
      return location.pathname === '/' || location.pathname === '/dashboard';
    }
    return location.pathname.startsWith(path);
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Header */}
      <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
        <Typography
          variant="h6"
          fontWeight="bold"
          sx={{
            background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
            backgroundClip: 'text',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}
        >
          T-Bot
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Trading System
        </Typography>
      </Box>

      {/* Navigation List */}
      <List sx={{ flexGrow: 1, py: 1 }}>
        {navigationItems.map((item) => (
          <ListItem key={item.path} disablePadding sx={{ mb: 0.5 }}>
            <ListItemButton
              onClick={() => handleNavigate(item.path)}
              selected={isActive(item.path)}
              sx={{
                mx: 1,
                borderRadius: 2,
                minHeight: 56,
                '&.Mui-selected': {
                  backgroundColor: alpha(theme.palette.primary.main, 0.12),
                  borderLeft: `4px solid ${theme.palette.primary.main}`,
                  '&:hover': {
                    backgroundColor: alpha(theme.palette.primary.main, 0.16),
                  },
                  '& .MuiListItemIcon-root': {
                    color: theme.palette.primary.main,
                  },
                  '& .MuiListItemText-primary': {
                    fontWeight: 600,
                    color: theme.palette.primary.main,
                  },
                },
                '&:hover': {
                  backgroundColor: alpha(theme.palette.text.primary, 0.04),
                },
              }}
            >
              <ListItemIcon
                sx={{
                  minWidth: 40,
                  color: isActive(item.path) ? 'primary.main' : 'text.secondary',
                }}
              >
                {item.icon}
              </ListItemIcon>
              <ListItemText
                primary={item.label}
                secondary={item.description}
                primaryTypographyProps={{
                  fontSize: '0.95rem',
                  fontWeight: isActive(item.path) ? 600 : 500,
                }}
                secondaryTypographyProps={{
                  fontSize: '0.75rem',
                  color: 'text.secondary',
                  sx: { mt: 0.25 },
                }}
              />
            </ListItemButton>
          </ListItem>
        ))}
      </List>

      <Divider />

      {/* Footer Navigation */}
      <List sx={{ py: 1 }}>
        <ListItem disablePadding>
          <ListItemButton
            sx={{
              mx: 1,
              borderRadius: 2,
              minHeight: 48,
              '&:hover': {
                backgroundColor: alpha(theme.palette.text.primary, 0.04),
              },
            }}
          >
            <ListItemIcon sx={{ minWidth: 40, color: 'text.secondary' }}>
              <AnalyticsIcon />
            </ListItemIcon>
            <ListItemText
              primary="Analytics"
              primaryTypographyProps={{
                fontSize: '0.9rem',
                fontWeight: 500,
              }}
            />
          </ListItemButton>
        </ListItem>

        <ListItem disablePadding>
          <ListItemButton
            sx={{
              mx: 1,
              borderRadius: 2,
              minHeight: 48,
              '&:hover': {
                backgroundColor: alpha(theme.palette.text.primary, 0.04),
              },
            }}
          >
            <ListItemIcon sx={{ minWidth: 40, color: 'text.secondary' }}>
              <SettingsIcon />
            </ListItemIcon>
            <ListItemText
              primary="Settings"
              primaryTypographyProps={{
                fontSize: '0.9rem',
                fontWeight: 500,
              }}
            />
          </ListItemButton>
        </ListItem>
      </List>

      {/* Status Indicator */}
      <Box sx={{ p: 2, borderTop: 1, borderColor: 'divider' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Box
            sx={{
              width: 8,
              height: 8,
              borderRadius: '50%',
              backgroundColor: 'success.main',
              animation: 'pulse 2s infinite',
            }}
          />
          <Typography variant="caption" color="text.secondary">
            System Online
          </Typography>
        </Box>
      </Box>

      <style>
        {`
          @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
          }
        `}
      </style>
    </Box>
  );
};

export default Navigation;