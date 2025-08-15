/**
 * Dashboard page component
 * Professional trading dashboard with real-time data, charts, and comprehensive metrics
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  Box,
  Grid,
  Typography,
  Paper,
  Tabs,
  Tab,
  Alert,
  Chip,
  IconButton,
  Tooltip,
  Switch,
  FormControlLabel,
  Button,
} from '@mui/material';
import {
  Refresh,
  Settings,
  Notifications,
  Timeline,
  TrendingUp,
  Security,
  Speed,
} from '@mui/icons-material';
import { motion } from 'framer-motion';
import { useAppSelector, useAppDispatch } from '@/store';
import { colors } from '@/theme/colors';
import { tradingTheme } from '@/theme';

// Import our enhanced components
import MetricCard from '@/components/Trading/MetricCard';
import PriceChart from '@/components/Trading/PriceChart';
import PositionCard, { Position } from '@/components/Trading/PositionCard';

// Mock data - In real app, this would come from Redux store/API
const mockPortfolioData = {
  totalValue: 12450.67,
  dailyChange: 245.32,
  dailyChangePercent: 2.01,
  activeBots: 3,
  runningBots: 2,
  pausedBots: 1,
  dailyPnL: 89.32,
  tradesExecuted: 15,
  riskLevel: 'Medium',
  portfolioExposure: 3.2,
  weeklyReturn: 5.67,
  monthlyReturn: 12.45,
  maxDrawdown: -2.34,
  sharpeRatio: 1.87,
  totalTrades: 847,
  winRate: 68.5,
  avgTradeReturn: 0.23,
};

const mockPriceData = Array.from({ length: 50 }, (_, i) => {
  const basePrice = 45000;
  const timestamp = Date.now() - (50 - i) * 60000; // 1 minute intervals
  const noise = (Math.random() - 0.5) * 1000;
  const trend = i * 10;
  const price = basePrice + trend + noise;
  
  return {
    timestamp,
    price,
    close: price,
    open: price - (Math.random() - 0.5) * 100,
    high: price + Math.random() * 200,
    low: price - Math.random() * 200,
    volume: Math.random() * 1000000,
  };
});

const mockPositions: Position[] = [
  {
    id: '1',
    symbol: 'BTC/USD',
    side: 'long',
    size: 0.5,
    entryPrice: 44200,
    currentPrice: 45100,
    stopLoss: 43000,
    takeProfit: 47000,
    unrealizedPnL: 450,
    unrealizedPnLPercent: 2.04,
    margin: 2210,
    leverage: 10,
    timestamp: Date.now() - 3600000,
    strategy: 'Momentum',
    exchange: 'Binance',
    status: 'open',
  },
  {
    id: '2',
    symbol: 'ETH/USD',
    side: 'short',
    size: 2.5,
    entryPrice: 2850,
    currentPrice: 2820,
    stopLoss: 2900,
    takeProfit: 2750,
    unrealizedPnL: 75,
    unrealizedPnLPercent: 1.05,
    margin: 712.5,
    leverage: 5,
    timestamp: Date.now() - 1800000,
    strategy: 'Mean Reversion',
    exchange: 'Coinbase',
    status: 'open',
  },
];

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel: React.FC<TabPanelProps> = ({ children, value, index }) => {
  return (
    <div role="tabpanel" hidden={value !== index}>
      {value === index && <Box sx={{ pt: 3 }}>{children}</Box>}
    </div>
  );
};

const DashboardPage: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [loading, setLoading] = useState(false);
  const [lastUpdate, setLastUpdate] = useState(new Date());
  
  // Simulate real-time updates
  useEffect(() => {
    if (!autoRefresh) return;
    
    const interval = setInterval(() => {
      setLastUpdate(new Date());
      // In real app, this would trigger data refresh
    }, 5000);
    
    return () => clearInterval(interval);
  }, [autoRefresh]);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const handleRefresh = useCallback(async () => {
    setLoading(true);
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1000));
    setLastUpdate(new Date());
    setLoading(false);
  }, []);

  const portfolioMetrics = useMemo(() => [
    {
      title: 'Portfolio Value',
      value: mockPortfolioData.totalValue,
      change: mockPortfolioData.dailyChange,
      changePercent: mockPortfolioData.dailyChangePercent,
      format: 'currency' as const,
      tooltip: 'Total value of all holdings including open positions',
      variant: 'highlighted' as const,
    },
    {
      title: 'Daily P&L',
      value: mockPortfolioData.dailyPnL,
      change: mockPortfolioData.dailyPnL,
      format: 'currency' as const,
      subtitle: `${mockPortfolioData.tradesExecuted} trades`,
      tooltip: 'Profit and loss for the current trading day',
    },
    {
      title: 'Active Bots',
      value: mockPortfolioData.activeBots,
      subtitle: `${mockPortfolioData.runningBots} running, ${mockPortfolioData.pausedBots} paused`,
      format: 'number' as const,
      precision: 0,
      tooltip: 'Number of active trading bots',
    },
    {
      title: 'Risk Level',
      value: mockPortfolioData.riskLevel,
      subtitle: `${mockPortfolioData.portfolioExposure}% exposure`,
      format: 'custom' as const,
      tooltip: 'Current portfolio risk assessment',
    },
    {
      title: 'Weekly Return',
      value: mockPortfolioData.weeklyReturn,
      format: 'percentage' as const,
      precision: 2,
      tooltip: 'Portfolio return over the last 7 days',
    },
    {
      title: 'Monthly Return',
      value: mockPortfolioData.monthlyReturn,
      format: 'percentage' as const,
      precision: 2,
      tooltip: 'Portfolio return over the last 30 days',
    },
    {
      title: 'Max Drawdown',
      value: mockPortfolioData.maxDrawdown,
      format: 'percentage' as const,
      precision: 2,
      tooltip: 'Maximum peak-to-trough decline',
    },
    {
      title: 'Sharpe Ratio',
      value: mockPortfolioData.sharpeRatio,
      format: 'number' as const,
      precision: 2,
      tooltip: 'Risk-adjusted return measure',
    },
  ], []);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <Box>
        {/* Header */}
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
          <Box>
            <Typography variant="h4" component="h1" gutterBottom sx={{ fontWeight: 600 }}>
              Trading Dashboard
            </Typography>
            <Typography variant="body1" color="text.secondary">
              Real-time portfolio monitoring and bot management • Last updated: {lastUpdate.toLocaleTimeString()}
            </Typography>
          </Box>
          
          <Box display="flex" alignItems="center" gap={2}>
            <FormControlLabel
              control={
                <Switch
                  checked={autoRefresh}
                  onChange={(e) => setAutoRefresh(e.target.checked)}
                  size="small"
                />
              }
              label="Auto-refresh"
            />
            
            <Tooltip title="Refresh data">
              <IconButton onClick={handleRefresh} disabled={loading}>
                <Refresh sx={{ animation: loading ? 'spin 1s linear infinite' : 'none' }} />
              </IconButton>
            </Tooltip>
            
            <Tooltip title="Dashboard settings">
              <IconButton>
                <Settings />
              </IconButton>
            </Tooltip>
            
            <Tooltip title="Notifications">
              <IconButton>
                <Notifications />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>

        {/* Quick Status Alert */}
        <Alert 
          severity="info" 
          sx={{ mb: 3 }}
          action={
            <Button color="inherit" size="small" startIcon={<TrendingUp />}>
              View Details
            </Button>
          }
        >
          <Box display="flex" alignItems="center" gap={2}>
            <Typography variant="body2">
              All systems operational •
            </Typography>
            <Chip label="2 Bots Running" size="small" color="success" />
            <Chip label="Low Risk" size="small" color="info" />
            <Chip label="+2.01% Today" size="small" color="success" />
          </Box>
        </Alert>

        {/* Metrics Grid */}
        <Grid container spacing={3} mb={4}>
          {portfolioMetrics.map((metric, index) => (
            <Grid item xs={12} sm={6} md={4} lg={3} key={metric.title}>
              <MetricCard
                {...metric}
                loading={loading}
                onRefresh={() => handleRefresh()}
              />
            </Grid>
          ))}
        </Grid>

        {/* Main Content Tabs */}
        <Paper sx={{ backgroundColor: 'background.paper' }}>
          <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tabs 
              value={tabValue} 
              onChange={handleTabChange}
              sx={{ px: 3, pt: 2 }}
            >
              <Tab icon={<Timeline />} label="Portfolio Performance" />
              <Tab icon={<TrendingUp />} label="Active Positions" />
              <Tab icon={<Speed />} label="Bot Activity" />
              <Tab icon={<Security />} label="Risk Overview" />
            </Tabs>
          </Box>

          {/* Portfolio Performance Tab */}
          <TabPanel value={tabValue} index={0}>
            <Box p={3}>
              <Grid container spacing={3}>
                <Grid item xs={12}>
                  <PriceChart
                    data={mockPriceData}
                    symbol="Portfolio Value"
                    height={500}
                    loading={loading}
                    onRefresh={handleRefresh}
                    chartType="area"
                  />
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <Paper sx={{ p: 3 }}>
                    <Typography variant="h6" gutterBottom>
                      Trading Statistics
                    </Typography>
                    <Box display="flex" flexDirection="column" gap={2}>
                      <Box display="flex" justifyContent="space-between">
                        <Typography color="text.secondary">Total Trades</Typography>
                        <Typography sx={{ fontFamily: 'monospace' }}>
                          {mockPortfolioData.totalTrades.toLocaleString()}
                        </Typography>
                      </Box>
                      <Box display="flex" justifyContent="space-between">
                        <Typography color="text.secondary">Win Rate</Typography>
                        <Typography sx={{ fontFamily: 'monospace' }}>
                          {mockPortfolioData.winRate}%
                        </Typography>
                      </Box>
                      <Box display="flex" justifyContent="space-between">
                        <Typography color="text.secondary">Avg Trade Return</Typography>
                        <Typography sx={{ fontFamily: 'monospace' }}>
                          {mockPortfolioData.avgTradeReturn}%
                        </Typography>
                      </Box>
                    </Box>
                  </Paper>
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <Paper sx={{ p: 3 }}>
                    <Typography variant="h6" gutterBottom>
                      Risk Metrics
                    </Typography>
                    <Box display="flex" flexDirection="column" gap={2}>
                      <Box display="flex" justifyContent="space-between">
                        <Typography color="text.secondary">Value at Risk (1d)</Typography>
                        <Typography sx={{ fontFamily: 'monospace', color: 'error.main' }}>
                          -$234.56
                        </Typography>
                      </Box>
                      <Box display="flex" justifyContent="space-between">
                        <Typography color="text.secondary">Beta</Typography>
                        <Typography sx={{ fontFamily: 'monospace' }}>
                          0.85
                        </Typography>
                      </Box>
                      <Box display="flex" justifyContent="space-between">
                        <Typography color="text.secondary">Correlation (BTC)</Typography>
                        <Typography sx={{ fontFamily: 'monospace' }}>
                          0.72
                        </Typography>
                      </Box>
                    </Box>
                  </Paper>
                </Grid>
              </Grid>
            </Box>
          </TabPanel>

          {/* Active Positions Tab */}
          <TabPanel value={tabValue} index={1}>
            <Box p={3}>
              <Grid container spacing={3}>
                {mockPositions.map((position) => (
                  <Grid item xs={12} md={6} lg={4} key={position.id}>
                    <PositionCard
                      position={position}
                      onClose={(id) => console.log('Close position:', id)}
                      onEdit={(id) => console.log('Edit position:', id)}
                    />
                  </Grid>
                ))}
                
                {mockPositions.length === 0 && (
                  <Grid item xs={12}>
                    <Box
                      display="flex"
                      flexDirection="column"
                      alignItems="center"
                      justifyContent="center"
                      minHeight={300}
                      sx={{ backgroundColor: colors.background.tertiary, borderRadius: 2 }}
                    >
                      <TrendingUp sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
                      <Typography variant="h6" color="text.secondary" gutterBottom>
                        No Active Positions
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Start a bot or place a manual trade to see positions here
                      </Typography>
                    </Box>
                  </Grid>
                )}
              </Grid>
            </Box>
          </TabPanel>

          {/* Bot Activity Tab */}
          <TabPanel value={tabValue} index={2}>
            <Box p={3}>
              <Typography variant="h6" gutterBottom>
                Bot Activity Monitor
              </Typography>
              <Typography color="text.secondary">
                Bot activity monitoring will be implemented here
              </Typography>
            </Box>
          </TabPanel>

          {/* Risk Overview Tab */}
          <TabPanel value={tabValue} index={3}>
            <Box p={3}>
              <Typography variant="h6" gutterBottom>
                Risk Management Overview
              </Typography>
              <Typography color="text.secondary">
                Risk monitoring and circuit breakers will be implemented here
              </Typography>
            </Box>
          </TabPanel>
        </Paper>
      </Box>
    </motion.div>
  );
};

export default DashboardPage;