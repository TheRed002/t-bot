/**
 * Bot Management page component
 * Comprehensive bot management interface with creation, monitoring, and control
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  Box,
  Typography,
  Grid,
  Paper,
  Button,
  IconButton,
  Tooltip,
  TextField,
  InputAdornment,
  ToggleButton,
  ToggleButtonGroup,
  Chip,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Tabs,
  Tab,
  Fab,
} from '@mui/material';
import {
  Add,
  Search,
  FilterList,
  Refresh,
  PlayArrow,
  Pause,
  Stop,
  TrendingUp,
  TrendingDown,
  Assessment,
  Settings,
  Speed,
  Security,
} from '@mui/icons-material';
import { motion } from 'framer-motion';
import { useAppSelector, useAppDispatch } from '@/store';
import { colors } from '@/theme/colors';

// Import bot components
import BotCard, { Bot } from '@/components/Bots/BotCard';
import MetricCard from '@/components/Trading/MetricCard';

// Mock data for development
const mockBots: Bot[] = [
  {
    id: '1',
    name: 'BTC Momentum Bot',
    strategy: 'Momentum',
    status: 'running',
    exchange: 'Binance',
    symbol: 'BTC/USD',
    balance: 5420.50,
    pnl: 342.18,
    pnlPercent: 6.74,
    trades: 127,
    winRate: 68.5,
    created: Date.now() - 86400000 * 7, // 7 days ago
    lastActivity: Date.now() - 300000, // 5 minutes ago
    riskLevel: 'medium',
    leverage: 10,
    maxDrawdown: -5.2,
    sharpeRatio: 1.85,
    description: 'Momentum-based trading strategy for BTC',
    version: '2.1.4',
    isActive: true,
  },
  {
    id: '2',
    name: 'ETH Mean Reversion',
    strategy: 'Mean Reversion',
    status: 'paused',
    exchange: 'Coinbase',
    symbol: 'ETH/USD',
    balance: 2850.00,
    pnl: -45.30,
    pnlPercent: -1.56,
    trades: 89,
    winRate: 72.1,
    created: Date.now() - 86400000 * 3, // 3 days ago
    lastActivity: Date.now() - 1800000, // 30 minutes ago
    riskLevel: 'low',
    leverage: 5,
    maxDrawdown: -3.1,
    sharpeRatio: 1.42,
    description: 'Mean reversion strategy for ETH',
    version: '1.8.2',
    isActive: true,
  },
  {
    id: '3',
    name: 'Multi-Pair Arbitrage',
    strategy: 'Arbitrage',
    status: 'stopped',
    exchange: 'OKX',
    symbol: 'MULTI',
    balance: 1200.00,
    pnl: 89.45,
    pnlPercent: 8.05,
    trades: 45,
    winRate: 95.6,
    created: Date.now() - 86400000 * 14, // 14 days ago
    lastActivity: Date.now() - 3600000, // 1 hour ago
    riskLevel: 'low',
    leverage: 3,
    maxDrawdown: -1.8,
    sharpeRatio: 2.34,
    description: 'Cross-exchange arbitrage opportunities',
    version: '3.0.1',
    isActive: false,
  },
  {
    id: '4',
    name: 'High-Freq Scalper',
    strategy: 'Scalping',
    status: 'error',
    exchange: 'Binance',
    symbol: 'ADA/USD',
    balance: 890.25,
    pnl: -125.60,
    pnlPercent: -12.35,
    trades: 2341,
    winRate: 58.3,
    created: Date.now() - 86400000 * 1, // 1 day ago
    lastActivity: Date.now() - 600000, // 10 minutes ago
    riskLevel: 'high',
    leverage: 20,
    maxDrawdown: -15.7,
    sharpeRatio: 0.87,
    description: 'High-frequency scalping strategy',
    version: '1.2.0',
    isActive: true,
  },
];

const strategies = [
  'All Strategies',
  'Momentum',
  'Mean Reversion',
  'Arbitrage',
  'Scalping',
  'Market Making',
  'Trend Following',
];

const exchanges = [
  'All Exchanges',
  'Binance',
  'Coinbase',
  'OKX',
  'Kraken',
];

const statusFilters = [
  'All Status',
  'running',
  'paused',
  'stopped',
  'error',
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

const BotManagementPage: React.FC = () => {
  const [bots, setBots] = useState<Bot[]>(mockBots);
  const [filteredBots, setFilteredBots] = useState<Bot[]>(mockBots);
  const [searchQuery, setSearchQuery] = useState('');
  const [strategyFilter, setStrategyFilter] = useState('All Strategies');
  const [exchangeFilter, setExchangeFilter] = useState('All Exchanges');
  const [statusFilter, setStatusFilter] = useState('All Status');
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [tabValue, setTabValue] = useState(0);
  const [createBotDialog, setCreateBotDialog] = useState(false);
  const [bulkActions, setBulkActions] = useState(false);
  const [selectedBots, setSelectedBots] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);

  // Calculate summary metrics
  const summaryMetrics = useMemo(() => {
    const totalBots = bots.length;
    const runningBots = bots.filter(bot => bot.status === 'running').length;
    const totalPnL = bots.reduce((sum, bot) => sum + bot.pnl, 0);
    const totalBalance = bots.reduce((sum, bot) => sum + bot.balance, 0);
    const avgWinRate = bots.reduce((sum, bot) => sum + bot.winRate, 0) / totalBots || 0;
    const totalTrades = bots.reduce((sum, bot) => sum + bot.trades, 0);

    return {
      totalBots,
      runningBots,
      totalPnL,
      totalBalance,
      avgWinRate,
      totalTrades,
    };
  }, [bots]);

  // Filter bots based on search and filters
  useEffect(() => {
    let filtered = bots;

    // Search filter
    if (searchQuery) {
      filtered = filtered.filter(bot =>
        bot.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        bot.strategy.toLowerCase().includes(searchQuery.toLowerCase()) ||
        bot.symbol.toLowerCase().includes(searchQuery.toLowerCase())
      );
    }

    // Strategy filter
    if (strategyFilter !== 'All Strategies') {
      filtered = filtered.filter(bot => bot.strategy === strategyFilter);
    }

    // Exchange filter
    if (exchangeFilter !== 'All Exchanges') {
      filtered = filtered.filter(bot => bot.exchange === exchangeFilter);
    }

    // Status filter
    if (statusFilter !== 'All Status') {
      filtered = filtered.filter(bot => bot.status === statusFilter);
    }

    setFilteredBots(filtered);
  }, [bots, searchQuery, strategyFilter, exchangeFilter, statusFilter]);

  const handleRefresh = useCallback(async () => {
    setLoading(true);
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1000));
    setLoading(false);
  }, []);

  const handleBotAction = useCallback((action: string, botId: string) => {
    setBots(prevBots =>
      prevBots.map(bot =>
        bot.id === botId
          ? { ...bot, status: action as any, lastActivity: Date.now() }
          : bot
      )
    );
  }, []);

  const handleBulkAction = useCallback((action: string) => {
    setBots(prevBots =>
      prevBots.map(bot =>
        selectedBots.includes(bot.id)
          ? { ...bot, status: action as any, lastActivity: Date.now() }
          : bot
      )
    );
    setSelectedBots([]);
  }, [selectedBots]);

  const handleCreateBot = useCallback(() => {
    setCreateBotDialog(true);
  }, []);

  const metrics = [
    {
      title: 'Total Bots',
      value: summaryMetrics.totalBots,
      subtitle: `${summaryMetrics.runningBots} running`,
      format: 'number' as const,
      precision: 0,
      tooltip: 'Total number of trading bots',
    },
    {
      title: 'Total Balance',
      value: summaryMetrics.totalBalance,
      format: 'currency' as const,
      tooltip: 'Combined balance across all bots',
    },
    {
      title: 'Total P&L',
      value: summaryMetrics.totalPnL,
      change: summaryMetrics.totalPnL,
      format: 'currency' as const,
      tooltip: 'Combined profit and loss',
      variant: summaryMetrics.totalPnL >= 0 ? 'highlighted' as const : 'default' as const,
    },
    {
      title: 'Avg Win Rate',
      value: summaryMetrics.avgWinRate,
      format: 'percentage' as const,
      precision: 1,
      tooltip: 'Average win rate across all bots',
    },
    {
      title: 'Total Trades',
      value: summaryMetrics.totalTrades,
      format: 'number' as const,
      precision: 0,
      tooltip: 'Total trades executed by all bots',
    },
  ];

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
              Bot Management
            </Typography>
            <Typography variant="body1" color="text.secondary">
              Monitor and control your automated trading bots
            </Typography>
          </Box>

          <Box display="flex" alignItems="center" gap={2}>
            <FormControlLabel
              control={
                <Switch
                  checked={bulkActions}
                  onChange={(e) => setBulkActions(e.target.checked)}
                  size="small"
                />
              }
              label="Bulk Actions"
            />

            <Tooltip title="Refresh data">
              <IconButton onClick={handleRefresh} disabled={loading}>
                <Refresh sx={{ animation: loading ? 'spin 1s linear infinite' : 'none' }} />
              </IconButton>
            </Tooltip>

            <Button
              variant="contained"
              startIcon={<Add />}
              onClick={handleCreateBot}
            >
              Create Bot
            </Button>
          </Box>
        </Box>

        {/* Summary Metrics */}
        <Grid container spacing={3} mb={4}>
          {metrics.map((metric, index) => (
            <Grid item xs={12} sm={6} md={4} lg={2.4} key={metric.title}>
              <MetricCard
                {...metric}
                loading={loading}
                size="small"
              />
            </Grid>
          ))}
        </Grid>

        {/* Bulk Actions Bar */}
        {bulkActions && selectedBots.length > 0 && (
          <Alert 
            severity="info" 
            sx={{ mb: 3 }}
            action={
              <Box display="flex" gap={1}>
                <Button
                  size="small"
                  startIcon={<PlayArrow />}
                  onClick={() => handleBulkAction('running')}
                >
                  Start All
                </Button>
                <Button
                  size="small"
                  startIcon={<Pause />}
                  onClick={() => handleBulkAction('paused')}
                >
                  Pause All
                </Button>
                <Button
                  size="small"
                  startIcon={<Stop />}
                  onClick={() => handleBulkAction('stopped')}
                >
                  Stop All
                </Button>
              </Box>
            }
          >
            {selectedBots.length} bot(s) selected
          </Alert>
        )}

        {/* Filters and Search */}
        <Paper sx={{ p: 3, mb: 3 }}>
          <Box display="flex" flexWrap="wrap" gap={2} alignItems="center" mb={2}>
            <TextField
              label="Search bots"
              variant="outlined"
              size="small"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <Search />
                  </InputAdornment>
                ),
              }}
              sx={{ minWidth: 200 }}
            />

            <FormControl size="small" sx={{ minWidth: 150 }}>
              <InputLabel>Strategy</InputLabel>
              <Select
                value={strategyFilter}
                onChange={(e) => setStrategyFilter(e.target.value)}
                label="Strategy"
              >
                {strategies.map((strategy) => (
                  <MenuItem key={strategy} value={strategy}>
                    {strategy}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <FormControl size="small" sx={{ minWidth: 150 }}>
              <InputLabel>Exchange</InputLabel>
              <Select
                value={exchangeFilter}
                onChange={(e) => setExchangeFilter(e.target.value)}
                label="Exchange"
              >
                {exchanges.map((exchange) => (
                  <MenuItem key={exchange} value={exchange}>
                    {exchange}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <FormControl size="small" sx={{ minWidth: 120 }}>
              <InputLabel>Status</InputLabel>
              <Select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                label="Status"
              >
                {statusFilters.map((status) => (
                  <MenuItem key={status} value={status}>
                    {status}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <ToggleButtonGroup
              value={viewMode}
              exclusive
              onChange={(_, value) => value && setViewMode(value)}
              size="small"
            >
              <ToggleButton value="grid">
                Grid
              </ToggleButton>
              <ToggleButton value="list">
                List
              </ToggleButton>
            </ToggleButtonGroup>
          </Box>

          <Typography variant="body2" color="text.secondary">
            Showing {filteredBots.length} of {bots.length} bots
          </Typography>
        </Paper>

        {/* Bot Grid/List */}
        <Box>
          {filteredBots.length > 0 ? (
            <Grid container spacing={3}>
              {filteredBots.map((bot) => (
                <Grid 
                  item 
                  xs={12} 
                  sm={viewMode === 'grid' ? 6 : 12} 
                  md={viewMode === 'grid' ? 4 : 12} 
                  lg={viewMode === 'grid' ? 3 : 12}
                  key={bot.id}
                >
                  <BotCard
                    bot={bot}
                    compact={viewMode === 'list'}
                    onStart={(id) => handleBotAction('running', id)}
                    onPause={(id) => handleBotAction('paused', id)}
                    onStop={(id) => handleBotAction('stopped', id)}
                    onEdit={(id) => console.log('Edit bot:', id)}
                    onDelete={(id) => console.log('Delete bot:', id)}
                    onToggleActive={(id, active) => console.log('Toggle bot:', id, active)}
                    onViewDetails={(id) => console.log('View bot details:', id)}
                  />
                </Grid>
              ))}
            </Grid>
          ) : (
            <Box
              display="flex"
              flexDirection="column"
              alignItems="center"
              justifyContent="center"
              minHeight={400}
              sx={{ backgroundColor: colors.background.tertiary, borderRadius: 2 }}
            >
              <Speed sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
              <Typography variant="h6" color="text.secondary" gutterBottom>
                No Bots Found
              </Typography>
              <Typography variant="body2" color="text.secondary" mb={3}>
                {searchQuery || strategyFilter !== 'All Strategies' || exchangeFilter !== 'All Exchanges' || statusFilter !== 'All Status'
                  ? 'Try adjusting your filters or search criteria'
                  : 'Create your first trading bot to get started'
                }
              </Typography>
              <Button
                variant="contained"
                startIcon={<Add />}
                onClick={handleCreateBot}
              >
                Create Your First Bot
              </Button>
            </Box>
          )}
        </Box>

        {/* Floating Action Button */}
        <Fab
          color="primary"
          aria-label="add"
          sx={{
            position: 'fixed',
            bottom: 16,
            right: 16,
          }}
          onClick={handleCreateBot}
        >
          <Add />
        </Fab>

        {/* Create Bot Dialog */}
        <Dialog 
          open={createBotDialog} 
          onClose={() => setCreateBotDialog(false)}
          maxWidth="md"
          fullWidth
        >
          <DialogTitle>Create New Trading Bot</DialogTitle>
          <DialogContent>
            <Typography color="text.secondary">
              Bot creation wizard will be implemented here
            </Typography>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setCreateBotDialog(false)}>Cancel</Button>
            <Button variant="contained">Create Bot</Button>
          </DialogActions>
        </Dialog>
      </Box>
    </motion.div>
  );
};

export default BotManagementPage;