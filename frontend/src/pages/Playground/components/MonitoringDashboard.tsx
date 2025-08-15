/**
 * Monitoring Dashboard Component for Playground
 * Real-time monitoring of execution status, logs, and performance metrics
 */

import React, { useState, useCallback, useEffect, useRef } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  Card,
  CardContent,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  InputAdornment,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  IconButton,
  Tooltip,
  Button,
  LinearProgress,
  Alert,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  useTheme,
  alpha,
  Switch,
  FormControlLabel
} from '@mui/material';
import {
  Search as SearchIcon,
  FilterList as FilterIcon,
  Refresh as RefreshIcon,
  Download as DownloadIcon,
  PlayArrow as PlayArrowIcon,
  Pause as PauseIcon,
  Error as ErrorIcon,
  Warning as WarningIcon,
  Info as InfoIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Timeline as TimelineIcon,
  Visibility as VisibilityIcon,
  VisibilityOff as VisibilityOffIcon,
  FullscreenExit as FullscreenExitIcon
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as ChartTooltip,
  ResponsiveContainer,
  AreaChart,
  Area
} from 'recharts';

import { PlaygroundConfiguration, PlaygroundExecution, PlaygroundLog, PlaygroundTrade } from '@/types';

interface MonitoringDashboardProps {
  execution: PlaygroundExecution | null;
  configuration: PlaygroundConfiguration | null;
}

// Mock data for demonstration
const generateMockEquityCurve = (numPoints: number = 100) => {
  const data = [];
  let equity = 10000;
  
  for (let i = 0; i < numPoints; i++) {
    const change = (Math.random() - 0.5) * 200; // Random change
    equity += change;
    data.push({
      timestamp: new Date(Date.now() - (numPoints - i) * 60000).toISOString(),
      equity: Math.max(equity, 5000), // Minimum equity
      drawdown: Math.max(0, 10000 - equity),
      returns: i === 0 ? 0 : ((equity - 10000) / 10000) * 100
    });
  }
  
  return data;
};

const MonitoringDashboard: React.FC<MonitoringDashboardProps> = ({
  execution,
  configuration
}) => {
  const theme = useTheme();
  const logsEndRef = useRef<HTMLDivElement>(null);

  // State management
  const [logFilter, setLogFilter] = useState<string>('');
  const [logLevelFilter, setLogLevelFilter] = useState<string>('all');
  const [logCategoryFilter, setLogCategoryFilter] = useState<string>('all');
  const [autoScrollLogs, setAutoScrollLogs] = useState<boolean>(true);
  const [showAdvancedMetrics, setShowAdvancedMetrics] = useState<boolean>(false);
  const [selectedTimeframe, setSelectedTimeframe] = useState<string>('1h');

  // Mock data for demonstration
  const [equityCurveData, setEquityCurveData] = useState(generateMockEquityCurve);
  const [recentTrades, setRecentTrades] = useState<PlaygroundTrade[]>([
    {
      id: '1',
      executionId: execution?.id || 'demo',
      symbol: 'BTC/USDT',
      side: 'buy',
      quantity: 0.1,
      price: 45000,
      timestamp: new Date().toISOString(),
      pnl: 150,
      commission: 4.5,
      reason: 'Trend following signal',
      confidence: 0.85
    },
    {
      id: '2',
      executionId: execution?.id || 'demo',
      symbol: 'ETH/USDT',
      side: 'sell',
      quantity: 2.5,
      price: 3200,
      timestamp: new Date(Date.now() - 300000).toISOString(),
      pnl: -50,
      commission: 8,
      reason: 'Stop loss triggered',
      confidence: 0.92
    }
  ]);

  // Auto-scroll logs to bottom
  useEffect(() => {
    if (autoScrollLogs && logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [execution?.logs, autoScrollLogs]);

  // Filter logs based on search criteria
  const filteredLogs = execution?.logs?.filter(log => {
    const matchesSearch = logFilter === '' || 
      log.message.toLowerCase().includes(logFilter.toLowerCase()) ||
      log.category.toLowerCase().includes(logFilter.toLowerCase());
    
    const matchesLevel = logLevelFilter === 'all' || log.level === logLevelFilter;
    const matchesCategory = logCategoryFilter === 'all' || log.category === logCategoryFilter;
    
    return matchesSearch && matchesLevel && matchesCategory;
  }) || [];

  // Get log icon based on level
  const getLogIcon = (level: string) => {
    switch (level) {
      case 'error': return <ErrorIcon color="error" />;
      case 'warning': return <WarningIcon color="warning" />;
      case 'info': return <InfoIcon color="info" />;
      default: return <InfoIcon />;
    }
  };

  // Get log color based on level
  const getLogColor = (level: string) => {
    switch (level) {
      case 'error': return theme.palette.error.main;
      case 'warning': return theme.palette.warning.main;
      case 'info': return theme.palette.info.main;
      default: return theme.palette.text.primary;
    }
  };

  // Format currency values
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value);
  };

  // Calculate performance metrics
  const currentEquity = equityCurveData[equityCurveData.length - 1]?.equity || 10000;
  const totalReturn = ((currentEquity - 10000) / 10000) * 100;
  const maxDrawdown = Math.max(...equityCurveData.map(d => d.drawdown));
  const maxDrawdownPercent = (maxDrawdown / 10000) * 100;

  if (!execution) {
    return (
      <Paper elevation={2} sx={{ p: 3, textAlign: 'center', minHeight: 400 }}>
        <Typography variant="h6" color="text.secondary" gutterBottom>
          No Active Execution
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Start an execution to monitor performance and view real-time data.
        </Typography>
      </Paper>
    );
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3, height: '100%' }}>
      {/* Performance Overview */}
      <Grid container spacing={2}>
        <Grid item xs={12} sm={6} md={3}>
          <Card elevation={2}>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Current Equity
              </Typography>
              <Typography variant="h5" fontWeight="bold">
                {formatCurrency(currentEquity)}
              </Typography>
              <Typography 
                variant="body2" 
                color={totalReturn >= 0 ? 'success.main' : 'error.main'}
                sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 0.5 }}
              >
                {totalReturn >= 0 ? <TrendingUpIcon fontSize="small" /> : <TrendingDownIcon fontSize="small" />}
                {totalReturn >= 0 ? '+' : ''}{totalReturn.toFixed(2)}%
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card elevation={2}>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Total Trades
              </Typography>
              <Typography variant="h5" fontWeight="bold">
                {execution.metrics?.totalTrades || 0}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Win Rate: {execution.metrics?.winRate.toFixed(1) || 0}%
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card elevation={2}>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Max Drawdown
              </Typography>
              <Typography variant="h5" fontWeight="bold" color="error.main">
                {maxDrawdownPercent.toFixed(2)}%
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {formatCurrency(maxDrawdown)}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card elevation={2}>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Sharpe Ratio
              </Typography>
              <Typography variant="h5" fontWeight="bold">
                {execution.metrics?.sharpeRatio?.toFixed(2) || 'N/A'}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Risk-adjusted return
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Execution Progress */}
      {execution.status === 'running' && (
        <Card elevation={2}>
          <CardContent>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">Execution Progress</Typography>
              <Chip 
                label={`${Math.round(execution.progress)}%`}
                color="primary"
                variant="filled"
              />
            </Box>
            <LinearProgress 
              variant="determinate" 
              value={execution.progress} 
              sx={{ height: 8, borderRadius: 4, mb: 1 }}
            />
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography variant="body2" color="text.secondary">
                Status: {execution.status.charAt(0).toUpperCase() + execution.status.slice(1)}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Duration: {execution.duration ? `${Math.round(execution.duration / 1000)}s` : 'N/A'}
              </Typography>
            </Box>
          </CardContent>
        </Card>
      )}

      {/* Main Content Grid */}
      <Grid container spacing={3} sx={{ flexGrow: 1 }}>
        {/* Equity Curve Chart */}
        <Grid item xs={12} lg={8}>
          <Card elevation={2} sx={{ height: 400 }}>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <TimelineIcon />
                  Equity Curve
                </Typography>
                <FormControl size="small" sx={{ minWidth: 120 }}>
                  <InputLabel>Timeframe</InputLabel>
                  <Select
                    value={selectedTimeframe}
                    label="Timeframe"
                    onChange={(e) => setSelectedTimeframe(e.target.value)}
                  >
                    <MenuItem value="1h">1 Hour</MenuItem>
                    <MenuItem value="4h">4 Hours</MenuItem>
                    <MenuItem value="1d">1 Day</MenuItem>
                    <MenuItem value="1w">1 Week</MenuItem>
                  </Select>
                </FormControl>
              </Box>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={equityCurveData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="timestamp" 
                    tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                  />
                  <YAxis tickFormatter={(value) => `$${value.toLocaleString()}`} />
                  <ChartTooltip 
                    formatter={(value, name) => [
                      name === 'equity' ? formatCurrency(value as number) : `${(value as number).toFixed(2)}%`,
                      name === 'equity' ? 'Equity' : 'Returns'
                    ]}
                    labelFormatter={(label) => new Date(label).toLocaleString()}
                  />
                  <Area
                    type="monotone"
                    dataKey="equity"
                    stroke={theme.palette.primary.main}
                    fill={alpha(theme.palette.primary.main, 0.3)}
                    strokeWidth={2}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Trades */}
        <Grid item xs={12} lg={4}>
          <Card elevation={2} sx={{ height: 400 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Recent Trades
              </Typography>
              <List sx={{ maxHeight: 300, overflow: 'auto' }}>
                {recentTrades.map((trade, index) => (
                  <React.Fragment key={trade.id}>
                    <ListItem>
                      <ListItemIcon>
                        {trade.side === 'buy' ? (
                          <TrendingUpIcon color="success" />
                        ) : (
                          <TrendingDownIcon color="error" />
                        )}
                      </ListItemIcon>
                      <ListItemText
                        primary={
                          <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                            <Typography variant="body2" fontWeight="medium">
                              {trade.symbol}
                            </Typography>
                            <Typography 
                              variant="body2" 
                              color={trade.pnl && trade.pnl >= 0 ? 'success.main' : 'error.main'}
                              fontWeight="medium"
                            >
                              {trade.pnl ? `${trade.pnl >= 0 ? '+' : ''}$${trade.pnl.toFixed(2)}` : 'Pending'}
                            </Typography>
                          </Box>
                        }
                        secondary={
                          <Box>
                            <Typography variant="caption" color="text.secondary">
                              {trade.side.toUpperCase()} {trade.quantity} @ ${trade.price.toLocaleString()}
                            </Typography>
                            <Typography variant="caption" display="block" color="text.secondary">
                              {new Date(trade.timestamp).toLocaleTimeString()}
                            </Typography>
                          </Box>
                        }
                      />
                    </ListItem>
                    {index < recentTrades.length - 1 && <Divider variant="inset" component="li" />}
                  </React.Fragment>
                ))}
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* Execution Logs */}
        <Grid item xs={12}>
          <Card elevation={2}>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">Execution Logs</Typography>
                <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={autoScrollLogs}
                        onChange={(e) => setAutoScrollLogs(e.target.checked)}
                        size="small"
                      />
                    }
                    label="Auto-scroll"
                  />
                  <Tooltip title="Download logs">
                    <IconButton size="small">
                      <DownloadIcon />
                    </IconButton>
                  </Tooltip>
                </Box>
              </Box>

              {/* Log Filters */}
              <Grid container spacing={2} sx={{ mb: 2 }}>
                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    size="small"
                    placeholder="Search logs..."
                    value={logFilter}
                    onChange={(e) => setLogFilter(e.target.value)}
                    InputProps={{
                      startAdornment: (
                        <InputAdornment position="start">
                          <SearchIcon />
                        </InputAdornment>
                      )
                    }}
                  />
                </Grid>
                <Grid item xs={12} md={3}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Level</InputLabel>
                    <Select
                      value={logLevelFilter}
                      label="Level"
                      onChange={(e) => setLogLevelFilter(e.target.value)}
                    >
                      <MenuItem value="all">All Levels</MenuItem>
                      <MenuItem value="debug">Debug</MenuItem>
                      <MenuItem value="info">Info</MenuItem>
                      <MenuItem value="warning">Warning</MenuItem>
                      <MenuItem value="error">Error</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12} md={3}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Category</InputLabel>
                    <Select
                      value={logCategoryFilter}
                      label="Category"
                      onChange={(e) => setLogCategoryFilter(e.target.value)}
                    >
                      <MenuItem value="all">All Categories</MenuItem>
                      <MenuItem value="strategy">Strategy</MenuItem>
                      <MenuItem value="risk">Risk</MenuItem>
                      <MenuItem value="execution">Execution</MenuItem>
                      <MenuItem value="data">Data</MenuItem>
                      <MenuItem value="system">System</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
              </Grid>

              {/* Log List */}
              <Box 
                sx={{ 
                  maxHeight: 300, 
                  overflow: 'auto',
                  border: 1,
                  borderColor: 'divider',
                  borderRadius: 1,
                  bgcolor: alpha(theme.palette.background.default, 0.5)
                }}
              >
                <List dense>
                  {filteredLogs.length === 0 ? (
                    <ListItem>
                      <ListItemText 
                        primary="No logs available"
                        secondary="Logs will appear here as the execution progresses"
                      />
                    </ListItem>
                  ) : (
                    filteredLogs.map((log) => (
                      <ListItem key={log.id}>
                        <ListItemIcon sx={{ minWidth: 32 }}>
                          {getLogIcon(log.level)}
                        </ListItemIcon>
                        <ListItemText
                          primary={
                            <Typography 
                              variant="body2" 
                              component="pre" 
                              sx={{ 
                                fontFamily: 'monospace', 
                                whiteSpace: 'pre-wrap',
                                color: getLogColor(log.level)
                              }}
                            >
                              [{new Date(log.timestamp).toLocaleTimeString()}] [{log.category.toUpperCase()}] {log.message}
                            </Typography>
                          }
                        />
                      </ListItem>
                    ))
                  )}
                  <div ref={logsEndRef} />
                </List>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default MonitoringDashboard;