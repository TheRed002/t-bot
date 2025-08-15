/**
 * Results Analysis Component for Playground
 * Comprehensive analysis of backtest results and performance metrics
 */

import React, { useState, useCallback, useMemo } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  Card,
  CardContent,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button,
  IconButton,
  Tooltip,
  Chip,
  Alert,
  Tabs,
  Tab,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  useTheme,
  alpha,
  LinearProgress
} from '@mui/material';
import {
  Download as DownloadIcon,
  Compare as CompareIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Assessment as AssessmentIcon,
  Timeline as TimelineIcon,
  PieChart as PieChartIcon,
  BarChart as BarChartIcon,
  ShowChart as ShowChartIcon,
  TableChart as TableChartIcon,
  Star as StarIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as ChartTooltip,
  Legend,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  ComposedChart
} from 'recharts';

import { PlaygroundExecution, PlaygroundMetrics, PlaygroundTrade } from '@/types';

interface ResultsAnalysisProps {
  executions: PlaygroundExecution[];
  comparisonExecutions: PlaygroundExecution[];
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

// Generate mock data for comprehensive analysis
const generateMockAnalysisData = () => {
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const monthlyReturns = months.map(month => ({
    month,
    returns: (Math.random() - 0.5) * 20,
    volatility: Math.random() * 15 + 5,
    trades: Math.floor(Math.random() * 50) + 10
  }));

  const drawdownData = Array.from({ length: 30 }, (_, i) => ({
    day: i + 1,
    drawdown: Math.random() * 10,
    underwater: Math.random() * 15
  }));

  const tradeDistribution = [
    { name: 'Winning Trades', value: 65, color: '#4caf50' },
    { name: 'Losing Trades', value: 35, color: '#f44336' }
  ];

  const riskMetrics = {
    sharpeRatio: 1.45,
    sortinoRatio: 1.82,
    calmarRatio: 1.23,
    maxDrawdown: 8.5,
    var95: 2.3,
    var99: 3.8,
    beta: 0.85,
    alpha: 0.04,
    informationRatio: 1.12,
    treynorRatio: 0.156
  };

  return { monthlyReturns, drawdownData, tradeDistribution, riskMetrics };
};

const ResultsAnalysis: React.FC<ResultsAnalysisProps> = ({
  executions,
  comparisonExecutions
}) => {
  const theme = useTheme();
  const [selectedTab, setSelectedTab] = useState(0);
  const [selectedExecution, setSelectedExecution] = useState<PlaygroundExecution | null>(
    executions.length > 0 ? executions[0] : null
  );
  const [comparisonMode, setComparisonMode] = useState<boolean>(false);
  const [selectedMetric, setSelectedMetric] = useState<string>('totalReturn');

  const mockData = useMemo(() => generateMockAnalysisData(), []);

  const handleTabChange = useCallback((event: React.SyntheticEvent, newValue: number) => {
    setSelectedTab(newValue);
  }, []);

  const handleExecutionChange = useCallback((executionId: string) => {
    const execution = executions.find(e => e.id === executionId);
    setSelectedExecution(execution || null);
  }, [executions]);

  // Calculate performance grade
  const getPerformanceGrade = (metrics: PlaygroundMetrics) => {
    let score = 0;
    
    if (metrics.sharpeRatio > 1.5) score += 20;
    else if (metrics.sharpeRatio > 1.0) score += 15;
    else if (metrics.sharpeRatio > 0.5) score += 10;
    
    if (metrics.maxDrawdown < 10) score += 20;
    else if (metrics.maxDrawdown < 20) score += 15;
    else if (metrics.maxDrawdown < 30) score += 10;
    
    if (metrics.winRate > 60) score += 20;
    else if (metrics.winRate > 50) score += 15;
    else if (metrics.winRate > 40) score += 10;
    
    if (metrics.totalReturn > 20) score += 20;
    else if (metrics.totalReturn > 10) score += 15;
    else if (metrics.totalReturn > 0) score += 10;
    
    if (metrics.volatility < 15) score += 20;
    else if (metrics.volatility < 25) score += 15;
    else if (metrics.volatility < 35) score += 10;
    
    if (score >= 85) return { grade: 'A+', color: '#4caf50' };
    if (score >= 75) return { grade: 'A', color: '#66bb6a' };
    if (score >= 65) return { grade: 'B+', color: '#9ccc65' };
    if (score >= 55) return { grade: 'B', color: '#ffca28' };
    if (score >= 45) return { grade: 'C', color: '#ff9800' };
    return { grade: 'D', color: '#f44336' };
  };

  // Format percentage values
  const formatPercent = (value: number) => `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;

  // Format currency values
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value);
  };

  const completedExecutions = executions.filter(e => e.status === 'completed');

  if (completedExecutions.length === 0) {
    return (
      <Paper elevation={2} sx={{ p: 4, textAlign: 'center', minHeight: 400 }}>
        <AssessmentIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
        <Typography variant="h5" color="text.secondary" gutterBottom>
          No Completed Executions
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Complete at least one execution to view detailed analysis and performance metrics.
        </Typography>
        <Button
          variant="outlined"
          startIcon={<TimelineIcon />}
          sx={{ mt: 2 }}
          disabled
        >
          Run Backtest First
        </Button>
      </Paper>
    );
  }

  return (
    <Box>
      {/* Header Controls */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h5" fontWeight="bold">
          Results Analysis
        </Typography>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <FormControl size="small" sx={{ minWidth: 200 }}>
            <InputLabel>Select Execution</InputLabel>
            <Select
              value={selectedExecution?.id || ''}
              label="Select Execution"
              onChange={(e) => handleExecutionChange(e.target.value)}
            >
              {completedExecutions.map((execution) => (
                <MenuItem key={execution.id} value={execution.id}>
                  <Box>
                    <Typography variant="body2">
                      {execution.mode} - {new Date(execution.startTime!).toLocaleDateString()}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {execution.metrics ? formatPercent(execution.metrics.totalReturn) : 'No metrics'}
                    </Typography>
                  </Box>
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <Button
            variant="outlined"
            startIcon={<CompareIcon />}
            onClick={() => setComparisonMode(!comparisonMode)}
          >
            Compare
          </Button>
          <Button
            variant="outlined"
            startIcon={<DownloadIcon />}
          >
            Export
          </Button>
        </Box>
      </Box>

      {selectedExecution && (
        <>
          {/* Performance Overview Cards */}
          <Grid container spacing={3} sx={{ mb: 3 }}>
            <Grid item xs={12} md={8}>
              <Grid container spacing={2}>
                <Grid item xs={6} sm={3}>
                  <Card elevation={2}>
                    <CardContent sx={{ textAlign: 'center' }}>
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        Total Return
                      </Typography>
                      <Typography
                        variant="h5"
                        fontWeight="bold"
                        color={selectedExecution.metrics!.totalReturn >= 0 ? 'success.main' : 'error.main'}
                      >
                        {formatPercent(selectedExecution.metrics!.totalReturn)}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Annualized: {formatPercent(selectedExecution.metrics!.annualizedReturn)}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>

                <Grid item xs={6} sm={3}>
                  <Card elevation={2}>
                    <CardContent sx={{ textAlign: 'center' }}>
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        Sharpe Ratio
                      </Typography>
                      <Typography variant="h5" fontWeight="bold">
                        {selectedExecution.metrics!.sharpeRatio.toFixed(2)}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Risk-adjusted
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>

                <Grid item xs={6} sm={3}>
                  <Card elevation={2}>
                    <CardContent sx={{ textAlign: 'center' }}>
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        Max Drawdown
                      </Typography>
                      <Typography variant="h5" fontWeight="bold" color="error.main">
                        -{selectedExecution.metrics!.maxDrawdown.toFixed(2)}%
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Peak to trough
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>

                <Grid item xs={6} sm={3}>
                  <Card elevation={2}>
                    <CardContent sx={{ textAlign: 'center' }}>
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        Win Rate
                      </Typography>
                      <Typography variant="h5" fontWeight="bold">
                        {selectedExecution.metrics!.winRate.toFixed(1)}%
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {selectedExecution.metrics!.totalTrades} trades
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
            </Grid>

            <Grid item xs={12} md={4}>
              <Card elevation={2} sx={{ height: '100%' }}>
                <CardContent sx={{ textAlign: 'center', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                  <Typography variant="h6" gutterBottom>
                    Performance Grade
                  </Typography>
                  <Typography
                    variant="h2"
                    fontWeight="bold"
                    sx={{ color: getPerformanceGrade(selectedExecution.metrics!).color, mb: 1 }}
                  >
                    {getPerformanceGrade(selectedExecution.metrics!).grade}
                  </Typography>
                  <Box sx={{ display: 'flex', justifyContent: 'center', gap: 1 }}>
                    {Array.from({ length: 5 }, (_, i) => (
                      <StarIcon
                        key={i}
                        sx={{
                          color: i < Math.ceil(85 / 20) ? '#ffca28' : 'grey.300',
                          fontSize: 20
                        }}
                      />
                    ))}
                  </Box>
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                    Based on risk-adjusted returns
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          {/* Detailed Analysis Tabs */}
          <Paper elevation={2}>
            <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
              <Tabs value={selectedTab} onChange={handleTabChange}>
                <Tab icon={<TimelineIcon />} label="Performance" iconPosition="start" />
                <Tab icon={<BarChartIcon />} label="Risk Analysis" iconPosition="start" />
                <Tab icon={<PieChartIcon />} label="Trade Analysis" iconPosition="start" />
                <Tab icon={<TableChartIcon />} label="Detailed Stats" iconPosition="start" />
                <Tab icon={<AssessmentIcon />} label="Benchmark" iconPosition="start" />
              </Tabs>
            </Box>

            {/* Performance Tab */}
            <TabPanel value={selectedTab} index={0}>
              <Grid container spacing={3}>
                {/* Monthly Returns Chart */}
                <Grid item xs={12} lg={8}>
                  <Typography variant="h6" gutterBottom>
                    Monthly Returns
                  </Typography>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={mockData.monthlyReturns}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="month" />
                      <YAxis tickFormatter={(value) => `${value}%`} />
                      <ChartTooltip formatter={(value) => [`${value}%`, 'Return']} />
                      <Bar
                        dataKey="returns"
                        fill={(datum: any) => datum.returns >= 0 ? theme.palette.success.main : theme.palette.error.main}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </Grid>

                {/* Performance Metrics */}
                <Grid item xs={12} lg={4}>
                  <Typography variant="h6" gutterBottom>
                    Key Metrics
                  </Typography>
                  <List>
                    <ListItem>
                      <ListItemText
                        primary="Final Balance"
                        secondary={formatCurrency(selectedExecution.metrics!.finalBalance)}
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText
                        primary="Peak Balance"
                        secondary={formatCurrency(selectedExecution.metrics!.peakBalance)}
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText
                        primary="Profit Factor"
                        secondary={selectedExecution.metrics!.profitFactor.toFixed(2)}
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText
                        primary="Avg Trade Size"
                        secondary={formatCurrency(selectedExecution.metrics!.avgTradeSize)}
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText
                        primary="Avg Holding Period"
                        secondary={`${selectedExecution.metrics!.avgHoldingPeriod} hours`}
                      />
                    </ListItem>
                  </List>
                </Grid>

                {/* Cumulative Returns Chart */}
                <Grid item xs={12}>
                  <Typography variant="h6" gutterBottom>
                    Cumulative Returns vs Drawdown
                  </Typography>
                  <ResponsiveContainer width="100%" height={300}>
                    <ComposedChart data={mockData.drawdownData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="day" />
                      <YAxis yAxisId="left" orientation="left" />
                      <YAxis yAxisId="right" orientation="right" />
                      <ChartTooltip />
                      <Area
                        yAxisId="right"
                        type="monotone"
                        dataKey="underwater"
                        fill={alpha(theme.palette.error.main, 0.3)}
                        stroke={theme.palette.error.main}
                        name="Underwater Curve (%)"
                      />
                      <Line
                        yAxisId="left"
                        type="monotone"
                        dataKey="drawdown"
                        stroke={theme.palette.primary.main}
                        strokeWidth={2}
                        name="Drawdown (%)"
                      />
                    </ComposedChart>
                  </ResponsiveContainer>
                </Grid>
              </Grid>
            </TabPanel>

            {/* Risk Analysis Tab */}
            <TabPanel value={selectedTab} index={1}>
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <Typography variant="h6" gutterBottom>
                    Risk Metrics
                  </Typography>
                  <TableContainer>
                    <Table>
                      <TableBody>
                        <TableRow>
                          <TableCell>Sharpe Ratio</TableCell>
                          <TableCell align="right" fontWeight="bold">
                            {mockData.riskMetrics.sharpeRatio}
                          </TableCell>
                          <TableCell>
                            {mockData.riskMetrics.sharpeRatio > 1 ? (
                              <CheckCircleIcon color="success" />
                            ) : (
                              <WarningIcon color="warning" />
                            )}
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Sortino Ratio</TableCell>
                          <TableCell align="right" fontWeight="bold">
                            {mockData.riskMetrics.sortinoRatio}
                          </TableCell>
                          <TableCell>
                            <CheckCircleIcon color="success" />
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Calmar Ratio</TableCell>
                          <TableCell align="right" fontWeight="bold">
                            {mockData.riskMetrics.calmarRatio}
                          </TableCell>
                          <TableCell>
                            <CheckCircleIcon color="success" />
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>VaR (95%)</TableCell>
                          <TableCell align="right" fontWeight="bold">
                            {mockData.riskMetrics.var95}%
                          </TableCell>
                          <TableCell>
                            <CheckCircleIcon color="success" />
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>VaR (99%)</TableCell>
                          <TableCell align="right" fontWeight="bold">
                            {mockData.riskMetrics.var99}%
                          </TableCell>
                          <TableCell>
                            <CheckCircleIcon color="success" />
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Beta</TableCell>
                          <TableCell align="right" fontWeight="bold">
                            {mockData.riskMetrics.beta}
                          </TableCell>
                          <TableCell>
                            <CheckCircleIcon color="success" />
                          </TableCell>
                        </TableRow>
                      </TableBody>
                    </Table>
                  </TableContainer>
                </Grid>

                <Grid item xs={12} md={6}>
                  <Typography variant="h6" gutterBottom>
                    Risk Distribution
                  </Typography>
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={mockData.tradeDistribution}
                        cx="50%"
                        cy="50%"
                        innerRadius={60}
                        outerRadius={100}
                        paddingAngle={5}
                        dataKey="value"
                      >
                        {mockData.tradeDistribution.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <ChartTooltip formatter={(value) => [`${value}%`, 'Percentage']} />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                </Grid>

                <Grid item xs={12}>
                  <Alert severity="info">
                    <Typography variant="body2">
                      <strong>Risk Assessment:</strong> The strategy shows good risk-adjusted returns with a Sharpe ratio of {mockData.riskMetrics.sharpeRatio}.
                      Maximum drawdown of {selectedExecution.metrics!.maxDrawdown.toFixed(1)}% is within acceptable limits.
                      Consider reducing position sizes if VaR exceeds your risk tolerance.
                    </Typography>
                  </Alert>
                </Grid>
              </Grid>
            </TabPanel>

            {/* Trade Analysis Tab */}
            <TabPanel value={selectedTab} index={2}>
              <Grid container spacing={3}>
                <Grid item xs={12} md={8}>
                  <Typography variant="h6" gutterBottom>
                    Trade Distribution
                  </Typography>
                  <ResponsiveContainer width="100%" height={300}>
                    <ScatterChart data={selectedExecution.trades?.map(trade => ({
                      duration: Math.random() * 24,
                      pnl: trade.pnl || 0,
                      size: trade.quantity
                    }))}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="duration" name="Duration (hours)" />
                      <YAxis dataKey="pnl" name="P&L ($)" />
                      <ChartTooltip
                        formatter={(value, name) => [
                          name === 'pnl' ? formatCurrency(value as number) : `${value} hrs`,
                          name === 'pnl' ? 'P&L' : 'Duration'
                        ]}
                      />
                      <Scatter
                        dataKey="pnl"
                        fill={theme.palette.primary.main}
                      />
                    </ScatterChart>
                  </ResponsiveContainer>
                </Grid>

                <Grid item xs={12} md={4}>
                  <Typography variant="h6" gutterBottom>
                    Trade Statistics
                  </Typography>
                  <List>
                    <ListItem>
                      <ListItemIcon>
                        <TrendingUpIcon color="success" />
                      </ListItemIcon>
                      <ListItemText
                        primary="Winning Trades"
                        secondary={`${Math.round(selectedExecution.metrics!.winRate)}% (${Math.round(selectedExecution.metrics!.totalTrades * selectedExecution.metrics!.winRate / 100)} trades)`}
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon>
                        <TrendingDownIcon color="error" />
                      </ListItemIcon>
                      <ListItemText
                        primary="Losing Trades"
                        secondary={`${Math.round(100 - selectedExecution.metrics!.winRate)}% (${Math.round(selectedExecution.metrics!.totalTrades * (100 - selectedExecution.metrics!.winRate) / 100)} trades)`}
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText
                        primary="Largest Winner"
                        secondary={formatCurrency(1250)} // Mock data
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText
                        primary="Largest Loser"
                        secondary={formatCurrency(-450)} // Mock data
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText
                        primary="Average Winner"
                        secondary={formatCurrency(185)} // Mock data
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText
                        primary="Average Loser"
                        secondary={formatCurrency(-95)} // Mock data
                      />
                    </ListItem>
                  </List>
                </Grid>

                <Grid item xs={12}>
                  <Typography variant="h6" gutterBottom>
                    Recent Trades
                  </Typography>
                  <TableContainer>
                    <Table>
                      <TableHead>
                        <TableRow>
                          <TableCell>Symbol</TableCell>
                          <TableCell>Side</TableCell>
                          <TableCell align="right">Quantity</TableCell>
                          <TableCell align="right">Price</TableCell>
                          <TableCell align="right">P&L</TableCell>
                          <TableCell align="right">Commission</TableCell>
                          <TableCell>Timestamp</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {selectedExecution.trades?.slice(0, 10).map((trade) => (
                          <TableRow key={trade.id}>
                            <TableCell>{trade.symbol}</TableCell>
                            <TableCell>
                              <Chip
                                label={trade.side.toUpperCase()}
                                color={trade.side === 'buy' ? 'success' : 'error'}
                                variant="outlined"
                                size="small"
                              />
                            </TableCell>
                            <TableCell align="right">{trade.quantity}</TableCell>
                            <TableCell align="right">{formatCurrency(trade.price)}</TableCell>
                            <TableCell
                              align="right"
                              sx={{
                                color: trade.pnl && trade.pnl >= 0 ? 'success.main' : 'error.main',
                                fontWeight: 'medium'
                              }}
                            >
                              {trade.pnl ? formatCurrency(trade.pnl) : 'Pending'}
                            </TableCell>
                            <TableCell align="right">{formatCurrency(trade.commission)}</TableCell>
                            <TableCell>
                              {new Date(trade.timestamp).toLocaleDateString()}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </Grid>
              </Grid>
            </TabPanel>

            {/* Detailed Stats Tab */}
            <TabPanel value={selectedTab} index={3}>
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <Typography variant="h6" gutterBottom>
                    Performance Statistics
                  </Typography>
                  <TableContainer>
                    <Table size="small">
                      <TableBody>
                        <TableRow>
                          <TableCell>Total Return</TableCell>
                          <TableCell align="right">
                            {formatPercent(selectedExecution.metrics!.totalReturn)}
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Annualized Return</TableCell>
                          <TableCell align="right">
                            {formatPercent(selectedExecution.metrics!.annualizedReturn)}
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Volatility</TableCell>
                          <TableCell align="right">
                            {selectedExecution.metrics!.volatility.toFixed(2)}%
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Sharpe Ratio</TableCell>
                          <TableCell align="right">
                            {selectedExecution.metrics!.sharpeRatio.toFixed(2)}
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Sortino Ratio</TableCell>
                          <TableCell align="right">
                            {selectedExecution.metrics!.sortinoRatio.toFixed(2)}
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Maximum Drawdown</TableCell>
                          <TableCell align="right">
                            -{selectedExecution.metrics!.maxDrawdown.toFixed(2)}%
                          </TableCell>
                        </TableRow>
                      </TableBody>
                    </Table>
                  </TableContainer>
                </Grid>

                <Grid item xs={12} md={6}>
                  <Typography variant="h6" gutterBottom>
                    Trading Statistics
                  </Typography>
                  <TableContainer>
                    <Table size="small">
                      <TableBody>
                        <TableRow>
                          <TableCell>Total Trades</TableCell>
                          <TableCell align="right">
                            {selectedExecution.metrics!.totalTrades}
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Win Rate</TableCell>
                          <TableCell align="right">
                            {selectedExecution.metrics!.winRate.toFixed(1)}%
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Profit Factor</TableCell>
                          <TableCell align="right">
                            {selectedExecution.metrics!.profitFactor.toFixed(2)}
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Average Trade Size</TableCell>
                          <TableCell align="right">
                            {formatCurrency(selectedExecution.metrics!.avgTradeSize)}
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Average Holding Period</TableCell>
                          <TableCell align="right">
                            {selectedExecution.metrics!.avgHoldingPeriod.toFixed(1)} hours
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Final Balance</TableCell>
                          <TableCell align="right">
                            {formatCurrency(selectedExecution.metrics!.finalBalance)}
                          </TableCell>
                        </TableRow>
                      </TableBody>
                    </Table>
                  </TableContainer>
                </Grid>
              </Grid>
            </TabPanel>

            {/* Benchmark Tab */}
            <TabPanel value={selectedTab} index={4}>
              <Grid container spacing={3}>
                <Grid item xs={12}>
                  <Alert severity="info">
                    <Typography variant="h6" gutterBottom>
                      Benchmark Comparison
                    </Typography>
                    <Typography variant="body1">
                      Your strategy outperformed the benchmark (BTC Buy & Hold) by{' '}
                      <strong>
                        {selectedExecution.metrics?.benchmark
                          ? formatPercent(selectedExecution.metrics.totalReturn - selectedExecution.metrics.benchmark.return)
                          : '+5.2%'
                        }
                      </strong>{' '}
                      with lower volatility and better risk-adjusted returns.
                    </Typography>
                  </Alert>
                </Grid>

                <Grid item xs={12} md={6}>
                  <Typography variant="h6" gutterBottom>
                    Strategy vs Benchmark
                  </Typography>
                  <TableContainer>
                    <Table>
                      <TableHead>
                        <TableRow>
                          <TableCell>Metric</TableCell>
                          <TableCell align="right">Strategy</TableCell>
                          <TableCell align="right">Benchmark</TableCell>
                          <TableCell align="right">Difference</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        <TableRow>
                          <TableCell>Total Return</TableCell>
                          <TableCell align="right">
                            {formatPercent(selectedExecution.metrics!.totalReturn)}
                          </TableCell>
                          <TableCell align="right">
                            {formatPercent(selectedExecution.metrics?.benchmark?.return || 12.5)}
                          </TableCell>
                          <TableCell
                            align="right"
                            sx={{
                              color: selectedExecution.metrics!.totalReturn > (selectedExecution.metrics?.benchmark?.return || 12.5)
                                ? 'success.main' : 'error.main'
                            }}
                          >
                            {formatPercent(selectedExecution.metrics!.totalReturn - (selectedExecution.metrics?.benchmark?.return || 12.5))}
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Volatility</TableCell>
                          <TableCell align="right">
                            {selectedExecution.metrics!.volatility.toFixed(2)}%
                          </TableCell>
                          <TableCell align="right">
                            {selectedExecution.metrics?.benchmark?.volatility?.toFixed(2) || '45.2'}%
                          </TableCell>
                          <TableCell
                            align="right"
                            sx={{
                              color: selectedExecution.metrics!.volatility < (selectedExecution.metrics?.benchmark?.volatility || 45.2)
                                ? 'success.main' : 'error.main'
                            }}
                          >
                            {(selectedExecution.metrics!.volatility - (selectedExecution.metrics?.benchmark?.volatility || 45.2)).toFixed(2)}%
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Sharpe Ratio</TableCell>
                          <TableCell align="right">
                            {selectedExecution.metrics!.sharpeRatio.toFixed(2)}
                          </TableCell>
                          <TableCell align="right">0.28</TableCell>
                          <TableCell align="right" sx={{ color: 'success.main' }}>
                            +{(selectedExecution.metrics!.sharpeRatio - 0.28).toFixed(2)}
                          </TableCell>
                        </TableRow>
                      </TableBody>
                    </Table>
                  </TableContainer>
                </Grid>

                <Grid item xs={12} md={6}>
                  <Typography variant="h6" gutterBottom>
                    Correlation Analysis
                  </Typography>
                  <Box sx={{ textAlign: 'center', py: 4 }}>
                    <Typography variant="h3" fontWeight="bold" color="primary">
                      {selectedExecution.metrics?.benchmark?.correlation?.toFixed(2) || '0.65'}
                    </Typography>
                    <Typography variant="body1" color="text.secondary">
                      Correlation with {selectedExecution.metrics?.benchmark?.symbol || 'BTC'}
                    </Typography>
                    <LinearProgress
                      variant="determinate"
                      value={(selectedExecution.metrics?.benchmark?.correlation || 0.65) * 100}
                      sx={{ mt: 2, height: 8, borderRadius: 4 }}
                    />
                    <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                      Lower correlation indicates better diversification
                    </Typography>
                  </Box>
                </Grid>
              </Grid>
            </TabPanel>
          </Paper>
        </>
      )}
    </Box>
  );
};

export default ResultsAnalysis;