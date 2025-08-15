/**
 * Batch Optimizer Component for Playground
 * Run multiple configurations to find optimal parameters and strategies
 */

import React, { useState, useCallback, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  Card,
  CardContent,
  Button,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Slider,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Tooltip,
  LinearProgress,
  Alert,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControlLabel,
  Switch,
  Checkbox,
  useTheme,
  alpha,
  Accordion,
  AccordionSummary,
  AccordionDetails
} from '@mui/material';
import {
  Speed as SpeedIcon,
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  Pause as PauseIcon,
  Refresh as RefreshIcon,
  Download as DownloadIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Assessment as AssessmentIcon,
  Science as ScienceIcon,
  Memory as MemoryIcon,
  Timer as TimerIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Warning as WarningIcon,
  ExpandMore as ExpandMoreIcon,
  Visibility as VisibilityIcon,
  Analytics as AnalyticsIcon
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as ChartTooltip,
  ResponsiveContainer,
  ReferenceLine
} from 'recharts';

import { PlaygroundBatch, PlaygroundConfiguration, PlaygroundMetrics } from '@/types';

// Generate mock batch results
const generateMockBatchResults = (configCount: number) => {
  return Array.from({ length: configCount }, (_, i) => ({
    configurationId: `config_${i + 1}`,
    name: `Configuration ${i + 1}`,
    metrics: {
      totalReturn: (Math.random() - 0.3) * 40, // -12% to +28%
      annualizedReturn: (Math.random() - 0.3) * 50,
      sharpeRatio: Math.random() * 3 + 0.1,
      sortinoRatio: Math.random() * 4 + 0.1,
      maxDrawdown: Math.random() * 25 + 2,
      volatility: Math.random() * 30 + 10,
      winRate: Math.random() * 40 + 30, // 30% to 70%
      profitFactor: Math.random() * 3 + 0.5,
      totalTrades: Math.floor(Math.random() * 500) + 50,
      avgTradeSize: Math.random() * 1000 + 100,
      avgHoldingPeriod: Math.random() * 48 + 1,
      finalBalance: 10000 + (Math.random() - 0.3) * 8000,
      peakBalance: 10000 + Math.random() * 12000
    } as PlaygroundMetrics,
    rank: 0, // Will be calculated
    parameters: {
      stopLoss: Math.random() * 4 + 1,
      takeProfit: Math.random() * 8 + 2,
      positionSize: Math.random() * 4 + 1,
      strategy: ['trend_following', 'mean_reversion', 'arbitrage'][Math.floor(Math.random() * 3)]
    },
    status: Math.random() > 0.1 ? 'completed' : Math.random() > 0.5 ? 'running' : 'error'
  }));
};

const BatchOptimizer: React.FC = () => {
  const theme = useTheme();

  // State management
  const [batchConfig, setBatchConfig] = useState({
    name: 'Optimization Batch',
    description: 'Find optimal parameters across multiple configurations',
    configurationCount: 50,
    optimizationMetric: 'sharpe_ratio',
    crossValidationFolds: 5,
    outOfSamplePercentage: 20,
    walkForwardWindows: 3,
    overfittingProtection: true,
    parallelExecution: true,
    maxConcurrentJobs: 5,
    resourceLimit: 80
  });

  const [parameterRanges, setParameterRanges] = useState({
    stopLossPercentage: { min: 1, max: 5, step: 0.1, enabled: true },
    takeProfitPercentage: { min: 2, max: 10, step: 0.1, enabled: true },
    positionSize: { min: 1, max: 5, step: 0.1, enabled: true },
    maxDrawdown: { min: 5, max: 25, step: 1, enabled: false }
  });

  const [selectedStrategies, setSelectedStrategies] = useState({
    trend_following: true,
    mean_reversion: true,
    arbitrage: false,
    market_making: false,
    ml_based: false
  });

  const [selectedSymbols, setSelectedSymbols] = useState({
    'BTC/USDT': true,
    'ETH/USDT': true,
    'ADA/USDT': false,
    'DOT/USDT': false,
    'LINK/USDT': false
  });

  const [currentBatch, setCurrentBatch] = useState<PlaygroundBatch | null>(null);
  const [batchResults, setBatchResults] = useState<any[]>([]);
  const [batchProgress, setBatchProgress] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [selectedResult, setSelectedResult] = useState<any>(null);
  const [advancedSettingsOpen, setAdvancedSettingsOpen] = useState(false);

  // Mock resource usage
  const [resourceUsage, setResourceUsage] = useState({
    cpu: 0,
    memory: 0,
    activeJobs: 0
  });

  // Start batch optimization
  const handleStartBatch = useCallback(() => {
    setIsRunning(true);
    setBatchProgress(0);
    setBatchResults([]);

    // Create batch configuration
    const batch: PlaygroundBatch = {
      id: `batch_${Date.now()}`,
      name: batchConfig.name,
      description: batchConfig.description,
      configurations: [], // Would be generated based on parameter ranges
      status: 'running',
      startTime: new Date().toISOString(),
      settings: {
        crossValidationFolds: batchConfig.crossValidationFolds,
        optimizationMetric: batchConfig.optimizationMetric as any,
        overfittingProtection: {
          enabled: batchConfig.overfittingProtection,
          walkForwardWindows: batchConfig.walkForwardWindows,
          outOfSamplePercentage: batchConfig.outOfSamplePercentage
        }
      }
    };

    setCurrentBatch(batch);

    // Simulate batch execution
    const totalConfigs = batchConfig.configurationCount;
    let completed = 0;

    const interval = setInterval(() => {
      completed += Math.floor(Math.random() * 3) + 1;
      
      if (completed >= totalConfigs) {
        completed = totalConfigs;
        clearInterval(interval);
        setIsRunning(false);
        
        // Generate and rank results
        const results = generateMockBatchResults(totalConfigs);
        const sortedResults = results
          .sort((a, b) => b.metrics.sharpeRatio - a.metrics.sharpeRatio)
          .map((result, index) => ({ ...result, rank: index + 1 }));
        
        setBatchResults(sortedResults);
        setCurrentBatch(prev => prev ? { ...prev, status: 'completed' } : null);
      }

      setBatchProgress((completed / totalConfigs) * 100);
      setResourceUsage({
        cpu: Math.random() * 60 + 20,
        memory: Math.random() * 50 + 30,
        activeJobs: Math.min(batchConfig.maxConcurrentJobs, totalConfigs - completed)
      });
    }, 500);
  }, [batchConfig]);

  // Stop batch optimization
  const handleStopBatch = useCallback(() => {
    setIsRunning(false);
    setCurrentBatch(prev => prev ? { ...prev, status: 'error' } : null);
  }, []);

  // Format metric values
  const formatMetric = (value: number, metric: string) => {
    switch (metric) {
      case 'totalReturn':
      case 'annualizedReturn':
      case 'maxDrawdown':
      case 'volatility':
      case 'winRate':
        return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
      case 'sharpeRatio':
      case 'sortinoRatio':
      case 'profitFactor':
        return value.toFixed(2);
      case 'totalTrades':
        return Math.round(value).toString();
      case 'avgTradeSize':
      case 'finalBalance':
      case 'peakBalance':
        return new Intl.NumberFormat('en-US', {
          style: 'currency',
          currency: 'USD'
        }).format(value);
      case 'avgHoldingPeriod':
        return `${value.toFixed(1)}h`;
      default:
        return value.toString();
    }
  };

  // Get rank color
  const getRankColor = (rank: number) => {
    if (rank === 1) return theme.palette.warning.main; // Gold
    if (rank === 2) return theme.palette.info.main;    // Silver
    if (rank === 3) return theme.palette.secondary.main; // Bronze
    if (rank <= 10) return theme.palette.success.main;
    if (rank <= 25) return theme.palette.primary.main;
    return theme.palette.text.secondary;
  };

  // Calculate overfitting risk
  const calculateOverfittingRisk = (metrics: PlaygroundMetrics) => {
    // Mock overfitting risk calculation
    let risk = 0;
    if (metrics.sharpeRatio > 3) risk += 0.3;
    if (metrics.winRate > 80) risk += 0.3;
    if (metrics.totalTrades < 30) risk += 0.4;
    
    if (risk > 0.7) return { level: 'High', color: 'error' };
    if (risk > 0.4) return { level: 'Medium', color: 'warning' };
    return { level: 'Low', color: 'success' };
  };

  return (
    <Box>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h5" fontWeight="bold">
          Batch Optimizer
        </Typography>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button
            variant="outlined"
            startIcon={<AnalyticsIcon />}
            onClick={() => setAdvancedSettingsOpen(true)}
            disabled={isRunning}
          >
            Advanced Settings
          </Button>
          {isRunning ? (
            <Button
              variant="outlined"
              color="error"
              startIcon={<StopIcon />}
              onClick={handleStopBatch}
            >
              Stop Batch
            </Button>
          ) : (
            <Button
              variant="contained"
              startIcon={<PlayIcon />}
              onClick={handleStartBatch}
              disabled={batchConfig.configurationCount === 0}
            >
              Start Batch
            </Button>
          )}
        </Box>
      </Box>

      <Grid container spacing={3}>
        {/* Configuration Panel */}
        <Grid item xs={12} lg={4}>
          <Card elevation={2}>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <ScienceIcon color="primary" />
                Batch Configuration
              </Typography>

              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    size="small"
                    label="Batch Name"
                    value={batchConfig.name}
                    onChange={(e) => setBatchConfig(prev => ({ ...prev, name: e.target.value }))}
                    disabled={isRunning}
                  />
                </Grid>

                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    size="small"
                    type="number"
                    label="Configuration Count"
                    value={batchConfig.configurationCount}
                    onChange={(e) => setBatchConfig(prev => ({ ...prev, configurationCount: Number(e.target.value) }))}
                    disabled={isRunning}
                    inputProps={{ min: 10, max: 1000 }}
                    helperText="Number of configurations to test"
                  />
                </Grid>

                <Grid item xs={12}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Optimization Metric</InputLabel>
                    <Select
                      value={batchConfig.optimizationMetric}
                      label="Optimization Metric"
                      onChange={(e) => setBatchConfig(prev => ({ ...prev, optimizationMetric: e.target.value }))}
                      disabled={isRunning}
                    >
                      <MenuItem value="sharpe_ratio">Sharpe Ratio</MenuItem>
                      <MenuItem value="return">Total Return</MenuItem>
                      <MenuItem value="calmar_ratio">Calmar Ratio</MenuItem>
                      <MenuItem value="sortino_ratio">Sortino Ratio</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>

                <Grid item xs={12}>
                  <Typography variant="body2" gutterBottom>
                    Max Concurrent Jobs: {batchConfig.maxConcurrentJobs}
                  </Typography>
                  <Slider
                    value={batchConfig.maxConcurrentJobs}
                    onChange={(_, value) => setBatchConfig(prev => ({ ...prev, maxConcurrentJobs: value as number }))}
                    min={1}
                    max={10}
                    step={1}
                    marks
                    disabled={isRunning}
                  />
                </Grid>

                <Grid item xs={12}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={batchConfig.overfittingProtection}
                        onChange={(e) => setBatchConfig(prev => ({ ...prev, overfittingProtection: e.target.checked }))}
                        disabled={isRunning}
                      />
                    }
                    label="Overfitting Protection"
                  />
                </Grid>
              </Grid>

              {/* Parameter Ranges */}
              <Typography variant="subtitle1" sx={{ mt: 3, mb: 2 }}>
                Parameter Ranges
              </Typography>
              {Object.entries(parameterRanges).map(([param, range]) => (
                <Box key={param} sx={{ mb: 2 }}>
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={range.enabled}
                        onChange={(e) => setParameterRanges(prev => ({
                          ...prev,
                          [param]: { ...prev[param as keyof typeof prev], enabled: e.target.checked }
                        }))}
                        disabled={isRunning}
                      />
                    }
                    label={param}
                  />
                  {range.enabled && (
                    <Box sx={{ px: 2 }}>
                      <Typography variant="caption" gutterBottom display="block">
                        Range: {range.min} - {range.max}
                      </Typography>
                      <Slider
                        value={[range.min, range.max]}
                        onChange={(_, value) => {
                          const [min, max] = value as number[];
                          setParameterRanges(prev => ({
                            ...prev,
                            [param]: { ...prev[param as keyof typeof prev], min, max }
                          }));
                        }}
                        min={0}
                        max={20}
                        step={range.step}
                        disabled={isRunning}
                        size="small"
                        valueLabelDisplay="auto"
                      />
                    </Box>
                  )}
                </Box>
              ))}
            </CardContent>
          </Card>

          {/* Resource Monitor */}
          {isRunning && (
            <Card elevation={2} sx={{ mt: 2 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <MemoryIcon color="primary" />
                  Resource Monitor
                </Typography>

                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Typography variant="body2" gutterBottom>
                      CPU Usage: {resourceUsage.cpu.toFixed(1)}%
                    </Typography>
                    <LinearProgress
                      variant="determinate"
                      value={resourceUsage.cpu}
                      sx={{ height: 8, borderRadius: 4 }}
                    />
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" gutterBottom>
                      Memory: {resourceUsage.memory.toFixed(1)}%
                    </Typography>
                    <LinearProgress
                      variant="determinate"
                      value={resourceUsage.memory}
                      sx={{ height: 8, borderRadius: 4 }}
                    />
                  </Grid>
                  <Grid item xs={12}>
                    <Typography variant="body2">
                      Active Jobs: {resourceUsage.activeJobs} / {batchConfig.maxConcurrentJobs}
                    </Typography>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          )}
        </Grid>

        {/* Results Panel */}
        <Grid item xs={12} lg={8}>
          <Card elevation={2} sx={{ height: 'fit-content' }}>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">
                  Batch Results
                </Typography>
                {batchResults.length > 0 && (
                  <Button
                    variant="outlined"
                    size="small"
                    startIcon={<DownloadIcon />}
                  >
                    Export Results
                  </Button>
                )}
              </Box>

              {/* Progress Bar */}
              {(isRunning || currentBatch) && (
                <Box sx={{ mb: 3 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2">
                      Progress: {Math.round(batchProgress)}%
                    </Typography>
                    <Typography variant="body2">
                      {isRunning ? 'Running...' : currentBatch?.status}
                    </Typography>
                  </Box>
                  <LinearProgress
                    variant="determinate"
                    value={batchProgress}
                    sx={{ height: 10, borderRadius: 5 }}
                  />
                </Box>
              )}

              {batchResults.length === 0 && !isRunning && (
                <Alert severity="info">
                  Configure your batch parameters and click "Start Batch" to begin optimization.
                </Alert>
              )}

              {/* Results Table */}
              {batchResults.length > 0 && (
                <TableContainer sx={{ maxHeight: 500 }}>
                  <Table stickyHeader size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Rank</TableCell>
                        <TableCell>Configuration</TableCell>
                        <TableCell align="right">Return</TableCell>
                        <TableCell align="right">Sharpe</TableCell>
                        <TableCell align="right">Max DD</TableCell>
                        <TableCell align="right">Win Rate</TableCell>
                        <TableCell align="right">Trades</TableCell>
                        <TableCell>Overfit Risk</TableCell>
                        <TableCell>Status</TableCell>
                        <TableCell>Actions</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {batchResults.slice(0, 50).map((result) => (
                        <TableRow
                          key={result.configurationId}
                          hover
                          sx={{
                            cursor: 'pointer',
                            backgroundColor: result.rank <= 3 ? alpha(getRankColor(result.rank), 0.1) : 'inherit'
                          }}
                          onClick={() => setSelectedResult(result)}
                        >
                          <TableCell>
                            <Chip
                              label={`#${result.rank}`}
                              size="small"
                              sx={{
                                fontWeight: 'bold',
                                color: 'white',
                                backgroundColor: getRankColor(result.rank)
                              }}
                            />
                          </TableCell>
                          <TableCell>
                            <Typography variant="body2" fontWeight="medium">
                              {result.name}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              {result.parameters.strategy}
                            </Typography>
                          </TableCell>
                          <TableCell
                            align="right"
                            sx={{
                              color: result.metrics.totalReturn >= 0 ? 'success.main' : 'error.main',
                              fontWeight: 'medium'
                            }}
                          >
                            {formatMetric(result.metrics.totalReturn, 'totalReturn')}
                          </TableCell>
                          <TableCell align="right" sx={{ fontWeight: 'medium' }}>
                            {formatMetric(result.metrics.sharpeRatio, 'sharpeRatio')}
                          </TableCell>
                          <TableCell align="right" sx={{ color: 'error.main' }}>
                            {formatMetric(result.metrics.maxDrawdown, 'maxDrawdown')}
                          </TableCell>
                          <TableCell align="right">
                            {formatMetric(result.metrics.winRate, 'winRate')}
                          </TableCell>
                          <TableCell align="right">
                            {formatMetric(result.metrics.totalTrades, 'totalTrades')}
                          </TableCell>
                          <TableCell>
                            <Chip
                              label={calculateOverfittingRisk(result.metrics).level}
                              size="small"
                              color={calculateOverfittingRisk(result.metrics).color as any}
                              variant="outlined"
                            />
                          </TableCell>
                          <TableCell>
                            <Chip
                              label={result.status}
                              size="small"
                              color={
                                result.status === 'completed' ? 'success' :
                                result.status === 'running' ? 'info' : 'error'
                              }
                              variant="filled"
                            />
                          </TableCell>
                          <TableCell>
                            <IconButton
                              size="small"
                              onClick={(e) => {
                                e.stopPropagation();
                                setSelectedResult(result);
                              }}
                            >
                              <VisibilityIcon />
                            </IconButton>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}

              {/* Top Performers Summary */}
              {batchResults.length > 0 && (
                <Box sx={{ mt: 3 }}>
                  <Typography variant="subtitle1" gutterBottom>
                    Top 3 Performers
                  </Typography>
                  <Grid container spacing={2}>
                    {batchResults.slice(0, 3).map((result, index) => (
                      <Grid item xs={12} md={4} key={result.configurationId}>
                        <Card
                          variant="outlined"
                          sx={{
                            border: 2,
                            borderColor: getRankColor(result.rank),
                            backgroundColor: alpha(getRankColor(result.rank), 0.05)
                          }}
                        >
                          <CardContent sx={{ textAlign: 'center' }}>
                            <Typography variant="h6" color={getRankColor(result.rank)} fontWeight="bold">
                              #{result.rank}
                            </Typography>
                            <Typography variant="body2" gutterBottom>
                              {result.name}
                            </Typography>
                            <Typography variant="h5" fontWeight="bold" color="primary">
                              {formatMetric(result.metrics.totalReturn, 'totalReturn')}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                              Sharpe: {formatMetric(result.metrics.sharpeRatio, 'sharpeRatio')}
                            </Typography>
                          </CardContent>
                        </Card>
                      </Grid>
                    ))}
                  </Grid>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Advanced Settings Dialog */}
      <Dialog
        open={advancedSettingsOpen}
        onClose={() => setAdvancedSettingsOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Advanced Batch Settings</DialogTitle>
        <DialogContent>
          <Grid container spacing={3} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                Cross-Validation Settings
              </Typography>
            </Grid>
            <Grid item xs={6}>
              <TextField
                fullWidth
                type="number"
                label="Cross-Validation Folds"
                value={batchConfig.crossValidationFolds}
                onChange={(e) => setBatchConfig(prev => ({ ...prev, crossValidationFolds: Number(e.target.value) }))}
                inputProps={{ min: 3, max: 10 }}
              />
            </Grid>
            <Grid item xs={6}>
              <TextField
                fullWidth
                type="number"
                label="Out-of-Sample %"
                value={batchConfig.outOfSamplePercentage}
                onChange={(e) => setBatchConfig(prev => ({ ...prev, outOfSamplePercentage: Number(e.target.value) }))}
                inputProps={{ min: 10, max: 50 }}
              />
            </Grid>
            <Grid item xs={6}>
              <TextField
                fullWidth
                type="number"
                label="Walk-Forward Windows"
                value={batchConfig.walkForwardWindows}
                onChange={(e) => setBatchConfig(prev => ({ ...prev, walkForwardWindows: Number(e.target.value) }))}
                inputProps={{ min: 1, max: 10 }}
              />
            </Grid>
            <Grid item xs={6}>
              <TextField
                fullWidth
                type="number"
                label="Resource Limit %"
                value={batchConfig.resourceLimit}
                onChange={(e) => setBatchConfig(prev => ({ ...prev, resourceLimit: Number(e.target.value) }))}
                inputProps={{ min: 10, max: 100 }}
              />
            </Grid>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={batchConfig.parallelExecution}
                    onChange={(e) => setBatchConfig(prev => ({ ...prev, parallelExecution: e.target.checked }))}
                  />
                }
                label="Enable parallel execution"
              />
            </Grid>
            <Grid item xs={12}>
              <Alert severity="info">
                <Typography variant="body2">
                  <strong>Overfitting Protection:</strong> Uses walk-forward analysis and out-of-sample testing 
                  to identify configurations that may not perform well in live trading.
                </Typography>
              </Alert>
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAdvancedSettingsOpen(false)}>Close</Button>
          <Button variant="contained">Save Settings</Button>
        </DialogActions>
      </Dialog>

      {/* Result Details Dialog */}
      {selectedResult && (
        <Dialog
          open={!!selectedResult}
          onClose={() => setSelectedResult(null)}
          maxWidth="lg"
          fullWidth
        >
          <DialogTitle>
            Configuration Details - Rank #{selectedResult.rank}
          </DialogTitle>
          <DialogContent>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Typography variant="h6" gutterBottom>Performance Metrics</Typography>
                <TableContainer>
                  <Table size="small">
                    <TableBody>
                      <TableRow>
                        <TableCell>Total Return</TableCell>
                        <TableCell align="right" sx={{ fontWeight: 'bold' }}>
                          {formatMetric(selectedResult.metrics.totalReturn, 'totalReturn')}
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Sharpe Ratio</TableCell>
                        <TableCell align="right" sx={{ fontWeight: 'bold' }}>
                          {formatMetric(selectedResult.metrics.sharpeRatio, 'sharpeRatio')}
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Max Drawdown</TableCell>
                        <TableCell align="right" sx={{ fontWeight: 'bold', color: 'error.main' }}>
                          {formatMetric(selectedResult.metrics.maxDrawdown, 'maxDrawdown')}
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Win Rate</TableCell>
                        <TableCell align="right" sx={{ fontWeight: 'bold' }}>
                          {formatMetric(selectedResult.metrics.winRate, 'winRate')}
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Total Trades</TableCell>
                        <TableCell align="right" sx={{ fontWeight: 'bold' }}>
                          {formatMetric(selectedResult.metrics.totalTrades, 'totalTrades')}
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Profit Factor</TableCell>
                        <TableCell align="right" sx={{ fontWeight: 'bold' }}>
                          {formatMetric(selectedResult.metrics.profitFactor, 'profitFactor')}
                        </TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </TableContainer>
              </Grid>

              <Grid item xs={12} md={6}>
                <Typography variant="h6" gutterBottom>Configuration Parameters</Typography>
                <TableContainer>
                  <Table size="small">
                    <TableBody>
                      <TableRow>
                        <TableCell>Strategy</TableCell>
                        <TableCell align="right" sx={{ fontWeight: 'bold' }}>
                          {selectedResult.parameters.strategy}
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Stop Loss</TableCell>
                        <TableCell align="right" sx={{ fontWeight: 'bold' }}>
                          {selectedResult.parameters.stopLoss.toFixed(2)}%
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Take Profit</TableCell>
                        <TableCell align="right" sx={{ fontWeight: 'bold' }}>
                          {selectedResult.parameters.takeProfit.toFixed(2)}%
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Position Size</TableCell>
                        <TableCell align="right" sx={{ fontWeight: 'bold' }}>
                          {selectedResult.parameters.positionSize.toFixed(2)}%
                        </TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </TableContainer>

                <Box sx={{ mt: 3 }}>
                  <Typography variant="subtitle1" gutterBottom>
                    Overfitting Assessment
                  </Typography>
                  <Alert severity={calculateOverfittingRisk(selectedResult.metrics).color as any}>
                    <Typography variant="body2">
                      Overfitting Risk: <strong>{calculateOverfittingRisk(selectedResult.metrics).level}</strong>
                    </Typography>
                    <Typography variant="body2" sx={{ mt: 1 }}>
                      {calculateOverfittingRisk(selectedResult.metrics).level === 'High' &&
                        'This configuration may not perform well in live trading due to potential overfitting.'
                      }
                      {calculateOverfittingRisk(selectedResult.metrics).level === 'Medium' &&
                        'Monitor this configuration carefully for signs of overfitting in live trading.'
                      }
                      {calculateOverfittingRisk(selectedResult.metrics).level === 'Low' &&
                        'This configuration shows good potential for live trading performance.'
                      }
                    </Typography>
                  </Alert>
                </Box>
              </Grid>
            </Grid>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setSelectedResult(null)}>Close</Button>
            <Button variant="outlined">Clone Configuration</Button>
            <Button variant="contained">Deploy to Live</Button>
          </DialogActions>
        </Dialog>
      )}
    </Box>
  );
};

export default BatchOptimizer;