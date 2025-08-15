/**
 * Execution Controls Component for Playground
 * Interface for controlling bot execution and monitoring settings
 */

import React, { useState, useCallback, useEffect } from 'react';
import {
  Paper,
  Typography,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Box,
  Card,
  CardContent,
  Slider,
  FormControlLabel,
  Switch,
  LinearProgress,
  Chip,
  Grid,
  Divider,
  Alert,
  IconButton,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  useTheme,
  alpha
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Pause as PauseIcon,
  Stop as StopIcon,
  Refresh as RefreshIcon,
  Speed as SpeedIcon,
  Warning as WarningIcon,
  Info as InfoIcon,
  Settings as SettingsIcon,
  DateRange as DateRangeIcon,
  AccountBalance as AccountBalanceIcon
} from '@mui/icons-material';
import { DatePicker } from '@mui/x-date-pickers/DatePicker';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';

import { PlaygroundConfiguration, PlaygroundExecution } from '@/types';

interface ExecutionControlsProps {
  configuration: PlaygroundConfiguration | null;
  execution: PlaygroundExecution | null;
  onExecutionStart: (execution: PlaygroundExecution) => void;
  onExecutionControl: (action: 'pause' | 'resume' | 'stop' | 'restart') => void;
}

type ExecutionMode = 'historical' | 'live' | 'sandbox' | 'production';

const ExecutionControls: React.FC<ExecutionControlsProps> = ({
  configuration,
  execution,
  onExecutionStart,
  onExecutionControl
}) => {
  const theme = useTheme();

  // Local state for execution settings
  const [executionMode, setExecutionMode] = useState<ExecutionMode>('historical');
  const [startDate, setStartDate] = useState<Date>(new Date(Date.now() - 30 * 24 * 60 * 60 * 1000)); // 30 days ago
  const [endDate, setEndDate] = useState<Date>(new Date());
  const [executionSpeed, setExecutionSpeed] = useState<number>(1);
  const [initialBalance, setInitialBalance] = useState<number>(10000);
  const [commission, setCommission] = useState<number>(0.1);
  const [autoRestart, setAutoRestart] = useState<boolean>(false);
  const [showAdvancedSettings, setShowAdvancedSettings] = useState<boolean>(false);
  const [productionWarningOpen, setProductionWarningOpen] = useState<boolean>(false);

  const isExecutionActive = execution && ['running', 'paused'].includes(execution.status);
  const canStart = configuration && !isExecutionActive;
  const canControl = execution && execution.status !== 'idle';

  // Handle execution mode change
  const handleModeChange = useCallback((mode: ExecutionMode) => {
    if (mode === 'production') {
      setProductionWarningOpen(true);
    } else {
      setExecutionMode(mode);
    }
  }, []);

  // Handle production mode confirmation
  const handleProductionConfirm = useCallback(() => {
    setExecutionMode('production');
    setProductionWarningOpen(false);
  }, []);

  // Start execution
  const handleStart = useCallback(() => {
    if (!configuration) return;

    const newExecution: PlaygroundExecution = {
      id: `exec_${Date.now()}`,
      configurationId: configuration.id || 'temp',
      mode: executionMode,
      status: 'running',
      progress: 0,
      startTime: new Date().toISOString(),
      settings: {
        startDate: executionMode === 'historical' ? startDate.toISOString() : undefined,
        endDate: executionMode === 'historical' ? endDate.toISOString() : undefined,
        speed: executionSpeed,
        initialBalance,
        commission: commission / 100 // Convert percentage to decimal
      },
      logs: [{
        id: `log_${Date.now()}`,
        timestamp: new Date().toISOString(),
        level: 'info',
        category: 'system',
        message: `Execution started in ${executionMode} mode`
      }]
    };

    onExecutionStart(newExecution);
  }, [configuration, executionMode, startDate, endDate, executionSpeed, initialBalance, commission, onExecutionStart]);

  // Speed control options
  const speedOptions = [
    { value: 0.1, label: '0.1x (Slow)', description: 'Detailed analysis' },
    { value: 0.5, label: '0.5x', description: 'Careful execution' },
    { value: 1, label: '1x (Normal)', description: 'Real-time speed' },
    { value: 2, label: '2x', description: 'Fast execution' },
    { value: 5, label: '5x', description: 'Very fast' },
    { value: 10, label: '10x', description: 'Ultra fast' },
    { value: 100, label: 'Max Speed', description: 'As fast as possible' }
  ];

  // Get status color
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'success';
      case 'paused': return 'warning';
      case 'error': return 'error';
      case 'completed': return 'info';
      default: return 'default';
    }
  };

  // Get mode description
  const getModeDescription = (mode: ExecutionMode) => {
    switch (mode) {
      case 'historical':
        return 'Test strategy on historical data with customizable date range';
      case 'live':
        return 'Paper trading with real-time market data (no real money)';
      case 'sandbox':
        return 'Simulated market conditions for strategy validation';
      case 'production':
        return 'Live trading with real money - USE WITH EXTREME CAUTION';
      default:
        return '';
    }
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, height: '100%' }}>
      {/* Execution Mode Selection */}
      <Card elevation={2}>
        <CardContent>
          <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <SettingsIcon color="primary" />
            Execution Mode
          </Typography>
          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel>Mode</InputLabel>
            <Select
              value={executionMode}
              label="Mode"
              onChange={(e) => handleModeChange(e.target.value as ExecutionMode)}
              disabled={isExecutionActive}
            >
              <MenuItem value="historical">
                <Box>
                  <Typography>Historical Backtesting</Typography>
                  <Typography variant="caption" color="text.secondary">
                    Test on past data
                  </Typography>
                </Box>
              </MenuItem>
              <MenuItem value="sandbox">
                <Box>
                  <Typography>Sandbox Mode</Typography>
                  <Typography variant="caption" color="text.secondary">
                    Simulated environment
                  </Typography>
                </Box>
              </MenuItem>
              <MenuItem value="live">
                <Box>
                  <Typography>Live Paper Trading</Typography>
                  <Typography variant="caption" color="text.secondary">
                    Real data, no real money
                  </Typography>
                </Box>
              </MenuItem>
              <MenuItem 
                value="production" 
                sx={{ 
                  backgroundColor: alpha(theme.palette.error.main, 0.1),
                  '&:hover': {
                    backgroundColor: alpha(theme.palette.error.main, 0.2)
                  }
                }}
              >
                <Box>
                  <Typography sx={{ color: 'error.main' }}>Production Trading</Typography>
                  <Typography variant="caption" color="error.main">
                    LIVE MONEY - High Risk
                  </Typography>
                </Box>
              </MenuItem>
            </Select>
          </FormControl>
          <Alert severity="info" icon={<InfoIcon />}>
            {getModeDescription(executionMode)}
          </Alert>
        </CardContent>
      </Card>

      {/* Execution Settings */}
      <Card elevation={2}>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6">Execution Settings</Typography>
            <IconButton
              onClick={() => setShowAdvancedSettings(!showAdvancedSettings)}
              size="small"
            >
              <SettingsIcon />
            </IconButton>
          </Box>

          <Grid container spacing={2}>
            {/* Date Range for Historical Mode */}
            {executionMode === 'historical' && (
              <>
                <Grid item xs={6}>
                  <LocalizationProvider dateAdapter={AdapterDateFns}>
                    <DatePicker
                      label="Start Date"
                      value={startDate}
                      onChange={(newValue) => newValue && setStartDate(newValue)}
                      disabled={isExecutionActive}
                      slotProps={{
                        textField: {
                          fullWidth: true,
                          size: 'small'
                        }
                      }}
                    />
                  </LocalizationProvider>
                </Grid>
                <Grid item xs={6}>
                  <LocalizationProvider dateAdapter={AdapterDateFns}>
                    <DatePicker
                      label="End Date"
                      value={endDate}
                      onChange={(newValue) => newValue && setEndDate(newValue)}
                      disabled={isExecutionActive}
                      minDate={startDate}
                      slotProps={{
                        textField: {
                          fullWidth: true,
                          size: 'small'
                        }
                      }}
                    />
                  </LocalizationProvider>
                </Grid>
              </>
            )}

            {/* Initial Balance */}
            <Grid item xs={6}>
              <TextField
                fullWidth
                size="small"
                type="number"
                label="Initial Balance (USD)"
                value={initialBalance}
                onChange={(e) => setInitialBalance(Number(e.target.value))}
                disabled={isExecutionActive}
                InputProps={{
                  startAdornment: <AccountBalanceIcon sx={{ mr: 1, color: 'text.secondary' }} />
                }}
              />
            </Grid>

            {/* Commission */}
            <Grid item xs={6}>
              <TextField
                fullWidth
                size="small"
                type="number"
                label="Commission (%)"
                value={commission}
                onChange={(e) => setCommission(Number(e.target.value))}
                disabled={isExecutionActive}
                inputProps={{ step: 0.01, min: 0, max: 5 }}
              />
            </Grid>

            {/* Speed Control for Historical Mode */}
            {executionMode === 'historical' && (
              <Grid item xs={12}>
                <Box sx={{ mt: 2 }}>
                  <Typography gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <SpeedIcon />
                    Execution Speed: {speedOptions.find(opt => opt.value === executionSpeed)?.label}
                  </Typography>
                  <Slider
                    value={executionSpeed}
                    onChange={(_, value) => setExecutionSpeed(value as number)}
                    min={0.1}
                    max={100}
                    step={null}
                    marks={speedOptions.map(opt => ({ value: opt.value, label: opt.label }))}
                    disabled={isExecutionActive}
                    valueLabelDisplay="off"
                    sx={{ mt: 2 }}
                  />
                  <Typography variant="caption" color="text.secondary">
                    {speedOptions.find(opt => opt.value === executionSpeed)?.description}
                  </Typography>
                </Box>
              </Grid>
            )}

            {/* Advanced Settings */}
            {showAdvancedSettings && (
              <Grid item xs={12}>
                <Divider sx={{ my: 2 }} />
                <Typography variant="subtitle2" gutterBottom>
                  Advanced Options
                </Typography>
                <FormControlLabel
                  control={
                    <Switch
                      checked={autoRestart}
                      onChange={(e) => setAutoRestart(e.target.checked)}
                      disabled={isExecutionActive}
                    />
                  }
                  label="Auto-restart on completion"
                />
              </Grid>
            )}
          </Grid>
        </CardContent>
      </Card>

      {/* Control Buttons */}
      <Card elevation={2}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Execution Controls
          </Typography>
          
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            {!isExecutionActive ? (
              <Button
                variant="contained"
                size="large"
                startIcon={<PlayIcon />}
                onClick={handleStart}
                disabled={!canStart}
                color={executionMode === 'production' ? 'error' : 'primary'}
                sx={{ py: 1.5 }}
              >
                {executionMode === 'production' ? 'START LIVE TRADING' : 'Start Execution'}
              </Button>
            ) : (
              <Grid container spacing={1}>
                {execution?.status === 'running' ? (
                  <Grid item xs={6}>
                    <Button
                      variant="outlined"
                      fullWidth
                      startIcon={<PauseIcon />}
                      onClick={() => onExecutionControl('pause')}
                      color="warning"
                    >
                      Pause
                    </Button>
                  </Grid>
                ) : (
                  <Grid item xs={6}>
                    <Button
                      variant="contained"
                      fullWidth
                      startIcon={<PlayIcon />}
                      onClick={() => onExecutionControl('resume')}
                      color="primary"
                    >
                      Resume
                    </Button>
                  </Grid>
                )}
                <Grid item xs={6}>
                  <Button
                    variant="outlined"
                    fullWidth
                    startIcon={<StopIcon />}
                    onClick={() => onExecutionControl('stop')}
                    color="error"
                  >
                    Stop
                  </Button>
                </Grid>
                <Grid item xs={12}>
                  <Button
                    variant="text"
                    fullWidth
                    startIcon={<RefreshIcon />}
                    onClick={() => onExecutionControl('restart')}
                    disabled={execution?.status === 'running'}
                  >
                    Restart
                  </Button>
                </Grid>
              </Grid>
            )}

            {/* Configuration Status */}
            {!configuration && (
              <Alert severity="warning">
                Please configure your strategy settings first
              </Alert>
            )}
          </Box>
        </CardContent>
      </Card>

      {/* Execution Status */}
      {execution && (
        <Card elevation={2}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Execution Status
            </Typography>
            
            <Box sx={{ mb: 2 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                <Chip
                  label={execution.status.toUpperCase()}
                  color={getStatusColor(execution.status) as any}
                  variant="filled"
                />
                <Typography variant="body2" color="text.secondary">
                  {Math.round(execution.progress)}% Complete
                </Typography>
              </Box>
              <LinearProgress 
                variant="determinate" 
                value={execution.progress} 
                sx={{ height: 8, borderRadius: 4 }}
                color={getStatusColor(execution.status) as any}
              />
            </Box>

            <Grid container spacing={1}>
              <Grid item xs={6}>
                <Typography variant="body2" color="text.secondary">Mode</Typography>
                <Typography variant="body1" fontWeight="medium">
                  {execution.mode.charAt(0).toUpperCase() + execution.mode.slice(1)}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2" color="text.secondary">Duration</Typography>
                <Typography variant="body1" fontWeight="medium">
                  {execution.duration ? `${Math.round(execution.duration / 1000)}s` : 'N/A'}
                </Typography>
              </Grid>
              {execution.metrics && (
                <>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">P&L</Typography>
                    <Typography
                      variant="body1"
                      fontWeight="medium"
                      color={execution.metrics.totalReturn >= 0 ? 'success.main' : 'error.main'}
                    >
                      {execution.metrics.totalReturn >= 0 ? '+' : ''}
                      {execution.metrics.totalReturn.toFixed(2)}%
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">Trades</Typography>
                    <Typography variant="body1" fontWeight="medium">
                      {execution.metrics.totalTrades}
                    </Typography>
                  </Grid>
                </>
              )}
            </Grid>
          </CardContent>
        </Card>
      )}

      {/* Production Warning Dialog */}
      <Dialog
        open={productionWarningOpen}
        onClose={() => setProductionWarningOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle sx={{ color: 'error.main', display: 'flex', alignItems: 'center', gap: 1 }}>
          <WarningIcon />
          PRODUCTION TRADING WARNING
        </DialogTitle>
        <DialogContent>
          <Alert severity="error" sx={{ mb: 2 }}>
            <Typography variant="body1" fontWeight="bold">
              You are about to enable LIVE TRADING with REAL MONEY.
            </Typography>
          </Alert>
          <Typography paragraph>
            Production mode will execute trades on live exchanges using real funds. This involves significant financial risk:
          </Typography>
          <ul>
            <li>All trades will use real money from your exchange accounts</li>
            <li>Losses can occur and may be substantial</li>
            <li>Market conditions can cause unexpected behavior</li>
            <li>No guarantee of profits or performance</li>
          </ul>
          <Typography paragraph color="error.main" fontWeight="bold">
            Only proceed if you fully understand the risks and are prepared for potential losses.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setProductionWarningOpen(false)}>
            Cancel
          </Button>
          <Button
            onClick={handleProductionConfirm}
            variant="contained"
            color="error"
            startIcon={<WarningIcon />}
          >
            I Understand the Risks - Proceed
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ExecutionControls;