/**
 * Configuration Panel Component for Playground
 * Comprehensive interface for setting up trading strategy configurations
 */

import React, { useState, useCallback, useEffect } from 'react';
import {
  Grid,
  Paper,
  Typography,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Box,
  Button,
  Card,
  CardContent,
  FormControlLabel,
  Switch,
  Slider,
  Tooltip,
  IconButton,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Alert,
  Autocomplete,
  useTheme,
  alpha
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Help as HelpIcon,
  Save as SaveIcon,
  Restore as RestoreIcon,
  Science as ScienceIcon,
  TrendingUp as TrendingUpIcon,
  Security as SecurityIcon,
  AccountBalance as PortfolioIcon,
  Psychology as StrategyIcon,
  Settings as ParametersIcon
} from '@mui/icons-material';

import { PlaygroundConfiguration, TimeInterval } from '@/types';
import { useAppDispatch, useAppSelector } from '@/store';
import { selectStrategies } from '@/store/slices/strategySlice';

interface ConfigurationPanelProps {
  configuration: PlaygroundConfiguration | null;
  onConfigurationChange: (config: PlaygroundConfiguration) => void;
  isExecutionActive: boolean;
}

// Sample trading symbols for demo
const SAMPLE_SYMBOLS = [
  'BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'DOT/USDT', 'LINK/USDT',
  'SOL/USDT', 'MATIC/USDT', 'AVAX/USDT', 'ATOM/USDT', 'ALGO/USDT'
];

// Sample strategy types
const STRATEGY_TYPES = [
  { value: 'trend_following', label: 'Trend Following', description: 'Follow market trends with momentum indicators' },
  { value: 'mean_reversion', label: 'Mean Reversion', description: 'Trade on price reversals to the mean' },
  { value: 'arbitrage', label: 'Arbitrage', description: 'Exploit price differences across exchanges' },
  { value: 'market_making', label: 'Market Making', description: 'Provide liquidity and capture spreads' },
  { value: 'ml_based', label: 'ML-Based', description: 'Machine learning driven predictions' },
  { value: 'hybrid', label: 'Hybrid Strategy', description: 'Combination of multiple approaches' }
];

// Model types for ML strategies
const MODEL_TYPES = [
  { value: 'lstm', label: 'LSTM Neural Network', description: 'Long Short-Term Memory for time series' },
  { value: 'transformer', label: 'Transformer', description: 'Attention-based sequence modeling' },
  { value: 'random_forest', label: 'Random Forest', description: 'Ensemble tree-based classifier' },
  { value: 'xgboost', label: 'XGBoost', description: 'Gradient boosting framework' },
  { value: 'ensemble', label: 'Ensemble Model', description: 'Combination of multiple models' }
];

const ConfigurationPanel: React.FC<ConfigurationPanelProps> = ({
  configuration,
  onConfigurationChange,
  isExecutionActive
}) => {
  const theme = useTheme();
  const dispatch = useAppDispatch();
  const strategies = useAppSelector(selectStrategies);

  // Local state for form management
  const [formData, setFormData] = useState<PlaygroundConfiguration>({
    name: '',
    description: '',
    symbols: [],
    positionSizing: {
      type: 'percentage',
      value: 2,
      maxPositions: 5
    },
    tradingSide: 'both',
    riskSettings: {
      stopLossPercentage: 2,
      takeProfitPercentage: 4,
      maxDrawdownPercentage: 10,
      maxRiskPerTrade: 2
    },
    portfolioSettings: {
      maxPositions: 5,
      allocationStrategy: 'equal_weight',
      rebalanceFrequency: 'daily'
    },
    strategy: {
      type: 'trend_following',
      parameters: {}
    },
    timeframe: '1h'
  });

  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({
    basic: true,
    symbols: true,
    position: true,
    risk: true,
    portfolio: false,
    strategy: false
  });

  const [validationErrors, setValidationErrors] = useState<Record<string, string>>({});
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);

  // Update form data when configuration prop changes
  useEffect(() => {
    if (configuration) {
      setFormData(configuration);
      setHasUnsavedChanges(false);
    }
  }, [configuration]);

  // Handle form field changes
  const handleFieldChange = useCallback((path: string[], value: any) => {
    setFormData(prev => {
      const newData = { ...prev };
      let current: any = newData;
      
      for (let i = 0; i < path.length - 1; i++) {
        if (!current[path[i]]) current[path[i]] = {};
        current = current[path[i]];
      }
      
      current[path[path.length - 1]] = value;
      return newData;
    });
    setHasUnsavedChanges(true);
  }, []);

  // Handle accordion expansion
  const handleAccordionChange = useCallback((panel: string) => (
    event: React.SyntheticEvent, isExpanded: boolean
  ) => {
    setExpandedSections(prev => ({
      ...prev,
      [panel]: isExpanded
    }));
  }, []);

  // Validate configuration
  const validateConfiguration = useCallback((): Record<string, string> => {
    const errors: Record<string, string> = {};

    if (!formData.name.trim()) {
      errors.name = 'Configuration name is required';
    }

    if (formData.symbols.length === 0) {
      errors.symbols = 'At least one trading symbol must be selected';
    }

    if (formData.positionSizing.value <= 0) {
      errors.positionSizing = 'Position size must be greater than 0';
    }

    if (formData.riskSettings.stopLossPercentage >= formData.riskSettings.takeProfitPercentage) {
      errors.riskSettings = 'Take profit must be greater than stop loss';
    }

    return errors;
  }, [formData]);

  // Save configuration
  const handleSaveConfiguration = useCallback(() => {
    const errors = validateConfiguration();
    setValidationErrors(errors);

    if (Object.keys(errors).length === 0) {
      onConfigurationChange(formData);
      setHasUnsavedChanges(false);
    }
  }, [formData, validateConfiguration, onConfigurationChange]);

  // Reset to original configuration
  const handleResetConfiguration = useCallback(() => {
    if (configuration) {
      setFormData(configuration);
    }
    setHasUnsavedChanges(false);
    setValidationErrors({});
  }, [configuration]);

  // Helper component for section headers
  const SectionHeader: React.FC<{
    icon: React.ReactNode;
    title: string;
    description: string;
  }> = ({ icon, title, description }) => (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
      <Box
        sx={{
          p: 1,
          borderRadius: 1,
          backgroundColor: alpha(theme.palette.primary.main, 0.1),
          color: 'primary.main'
        }}
      >
        {icon}
      </Box>
      <Box>
        <Typography variant="h6" fontWeight="bold">
          {title}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          {description}
        </Typography>
      </Box>
    </Box>
  );

  return (
    <Box>
      {/* Header Actions */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h5" fontWeight="bold">
          Configuration Setup
        </Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          {hasUnsavedChanges && (
            <Alert severity="warning" sx={{ mr: 2 }}>
              You have unsaved changes
            </Alert>
          )}
          <Button
            variant="outlined"
            startIcon={<RestoreIcon />}
            onClick={handleResetConfiguration}
            disabled={!hasUnsavedChanges}
          >
            Reset
          </Button>
          <Button
            variant="contained"
            startIcon={<SaveIcon />}
            onClick={handleSaveConfiguration}
            disabled={isExecutionActive}
          >
            Save Configuration
          </Button>
        </Box>
      </Box>

      <Grid container spacing={3}>
        {/* Basic Information */}
        <Grid item xs={12}>
          <Accordion
            expanded={expandedSections.basic}
            onChange={handleAccordionChange('basic')}
            elevation={2}
          >
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <SectionHeader
                icon={<ScienceIcon />}
                title="Basic Information"
                description="Name and describe your configuration"
              />
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="Configuration Name"
                    value={formData.name}
                    onChange={(e) => handleFieldChange(['name'], e.target.value)}
                    error={!!validationErrors.name}
                    helperText={validationErrors.name || 'Give your configuration a memorable name'}
                    disabled={isExecutionActive}
                    required
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <FormControl fullWidth>
                    <InputLabel>Timeframe</InputLabel>
                    <Select
                      value={formData.timeframe}
                      label="Timeframe"
                      onChange={(e) => handleFieldChange(['timeframe'], e.target.value)}
                      disabled={isExecutionActive}
                    >
                      <MenuItem value="1m">1 Minute</MenuItem>
                      <MenuItem value="5m">5 Minutes</MenuItem>
                      <MenuItem value="15m">15 Minutes</MenuItem>
                      <MenuItem value="30m">30 Minutes</MenuItem>
                      <MenuItem value="1h">1 Hour</MenuItem>
                      <MenuItem value="4h">4 Hours</MenuItem>
                      <MenuItem value="1d">1 Day</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    multiline
                    rows={3}
                    label="Description (Optional)"
                    value={formData.description || ''}
                    onChange={(e) => handleFieldChange(['description'], e.target.value)}
                    helperText="Describe the purpose or characteristics of this configuration"
                    disabled={isExecutionActive}
                  />
                </Grid>
              </Grid>
            </AccordionDetails>
          </Accordion>
        </Grid>

        {/* Symbol Selection */}
        <Grid item xs={12}>
          <Accordion
            expanded={expandedSections.symbols}
            onChange={handleAccordionChange('symbols')}
            elevation={2}
          >
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <SectionHeader
                icon={<TrendingUpIcon />}
                title="Trading Symbols"
                description="Select the cryptocurrency pairs to trade"
              />
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={3}>
                <Grid item xs={12} md={8}>
                  <Autocomplete
                    multiple
                    options={SAMPLE_SYMBOLS}
                    value={formData.symbols}
                    onChange={(_, newValue) => handleFieldChange(['symbols'], newValue)}
                    renderTags={(value, getTagProps) =>
                      value.map((option, index) => (
                        <Chip
                          {...getTagProps({ index })}
                          key={option}
                          label={option}
                          color="primary"
                          variant="outlined"
                        />
                      ))
                    }
                    renderInput={(params) => (
                      <TextField
                        {...params}
                        label="Select Trading Pairs"
                        placeholder="Choose symbols..."
                        error={!!validationErrors.symbols}
                        helperText={validationErrors.symbols || 'Select one or more trading pairs'}
                      />
                    )}
                    disabled={isExecutionActive}
                  />
                </Grid>
                <Grid item xs={12} md={4}>
                  <FormControl fullWidth>
                    <InputLabel>Trading Side</InputLabel>
                    <Select
                      value={formData.tradingSide}
                      label="Trading Side"
                      onChange={(e) => handleFieldChange(['tradingSide'], e.target.value)}
                      disabled={isExecutionActive}
                    >
                      <MenuItem value="long">Long Only</MenuItem>
                      <MenuItem value="short">Short Only</MenuItem>
                      <MenuItem value="both">Both Long & Short</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
              </Grid>
            </AccordionDetails>
          </Accordion>
        </Grid>

        {/* Position Sizing */}
        <Grid item xs={12}>
          <Accordion
            expanded={expandedSections.position}
            onChange={handleAccordionChange('position')}
            elevation={2}
          >
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <SectionHeader
                icon={<ParametersIcon />}
                title="Position Sizing"
                description="Configure how positions are sized"
              />
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={3}>
                <Grid item xs={12} md={4}>
                  <FormControl fullWidth>
                    <InputLabel>Sizing Method</InputLabel>
                    <Select
                      value={formData.positionSizing.type}
                      label="Sizing Method"
                      onChange={(e) => handleFieldChange(['positionSizing', 'type'], e.target.value)}
                      disabled={isExecutionActive}
                    >
                      <MenuItem value="fixed">Fixed Amount</MenuItem>
                      <MenuItem value="percentage">Percentage of Portfolio</MenuItem>
                      <MenuItem value="kelly_criterion">Kelly Criterion</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12} md={4}>
                  <TextField
                    fullWidth
                    type="number"
                    label={`Position Size ${formData.positionSizing.type === 'fixed' ? '(USD)' : '(%)'}`}
                    value={formData.positionSizing.value}
                    onChange={(e) => handleFieldChange(['positionSizing', 'value'], Number(e.target.value))}
                    error={!!validationErrors.positionSizing}
                    helperText={validationErrors.positionSizing}
                    disabled={isExecutionActive}
                    InputProps={{
                      inputProps: { min: 0, step: formData.positionSizing.type === 'percentage' ? 0.1 : 10 }
                    }}
                  />
                </Grid>
                <Grid item xs={12} md={4}>
                  <TextField
                    fullWidth
                    type="number"
                    label="Max Concurrent Positions"
                    value={formData.positionSizing.maxPositions}
                    onChange={(e) => handleFieldChange(['positionSizing', 'maxPositions'], Number(e.target.value))}
                    disabled={isExecutionActive}
                    InputProps={{
                      inputProps: { min: 1, max: 20 }
                    }}
                  />
                </Grid>
              </Grid>
            </AccordionDetails>
          </Accordion>
        </Grid>

        {/* Risk Management */}
        <Grid item xs={12}>
          <Accordion
            expanded={expandedSections.risk}
            onChange={handleAccordionChange('risk')}
            elevation={2}
          >
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <SectionHeader
                icon={<SecurityIcon />}
                title="Risk Management"
                description="Set stop-loss, take-profit, and risk limits"
              />
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <Typography gutterBottom>
                    Stop Loss: {formData.riskSettings.stopLossPercentage}%
                    <Tooltip title="Percentage loss at which positions are automatically closed">
                      <IconButton size="small" sx={{ ml: 1 }}>
                        <HelpIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </Typography>
                  <Slider
                    value={formData.riskSettings.stopLossPercentage}
                    onChange={(_, value) => handleFieldChange(['riskSettings', 'stopLossPercentage'], value)}
                    min={0.5}
                    max={10}
                    step={0.1}
                    marks={[
                      { value: 1, label: '1%' },
                      { value: 2, label: '2%' },
                      { value: 5, label: '5%' },
                      { value: 10, label: '10%' }
                    ]}
                    disabled={isExecutionActive}
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <Typography gutterBottom>
                    Take Profit: {formData.riskSettings.takeProfitPercentage}%
                    <Tooltip title="Percentage gain at which positions are automatically closed">
                      <IconButton size="small" sx={{ ml: 1 }}>
                        <HelpIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </Typography>
                  <Slider
                    value={formData.riskSettings.takeProfitPercentage}
                    onChange={(_, value) => handleFieldChange(['riskSettings', 'takeProfitPercentage'], value)}
                    min={1}
                    max={20}
                    step={0.1}
                    marks={[
                      { value: 2, label: '2%' },
                      { value: 5, label: '5%' },
                      { value: 10, label: '10%' },
                      { value: 20, label: '20%' }
                    ]}
                    disabled={isExecutionActive}
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <Typography gutterBottom>
                    Max Drawdown: {formData.riskSettings.maxDrawdownPercentage}%
                  </Typography>
                  <Slider
                    value={formData.riskSettings.maxDrawdownPercentage}
                    onChange={(_, value) => handleFieldChange(['riskSettings', 'maxDrawdownPercentage'], value)}
                    min={5}
                    max={50}
                    step={1}
                    marks={[
                      { value: 10, label: '10%' },
                      { value: 20, label: '20%' },
                      { value: 30, label: '30%' },
                      { value: 50, label: '50%' }
                    ]}
                    disabled={isExecutionActive}
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <Typography gutterBottom>
                    Max Risk per Trade: {formData.riskSettings.maxRiskPerTrade}%
                  </Typography>
                  <Slider
                    value={formData.riskSettings.maxRiskPerTrade}
                    onChange={(_, value) => handleFieldChange(['riskSettings', 'maxRiskPerTrade'], value)}
                    min={0.1}
                    max={10}
                    step={0.1}
                    marks={[
                      { value: 1, label: '1%' },
                      { value: 2, label: '2%' },
                      { value: 5, label: '5%' },
                      { value: 10, label: '10%' }
                    ]}
                    disabled={isExecutionActive}
                  />
                </Grid>
              </Grid>
              {validationErrors.riskSettings && (
                <Alert severity="error" sx={{ mt: 2 }}>
                  {validationErrors.riskSettings}
                </Alert>
              )}
            </AccordionDetails>
          </Accordion>
        </Grid>

        {/* Portfolio Settings */}
        <Grid item xs={12}>
          <Accordion
            expanded={expandedSections.portfolio}
            onChange={handleAccordionChange('portfolio')}
            elevation={2}
          >
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <SectionHeader
                icon={<PortfolioIcon />}
                title="Portfolio Settings"
                description="Configure portfolio allocation and rebalancing"
              />
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={3}>
                <Grid item xs={12} md={4}>
                  <TextField
                    fullWidth
                    type="number"
                    label="Max Portfolio Positions"
                    value={formData.portfolioSettings.maxPositions}
                    onChange={(e) => handleFieldChange(['portfolioSettings', 'maxPositions'], Number(e.target.value))}
                    disabled={isExecutionActive}
                    InputProps={{
                      inputProps: { min: 1, max: 50 }
                    }}
                  />
                </Grid>
                <Grid item xs={12} md={4}>
                  <FormControl fullWidth>
                    <InputLabel>Allocation Strategy</InputLabel>
                    <Select
                      value={formData.portfolioSettings.allocationStrategy}
                      label="Allocation Strategy"
                      onChange={(e) => handleFieldChange(['portfolioSettings', 'allocationStrategy'], e.target.value)}
                      disabled={isExecutionActive}
                    >
                      <MenuItem value="equal_weight">Equal Weight</MenuItem>
                      <MenuItem value="risk_parity">Risk Parity</MenuItem>
                      <MenuItem value="market_cap">Market Cap Weighted</MenuItem>
                      <MenuItem value="custom">Custom Allocation</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12} md={4}>
                  <FormControl fullWidth>
                    <InputLabel>Rebalance Frequency</InputLabel>
                    <Select
                      value={formData.portfolioSettings.rebalanceFrequency}
                      label="Rebalance Frequency"
                      onChange={(e) => handleFieldChange(['portfolioSettings', 'rebalanceFrequency'], e.target.value)}
                      disabled={isExecutionActive}
                    >
                      <MenuItem value="never">Never</MenuItem>
                      <MenuItem value="daily">Daily</MenuItem>
                      <MenuItem value="weekly">Weekly</MenuItem>
                      <MenuItem value="monthly">Monthly</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
              </Grid>
            </AccordionDetails>
          </Accordion>
        </Grid>

        {/* Strategy Configuration */}
        <Grid item xs={12}>
          <Accordion
            expanded={expandedSections.strategy}
            onChange={handleAccordionChange('strategy')}
            elevation={2}
          >
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <SectionHeader
                icon={<StrategyIcon />}
                title="Strategy Configuration"
                description="Select and configure your trading strategy"
              />
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <FormControl fullWidth>
                    <InputLabel>Strategy Type</InputLabel>
                    <Select
                      value={formData.strategy.type}
                      label="Strategy Type"
                      onChange={(e) => handleFieldChange(['strategy', 'type'], e.target.value)}
                      disabled={isExecutionActive}
                    >
                      {STRATEGY_TYPES.map((strategy) => (
                        <MenuItem key={strategy.value} value={strategy.value}>
                          <Box>
                            <Typography>{strategy.label}</Typography>
                            <Typography variant="caption" color="text.secondary">
                              {strategy.description}
                            </Typography>
                          </Box>
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>
                {formData.strategy.type === 'ml_based' && (
                  <Grid item xs={12} md={6}>
                    <FormControl fullWidth>
                      <InputLabel>Model Type</InputLabel>
                      <Select
                        value={formData.model?.type || ''}
                        label="Model Type"
                        onChange={(e) => handleFieldChange(['model', 'type'], e.target.value)}
                        disabled={isExecutionActive}
                      >
                        {MODEL_TYPES.map((model) => (
                          <MenuItem key={model.value} value={model.value}>
                            <Box>
                              <Typography>{model.label}</Typography>
                              <Typography variant="caption" color="text.secondary">
                                {model.description}
                              </Typography>
                            </Box>
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                  </Grid>
                )}
              </Grid>
            </AccordionDetails>
          </Accordion>
        </Grid>
      </Grid>
    </Box>
  );
};

export default ConfigurationPanel;