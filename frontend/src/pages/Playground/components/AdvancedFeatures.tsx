/**
 * Advanced Features Component for Playground
 * A/B testing, parameter optimization, and multi-instance execution
 */

import React, { useState, useCallback } from 'react';
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
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  ListItemSecondaryAction,
  Divider,
  LinearProgress,
  useTheme,
  alpha,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  FormControlLabel,
  Switch
} from '@mui/material';
import {
  Science as ScienceIcon,
  CompareArrows as CompareIcon,
  Tune as TuneIcon,
  Speed as SpeedIcon,
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  PlayArrow as PlayIcon,
  ExpandMore as ExpandMoreIcon,
  Assessment as AssessmentIcon,
  Timeline as TimelineIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  Psychology as PsychologyIcon,
  AutoFixHigh as OptimizeIcon
} from '@mui/icons-material';

import {
  PlaygroundConfiguration,
  PlaygroundExecution,
  ABTest,
  ParameterOptimization
} from '@/types';

interface AdvancedFeaturesProps {
  configuration: PlaygroundConfiguration | null;
  executions: PlaygroundExecution[];
}

// Mock A/B test data
const mockABTests: ABTest[] = [
  {
    id: 'ab_1',
    name: 'Stop Loss Comparison',
    configurations: {
      control: {
        name: 'Conservative SL (2%)',
        description: 'Lower risk approach',
        symbols: ['BTC/USDT'],
        positionSizing: { type: 'percentage', value: 2, maxPositions: 5 },
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
        strategy: { type: 'trend_following', parameters: {} },
        timeframe: '1h'
      },
      treatment: {
        name: 'Aggressive SL (3%)',
        description: 'Higher risk for better returns',
        symbols: ['BTC/USDT'],
        positionSizing: { type: 'percentage', value: 2, maxPositions: 5 },
        tradingSide: 'both',
        riskSettings: {
          stopLossPercentage: 3,
          takeProfitPercentage: 6,
          maxDrawdownPercentage: 15,
          maxRiskPerTrade: 3
        },
        portfolioSettings: {
          maxPositions: 5,
          allocationStrategy: 'equal_weight',
          rebalanceFrequency: 'daily'
        },
        strategy: { type: 'trend_following', parameters: {} },
        timeframe: '1h'
      }
    },
    executions: {} as any,
    results: {
      significanceLevel: 0.05,
      pValue: 0.023,
      confidenceInterval: [1.2, 4.8],
      winner: 'treatment',
      effect: {
        magnitude: 3.2,
        direction: 'positive',
        metric: 'Total Return'
      }
    },
    status: 'completed',
    createdAt: new Date().toISOString()
  }
];

// Mock parameter optimization data
const mockParameters: ParameterOptimization[] = [
  {
    parameter: 'stopLossPercentage',
    type: 'range',
    min: 1,
    max: 5,
    step: 0.1,
    current: 2,
    optimal: 2.8,
    sensitivity: 0.65
  },
  {
    parameter: 'takeProfitPercentage',
    type: 'range',
    min: 2,
    max: 10,
    step: 0.1,
    current: 4,
    optimal: 5.2,
    sensitivity: 0.43
  },
  {
    parameter: 'positionSize',
    type: 'range',
    min: 1,
    max: 5,
    step: 0.1,
    current: 2,
    optimal: 2.3,
    sensitivity: 0.78
  }
];

const AdvancedFeatures: React.FC<AdvancedFeaturesProps> = ({
  configuration,
  executions
}) => {
  const theme = useTheme();

  // State management
  const [abTests, setAbTests] = useState<ABTest[]>(mockABTests);
  const [selectedABTest, setSelectedABTest] = useState<ABTest | null>(null);
  const [createABTestOpen, setCreateABTestOpen] = useState<boolean>(false);
  const [optimizationOpen, setOptimizationOpen] = useState<boolean>(false);
  const [multiInstanceOpen, setMultiInstanceOpen] = useState<boolean>(false);
  const [parameters, setParameters] = useState<ParameterOptimization[]>(mockParameters);
  const [optimizationMetric, setOptimizationMetric] = useState<string>('sharpe_ratio');
  const [optimizationProgress, setOptimizationProgress] = useState<number>(0);
  const [isOptimizing, setIsOptimizing] = useState<boolean>(false);

  // A/B Test Creation Form State
  const [abTestForm, setABTestForm] = useState({
    name: '',
    description: '',
    controlName: 'Control',
    treatmentName: 'Treatment',
    testParameter: 'stopLossPercentage',
    controlValue: 2,
    treatmentValue: 3,
    significance: 0.05,
    duration: 30
  });

  // Multi-instance state
  const [multiInstanceConfig, setMultiInstanceConfig] = useState({
    instanceCount: 3,
    symbols: ['BTC/USDT', 'ETH/USDT', 'ADA/USDT'],
    strategies: ['trend_following', 'mean_reversion'],
    parallelExecution: true,
    resourceLimit: 80
  });

  // Handle A/B test creation
  const handleCreateABTest = useCallback(() => {
    if (!configuration) return;

    const newABTest: ABTest = {
      id: `ab_${Date.now()}`,
      name: abTestForm.name,
      configurations: {
        control: {
          ...configuration,
          name: abTestForm.controlName,
          riskSettings: {
            ...configuration.riskSettings,
            [abTestForm.testParameter]: abTestForm.controlValue
          }
        },
        treatment: {
          ...configuration,
          name: abTestForm.treatmentName,
          riskSettings: {
            ...configuration.riskSettings,
            [abTestForm.testParameter]: abTestForm.treatmentValue
          }
        }
      },
      executions: {} as any,
      status: 'setup',
      createdAt: new Date().toISOString()
    };

    setAbTests(prev => [...prev, newABTest]);
    setCreateABTestOpen(false);
  }, [configuration, abTestForm]);

  // Handle parameter optimization
  const handleStartOptimization = useCallback(() => {
    setIsOptimizing(true);
    setOptimizationProgress(0);

    // Simulate optimization process
    const interval = setInterval(() => {
      setOptimizationProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsOptimizing(false);
          return 100;
        }
        return prev + Math.random() * 10;
      });
    }, 500);
  }, []);

  // Get significance badge color
  const getSignificanceBadgeColor = (pValue: number, alpha: number = 0.05) => {
    if (pValue < alpha) return 'success';
    if (pValue < alpha * 2) return 'warning';
    return 'error';
  };

  // Format parameter value
  const formatParameterValue = (param: ParameterOptimization, value: number) => {
    if (param.parameter.includes('Percentage')) {
      return `${value.toFixed(1)}%`;
    }
    return value.toFixed(2);
  };

  return (
    <Box>
      {/* Header */}
      <Typography variant="h5" fontWeight="bold" gutterBottom sx={{ mb: 3 }}>
        Advanced Features
      </Typography>

      <Grid container spacing={3}>
        {/* A/B Testing Section */}
        <Grid item xs={12} lg={6}>
          <Card elevation={2} sx={{ height: 'fit-content' }}>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <ScienceIcon color="primary" />
                  A/B Testing
                </Typography>
                <Button
                  variant="outlined"
                  size="small"
                  startIcon={<AddIcon />}
                  onClick={() => setCreateABTestOpen(true)}
                  disabled={!configuration}
                >
                  Create Test
                </Button>
              </Box>

              <Typography variant="body2" color="text.secondary" paragraph>
                Compare different configurations to find the optimal settings scientifically.
              </Typography>

              {abTests.length === 0 ? (
                <Alert severity="info" sx={{ mb: 2 }}>
                  No A/B tests created yet. Create your first test to compare different configurations.
                </Alert>
              ) : (
                <List>
                  {abTests.map((test, index) => (
                    <React.Fragment key={test.id}>
                      <ListItem>
                        <ListItemIcon>
                          <CompareIcon />
                        </ListItemIcon>
                        <ListItemText
                          primary={test.name}
                          secondary={
                            <Box>
                              <Typography variant="caption" display="block">
                                Status: {test.status}
                              </Typography>
                              {test.results && (
                                <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
                                  <Chip
                                    size="small"
                                    label={`p-value: ${test.results.pValue.toFixed(3)}`}
                                    color={getSignificanceBadgeColor(test.results.pValue)}
                                  />
                                  <Chip
                                    size="small"
                                    label={`Winner: ${test.results.winner}`}
                                    color="primary"
                                    variant="outlined"
                                  />
                                </Box>
                              )}
                            </Box>
                          }
                        />
                        <ListItemSecondaryAction>
                          <IconButton
                            size="small"
                            onClick={() => setSelectedABTest(test)}
                          >
                            <AssessmentIcon />
                          </IconButton>
                        </ListItemSecondaryAction>
                      </ListItem>
                      {index < abTests.length - 1 && <Divider variant="inset" component="li" />}
                    </React.Fragment>
                  ))}
                </List>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Parameter Optimization Section */}
        <Grid item xs={12} lg={6}>
          <Card elevation={2} sx={{ height: 'fit-content' }}>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <TuneIcon color="primary" />
                  Parameter Optimization
                </Typography>
                <Button
                  variant="outlined"
                  size="small"
                  startIcon={<OptimizeIcon />}
                  onClick={() => setOptimizationOpen(true)}
                  disabled={!configuration}
                >
                  Optimize
                </Button>
              </Box>

              <Typography variant="body2" color="text.secondary" paragraph>
                Find optimal parameter values using systematic search algorithms.
              </Typography>

              {isOptimizing && (
                <Box sx={{ mb: 2 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2">Optimization Progress</Typography>
                    <Typography variant="body2">{Math.round(optimizationProgress)}%</Typography>
                  </Box>
                  <LinearProgress
                    variant="determinate"
                    value={optimizationProgress}
                    sx={{ height: 8, borderRadius: 4 }}
                  />
                </Box>
              )}

              <List>
                {parameters.map((param, index) => (
                  <React.Fragment key={param.parameter}>
                    <ListItem>
                      <ListItemText
                        primary={param.parameter}
                        secondary={
                          <Box>
                            <Typography variant="caption" display="block">
                              Current: {formatParameterValue(param, param.current || 0)} → 
                              Optimal: {formatParameterValue(param, param.optimal || 0)}
                            </Typography>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 1 }}>
                              <Typography variant="caption">Sensitivity:</Typography>
                              <LinearProgress
                                variant="determinate"
                                value={(param.sensitivity || 0) * 100}
                                sx={{ flexGrow: 1, height: 4, borderRadius: 2 }}
                              />
                              <Typography variant="caption">
                                {((param.sensitivity || 0) * 100).toFixed(0)}%
                              </Typography>
                            </Box>
                          </Box>
                        }
                      />
                    </ListItem>
                    {index < parameters.length - 1 && <Divider variant="inset" component="li" />}
                  </React.Fragment>
                ))}
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* Multi-Instance Execution */}
        <Grid item xs={12}>
          <Card elevation={2}>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <SpeedIcon color="primary" />
                  Multi-Instance Execution
                </Typography>
                <Button
                  variant="outlined"
                  size="small"
                  startIcon={<PlayIcon />}
                  onClick={() => setMultiInstanceOpen(true)}
                  disabled={!configuration}
                >
                  Configure
                </Button>
              </Box>

              <Typography variant="body2" color="text.secondary" paragraph>
                Run multiple configurations simultaneously to compare performance across different parameters.
              </Typography>

              <Grid container spacing={2}>
                <Grid item xs={12} sm={6} md={3}>
                  <Card variant="outlined">
                    <CardContent sx={{ textAlign: 'center' }}>
                      <Typography variant="h4" color="primary" fontWeight="bold">
                        {multiInstanceConfig.instanceCount}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Active Instances
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Card variant="outlined">
                    <CardContent sx={{ textAlign: 'center' }}>
                      <Typography variant="h4" color="primary" fontWeight="bold">
                        {multiInstanceConfig.symbols.length}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Trading Pairs
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Card variant="outlined">
                    <CardContent sx={{ textAlign: 'center' }}>
                      <Typography variant="h4" color="primary" fontWeight="bold">
                        {multiInstanceConfig.strategies.length}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Strategies
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Card variant="outlined">
                    <CardContent sx={{ textAlign: 'center' }}>
                      <Typography variant="h4" color="primary" fontWeight="bold">
                        {multiInstanceConfig.resourceLimit}%
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Resource Limit
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>

              {multiInstanceConfig.parallelExecution && (
                <Alert severity="warning" sx={{ mt: 2 }}>
                  <Typography variant="body2">
                    <strong>Warning:</strong> Parallel execution will consume significant system resources. 
                    Monitor CPU and memory usage during execution.
                  </Typography>
                </Alert>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Sensitivity Analysis */}
        <Grid item xs={12}>
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <PsychologyIcon color="primary" />
                Sensitivity Analysis
              </Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Typography variant="body2" color="text.secondary" paragraph>
                Analyze how sensitive your strategy is to parameter changes. Higher sensitivity indicates 
                parameters that require careful tuning.
              </Typography>
              
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle1" gutterBottom>
                    Parameter Sensitivity Ranking
                  </Typography>
                  <TableContainer>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>Parameter</TableCell>
                          <TableCell align="right">Sensitivity</TableCell>
                          <TableCell align="right">Impact</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {parameters
                          .sort((a, b) => (b.sensitivity || 0) - (a.sensitivity || 0))
                          .map((param) => (
                            <TableRow key={param.parameter}>
                              <TableCell>{param.parameter}</TableCell>
                              <TableCell align="right">
                                {((param.sensitivity || 0) * 100).toFixed(1)}%
                              </TableCell>
                              <TableCell align="right">
                                <Chip
                                  size="small"
                                  label={
                                    (param.sensitivity || 0) > 0.7 ? 'High' :
                                    (param.sensitivity || 0) > 0.4 ? 'Medium' : 'Low'
                                  }
                                  color={
                                    (param.sensitivity || 0) > 0.7 ? 'error' :
                                    (param.sensitivity || 0) > 0.4 ? 'warning' : 'success'
                                  }
                                  variant="outlined"
                                />
                              </TableCell>
                            </TableRow>
                          ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </Grid>

                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle1" gutterBottom>
                    Optimization Recommendations
                  </Typography>
                  <List>
                    <ListItem>
                      <ListItemIcon>
                        <CheckCircleIcon color="success" />
                      </ListItemIcon>
                      <ListItemText
                        primary="Position Size Optimization"
                        secondary="High sensitivity detected. Consider using Kelly Criterion for optimal sizing."
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon>
                        <WarningIcon color="warning" />
                      </ListItemIcon>
                      <ListItemText
                        primary="Stop Loss Tuning"
                        secondary="Medium sensitivity. Test values between 1.5% and 3.5% for optimal results."
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon>
                        <CheckCircleIcon color="success" />
                      </ListItemIcon>
                      <ListItemText
                        primary="Take Profit Settings"
                        secondary="Low sensitivity. Current settings are robust across different market conditions."
                      />
                    </ListItem>
                  </List>
                </Grid>
              </Grid>
            </AccordionDetails>
          </Accordion>
        </Grid>
      </Grid>

      {/* Create A/B Test Dialog */}
      <Dialog
        open={createABTestOpen}
        onClose={() => setCreateABTestOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Create A/B Test</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Test Name"
                value={abTestForm.name}
                onChange={(e) => setAbTestForm(prev => ({ ...prev, name: e.target.value }))}
                required
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                multiline
                rows={2}
                label="Description"
                value={abTestForm.description}
                onChange={(e) => setAbTestForm(prev => ({ ...prev, description: e.target.value }))}
              />
            </Grid>
            <Grid item xs={6}>
              <TextField
                fullWidth
                label="Control Name"
                value={abTestForm.controlName}
                onChange={(e) => setAbTestForm(prev => ({ ...prev, controlName: e.target.value }))}
              />
            </Grid>
            <Grid item xs={6}>
              <TextField
                fullWidth
                label="Treatment Name"
                value={abTestForm.treatmentName}
                onChange={(e) => setAbTestForm(prev => ({ ...prev, treatmentName: e.target.value }))}
              />
            </Grid>
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>Test Parameter</InputLabel>
                <Select
                  value={abTestForm.testParameter}
                  label="Test Parameter"
                  onChange={(e) => setAbTestForm(prev => ({ ...prev, testParameter: e.target.value }))}
                >
                  <MenuItem value="stopLossPercentage">Stop Loss %</MenuItem>
                  <MenuItem value="takeProfitPercentage">Take Profit %</MenuItem>
                  <MenuItem value="positionSize">Position Size</MenuItem>
                  <MenuItem value="maxDrawdownPercentage">Max Drawdown %</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={6}>
              <TextField
                fullWidth
                type="number"
                label="Control Value"
                value={abTestForm.controlValue}
                onChange={(e) => setAbTestForm(prev => ({ ...prev, controlValue: Number(e.target.value) }))}
                inputProps={{ step: 0.1, min: 0 }}
              />
            </Grid>
            <Grid item xs={6}>
              <TextField
                fullWidth
                type="number"
                label="Treatment Value"
                value={abTestForm.treatmentValue}
                onChange={(e) => setAbTestForm(prev => ({ ...prev, treatmentValue: Number(e.target.value) }))}
                inputProps={{ step: 0.1, min: 0 }}
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateABTestOpen(false)}>Cancel</Button>
          <Button
            onClick={handleCreateABTest}
            variant="contained"
            disabled={!abTestForm.name || !configuration}
          >
            Create Test
          </Button>
        </DialogActions>
      </Dialog>

      {/* Parameter Optimization Dialog */}
      <Dialog
        open={optimizationOpen}
        onClose={() => setOptimizationOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Parameter Optimization</DialogTitle>
        <DialogContent>
          <Grid container spacing={3} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>Optimization Metric</InputLabel>
                <Select
                  value={optimizationMetric}
                  label="Optimization Metric"
                  onChange={(e) => setOptimizationMetric(e.target.value)}
                >
                  <MenuItem value="sharpe_ratio">Sharpe Ratio</MenuItem>
                  <MenuItem value="return">Total Return</MenuItem>
                  <MenuItem value="calmar_ratio">Calmar Ratio</MenuItem>
                  <MenuItem value="sortino_ratio">Sortino Ratio</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            {parameters.map((param) => (
              <Grid item xs={12} key={param.parameter}>
                <Paper variant="outlined" sx={{ p: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    {param.parameter}
                  </Typography>
                  {param.type === 'range' && (
                    <Box sx={{ px: 2 }}>
                      <Slider
                        value={[param.min || 0, param.max || 10]}
                        onChange={(_, value) => {
                          const [min, max] = value as number[];
                          setParameters(prev => prev.map(p => 
                            p.parameter === param.parameter 
                              ? { ...p, min, max }
                              : p
                          ));
                        }}
                        min={0}
                        max={20}
                        step={param.step || 0.1}
                        marks
                        valueLabelDisplay="auto"
                        valueLabelFormat={(value) => formatParameterValue(param, value)}
                      />
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
                        <Typography variant="caption">Min: {formatParameterValue(param, param.min || 0)}</Typography>
                        <Typography variant="caption">Max: {formatParameterValue(param, param.max || 10)}</Typography>
                      </Box>
                    </Box>
                  )}
                </Paper>
              </Grid>
            ))}

            <Grid item xs={12}>
              <Alert severity="info">
                Optimization will test multiple parameter combinations to find the optimal settings 
                based on the selected metric. This process may take several minutes.
              </Alert>
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOptimizationOpen(false)}>Cancel</Button>
          <Button
            onClick={() => {
              handleStartOptimization();
              setOptimizationOpen(false);
            }}
            variant="contained"
            disabled={isOptimizing}
          >
            Start Optimization
          </Button>
        </DialogActions>
      </Dialog>

      {/* Multi-Instance Configuration Dialog */}
      <Dialog
        open={multiInstanceOpen}
        onClose={() => setMultiInstanceOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Multi-Instance Configuration</DialogTitle>
        <DialogContent>
          <Grid container spacing={3} sx={{ mt: 1 }}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                type="number"
                label="Number of Instances"
                value={multiInstanceConfig.instanceCount}
                onChange={(e) => setMultiInstanceConfig(prev => ({
                  ...prev,
                  instanceCount: Number(e.target.value)
                }))}
                inputProps={{ min: 1, max: 10 }}
                helperText="Maximum 10 instances recommended"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                type="number"
                label="Resource Limit (%)"
                value={multiInstanceConfig.resourceLimit}
                onChange={(e) => setMultiInstanceConfig(prev => ({
                  ...prev,
                  resourceLimit: Number(e.target.value)
                }))}
                inputProps={{ min: 10, max: 100 }}
                helperText="CPU/Memory usage limit"
              />
            </Grid>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={multiInstanceConfig.parallelExecution}
                    onChange={(e) => setMultiInstanceConfig(prev => ({
                      ...prev,
                      parallelExecution: e.target.checked
                    }))}
                  />
                }
                label="Enable parallel execution"
              />
              <Typography variant="body2" color="text.secondary">
                Run instances simultaneously for faster results (requires more resources)
              </Typography>
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setMultiInstanceOpen(false)}>Cancel</Button>
          <Button variant="contained">
            Start Multi-Instance Execution
          </Button>
        </DialogActions>
      </Dialog>

      {/* A/B Test Results Dialog */}
      {selectedABTest && (
        <Dialog
          open={!!selectedABTest}
          onClose={() => setSelectedABTest(null)}
          maxWidth="md"
          fullWidth
        >
          <DialogTitle>{selectedABTest.name} - Results</DialogTitle>
          <DialogContent>
            {selectedABTest.results ? (
              <Grid container spacing={3}>
                <Grid item xs={12}>
                  <Alert severity={selectedABTest.results.pValue < 0.05 ? 'success' : 'warning'}>
                    <Typography variant="body1" fontWeight="bold">
                      {selectedABTest.results.winner === 'treatment' ? 'Treatment' : 'Control'} configuration wins!
                    </Typography>
                    <Typography variant="body2">
                      P-value: {selectedABTest.results.pValue.toFixed(3)} 
                      {selectedABTest.results.pValue < 0.05 ? ' (Statistically significant)' : ' (Not significant)'}
                    </Typography>
                  </Alert>
                </Grid>
                <Grid item xs={12} md={6}>
                  <Typography variant="h6" gutterBottom>Control Configuration</Typography>
                  <Typography variant="body2" paragraph>
                    {selectedABTest.configurations.control.name}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {selectedABTest.configurations.control.description}
                  </Typography>
                </Grid>
                <Grid item xs={12} md={6}>
                  <Typography variant="h6" gutterBottom>Treatment Configuration</Typography>
                  <Typography variant="body2" paragraph>
                    {selectedABTest.configurations.treatment.name}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {selectedABTest.configurations.treatment.description}
                  </Typography>
                </Grid>
                <Grid item xs={12}>
                  <Typography variant="h6" gutterBottom>Statistical Results</Typography>
                  <TableContainer>
                    <Table>
                      <TableBody>
                        <TableRow>
                          <TableCell>Effect Size</TableCell>
                          <TableCell align="right">
                            {selectedABTest.results.effect.magnitude.toFixed(2)}%
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Confidence Interval</TableCell>
                          <TableCell align="right">
                            [{selectedABTest.results.confidenceInterval[0].toFixed(2)}%, {selectedABTest.results.confidenceInterval[1].toFixed(2)}%]
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Significance Level</TableCell>
                          <TableCell align="right">
                            {(selectedABTest.results.significanceLevel * 100).toFixed(0)}%
                          </TableCell>
                        </TableRow>
                      </TableBody>
                    </Table>
                  </TableContainer>
                </Grid>
              </Grid>
            ) : (
              <Typography>No results available yet. Run the test to see results.</Typography>
            )}
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setSelectedABTest(null)}>Close</Button>
            {!selectedABTest.results && (
              <Button variant="contained">
                Run A/B Test
              </Button>
            )}
          </DialogActions>
        </Dialog>
      )}
    </Box>
  );
};

export default AdvancedFeatures;