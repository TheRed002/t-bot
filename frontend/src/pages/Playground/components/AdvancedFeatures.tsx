/**
 * Advanced Features Component for Playground
 * A/B testing, parameter optimization, and multi-instance execution
 */

import React, { useState, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Alert } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { Switch } from '@/components/ui/switch';
import { Slider } from '@/components/ui/slider';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Textarea } from '@/components/ui/textarea';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Separator } from '@/components/ui/separator';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';

import {
  Beaker,
  ArrowLeftRight,
  Settings2,
  Gauge,
  Plus,
  Edit,
  Trash2,
  Play,
  ChevronDown,
  BarChart3,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  Brain,
  Sparkles,
  Activity
} from 'lucide-react';

import {
  PlaygroundConfiguration,
  PlaygroundExecution,
  ABTest,
  ParameterOptimization,
  PositionSizeMethod,
  AllocationStrategy,
  StrategyType
} from '@/types';
import { cn } from '@/lib/utils';

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
        positionSizing: { method: PositionSizeMethod.PERCENTAGE, value: 2, maxPositions: 5 },
        tradingSide: 'both',
        riskSettings: {
          stopLossPercentage: 2,
          takeProfitPercentage: 4,
          maxDrawdownPercentage: 10,
          maxRiskPerTrade: 2,
          correlationLimit: 0.7,
          enableCircuitBreaker: true,
          dailyLossLimit: 5
        },
        portfolioSettings: {
          maxPositions: 5,
          allocationStrategy: AllocationStrategy.EQUAL_WEIGHT,
          rebalanceFrequency: 'daily'
        },
        strategyTemplate: { 
          templateId: 'trend_following_template', 
          strategyType: StrategyType.TREND_FOLLOWING, 
          parameters: {} 
        },
        timeframe: '1h'
      },
      treatment: {
        name: 'Aggressive SL (3%)',
        description: 'Higher risk for better returns',
        symbols: ['BTC/USDT'],
        positionSizing: { method: PositionSizeMethod.PERCENTAGE, value: 2, maxPositions: 5 },
        tradingSide: 'both',
        riskSettings: {
          stopLossPercentage: 3,
          takeProfitPercentage: 6,
          maxDrawdownPercentage: 15,
          maxRiskPerTrade: 3,
          correlationLimit: 0.7,
          enableCircuitBreaker: true,
          dailyLossLimit: 8
        },
        portfolioSettings: {
          maxPositions: 5,
          allocationStrategy: AllocationStrategy.EQUAL_WEIGHT,
          rebalanceFrequency: 'daily'
        },
        strategyTemplate: { 
          templateId: 'trend_following_template', 
          strategyType: StrategyType.TREND_FOLLOWING, 
          parameters: {} 
        },
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
  const [sensitivityExpanded, setSensitivityExpanded] = useState<boolean>(false);

  // A/B Test Creation Form State
  interface ABTestFormData {
    name: string;
    description: string;
    controlName: string;
    treatmentName: string;
    testParameter: string;
    controlValue: number;
    treatmentValue: number;
    significance: number;
    duration: number;
  }
  
  const [abTestForm, setABTestForm] = useState<ABTestFormData>({
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
  const getSignificanceBadgeVariant = (pValue: number, alpha: number = 0.05) => {
    if (pValue < alpha) return 'default';
    if (pValue < alpha * 2) return 'secondary';
    return 'destructive';
  };

  // Format parameter value
  const formatParameterValue = (param: ParameterOptimization, value: number) => {
    if (param.parameter.includes('Percentage')) {
      return `${value.toFixed(1)}%`;
    }
    return value.toFixed(2);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold mb-2">Advanced Features</h2>
        <p className="text-muted-foreground">
          A/B testing, parameter optimization, and multi-instance execution tools
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* A/B Testing Section */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-2">
                <Beaker className="h-5 w-5 text-primary" />
                A/B Testing
              </CardTitle>
              <Dialog open={createABTestOpen} onOpenChange={setCreateABTestOpen}>
                <DialogTrigger asChild>
                  <Button variant="outline" size="sm" disabled={!configuration}>
                    <Plus className="h-4 w-4 mr-2" />
                    Create Test
                  </Button>
                </DialogTrigger>
                <DialogContent className="sm:max-w-md">
                  <DialogHeader>
                    <DialogTitle>Create A/B Test</DialogTitle>
                    <DialogDescription>
                      Compare different configurations to find optimal settings scientifically.
                    </DialogDescription>
                  </DialogHeader>
                  <div className="space-y-4">
                    <div>
                      <Label htmlFor="test-name">Test Name</Label>
                      <Input
                        id="test-name"
                        value={abTestForm.name}
                        onChange={(e) => setABTestForm(prev => ({ ...prev, name: e.target.value }))}
                        placeholder="Stop Loss Comparison"
                        required
                      />
                    </div>
                    <div>
                      <Label htmlFor="description">Description</Label>
                      <Textarea
                        id="description"
                        value={abTestForm.description}
                        onChange={(e) => setABTestForm(prev => ({ ...prev, description: e.target.value }))}
                        placeholder="Test different stop loss values..."
                        rows={2}
                      />
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <Label htmlFor="control-name">Control Name</Label>
                        <Input
                          id="control-name"
                          value={abTestForm.controlName}
                          onChange={(e) => setABTestForm(prev => ({ ...prev, controlName: e.target.value }))}
                          placeholder="Control"
                        />
                      </div>
                      <div>
                        <Label htmlFor="treatment-name">Treatment Name</Label>
                        <Input
                          id="treatment-name"
                          value={abTestForm.treatmentName}
                          onChange={(e) => setABTestForm(prev => ({ ...prev, treatmentName: e.target.value }))}
                          placeholder="Treatment"
                        />
                      </div>
                    </div>
                    <div>
                      <Label>Test Parameter</Label>
                      <Select value={abTestForm.testParameter} onValueChange={(value) => setABTestForm(prev => ({ ...prev, testParameter: value }))}>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="stopLossPercentage">Stop Loss %</SelectItem>
                          <SelectItem value="takeProfitPercentage">Take Profit %</SelectItem>
                          <SelectItem value="positionSize">Position Size</SelectItem>
                          <SelectItem value="maxDrawdownPercentage">Max Drawdown %</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <Label htmlFor="control-value">Control Value</Label>
                        <Input
                          id="control-value"
                          type="number"
                          value={abTestForm.controlValue}
                          onChange={(e) => setABTestForm(prev => ({ ...prev, controlValue: Number(e.target.value) }))}
                          step="0.1"
                          min="0"
                        />
                      </div>
                      <div>
                        <Label htmlFor="treatment-value">Treatment Value</Label>
                        <Input
                          id="treatment-value"
                          type="number"
                          value={abTestForm.treatmentValue}
                          onChange={(e) => setABTestForm(prev => ({ ...prev, treatmentValue: Number(e.target.value) }))}
                          step="0.1"
                          min="0"
                        />
                      </div>
                    </div>
                    <div className="flex justify-end gap-2">
                      <Button variant="outline" onClick={() => setCreateABTestOpen(false)}>Cancel</Button>
                      <Button onClick={handleCreateABTest} disabled={!abTestForm.name || !configuration}>
                        Create Test
                      </Button>
                    </div>
                  </div>
                </DialogContent>
              </Dialog>
            </div>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground mb-4">
              Compare different configurations to find the optimal settings scientifically.
            </p>

            {abTests.length === 0 ? (
              <Alert>
                <Activity className="h-4 w-4" />
                <div>
                  <div className="font-medium">No A/B tests created yet</div>
                  <div className="text-sm">Create your first test to compare different configurations.</div>
                </div>
              </Alert>
            ) : (
              <div className="space-y-3">
                {abTests.map((test) => (
                  <Card key={test.id} className="border">
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between">
                        <div className="flex items-start gap-3">
                          <ArrowLeftRight className="h-4 w-4 mt-1 text-muted-foreground" />
                          <div>
                            <div className="font-medium">{test.name}</div>
                            <div className="text-sm text-muted-foreground">Status: {test.status}</div>
                            {test.results && (
                              <div className="flex gap-2 mt-2">
                                <Badge variant={getSignificanceBadgeVariant(test.results.pValue)}>
                                  p-value: {test.results.pValue.toFixed(3)}
                                </Badge>
                                <Badge variant="outline">
                                  Winner: {test.results.winner}
                                </Badge>
                              </div>
                            )}
                          </div>
                        </div>
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={() => setSelectedABTest(test)}
                        >
                          <BarChart3 className="h-4 w-4" />
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Parameter Optimization Section */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-2">
                <Settings2 className="h-5 w-5 text-primary" />
                Parameter Optimization
              </CardTitle>
              <Dialog open={optimizationOpen} onOpenChange={setOptimizationOpen}>
                <DialogTrigger asChild>
                  <Button variant="outline" size="sm" disabled={!configuration}>
                    <Sparkles className="h-4 w-4 mr-2" />
                    Optimize
                  </Button>
                </DialogTrigger>
                <DialogContent className="sm:max-w-2xl">
                  <DialogHeader>
                    <DialogTitle>Parameter Optimization</DialogTitle>
                    <DialogDescription>
                      Find optimal parameter values using systematic search algorithms.
                    </DialogDescription>
                  </DialogHeader>
                  <div className="space-y-6">
                    <div>
                      <Label>Optimization Metric</Label>
                      <Select value={optimizationMetric} onValueChange={setOptimizationMetric}>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="sharpe_ratio">Sharpe Ratio</SelectItem>
                          <SelectItem value="return">Total Return</SelectItem>
                          <SelectItem value="calmar_ratio">Calmar Ratio</SelectItem>
                          <SelectItem value="sortino_ratio">Sortino Ratio</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    {parameters.map((param) => (
                      <Card key={param.parameter} className="border">
                        <CardContent className="p-4">
                          <div className="space-y-4">
                            <Label>{param.parameter}</Label>
                            {param.type === 'range' && (
                              <div className="px-2">
                                <Slider
                                  value={[param.min || 0, param.max || 10]}
                                  onValueChange={([min, max]) => {
                                    setParameters(prev => prev.map(p => 
                                      p.parameter === param.parameter 
                                        ? { ...p, min, max }
                                        : p
                                    ));
                                  }}
                                  min={0}
                                  max={20}
                                  step={param.step || 0.1}
                                  className="w-full"
                                />
                                <div className="flex justify-between text-sm text-muted-foreground mt-2">
                                  <span>Min: {formatParameterValue(param, param.min || 0)}</span>
                                  <span>Max: {formatParameterValue(param, param.max || 10)}</span>
                                </div>
                              </div>
                            )}
                          </div>
                        </CardContent>
                      </Card>
                    ))}

                    <Alert>
                      <Activity className="h-4 w-4" />
                      <div>
                        <div className="font-medium">Note</div>
                        <div className="text-sm">
                          Optimization will test multiple parameter combinations to find the optimal settings 
                          based on the selected metric. This process may take several minutes.
                        </div>
                      </div>
                    </Alert>

                    <div className="flex justify-end gap-2">
                      <Button variant="outline" onClick={() => setOptimizationOpen(false)}>Cancel</Button>
                      <Button
                        onClick={() => {
                          handleStartOptimization();
                          setOptimizationOpen(false);
                        }}
                        disabled={isOptimizing}
                      >
                        Start Optimization
                      </Button>
                    </div>
                  </div>
                </DialogContent>
              </Dialog>
            </div>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground mb-4">
              Find optimal parameter values using systematic search algorithms.
            </p>

            {isOptimizing && (
              <div className="mb-4">
                <div className="flex justify-between text-sm mb-2">
                  <span>Optimization Progress</span>
                  <span>{Math.round(optimizationProgress)}%</span>
                </div>
                <Progress value={optimizationProgress} className="h-2" />
              </div>
            )}

            <div className="space-y-3">
              {parameters.map((param) => (
                <Card key={param.parameter} className="border">
                  <CardContent className="p-4">
                    <div className="space-y-2">
                      <div className="font-medium">{param.parameter}</div>
                      <div className="text-sm text-muted-foreground">
                        Current: {formatParameterValue(param, param.current || 0)} → 
                        Optimal: {formatParameterValue(param, param.optimal || 0)}
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-muted-foreground">Sensitivity:</span>
                        <Progress 
                          value={(param.sensitivity || 0) * 100} 
                          className="flex-1 h-1" 
                        />
                        <span className="text-xs text-muted-foreground">
                          {((param.sensitivity || 0) * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Multi-Instance Execution */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <Gauge className="h-5 w-5 text-primary" />
              Multi-Instance Execution
            </CardTitle>
            <Dialog open={multiInstanceOpen} onOpenChange={setMultiInstanceOpen}>
              <DialogTrigger asChild>
                <Button variant="outline" size="sm" disabled={!configuration}>
                  <Play className="h-4 w-4 mr-2" />
                  Configure
                </Button>
              </DialogTrigger>
              <DialogContent className="sm:max-w-md">
                <DialogHeader>
                  <DialogTitle>Multi-Instance Configuration</DialogTitle>
                  <DialogDescription>
                    Configure multiple strategy instances to run simultaneously.
                  </DialogDescription>
                </DialogHeader>
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="instance-count">Number of Instances</Label>
                      <Input
                        id="instance-count"
                        type="number"
                        value={multiInstanceConfig.instanceCount}
                        onChange={(e) => setMultiInstanceConfig(prev => ({
                          ...prev,
                          instanceCount: Number(e.target.value)
                        }))}
                        min="1"
                        max="10"
                      />
                      <p className="text-xs text-muted-foreground mt-1">Maximum 10 instances recommended</p>
                    </div>
                    <div>
                      <Label htmlFor="resource-limit">Resource Limit (%)</Label>
                      <Input
                        id="resource-limit"
                        type="number"
                        value={multiInstanceConfig.resourceLimit}
                        onChange={(e) => setMultiInstanceConfig(prev => ({
                          ...prev,
                          resourceLimit: Number(e.target.value)
                        }))}
                        min="10"
                        max="100"
                      />
                      <p className="text-xs text-muted-foreground mt-1">CPU/Memory usage limit</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Switch
                      id="parallel-execution"
                      checked={multiInstanceConfig.parallelExecution}
                      onCheckedChange={(checked) => setMultiInstanceConfig(prev => ({
                        ...prev,
                        parallelExecution: checked
                      }))}
                    />
                    <div>
                      <Label htmlFor="parallel-execution">Enable parallel execution</Label>
                      <p className="text-xs text-muted-foreground">
                        Run instances simultaneously for faster results (requires more resources)
                      </p>
                    </div>
                  </div>
                  <div className="flex justify-end gap-2">
                    <Button variant="outline" onClick={() => setMultiInstanceOpen(false)}>Cancel</Button>
                    <Button>Start Multi-Instance Execution</Button>
                  </div>
                </div>
              </DialogContent>
            </Dialog>
          </div>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground mb-4">
            Run multiple configurations simultaneously to compare performance across different parameters.
          </p>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
            <Card className="text-center">
              <CardContent className="p-4">
                <div className="text-2xl font-bold text-primary">{multiInstanceConfig.instanceCount}</div>
                <div className="text-sm text-muted-foreground">Active Instances</div>
              </CardContent>
            </Card>
            <Card className="text-center">
              <CardContent className="p-4">
                <div className="text-2xl font-bold text-primary">{multiInstanceConfig.symbols.length}</div>
                <div className="text-sm text-muted-foreground">Trading Pairs</div>
              </CardContent>
            </Card>
            <Card className="text-center">
              <CardContent className="p-4">
                <div className="text-2xl font-bold text-primary">{multiInstanceConfig.strategies.length}</div>
                <div className="text-sm text-muted-foreground">Strategies</div>
              </CardContent>
            </Card>
            <Card className="text-center">
              <CardContent className="p-4">
                <div className="text-2xl font-bold text-primary">{multiInstanceConfig.resourceLimit}%</div>
                <div className="text-sm text-muted-foreground">Resource Limit</div>
              </CardContent>
            </Card>
          </div>

          {multiInstanceConfig.parallelExecution && (
            <Alert>
              <AlertTriangle className="h-4 w-4" />
              <div>
                <div className="font-medium">Warning</div>
                <div className="text-sm">
                  Parallel execution will consume significant system resources. 
                  Monitor CPU and memory usage during execution.
                </div>
              </div>
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Sensitivity Analysis */}
      <Collapsible open={sensitivityExpanded} onOpenChange={setSensitivityExpanded}>
        <CollapsibleTrigger asChild>
          <Card className="cursor-pointer">
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Brain className="h-5 w-5 text-primary" />
                  Sensitivity Analysis
                </div>
                <ChevronDown className={cn("h-4 w-4 transition-transform", sensitivityExpanded && "rotate-180")} />
              </CardTitle>
            </CardHeader>
          </Card>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <Card>
            <CardContent className="p-6">
              <p className="text-sm text-muted-foreground mb-4">
                Analyze how sensitive your strategy is to parameter changes. Higher sensitivity indicates 
                parameters that require careful tuning.
              </p>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h3 className="text-lg font-medium mb-4">Parameter Sensitivity Ranking</h3>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Parameter</TableHead>
                        <TableHead className="text-right">Sensitivity</TableHead>
                        <TableHead className="text-right">Impact</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {parameters
                        .sort((a, b) => (b.sensitivity || 0) - (a.sensitivity || 0))
                        .map((param) => (
                          <TableRow key={param.parameter}>
                            <TableCell>{param.parameter}</TableCell>
                            <TableCell className="text-right">
                              {((param.sensitivity || 0) * 100).toFixed(1)}%
                            </TableCell>
                            <TableCell className="text-right">
                              <Badge
                                variant={
                                  (param.sensitivity || 0) > 0.7 ? 'destructive' :
                                  (param.sensitivity || 0) > 0.4 ? 'secondary' : 'default'
                                }
                              >
                                {(param.sensitivity || 0) > 0.7 ? 'High' :
                                 (param.sensitivity || 0) > 0.4 ? 'Medium' : 'Low'}
                              </Badge>
                            </TableCell>
                          </TableRow>
                        ))}
                    </TableBody>
                  </Table>
                </div>

                <div>
                  <h3 className="text-lg font-medium mb-4">Optimization Recommendations</h3>
                  <div className="space-y-4">
                    <div className="flex gap-3">
                      <CheckCircle className="h-5 w-5 text-green-500 mt-0.5 flex-shrink-0" />
                      <div>
                        <div className="font-medium">Position Size Optimization</div>
                        <div className="text-sm text-muted-foreground">
                          High sensitivity detected. Consider using Kelly Criterion for optimal sizing.
                        </div>
                      </div>
                    </div>
                    <div className="flex gap-3">
                      <AlertTriangle className="h-5 w-5 text-yellow-500 mt-0.5 flex-shrink-0" />
                      <div>
                        <div className="font-medium">Stop Loss Tuning</div>
                        <div className="text-sm text-muted-foreground">
                          Medium sensitivity. Test values between 1.5% and 3.5% for optimal results.
                        </div>
                      </div>
                    </div>
                    <div className="flex gap-3">
                      <CheckCircle className="h-5 w-5 text-green-500 mt-0.5 flex-shrink-0" />
                      <div>
                        <div className="font-medium">Take Profit Settings</div>
                        <div className="text-sm text-muted-foreground">
                          Low sensitivity. Current settings are robust across different market conditions.
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </CollapsibleContent>
      </Collapsible>

      {/* A/B Test Results Dialog */}
      {selectedABTest && (
        <Dialog open={!!selectedABTest} onOpenChange={() => setSelectedABTest(null)}>
          <DialogContent className="sm:max-w-2xl">
            <DialogHeader>
              <DialogTitle>{selectedABTest.name} - Results</DialogTitle>
            </DialogHeader>
            {selectedABTest.results ? (
              <div className="space-y-6">
                <Alert className={selectedABTest.results.pValue < 0.05 ? 'border-green-200' : 'border-yellow-200'}>
                  <CheckCircle className="h-4 w-4" />
                  <div>
                    <div className="font-medium">
                      {selectedABTest.results.winner === 'treatment' ? 'Treatment' : 'Control'} configuration wins!
                    </div>
                    <div className="text-sm">
                      P-value: {selectedABTest.results.pValue.toFixed(3)} 
                      {selectedABTest.results.pValue < 0.05 ? ' (Statistically significant)' : ' (Not significant)'}
                    </div>
                  </div>
                </Alert>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h3 className="text-lg font-medium mb-2">Control Configuration</h3>
                    <p className="text-sm font-medium">{selectedABTest.configurations.control.name}</p>
                    <p className="text-sm text-muted-foreground">{selectedABTest.configurations.control.description}</p>
                  </div>
                  <div>
                    <h3 className="text-lg font-medium mb-2">Treatment Configuration</h3>
                    <p className="text-sm font-medium">{selectedABTest.configurations.treatment.name}</p>
                    <p className="text-sm text-muted-foreground">{selectedABTest.configurations.treatment.description}</p>
                  </div>
                </div>

                <div>
                  <h3 className="text-lg font-medium mb-4">Statistical Results</h3>
                  <Table>
                    <TableBody>
                      <TableRow>
                        <TableCell>Effect Size</TableCell>
                        <TableCell className="text-right">
                          {selectedABTest.results.effect.magnitude.toFixed(2)}%
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Confidence Interval</TableCell>
                        <TableCell className="text-right">
                          [{selectedABTest.results.confidenceInterval[0].toFixed(2)}%, {selectedABTest.results.confidenceInterval[1].toFixed(2)}%]
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Significance Level</TableCell>
                        <TableCell className="text-right">
                          {(selectedABTest.results.significanceLevel * 100).toFixed(0)}%
                        </TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </div>

                <div className="flex justify-end gap-2">
                  <Button onClick={() => setSelectedABTest(null)}>Close</Button>
                </div>
              </div>
            ) : (
              <div className="text-center py-8">
                <p className="text-muted-foreground mb-4">No results available yet. Run the test to see results.</p>
                <div className="flex justify-center gap-2">
                  <Button variant="outline" onClick={() => setSelectedABTest(null)}>Close</Button>
                  <Button>Run A/B Test</Button>
                </div>
              </div>
            )}
          </DialogContent>
        </Dialog>
      )}
    </div>
  );
};

export default AdvancedFeatures;