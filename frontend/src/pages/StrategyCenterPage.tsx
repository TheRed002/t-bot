import React, { useState, useEffect, useMemo } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Badge } from '../components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs';
import { Input } from '../components/ui/input';
import { Label } from '../components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../components/ui/select';
import { Textarea } from '../components/ui/textarea';
import { Switch } from '../components/ui/switch';
import { Slider } from '../components/ui/slider';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '../components/ui/table';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '../components/ui/dialog';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  Area,
  AreaChart
} from 'recharts';
import { 
  TrendingUp, 
  TrendingDown, 
  Activity, 
  Settings, 
  Play, 
  Pause, 
  Download, 
  Upload, 
  Copy, 
  Edit, 
  Trash2,
  Plus,
  Filter,
  Search,
  AlertTriangle,
  CheckCircle,
  Clock,
  BarChart3,
  PieChart as PieChartIcon,
  Target,
  Zap,
  Shield,
  Layers,
  GitBranch,
  DollarSign
} from 'lucide-react';
import { RootState } from '../store';
import { fetchStrategies, deployStrategy, updateStrategy } from '../store/slices/strategySlice';

interface Strategy {
  id: string;
  name: string;
  description: string;
  type: 'momentum' | 'arbitrage' | 'mean_reversion' | 'market_making' | 'trend_following' | 'breakout';
  riskLevel: 'low' | 'medium' | 'high';
  status: 'active' | 'paused' | 'draft';
  performance: {
    totalReturn: number;
    sharpeRatio: number;
    maxDrawdown: number;
    winRate: number;
    totalTrades: number;
    profitFactor: number;
    avgWin: number;
    avgLoss: number;
  };
  backtest: {
    startDate: string;
    endDate: string;
    initialCapital: number;
    finalValue: number;
    returns: Array<{ date: string; value: number; drawdown: number }>;
  };
  parameters: Record<string, any>;
  deployed: boolean;
  createdAt: string;
  lastModified: string;
}

const StrategyCenterPage: React.FC = () => {
  const dispatch = useDispatch();
  const { strategies, isLoading: loading } = useSelector((state: RootState) => state.strategies);
  const [selectedStrategy, setSelectedStrategy] = useState<Strategy | null>(null);
  const [activeTab, setActiveTab] = useState('library');
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState<string>('all');
  const [filterRisk, setFilterRisk] = useState<string>('all');
  const [isEditDialogOpen, setIsEditDialogOpen] = useState(false);
  const [isCompareDialogOpen, setIsCompareDialogOpen] = useState(false);
  const [compareStrategies, setCompareStrategies] = useState<string[]>([]);

  // Mock data for demonstration
  const mockStrategies: Strategy[] = [
    {
      id: 'momentum_1',
      name: 'Adaptive Momentum',
      description: 'Dynamic momentum strategy with volatility adjustment',
      type: 'momentum',
      riskLevel: 'medium',
      status: 'active',
      performance: {
        totalReturn: 23.4,
        sharpeRatio: 1.8,
        maxDrawdown: -8.2,
        winRate: 68.5,
        totalTrades: 142,
        profitFactor: 2.1,
        avgWin: 2.8,
        avgLoss: -1.3
      },
      backtest: {
        startDate: '2024-01-01',
        endDate: '2024-08-22',
        initialCapital: 10000,
        finalValue: 12340,
        returns: Array.from({ length: 50 }, (_, i) => ({
          date: new Date(2024, 0, i * 5).toISOString().split('T')[0],
          value: 10000 + Math.random() * 3000 - 500,
          drawdown: -(Math.random() * 10)
        }))
      },
      parameters: {
        lookbackPeriod: 20,
        momentumThreshold: 0.02,
        volatilityWindow: 10,
        riskPerTrade: 0.02
      },
      deployed: true,
      createdAt: '2024-01-15',
      lastModified: '2024-08-20'
    },
    {
      id: 'arbitrage_1',
      name: 'Cross-Exchange Arbitrage',
      description: 'Multi-exchange price difference exploitation',
      type: 'arbitrage',
      riskLevel: 'low',
      status: 'active',
      performance: {
        totalReturn: 15.2,
        sharpeRatio: 2.3,
        maxDrawdown: -3.1,
        winRate: 89.2,
        totalTrades: 298,
        profitFactor: 3.8,
        avgWin: 0.8,
        avgLoss: -0.2
      },
      backtest: {
        startDate: '2024-01-01',
        endDate: '2024-08-22',
        initialCapital: 10000,
        finalValue: 11520,
        returns: Array.from({ length: 50 }, (_, i) => ({
          date: new Date(2024, 0, i * 5).toISOString().split('T')[0],
          value: 10000 + i * 30 + Math.random() * 200 - 100,
          drawdown: -(Math.random() * 5)
        }))
      },
      parameters: {
        minSpread: 0.001,
        maxExposure: 0.1,
        exchanges: ['binance', 'coinbase', 'okx'],
        executionDelay: 100
      },
      deployed: true,
      createdAt: '2024-02-01',
      lastModified: '2024-08-18'
    },
    {
      id: 'mean_reversion_1',
      name: 'Statistical Mean Reversion',
      description: 'Mean reversion with statistical significance testing',
      type: 'mean_reversion',
      riskLevel: 'medium',
      status: 'paused',
      performance: {
        totalReturn: 11.7,
        sharpeRatio: 1.4,
        maxDrawdown: -12.5,
        winRate: 72.1,
        totalTrades: 87,
        profitFactor: 1.9,
        avgWin: 3.2,
        avgLoss: -1.7
      },
      backtest: {
        startDate: '2024-03-01',
        endDate: '2024-08-22',
        initialCapital: 10000,
        finalValue: 11170,
        returns: Array.from({ length: 40 }, (_, i) => ({
          date: new Date(2024, 2, i * 5).toISOString().split('T')[0],
          value: 10000 + Math.sin(i * 0.3) * 500 + i * 20,
          drawdown: -(Math.random() * 15)
        }))
      },
      parameters: {
        lookbackPeriod: 50,
        entryThreshold: 2.0,
        exitThreshold: 0.5,
        maxHoldingPeriod: 48
      },
      deployed: false,
      createdAt: '2024-03-10',
      lastModified: '2024-08-15'
    }
  ];

  const filteredStrategies = useMemo(() => {
    return mockStrategies.filter(strategy => {
      const matchesSearch = strategy.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           strategy.description.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesType = filterType === 'all' || strategy.type === filterType;
      const matchesRisk = filterRisk === 'all' || strategy.riskLevel === filterRisk;
      return matchesSearch && matchesType && matchesRisk;
    });
  }, [searchTerm, filterType, filterRisk]);

  const getRiskColor = (level: string) => {
    switch (level) {
      case 'low': return 'text-green-500';
      case 'medium': return 'text-yellow-500';
      case 'high': return 'text-red-500';
      default: return 'text-gray-500';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'text-green-500';
      case 'paused': return 'text-yellow-500';
      case 'draft': return 'text-gray-500';
      default: return 'text-gray-500';
    }
  };

  const formatPercentage = (value: number) => {
    const color = value >= 0 ? 'text-green-500' : 'text-red-500';
    return <span className={color}>{value >= 0 ? '+' : ''}{value.toFixed(2)}%</span>;
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value);
  };

  const performanceData = selectedStrategy ? selectedStrategy.backtest.returns.map((item, index) => ({
    ...item,
    cumulativeReturn: ((item.value - selectedStrategy.backtest.initialCapital) / selectedStrategy.backtest.initialCapital) * 100
  })) : [];

  const ComparisonChart = () => {
    if (compareStrategies.length === 0) return null;

    const comparedStrategies = mockStrategies.filter(s => compareStrategies.includes(s.id));
    const colors = ['#ef4444', '#f97316', '#eab308', '#22c55e', '#3b82f6', '#8b5cf6'];

    return (
      <div className="h-96">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="date" stroke="#9ca3af" />
            <YAxis stroke="#9ca3af" />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#1f2937', 
                border: '1px solid #374151',
                borderRadius: '8px'
              }} 
            />
            <Legend />
            {comparedStrategies.map((strategy, index) => (
              <Line
                key={strategy.id}
                dataKey={`returns_${strategy.id}`}
                stroke={colors[index % colors.length]}
                strokeWidth={2}
                dot={false}
                name={strategy.name}
                data={strategy.backtest.returns.map(item => ({
                  date: item.date,
                  [`returns_${strategy.id}`]: ((item.value - strategy.backtest.initialCapital) / strategy.backtest.initialCapital) * 100
                }))}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-950 text-white p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold text-white flex items-center gap-3">
              <Layers className="text-red-500" />
              Strategy Center
            </h1>
            <p className="text-gray-400 mt-2">
              Develop, test, and deploy automated trading strategies
            </p>
          </div>
          <div className="flex gap-3">
            <Button variant="outline" className="border-gray-600">
              <Upload className="w-4 h-4 mr-2" />
              Import
            </Button>
            <Button variant="outline" className="border-gray-600">
              <Download className="w-4 h-4 mr-2" />
              Export
            </Button>
            <Button className="bg-red-600 hover:bg-red-700">
              <Plus className="w-4 h-4 mr-2" />
              New Strategy
            </Button>
          </div>
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-5 bg-gray-900">
            <TabsTrigger value="library" className="data-[state=active]:bg-red-600">Library</TabsTrigger>
            <TabsTrigger value="builder" className="data-[state=active]:bg-red-600">Builder</TabsTrigger>
            <TabsTrigger value="backtest" className="data-[state=active]:bg-red-600">Backtest</TabsTrigger>
            <TabsTrigger value="performance" className="data-[state=active]:bg-red-600">Performance</TabsTrigger>
            <TabsTrigger value="comparison" className="data-[state=active]:bg-red-600">Compare</TabsTrigger>
          </TabsList>

          {/* Strategy Library Tab */}
          <TabsContent value="library" className="space-y-6">
            {/* Filters and Search */}
            <Card className="bg-gray-900 border-gray-700">
              <CardHeader>
                <CardTitle className="text-white">Strategy Library</CardTitle>
                <CardDescription>Browse and manage your trading strategies</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-4 mb-6">
                  <div className="flex-1 min-w-64">
                    <div className="relative">
                      <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
                      <Input
                        placeholder="Search strategies..."
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        className="pl-10 bg-gray-800 border-gray-600"
                      />
                    </div>
                  </div>
                  <Select value={filterType} onValueChange={setFilterType}>
                    <SelectTrigger className="w-48 bg-gray-800 border-gray-600">
                      <SelectValue placeholder="Strategy Type" />
                    </SelectTrigger>
                    <SelectContent className="bg-gray-800 border-gray-600">
                      <SelectItem value="all">All Types</SelectItem>
                      <SelectItem value="momentum">Momentum</SelectItem>
                      <SelectItem value="arbitrage">Arbitrage</SelectItem>
                      <SelectItem value="mean_reversion">Mean Reversion</SelectItem>
                      <SelectItem value="market_making">Market Making</SelectItem>
                      <SelectItem value="trend_following">Trend Following</SelectItem>
                      <SelectItem value="breakout">Breakout</SelectItem>
                    </SelectContent>
                  </Select>
                  <Select value={filterRisk} onValueChange={setFilterRisk}>
                    <SelectTrigger className="w-48 bg-gray-800 border-gray-600">
                      <SelectValue placeholder="Risk Level" />
                    </SelectTrigger>
                    <SelectContent className="bg-gray-800 border-gray-600">
                      <SelectItem value="all">All Risk Levels</SelectItem>
                      <SelectItem value="low">Low Risk</SelectItem>
                      <SelectItem value="medium">Medium Risk</SelectItem>
                      <SelectItem value="high">High Risk</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {/* Strategy Cards Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {filteredStrategies.map((strategy) => (
                    <Card key={strategy.id} className="bg-gray-800 border-gray-700 hover:border-red-500 transition-colors cursor-pointer">
                      <CardHeader className="pb-3">
                        <div className="flex items-start justify-between">
                          <div>
                            <CardTitle className="text-lg text-white">{strategy.name}</CardTitle>
                            <CardDescription className="text-gray-400">{strategy.description}</CardDescription>
                          </div>
                          <Badge className={`${getRiskColor(strategy.riskLevel)} bg-gray-700`}>
                            {strategy.riskLevel}
                          </Badge>
                        </div>
                        <div className="flex items-center gap-2 mt-2">
                          <Badge variant="outline" className="text-xs">
                            {strategy.type.replace('_', ' ')}
                          </Badge>
                          <Badge className={`${getStatusColor(strategy.status)} bg-gray-700 text-xs`}>
                            {strategy.status}
                          </Badge>
                          {strategy.deployed && (
                            <Badge className="bg-green-900 text-green-300 text-xs">
                              <Zap className="w-3 h-3 mr-1" />
                              Live
                            </Badge>
                          )}
                        </div>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        <div className="grid grid-cols-2 gap-4">
                          <div className="space-y-2">
                            <div className="flex justify-between text-sm">
                              <span className="text-gray-400">Total Return:</span>
                              {formatPercentage(strategy.performance.totalReturn)}
                            </div>
                            <div className="flex justify-between text-sm">
                              <span className="text-gray-400">Sharpe Ratio:</span>
                              <span className="text-white">{strategy.performance.sharpeRatio.toFixed(2)}</span>
                            </div>
                            <div className="flex justify-between text-sm">
                              <span className="text-gray-400">Max Drawdown:</span>
                              <span className="text-red-400">{strategy.performance.maxDrawdown.toFixed(2)}%</span>
                            </div>
                          </div>
                          <div className="space-y-2">
                            <div className="flex justify-between text-sm">
                              <span className="text-gray-400">Win Rate:</span>
                              <span className="text-green-400">{strategy.performance.winRate.toFixed(1)}%</span>
                            </div>
                            <div className="flex justify-between text-sm">
                              <span className="text-gray-400">Total Trades:</span>
                              <span className="text-white">{strategy.performance.totalTrades}</span>
                            </div>
                            <div className="flex justify-between text-sm">
                              <span className="text-gray-400">Profit Factor:</span>
                              <span className="text-white">{strategy.performance.profitFactor.toFixed(2)}</span>
                            </div>
                          </div>
                        </div>

                        <div className="flex gap-2 pt-2">
                          <Button
                            size="sm"
                            variant="outline"
                            className="flex-1 border-gray-600"
                            onClick={() => setSelectedStrategy(strategy)}
                          >
                            <BarChart3 className="w-4 h-4 mr-1" />
                            View
                          </Button>
                          <Button
                            size="sm"
                            variant="outline"
                            className="border-gray-600"
                            onClick={() => {
                              setSelectedStrategy(strategy);
                              setIsEditDialogOpen(true);
                            }}
                          >
                            <Edit className="w-4 h-4" />
                          </Button>
                          <Button
                            size="sm"
                            className={strategy.deployed ? "bg-yellow-600 hover:bg-yellow-700" : "bg-green-600 hover:bg-green-700"}
                          >
                            {strategy.deployed ? (
                              <>
                                <Pause className="w-4 h-4" />
                              </>
                            ) : (
                              <>
                                <Play className="w-4 h-4" />
                              </>
                            )}
                          </Button>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Strategy Builder Tab */}
          <TabsContent value="builder" className="space-y-6">
            <Card className="bg-gray-900 border-gray-700">
              <CardHeader>
                <CardTitle className="text-white">Strategy Builder</CardTitle>
                <CardDescription>Create and configure custom trading strategies</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <div>
                      <Label htmlFor="strategyName">Strategy Name</Label>
                      <Input 
                        id="strategyName" 
                        placeholder="My Custom Strategy" 
                        className="bg-gray-800 border-gray-600 mt-1"
                      />
                    </div>
                    <div>
                      <Label htmlFor="strategyDescription">Description</Label>
                      <Textarea 
                        id="strategyDescription"
                        placeholder="Describe your strategy..."
                        className="bg-gray-800 border-gray-600 mt-1"
                        rows={3}
                      />
                    </div>
                    <div>
                      <Label htmlFor="strategyType">Strategy Type</Label>
                      <Select>
                        <SelectTrigger className="bg-gray-800 border-gray-600 mt-1">
                          <SelectValue placeholder="Select strategy type" />
                        </SelectTrigger>
                        <SelectContent className="bg-gray-800 border-gray-600">
                          <SelectItem value="momentum">Momentum</SelectItem>
                          <SelectItem value="mean_reversion">Mean Reversion</SelectItem>
                          <SelectItem value="arbitrage">Arbitrage</SelectItem>
                          <SelectItem value="market_making">Market Making</SelectItem>
                          <SelectItem value="trend_following">Trend Following</SelectItem>
                          <SelectItem value="breakout">Breakout</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div>
                      <Label htmlFor="riskLevel">Risk Level</Label>
                      <Select>
                        <SelectTrigger className="bg-gray-800 border-gray-600 mt-1">
                          <SelectValue placeholder="Select risk level" />
                        </SelectTrigger>
                        <SelectContent className="bg-gray-800 border-gray-600">
                          <SelectItem value="low">Low Risk</SelectItem>
                          <SelectItem value="medium">Medium Risk</SelectItem>
                          <SelectItem value="high">High Risk</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>

                  <div className="space-y-4">
                    <h3 className="text-lg font-semibold text-white">Parameters</h3>
                    <div className="space-y-4">
                      <div>
                        <Label>Risk Per Trade: 2%</Label>
                        <Slider
                          defaultValue={[2]}
                          max={10}
                          min={0.1}
                          step={0.1}
                          className="mt-2"
                        />
                      </div>
                      <div>
                        <Label>Lookback Period: 20</Label>
                        <Slider
                          defaultValue={[20]}
                          max={200}
                          min={5}
                          step={5}
                          className="mt-2"
                        />
                      </div>
                      <div>
                        <Label>Entry Threshold: 0.02</Label>
                        <Slider
                          defaultValue={[0.02]}
                          max={0.1}
                          min={0.001}
                          step={0.001}
                          className="mt-2"
                        />
                      </div>
                      <div>
                        <Label>Exit Threshold: 0.01</Label>
                        <Slider
                          defaultValue={[0.01]}
                          max={0.05}
                          min={0.001}
                          step={0.001}
                          className="mt-2"
                        />
                      </div>
                    </div>
                  </div>
                </div>

                <div className="flex gap-3 pt-4">
                  <Button className="bg-red-600 hover:bg-red-700">
                    <Play className="w-4 h-4 mr-2" />
                    Test Strategy
                  </Button>
                  <Button variant="outline" className="border-gray-600">
                    <Download className="w-4 h-4 mr-2" />
                    Save Draft
                  </Button>
                  <Button variant="outline" className="border-gray-600">
                    <Copy className="w-4 h-4 mr-2" />
                    Clone Existing
                  </Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Backtest Tab */}
          <TabsContent value="backtest" className="space-y-6">
            {selectedStrategy && (
              <Card className="bg-gray-900 border-gray-700">
                <CardHeader>
                  <CardTitle className="text-white">
                    Backtest Results - {selectedStrategy.name}
                  </CardTitle>
                  <CardDescription>
                    Historical performance from {selectedStrategy.backtest.startDate} to {selectedStrategy.backtest.endDate}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 mb-6">
                    <Card className="bg-gray-800 border-gray-700">
                      <CardContent className="p-4">
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="text-sm text-gray-400">Total Return</p>
                            <p className="text-2xl font-bold text-green-400">
                              +{selectedStrategy.performance.totalReturn.toFixed(2)}%
                            </p>
                          </div>
                          <TrendingUp className="w-8 h-8 text-green-400" />
                        </div>
                      </CardContent>
                    </Card>
                    <Card className="bg-gray-800 border-gray-700">
                      <CardContent className="p-4">
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="text-sm text-gray-400">Sharpe Ratio</p>
                            <p className="text-2xl font-bold text-white">
                              {selectedStrategy.performance.sharpeRatio.toFixed(2)}
                            </p>
                          </div>
                          <Target className="w-8 h-8 text-blue-400" />
                        </div>
                      </CardContent>
                    </Card>
                    <Card className="bg-gray-800 border-gray-700">
                      <CardContent className="p-4">
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="text-sm text-gray-400">Max Drawdown</p>
                            <p className="text-2xl font-bold text-red-400">
                              {selectedStrategy.performance.maxDrawdown.toFixed(2)}%
                            </p>
                          </div>
                          <TrendingDown className="w-8 h-8 text-red-400" />
                        </div>
                      </CardContent>
                    </Card>
                    <Card className="bg-gray-800 border-gray-700">
                      <CardContent className="p-4">
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="text-sm text-gray-400">Win Rate</p>
                            <p className="text-2xl font-bold text-green-400">
                              {selectedStrategy.performance.winRate.toFixed(1)}%
                            </p>
                          </div>
                          <CheckCircle className="w-8 h-8 text-green-400" />
                        </div>
                      </CardContent>
                    </Card>
                  </div>

                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <Card className="bg-gray-800 border-gray-700">
                      <CardHeader>
                        <CardTitle className="text-white">Cumulative Returns</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="h-64">
                          <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={performanceData}>
                              <defs>
                                <linearGradient id="colorReturns" x1="0" y1="0" x2="0" y2="1">
                                  <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3}/>
                                  <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
                                </linearGradient>
                              </defs>
                              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                              <XAxis dataKey="date" stroke="#9ca3af" />
                              <YAxis stroke="#9ca3af" />
                              <Tooltip 
                                contentStyle={{ 
                                  backgroundColor: '#1f2937', 
                                  border: '1px solid #374151',
                                  borderRadius: '8px'
                                }} 
                              />
                              <Area 
                                type="monotone" 
                                dataKey="cumulativeReturn" 
                                stroke="#ef4444" 
                                fillOpacity={1} 
                                fill="url(#colorReturns)" 
                              />
                            </AreaChart>
                          </ResponsiveContainer>
                        </div>
                      </CardContent>
                    </Card>

                    <Card className="bg-gray-800 border-gray-700">
                      <CardHeader>
                        <CardTitle className="text-white">Drawdown Chart</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="h-64">
                          <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={performanceData}>
                              <defs>
                                <linearGradient id="colorDrawdown" x1="0" y1="0" x2="0" y2="1">
                                  <stop offset="5%" stopColor="#ef4444" stopOpacity={0.5}/>
                                  <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
                                </linearGradient>
                              </defs>
                              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                              <XAxis dataKey="date" stroke="#9ca3af" />
                              <YAxis stroke="#9ca3af" />
                              <Tooltip 
                                contentStyle={{ 
                                  backgroundColor: '#1f2937', 
                                  border: '1px solid #374151',
                                  borderRadius: '8px'
                                }} 
                              />
                              <Area 
                                type="monotone" 
                                dataKey="drawdown" 
                                stroke="#ef4444" 
                                fillOpacity={1} 
                                fill="url(#colorDrawdown)" 
                              />
                            </AreaChart>
                          </ResponsiveContainer>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          {/* Performance Analytics Tab */}
          <TabsContent value="performance" className="space-y-6">
            <Card className="bg-gray-900 border-gray-700">
              <CardHeader>
                <CardTitle className="text-white">Performance Analytics</CardTitle>
                <CardDescription>Detailed performance metrics and analysis</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <Table>
                    <TableHeader>
                      <TableRow className="border-gray-700">
                        <TableHead className="text-gray-300">Strategy</TableHead>
                        <TableHead className="text-gray-300">Return</TableHead>
                        <TableHead className="text-gray-300">Sharpe</TableHead>
                        <TableHead className="text-gray-300">Drawdown</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {mockStrategies.map((strategy) => (
                        <TableRow key={strategy.id} className="border-gray-700">
                          <TableCell className="text-white">{strategy.name}</TableCell>
                          <TableCell>{formatPercentage(strategy.performance.totalReturn)}</TableCell>
                          <TableCell className="text-white">{strategy.performance.sharpeRatio.toFixed(2)}</TableCell>
                          <TableCell className="text-red-400">{strategy.performance.maxDrawdown.toFixed(2)}%</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>

                  <div className="space-y-4">
                    <h3 className="text-lg font-semibold text-white">Risk-Return Profile</h3>
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={mockStrategies.map(s => ({
                          name: s.name.split(' ').slice(0, 2).join(' '),
                          return: s.performance.totalReturn,
                          risk: Math.abs(s.performance.maxDrawdown)
                        }))}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                          <XAxis dataKey="name" stroke="#9ca3af" />
                          <YAxis stroke="#9ca3af" />
                          <Tooltip 
                            contentStyle={{ 
                              backgroundColor: '#1f2937', 
                              border: '1px solid #374151',
                              borderRadius: '8px'
                            }} 
                          />
                          <Bar dataKey="return" fill="#22c55e" name="Return %" />
                          <Bar dataKey="risk" fill="#ef4444" name="Risk %" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Strategy Comparison Tab */}
          <TabsContent value="comparison" className="space-y-6">
            <Card className="bg-gray-900 border-gray-700">
              <CardHeader>
                <CardTitle className="text-white">Strategy Comparison</CardTitle>
                <CardDescription>Compare multiple strategies side by side</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-2 mb-6">
                  {mockStrategies.map((strategy) => (
                    <Badge
                      key={strategy.id}
                      variant={compareStrategies.includes(strategy.id) ? "default" : "outline"}
                      className={`cursor-pointer ${
                        compareStrategies.includes(strategy.id) 
                          ? "bg-red-600 text-white" 
                          : "border-gray-600 text-gray-300"
                      }`}
                      onClick={() => {
                        if (compareStrategies.includes(strategy.id)) {
                          setCompareStrategies(prev => prev.filter(id => id !== strategy.id));
                        } else {
                          setCompareStrategies(prev => [...prev, strategy.id]);
                        }
                      }}
                    >
                      {strategy.name}
                    </Badge>
                  ))}
                </div>

                {compareStrategies.length > 0 ? (
                  <>
                    <ComparisonChart />
                    <div className="mt-6">
                      <Table>
                        <TableHeader>
                          <TableRow className="border-gray-700">
                            <TableHead className="text-gray-300">Metric</TableHead>
                            {compareStrategies.map((id) => {
                              const strategy = mockStrategies.find(s => s.id === id);
                              return (
                                <TableHead key={id} className="text-gray-300">
                                  {strategy?.name}
                                </TableHead>
                              );
                            })}
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          <TableRow className="border-gray-700">
                            <TableCell className="text-white font-medium">Total Return</TableCell>
                            {compareStrategies.map((id) => {
                              const strategy = mockStrategies.find(s => s.id === id);
                              return (
                                <TableCell key={id}>
                                  {formatPercentage(strategy?.performance.totalReturn || 0)}
                                </TableCell>
                              );
                            })}
                          </TableRow>
                          <TableRow className="border-gray-700">
                            <TableCell className="text-white font-medium">Sharpe Ratio</TableCell>
                            {compareStrategies.map((id) => {
                              const strategy = mockStrategies.find(s => s.id === id);
                              return (
                                <TableCell key={id} className="text-white">
                                  {strategy?.performance.sharpeRatio.toFixed(2)}
                                </TableCell>
                              );
                            })}
                          </TableRow>
                          <TableRow className="border-gray-700">
                            <TableCell className="text-white font-medium">Max Drawdown</TableCell>
                            {compareStrategies.map((id) => {
                              const strategy = mockStrategies.find(s => s.id === id);
                              return (
                                <TableCell key={id} className="text-red-400">
                                  {strategy?.performance.maxDrawdown.toFixed(2)}%
                                </TableCell>
                              );
                            })}
                          </TableRow>
                          <TableRow className="border-gray-700">
                            <TableCell className="text-white font-medium">Win Rate</TableCell>
                            {compareStrategies.map((id) => {
                              const strategy = mockStrategies.find(s => s.id === id);
                              return (
                                <TableCell key={id} className="text-green-400">
                                  {strategy?.performance.winRate.toFixed(1)}%
                                </TableCell>
                              );
                            })}
                          </TableRow>
                        </TableBody>
                      </Table>
                    </div>
                  </>
                ) : (
                  <div className="text-center py-12">
                    <GitBranch className="w-16 h-16 text-gray-600 mx-auto mb-4" />
                    <p className="text-gray-400">
                      Select strategies above to compare their performance
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default StrategyCenterPage;