import React, { useState, useEffect, useMemo } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Badge } from '../components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs';
import { Input } from '../components/ui/input';
import { Label } from '../components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../components/ui/select';
import { Switch } from '../components/ui/switch';
import { Slider } from '../components/ui/slider';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '../components/ui/table';
import { Progress } from '../components/ui/progress';
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
  AreaChart,
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ScatterChart,
  Scatter,
  Treemap,
  ReferenceLine
} from 'recharts';
import { 
  AlertTriangle, 
  Shield, 
  TrendingUp, 
  TrendingDown, 
  Activity, 
  Settings, 
  AlertCircle,
  CheckCircle,
  Clock,
  Target,
  Zap,
  DollarSign,
  Percent,
  BarChart3,
  PieChart as PieChartIcon,
  Activity as ActivityIcon,
  RefreshCw,
  Download,
  Upload,
  Bell,
  BellOff,
  Eye,
  EyeOff,
  Filter,
  Search,
  CircuitBoard,
  Gauge,
  Calculator,
  ArrowRight,
  AlertOctagon,
  ShieldAlert,
  BarChart4
} from 'lucide-react';
import { RootState } from '../store';
import { Position as PortfolioPosition } from '@/types';

interface RiskPosition {
  symbol: string;
  size: number;
  value: number;
  risk: number;
  exposure: number;
  maxLoss: number;
  correlation: number;
}

interface RiskMetric {
  name: string;
  value: number;
  threshold: number;
  status: 'safe' | 'warning' | 'danger';
  change24h: number;
}

interface Position {
  symbol: string;
  size: number;
  value: number;
  risk: number;
  exposure: number;
  maxLoss: number;
  correlation: number;
}

interface CircuitBreaker {
  id: string;
  name: string;
  type: 'position' | 'portfolio' | 'drawdown' | 'loss';
  threshold: number;
  current: number;
  status: 'active' | 'triggered' | 'disabled';
  triggeredAt?: string;
}

interface VaRData {
  timeframe: '1d' | '1w' | '1m';
  confidence: 95 | 99;
  value: number;
  stressed: number;
  historical: Array<{ date: string; var: number; return: number }>;
}

const RiskDashboardPage: React.FC = () => {
  const dispatch = useDispatch();
  const { summary, positions, balances } = useSelector((state: RootState) => state.portfolio);
  const [activeTab, setActiveTab] = useState('overview');
  const [timeframe, setTimeframe] = useState('1d');
  const [confidence, setConfidence] = useState(95);
  const [showAlerts, setShowAlerts] = useState(true);

  // Mock data for demonstration
  const riskMetrics: RiskMetric[] = [
    { name: 'Portfolio VaR (1D, 95%)', value: -12500, threshold: -15000, status: 'warning', change24h: 8.2 },
    { name: 'Max Drawdown', value: -5.8, threshold: -10, status: 'safe', change24h: -2.1 },
    { name: 'Sharpe Ratio', value: 1.42, threshold: 1.0, status: 'safe', change24h: 0.15 },
    { name: 'Volatility (30D)', value: 18.5, threshold: 25, status: 'safe', change24h: -1.2 },
    { name: 'Leverage Ratio', value: 2.1, threshold: 3.0, status: 'safe', change24h: 0.05 },
    { name: 'Concentration Risk', value: 35.2, threshold: 40, status: 'warning', change24h: 3.4 }
  ];

  const mockPositions: RiskPosition[] = [
    { symbol: 'BTC-USD', size: 2.5, value: 125000, risk: 8.5, exposure: 0.35, maxLoss: -15000, correlation: 1.0 },
    { symbol: 'ETH-USD', size: 45.2, value: 85000, risk: 12.2, exposure: 0.24, maxLoss: -12000, correlation: 0.78 },
    { symbol: 'SOL-USD', size: 850, value: 45000, risk: 15.8, exposure: 0.13, maxLoss: -8500, correlation: 0.65 },
    { symbol: 'ADA-USD', size: 25000, value: 35000, risk: 18.5, exposure: 0.10, maxLoss: -7000, correlation: 0.55 },
    { symbol: 'MATIC-USD', size: 12000, value: 28000, risk: 20.1, exposure: 0.08, maxLoss: -6500, correlation: 0.42 },
    { symbol: 'AVAX-USD', size: 750, value: 32000, risk: 22.3, exposure: 0.09, maxLoss: -7200, correlation: 0.58 }
  ];

  const circuitBreakers: CircuitBreaker[] = [
    { id: 'daily_loss', name: 'Daily Loss Limit', type: 'loss', threshold: -20000, current: -8500, status: 'active' },
    { id: 'position_size', name: 'Max Position Size', type: 'position', threshold: 0.4, current: 0.35, status: 'active' },
    { id: 'drawdown', name: 'Max Drawdown', type: 'drawdown', threshold: -0.15, current: -0.058, status: 'active' },
    { id: 'leverage', name: 'Leverage Limit', type: 'portfolio', threshold: 3.0, current: 2.1, status: 'active' },
    { id: 'correlation', name: 'Correlation Breach', type: 'portfolio', threshold: 0.8, current: 0.78, status: 'triggered', triggeredAt: '2024-08-22T14:30:00Z' }
  ];

  const varData: VaRData = {
    timeframe: '1d',
    confidence: 95,
    value: -12500,
    stressed: -18750,
    historical: Array.from({ length: 30 }, (_, i) => ({
      date: new Date(Date.now() - (29 - i) * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
      var: -10000 - Math.random() * 8000,
      return: (Math.random() - 0.5) * 20000
    }))
  };

  const correlationData = mockPositions.map(p1 => ({
    name: p1.symbol,
    ...mockPositions.reduce((acc, p2) => {
      acc[p2.symbol] = p1.symbol === p2.symbol ? 1 : Math.random() * 0.8 + 0.1;
      return acc;
    }, {} as Record<string, number>)
  }));

  const stressTestResults = [
    { scenario: '2008 Financial Crisis', impact: -25.8, probability: 0.05 },
    { scenario: 'COVID-19 Crash', impact: -18.2, probability: 0.12 },
    { scenario: 'Flash Crash (2010)', impact: -12.5, probability: 0.25 },
    { scenario: 'Crypto Winter', impact: -45.2, probability: 0.08 },
    { scenario: 'Interest Rate Shock', impact: -15.3, probability: 0.18 }
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'safe': return 'text-green-500';
      case 'warning': return 'text-yellow-500';
      case 'danger': return 'text-red-500';
      case 'active': return 'text-green-500';
      case 'triggered': return 'text-red-500';
      case 'disabled': return 'text-gray-500';
      default: return 'text-gray-500';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'safe': return <CheckCircle className="w-4 h-4" />;
      case 'warning': return <AlertTriangle className="w-4 h-4" />;
      case 'danger': return <AlertCircle className="w-4 h-4" />;
      case 'active': return <Shield className="w-4 h-4" />;
      case 'triggered': return <AlertOctagon className="w-4 h-4" />;
      case 'disabled': return <EyeOff className="w-4 h-4" />;
      default: return <Clock className="w-4 h-4" />;
    }
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(value);
  };

  const formatPercentage = (value: number, showSign = true) => {
    const color = value >= 0 ? 'text-green-500' : 'text-red-500';
    const sign = showSign && value >= 0 ? '+' : '';
    return <span className={color}>{sign}{value.toFixed(2)}%</span>;
  };

  const CorrelationHeatmap = () => (
    <div className="grid grid-cols-6 gap-1">
      {mockPositions.map((p1, i) => 
        mockPositions.map((p2, j) => {
          const correlation = i === j ? 1 : Math.random() * 0.8 + 0.1;
          const intensity = Math.abs(correlation);
          const color = correlation > 0.7 ? 'bg-red-500' : correlation > 0.4 ? 'bg-yellow-500' : 'bg-green-500';
          return (
            <div
              key={`${i}-${j}`}
              className={`aspect-square ${color} rounded-sm flex items-center justify-center text-xs text-white font-medium`}
              style={{ opacity: intensity }}
              title={`${p1.symbol} vs ${p2.symbol}: ${correlation.toFixed(2)}`}
            >
              {i === j ? '1' : correlation.toFixed(1).slice(-1)}
            </div>
          );
        })
      )}
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-950 text-white p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold text-white flex items-center gap-3">
              <Shield className="text-red-500" />
              Risk Dashboard
            </h1>
            <p className="text-gray-400 mt-2">
              Monitor portfolio risk metrics and exposure limits
            </p>
          </div>
          <div className="flex gap-3">
            <Button variant="outline" className="border-gray-600">
              <Download className="w-4 h-4 mr-2" />
              Export Report
            </Button>
            <Button variant="outline" className="border-gray-600">
              <RefreshCw className="w-4 h-4 mr-2" />
              Refresh Data
            </Button>
            <Button 
              className={`${showAlerts ? 'bg-red-600 hover:bg-red-700' : 'bg-gray-600 hover:bg-gray-700'}`}
              onClick={() => setShowAlerts(!showAlerts)}
            >
              {showAlerts ? <Bell className="w-4 h-4 mr-2" /> : <BellOff className="w-4 h-4 mr-2" />}
              Alerts {showAlerts ? 'On' : 'Off'}
            </Button>
          </div>
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-6 bg-gray-900">
            <TabsTrigger value="overview" className="data-[state=active]:bg-red-600">Overview</TabsTrigger>
            <TabsTrigger value="var" className="data-[state=active]:bg-red-600">VaR Analysis</TabsTrigger>
            <TabsTrigger value="mockPositions" className="data-[state=active]:bg-red-600">Positions</TabsTrigger>
            <TabsTrigger value="correlation" className="data-[state=active]:bg-red-600">Correlation</TabsTrigger>
            <TabsTrigger value="circuit-breakers" className="data-[state=active]:bg-red-600">Circuit Breakers</TabsTrigger>
            <TabsTrigger value="stress-testing" className="data-[state=active]:bg-red-600">Stress Testing</TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-6">
            {/* Risk Metrics Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {riskMetrics.map((metric, index) => (
                <Card key={index} className="bg-gray-900 border-gray-700">
                  <CardContent className="p-6">
                    <div className="flex items-center justify-between mb-2">
                      <div className={`flex items-center gap-2 ${getStatusColor(metric.status)}`}>
                        {getStatusIcon(metric.status)}
                        <span className="text-sm font-medium">{metric.status.toUpperCase()}</span>
                      </div>
                      {formatPercentage(metric.change24h)}
                    </div>
                    <h3 className="text-sm text-gray-400 mb-1">{metric.name}</h3>
                    <p className="text-2xl font-bold text-white mb-2">
                      {metric.name.includes('VaR') || metric.name.includes('Loss') 
                        ? formatCurrency(metric.value)
                        : `${metric.value}${metric.name.includes('%') || metric.name.includes('Ratio') || metric.name.includes('Volatility') ? '%' : ''}`
                      }
                    </p>
                    <div className="w-full bg-gray-700 rounded-full h-2">
                      <div 
                        className={`h-2 rounded-full ${
                          metric.status === 'safe' ? 'bg-green-500' : 
                          metric.status === 'warning' ? 'bg-yellow-500' : 'bg-red-500'
                        }`}
                        style={{ 
                          width: `${Math.min(100, (Math.abs(metric.value) / Math.abs(metric.threshold)) * 100)}%` 
                        }}
                      />
                    </div>
                    <p className="text-xs text-gray-500 mt-1">
                      Threshold: {metric.name.includes('VaR') ? formatCurrency(metric.threshold) : `${metric.threshold}%`}
                    </p>
                  </CardContent>
                </Card>
              ))}
            </div>

            {/* Portfolio Risk Summary */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card className="bg-gray-900 border-gray-700">
                <CardHeader>
                  <CardTitle className="text-white">Portfolio Composition by Risk</CardTitle>
                  <CardDescription>Risk-weighted position allocation</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={mockPositions.map(p => ({ name: p.symbol, value: p.risk, exposure: p.exposure }))}
                          cx="50%"
                          cy="50%"
                          innerRadius={60}
                          outerRadius={100}
                          paddingAngle={5}
                          dataKey="value"
                        >
                          {mockPositions.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={`hsl(${index * 60}, 70%, 50%)`} />
                          ))}
                        </Pie>
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: '#1f2937', 
                            border: '1px solid #374151',
                            borderRadius: '8px'
                          }}
                          formatter={(value: number, name: string, props: any) => [
                            `${value.toFixed(1)}% risk`,
                            `${props.payload.name} (${(props.payload.exposure * 100).toFixed(1)}% exposure)`
                          ]}
                        />
                        <Legend />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-gray-900 border-gray-700">
                <CardHeader>
                  <CardTitle className="text-white">Risk Trend (30 Days)</CardTitle>
                  <CardDescription>Portfolio risk evolution over time</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={Array.from({ length: 30 }, (_, i) => ({
                        date: new Date(Date.now() - (29 - i) * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
                        portfolio_var: -8000 - Math.random() * 6000,
                        drawdown: -(Math.random() * 8),
                        volatility: 15 + Math.random() * 10
                      }))}>
                        <defs>
                          <linearGradient id="colorVaR" x1="0" y1="0" x2="0" y2="1">
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
                          dataKey="portfolio_var" 
                          stroke="#ef4444" 
                          fillOpacity={1} 
                          fill="url(#colorVaR)" 
                          name="Portfolio VaR"
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Active Alerts */}
            {showAlerts && (
              <Card className="bg-gray-900 border-gray-700 border-l-4 border-l-red-500">
                <CardHeader>
                  <CardTitle className="text-white flex items-center gap-2">
                    <AlertTriangle className="text-red-500" />
                    Active Risk Alerts
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between p-3 bg-red-900/20 rounded-lg border border-red-500/30">
                      <div className="flex items-center gap-3">
                        <AlertCircle className="text-red-500 w-5 h-5" />
                        <div>
                          <p className="text-white font-medium">High Correlation Alert</p>
                          <p className="text-gray-400 text-sm">BTC-ETH correlation exceeded 0.8 threshold</p>
                        </div>
                      </div>
                      <Badge className="bg-red-600 text-white">Critical</Badge>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-yellow-900/20 rounded-lg border border-yellow-500/30">
                      <div className="flex items-center gap-3">
                        <AlertTriangle className="text-yellow-500 w-5 h-5" />
                        <div>
                          <p className="text-white font-medium">VaR Approaching Limit</p>
                          <p className="text-gray-400 text-sm">Portfolio VaR at 83% of daily limit</p>
                        </div>
                      </div>
                      <Badge className="bg-yellow-600 text-white">Warning</Badge>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          {/* VaR Analysis Tab */}
          <TabsContent value="var" className="space-y-6">
            <Card className="bg-gray-900 border-gray-700">
              <CardHeader>
                <CardTitle className="text-white">Value at Risk (VaR) Analysis</CardTitle>
                <CardDescription>Risk exposure at different confidence levels and time horizons</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex gap-4 mb-6">
                  <div>
                    <Label>Time Horizon</Label>
                    <Select value={timeframe} onValueChange={setTimeframe}>
                      <SelectTrigger className="w-32 bg-gray-800 border-gray-600">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent className="bg-gray-800 border-gray-600">
                        <SelectItem value="1d">1 Day</SelectItem>
                        <SelectItem value="1w">1 Week</SelectItem>
                        <SelectItem value="1m">1 Month</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label>Confidence Level</Label>
                    <Select value={confidence.toString()} onValueChange={(value) => setConfidence(Number(value))}>
                      <SelectTrigger className="w-32 bg-gray-800 border-gray-600">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent className="bg-gray-800 border-gray-600">
                        <SelectItem value="95">95%</SelectItem>
                        <SelectItem value="99">99%</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
                  <Card className="bg-gray-800 border-gray-700">
                    <CardContent className="p-6">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm text-gray-400">Normal VaR</p>
                          <p className="text-2xl font-bold text-red-400">
                            {formatCurrency(varData.value)}
                          </p>
                        </div>
                        <Calculator className="w-8 h-8 text-blue-400" />
                      </div>
                    </CardContent>
                  </Card>
                  <Card className="bg-gray-800 border-gray-700">
                    <CardContent className="p-6">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm text-gray-400">Stressed VaR</p>
                          <p className="text-2xl font-bold text-red-500">
                            {formatCurrency(varData.stressed)}
                          </p>
                        </div>
                        <ShieldAlert className="w-8 h-8 text-red-400" />
                      </div>
                    </CardContent>
                  </Card>
                  <Card className="bg-gray-800 border-gray-700">
                    <CardContent className="p-6">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm text-gray-400">Stress Multiple</p>
                          <p className="text-2xl font-bold text-yellow-400">
                            {(varData.stressed / varData.value).toFixed(1)}x
                          </p>
                        </div>
                        <Gauge className="w-8 h-8 text-yellow-400" />
                      </div>
                    </CardContent>
                  </Card>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <Card className="bg-gray-800 border-gray-700">
                    <CardHeader>
                      <CardTitle className="text-white">VaR vs Returns (30 Days)</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="h-64">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={varData.historical}>
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
                            <Line type="monotone" dataKey="var" stroke="#ef4444" strokeWidth={2} name="VaR" />
                            <Line type="monotone" dataKey="return" stroke="#22c55e" strokeWidth={2} name="Daily Return" />
                            <ReferenceLine y={varData.value} stroke="#ef4444" strokeDasharray="5 5" />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="bg-gray-800 border-gray-700">
                    <CardHeader>
                      <CardTitle className="text-white">VaR Breakdown by Asset</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="h-64">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={mockPositions.map(p => ({
                            name: p.symbol.replace('-USD', ''),
                            var: -(p.value * p.risk / 100),
                            contribution: (p.value * p.risk / 100) / 12500 * 100
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
                              formatter={(value: number, name: string) => [
                                name === 'var' ? formatCurrency(value) : `${value.toFixed(1)}%`,
                                name === 'var' ? 'VaR' : 'Contribution'
                              ]}
                            />
                            <Bar dataKey="var" fill="#ef4444" name="VaR" />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Positions Tab */}
          <TabsContent value="mockPositions" className="space-y-6">
            <Card className="bg-gray-900 border-gray-700">
              <CardHeader>
                <CardTitle className="text-white">Position Risk Analysis</CardTitle>
                <CardDescription>Individual position metrics and risk contribution</CardDescription>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow className="border-gray-700">
                      <TableHead className="text-gray-300">Symbol</TableHead>
                      <TableHead className="text-gray-300">Size</TableHead>
                      <TableHead className="text-gray-300">Value</TableHead>
                      <TableHead className="text-gray-300">Risk %</TableHead>
                      <TableHead className="text-gray-300">Exposure</TableHead>
                      <TableHead className="text-gray-300">Max Loss</TableHead>
                      <TableHead className="text-gray-300">Status</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {mockPositions.map((position) => (
                      <TableRow key={position.symbol} className="border-gray-700">
                        <TableCell className="text-white font-medium">{position.symbol}</TableCell>
                        <TableCell className="text-white">{position.size.toLocaleString()}</TableCell>
                        <TableCell className="text-white">{formatCurrency(position.value)}</TableCell>
                        <TableCell className="text-white">{position.risk.toFixed(1)}%</TableCell>
                        <TableCell>
                          <div className="flex items-center gap-2">
                            <div className="w-16 bg-gray-700 rounded-full h-2">
                              <div 
                                className="h-2 rounded-full bg-red-500" 
                                style={{ width: `${position.exposure * 100}%` }}
                              />
                            </div>
                            <span className="text-white text-sm">
                              {(position.exposure * 100).toFixed(1)}%
                            </span>
                          </div>
                        </TableCell>
                        <TableCell className="text-red-400">{formatCurrency(position.maxLoss)}</TableCell>
                        <TableCell>
                          <Badge className={`${
                            position.risk > 20 ? 'bg-red-600' : 
                            position.risk > 15 ? 'bg-yellow-600' : 'bg-green-600'
                          }`}>
                            {position.risk > 20 ? 'High' : position.risk > 15 ? 'Medium' : 'Low'}
                          </Badge>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card className="bg-gray-900 border-gray-700">
                <CardHeader>
                  <CardTitle className="text-white">Risk vs Return</CardTitle>
                  <CardDescription>Position risk-return profile</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <ScatterChart data={mockPositions.map(p => ({
                        name: p.symbol.replace('-USD', ''),
                        risk: p.risk,
                        return: 5 + Math.random() * 20 - 10,
                        size: p.value / 1000
                      }))}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                        <XAxis dataKey="risk" stroke="#9ca3af" name="Risk %" />
                        <YAxis dataKey="return" stroke="#9ca3af" name="Return %" />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: '#1f2937', 
                            border: '1px solid #374151',
                            borderRadius: '8px'
                          }}
                          formatter={(value: number, name: string) => [
                            `${value.toFixed(1)}%`,
                            name === 'risk' ? 'Risk' : 'Return'
                          ]}
                        />
                        <Scatter dataKey="return" fill="#ef4444" />
                      </ScatterChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-gray-900 border-gray-700">
                <CardHeader>
                  <CardTitle className="text-white">Concentration Analysis</CardTitle>
                  <CardDescription>Portfolio concentration by asset</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <Treemap
                        data={mockPositions.map(p => ({
                          name: p.symbol.replace('-USD', ''),
                          size: p.value,
                          risk: p.risk
                        }))}
                        dataKey="size"
                        aspectRatio={2}
                        stroke="#374151"
                        fill="#ef4444"
                      />
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Correlation Tab */}
          <TabsContent value="correlation" className="space-y-6">
            <Card className="bg-gray-900 border-gray-700">
              <CardHeader>
                <CardTitle className="text-white">Asset Correlation Matrix</CardTitle>
                <CardDescription>Cross-asset correlation analysis</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div>
                    <h3 className="text-lg font-semibold text-white mb-4">Correlation Heatmap</h3>
                    <div className="space-y-2">
                      <div className="grid grid-cols-7 gap-1 text-xs">
                        <div></div>
                        {mockPositions.map(p => (
                          <div key={p.symbol} className="text-center text-gray-400">
                            {p.symbol.replace('-USD', '')}
                          </div>
                        ))}
                      </div>
                      {mockPositions.map((p1, i) => (
                        <div key={p1.symbol} className="grid grid-cols-7 gap-1">
                          <div className="text-xs text-gray-400 flex items-center">
                            {p1.symbol.replace('-USD', '')}
                          </div>
                          {mockPositions.map((p2, j) => {
                            const correlation = i === j ? 1 : 0.3 + Math.random() * 0.6;
                            const intensity = Math.abs(correlation);
                            const bgColor = correlation > 0.7 ? 'bg-red-500' : 
                                           correlation > 0.4 ? 'bg-yellow-500' : 'bg-green-500';
                            return (
                              <div
                                key={`${i}-${j}`}
                                className={`aspect-square ${bgColor} rounded-sm flex items-center justify-center text-xs text-white font-medium`}
                                style={{ opacity: intensity }}
                                title={`Correlation: ${correlation.toFixed(3)}`}
                              >
                                {correlation.toFixed(2)}
                              </div>
                            );
                          })}
                        </div>
                      ))}
                    </div>
                    <div className="flex items-center justify-between mt-4 text-xs">
                      <div className="flex items-center gap-2">
                        <div className="w-4 h-4 bg-green-500 rounded-sm"></div>
                        <span className="text-gray-400">Low (&lt;0.4)</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="w-4 h-4 bg-yellow-500 rounded-sm"></div>
                        <span className="text-gray-400">Medium (0.4-0.7)</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="w-4 h-4 bg-red-500 rounded-sm"></div>
                        <span className="text-gray-400">High (&gt;0.7)</span>
                      </div>
                    </div>
                  </div>

                  <div>
                    <h3 className="text-lg font-semibold text-white mb-4">Risk Contribution</h3>
                    <div className="space-y-3">
                      {mockPositions.map((position, index) => (
                        <div key={position.symbol} className="space-y-2">
                          <div className="flex items-center justify-between">
                            <span className="text-white text-sm">{position.symbol}</span>
                            <span className="text-gray-400 text-sm">
                              {(position.risk * position.correlation).toFixed(1)}%
                            </span>
                          </div>
                          <Progress 
                            value={(position.risk * position.correlation) / 20 * 100} 
                            className="h-2"
                          />
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Circuit Breakers Tab */}
          <TabsContent value="circuit-breakers" className="space-y-6">
            <Card className="bg-gray-900 border-gray-700">
              <CardHeader>
                <CardTitle className="text-white">Circuit Breaker Status</CardTitle>
                <CardDescription>Automated risk controls and emergency measures</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 gap-4">
                  {circuitBreakers.map((breaker) => (
                    <Card key={breaker.id} className={`bg-gray-800 border-gray-700 ${
                      breaker.status === 'triggered' ? 'border-l-4 border-l-red-500' : ''
                    }`}>
                      <CardContent className="p-4">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-3">
                            <div className={`${getStatusColor(breaker.status)}`}>
                              {getStatusIcon(breaker.status)}
                            </div>
                            <div>
                              <h3 className="text-white font-medium">{breaker.name}</h3>
                              <p className="text-gray-400 text-sm capitalize">{breaker.type} control</p>
                            </div>
                          </div>
                          <div className="text-right">
                            <Badge className={`${
                              breaker.status === 'active' ? 'bg-green-600' :
                              breaker.status === 'triggered' ? 'bg-red-600' : 'bg-gray-600'
                            }`}>
                              {breaker.status}
                            </Badge>
                            {breaker.triggeredAt && (
                              <p className="text-xs text-gray-500 mt-1">
                                Triggered: {new Date(breaker.triggeredAt).toLocaleTimeString()}
                              </p>
                            )}
                          </div>
                        </div>
                        <div className="mt-4">
                          <div className="flex items-center justify-between text-sm mb-2">
                            <span className="text-gray-400">Current</span>
                            <span className="text-white">
                              {breaker.type === 'loss' || breaker.type === 'portfolio' 
                                ? (breaker.current < 0 ? formatCurrency(breaker.current) : breaker.current.toFixed(2))
                                : `${(breaker.current * 100).toFixed(1)}%`
                              }
                            </span>
                          </div>
                          <div className="flex items-center justify-between text-sm mb-2">
                            <span className="text-gray-400">Threshold</span>
                            <span className="text-red-400">
                              {breaker.type === 'loss' || breaker.type === 'portfolio'
                                ? (breaker.threshold < 0 ? formatCurrency(breaker.threshold) : breaker.threshold.toFixed(2))
                                : `${(breaker.threshold * 100).toFixed(1)}%`
                              }
                            </span>
                          </div>
                          <Progress 
                            value={Math.abs(breaker.current) / Math.abs(breaker.threshold) * 100}
                            className="h-2"
                          />
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Stress Testing Tab */}
          <TabsContent value="stress-testing" className="space-y-6">
            <Card className="bg-gray-900 border-gray-700">
              <CardHeader>
                <CardTitle className="text-white">Stress Test Results</CardTitle>
                <CardDescription>Portfolio performance under adverse market conditions</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  <Table>
                    <TableHeader>
                      <TableRow className="border-gray-700">
                        <TableHead className="text-gray-300">Scenario</TableHead>
                        <TableHead className="text-gray-300">Portfolio Impact</TableHead>
                        <TableHead className="text-gray-300">Probability</TableHead>
                        <TableHead className="text-gray-300">Risk Level</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {stressTestResults.map((result, index) => (
                        <TableRow key={index} className="border-gray-700">
                          <TableCell className="text-white">{result.scenario}</TableCell>
                          <TableCell className="text-red-400 font-medium">
                            {result.impact.toFixed(1)}%
                          </TableCell>
                          <TableCell className="text-gray-300">
                            {(result.probability * 100).toFixed(0)}%
                          </TableCell>
                          <TableCell>
                            <Badge className={`${
                              Math.abs(result.impact) > 30 ? 'bg-red-600' :
                              Math.abs(result.impact) > 15 ? 'bg-yellow-600' : 'bg-green-600'
                            }`}>
                              {Math.abs(result.impact) > 30 ? 'Extreme' :
                               Math.abs(result.impact) > 15 ? 'High' : 'Moderate'}
                            </Badge>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>

                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <Card className="bg-gray-800 border-gray-700">
                      <CardHeader>
                        <CardTitle className="text-white">Stress Test Impact</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="h-64">
                          <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={stressTestResults}>
                              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                              <XAxis 
                                dataKey="scenario" 
                                stroke="#9ca3af" 
                                angle={-45} 
                                textAnchor="end" 
                                height={80}
                                fontSize={10}
                              />
                              <YAxis stroke="#9ca3af" />
                              <Tooltip 
                                contentStyle={{ 
                                  backgroundColor: '#1f2937', 
                                  border: '1px solid #374151',
                                  borderRadius: '8px'
                                }}
                                formatter={(value: number) => [`${value.toFixed(1)}%`, 'Impact']}
                              />
                              <Bar dataKey="impact" fill="#ef4444" />
                            </BarChart>
                          </ResponsiveContainer>
                        </div>
                      </CardContent>
                    </Card>

                    <Card className="bg-gray-800 border-gray-700">
                      <CardHeader>
                        <CardTitle className="text-white">Risk-Probability Matrix</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="h-64">
                          <ResponsiveContainer width="100%" height="100%">
                            <ScatterChart data={stressTestResults.map(r => ({
                              name: r.scenario.split(' ')[0],
                              impact: Math.abs(r.impact),
                              probability: r.probability * 100,
                              size: Math.abs(r.impact) * r.probability * 100
                            }))}>
                              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                              <XAxis 
                                dataKey="probability" 
                                stroke="#9ca3af" 
                                name="Probability %" 
                                domain={[0, 30]}
                              />
                              <YAxis 
                                dataKey="impact" 
                                stroke="#9ca3af" 
                                name="Impact %" 
                                domain={[0, 50]}
                              />
                              <Tooltip 
                                contentStyle={{ 
                                  backgroundColor: '#1f2937', 
                                  border: '1px solid #374151',
                                  borderRadius: '8px'
                                }}
                              />
                              <Scatter dataKey="impact" fill="#ef4444" />
                            </ScatterChart>
                          </ResponsiveContainer>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default RiskDashboardPage;