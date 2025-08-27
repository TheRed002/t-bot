/**
 * Portfolio Page - Comprehensive portfolio management interface
 * Displays positions, balances, P&L, and performance metrics using Shadcn/ui components
 */

import React, { useState, useEffect, useMemo } from 'react';
import { motion } from 'framer-motion';
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as ChartTooltip,
  Legend,
  Area,
  AreaChart,
  BarChart,
  Bar,
} from 'recharts';
import {
  TrendingUp,
  TrendingDown,
  Wallet,
  BarChart3,
  PieChart as PieChartIcon,
  Activity,
  DollarSign,
  Target,
  AlertTriangle,
  Download,
  RefreshCw,
  Calendar,
  ArrowUpRight,
  ArrowDownRight,
  Eye,
  EyeOff,
  Filter,
  Search,
  MoreHorizontal,
  Plus,
  Minus,
  Info,
} from 'lucide-react';

import { useAppDispatch, useAppSelector } from '@/store';
import {
  fetchPortfolioSummary,
  fetchPositions,
  fetchBalances,
  selectPortfolioState,
} from '@/store/slices/portfolioSlice';
import { websocketService } from '@/services/websocket';
import { cn, formatPrice, formatPnL, formatPercent, getPnLColor, getPnLBg } from '@/lib/utils';

// Shadcn/ui components
import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Skeleton } from '@/components/ui/skeleton';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { Switch } from '@/components/ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';

// Types
import { Position, Balance, PortfolioSummary } from '@/types';

// Portfolio data interfaces
interface AssetAllocation {
  symbol: string;
  name: string;
  value: number;
  percentage: number;
  amount: number;
  change24h: number;
  color: string;
}

interface PortfolioHistoryPoint {
  date: string;
  value: number;
  pnl: number;
  timestamp: number;
}

interface TradingActivity {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  price: number;
  value: number;
  timestamp: string;
  exchange: string;
  pnl?: number;
}

interface RiskMetric {
  name: string;
  value: number;
  threshold: number;
  status: 'good' | 'warning' | 'danger';
  description: string;
}

interface TimeRange {
  label: string;
  value: string;
  days: number;
}

// Sample data for demonstration
const mockPortfolioHistory: PortfolioHistoryPoint[] = [
  { date: '2024-01-01', value: 10000, pnl: 0, timestamp: 1704067200000 },
  { date: '2024-01-02', value: 10250, pnl: 250, timestamp: 1704153600000 },
  { date: '2024-01-03', value: 10500, pnl: 500, timestamp: 1704240000000 },
  { date: '2024-01-04', value: 10300, pnl: 300, timestamp: 1704326400000 },
  { date: '2024-01-05', value: 10800, pnl: 800, timestamp: 1704412800000 },
  { date: '2024-01-06', value: 11200, pnl: 1200, timestamp: 1704499200000 },
  { date: '2024-01-07', value: 11500, pnl: 1500, timestamp: 1704585600000 },
];

const mockAllocation: AssetAllocation[] = [
  { 
    symbol: 'BTC', 
    name: 'Bitcoin', 
    value: 45, 
    percentage: 45, 
    amount: 5175, 
    change24h: 2.5,
    color: '#F7931A' 
  },
  { 
    symbol: 'ETH', 
    name: 'Ethereum', 
    value: 30, 
    percentage: 30, 
    amount: 3450, 
    change24h: -1.2,
    color: '#627EEA' 
  },
  { 
    symbol: 'SOL', 
    name: 'Solana', 
    value: 15, 
    percentage: 15, 
    amount: 1725, 
    change24h: 5.8,
    color: '#00FFA3' 
  },
  { 
    symbol: 'OTHERS', 
    name: 'Others', 
    value: 10, 
    percentage: 10, 
    amount: 1150, 
    change24h: 0.5,
    color: '#8B5CF6' 
  },
];

const mockTradingActivity: TradingActivity[] = [
  {
    id: '1',
    symbol: 'BTC/USDT',
    side: 'buy',
    quantity: 0.025,
    price: 42000,
    value: 1050,
    timestamp: '2024-01-07T10:30:00Z',
    exchange: 'Binance',
    pnl: 125
  },
  {
    id: '2',
    symbol: 'ETH/USDT',
    side: 'sell',
    quantity: 1.5,
    price: 2500,
    value: 3750,
    timestamp: '2024-01-07T09:15:00Z',
    exchange: 'Coinbase',
    pnl: -75
  },
  {
    id: '3',
    symbol: 'SOL/USDT',
    side: 'buy',
    quantity: 10,
    price: 85,
    value: 850,
    timestamp: '2024-01-06T16:45:00Z',
    exchange: 'OKX',
    pnl: 50
  },
];

const mockRiskMetrics: RiskMetric[] = [
  {
    name: 'Portfolio Beta',
    value: 1.25,
    threshold: 1.5,
    status: 'good',
    description: 'Measures portfolio volatility relative to market'
  },
  {
    name: 'Value at Risk (95%)',
    value: -850,
    threshold: -1000,
    status: 'good',
    description: 'Maximum expected loss over 1 day at 95% confidence'
  },
  {
    name: 'Correlation (BTC)',
    value: 0.78,
    threshold: 0.9,
    status: 'warning',
    description: 'Portfolio correlation with Bitcoin price movements'
  },
  {
    name: 'Concentration Risk',
    value: 0.45,
    threshold: 0.5,
    status: 'good',
    description: 'Largest single asset allocation percentage'
  },
];

const timeRanges: TimeRange[] = [
  { label: '1D', value: '1D', days: 1 },
  { label: '7D', value: '7D', days: 7 },
  { label: '30D', value: '30D', days: 30 },
  { label: '3M', value: '3M', days: 90 },
  { label: '1Y', value: '1Y', days: 365 },
  { label: 'All', value: 'All', days: 0 },
];

const COLORS = ['#F7931A', '#627EEA', '#00FFA3', '#8B5CF6', '#EF4444', '#10B981'];

const PortfolioPage: React.FC = () => {
  const dispatch = useAppDispatch();
  const { summary, positions, balances, isLoading, error } = useAppSelector(selectPortfolioState);
  
  const [activeTab, setActiveTab] = useState('overview');
  const [refreshing, setRefreshing] = useState(false);
  const [timeRange, setTimeRange] = useState('7D');
  const [showZeroBalances, setShowZeroBalances] = useState(false);
  const [selectedCurrency, setSelectedCurrency] = useState('USD');
  const [searchTerm, setSearchTerm] = useState('');

  // Fetch portfolio data on mount
  useEffect(() => {
    dispatch(fetchPortfolioSummary());
    dispatch(fetchPositions({}));
    dispatch(fetchBalances({}));

    // Subscribe to WebSocket channels for real-time updates
    websocketService.subscribe(['portfolio', 'positions']);

    // Set up periodic refresh for portfolio data
    const refreshInterval = setInterval(() => {
      dispatch(fetchPositions({}));
      dispatch(fetchBalances({}));
    }, 30000); // Refresh every 30 seconds

    return () => {
      websocketService.unsubscribe(['portfolio', 'positions']);
      clearInterval(refreshInterval);
    };
  }, [dispatch]);

  const handleRefresh = async () => {
    setRefreshing(true);
    await Promise.all([
      dispatch(fetchPortfolioSummary()),
      dispatch(fetchPositions({})),
      dispatch(fetchBalances({})),
    ]);
    setRefreshing(false);
  };

  const handleExport = () => {
    // Export portfolio data to CSV
    const data = {
      summary: portfolioSummary,
      positions,
      balances,
      timestamp: new Date().toISOString(),
    };
    
    const dataStr = JSON.stringify(data, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    
    const link = document.createElement('a');
    link.href = URL.createObjectURL(dataBlob);
    link.download = `portfolio-report-${new Date().toISOString().split('T')[0]}.json`;
    link.click();
  };

  const handleTimeRangeChange = (value: string) => {
    setTimeRange(value);
  };

  // Calculate mock summary if not available
  const portfolioSummary: PortfolioSummary = summary || {
    totalValue: 11500,
    totalPnl: 1500,
    dailyPnl: 300,
    dailyPnlPercentage: 2.68,
    weeklyPnl: 1500,
    weeklyPnlPercentage: 15.0,
    monthlyPnl: 1500,
    monthlyPnlPercentage: 15.0,
    positions: positions || [],
    balances: balances || [],
    openOrders: 2,
    winRate: 68.5,
    sharpeRatio: 1.85,
    maxDrawdown: -8.5,
    lastUpdated: new Date().toISOString(),
  };

  // Memoized calculations for performance
  const filteredPositions = useMemo(() => {
    return positions?.filter(position => 
      position.symbol.toLowerCase().includes(searchTerm.toLowerCase())
    ) || [];
  }, [positions, searchTerm]);

  const filteredBalances = useMemo(() => {
    return balances?.filter(balance => {
      const matchesSearch = balance.currency.toLowerCase().includes(searchTerm.toLowerCase());
      const hasBalance = showZeroBalances || balance.total > 0;
      return matchesSearch && hasBalance;
    }) || [];
  }, [balances, searchTerm, showZeroBalances]);

  if (error) {
    return (
      <div className="p-6">
        <Alert>
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>
            {error}
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  return (
    <TooltipProvider>
      <div className="flex-1 space-y-6 p-6">
        {/* Page Header */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="space-y-1">
              <h2 className="text-3xl font-bold tracking-tight">Portfolio Overview</h2>
              <p className="text-muted-foreground">
                Track your positions, balances, and performance metrics
              </p>
            </div>
            <div className="flex items-center space-x-2">
              <div className="flex items-center space-x-2">
                <Select value={selectedCurrency} onValueChange={setSelectedCurrency}>
                  <SelectTrigger className="w-[100px]">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="USD">USD</SelectItem>
                    <SelectItem value="BTC">BTC</SelectItem>
                    <SelectItem value="ETH">ETH</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="outline" size="sm" onClick={handleExport}>
                    <Download className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  Export portfolio data
                </TooltipContent>
              </Tooltip>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button 
                    variant="outline" 
                    size="sm" 
                    onClick={handleRefresh}
                    disabled={refreshing}
                  >
                    <RefreshCw className={cn("h-4 w-4", refreshing && "animate-spin")} />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  Refresh data
                </TooltipContent>
              </Tooltip>
            </div>
          </div>
          {refreshing && <Progress value={33} className="h-1" />}
        </div>

        {/* Portfolio Summary Cards */}
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            <Card className="border-primary/20 bg-gradient-to-br from-primary/5 to-primary/10">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Total Value</CardTitle>
                <Wallet className="h-4 w-4 text-primary" />
              </CardHeader>
              <CardContent>
                {isLoading ? (
                  <Skeleton className="h-8 w-24" />
                ) : (
                  <div className="space-y-1">
                    <div className="text-2xl font-bold">{formatPrice(portfolioSummary.totalValue)}</div>
                    <Badge variant="outline" className="text-xs">
                      {portfolioSummary.positions.length} positions
                    </Badge>
                  </div>
                )}
              </CardContent>
            </Card>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.1 }}
          >
            <Card className={cn(
              "border-opacity-20 bg-gradient-to-br to-opacity-10",
              portfolioSummary.dailyPnl >= 0 
                ? "border-green-500 from-green-500/5 to-green-500/10" 
                : "border-red-500 from-red-500/5 to-red-500/10"
            )}>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Daily P&L</CardTitle>
                {portfolioSummary.dailyPnl >= 0 ? (
                  <TrendingUp className="h-4 w-4 text-green-500" />
                ) : (
                  <TrendingDown className="h-4 w-4 text-red-500" />
                )}
              </CardHeader>
              <CardContent>
                {isLoading ? (
                  <Skeleton className="h-8 w-24" />
                ) : (
                  <div className="space-y-1">
                    <div className={cn("text-2xl font-bold", getPnLColor(portfolioSummary.dailyPnl))}>
                      {formatPnL(portfolioSummary.dailyPnl)}
                    </div>
                    <div className="flex items-center space-x-1 text-sm">
                      {portfolioSummary.dailyPnlPercentage >= 0 ? (
                        <ArrowUpRight className="h-3 w-3 text-green-500" />
                      ) : (
                        <ArrowDownRight className="h-3 w-3 text-red-500" />
                      )}
                      <span className={getPnLColor(portfolioSummary.dailyPnlPercentage)}>
                        {formatPercent(portfolioSummary.dailyPnlPercentage)}
                      </span>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.2 }}
          >
            <Card className="border-blue-500/20 bg-gradient-to-br from-blue-500/5 to-blue-500/10">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Win Rate</CardTitle>
                <Target className="h-4 w-4 text-blue-500" />
              </CardHeader>
              <CardContent>
                {isLoading ? (
                  <Skeleton className="h-8 w-24" />
                ) : (
                  <div className="space-y-1">
                    <div className="text-2xl font-bold">{portfolioSummary.winRate.toFixed(1)}%</div>
                    <p className="text-xs text-muted-foreground">
                      Sharpe: {portfolioSummary.sharpeRatio.toFixed(2)}
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.3 }}
          >
            <Card className="border-orange-500/20 bg-gradient-to-br from-orange-500/5 to-orange-500/10">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Max Drawdown</CardTitle>
                <AlertTriangle className="h-4 w-4 text-orange-500" />
              </CardHeader>
              <CardContent>
                {isLoading ? (
                  <Skeleton className="h-8 w-24" />
                ) : (
                  <div className="space-y-1">
                    <div className="text-2xl font-bold text-orange-600">
                      {portfolioSummary.maxDrawdown.toFixed(1)}%
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Risk level: Medium
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>
          </motion.div>
        </div>

        {/* Charts Section */}
        <div className="grid gap-4 lg:grid-cols-3">
          <Card className="lg:col-span-2">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Portfolio Value</CardTitle>
                  <CardDescription>Historical performance over time</CardDescription>
                </div>
                <div className="flex items-center space-x-1">
                  {timeRanges.map((range) => (
                    <Button
                      key={range.value}
                      variant={timeRange === range.value ? 'default' : 'outline'}
                      size="sm"
                      onClick={() => handleTimeRangeChange(range.value)}
                      className="h-8 w-12 text-xs"
                    >
                      {range.label}
                    </Button>
                  ))}
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="h-[300px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={mockPortfolioHistory}>
                    <defs>
                      <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                    <XAxis
                      dataKey="date"
                      tick={{ fontSize: 12 }}
                      className="text-xs"
                      tickFormatter={(value) => {
                        const date = new Date(value);
                        return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
                      }}
                    />
                    <YAxis
                      tick={{ fontSize: 12 }}
                      className="text-xs"
                      tickFormatter={(value) => `$${(value / 1000).toFixed(1)}k`}
                    />
                    <ChartTooltip
                      content={({ active, payload, label }) => {
                        if (active && payload && payload.length) {
                          return (
                            <div className="rounded-lg border bg-background p-2 shadow-sm">
                              <p className="text-xs text-muted-foreground">
                                {new Date(label).toLocaleDateString()}
                              </p>
                              <p className="font-medium">
                                {formatPrice(payload[0]?.value as number)}
                              </p>
                            </div>
                          );
                        }
                        return null;
                      }}
                    />
                    <Area
                      type="monotone"
                      dataKey="value"
                      stroke="hsl(var(--primary))"
                      strokeWidth={2}
                      fill="url(#colorValue)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Asset Allocation</CardTitle>
              <CardDescription>Portfolio distribution by asset</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[280px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={mockAllocation}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, value }) => `${name}: ${value}%`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {mockAllocation.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <ChartTooltip
                      content={({ active, payload }) => {
                        if (active && payload && payload.length) {
                          const data = payload[0].payload as AssetAllocation;
                          return (
                            <div className="rounded-lg border bg-background p-2 shadow-sm">
                              <p className="font-medium">{data.name}</p>
                              <p className="text-sm text-muted-foreground">
                                {data.percentage}% • {formatPrice(data.amount)}
                              </p>
                            </div>
                          );
                        }
                        return null;
                      }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-4 space-y-2">
                {mockAllocation.map((asset, index) => (
                  <div key={asset.symbol} className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <div 
                        className="h-3 w-3 rounded-full" 
                        style={{ backgroundColor: asset.color }}
                      />
                      <span className="text-sm font-medium">{asset.name}</span>
                      <Badge 
                        variant="secondary" 
                        className={cn("text-xs", getPnLColor(asset.change24h))}
                      >
                        {formatPercent(asset.change24h)}
                      </Badge>
                    </div>
                    <span className="text-sm font-medium">{formatPrice(asset.amount)}</span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Main Content Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
          <TabsList className="grid w-full grid-cols-5">
            <TabsTrigger value="overview" className="flex items-center space-x-2">
              <BarChart3 className="h-4 w-4" />
              <span>Overview</span>
            </TabsTrigger>
            <TabsTrigger value="positions" className="flex items-center space-x-2">
              <Activity className="h-4 w-4" />
              <span>Positions</span>
            </TabsTrigger>
            <TabsTrigger value="balances" className="flex items-center space-x-2">
              <Wallet className="h-4 w-4" />
              <span>Balances</span>
            </TabsTrigger>
            <TabsTrigger value="analytics" className="flex items-center space-x-2">
              <PieChartIcon className="h-4 w-4" />
              <span>Analytics</span>
            </TabsTrigger>
            <TabsTrigger value="activity" className="flex items-center space-x-2">
              <Activity className="h-4 w-4" />
              <span>Activity</span>
            </TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-4">
            <OverviewTab 
              summary={portfolioSummary}
              positions={filteredPositions}
              isLoading={isLoading}
            />
          </TabsContent>

          <TabsContent value="positions" className="space-y-4">
            <PositionsTab 
              positions={filteredPositions}
              isLoading={isLoading}
              searchTerm={searchTerm}
              onSearchChange={setSearchTerm}
            />
          </TabsContent>

          <TabsContent value="balances" className="space-y-4">
            <BalancesTab 
              balances={filteredBalances}
              isLoading={isLoading}
              searchTerm={searchTerm}
              onSearchChange={setSearchTerm}
              showZeroBalances={showZeroBalances}
              onShowZeroBalancesChange={setShowZeroBalances}
            />
          </TabsContent>

          <TabsContent value="analytics" className="space-y-4">
            <AnalyticsTab 
              summary={portfolioSummary}
              riskMetrics={mockRiskMetrics}
              isLoading={isLoading}
            />
          </TabsContent>

          <TabsContent value="activity" className="space-y-4">
            <ActivityTab 
              activities={mockTradingActivity}
              isLoading={isLoading}
            />
          </TabsContent>
        </Tabs>
      </div>
    </TooltipProvider>
  );
};

// Overview Tab Component
const OverviewTab: React.FC<{
  summary: PortfolioSummary;
  positions: Position[];
  isLoading: boolean;
}> = ({ summary, positions, isLoading }) => {
  return (
    <div className="space-y-6">
      {/* Quick Stats */}
      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Total Positions</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{positions.length}</div>
            <p className="text-xs text-muted-foreground">
              {positions.filter(p => p.side === 'long').length} long, {positions.filter(p => p.side === 'short').length} short
            </p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Open Orders</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{summary.openOrders}</div>
            <p className="text-xs text-muted-foreground">Active in market</p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Total P&L</CardTitle>
          </CardHeader>
          <CardContent>
            <div className={cn("text-2xl font-bold", getPnLColor(summary.totalPnl))}>
              {formatPnL(summary.totalPnl)}
            </div>
            <p className="text-xs text-muted-foreground">All-time performance</p>
          </CardContent>
        </Card>
      </div>

      {/* Recent Positions */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Positions</CardTitle>
          <CardDescription>Your most recently updated positions</CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="space-y-2">
              {[...Array(3)].map((_, i) => (
                <div key={i} className="flex items-center space-x-4">
                  <Skeleton className="h-12 w-12 rounded-full" />
                  <div className="space-y-2">
                    <Skeleton className="h-4 w-[250px]" />
                    <Skeleton className="h-4 w-[200px]" />
                  </div>
                </div>
              ))}
            </div>
          ) : positions.length === 0 ? (
            <div className="text-center py-6">
              <p className="text-muted-foreground">No positions found</p>
            </div>
          ) : (
            <div className="space-y-4">
              {positions.slice(0, 5).map((position) => (
                <div key={position.id} className="flex items-center justify-between border-b pb-4 last:border-b-0">
                  <div className="flex items-center space-x-3">
                    <Avatar className="h-10 w-10">
                      <AvatarImage src={`https://cryptoicons.org/api/icon/${position.symbol.split('/')[0].toLowerCase()}/200`} />
                      <AvatarFallback>{position.symbol.split('/')[0].slice(0, 2)}</AvatarFallback>
                    </Avatar>
                    <div>
                      <p className="font-medium">{position.symbol}</p>
                      <p className="text-sm text-muted-foreground">
                        {position.side} • {position.quantity} @ {formatPrice(position.entryPrice)}
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="font-medium">{formatPrice(position.currentPrice)}</p>
                    <p className={cn("text-sm", getPnLColor(position.unrealizedPnl))}>
                      {formatPnL(position.unrealizedPnl)}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

// Positions Tab Component
const PositionsTab: React.FC<{
  positions: Position[];
  isLoading: boolean;
  searchTerm: string;
  onSearchChange: (value: string) => void;
}> = ({ positions, isLoading, searchTerm, onSearchChange }) => {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Positions</CardTitle>
            <CardDescription>Manage your trading positions</CardDescription>
          </div>
          <div className="flex items-center space-x-2">
            <div className="relative">
              <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
              <input
                placeholder="Search positions..."
                value={searchTerm}
                onChange={(e) => onSearchChange(e.target.value)}
                className="pl-8 h-9 w-[200px] rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm transition-colors placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring"
              />
            </div>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="space-y-2">
            {[...Array(5)].map((_, i) => (
              <Skeleton key={i} className="h-16 w-full" />
            ))}
          </div>
        ) : (
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Symbol</TableHead>
                  <TableHead>Side</TableHead>
                  <TableHead className="text-right">Quantity</TableHead>
                  <TableHead className="text-right">Entry Price</TableHead>
                  <TableHead className="text-right">Current Price</TableHead>
                  <TableHead className="text-right">P&L</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {positions.map((position) => (
                  <TableRow key={position.id}>
                    <TableCell className="font-medium">
                      <div className="flex items-center space-x-2">
                        <Avatar className="h-6 w-6">
                          <AvatarImage src={`https://cryptoicons.org/api/icon/${position.symbol.split('/')[0].toLowerCase()}/200`} />
                          <AvatarFallback className="text-xs">{position.symbol.split('/')[0].slice(0, 2)}</AvatarFallback>
                        </Avatar>
                        <span>{position.symbol}</span>
                      </div>
                    </TableCell>
                    <TableCell>
                      <Badge variant={position.side === 'long' ? 'default' : 'secondary'}>
                        {position.side.toUpperCase()}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-right">{position.quantity}</TableCell>
                    <TableCell className="text-right">{formatPrice(position.entryPrice)}</TableCell>
                    <TableCell className="text-right">{formatPrice(position.currentPrice)}</TableCell>
                    <TableCell className={cn("text-right font-medium", getPnLColor(position.unrealizedPnl))}>
                      {formatPnL(position.unrealizedPnl)}
                    </TableCell>
                    <TableCell className="text-right">
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button variant="ghost" className="h-8 w-8 p-0">
                            <MoreHorizontal className="h-4 w-4" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          <DropdownMenuLabel>Actions</DropdownMenuLabel>
                          <DropdownMenuItem>
                            <Plus className="mr-2 h-4 w-4" />
                            Add to position
                          </DropdownMenuItem>
                          <DropdownMenuItem>
                            <Minus className="mr-2 h-4 w-4" />
                            Reduce position
                          </DropdownMenuItem>
                          <DropdownMenuSeparator />
                          <DropdownMenuItem className="text-red-600">
                            Close position
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
            {positions.length === 0 && (
              <div className="text-center py-10">
                <p className="text-muted-foreground">No positions found</p>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
};

// Balances Tab Component
const BalancesTab: React.FC<{
  balances: Balance[];
  isLoading: boolean;
  searchTerm: string;
  onSearchChange: (value: string) => void;
  showZeroBalances: boolean;
  onShowZeroBalancesChange: (value: boolean) => void;
}> = ({ balances, isLoading, searchTerm, onSearchChange, showZeroBalances, onShowZeroBalancesChange }) => {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Balances</CardTitle>
            <CardDescription>Your exchange balances and holdings</CardDescription>
          </div>
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <Switch
                checked={showZeroBalances}
                onCheckedChange={onShowZeroBalancesChange}
              />
              <span className="text-sm">Show zero balances</span>
            </div>
            <div className="relative">
              <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
              <input
                placeholder="Search balances..."
                value={searchTerm}
                onChange={(e) => onSearchChange(e.target.value)}
                className="pl-8 h-9 w-[200px] rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm transition-colors placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring"
              />
            </div>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="space-y-2">
            {[...Array(8)].map((_, i) => (
              <Skeleton key={i} className="h-14 w-full" />
            ))}
          </div>
        ) : (
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Currency</TableHead>
                  <TableHead>Exchange</TableHead>
                  <TableHead className="text-right">Free</TableHead>
                  <TableHead className="text-right">Locked</TableHead>
                  <TableHead className="text-right">Total</TableHead>
                  <TableHead className="text-right">USD Value</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {balances.map((balance, index) => (
                  <TableRow key={`${balance.currency}-${balance.exchange}-${index}`}>
                    <TableCell className="font-medium">
                      <div className="flex items-center space-x-2">
                        <Avatar className="h-6 w-6">
                          <AvatarImage src={`https://cryptoicons.org/api/icon/${balance.currency.toLowerCase()}/200`} />
                          <AvatarFallback className="text-xs">{balance.currency.slice(0, 2)}</AvatarFallback>
                        </Avatar>
                        <span>{balance.currency}</span>
                      </div>
                    </TableCell>
                    <TableCell>
                      <Badge variant="outline">{balance.exchange}</Badge>
                    </TableCell>
                    <TableCell className="text-right font-mono">{balance.free.toFixed(8)}</TableCell>
                    <TableCell className="text-right font-mono">{balance.locked.toFixed(8)}</TableCell>
                    <TableCell className="text-right font-mono font-medium">{balance.total.toFixed(8)}</TableCell>
                    <TableCell className="text-right font-medium">{formatPrice(balance.usdValue)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
            {balances.length === 0 && (
              <div className="text-center py-10">
                <p className="text-muted-foreground">No balances found</p>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
};

// Analytics Tab Component
const AnalyticsTab: React.FC<{
  summary: PortfolioSummary;
  riskMetrics: RiskMetric[];
  isLoading: boolean;
}> = ({ summary, riskMetrics, isLoading }) => {
  return (
    <div className="space-y-6">
      {/* Performance Metrics */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Sharpe Ratio</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{summary.sharpeRatio.toFixed(2)}</div>
            <p className="text-xs text-muted-foreground">Risk-adjusted returns</p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Max Drawdown</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-red-600">{summary.maxDrawdown.toFixed(1)}%</div>
            <p className="text-xs text-muted-foreground">Worst peak-to-trough</p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Win Rate</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">{summary.winRate.toFixed(1)}%</div>
            <p className="text-xs text-muted-foreground">Successful trades</p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Volatility</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">12.5%</div>
            <p className="text-xs text-muted-foreground">Annualized volatility</p>
          </CardContent>
        </Card>
      </div>

      {/* Risk Metrics */}
      <Card>
        <CardHeader>
          <CardTitle>Risk Metrics</CardTitle>
          <CardDescription>Portfolio risk assessment and monitoring</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {riskMetrics.map((metric) => (
            <div key={metric.name} className="space-y-2">
              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <p className="font-medium">{metric.name}</p>
                  <p className="text-xs text-muted-foreground">{metric.description}</p>
                </div>
                <div className="text-right">
                  <p className="font-mono font-medium">
                    {typeof metric.value === 'number' && metric.value < 0 
                      ? formatPnL(metric.value) 
                      : metric.value.toFixed(2)
                    }
                  </p>
                  <Badge 
                    variant={metric.status === 'good' ? 'default' : metric.status === 'warning' ? 'secondary' : 'destructive'}
                    className="text-xs"
                  >
                    {metric.status}
                  </Badge>
                </div>
              </div>
              <Progress 
                value={Math.abs(metric.value / metric.threshold) * 100} 
                className="h-2" 
              />
            </div>
          ))}
        </CardContent>
      </Card>

      {/* P&L Distribution Chart */}
      <Card>
        <CardHeader>
          <CardTitle>P&L Distribution</CardTitle>
          <CardDescription>Historical profit and loss distribution</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-[300px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={mockPortfolioHistory}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                <XAxis
                  dataKey="date"
                  tick={{ fontSize: 12 }}
                  className="text-xs"
                  tickFormatter={(value) => {
                    const date = new Date(value);
                    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
                  }}
                />
                <YAxis
                  tick={{ fontSize: 12 }}
                  className="text-xs"
                  tickFormatter={(value) => formatPrice(value, 0)}
                />
                <ChartTooltip
                  content={({ active, payload, label }) => {
                    if (active && payload && payload.length) {
                      const pnl = payload[0]?.value as number;
                      return (
                        <div className="rounded-lg border bg-background p-2 shadow-sm">
                          <p className="text-xs text-muted-foreground">
                            {new Date(label).toLocaleDateString()}
                          </p>
                          <p className={cn("font-medium", getPnLColor(pnl))}>
                            {formatPnL(pnl)}
                          </p>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Bar
                  dataKey="pnl"
                  fill="hsl(var(--primary))"
                  className="fill-current"
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

// Activity Tab Component
const ActivityTab: React.FC<{
  activities: TradingActivity[];
  isLoading: boolean;
}> = ({ activities, isLoading }) => {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Trading Activity</CardTitle>
        <CardDescription>Recent trades and transactions</CardDescription>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="space-y-4">
            {[...Array(5)].map((_, i) => (
              <div key={i} className="flex items-center space-x-4">
                <Skeleton className="h-10 w-10 rounded-full" />
                <div className="space-y-2 flex-1">
                  <Skeleton className="h-4 w-full" />
                  <Skeleton className="h-4 w-3/4" />
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="space-y-4">
            {activities.map((activity) => (
              <div key={activity.id} className="flex items-center justify-between border-b pb-4 last:border-b-0">
                <div className="flex items-center space-x-3">
                  <Avatar className="h-10 w-10">
                    <AvatarImage src={`https://cryptoicons.org/api/icon/${activity.symbol.split('/')[0].toLowerCase()}/200`} />
                    <AvatarFallback>{activity.symbol.split('/')[0].slice(0, 2)}</AvatarFallback>
                  </Avatar>
                  <div>
                    <div className="flex items-center space-x-2">
                      <p className="font-medium">{activity.symbol}</p>
                      <Badge variant={activity.side === 'buy' ? 'default' : 'secondary'}>
                        {activity.side.toUpperCase()}
                      </Badge>
                    </div>
                    <p className="text-sm text-muted-foreground">
                      {activity.quantity} @ {formatPrice(activity.price)} • {activity.exchange}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {new Date(activity.timestamp).toLocaleString()}
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="font-medium">{formatPrice(activity.value)}</p>
                  {activity.pnl && (
                    <p className={cn("text-sm", getPnLColor(activity.pnl))}>
                      {formatPnL(activity.pnl)}
                    </p>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default PortfolioPage;