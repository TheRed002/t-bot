/**
 * Dashboard page component
 * Professional trading dashboard with real-time data, charts, and comprehensive metrics
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import {
  Activity,
  ArrowUpIcon,
  ArrowDownIcon,
  BarChart3,
  Bot,
  DollarSign,
  RefreshCw,
  Settings,
  TrendingUp,
  TrendingDown,
  Wallet,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Clock,
  Eye,
  EyeOff,
  Users,
} from 'lucide-react';

// Shadcn UI components
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Skeleton } from '@/components/ui/skeleton';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';

// Utils
import { cn, formatPrice, formatPercent, formatPnL, getPnLColor, formatDateTime, formatTime } from '@/lib/utils';
import type { RootState } from '@/store';
import type { PortfolioSummary, Position, Trade, BotInstance } from '@/types';

// Mock data - In real app, this would come from Redux store/API
const mockPortfolioData = {
  totalValue: 125450.67,
  dailyChange: 2345.32,
  dailyChangePercent: 1.91,
  activeBots: 5,
  runningBots: 3,
  pausedBots: 2,
  totalPositions: 8,
  todayVolume: 89432.12,
  dailyPnL: 1892.45,
  winRate: 72.5,
  riskLevel: 'Medium',
  portfolioExposure: 3.2,
  weeklyReturn: 5.67,
  monthlyReturn: 12.45,
  maxDrawdown: -2.34,
  sharpeRatio: 1.87,
  totalTrades: 1247,
  avgTradeReturn: 0.23,
};

const mockPortfolioChartData = Array.from({ length: 30 }, (_, i) => {
  const baseValue = 120000;
  const date = new Date();
  date.setDate(date.getDate() - (30 - i));
  const noise = (Math.random() - 0.5) * 5000;
  const trend = i * 150;
  const value = baseValue + trend + noise;
  
  return {
    date: date.toISOString().split('T')[0],
    value,
    timestamp: date.getTime(),
  };
});

const mockPnLDistribution = [
  { range: '-5% to -3%', count: 12, pnl: -1250 },
  { range: '-3% to -1%', count: 23, pnl: -890 },
  { range: '-1% to 0%', count: 45, pnl: -345 },
  { range: '0% to 1%', count: 78, pnl: 234 },
  { range: '1% to 3%', count: 156, pnl: 1890 },
  { range: '3% to 5%', count: 89, pnl: 2340 },
  { range: '5%+', count: 34, pnl: 4560 },
];

const mockRecentTrades = [
  {
    id: '1',
    time: new Date(Date.now() - 1800000).toISOString(),
    pair: 'BTC/USDT',
    side: 'buy' as const,
    price: 45123.45,
    amount: 0.0234,
    pnl: 234.56,
    exchange: 'Binance',
    bot: 'Momentum Bot',
  },
  {
    id: '2',
    time: new Date(Date.now() - 2400000).toISOString(),
    pair: 'ETH/USDT',
    side: 'sell' as const,
    price: 2845.67,
    amount: 1.456,
    pnl: -45.23,
    exchange: 'Coinbase',
    bot: 'Mean Reversion Bot',
  },
  {
    id: '3',
    time: new Date(Date.now() - 3600000).toISOString(),
    pair: 'ADA/USDT',
    side: 'buy' as const,
    price: 0.4567,
    amount: 1000,
    pnl: 89.12,
    exchange: 'Binance',
    bot: 'Grid Bot',
  },
  {
    id: '4',
    time: new Date(Date.now() - 4800000).toISOString(),
    pair: 'SOL/USDT',
    side: 'sell' as const,
    price: 98.34,
    amount: 5.23,
    pnl: 156.78,
    exchange: 'OKX',
    bot: 'DCA Bot',
  },
];

const mockActiveBots = [
  {
    id: '1',
    name: 'BTC Momentum Bot',
    status: 'running' as const,
    strategy: 'Momentum',
    pnl: 1234.56,
    trades: 45,
    winRate: 78.5,
    uptime: '2d 14h',
  },
  {
    id: '2',
    name: 'ETH Grid Bot',
    status: 'running' as const,
    strategy: 'Grid Trading',
    pnl: 567.89,
    trades: 123,
    winRate: 65.2,
    uptime: '5d 8h',
  },
  {
    id: '3',
    name: 'DCA Accumulator',
    status: 'paused' as const,
    strategy: 'DCA',
    pnl: 89.45,
    trades: 23,
    winRate: 87.0,
    uptime: '1d 3h',
  },
];

const mockMarketData = [
  { symbol: 'BTC/USDT', price: 45234.67, change: 2.34, volume: '2.45B' },
  { symbol: 'ETH/USDT', price: 2845.23, change: 1.89, volume: '1.23B' },
  { symbol: 'BNB/USDT', price: 334.56, change: -0.87, volume: '456M' },
  { symbol: 'ADA/USDT', price: 0.4567, change: 5.67, volume: '234M' },
  { symbol: 'SOL/USDT', price: 98.34, change: -2.13, volume: '189M' },
  { symbol: 'DOT/USDT', price: 6.78, change: 3.45, volume: '67M' },
];

interface MetricCardProps {
  title: string;
  value: string | number;
  change?: number;
  icon: React.ReactNode;
  loading?: boolean;
  className?: string;
}

const MetricCard: React.FC<MetricCardProps> = ({ title, value, change, icon, loading, className }) => {
  if (loading) {
    return (
      <Card className={className}>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <Skeleton className="h-4 w-20" />
          <Skeleton className="h-4 w-4" />
        </CardHeader>
        <CardContent>
          <Skeleton className="h-8 w-full mb-1" />
          <Skeleton className="h-3 w-16" />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={className}>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground">{title}</CardTitle>
        <div className="text-muted-foreground">{icon}</div>
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold font-mono">{value}</div>
        {change !== undefined && (
          <div className={cn(
            "text-xs flex items-center gap-1",
            change >= 0 ? "text-green-500" : "text-red-500"
          )}>
            {change >= 0 ? (
              <ArrowUpIcon className="h-3 w-3" />
            ) : (
              <ArrowDownIcon className="h-3 w-3" />
            )}
            {formatPercent(Math.abs(change))}
          </div>
        )}
      </CardContent>
    </Card>
  );
};

interface DashboardProps {
  // Add any props if needed
}

const DashboardPage: React.FC<DashboardProps> = () => {
  const [loading, setLoading] = useState(false);
  const [lastUpdate, setLastUpdate] = useState(new Date());
  const [autoRefresh, setAutoRefresh] = useState(true);
  
  // In real app, these would come from Redux store
  // const portfolio = useSelector((state: RootState) => state.portfolio.summary);
  // const bots = useSelector((state: RootState) => state.bot.bots);
  // const websocketConnected = useSelector((state: RootState) => state.websocket.connected);
  
  // Simulate real-time updates
  useEffect(() => {
    if (!autoRefresh) return;
    
    const interval = setInterval(() => {
      setLastUpdate(new Date());
      // In real app, this would trigger data refresh from WebSocket
    }, 5000);
    
    return () => clearInterval(interval);
  }, [autoRefresh]);

  const handleRefresh = useCallback(async () => {
    setLoading(true);
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1000));
    setLastUpdate(new Date());
    setLoading(false);
  }, []);

  const connectionStatus = 'connected'; // In real app: websocketConnected ? 'connected' : 'disconnected'

  return (
    <TooltipProvider>
      <div className="space-y-6 p-6">
        {/* Header */}
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">Trading Dashboard</h1>
            <p className="text-muted-foreground">
              Real-time portfolio monitoring and bot management • Last updated: {formatTime(lastUpdate)}
            </p>
          </div>
          
          <div className="flex items-center gap-2">
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={handleRefresh}
                    disabled={loading}
                  >
                    <RefreshCw className={cn("h-4 w-4", loading && "animate-spin")} />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Refresh data</TooltipContent>
              </Tooltip>
            </TooltipProvider>
            
            <Button variant="outline" size="icon">
              <Settings className="h-4 w-4" />
            </Button>
            
            <Badge variant={connectionStatus === 'connected' ? 'default' : 'destructive'}>
              {connectionStatus === 'connected' ? 'Live' : 'Offline'}
            </Badge>
          </div>
        </div>

        {/* Status Alert */}
        <Alert>
          <CheckCircle className="h-4 w-4" />
          <AlertDescription>
            All systems operational • {mockActiveBots.filter(b => b.status === 'running').length} bots running • 
            Portfolio up {formatPercent(mockPortfolioData.dailyChangePercent)} today
          </AlertDescription>
        </Alert>

        {/* Key Metrics Grid */}
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <MetricCard
            title="Total Portfolio Value"
            value={formatPrice(mockPortfolioData.totalValue)}
            change={mockPortfolioData.dailyChangePercent}
            icon={<Wallet className="h-4 w-4" />}
            loading={loading}
          />
          <MetricCard
            title="24h P&L"
            value={formatPnL(mockPortfolioData.dailyPnL)}
            change={mockPortfolioData.dailyChangePercent}
            icon={<TrendingUp className="h-4 w-4" />}
            loading={loading}
          />
          <MetricCard
            title="Active Bots"
            value={`${mockPortfolioData.runningBots}/${mockPortfolioData.activeBots}`}
            icon={<Bot className="h-4 w-4" />}
            loading={loading}
          />
          <MetricCard
            title="Total Positions"
            value={mockPortfolioData.totalPositions}
            icon={<BarChart3 className="h-4 w-4" />}
            loading={loading}
          />
          <MetricCard
            title="Today's Volume"
            value={formatPrice(mockPortfolioData.todayVolume)}
            icon={<Activity className="h-4 w-4" />}
            loading={loading}
          />
          <MetricCard
            title="Win Rate"
            value={`${mockPortfolioData.winRate}%`}
            icon={<TrendingUp className="h-4 w-4" />}
            loading={loading}
          />
        </div>

        {/* Main Content */}
        <Tabs defaultValue="overview" className="space-y-6">
          <TabsList>
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="trades">Recent Trades</TabsTrigger>
            <TabsTrigger value="bots">Active Bots</TabsTrigger>
            <TabsTrigger value="market">Market</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-6">
            <div className="grid gap-6 md:grid-cols-2">
              {/* Portfolio Value Chart */}
              <Card className="col-span-full">
                <CardHeader>
                  <CardTitle>Portfolio Value Over Time</CardTitle>
                  <CardDescription>Last 30 days performance</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-[300px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={mockPortfolioChartData}>
                        <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                        <XAxis 
                          dataKey="date" 
                          tick={{ fontSize: 12 }}
                          tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                        />
                        <YAxis 
                          tick={{ fontSize: 12 }}
                          tickFormatter={(value) => `$${(value / 1000).toFixed(0)}K`}
                        />
                        <RechartsTooltip 
                          formatter={(value: any) => [formatPrice(value), 'Portfolio Value']}
                          labelFormatter={(label) => new Date(label).toLocaleDateString()}
                          contentStyle={{ 
                            backgroundColor: 'hsl(var(--background))', 
                            border: '1px solid hsl(var(--border))',
                            borderRadius: '6px'
                          }}
                        />
                        <Line 
                          type="monotone" 
                          dataKey="value" 
                          stroke="hsl(var(--primary))" 
                          strokeWidth={2}
                          dot={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>

              {/* P&L Distribution Chart */}
              <Card>
                <CardHeader>
                  <CardTitle>P&L Distribution</CardTitle>
                  <CardDescription>Trade outcome distribution</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-[300px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={mockPnLDistribution}>
                        <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                        <XAxis 
                          dataKey="range" 
                          tick={{ fontSize: 10 }}
                          angle={-45}
                          textAnchor="end"
                          height={60}
                        />
                        <YAxis tick={{ fontSize: 12 }} />
                        <RechartsTooltip 
                          formatter={(value: any, name: string) => [
                            name === 'count' ? value : formatPrice(value),
                            name === 'count' ? 'Trades' : 'Total P&L'
                          ]}
                          contentStyle={{ 
                            backgroundColor: 'hsl(var(--background))', 
                            border: '1px solid hsl(var(--border))',
                            borderRadius: '6px'
                          }}
                        />
                        <Bar dataKey="count" fill="hsl(var(--primary))" radius={[2, 2, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>

              {/* Trading Statistics */}
              <Card>
                <CardHeader>
                  <CardTitle>Trading Statistics</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">Total Trades</span>
                    <span className="font-mono">{mockPortfolioData.totalTrades.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">Win Rate</span>
                    <span className="font-mono text-green-500">{mockPortfolioData.winRate}%</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">Sharpe Ratio</span>
                    <span className="font-mono">{mockPortfolioData.sharpeRatio}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">Max Drawdown</span>
                    <span className="font-mono text-red-500">{formatPercent(mockPortfolioData.maxDrawdown)}</span>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="trades" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Recent Trades</CardTitle>
                <CardDescription>Latest executed trades across all bots</CardDescription>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Time</TableHead>
                      <TableHead>Pair</TableHead>
                      <TableHead>Side</TableHead>
                      <TableHead className="text-right">Price</TableHead>
                      <TableHead className="text-right">Amount</TableHead>
                      <TableHead className="text-right">P&L</TableHead>
                      <TableHead>Bot</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {mockRecentTrades.map((trade) => (
                      <TableRow key={trade.id}>
                        <TableCell className="font-mono text-sm">
                          {formatTime(trade.time)}
                        </TableCell>
                        <TableCell className="font-mono">{trade.pair}</TableCell>
                        <TableCell>
                          <Badge variant={trade.side === 'buy' ? 'default' : 'destructive'}>
                            {trade.side.toUpperCase()}
                          </Badge>
                        </TableCell>
                        <TableCell className="text-right font-mono">
                          {formatPrice(trade.price)}
                        </TableCell>
                        <TableCell className="text-right font-mono">
                          {trade.amount.toFixed(6)}
                        </TableCell>
                        <TableCell className={cn(
                          "text-right font-mono",
                          getPnLColor(trade.pnl)
                        )}>
                          {formatPnL(trade.pnl)}
                        </TableCell>
                        <TableCell className="text-sm text-muted-foreground">
                          {trade.bot}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="bots" className="space-y-6">
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {mockActiveBots.map((bot) => (
                <Card key={bot.id}>
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-lg">{bot.name}</CardTitle>
                      <Badge variant={bot.status === 'running' ? 'default' : 'secondary'}>
                        {bot.status === 'running' ? (
                          <><CheckCircle className="w-3 h-3 mr-1" /> Running</>
                        ) : (
                          <><XCircle className="w-3 h-3 mr-1" /> Paused</>
                        )}
                      </Badge>
                    </div>
                    <CardDescription>{bot.strategy}</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-muted-foreground">P&L</span>
                      <span className={cn(
                        "font-mono text-sm",
                        getPnLColor(bot.pnl)
                      )}>
                        {formatPnL(bot.pnl)}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-muted-foreground">Trades</span>
                      <span className="font-mono text-sm">{bot.trades}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-muted-foreground">Win Rate</span>
                      <span className="font-mono text-sm text-green-500">{bot.winRate}%</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-muted-foreground">Uptime</span>
                      <span className="font-mono text-sm">{bot.uptime}</span>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="market" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Market Overview</CardTitle>
                <CardDescription>Top cryptocurrency pairs by volume</CardDescription>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Symbol</TableHead>
                      <TableHead className="text-right">Price</TableHead>
                      <TableHead className="text-right">24h Change</TableHead>
                      <TableHead className="text-right">Volume</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {mockMarketData.map((market) => (
                      <TableRow key={market.symbol}>
                        <TableCell className="font-mono">{market.symbol}</TableCell>
                        <TableCell className="text-right font-mono">
                          {formatPrice(market.price)}
                        </TableCell>
                        <TableCell className={cn(
                          "text-right font-mono",
                          getPnLColor(market.change)
                        )}>
                          {formatPercent(market.change)}
                        </TableCell>
                        <TableCell className="text-right font-mono text-muted-foreground">
                          {market.volume}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </TooltipProvider>
  );
};

export default DashboardPage;