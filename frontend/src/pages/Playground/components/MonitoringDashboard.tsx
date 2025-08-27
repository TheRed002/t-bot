/**
 * Monitoring Dashboard Component for Playground
 * Real-time monitoring of execution status, logs, and performance metrics
 */

import React, { useState, useCallback, useEffect, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Switch } from '@/components/ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import {
  Search,
  Filter,
  RefreshCw,
  Download,
  Play,
  Pause,
  AlertTriangle,
  AlertCircle,
  Info,
  TrendingUp,
  TrendingDown,
  LineChart,
  Eye,
  EyeOff
} from 'lucide-react';
import {
  LineChart as RechartsLineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as ChartTooltip,
  ResponsiveContainer,
  AreaChart,
  Area
} from 'recharts';

import { PlaygroundConfiguration, PlaygroundExecution, PlaygroundLog, PlaygroundTrade } from '@/types';

interface MonitoringDashboardProps {
  execution: PlaygroundExecution | null;
  configuration: PlaygroundConfiguration | null;
}

// Mock data for demonstration
const generateMockEquityCurve = (numPoints: number = 100) => {
  const data = [];
  let equity = 10000;
  
  for (let i = 0; i < numPoints; i++) {
    const change = (Math.random() - 0.5) * 200; // Random change
    equity += change;
    data.push({
      timestamp: new Date(Date.now() - (numPoints - i) * 60000).toISOString(),
      equity: Math.max(equity, 5000), // Minimum equity
      drawdown: Math.max(0, 10000 - equity),
      returns: i === 0 ? 0 : ((equity - 10000) / 10000) * 100
    });
  }
  
  return data;
};

const MonitoringDashboard: React.FC<MonitoringDashboardProps> = ({
  execution,
  configuration
}) => {
  const logsEndRef = useRef<HTMLDivElement>(null);

  // State management
  const [logFilter, setLogFilter] = useState<string>('');
  const [logLevelFilter, setLogLevelFilter] = useState<string>('all');
  const [logCategoryFilter, setLogCategoryFilter] = useState<string>('all');
  const [autoScrollLogs, setAutoScrollLogs] = useState<boolean>(true);
  const [showAdvancedMetrics, setShowAdvancedMetrics] = useState<boolean>(false);
  const [selectedTimeframe, setSelectedTimeframe] = useState<string>('1h');

  // Mock data for demonstration
  const [equityCurveData, setEquityCurveData] = useState(generateMockEquityCurve);
  const [recentTrades, setRecentTrades] = useState<PlaygroundTrade[]>([
    {
      id: '1',
      executionId: execution?.id || 'demo',
      symbol: 'BTC/USDT',
      side: 'buy',
      quantity: 0.1,
      price: 45000,
      timestamp: new Date().toISOString(),
      pnl: 150,
      commission: 4.5,
      reason: 'Trend following signal',
      confidence: 0.85
    },
    {
      id: '2',
      executionId: execution?.id || 'demo',
      symbol: 'ETH/USDT',
      side: 'sell',
      quantity: 2.5,
      price: 3200,
      timestamp: new Date(Date.now() - 300000).toISOString(),
      pnl: -50,
      commission: 8,
      reason: 'Stop loss triggered',
      confidence: 0.92
    }
  ]);

  // Auto-scroll logs to bottom
  useEffect(() => {
    if (autoScrollLogs && logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [execution?.logs, autoScrollLogs]);

  // Filter logs based on search criteria
  const filteredLogs = execution?.logs?.filter(log => {
    const matchesSearch = logFilter === '' || 
      log.message.toLowerCase().includes(logFilter.toLowerCase()) ||
      log.category.toLowerCase().includes(logFilter.toLowerCase());
    
    const matchesLevel = logLevelFilter === 'all' || log.level === logLevelFilter;
    const matchesCategory = logCategoryFilter === 'all' || log.category === logCategoryFilter;
    
    return matchesSearch && matchesLevel && matchesCategory;
  }) || [];

  // Get log icon based on level
  const getLogIcon = (level: string) => {
    switch (level) {
      case 'error': return <AlertTriangle className="h-4 w-4 text-red-500" />;
      case 'warning': return <AlertCircle className="h-4 w-4 text-yellow-500" />;
      case 'info': return <Info className="h-4 w-4 text-blue-500" />;
      default: return <Info className="h-4 w-4" />;
    }
  };

  // Get log color based on level
  const getLogColor = (level: string) => {
    switch (level) {
      case 'error': return 'text-red-500';
      case 'warning': return 'text-yellow-500';
      case 'info': return 'text-blue-500';
      default: return 'text-foreground';
    }
  };

  // Format currency values
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value);
  };

  // Calculate performance metrics
  const currentEquity = equityCurveData[equityCurveData.length - 1]?.equity || 10000;
  const totalReturn = ((currentEquity - 10000) / 10000) * 100;
  const maxDrawdown = Math.max(...equityCurveData.map(d => d.drawdown));
  const maxDrawdownPercent = (maxDrawdown / 10000) * 100;

  if (!execution) {
    return (
      <Card className="p-6 text-center min-h-96">
        <CardContent>
          <h3 className="text-lg font-semibold text-muted-foreground mb-2">
            No Active Execution
          </h3>
          <p className="text-muted-foreground">
            Start an execution to monitor performance and view real-time data.
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="flex flex-col gap-6 h-full">
      {/* Performance Overview */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardContent className="text-center p-4">
            <p className="text-sm text-muted-foreground mb-1">
              Current Equity
            </p>
            <h3 className="text-2xl font-bold">
              {formatCurrency(currentEquity)}
            </h3>
            <div className={`flex items-center justify-center gap-1 text-sm ${
              totalReturn >= 0 ? 'text-green-600' : 'text-red-600'
            }`}>
              {totalReturn >= 0 ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />}
              {totalReturn >= 0 ? '+' : ''}{totalReturn.toFixed(2)}%
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="text-center p-4">
            <p className="text-sm text-muted-foreground mb-1">
              Total Trades
            </p>
            <h3 className="text-2xl font-bold">
              {execution.metrics?.totalTrades || 0}
            </h3>
            <p className="text-sm text-muted-foreground">
              Win Rate: {execution.metrics?.winRate?.toFixed(1) || 0}%
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="text-center p-4">
            <p className="text-sm text-muted-foreground mb-1">
              Max Drawdown
            </p>
            <h3 className="text-2xl font-bold text-red-600">
              {maxDrawdownPercent.toFixed(2)}%
            </h3>
            <p className="text-sm text-muted-foreground">
              {formatCurrency(maxDrawdown)}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="text-center p-4">
            <p className="text-sm text-muted-foreground mb-1">
              Sharpe Ratio
            </p>
            <h3 className="text-2xl font-bold">
              {execution.metrics?.sharpeRatio?.toFixed(2) || 'N/A'}
            </h3>
            <p className="text-sm text-muted-foreground">
              Risk-adjusted return
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Execution Progress */}
      {execution.status === 'running' && (
        <Card>
          <CardContent className="p-4">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold">Execution Progress</h3>
              <Badge variant="default">
                {Math.round(execution.progress)}%
              </Badge>
            </div>
            <Progress 
              value={execution.progress} 
              className="h-2 mb-2"
            />
            <div className="flex justify-between text-sm text-muted-foreground">
              <span>
                Status: {execution.status.charAt(0).toUpperCase() + execution.status.slice(1)}
              </span>
              <span>
                Duration: {execution.duration ? `${Math.round(execution.duration / 1000)}s` : 'N/A'}
              </span>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 flex-1">
        {/* Equity Curve Chart */}
        <div className="lg:col-span-2">
          <Card className="h-96">
            <CardHeader>
              <div className="flex justify-between items-center">
                <CardTitle className="flex items-center gap-2">
                  <LineChart className="h-5 w-5" />
                  Equity Curve
                </CardTitle>
                <Select value={selectedTimeframe} onValueChange={setSelectedTimeframe}>
                  <SelectTrigger className="w-32">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="1h">1 Hour</SelectItem>
                    <SelectItem value="4h">4 Hours</SelectItem>
                    <SelectItem value="1d">1 Day</SelectItem>
                    <SelectItem value="1w">1 Week</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={250}>
                <AreaChart data={equityCurveData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="timestamp" 
                    tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                  />
                  <YAxis tickFormatter={(value) => `$${value.toLocaleString()}`} />
                  <ChartTooltip 
                    formatter={(value, name) => [
                      name === 'equity' ? formatCurrency(value as number) : `${(value as number).toFixed(2)}%`,
                      name === 'equity' ? 'Equity' : 'Returns'
                    ]}
                    labelFormatter={(label) => new Date(label).toLocaleString()}
                  />
                  <Area
                    type="monotone"
                    dataKey="equity"
                    stroke="hsl(var(--primary))"
                    fill="hsl(var(--primary) / 0.3)"
                    strokeWidth={2}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </div>

        {/* Recent Trades */}
        <div>
          <Card className="h-96">
            <CardHeader>
              <CardTitle>Recent Trades</CardTitle>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-64">
                <div className="space-y-4">
                  {recentTrades.map((trade, index) => (
                    <div key={trade.id}>
                      <div className="flex items-start space-x-3">
                        <div className="flex-shrink-0">
                          {trade.side === 'buy' ? (
                            <TrendingUp className="h-4 w-4 text-green-600" />
                          ) : (
                            <TrendingDown className="h-4 w-4 text-red-600" />
                          )}
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex justify-between items-start">
                            <p className="text-sm font-medium text-foreground">
                              {trade.symbol}
                            </p>
                            <p className={`text-sm font-medium ${
                              trade.pnl && trade.pnl >= 0 ? 'text-green-600' : 'text-red-600'
                            }`}>
                              {trade.pnl ? `${trade.pnl >= 0 ? '+' : ''}$${trade.pnl.toFixed(2)}` : 'Pending'}
                            </p>
                          </div>
                          <p className="text-xs text-muted-foreground">
                            {trade.side.toUpperCase()} {trade.quantity} @ ${trade.price.toLocaleString()}
                          </p>
                          <p className="text-xs text-muted-foreground">
                            {new Date(trade.timestamp).toLocaleTimeString()}
                          </p>
                        </div>
                      </div>
                      {index < recentTrades.length - 1 && <Separator className="mt-3" />}
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </div>

      </div>

      {/* Execution Logs */}
      <div className="col-span-full">
        <Card>
          <CardHeader>
            <div className="flex justify-between items-center">
              <CardTitle>Execution Logs</CardTitle>
              <div className="flex items-center gap-4">
                <div className="flex items-center space-x-2">
                  <Switch
                    id="auto-scroll"
                    checked={autoScrollLogs}
                    onCheckedChange={setAutoScrollLogs}
                  />
                  <Label htmlFor="auto-scroll" className="text-sm">Auto-scroll</Label>
                </div>
                <Button size="sm" variant="outline">
                  <Download className="h-4 w-4 mr-2" />
                  Download
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent>

            {/* Log Filters */}
            <div className="grid grid-cols-1 md:grid-cols-6 gap-4 mb-4">
              <div className="md:col-span-3">
                <div className="relative">
                  <Search className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                  <Input
                    placeholder="Search logs..."
                    value={logFilter}
                    onChange={(e) => setLogFilter(e.target.value)}
                    className="pl-10"
                  />
                </div>
              </div>
              <div className="md:col-span-1.5">
                <Select value={logLevelFilter} onValueChange={setLogLevelFilter}>
                  <SelectTrigger>
                    <SelectValue placeholder="Level" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Levels</SelectItem>
                    <SelectItem value="debug">Debug</SelectItem>
                    <SelectItem value="info">Info</SelectItem>
                    <SelectItem value="warning">Warning</SelectItem>
                    <SelectItem value="error">Error</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="md:col-span-1.5">
                <Select value={logCategoryFilter} onValueChange={setLogCategoryFilter}>
                  <SelectTrigger>
                    <SelectValue placeholder="Category" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Categories</SelectItem>
                    <SelectItem value="strategy">Strategy</SelectItem>
                    <SelectItem value="risk">Risk</SelectItem>
                    <SelectItem value="execution">Execution</SelectItem>
                    <SelectItem value="data">Data</SelectItem>
                    <SelectItem value="system">System</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            {/* Log List */}
            <div className="border rounded-lg bg-muted/10 max-h-80">
              <ScrollArea className="h-80">
                <div className="p-4">
                  {filteredLogs.length === 0 ? (
                    <div className="text-center py-8">
                      <p className="font-medium">No logs available</p>
                      <p className="text-sm text-muted-foreground">
                        Logs will appear here as the execution progresses
                      </p>
                    </div>
                  ) : (
                    <div className="space-y-2">
                      {filteredLogs.map((log) => (
                        <div key={log.id} className="flex items-start gap-3 py-1">
                          <div className="flex-shrink-0 mt-1">
                            {getLogIcon(log.level)}
                          </div>
                          <div className="flex-1">
                            <pre className={`text-sm font-mono whitespace-pre-wrap ${getLogColor(log.level)}`}>
                              [{new Date(log.timestamp).toLocaleTimeString()}] [{log.category.toUpperCase()}] {log.message}
                            </pre>
                          </div>
                        </div>
                      ))}
                      <div ref={logsEndRef} />
                    </div>
                  )}
                </div>
              </ScrollArea>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default MonitoringDashboard;