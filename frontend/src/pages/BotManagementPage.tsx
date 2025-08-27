/**
 * Bot Management page component
 * Comprehensive bot management interface with creation, monitoring, and control
 * Built with Shadcn/ui components for a modern trading interface
 * Fully integrated with Redux store and backend API
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { motion } from 'framer-motion';
import { useDispatch, useSelector } from 'react-redux';
import { AppDispatch } from '@/store';

// Shadcn/ui components
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { Separator } from '@/components/ui/separator';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { Switch } from '@/components/ui/switch';
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuLabel, DropdownMenuSeparator, DropdownMenuTrigger } from '@/components/ui/dropdown-menu';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';

// Lucide React icons
import {
  Plus,
  Search,
  RefreshCw,
  Play,
  Pause,
  Square,
  TrendingUp,
  TrendingDown,
  Bot,
  DollarSign,
  Target,
  Activity,
  Settings,
  MoreHorizontal,
  Filter,
  SortAsc,
  SortDesc,
  Eye,
  Edit,
  Trash2,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Clock,
  Zap,
  BarChart3,
  PieChart,
  LineChart,
  ArrowUpDown,
  Grid3X3,
  List,
  X,
} from 'lucide-react';

// Utility functions
import { cn, formatPrice, formatPercent, getPnLColor, formatDateTime, formatUptime, timeAgo } from '@/lib/utils';

// Import bot components
import BotCreationWizard from '@/components/Bots/BotCreationWizardShadcn';

// Redux imports
import {
  fetchBots,
  startBot,
  stopBot,
  pauseBot,
  resumeBot,
  deleteBot,
  updateFilters,
  clearFilters,
  clearError,
  setSelectedBot,
  selectBots,
  selectFilteredBots,
  selectBotLoading,
  selectBotError,
  selectBotFilters,
  selectBotStatusCounts,
} from '@/store/slices/botSlice';

// Types from backend integration
import { BotInstance, BotStatus, BotFilters } from '@/types';

// Constants for filter options
const STRATEGY_OPTIONS = [
  { value: 'all', label: 'All Strategies' },
  { value: 'momentum', label: 'Momentum' },
  { value: 'mean_reversion', label: 'Mean Reversion' },
  { value: 'arbitrage', label: 'Arbitrage' },
  { value: 'scalping', label: 'Scalping' },
  { value: 'market_making', label: 'Market Making' },
  { value: 'trend_following', label: 'Trend Following' },
];

const EXCHANGE_OPTIONS = [
  { value: 'all', label: 'All Exchanges' },
  { value: 'binance', label: 'Binance' },
  { value: 'coinbase', label: 'Coinbase' },
  { value: 'okx', label: 'OKX' },
  { value: 'kraken', label: 'Kraken' },
];

const STATUS_FILTER_OPTIONS = [
  { value: 'all', label: 'All Status' },
  { value: BotStatus.RUNNING, label: 'Running' },
  { value: BotStatus.PAUSED, label: 'Paused' },
  { value: BotStatus.STOPPED, label: 'Stopped' },
  { value: BotStatus.ERROR, label: 'Error' },
  { value: BotStatus.INITIALIZING, label: 'Initializing' },
  { value: BotStatus.READY, label: 'Ready' },
];

const SORT_OPTIONS = [
  { value: 'total_pnl', label: 'P&L' },
  { value: 'total_trades', label: 'Trades' },
  { value: 'win_rate', label: 'Win Rate' },
  { value: 'bot_name', label: 'Name' },
  { value: 'createdAt', label: 'Created' },
];

type SortField = 'total_pnl' | 'total_trades' | 'win_rate' | 'bot_name' | 'createdAt';
type SortOrder = 'asc' | 'desc';
type ViewMode = 'grid' | 'list';

// Helper functions
const getStatusColor = (status: BotStatus): string => {
  switch (status) {
    case BotStatus.RUNNING:
      return 'bg-green-500';
    case BotStatus.PAUSED:
      return 'bg-yellow-500';
    case BotStatus.STOPPED:
    case BotStatus.STOPPING:
      return 'bg-gray-500';
    case BotStatus.ERROR:
      return 'bg-red-500';
    case BotStatus.INITIALIZING:
      return 'bg-blue-500';
    case BotStatus.READY:
      return 'bg-cyan-500';
    case BotStatus.MAINTENANCE:
      return 'bg-orange-500';
    default:
      return 'bg-gray-500';
  }
};

const getStatusIcon = (status: BotStatus) => {
  switch (status) {
    case BotStatus.RUNNING:
      return <Play className="h-3 w-3" />;
    case BotStatus.PAUSED:
      return <Pause className="h-3 w-3" />;
    case BotStatus.STOPPED:
    case BotStatus.STOPPING:
      return <Square className="h-3 w-3" />;
    case BotStatus.ERROR:
      return <AlertTriangle className="h-3 w-3" />;
    case BotStatus.INITIALIZING:
      return <Clock className="h-3 w-3" />;
    case BotStatus.READY:
      return <CheckCircle className="h-3 w-3" />;
    case BotStatus.MAINTENANCE:
      return <Settings className="h-3 w-3" />;
    default:
      return <Square className="h-3 w-3" />;
  }
};

const getStatusVariant = (status: BotStatus): 'default' | 'secondary' | 'destructive' | 'outline' => {
  switch (status) {
    case BotStatus.RUNNING:
      return 'default';
    case BotStatus.PAUSED:
    case BotStatus.READY:
      return 'secondary';
    case BotStatus.ERROR:
      return 'destructive';
    default:
      return 'outline';
  }
};

const getRiskLevel = (config?: { risk_percentage?: number }): 'low' | 'medium' | 'high' => {
  if (!config?.risk_percentage) return 'low';
  
  if (config.risk_percentage > 5) return 'high';
  if (config.risk_percentage > 2) return 'medium';
  return 'low';
};

const getRiskBadgeVariant = (risk: 'low' | 'medium' | 'high'): 'default' | 'secondary' | 'destructive' => {
  switch (risk) {
    case 'low':
      return 'default';
    case 'medium':
      return 'secondary';
    case 'high':
      return 'destructive';
    default:
      return 'default';
  }
};

const BotManagementPage: React.FC = () => {
  const dispatch = useDispatch<AppDispatch>();
  
  // Redux selectors
  const bots = useSelector(selectFilteredBots);
  const allBots = useSelector(selectBots);
  const isLoading = useSelector(selectBotLoading);
  const error = useSelector(selectBotError);
  const filters = useSelector(selectBotFilters);
  const statusCounts = useSelector(selectBotStatusCounts);
  
  // Local state
  const [viewMode, setViewMode] = useState<ViewMode>('grid');
  const [sortBy, setSortBy] = useState<SortField>('total_pnl');
  const [sortOrder, setSortOrder] = useState<SortOrder>('desc');
  const [createBotDialog, setCreateBotDialog] = useState(false);
  const [bulkActions, setBulkActions] = useState(false);
  const [selectedBots, setSelectedBots] = useState<string[]>([]);
  const [selectedBot, setSelectedBot] = useState<BotInstance | null>(null);
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [detailsDialogOpen, setDetailsDialogOpen] = useState(false);
  const [refreshing, setRefreshing] = useState(false);

  // Calculate summary metrics from Redux data
  const summaryMetrics = useMemo(() => {
    if (!Array.isArray(allBots) || allBots.length === 0) {
      return {
        totalBots: 0,
        runningBots: 0,
        pausedBots: 0,
        stoppedBots: 0,
        errorBots: 0,
        totalPnL: 0,
        todayPnL: 0,
        totalAllocatedCapital: 0,
        avgWinRate: 0,
        totalTrades: 0,
      };
    }

    const totalPnL = allBots.reduce((sum, bot) => {
      return sum + (bot.metrics?.total_pnl || 0);
    }, 0);

    const todayPnL = allBots.reduce((sum, bot) => {
      // Calculate today's P&L from metrics timestamp if available
      if (bot.metrics?.timestamp) {
        const today = new Date();
        const metricsDate = new Date(bot.metrics.timestamp);
        const isToday = 
          today.getDate() === metricsDate.getDate() &&
          today.getMonth() === metricsDate.getMonth() &&
          today.getFullYear() === metricsDate.getFullYear();
        
        if (isToday) {
          return sum + (bot.metrics.total_pnl || 0);
        }
      }
      return sum;
    }, 0);

    const totalAllocatedCapital = allBots.reduce((sum, bot) => {
      return sum + (bot.config?.allocated_capital || 0);
    }, 0);

    const avgWinRate = allBots.reduce((sum, bot) => {
      return sum + (bot.metrics?.win_rate || 0);
    }, 0) / allBots.length;

    const totalTrades = allBots.reduce((sum, bot) => {
      return sum + (bot.metrics?.total_trades || 0);
    }, 0);

    return {
      totalBots: statusCounts.total,
      runningBots: statusCounts.running,
      pausedBots: statusCounts.paused,
      stoppedBots: statusCounts.stopped,
      errorBots: statusCounts.error,
      totalPnL,
      todayPnL,
      totalAllocatedCapital,
      avgWinRate,
      totalTrades,
    };
  }, [allBots, statusCounts]);

  // Sorted and filtered bots (local sorting on Redux filtered data)
  const sortedBots = useMemo(() => {
    if (!Array.isArray(bots) || bots.length === 0) {
      return [];
    }

    const sorted = [...bots].sort((a, b) => {
      let aValue: number | string;
      let bValue: number | string;

      switch (sortBy) {
        case 'total_pnl':
          aValue = a.metrics?.total_pnl || 0;
          bValue = b.metrics?.total_pnl || 0;
          break;
        case 'total_trades':
          aValue = a.metrics?.total_trades || 0;
          bValue = b.metrics?.total_trades || 0;
          break;
        case 'win_rate':
          aValue = a.metrics?.win_rate || 0;
          bValue = b.metrics?.win_rate || 0;
          break;
        case 'bot_name':
          aValue = a.bot_name.toLowerCase();
          bValue = b.bot_name.toLowerCase();
          break;
        case 'createdAt':
          aValue = new Date(a.createdAt).getTime();
          bValue = new Date(b.createdAt).getTime();
          break;
        default:
          return 0;
      }

      if (typeof aValue === 'string' && typeof bValue === 'string') {
        return sortOrder === 'asc' ? aValue.localeCompare(bValue) : bValue.localeCompare(aValue);
      }

      const numA = aValue as number;
      const numB = bValue as number;
      return sortOrder === 'asc' ? numA - numB : numB - numA;
    });

    return sorted;
  }, [bots, sortBy, sortOrder]);

  // Data fetching with Redux
  const handleRefresh = useCallback(async () => {
    setRefreshing(true);
    try {
      await dispatch(fetchBots({})).unwrap();
    } catch (error) {
      console.error('Error refreshing bots:', error);
      // Error is handled by Redux slice
    } finally {
      setRefreshing(false);
    }
  }, [dispatch]);

  // Initial data fetch
  useEffect(() => {
    dispatch(fetchBots({})).catch(console.error);
  }, [dispatch]);

  // Clear error when component unmounts
  useEffect(() => {
    return () => {
      if (error) {
        dispatch(clearError());
      }
    };
  }, [error, dispatch]);

  // Bot action handlers using Redux thunks
  const handleBotAction = useCallback(async (action: string, botId: string) => {
    try {
      switch (action) {
        case 'start':
        case BotStatus.RUNNING:
          await dispatch(startBot(botId)).unwrap();
          break;
        case 'pause':
        case BotStatus.PAUSED:
          await dispatch(pauseBot(botId)).unwrap();
          break;
        case 'resume':
          await dispatch(resumeBot(botId)).unwrap();
          break;
        case 'stop':
        case BotStatus.STOPPED:
          await dispatch(stopBot(botId)).unwrap();
          break;
        default:
          console.warn(`Unknown bot action: ${action}`);
      }
    } catch (error) {
      console.error(`Failed to ${action} bot ${botId}:`, error);
      // Error handled by Redux slice
    }
  }, [dispatch]);

  const handleBulkAction = useCallback(async (action: string) => {
    const promises = selectedBots.map(botId => handleBotAction(action, botId));
    await Promise.allSettled(promises);
    setSelectedBots([]);
  }, [selectedBots, handleBotAction]);

  const handleDeleteBot = useCallback(async (botId: string) => {
    if (window.confirm('Are you sure you want to delete this bot? This action cannot be undone.')) {
      try {
        await dispatch(deleteBot(botId)).unwrap();
      } catch (error) {
        console.error(`Failed to delete bot ${botId}:`, error);
        // Error handled by Redux slice
      }
    }
  }, [dispatch]);

  const handleCreateBot = useCallback(() => {
    setCreateBotDialog(true);
  }, []);

  // Filter handlers
  const handleSearchChange = useCallback((searchTerm: string) => {
    dispatch(updateFilters({ searchTerm }));
  }, [dispatch]);

  const handleStatusFilterChange = useCallback((status: string) => {
    const statusArray = status === 'all' ? undefined : [status as BotStatus];
    dispatch(updateFilters({ status: statusArray }));
  }, [dispatch]);

  const handleExchangeFilterChange = useCallback((exchange: string) => {
    const exchangeArray = exchange === 'all' ? undefined : [exchange];
    dispatch(updateFilters({ exchange: exchangeArray }));
  }, [dispatch]);

  const handleStrategyFilterChange = useCallback((strategy: string) => {
    const strategyArray = strategy === 'all' ? undefined : [strategy];
    dispatch(updateFilters({ strategy: strategyArray }));
  }, [dispatch]);

  // Performance summary cards with proper error handling
  const PerformanceSummary = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4 mb-6">
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Total Bots</CardTitle>
          <Bot className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{summaryMetrics.totalBots}</div>
          <p className="text-xs text-muted-foreground">
            {summaryMetrics.runningBots} active
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Today P&L</CardTitle>
          <TrendingUp className={cn("h-4 w-4", getPnLColor(summaryMetrics.todayPnL))} />
        </CardHeader>
        <CardContent>
          <div className={cn("text-2xl font-bold", getPnLColor(summaryMetrics.todayPnL))}>
            {formatPrice(summaryMetrics.todayPnL)}
          </div>
          <p className="text-xs text-muted-foreground">
            {summaryMetrics.totalAllocatedCapital > 0 
              ? formatPercent((summaryMetrics.todayPnL / summaryMetrics.totalAllocatedCapital) * 100)
              : '0.00%'
            }
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Total P&L</CardTitle>
          <DollarSign className={cn("h-4 w-4", getPnLColor(summaryMetrics.totalPnL))} />
        </CardHeader>
        <CardContent>
          <div className={cn("text-2xl font-bold", getPnLColor(summaryMetrics.totalPnL))}>
            {formatPrice(summaryMetrics.totalPnL)}
          </div>
          <p className="text-xs text-muted-foreground">
            {summaryMetrics.totalAllocatedCapital > 0 
              ? formatPercent((summaryMetrics.totalPnL / summaryMetrics.totalAllocatedCapital) * 100)
              : '0.00%'
            }
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Avg Win Rate</CardTitle>
          <Target className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">
            {summaryMetrics.avgWinRate.toFixed(1)}%
          </div>
          <p className="text-xs text-muted-foreground">
            Across all bots
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Total Trades</CardTitle>
          <Activity className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">
            {summaryMetrics.totalTrades.toLocaleString()}
          </div>
          <p className="text-xs text-muted-foreground">
            All time
          </p>
        </CardContent>
      </Card>
    </div>
  );

  // Bot Card Component with updated data structure
  const BotCard = ({ bot }: { bot: BotInstance }) => {
    const riskLevel = getRiskLevel(bot.config);
    const todayPnL = bot.metrics?.total_pnl || 0; // Simplified for now
    const uptime = bot.metrics?.uptime_seconds || 0;
    const symbol = bot.config?.symbols?.[0] || 'N/A';
    const winRate = bot.metrics?.win_rate || 0;
    
    return (
      <Card className="group hover:shadow-lg transition-all duration-200 border-l-4" style={{
        borderLeftColor: bot.status === BotStatus.RUNNING ? '#22c55e' : 
                        bot.status === BotStatus.PAUSED ? '#eab308' :
                        bot.status === BotStatus.ERROR ? '#ef4444' : '#6b7280'
      }}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Avatar className="h-8 w-8">
              <AvatarFallback className={cn(
                "text-xs font-semibold text-white",
                getStatusColor(bot.status)
              )}>
                {getStatusIcon(bot.status)}
              </AvatarFallback>
            </Avatar>
            <div>
              <CardTitle className="text-base font-semibold">{bot.bot_name}</CardTitle>
              <CardDescription className="text-sm">{symbol} on {bot.exchange}</CardDescription>
            </div>
          </div>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" className="h-8 w-8 p-0">
                <MoreHorizontal className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuLabel>Actions</DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={() => {
                setSelectedBot(bot);
                setDetailsDialogOpen(true);
              }}>
                <Eye className="mr-2 h-4 w-4" />
                View Details
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => {
                setSelectedBot(bot);
                setEditDialogOpen(true);
              }}>
                <Edit className="mr-2 h-4 w-4" />
                Edit Settings
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              {bot.status === BotStatus.RUNNING ? (
                <DropdownMenuItem onClick={() => handleBotAction('pause', bot.bot_id)}>
                  <Pause className="mr-2 h-4 w-4" />
                  Pause Bot
                </DropdownMenuItem>
              ) : bot.status === BotStatus.PAUSED ? (
                <DropdownMenuItem onClick={() => handleBotAction('resume', bot.bot_id)}>
                  <Play className="mr-2 h-4 w-4" />
                  Resume Bot
                </DropdownMenuItem>
              ) : (
                <DropdownMenuItem onClick={() => handleBotAction('start', bot.bot_id)}>
                  <Play className="mr-2 h-4 w-4" />
                  Start Bot
                </DropdownMenuItem>
              )}
              <DropdownMenuItem onClick={() => handleBotAction('stop', bot.bot_id)}>
                <Square className="mr-2 h-4 w-4" />
                Stop Bot
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem className="text-destructive" onClick={() => handleDeleteBot(bot.bot_id)}>
                <Trash2 className="mr-2 h-4 w-4" />
                Delete Bot
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
        <div className="flex items-center space-x-2 mt-2">
          <Badge variant={getRiskBadgeVariant(riskLevel)}>
            {riskLevel} risk
          </Badge>
          <Badge variant="outline">
            {bot.strategy_name}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <p className="text-muted-foreground">P&L Today</p>
            <p className={cn("font-semibold", getPnLColor(todayPnL))}>
              {formatPrice(todayPnL)}
            </p>
          </div>
          <div>
            <p className="text-muted-foreground">Total P&L</p>
            <p className={cn("font-semibold", getPnLColor(bot.metrics?.total_pnl || 0))}>
              {formatPrice(bot.metrics?.total_pnl || 0)}
            </p>
          </div>
          <div>
            <p className="text-muted-foreground">Trades</p>
            <p className="font-semibold">{(bot.metrics?.total_trades || 0).toLocaleString()}</p>
          </div>
          <div>
            <p className="text-muted-foreground">Win Rate</p>
            <p className="font-semibold">{winRate.toFixed(1)}%</p>
          </div>
          <div>
            <p className="text-muted-foreground">Uptime</p>
            <p className="font-semibold">{formatUptime(uptime)}</p>
          </div>
          <div>
            <p className="text-muted-foreground">Capital</p>
            <p className="font-semibold">{formatPrice(bot.config?.allocated_capital || 0)}</p>
          </div>
        </div>
        
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">Win Rate</span>
            <span>{winRate.toFixed(1)}%</span>
          </div>
          <Progress value={winRate} className="h-2" />
        </div>
      </CardContent>
      <CardFooter className="pt-0">
        <div className="flex w-full space-x-2">
          {bot.status === BotStatus.RUNNING ? (
            <Button size="sm" variant="outline" onClick={() => handleBotAction('pause', bot.bot_id)}>
              <Pause className="mr-1 h-3 w-3" />
              Pause
            </Button>
          ) : bot.status === BotStatus.PAUSED ? (
            <Button size="sm" onClick={() => handleBotAction('resume', bot.bot_id)}>
              <Play className="mr-1 h-3 w-3" />
              Resume
            </Button>
          ) : (
            <Button size="sm" onClick={() => handleBotAction('start', bot.bot_id)}>
              <Play className="mr-1 h-3 w-3" />
              Start
            </Button>
          )}
          <Button size="sm" variant="outline" onClick={() => handleBotAction('stop', bot.bot_id)}>
            <Square className="mr-1 h-3 w-3" />
            Stop
          </Button>
        </div>
      </CardFooter>
    </Card>
  );
  };

  return (
    <TooltipProvider>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="container mx-auto px-4 py-6 space-y-6"
      >
        {/* Header */}
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-6">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">Bot Management</h1>
            <p className="text-muted-foreground mt-1">
              Monitor and control your automated trading bots
            </p>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center space-x-2">
              <Switch
                id="bulk-actions"
                checked={bulkActions}
                onCheckedChange={setBulkActions}
              />
              <Label htmlFor="bulk-actions" className="text-sm">Bulk Actions</Label>
            </div>

            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="outline"
                  size="icon"
                  onClick={handleRefresh}
                  disabled={isLoading || refreshing}
                >
                  <RefreshCw className={cn("h-4 w-4", (isLoading || refreshing) && "animate-spin")} />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Refresh data</TooltipContent>
            </Tooltip>

            <Button onClick={handleCreateBot}>
              <Plus className="mr-2 h-4 w-4" />
              Create Bot
            </Button>
          </div>
        </div>

        {/* Performance Summary */}
        <PerformanceSummary />

        {/* Bulk Actions Bar */}
        {bulkActions && selectedBots.length > 0 && (
          <Alert className="mb-6">
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>Bulk Actions</AlertTitle>
            <AlertDescription>
              {selectedBots.length} bot(s) selected
            </AlertDescription>
            <div className="flex gap-2 mt-3">
              <Button
                size="sm"
                onClick={() => handleBulkAction('running')}
              >
                <Play className="mr-1 h-3 w-3" />
                Start All
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={() => handleBulkAction('paused')}
              >
                <Pause className="mr-1 h-3 w-3" />
                Pause All
              </Button>
              <Button
                size="sm"
                variant="destructive"
                onClick={() => handleBulkAction('stopped')}
              >
                <Square className="mr-1 h-3 w-3" />
                Stop All
              </Button>
            </div>
          </Alert>
        )}

        {/* Filters and Search */}
        <Card className="mb-6">
          <CardContent className="p-6">
            <div className="flex flex-col sm:flex-row gap-4 mb-4">
              <div className="relative flex-1 min-w-[200px]">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
                <Input
                  placeholder="Search bots..."
                  value={filters.searchTerm || ''}
                  onChange={(e) => handleSearchChange(e.target.value)}
                  className="pl-10"
                />
              </div>
              
              <div className="flex flex-wrap gap-3">
                <Select 
                  value={filters.strategy?.[0] || 'all'} 
                  onValueChange={handleStrategyFilterChange}
                >
                  <SelectTrigger className="w-[150px]">
                    <SelectValue placeholder="Strategy" />
                  </SelectTrigger>
                  <SelectContent>
                    {STRATEGY_OPTIONS.map((strategy) => (
                      <SelectItem key={strategy.value} value={strategy.value}>
                        {strategy.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>

                <Select 
                  value={filters.exchange?.[0] || 'all'} 
                  onValueChange={handleExchangeFilterChange}
                >
                  <SelectTrigger className="w-[150px]">
                    <SelectValue placeholder="Exchange" />
                  </SelectTrigger>
                  <SelectContent>
                    {EXCHANGE_OPTIONS.map((exchange) => (
                      <SelectItem key={exchange.value} value={exchange.value}>
                        {exchange.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>

                <Select 
                  value={filters.status?.[0] || 'all'} 
                  onValueChange={handleStatusFilterChange}
                >
                  <SelectTrigger className="w-[120px]">
                    <SelectValue placeholder="Status" />
                  </SelectTrigger>
                  <SelectContent>
                    {STATUS_FILTER_OPTIONS.map((status) => (
                      <SelectItem key={status.value} value={status.value}>
                        {status.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>

                <Select value={sortBy} onValueChange={(value: SortField) => setSortBy(value)}>
                  <SelectTrigger className="w-[120px]">
                    <SelectValue placeholder="Sort by" />
                  </SelectTrigger>
                  <SelectContent>
                    {SORT_OPTIONS.map((option) => (
                      <SelectItem key={option.value} value={option.value}>
                        {option.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>

                <Button
                  variant="outline"
                  size="icon"
                  onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
                >
                  {sortOrder === 'asc' ? <SortAsc className="h-4 w-4" /> : <SortDesc className="h-4 w-4" />}
                </Button>

                <div className="flex border rounded-lg">
                  <Button
                    variant={viewMode === 'grid' ? 'default' : 'ghost'}
                    size="sm"
                    onClick={() => setViewMode('grid')}
                    className="rounded-r-none"
                  >
                    <Grid3X3 className="h-4 w-4" />
                  </Button>
                  <Button
                    variant={viewMode === 'list' ? 'default' : 'ghost'}
                    size="sm"
                    onClick={() => setViewMode('list')}
                    className="rounded-l-none"
                  >
                    <List className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </div>

            <div className="flex justify-between items-center text-sm text-muted-foreground">
              <span>Showing {sortedBots.length} of {allBots.length} bots</span>
              <div className="flex items-center gap-4">
                <span className="flex items-center gap-1">
                  <div className="w-2 h-2 rounded-full bg-green-500"></div>
                  {summaryMetrics.runningBots} Running
                </span>
                <span className="flex items-center gap-1">
                  <div className="w-2 h-2 rounded-full bg-yellow-500"></div>
                  {summaryMetrics.pausedBots} Paused
                </span>
                <span className="flex items-center gap-1">
                  <div className="w-2 h-2 rounded-full bg-gray-500"></div>
                  {summaryMetrics.stoppedBots} Stopped
                </span>
                <span className="flex items-center gap-1">
                  <div className="w-2 h-2 rounded-full bg-red-500"></div>
                  {summaryMetrics.errorBots} Error
                </span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Bot Grid/List */}
        {isLoading ? (
          <div className="flex items-center justify-center min-h-[400px]">
            <div className="flex items-center space-x-2">
              <RefreshCw className="h-6 w-6 animate-spin" />
              <span>Loading bots...</span>
            </div>
          </div>
        ) : error ? (
          <Alert variant="destructive" className="mb-6">
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>
              {error}
              <Button 
                variant="outline" 
                size="sm" 
                className="ml-3"
                onClick={handleRefresh}
              >
                <RefreshCw className="h-4 w-4 mr-2" />
                Retry
              </Button>
            </AlertDescription>
          </Alert>
        ) : sortedBots.length > 0 ? (
          viewMode === 'grid' ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
              {sortedBots.map((bot) => (
                <BotCard key={bot.bot_id} bot={bot} />
              ))}
            </div>
          ) : (
            <Card>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Bot</TableHead>
                    <TableHead>Strategy</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Exchange/Symbol</TableHead>
                    <TableHead className="text-right">Today P&L</TableHead>
                    <TableHead className="text-right">Total P&L</TableHead>
                    <TableHead className="text-right">Trades</TableHead>
                    <TableHead className="text-right">Win Rate</TableHead>
                    <TableHead className="text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {sortedBots.map((bot) => {
                    const riskLevel = getRiskLevel(bot.config);
                    const todayPnL = bot.metrics?.total_pnl || 0;
                    const symbol = bot.config?.symbols?.[0] || 'N/A';
                    const winRate = bot.metrics?.win_rate || 0;
                    
                    return (
                    <TableRow key={bot.bot_id}>
                      <TableCell>
                        <div className="flex items-center space-x-2">
                          <Avatar className="h-8 w-8">
                            <AvatarFallback className={cn(
                              "text-xs font-semibold text-white",
                              getStatusColor(bot.status)
                            )}>
                              {bot.bot_name.substring(0, 2).toUpperCase()}
                            </AvatarFallback>
                          </Avatar>
                          <div>
                            <p className="font-medium">{bot.bot_name}</p>
                            <p className="text-sm text-muted-foreground">{timeAgo(bot.createdAt)}</p>
                          </div>
                        </div>
                      </TableCell>
                      <TableCell>
                        <Badge variant="outline">{bot.strategy_name}</Badge>
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center space-x-1">
                          <div className={cn("w-2 h-2 rounded-full", getStatusColor(bot.status))}></div>
                          <span className="capitalize text-sm">{bot.status}</span>
                        </div>
                      </TableCell>
                      <TableCell>
                        <div>
                          <p className="font-medium">{bot.exchange}</p>
                          <p className="text-sm text-muted-foreground">{symbol}</p>
                        </div>
                      </TableCell>
                      <TableCell className={cn("text-right font-medium", getPnLColor(todayPnL))}>
                        {formatPrice(todayPnL)}
                      </TableCell>
                      <TableCell className={cn("text-right font-medium", getPnLColor(bot.metrics?.total_pnl || 0))}>
                        <div>
                          <p>{formatPrice(bot.metrics?.total_pnl || 0)}</p>
                          <p className="text-sm">ROI: {bot.config?.allocated_capital ? formatPercent((bot.metrics?.total_pnl || 0) / bot.config.allocated_capital * 100) : 'N/A'}</p>
                        </div>
                      </TableCell>
                      <TableCell className="text-right font-medium">
                        {(bot.metrics?.total_trades || 0).toLocaleString()}
                      </TableCell>
                      <TableCell className="text-right">
                        <div className="flex items-center justify-end space-x-2">
                          <span className="font-medium">{winRate.toFixed(1)}%</span>
                          <div className="w-12">
                            <Progress value={winRate} className="h-2" />
                          </div>
                        </div>
                      </TableCell>
                      <TableCell className="text-right">
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button variant="ghost" className="h-8 w-8 p-0">
                              <MoreHorizontal className="h-4 w-4" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end">
                            <DropdownMenuItem onClick={() => {
                              setSelectedBot(bot);
                              setDetailsDialogOpen(true);
                            }}>
                              <Eye className="mr-2 h-4 w-4" />
                              View Details
                            </DropdownMenuItem>
                            <DropdownMenuItem onClick={() => {
                              setSelectedBot(bot);
                              setEditDialogOpen(true);
                            }}>
                              <Edit className="mr-2 h-4 w-4" />
                              Edit
                            </DropdownMenuItem>
                            <DropdownMenuSeparator />
                            {bot.status === BotStatus.RUNNING ? (
                              <DropdownMenuItem onClick={() => handleBotAction('pause', bot.bot_id)}>
                                <Pause className="mr-2 h-4 w-4" />
                                Pause
                              </DropdownMenuItem>
                            ) : bot.status === BotStatus.PAUSED ? (
                              <DropdownMenuItem onClick={() => handleBotAction('resume', bot.bot_id)}>
                                <Play className="mr-2 h-4 w-4" />
                                Resume
                              </DropdownMenuItem>
                            ) : (
                              <DropdownMenuItem onClick={() => handleBotAction('start', bot.bot_id)}>
                                <Play className="mr-2 h-4 w-4" />
                                Start
                              </DropdownMenuItem>
                            )}
                            <DropdownMenuItem onClick={() => handleBotAction('stop', bot.bot_id)}>
                              <Square className="mr-2 h-4 w-4" />
                              Stop
                            </DropdownMenuItem>
                            <DropdownMenuSeparator />
                            <DropdownMenuItem className="text-destructive" onClick={() => handleDeleteBot(bot.bot_id)}>
                              <Trash2 className="mr-2 h-4 w-4" />
                              Delete
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </TableCell>
                    </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </Card>
          )
        ) : (
          <Card className="flex flex-col items-center justify-center min-h-[400px] text-center">
            <Bot className="h-16 w-16 text-muted-foreground mb-4" />
            <h3 className="text-lg font-semibold mb-2">No Bots Found</h3>
            <p className="text-muted-foreground mb-6 max-w-md">
              {filters.searchTerm || filters.strategy?.length || filters.exchange?.length || filters.status?.length
                ? 'Try adjusting your filters or search criteria'
                : 'Create your first trading bot to get started'
              }
            </p>
            <Button onClick={() => setCreateBotDialog(true)}>
              <Plus className="mr-2 h-4 w-4" />
              Create Your First Bot
            </Button>
          </Card>
        )}

        {/* Bot Details Dialog */}
        <Dialog open={detailsDialogOpen} onOpenChange={setDetailsDialogOpen}>
          <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
            {selectedBot && (
              <>
                <DialogHeader>
                  <DialogTitle className="flex items-center space-x-2">
                    <Avatar className="h-8 w-8">
                      <AvatarFallback className={cn(
                        "text-xs font-semibold text-white",
                        getStatusColor(selectedBot?.status || BotStatus.STOPPED)
                      )}>
                        {getStatusIcon(selectedBot?.status || BotStatus.STOPPED)}
                      </AvatarFallback>
                    </Avatar>
                    <span>{selectedBot?.bot_name || 'Unknown Bot'}</span>
                    <Badge variant={getRiskBadgeVariant(getRiskLevel(selectedBot?.config))}>
                      {getRiskLevel(selectedBot?.config)} risk
                    </Badge>
                  </DialogTitle>
                  <DialogDescription>
                    {selectedBot?.config?.strategy_config?.description || `${selectedBot?.strategy_name || 'Unknown'} strategy on ${selectedBot?.exchange || 'Unknown'}`}
                  </DialogDescription>
                </DialogHeader>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mt-6">
                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm font-medium">Performance</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Today P&L:</span>
                        <span className={cn("font-medium", getPnLColor(selectedBot?.metrics?.total_pnl || 0))}>
                          {formatPrice(selectedBot?.metrics?.total_pnl || 0)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Total P&L:</span>
                        <span className={cn("font-medium", getPnLColor(selectedBot?.metrics?.total_pnl || 0))}>
                          {formatPrice(selectedBot?.metrics?.total_pnl || 0)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Win Rate:</span>
                        <span className="font-medium">{(selectedBot?.metrics?.win_rate || 0).toFixed(1)}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Sharpe Ratio:</span>
                        <span className="font-medium">{(selectedBot?.config?.strategy_config?.sharpe_ratio || 0).toFixed(2)}</span>
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm font-medium">Trading</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Total Trades:</span>
                        <span className="font-medium">{selectedBot?.metrics?.total_trades || 0}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Leverage:</span>
                        <span className="font-medium">{selectedBot?.config?.strategy_config?.leverage || 1}x</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Max Drawdown:</span>
                        <span className="font-medium text-red-500">{(selectedBot?.config?.strategy_config?.max_drawdown || 0).toFixed(1)}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Balance:</span>
                        <span className="font-medium">{formatPrice(selectedBot?.config?.allocated_capital || 0)}</span>
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm font-medium">System</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Status:</span>
                        <Badge variant={selectedBot?.status === 'running' ? 'default' : 'secondary'}>
                          {selectedBot?.status || 'unknown'}
                        </Badge>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Uptime:</span>
                        <span className="font-medium">{formatUptime(selectedBot?.metrics?.uptime_seconds || 0)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Version:</span>
                        <span className="font-medium">v{selectedBot?.config?.strategy_config?.version || '1.0.0'}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Created:</span>
                        <span className="font-medium">{formatDateTime(selectedBot?.createdAt || new Date().toISOString())}</span>
                      </div>
                    </CardContent>
                  </Card>
                </div>

                <div className="flex justify-end space-x-2 mt-6">
                  <Button
                    variant="outline"
                    onClick={() => {
                      setEditDialogOpen(true);
                      setDetailsDialogOpen(false);
                    }}
                  >
                    <Edit className="mr-2 h-4 w-4" />
                    Edit Settings
                  </Button>
                  {selectedBot?.status === BotStatus.RUNNING ? (
                    <Button
                      variant="outline"
                      onClick={() => {
                        if (selectedBot) handleBotAction('pause', selectedBot.bot_id);
                        setDetailsDialogOpen(false);
                      }}
                    >
                      <Pause className="mr-2 h-4 w-4" />
                      Pause Bot
                    </Button>
                  ) : selectedBot?.status === BotStatus.PAUSED ? (
                    <Button
                      onClick={() => {
                        if (selectedBot) handleBotAction('resume', selectedBot.bot_id);
                        setDetailsDialogOpen(false);
                      }}
                    >
                      <Play className="mr-2 h-4 w-4" />
                      Resume Bot
                    </Button>
                  ) : (
                    <Button
                      onClick={() => {
                        if (selectedBot) handleBotAction('start', selectedBot.bot_id);
                        setDetailsDialogOpen(false);
                      }}
                    >
                      <Play className="mr-2 h-4 w-4" />
                      Start Bot
                    </Button>
                  )}
                </div>
              </>
            )}
          </DialogContent>
        </Dialog>

        {/* Floating Action Button */}
        <Button
          className="fixed bottom-6 right-6 h-14 w-14 rounded-full shadow-lg z-50"
          size="icon"
          onClick={() => setCreateBotDialog(true)}
        >
          <Plus className="h-6 w-6" />
        </Button>

        {/* Bot Creation Wizard */}
        <BotCreationWizard
          open={createBotDialog}
          onClose={() => setCreateBotDialog(false)}
          onSuccess={(botId) => {
            // Refresh bot list after successful creation
            dispatch(fetchBots({})).catch(console.error);
            // Optionally show success message
            console.log(`Bot created successfully: ${botId}`);
          }}
        />
      </motion.div>
    </TooltipProvider>
  );
};

export default BotManagementPage;
