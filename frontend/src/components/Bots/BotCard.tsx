/**
 * BotCard component - Trading bot display and control interface
 * Shows bot status, performance, and management controls
 */

import React, { useState, useMemo } from 'react';
import {
  Play,
  Pause,
  Square,
  Edit,
  Trash2,
  BarChart3,
  MoreVertical,
} from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { cn } from '@/lib/utils';
import { motion } from 'framer-motion';

export interface Bot {
  id: string;
  name: string;
  strategy: string;
  status: 'running' | 'paused' | 'stopped' | 'error';
  exchange: string;
  symbol: string;
  balance: number;
  pnl: number;
  pnlPercent: number;
  trades: number;
  winRate: number;
  created: number;
  lastActivity: number;
  riskLevel: 'low' | 'medium' | 'high';
  leverage: number;
  maxDrawdown: number;
  sharpeRatio: number;
  description?: string;
  version?: string;
  isActive: boolean;
}

interface BotCardProps {
  bot: Bot;
  onStart?: (botId: string) => void;
  onPause?: (botId: string) => void;
  onStop?: (botId: string) => void;
  onEdit?: (botId: string) => void;
  onDelete?: (botId: string) => void;
  onToggleActive?: (botId: string, active: boolean) => void;
  onViewDetails?: (botId: string) => void;
  compact?: boolean;
  showActions?: boolean;
}

const formatCurrency = (value: number): string => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
};

const formatPercentage = (value: number): string => {
  const prefix = value >= 0 ? '+' : '';
  return `${prefix}${value.toFixed(2)}%`;
};

const getStatusColor = (status: string) => {
  switch (status) {
    case 'running':
      return 'bg-green-600';
    case 'paused':
      return 'bg-yellow-600';
    case 'stopped':
      return 'bg-gray-600';
    case 'error':
      return 'bg-red-600';
    default:
      return 'bg-blue-600';
  }
};

const getRiskColor = (risk: string) => {
  switch (risk) {
    case 'low':
      return 'text-green-600 border-green-600';
    case 'medium':
      return 'text-yellow-600 border-yellow-600';
    case 'high':
      return 'text-red-600 border-red-600';
    default:
      return 'text-blue-600 border-blue-600';
  }
};

export const BotCard: React.FC<BotCardProps> = ({
  bot,
  onStart,
  onPause,
  onStop,
  onEdit,
  onDelete,
  onToggleActive,
  onViewDetails,
  compact = false,
  showActions = true,
}) => {
  const [menuAnchor, setMenuAnchor] = useState<null | HTMLElement>(null);
  const [deleteDialog, setDeleteDialog] = useState(false);
  const [editDialog, setEditDialog] = useState(false);
  const [editName, setEditName] = useState(bot.name);

  const {
    id,
    name,
    strategy,
    status,
    exchange,
    symbol,
    balance,
    pnl,
    pnlPercent,
    trades,
    winRate,
    lastActivity,
    riskLevel,
    leverage,
    sharpeRatio,
    version,
    isActive,
  } = bot;

  const pnlColor = pnl >= 0 ? 'text-green-500' : 'text-red-500';
  const statusColor = getStatusColor(status);
  const riskColor = getRiskColor(riskLevel);

  const isRunning = status === 'running';
  const isPaused = status === 'paused';
  const isStopped = status === 'stopped';
  const hasError = status === 'error';

  const timeSinceLastActivity = useMemo(() => {
    const diff = Date.now() - lastActivity;
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);

    if (days > 0) return `${days}d ago`;
    if (hours > 0) return `${hours}h ago`;
    if (minutes > 0) return `${minutes}m ago`;
    return 'Just now';
  }, [lastActivity]);

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setMenuAnchor(event.currentTarget);
  };

  const handleMenuClose = () => {
    setMenuAnchor(null);
  };

  const handleStart = () => {
    if (onStart) onStart(id);
  };

  const handlePause = () => {
    if (onPause) onPause(id);
  };

  const handleStop = () => {
    if (onStop) onStop(id);
  };

  const handleEdit = () => {
    setEditDialog(true);
    handleMenuClose();
  };

  const handleSaveEdit = () => {
    if (onEdit && editName.trim()) {
      onEdit(id);
      setEditDialog(false);
    }
  };

  const handleDelete = () => {
    if (onDelete) onDelete(id);
    setDeleteDialog(false);
    handleMenuClose();
  };

  const handleToggleActive = (checked: boolean) => {
    if (onToggleActive) onToggleActive(id, checked);
  };

  const handleViewDetails = () => {
    if (onViewDetails) onViewDetails(id);
  };

  return (
    <>
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.95 }}
        transition={{ duration: 0.2 }}
        whileHover={{ scale: 1.02 }}
      >
        <Card className={cn(
          "relative transition-shadow hover:shadow-lg border-l-4",
          hasError ? "border-red-500" : "border-gray-200",
          statusColor.replace('bg-', 'border-l-'),
          isActive ? "opacity-100" : "opacity-70"
        )}>
          <CardContent className={compact ? 'p-4' : 'p-6'}>
            {/* Header */}
            <div className="flex justify-between items-start mb-4">
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-2">
                  <h3 
                    className={cn(
                      "font-semibold cursor-pointer hover:text-primary",
                      compact ? "text-lg" : "text-xl"
                    )}
                    onClick={handleViewDetails}
                  >
                    {name}
                  </h3>
                  
                  <Badge className={cn(
                    "text-white font-semibold min-w-[60px] justify-center",
                    statusColor
                  )}>
                    {status.toUpperCase()}
                  </Badge>
                  
                  <Badge variant="outline" className={riskColor}>
                    {riskLevel.toUpperCase()}
                  </Badge>
                </div>

                <p className="text-sm text-muted-foreground mb-1">
                  {strategy} • {exchange} • {symbol}
                </p>

                {version && (
                  <p className="text-xs text-muted-foreground">
                    v{version}
                  </p>
                )}
              </div>

              {showActions && (
                <div className="flex items-center gap-2">
                  <Switch
                    checked={isActive}
                    onCheckedChange={handleToggleActive}
                  />
                  
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="ghost" size="icon">
                        <MoreVertical className="h-4 w-4" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem onClick={handleEdit}>
                        <Edit className="w-4 h-4 mr-2" />
                        Edit
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={() => setDeleteDialog(true)}>
                        <Trash2 className="w-4 h-4 mr-2" />
                        Delete
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={handleViewDetails}>
                        <BarChart3 className="w-4 h-4 mr-2" />
                        View Details
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
              )}
            </div>

            {/* Error Alert */}
            {hasError && (
              <Alert className="mb-4 text-xs">
                <AlertDescription>
                  Bot encountered an error. Check logs for details.
                </AlertDescription>
              </Alert>
            )}

            {/* Performance Metrics */}
            <div className={cn(
              "flex gap-4 mb-4",
              compact ? "flex-col" : "flex-row"
            )}>
              {/* Left Column */}
              <div className="flex-1 space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-muted-foreground">Balance</span>
                  <span className="text-sm font-mono">{formatCurrency(balance)}</span>
                </div>

                <div className="flex justify-between">
                  <span className="text-sm text-muted-foreground">P&L</span>
                  <div className="text-right">
                    <div className={cn(
                      "text-sm font-mono font-semibold",
                      pnlColor
                    )}>
                      {formatCurrency(pnl)}
                    </div>
                    <div className={cn(
                      "text-xs font-mono",
                      pnlColor
                    )}>
                      {formatPercentage(pnlPercent)}
                    </div>
                  </div>
                </div>

                <div className="flex justify-between">
                  <span className="text-sm text-muted-foreground">Trades</span>
                  <span className="text-sm font-mono">{trades.toLocaleString()}</span>
                </div>
              </div>

              {/* Right Column */}
              {!compact && (
                <div className="flex-1 space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Win Rate</span>
                    <span className="text-sm font-mono">{winRate.toFixed(1)}%</span>
                  </div>

                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Leverage</span>
                    <span className="text-sm font-mono">{leverage}x</span>
                  </div>

                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Sharpe Ratio</span>
                    <span className="text-sm font-mono">{sharpeRatio.toFixed(2)}</span>
                  </div>
                </div>
              )}
            </div>

            {/* Activity Indicator */}
            <div className="mb-4">
              <div className="flex justify-between items-center mb-2">
                <span className="text-xs text-muted-foreground">
                  Last Activity: {timeSinceLastActivity}
                </span>
                {isRunning && (
                  <div className="flex items-center gap-1">
                    <div className={cn(
                      "w-2 h-2 rounded-full animate-pulse",
                      statusColor
                    )} />
                    <span className="text-xs text-muted-foreground">Active</span>
                  </div>
                )}
              </div>

              {isRunning && (
                <Progress value={undefined} className="h-0.5" />
              )}
            </div>

            {/* Control Buttons */}
            {showActions && (
              <div className="flex gap-2 flex-wrap">
                {isStopped && (
                  <Button
                    size="sm"
                    className="bg-green-600 hover:bg-green-700"
                    onClick={handleStart}
                    disabled={!isActive}
                  >
                    <Play className="w-4 h-4 mr-1" />
                    Start
                  </Button>
                )}

                {isRunning && (
                  <>
                    <Button
                      size="sm"
                      variant="outline"
                      className="text-yellow-600 border-yellow-200 hover:bg-yellow-50"
                      onClick={handlePause}
                    >
                      <Pause className="w-4 h-4 mr-1" />
                      Pause
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      className="text-red-600 border-red-200 hover:bg-red-50"
                      onClick={handleStop}
                    >
                      <Square className="w-4 h-4 mr-1" />
                      Stop
                    </Button>
                  </>
                )}

                {isPaused && (
                  <>
                    <Button
                      size="sm"
                      className="bg-green-600 hover:bg-green-700"
                      onClick={handleStart}
                      disabled={!isActive}
                    >
                      <Play className="w-4 h-4 mr-1" />
                      Resume
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      className="text-red-600 border-red-200 hover:bg-red-50"
                      onClick={handleStop}
                    >
                      <Square className="w-4 h-4 mr-1" />
                      Stop
                    </Button>
                  </>
                )}

                <Button
                  size="sm"
                  variant="outline"
                  onClick={handleViewDetails}
                >
                  <BarChart3 className="w-4 h-4 mr-1" />
                  Details
                </Button>
              </div>
            )}
          </CardContent>
        </Card>
      </motion.div>

      {/* Edit Dialog */}
      <Dialog open={editDialog} onOpenChange={setEditDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit Bot</DialogTitle>
          </DialogHeader>
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="botName">Bot Name</Label>
              <Input
                id="botName"
                value={editName}
                onChange={(e) => setEditName(e.target.value)}
                placeholder="Enter bot name"
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setEditDialog(false)}>
              Cancel
            </Button>
            <Button onClick={handleSaveEdit}>
              Save
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteDialog} onOpenChange={setDeleteDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete Bot</DialogTitle>
          </DialogHeader>
          <p className="text-sm text-muted-foreground">
            Are you sure you want to delete the bot "{name}"? This action cannot be undone.
          </p>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteDialog(false)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={handleDelete}>
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
};

export default BotCard;