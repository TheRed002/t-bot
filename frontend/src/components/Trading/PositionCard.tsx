/**
 * PositionCard component - Trading position display with P&L and controls
 * Shows position details, unrealized P&L, and management actions
 */

import React, { useMemo } from 'react';
import {
  TrendingUp,
  X,
  Edit,
  AlertTriangle,
  Shield,
} from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { Separator } from '@/components/ui/separator';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { cn } from '@/lib/utils';
import { motion } from 'framer-motion';

export interface Position {
  id: string;
  symbol: string;
  side: 'long' | 'short';
  size: number;
  entryPrice: number;
  currentPrice: number;
  stopLoss?: number;
  takeProfit?: number;
  unrealizedPnL: number;
  unrealizedPnLPercent: number;
  margin: number;
  leverage: number;
  timestamp: number;
  strategy?: string;
  exchange: string;
  status: 'open' | 'closing' | 'closed';
}

interface PositionCardProps {
  position: Position;
  onClose?: (positionId: string) => void;
  onEdit?: (positionId: string) => void;
  onSetStopLoss?: (positionId: string, price: number) => void;
  onSetTakeProfit?: (positionId: string, price: number) => void;
  compact?: boolean;
  showActions?: boolean;
}

const formatCurrency = (value: number, precision: number = 2): string => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: precision,
    maximumFractionDigits: precision,
  }).format(value);
};

const formatNumber = (value: number, precision: number = 4): string => {
  return new Intl.NumberFormat('en-US', {
    minimumFractionDigits: precision,
    maximumFractionDigits: precision,
  }).format(value);
};

const formatPercentage = (value: number, precision: number = 2): string => {
  const prefix = value >= 0 ? '+' : '';
  return `${prefix}${value.toFixed(precision)}%`;
};

export const PositionCard: React.FC<PositionCardProps> = ({
  position,
  onClose,
  onEdit,
  onSetStopLoss,
  onSetTakeProfit,
  compact = false,
  showActions = true,
}) => {
  const {
    id,
    symbol,
    side,
    size,
    entryPrice,
    currentPrice,
    stopLoss,
    takeProfit,
    unrealizedPnL,
    unrealizedPnLPercent,
    margin,
    leverage,
    timestamp,
    strategy,
    exchange,
    status,
  } = position;

  const pnlColor = unrealizedPnL >= 0 ? 'text-green-500' : 'text-red-500';
  const sideColor = side === 'long' ? 'bg-green-600' : 'bg-red-600';

  // Calculate risk metrics
  const riskMetrics = useMemo(() => {
    const marketValue = size * currentPrice;
    const riskRewardRatio = stopLoss && takeProfit
      ? Math.abs(takeProfit - entryPrice) / Math.abs(entryPrice - stopLoss)
      : null;
    
    const liquidationPrice = side === 'long'
      ? entryPrice - (margin / size)
      : entryPrice + (margin / size);

    const distanceToLiquidation = Math.abs((currentPrice - liquidationPrice) / currentPrice) * 100;

    return {
      marketValue,
      riskRewardRatio,
      liquidationPrice,
      distanceToLiquidation,
    };
  }, [size, currentPrice, entryPrice, stopLoss, takeProfit, margin, side]);

  const isAtRisk = riskMetrics.distanceToLiquidation < 20; // Less than 20% to liquidation

  const handleClose = () => {
    if (onClose) onClose(id);
  };

  const handleEdit = () => {
    if (onEdit) onEdit(id);
  };

  return (
    <TooltipProvider>
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.95 }}
        transition={{ duration: 0.2 }}
      >
        <Card className={cn(
          "relative transition-shadow hover:shadow-lg",
          "border-l-4",
          unrealizedPnL >= 0 ? "border-l-green-500" : "border-l-red-500"
        )}>
          <CardContent className={compact ? 'p-4' : 'p-6'}>
            {/* Header */}
            <div className="flex justify-between items-start mb-4">
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <h3 className="text-lg font-semibold">{symbol}</h3>
                  <Badge className={cn(
                    "text-white font-semibold min-w-[50px] justify-center",
                    sideColor
                  )}>
                    {side.toUpperCase()}
                  </Badge>
                  <Badge variant="outline">
                    {leverage}x
                  </Badge>
                  {strategy && (
                    <Badge variant="outline" className="text-xs">
                      {strategy}
                    </Badge>
                  )}
                </div>

                <p className="text-xs text-muted-foreground">
                  {exchange} • {new Date(timestamp).toLocaleString()}
                </p>
              </div>

              {showActions && (
                <div className="flex gap-1">
                  {onEdit && (
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Button variant="ghost" size="icon" onClick={handleEdit}>
                          <Edit className="h-4 w-4" />
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent>Edit Position</TooltipContent>
                    </Tooltip>
                  )}
                  {onClose && status === 'open' && (
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Button variant="ghost" size="icon" onClick={handleClose} className="text-red-500 hover:text-red-600">
                          <X className="h-4 w-4" />
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent>Close Position</TooltipContent>
                    </Tooltip>
                  )}
                </div>
              )}
            </div>

            {/* Risk Warning */}
            {isAtRisk && (
              <Alert className="mb-4 text-xs">
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>
                  High risk: {riskMetrics.distanceToLiquidation.toFixed(1)}% to liquidation
                </AlertDescription>
              </Alert>
            )}

            {/* Position Details */}
            <div className={cn(
              "flex gap-6",
              compact ? "flex-col" : "flex-row"
            )}>
              {/* Size and Prices */}
              <div className="flex-1 space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-muted-foreground">Size</span>
                  <span className="text-sm font-mono">{formatNumber(size, 6)}</span>
                </div>

                <div className="flex justify-between">
                  <span className="text-sm text-muted-foreground">Entry Price</span>
                  <span className="text-sm font-mono">{formatCurrency(entryPrice)}</span>
                </div>

                <div className="flex justify-between">
                  <span className="text-sm text-muted-foreground">Current Price</span>
                  <span className="text-sm font-mono">{formatCurrency(currentPrice)}</span>
                </div>

                {!compact && (
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Market Value</span>
                    <span className="text-sm font-mono">{formatCurrency(riskMetrics.marketValue)}</span>
                  </div>
                )}
              </div>

              {/* P&L and Risk */}
              <div className="flex-1 space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-muted-foreground">Unrealized P&L</span>
                  <div className="text-right">
                    <div className={cn(
                      "text-sm font-mono font-semibold",
                      pnlColor
                    )}>
                      {formatCurrency(unrealizedPnL)}
                    </div>
                    <div className={cn(
                      "text-xs font-mono",
                      pnlColor
                    )}>
                      {formatPercentage(unrealizedPnLPercent)}
                    </div>
                  </div>
                </div>

                {stopLoss && (
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Stop Loss</span>
                    <span className="text-sm font-mono text-red-500">
                      {formatCurrency(stopLoss)}
                    </span>
                  </div>
                )}

                {takeProfit && (
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Take Profit</span>
                    <span className="text-sm font-mono text-green-500">
                      {formatCurrency(takeProfit)}
                    </span>
                  </div>
                )}

                {riskMetrics.riskRewardRatio && (
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Risk/Reward</span>
                    <span className="text-sm font-mono">
                      1:{riskMetrics.riskRewardRatio.toFixed(2)}
                    </span>
                  </div>
                )}
              </div>
            </div>

            {/* Risk Management Actions */}
            {!compact && showActions && (
              <>
                <Separator className="my-4" />
                <div className="flex gap-2">
                  {!stopLoss && onSetStopLoss && (
                    <Button
                      size="sm"
                      variant="outline"
                      className="text-red-600 border-red-200 hover:bg-red-50"
                      onClick={() => onSetStopLoss(id, 0)}
                    >
                      <Shield className="w-4 h-4 mr-1" />
                      Set Stop Loss
                    </Button>
                  )}
                  {!takeProfit && onSetTakeProfit && (
                    <Button
                      size="sm"
                      variant="outline"
                      className="text-green-600 border-green-200 hover:bg-green-50"
                      onClick={() => onSetTakeProfit(id, 0)}
                    >
                      <TrendingUp className="w-4 h-4 mr-1" />
                      Set Take Profit
                    </Button>
                  )}
                </div>
              </>
            )}
          </CardContent>

          {/* Status indicator */}
          <div className={cn(
            "absolute top-2 right-2 w-2 h-2 rounded-full",
            status === 'open' ? 'bg-green-400' :
            status === 'closing' ? 'bg-yellow-400' : 'bg-gray-400'
          )} />
        </Card>
      </motion.div>
    </TooltipProvider>
  );
};

export default PositionCard;