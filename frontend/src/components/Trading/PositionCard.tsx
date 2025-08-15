/**
 * PositionCard component - Trading position display with P&L and controls
 * Shows position details, unrealized P&L, and management actions
 */

import React, { useMemo } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  IconButton,
  Button,
  Tooltip,
  Divider,
  Alert,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Close,
  Edit,
  Warning,
  Security,
} from '@mui/icons-material';
import { motion } from 'framer-motion';
import { tradingTheme } from '@/theme';
import { colors } from '@/theme/colors';

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

  const pnlColor = tradingTheme.getPriceChangeColor(unrealizedPnL);
  const sideColor = side === 'long' ? colors.financial.profit : colors.financial.loss;

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
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.95 }}
      transition={{ duration: 0.2 }}
    >
      <Card
        sx={{
          position: 'relative',
          backgroundColor: 'background.paper',
          border: '1px solid',
          borderColor: unrealizedPnL >= 0 ? 'success.main' : 'error.main',
          borderWidth: '1px',
          borderLeftWidth: '4px',
          '&:hover': {
            boxShadow: 3,
          },
        }}
      >
        <CardContent sx={{ p: compact ? 2 : 3 }}>
          {/* Header */}
          <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={2}>
            <Box>
              <Box display="flex" alignItems="center" gap={1} mb={1}>
                <Typography variant="h6" sx={{ fontWeight: 600 }}>
                  {symbol}
                </Typography>
                <Chip
                  label={side.toUpperCase()}
                  size="small"
                  sx={{
                    backgroundColor: sideColor,
                    color: 'white',
                    fontWeight: 600,
                    minWidth: 50,
                  }}
                />
                <Chip
                  label={`${leverage}x`}
                  size="small"
                  variant="outlined"
                />
                {strategy && (
                  <Chip
                    label={strategy}
                    size="small"
                    variant="outlined"
                    sx={{ fontSize: '0.7rem' }}
                  />
                )}
              </Box>

              <Typography variant="caption" color="text.secondary">
                {exchange} • {new Date(timestamp).toLocaleString()}
              </Typography>
            </Box>

            {showActions && (
              <Box display="flex" gap={0.5}>
                {onEdit && (
                  <Tooltip title="Edit Position">
                    <IconButton size="small" onClick={handleEdit}>
                      <Edit fontSize="small" />
                    </IconButton>
                  </Tooltip>
                )}
                {onClose && status === 'open' && (
                  <Tooltip title="Close Position">
                    <IconButton 
                      size="small" 
                      onClick={handleClose}
                      sx={{ color: 'error.main' }}
                    >
                      <Close fontSize="small" />
                    </IconButton>
                  </Tooltip>
                )}
              </Box>
            )}
          </Box>

          {/* Risk Warning */}
          {isAtRisk && (
            <Alert 
              severity="warning" 
              size="small" 
              sx={{ mb: 2, fontSize: '0.75rem' }}
              icon={<Warning fontSize="small" />}
            >
              High risk: {riskMetrics.distanceToLiquidation.toFixed(1)}% to liquidation
            </Alert>
          )}

          {/* Position Details */}
          <Box display="flex" flexDirection={compact ? 'column' : 'row'} gap={3}>
            {/* Size and Prices */}
            <Box flex={1}>
              <Box display="flex" justifyContent="space-between" mb={1}>
                <Typography variant="body2" color="text.secondary">
                  Size
                </Typography>
                <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                  {formatNumber(size, 6)}
                </Typography>
              </Box>

              <Box display="flex" justifyContent="space-between" mb={1}>
                <Typography variant="body2" color="text.secondary">
                  Entry Price
                </Typography>
                <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                  {formatCurrency(entryPrice)}
                </Typography>
              </Box>

              <Box display="flex" justifyContent="space-between" mb={1}>
                <Typography variant="body2" color="text.secondary">
                  Current Price
                </Typography>
                <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                  {formatCurrency(currentPrice)}
                </Typography>
              </Box>

              {!compact && (
                <Box display="flex" justifyContent="space-between" mb={1}>
                  <Typography variant="body2" color="text.secondary">
                    Market Value
                  </Typography>
                  <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                    {formatCurrency(riskMetrics.marketValue)}
                  </Typography>
                </Box>
              )}
            </Box>

            {/* P&L and Risk */}
            <Box flex={1}>
              <Box display="flex" justifyContent="space-between" mb={1}>
                <Typography variant="body2" color="text.secondary">
                  Unrealized P&L
                </Typography>
                <Box textAlign="right">
                  <Typography 
                    variant="body2" 
                    sx={{ 
                      fontFamily: 'monospace',
                      color: pnlColor,
                      fontWeight: 600,
                    }}
                  >
                    {formatCurrency(unrealizedPnL)}
                  </Typography>
                  <Typography 
                    variant="caption" 
                    sx={{ 
                      color: pnlColor,
                      fontFamily: 'monospace',
                    }}
                  >
                    {formatPercentage(unrealizedPnLPercent)}
                  </Typography>
                </Box>
              </Box>

              {stopLoss && (
                <Box display="flex" justifyContent="space-between" mb={1}>
                  <Typography variant="body2" color="text.secondary">
                    Stop Loss
                  </Typography>
                  <Typography 
                    variant="body2" 
                    sx={{ 
                      fontFamily: 'monospace',
                      color: 'error.main',
                    }}
                  >
                    {formatCurrency(stopLoss)}
                  </Typography>
                </Box>
              )}

              {takeProfit && (
                <Box display="flex" justifyContent="space-between" mb={1}>
                  <Typography variant="body2" color="text.secondary">
                    Take Profit
                  </Typography>
                  <Typography 
                    variant="body2" 
                    sx={{ 
                      fontFamily: 'monospace',
                      color: 'success.main',
                    }}
                  >
                    {formatCurrency(takeProfit)}
                  </Typography>
                </Box>
              )}

              {riskMetrics.riskRewardRatio && (
                <Box display="flex" justifyContent="space-between" mb={1}>
                  <Typography variant="body2" color="text.secondary">
                    Risk/Reward
                  </Typography>
                  <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                    1:{riskMetrics.riskRewardRatio.toFixed(2)}
                  </Typography>
                </Box>
              )}
            </Box>
          </Box>

          {/* Risk Management Actions */}
          {!compact && showActions && (
            <>
              <Divider sx={{ my: 2 }} />
              <Box display="flex" gap={1}>
                {!stopLoss && onSetStopLoss && (
                  <Button
                    size="small"
                    variant="outlined"
                    color="error"
                    startIcon={<Security />}
                    onClick={() => onSetStopLoss(id, 0)}
                  >
                    Set Stop Loss
                  </Button>
                )}
                {!takeProfit && onSetTakeProfit && (
                  <Button
                    size="small"
                    variant="outlined"
                    color="success"
                    startIcon={<TrendingUp />}
                    onClick={() => onSetTakeProfit(id, 0)}
                  >
                    Set Take Profit
                  </Button>
                )}
              </Box>
            </>
          )}
        </CardContent>

        {/* Status indicator */}
        <Box
          sx={{
            position: 'absolute',
            top: 8,
            right: 8,
            width: 8,
            height: 8,
            borderRadius: '50%',
            backgroundColor: tradingTheme.getStatusColor(status),
          }}
        />
      </Card>
    </motion.div>
  );
};

export default PositionCard;