/**
 * OrderBook component - Real-time market depth display
 * Shows buy/sell orders with price levels and volumes
 */

import React, { useMemo, useState, useCallback } from 'react';
import {
  Paper,
  Box,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Tooltip,
  ToggleButton,
  ToggleButtonGroup,
  LinearProgress,
} from '@mui/material';
import {
  Refresh,
  Settings,
  ZoomIn,
  ZoomOut,
  Fullscreen,
} from '@mui/icons-material';
import { motion } from 'framer-motion';
import { colors } from '@/theme/colors';
import { tradingTheme } from '@/theme';

export interface OrderBookEntry {
  price: number;
  volume: number;
  total: number;
  orders?: number;
}

export interface OrderBookData {
  bids: OrderBookEntry[];
  asks: OrderBookEntry[];
  spread: number;
  spreadPercent: number;
  lastUpdate: number;
}

interface OrderBookProps {
  data: OrderBookData;
  symbol?: string;
  currentPrice?: number;
  onPriceClick?: (price: number) => void;
  onRefresh?: () => void;
  loading?: boolean;
  precision?: number;
  maxRows?: number;
  showSpread?: boolean;
  showVolume?: boolean;
  compact?: boolean;
}

const formatPrice = (price: number, precision: number = 2): string => {
  return price.toFixed(precision);
};

const formatVolume = (volume: number): string => {
  if (volume >= 1000000) return `${(volume / 1000000).toFixed(2)}M`;
  if (volume >= 1000) return `${(volume / 1000).toFixed(2)}K`;
  return volume.toFixed(4);
};

const calculateDepthPercentage = (volume: number, maxVolume: number): number => {
  return maxVolume > 0 ? (volume / maxVolume) * 100 : 0;
};

export const OrderBook: React.FC<OrderBookProps> = ({
  data,
  symbol = 'BTC/USD',
  currentPrice,
  onPriceClick,
  onRefresh,
  loading = false,
  precision = 2,
  maxRows = 10,
  showSpread = true,
  showVolume = true,
  compact = false,
}) => {
  const [displayMode, setDisplayMode] = useState<'both' | 'bids' | 'asks'>('both');
  const [zoomLevel, setZoomLevel] = useState(1);

  const { bids, asks, spread, spreadPercent } = data;

  // Calculate max volumes for depth visualization
  const maxBidVolume = useMemo(() => 
    Math.max(...bids.slice(0, maxRows).map(bid => bid.volume), 0), 
    [bids, maxRows]
  );

  const maxAskVolume = useMemo(() => 
    Math.max(...asks.slice(0, maxRows).map(ask => ask.volume), 0), 
    [asks, maxRows]
  );

  const maxVolume = Math.max(maxBidVolume, maxAskVolume);

  // Filter and limit orders based on zoom and maxRows
  const visibleBids = useMemo(() => 
    bids.slice(0, Math.floor(maxRows / zoomLevel)), 
    [bids, maxRows, zoomLevel]
  );

  const visibleAsks = useMemo(() => 
    asks.slice(0, Math.floor(maxRows / zoomLevel)), 
    [asks, maxRows, zoomLevel]
  );

  const handleDisplayModeChange = useCallback((
    event: React.MouseEvent<HTMLElement>,
    newMode: string | null,
  ) => {
    if (newMode) {
      setDisplayMode(newMode as 'both' | 'bids' | 'asks');
    }
  }, []);

  const handlePriceClick = useCallback((price: number) => {
    if (onPriceClick) {
      onPriceClick(price);
    }
  }, [onPriceClick]);

  const renderOrderRow = (
    entry: OrderBookEntry, 
    type: 'bid' | 'ask', 
    index: number
  ) => {
    const depthPercentage = calculateDepthPercentage(entry.volume, maxVolume);
    const isClickable = !!onPriceClick;

    return (
      <TableRow
        key={`${type}-${index}`}
        sx={{
          position: 'relative',
          cursor: isClickable ? 'pointer' : 'default',
          '&:hover': isClickable ? {
            backgroundColor: 'action.hover',
          } : {},
        }}
        onClick={() => isClickable && handlePriceClick(entry.price)}
      >
        {/* Volume Column */}
        <TableCell 
          sx={{ 
            position: 'relative',
            overflow: 'hidden',
            py: compact ? 0.5 : 1,
            fontSize: compact ? '0.75rem' : '0.875rem',
            fontFamily: 'monospace',
          }}
        >
          {/* Depth Background */}
          <Box
            sx={{
              position: 'absolute',
              top: 0,
              right: 0,
              bottom: 0,
              width: `${depthPercentage}%`,
              backgroundColor: type === 'bid' 
                ? 'rgba(76, 175, 80, 0.1)' 
                : 'rgba(244, 67, 54, 0.1)',
              zIndex: 0,
            }}
          />
          <Box sx={{ position: 'relative', zIndex: 1 }}>
            {formatVolume(entry.volume)}
          </Box>
        </TableCell>

        {/* Price Column */}
        <TableCell 
          sx={{ 
            py: compact ? 0.5 : 1,
            fontSize: compact ? '0.75rem' : '0.875rem',
            fontFamily: 'monospace',
            fontWeight: 600,
            color: type === 'bid' ? colors.financial.profit : colors.financial.loss,
          }}
        >
          {formatPrice(entry.price, precision)}
        </TableCell>

        {/* Total Column */}
        <TableCell 
          sx={{ 
            py: compact ? 0.5 : 1,
            fontSize: compact ? '0.75rem' : '0.875rem',
            fontFamily: 'monospace',
          }}
        >
          {formatVolume(entry.total)}
        </TableCell>
      </TableRow>
    );
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <Paper sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
        {/* Header */}
        <Box
          p={2}
          display="flex"
          justifyContent="space-between"
          alignItems="center"
          borderBottom="1px solid"
          borderColor="divider"
        >
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            Order Book - {symbol}
          </Typography>

          <Box display="flex" alignItems="center" gap={1}>
            {/* Display Mode Toggle */}
            <ToggleButtonGroup
              value={displayMode}
              exclusive
              onChange={handleDisplayModeChange}
              size="small"
            >
              <ToggleButton value="bids">Bids</ToggleButton>
              <ToggleButton value="both">Both</ToggleButton>
              <ToggleButton value="asks">Asks</ToggleButton>
            </ToggleButtonGroup>

            {/* Zoom Controls */}
            <Tooltip title="Zoom In">
              <IconButton
                size="small"
                onClick={() => setZoomLevel(Math.min(zoomLevel + 0.5, 3))}
                disabled={zoomLevel >= 3}
              >
                <ZoomIn />
              </IconButton>
            </Tooltip>

            <Tooltip title="Zoom Out">
              <IconButton
                size="small"
                onClick={() => setZoomLevel(Math.max(zoomLevel - 0.5, 0.5))}
                disabled={zoomLevel <= 0.5}
              >
                <ZoomOut />
              </IconButton>
            </Tooltip>

            {/* Refresh Button */}
            {onRefresh && (
              <Tooltip title="Refresh">
                <IconButton size="small" onClick={onRefresh}>
                  <Refresh />
                </IconButton>
              </Tooltip>
            )}

            {/* Settings */}
            <Tooltip title="Settings">
              <IconButton size="small">
                <Settings />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>

        {/* Loading Bar */}
        {loading && <LinearProgress />}

        {/* Order Book Content */}
        <Box flex={1} overflow="hidden">
          <TableContainer sx={{ height: '100%' }}>
            <Table stickyHeader size={compact ? 'small' : 'medium'}>
              <TableHead>
                <TableRow>
                  <TableCell sx={{ fontWeight: 600, fontSize: '0.75rem' }}>
                    Volume
                  </TableCell>
                  <TableCell sx={{ fontWeight: 600, fontSize: '0.75rem' }}>
                    Price
                  </TableCell>
                  <TableCell sx={{ fontWeight: 600, fontSize: '0.75rem' }}>
                    Total
                  </TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {/* Asks (Sell Orders) - Shown in reverse order */}
                {(displayMode === 'both' || displayMode === 'asks') &&
                  visibleAsks.reverse().map((ask, index) =>
                    renderOrderRow(ask, 'ask', index)
                  )}

                {/* Spread Indicator */}
                {showSpread && displayMode === 'both' && (
                  <TableRow>
                    <TableCell colSpan={3} sx={{ py: 1, textAlign: 'center' }}>
                      <Box
                        sx={{
                          backgroundColor: 'background.default',
                          border: '1px solid',
                          borderColor: 'divider',
                          borderRadius: 1,
                          py: 0.5,
                          px: 1,
                        }}
                      >
                        <Typography
                          variant="caption"
                          sx={{
                            fontFamily: 'monospace',
                            fontWeight: 600,
                            color: 'warning.main',
                          }}
                        >
                          Spread: {formatPrice(spread, precision)} ({spreadPercent.toFixed(2)}%)
                        </Typography>
                      </Box>
                    </TableCell>
                  </TableRow>
                )}

                {/* Bids (Buy Orders) */}
                {(displayMode === 'both' || displayMode === 'bids') &&
                  visibleBids.map((bid, index) =>
                    renderOrderRow(bid, 'bid', index)
                  )}
              </TableBody>
            </Table>
          </TableContainer>
        </Box>

        {/* Footer with Last Update */}
        <Box
          p={1}
          borderTop="1px solid"
          borderColor="divider"
          textAlign="center"
        >
          <Typography variant="caption" color="text.secondary">
            Last updated: {new Date(data.lastUpdate).toLocaleTimeString()}
          </Typography>
        </Box>
      </Paper>
    </motion.div>
  );
};

// Mock data generator for development
export const generateMockOrderBookData = (
  basePrice: number = 45000,
  depth: number = 20
): OrderBookData => {
  const bids: OrderBookEntry[] = [];
  const asks: OrderBookEntry[] = [];

  let bidTotal = 0;
  let askTotal = 0;

  // Generate bids (buy orders) - below current price
  for (let i = 0; i < depth; i++) {
    const price = basePrice - (i + 1) * (Math.random() * 10 + 1);
    const volume = Math.random() * 10 + 0.1;
    bidTotal += volume;
    
    bids.push({
      price,
      volume,
      total: bidTotal,
      orders: Math.floor(Math.random() * 10) + 1,
    });
  }

  // Generate asks (sell orders) - above current price
  for (let i = 0; i < depth; i++) {
    const price = basePrice + (i + 1) * (Math.random() * 10 + 1);
    const volume = Math.random() * 10 + 0.1;
    askTotal += volume;
    
    asks.push({
      price,
      volume,
      total: askTotal,
      orders: Math.floor(Math.random() * 10) + 1,
    });
  }

  // Sort orders
  bids.sort((a, b) => b.price - a.price); // Highest bid first
  asks.sort((a, b) => a.price - b.price); // Lowest ask first

  const spread = asks[0]?.price - bids[0]?.price || 0;
  const spreadPercent = (spread / basePrice) * 100;

  return {
    bids,
    asks,
    spread,
    spreadPercent,
    lastUpdate: Date.now(),
  };
};

export default OrderBook;