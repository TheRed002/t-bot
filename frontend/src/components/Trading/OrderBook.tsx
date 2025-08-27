/**
 * OrderBook component - Real-time market depth display
 * Shows buy/sell orders with price levels and volumes
 */

import React, { useMemo, useState, useCallback } from 'react';
import {
  RefreshCw,
  Settings,
  ZoomIn,
  ZoomOut,
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';
import { motion } from 'framer-motion';

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
  onPriceClick,
  onRefresh,
  loading = false,
  precision = 2,
  maxRows = 10,
  showSpread = true,
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
    _event: React.MouseEvent<HTMLElement>,
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
        className={cn(
          "relative",
          isClickable ? "cursor-pointer hover:bg-muted/50" : "cursor-default"
        )}
        onClick={() => isClickable && handlePriceClick(entry.price)}
      >
        {/* Volume Column */}
        <TableCell className={cn(
          "relative overflow-hidden font-mono",
          compact ? "py-1 text-xs" : "py-2 text-sm"
        )}>
          {/* Depth Background */}
          <div
            className={cn(
              "absolute top-0 right-0 bottom-0 z-0",
              type === 'bid' 
                ? 'bg-green-500/10' 
                : 'bg-red-500/10'
            )}
            style={{ width: `${depthPercentage}%` }}
          />
          <div className="relative z-10">
            {formatVolume(entry.volume)}
          </div>
        </TableCell>

        {/* Price Column */}
        <TableCell className={cn(
          "font-mono font-semibold",
          compact ? "py-1 text-xs" : "py-2 text-sm",
          type === 'bid' ? 'text-green-500' : 'text-red-500'
        )}>
          {formatPrice(entry.price, precision)}
        </TableCell>

        {/* Total Column */}
        <TableCell className={cn(
          "font-mono",
          compact ? "py-1 text-xs" : "py-2 text-sm"
        )}>
          {formatVolume(entry.total)}
        </TableCell>
      </TableRow>
    );
  };

  return (
    <TooltipProvider>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        <Card className="h-full flex flex-col">
          {/* Header */}
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 border-b">
            <CardTitle className="text-lg font-semibold">
              Order Book - {symbol}
            </CardTitle>

            <div className="flex items-center gap-1">
              {/* Display Mode Toggle */}
              <div className="flex border rounded-md">
                <Button
                  variant={displayMode === 'bids' ? 'default' : 'ghost'}
                  size="sm"
                  onClick={() => setDisplayMode('bids')}
                  className="rounded-r-none"
                >
                  Bids
                </Button>
                <Button
                  variant={displayMode === 'both' ? 'default' : 'ghost'}
                  size="sm"
                  onClick={() => setDisplayMode('both')}
                  className="rounded-none border-x-0"
                >
                  Both
                </Button>
                <Button
                  variant={displayMode === 'asks' ? 'default' : 'ghost'}
                  size="sm"
                  onClick={() => setDisplayMode('asks')}
                  className="rounded-l-none"
                >
                  Asks
                </Button>
              </div>

              {/* Zoom Controls */}
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => setZoomLevel(Math.min(zoomLevel + 0.5, 3))}
                    disabled={zoomLevel >= 3}
                  >
                    <ZoomIn className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Zoom In</TooltipContent>
              </Tooltip>

              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => setZoomLevel(Math.max(zoomLevel - 0.5, 0.5))}
                    disabled={zoomLevel <= 0.5}
                  >
                    <ZoomOut className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Zoom Out</TooltipContent>
              </Tooltip>

              {/* Refresh Button */}
              {onRefresh && (
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button variant="ghost" size="icon" onClick={onRefresh}>
                      <RefreshCw className="h-4 w-4" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Refresh</TooltipContent>
                </Tooltip>
              )}

              {/* Settings */}
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="ghost" size="icon">
                    <Settings className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Settings</TooltipContent>
              </Tooltip>
            </div>
          </CardHeader>

          {/* Loading Bar */}
          {loading && (
            <div className="px-4">
              <Progress value={undefined} className="h-1" />
            </div>
          )}

          {/* Order Book Content */}
          <CardContent className="flex-1 overflow-hidden p-0">
            <div className="h-full overflow-auto">
              <Table>
                <TableHeader className="sticky top-0 bg-background">
                  <TableRow>
                    <TableCell className="font-semibold text-xs">
                      Volume
                    </TableCell>
                    <TableCell className="font-semibold text-xs">
                      Price
                    </TableCell>
                    <TableCell className="font-semibold text-xs">
                      Total
                    </TableCell>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {/* Asks (Sell Orders) - Shown in reverse order */}
                  {(displayMode === 'both' || displayMode === 'asks') &&
                    visibleAsks.reverse().map((ask, index) =>
                      renderOrderRow(ask, 'ask', index)
                    )}

                  {/* Spread Indicator */}
                  {showSpread && displayMode === 'both' && (
                    <TableRow>
                      <TableCell colSpan={3} className="py-2 text-center">
                        <Badge variant="outline" className="font-mono font-semibold text-yellow-500">
                          Spread: {formatPrice(spread, precision)} ({spreadPercent.toFixed(2)}%)
                        </Badge>
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
            </div>
          </CardContent>

          {/* Footer with Last Update */}
          <div className="border-t p-2 text-center">
            <p className="text-xs text-muted-foreground">
              Last updated: {new Date(data.lastUpdate).toLocaleTimeString()}
            </p>
          </div>
        </Card>
      </motion.div>
    </TooltipProvider>
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