/**
 * PriceChart component - Professional trading chart with real-time updates
 * Displays candlestick charts, line charts, and technical indicators
 */

import React, { useMemo, useCallback } from 'react';
import {
  Maximize,
  Minimize,
  RefreshCw,
  Settings,
  TrendingUp,
} from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { cn } from '@/lib/utils';
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as ChartTooltip,
} from 'recharts';
import { motion } from 'framer-motion';

interface PriceData {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  price?: number;
}

interface PriceChartProps {
  data: PriceData[];
  loading?: boolean;
  error?: string;
  symbol?: string;
  timeframe?: string;
  onTimeframeChange?: (timeframe: string) => void;
  onRefresh?: () => void;
  onSettingsClick?: () => void;
  height?: number;
  showVolume?: boolean;
  showIndicators?: boolean;
  fullscreen?: boolean;
  onFullscreenToggle?: () => void;
  chartType?: 'candlestick' | 'line' | 'area';
  onChartTypeChange?: (type: 'candlestick' | 'line' | 'area') => void;
}

const timeframes = [
  { value: '1m', label: '1M' },
  { value: '5m', label: '5M' },
  { value: '15m', label: '15M' },
  { value: '1h', label: '1H' },
  { value: '4h', label: '4H' },
  { value: '1d', label: '1D' },
  { value: '1w', label: '1W' },
];

const chartTypes = [
  { value: 'line', label: 'Line' },
  { value: 'area', label: 'Area' },
  { value: 'candlestick', label: 'Candles' },
];

const formatPrice = (value: number, decimals: number = 2): string => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value);
};

const formatVolume = (value: number): string => {
  if (value >= 1e9) return `${(value / 1e9).toFixed(1)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
  return value.toFixed(0);
};

const formatTimestamp = (timestamp: number): string => {
  return new Date(timestamp).toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
  });
};

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload || !payload.length) return null;

  const data = payload[0]?.payload;
  if (!data) return null;

  return (
    <Card className="p-3 bg-background border border-border rounded-lg shadow-lg">
      <p className="text-xs text-muted-foreground mb-2">
        {new Date(label).toLocaleString()}
      </p>
      
      {data.open !== undefined && (
        <div className="space-y-1">
          <p className="text-sm">O: {formatPrice(data.open)}</p>
          <p className="text-sm">H: {formatPrice(data.high)}</p>
          <p className="text-sm">L: {formatPrice(data.low)}</p>
          <p className="text-sm">C: {formatPrice(data.close)}</p>
          <p className="text-sm">V: {formatVolume(data.volume)}</p>
        </div>
      )}
      
      {data.price !== undefined && (
        <p className="text-sm">
          Price: {formatPrice(data.price)}
        </p>
      )}
    </Card>
  );
};

export const PriceChart: React.FC<PriceChartProps> = ({
  data,
  loading = false,
  error,
  symbol = 'BTC/USD',
  timeframe = '1h',
  onTimeframeChange,
  onRefresh,
  onSettingsClick,
  height = 400,
  fullscreen = false,
  onFullscreenToggle,
  chartType = 'line',
  onChartTypeChange,
}) => {
  // Calculate price change
  const priceChange = useMemo(() => {
    if (data.length < 2) return null;
    const currentPrice = data[data.length - 1]?.close || data[data.length - 1]?.price || 0;
    const previousPrice = data[data.length - 2]?.close || data[data.length - 2]?.price || 0;
    const change = currentPrice - previousPrice;
    const changePercent = (change / previousPrice) * 100;
    return { change, changePercent, currentPrice };
  }, [data]);

  // Prepare chart data
  const chartData = useMemo(() => {
    return data.map(item => ({
      ...item,
      time: formatTimestamp(item.timestamp),
      displayPrice: item.close || item.price || 0,
    }));
  }, [data]);

  const handleTimeframeChange = useCallback((
    _event: React.MouseEvent<HTMLElement>,
    newTimeframe: string | null,
  ) => {
    if (newTimeframe && onTimeframeChange) {
      onTimeframeChange(newTimeframe);
    }
  }, [onTimeframeChange]);

  const handleChartTypeChange = useCallback((
    _event: React.MouseEvent<HTMLElement>,
    newType: string | null,
  ) => {
    if (newType && onChartTypeChange) {
      onChartTypeChange(newType as 'candlestick' | 'line' | 'area');
    }
  }, [onChartTypeChange]);

  const renderChart = () => {
    if (loading) {
      return (
        <div
          className="flex items-center justify-center"
          style={{ height: height - 100 }}
        >
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
        </div>
      );
    }

    if (error) {
      return (
        <div
          className="flex items-center justify-center"
          style={{ height: height - 100 }}
        >
          <Alert className="max-w-sm">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        </div>
      );
    }

    if (!chartData.length) {
      return (
        <div
          className="flex items-center justify-center"
          style={{ height: height - 100 }}
        >
          <p className="text-muted-foreground">
            No data available
          </p>
        </div>
      );
    }

    return (
      <ResponsiveContainer width="100%" height={height - 100}>
        <ComposedChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
          <XAxis 
            dataKey="time" 
            axisLine={false}
            tickLine={false}
            tick={{ fontSize: 12, fill: 'hsl(var(--muted-foreground))' }}
          />
          <YAxis 
            domain={['dataMin - 5', 'dataMax + 5']}
            axisLine={false}
            tickLine={false}
            tick={{ fontSize: 12, fill: 'hsl(var(--muted-foreground))' }}
            tickFormatter={(value) => formatPrice(value, 0)}
          />
          <ChartTooltip content={<CustomTooltip />} />
          
          {chartType === 'line' && (
            <Line
              type="monotone"
              dataKey="displayPrice"
              stroke="hsl(var(--primary))"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4, fill: 'hsl(var(--primary))' }}
            />
          )}
          
          {chartType === 'area' && (
            <Area
              type="monotone"
              dataKey="displayPrice"
              stroke="hsl(var(--primary))"
              strokeWidth={2}
              fill={`url(#areaGradient)`}
              fillOpacity={0.3}
            />
          )}

          <defs>
            <linearGradient id="areaGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.8} />
              <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0.1} />
            </linearGradient>
          </defs>
        </ComposedChart>
      </ResponsiveContainer>
    );
  };

  return (
    <TooltipProvider>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        <Card className={cn(
          "flex flex-col bg-background",
          fullscreen ? "fixed inset-0 z-50" : "relative"
        )} style={{ height: fullscreen ? '100vh' : height }}>
          {/* Header */}
          <div className="flex justify-between items-center p-4 border-b border-border">
            <div className="flex items-center gap-4">
              <div>
                <h2 className="text-lg font-semibold">{symbol}</h2>
                {priceChange && (
                  <div className="flex items-center gap-2">
                    <span className="text-xl font-mono">
                      {formatPrice(priceChange.currentPrice)}
                    </span>
                    <div className="flex items-center gap-1">
                      <TrendingUp 
                        className={cn(
                          "h-4 w-4",
                          priceChange.change >= 0 ? "text-green-500" : "text-red-500 rotate-180"
                        )}
                      />
                      <span className={cn(
                        "text-sm font-mono",
                        priceChange.change >= 0 ? "text-green-500" : "text-red-500"
                      )}>
                        {priceChange.change >= 0 ? '+' : ''}
                        {formatPrice(priceChange.change)} ({priceChange.changePercent.toFixed(2)}%)
                      </span>
                    </div>
                  </div>
                )}
              </div>
            </div>

            <div className="flex items-center gap-2">
              {/* Chart type selector */}
              <div className="flex border rounded-md">
                {chartTypes.map((type, index) => (
                  <Button
                    key={type.value}
                    variant={chartType === type.value ? 'default' : 'ghost'}
                    size="sm"
                    onClick={() => handleChartTypeChange({} as any, type.value)}
                    className={cn(
                      index === 0 ? 'rounded-r-none' : 
                      index === chartTypes.length - 1 ? 'rounded-l-none' : 'rounded-none border-x-0'
                    )}
                  >
                    {type.label}
                  </Button>
                ))}
              </div>

              {/* Timeframe selector */}
              <div className="flex border rounded-md">
                {timeframes.map((tf, index) => (
                  <Button
                    key={tf.value}
                    variant={timeframe === tf.value ? 'default' : 'ghost'}
                    size="sm"
                    onClick={() => handleTimeframeChange({} as any, tf.value)}
                    className={cn(
                      "px-2",
                      index === 0 ? 'rounded-r-none' : 
                      index === timeframes.length - 1 ? 'rounded-l-none' : 'rounded-none border-x-0'
                    )}
                  >
                    {tf.label}
                  </Button>
                ))}
              </div>

              {/* Action buttons */}
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

              {onSettingsClick && (
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button variant="ghost" size="icon" onClick={onSettingsClick}>
                      <Settings className="h-4 w-4" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Settings</TooltipContent>
                </Tooltip>
              )}

              {onFullscreenToggle && (
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button variant="ghost" size="icon" onClick={onFullscreenToggle}>
                      {fullscreen ? <Minimize className="h-4 w-4" /> : <Maximize className="h-4 w-4" />}
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>{fullscreen ? "Exit Fullscreen" : "Fullscreen"}</TooltipContent>
                </Tooltip>
              )}
            </div>
          </div>

          {/* Chart area */}
          <div className="flex-1 relative">
            {renderChart()}
          </div>
        </Card>
      </motion.div>
    </TooltipProvider>
  );
};

export default PriceChart;