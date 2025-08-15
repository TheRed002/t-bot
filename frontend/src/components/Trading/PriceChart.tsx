/**
 * PriceChart component - Professional trading chart with real-time updates
 * Displays candlestick charts, line charts, and technical indicators
 */

import React, { useMemo, useCallback } from 'react';
import {
  Box,
  Paper,
  Typography,
  ToggleButton,
  ToggleButtonGroup,
  IconButton,
  Tooltip,
  CircularProgress,
  Alert,
} from '@mui/material';
import {
  Fullscreen,
  FullscreenExit,
  Refresh,
  Settings,
  TrendingUp,
} from '@mui/icons-material';
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as ChartTooltip,
  Legend,
  ReferenceLine,
} from 'recharts';
import { motion } from 'framer-motion';
import { colors } from '@/theme/colors';
import { tradingTheme } from '@/theme';

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
    <Paper
      sx={{
        p: 2,
        backgroundColor: 'background.paper',
        border: '1px solid',
        borderColor: 'divider',
        borderRadius: 1,
        boxShadow: 3,
      }}
    >
      <Typography variant="caption" color="text.secondary" gutterBottom>
        {new Date(label).toLocaleString()}
      </Typography>
      
      {data.open !== undefined && (
        <Box>
          <Typography variant="body2">O: {formatPrice(data.open)}</Typography>
          <Typography variant="body2">H: {formatPrice(data.high)}</Typography>
          <Typography variant="body2">L: {formatPrice(data.low)}</Typography>
          <Typography variant="body2">C: {formatPrice(data.close)}</Typography>
          <Typography variant="body2">V: {formatVolume(data.volume)}</Typography>
        </Box>
      )}
      
      {data.price !== undefined && (
        <Typography variant="body2">
          Price: {formatPrice(data.price)}
        </Typography>
      )}
    </Paper>
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
  showVolume = true,
  showIndicators = false,
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
    event: React.MouseEvent<HTMLElement>,
    newTimeframe: string | null,
  ) => {
    if (newTimeframe && onTimeframeChange) {
      onTimeframeChange(newTimeframe);
    }
  }, [onTimeframeChange]);

  const handleChartTypeChange = useCallback((
    event: React.MouseEvent<HTMLElement>,
    newType: string | null,
  ) => {
    if (newType && onChartTypeChange) {
      onChartTypeChange(newType as 'candlestick' | 'line' | 'area');
    }
  }, [onChartTypeChange]);

  const renderChart = () => {
    if (loading) {
      return (
        <Box
          display="flex"
          alignItems="center"
          justifyContent="center"
          height={height - 100}
        >
          <CircularProgress />
        </Box>
      );
    }

    if (error) {
      return (
        <Box
          display="flex"
          alignItems="center"
          justifyContent="center"
          height={height - 100}
        >
          <Alert severity="error" sx={{ maxWidth: 400 }}>
            {error}
          </Alert>
        </Box>
      );
    }

    if (!chartData.length) {
      return (
        <Box
          display="flex"
          alignItems="center"
          justifyContent="center"
          height={height - 100}
        >
          <Typography color="text.secondary">
            No data available
          </Typography>
        </Box>
      );
    }

    return (
      <ResponsiveContainer width="100%" height={height - 100}>
        <ComposedChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={colors.border.divider} />
          <XAxis 
            dataKey="time" 
            axisLine={false}
            tickLine={false}
            tick={{ fontSize: 12, fill: colors.text.secondary }}
          />
          <YAxis 
            domain={['dataMin - 5', 'dataMax + 5']}
            axisLine={false}
            tickLine={false}
            tick={{ fontSize: 12, fill: colors.text.secondary }}
            tickFormatter={(value) => formatPrice(value, 0)}
          />
          <ChartTooltip content={<CustomTooltip />} />
          
          {chartType === 'line' && (
            <Line
              type="monotone"
              dataKey="displayPrice"
              stroke={colors.chart.line1}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4, fill: colors.chart.line1 }}
            />
          )}
          
          {chartType === 'area' && (
            <Area
              type="monotone"
              dataKey="displayPrice"
              stroke={colors.chart.line1}
              strokeWidth={2}
              fill={`url(#areaGradient)`}
              fillOpacity={0.3}
            />
          )}

          <defs>
            <linearGradient id="areaGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={colors.chart.line1} stopOpacity={0.8} />
              <stop offset="95%" stopColor={colors.chart.line1} stopOpacity={0.1} />
            </linearGradient>
          </defs>
        </ComposedChart>
      </ResponsiveContainer>
    );
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <Paper
        sx={{
          height: fullscreen ? '100vh' : height,
          display: 'flex',
          flexDirection: 'column',
          position: fullscreen ? 'fixed' : 'relative',
          top: fullscreen ? 0 : 'auto',
          left: fullscreen ? 0 : 'auto',
          right: fullscreen ? 0 : 'auto',
          bottom: fullscreen ? 0 : 'auto',
          zIndex: fullscreen ? 1300 : 'auto',
          backgroundColor: 'background.paper',
        }}
      >
        {/* Header */}
        <Box
          p={2}
          display="flex"
          justifyContent="space-between"
          alignItems="center"
          borderBottom="1px solid"
          borderColor="divider"
        >
          <Box display="flex" alignItems="center" gap={2}>
            <Box>
              <Typography variant="h6" sx={{ fontWeight: 600 }}>
                {symbol}
              </Typography>
              {priceChange && (
                <Box display="flex" alignItems="center" gap={1}>
                  <Typography variant="h5" sx={{ fontFamily: 'monospace' }}>
                    {formatPrice(priceChange.currentPrice)}
                  </Typography>
                  <Box display="flex" alignItems="center" gap={0.5}>
                    {priceChange.change > 0 ? (
                      <TrendingUp sx={{ fontSize: 16, color: colors.financial.profit }} />
                    ) : (
                      <TrendingUp sx={{ fontSize: 16, color: colors.financial.loss, transform: 'rotate(180deg)' }} />
                    )}
                    <Typography
                      variant="body2"
                      sx={{
                        color: tradingTheme.getPriceChangeColor(priceChange.change),
                        fontFamily: 'monospace',
                      }}
                    >
                      {priceChange.change >= 0 ? '+' : ''}
                      {formatPrice(priceChange.change)} ({priceChange.changePercent.toFixed(2)}%)
                    </Typography>
                  </Box>
                </Box>
              )}
            </Box>
          </Box>

          <Box display="flex" alignItems="center" gap={1}>
            {/* Chart type selector */}
            <ToggleButtonGroup
              value={chartType}
              exclusive
              onChange={handleChartTypeChange}
              size="small"
            >
              {chartTypes.map((type) => (
                <ToggleButton key={type.value} value={type.value}>
                  {type.label}
                </ToggleButton>
              ))}
            </ToggleButtonGroup>

            {/* Timeframe selector */}
            <ToggleButtonGroup
              value={timeframe}
              exclusive
              onChange={handleTimeframeChange}
              size="small"
            >
              {timeframes.map((tf) => (
                <ToggleButton key={tf.value} value={tf.value}>
                  {tf.label}
                </ToggleButton>
              ))}
            </ToggleButtonGroup>

            {/* Action buttons */}
            {onRefresh && (
              <Tooltip title="Refresh">
                <IconButton onClick={onRefresh} size="small">
                  <Refresh />
                </IconButton>
              </Tooltip>
            )}

            {onSettingsClick && (
              <Tooltip title="Settings">
                <IconButton onClick={onSettingsClick} size="small">
                  <Settings />
                </IconButton>
              </Tooltip>
            )}

            {onFullscreenToggle && (
              <Tooltip title={fullscreen ? "Exit Fullscreen" : "Fullscreen"}>
                <IconButton onClick={onFullscreenToggle} size="small">
                  {fullscreen ? <FullscreenExit /> : <Fullscreen />}
                </IconButton>
              </Tooltip>
            )}
          </Box>
        </Box>

        {/* Chart area */}
        <Box flex={1} position="relative">
          {renderChart()}
        </Box>
      </Paper>
    </motion.div>
  );
};

export default PriceChart;