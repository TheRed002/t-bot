/**
 * OrderForm component - Professional order entry interface
 * Handles market orders, limit orders, stop orders with validation
 */

import React, { useState, useCallback, useMemo } from 'react';
import {
  Paper,
  Box,
  Typography,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  ToggleButton,
  ToggleButtonGroup,
  Slider,
  Chip,
  Alert,
  Divider,
  InputAdornment,
  Tooltip,
  IconButton,
  Switch,
  FormControlLabel,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Calculate,
  Info,
  Security,
  Speed,
  Timer,
} from '@mui/icons-material';
import { motion } from 'framer-motion';
import { useForm, Controller } from 'react-hook-form';
import { colors } from '@/theme/colors';
import { tradingTheme } from '@/theme';

export interface OrderData {
  symbol: string;
  side: 'buy' | 'sell';
  type: 'market' | 'limit' | 'stop' | 'stop_limit';
  quantity: number;
  price?: number;
  stopPrice?: number;
  timeInForce: 'GTC' | 'IOC' | 'FOK' | 'GTD';
  leverage?: number;
  reduceOnly?: boolean;
  postOnly?: boolean;
}

interface OrderFormProps {
  symbol?: string;
  currentPrice?: number;
  availableBalance?: number;
  maxLeverage?: number;
  onSubmit: (order: OrderData) => Promise<void>;
  onCalculateSize?: (percentage: number) => number;
  loading?: boolean;
  error?: string;
}

const leverageMarks = [
  { value: 1, label: '1x' },
  { value: 5, label: '5x' },
  { value: 10, label: '10x' },
  { value: 20, label: '20x' },
  { value: 50, label: '50x' },
  { value: 100, label: '100x' },
];

const sizePercentages = [10, 25, 50, 75, 100];

export const OrderForm: React.FC<OrderFormProps> = ({
  symbol = 'BTC/USD',
  currentPrice = 45000,
  availableBalance = 1000,
  maxLeverage = 100,
  onSubmit,
  onCalculateSize,
  loading = false,
  error,
}) => {
  const [side, setSide] = useState<'buy' | 'sell'>('buy');
  const [orderType, setOrderType] = useState<'market' | 'limit' | 'stop' | 'stop_limit'>('market');
  const [advancedMode, setAdvancedMode] = useState(false);

  const { control, handleSubmit, watch, setValue, formState: { errors } } = useForm<OrderData>({
    defaultValues: {
      symbol,
      side: 'buy',
      type: 'market',
      quantity: 0,
      leverage: 1,
      timeInForce: 'GTC',
      reduceOnly: false,
      postOnly: false,
    },
  });

  const watchedValues = watch();
  const { quantity, price, leverage } = watchedValues;

  // Calculate order value and fees
  const orderCalculations = useMemo(() => {
    const orderPrice = orderType === 'market' ? currentPrice : (price || currentPrice);
    const orderValue = quantity * orderPrice;
    const leveragedValue = orderValue * (leverage || 1);
    const margin = orderValue / (leverage || 1);
    const fee = orderValue * 0.001; // 0.1% fee
    const maxQuantity = availableBalance / orderPrice * (leverage || 1);

    return {
      orderPrice,
      orderValue,
      leveragedValue,
      margin,
      fee,
      maxQuantity,
      total: margin + fee,
    };
  }, [quantity, price, leverage, orderType, currentPrice, availableBalance]);

  const handleSideChange = useCallback((
    event: React.MouseEvent<HTMLElement>,
    newSide: 'buy' | 'sell' | null,
  ) => {
    if (newSide) {
      setSide(newSide);
      setValue('side', newSide);
    }
  }, [setValue]);

  const handleOrderTypeChange = useCallback((
    event: React.MouseEvent<HTMLElement>,
    newType: string | null,
  ) => {
    if (newType) {
      setOrderType(newType as any);
      setValue('type', newType as any);
    }
  }, [setValue]);

  const handleSizePercentage = useCallback((percentage: number) => {
    const calculatedSize = onCalculateSize ? 
      onCalculateSize(percentage) : 
      (orderCalculations.maxQuantity * percentage / 100);
    setValue('quantity', calculatedSize);
  }, [setValue, onCalculateSize, orderCalculations.maxQuantity]);

  const handleFormSubmit = async (data: OrderData) => {
    try {
      await onSubmit(data);
    } catch (err) {
      console.error('Order submission error:', err);
    }
  };

  const isFormValid = useMemo(() => {
    return quantity > 0 && 
           orderCalculations.total <= availableBalance &&
           (orderType === 'market' || (price && price > 0));
  }, [quantity, orderCalculations.total, availableBalance, orderType, price]);

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3 }}
    >
      <Paper sx={{ p: 3 }}>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            Place Order
          </Typography>
          <FormControlLabel
            control={
              <Switch
                checked={advancedMode}
                onChange={(e) => setAdvancedMode(e.target.checked)}
                size="small"
              />
            }
            label="Advanced"
          />
        </Box>

        <form onSubmit={handleSubmit(handleFormSubmit)}>
          {/* Symbol and Current Price */}
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
            <Typography variant="h5" sx={{ fontWeight: 600 }}>
              {symbol}
            </Typography>
            <Box textAlign="right">
              <Typography variant="body2" color="text.secondary">
                Market Price
              </Typography>
              <Typography variant="h6" sx={{ fontFamily: 'monospace' }}>
                ${currentPrice.toLocaleString()}
              </Typography>
            </Box>
          </Box>

          {/* Side Selection */}
          <Box mb={3}>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Order Side
            </Typography>
            <ToggleButtonGroup
              value={side}
              exclusive
              onChange={handleSideChange}
              fullWidth
              sx={{ mb: 2 }}
            >
              <ToggleButton 
                value="buy" 
                sx={{ 
                  color: side === 'buy' ? 'white' : colors.financial.profit,
                  backgroundColor: side === 'buy' ? colors.financial.profit : 'transparent',
                  '&:hover': {
                    backgroundColor: side === 'buy' ? colors.financial.profit : 'rgba(76, 175, 80, 0.1)',
                  },
                }}
              >
                <TrendingUp sx={{ mr: 1 }} />
                Buy / Long
              </ToggleButton>
              <ToggleButton 
                value="sell"
                sx={{ 
                  color: side === 'sell' ? 'white' : colors.financial.loss,
                  backgroundColor: side === 'sell' ? colors.financial.loss : 'transparent',
                  '&:hover': {
                    backgroundColor: side === 'sell' ? colors.financial.loss : 'rgba(244, 67, 54, 0.1)',
                  },
                }}
              >
                <TrendingDown sx={{ mr: 1 }} />
                Sell / Short
              </ToggleButton>
            </ToggleButtonGroup>
          </Box>

          {/* Order Type */}
          <Box mb={3}>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Order Type
            </Typography>
            <ToggleButtonGroup
              value={orderType}
              exclusive
              onChange={handleOrderTypeChange}
              fullWidth
              size="small"
            >
              <ToggleButton value="market">
                <Speed sx={{ mr: 1 }} />
                Market
              </ToggleButton>
              <ToggleButton value="limit">
                <Calculate sx={{ mr: 1 }} />
                Limit
              </ToggleButton>
              <ToggleButton value="stop">
                <Security sx={{ mr: 1 }} />
                Stop
              </ToggleButton>
              <ToggleButton value="stop_limit">
                <Timer sx={{ mr: 1 }} />
                Stop Limit
              </ToggleButton>
            </ToggleButtonGroup>
          </Box>

          {/* Price Input (for limit/stop orders) */}
          {(orderType === 'limit' || orderType === 'stop_limit') && (
            <Box mb={3}>
              <Controller
                name="price"
                control={control}
                rules={{ required: 'Price is required', min: { value: 0.01, message: 'Price must be positive' } }}
                render={({ field }) => (
                  <TextField
                    {...field}
                    label="Limit Price"
                    type="number"
                    fullWidth
                    InputProps={{
                      startAdornment: <InputAdornment position="start">$</InputAdornment>,
                    }}
                    error={!!errors.price}
                    helperText={errors.price?.message}
                  />
                )}
              />
            </Box>
          )}

          {/* Stop Price (for stop orders) */}
          {(orderType === 'stop' || orderType === 'stop_limit') && (
            <Box mb={3}>
              <Controller
                name="stopPrice"
                control={control}
                rules={{ required: 'Stop price is required', min: { value: 0.01, message: 'Stop price must be positive' } }}
                render={({ field }) => (
                  <TextField
                    {...field}
                    label="Stop Price"
                    type="number"
                    fullWidth
                    InputProps={{
                      startAdornment: <InputAdornment position="start">$</InputAdornment>,
                    }}
                    error={!!errors.stopPrice}
                    helperText={errors.stopPrice?.message}
                  />
                )}
              />
            </Box>
          )}

          {/* Leverage Slider */}
          <Box mb={3}>
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
              <Typography variant="body2" color="text.secondary">
                Leverage
              </Typography>
              <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                {leverage}x
              </Typography>
            </Box>
            <Controller
              name="leverage"
              control={control}
              render={({ field }) => (
                <Slider
                  {...field}
                  min={1}
                  max={maxLeverage}
                  marks={leverageMarks.filter(mark => mark.value <= maxLeverage)}
                  valueLabelDisplay="auto"
                  sx={{
                    color: leverage && leverage > 10 ? colors.financial.warning : colors.primary[500],
                  }}
                />
              )}
            />
          </Box>

          {/* Quantity Input */}
          <Box mb={3}>
            <Controller
              name="quantity"
              control={control}
              rules={{ 
                required: 'Quantity is required', 
                min: { value: 0.00001, message: 'Quantity must be positive' },
                max: { value: orderCalculations.maxQuantity, message: 'Insufficient balance' }
              }}
              render={({ field }) => (
                <TextField
                  {...field}
                  label="Quantity"
                  type="number"
                  fullWidth
                  InputProps={{
                    endAdornment: (
                      <InputAdornment position="end">
                        <Tooltip title="Calculate max quantity">
                          <IconButton
                            size="small"
                            onClick={() => handleSizePercentage(100)}
                          >
                            <Calculate />
                          </IconButton>
                        </Tooltip>
                      </InputAdornment>
                    ),
                  }}
                  error={!!errors.quantity}
                  helperText={errors.quantity?.message || `Max: ${orderCalculations.maxQuantity.toFixed(6)}`}
                />
              )}
            />

            {/* Size Percentage Buttons */}
            <Box display="flex" gap={1} mt={1}>
              {sizePercentages.map((percentage) => (
                <Chip
                  key={percentage}
                  label={`${percentage}%`}
                  variant="outlined"
                  size="small"
                  onClick={() => handleSizePercentage(percentage)}
                  sx={{ cursor: 'pointer' }}
                />
              ))}
            </Box>
          </Box>

          {/* Advanced Options */}
          {advancedMode && (
            <Box mb={3}>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Advanced Options
              </Typography>
              
              <Controller
                name="timeInForce"
                control={control}
                render={({ field }) => (
                  <FormControl fullWidth size="small" sx={{ mb: 2 }}>
                    <InputLabel>Time in Force</InputLabel>
                    <Select {...field} label="Time in Force">
                      <MenuItem value="GTC">Good Till Canceled</MenuItem>
                      <MenuItem value="IOC">Immediate or Cancel</MenuItem>
                      <MenuItem value="FOK">Fill or Kill</MenuItem>
                      <MenuItem value="GTD">Good Till Date</MenuItem>
                    </Select>
                  </FormControl>
                )}
              />

              <Box display="flex" gap={2}>
                <Controller
                  name="reduceOnly"
                  control={control}
                  render={({ field }) => (
                    <FormControlLabel
                      control={<Switch {...field} size="small" />}
                      label="Reduce Only"
                    />
                  )}
                />
                
                <Controller
                  name="postOnly"
                  control={control}
                  render={({ field }) => (
                    <FormControlLabel
                      control={<Switch {...field} size="small" />}
                      label="Post Only"
                    />
                  )}
                />
              </Box>
            </Box>
          )}

          <Divider sx={{ my: 3 }} />

          {/* Order Summary */}
          <Box mb={3}>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Order Summary
            </Typography>
            <Box display="flex" flexDirection="column" gap={1}>
              <Box display="flex" justifyContent="space-between">
                <Typography variant="body2">Order Value:</Typography>
                <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                  ${orderCalculations.orderValue.toFixed(2)}
                </Typography>
              </Box>
              <Box display="flex" justifyContent="space-between">
                <Typography variant="body2">Margin Required:</Typography>
                <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                  ${orderCalculations.margin.toFixed(2)}
                </Typography>
              </Box>
              <Box display="flex" justifyContent="space-between">
                <Typography variant="body2">Est. Fee:</Typography>
                <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                  ${orderCalculations.fee.toFixed(2)}
                </Typography>
              </Box>
              <Box display="flex" justifyContent="space-between">
                <Typography variant="body2" sx={{ fontWeight: 600 }}>Total:</Typography>
                <Typography variant="body2" sx={{ fontFamily: 'monospace', fontWeight: 600 }}>
                  ${orderCalculations.total.toFixed(2)}
                </Typography>
              </Box>
            </Box>
          </Box>

          {/* Risk Warning */}
          {leverage && leverage > 10 && (
            <Alert severity="warning" sx={{ mb: 3 }}>
              <Typography variant="body2">
                High leverage increases both potential profits and losses. 
                Consider your risk tolerance carefully.
              </Typography>
            </Alert>
          )}

          {/* Error Display */}
          {error && (
            <Alert severity="error" sx={{ mb: 3 }}>
              {error}
            </Alert>
          )}

          {/* Submit Button */}
          <Button
            type="submit"
            variant="contained"
            fullWidth
            size="large"
            disabled={!isFormValid || loading}
            sx={{
              backgroundColor: side === 'buy' ? colors.financial.profit : colors.financial.loss,
              '&:hover': {
                backgroundColor: side === 'buy' ? 
                  'rgba(76, 175, 80, 0.8)' : 
                  'rgba(244, 67, 54, 0.8)',
              },
              py: 1.5,
              fontWeight: 600,
            }}
          >
            {loading ? 'Placing Order...' : `${side.toUpperCase()} ${symbol}`}
          </Button>

          {/* Balance Info */}
          <Box mt={2} textAlign="center">
            <Typography variant="caption" color="text.secondary">
              Available Balance: ${availableBalance.toFixed(2)}
            </Typography>
          </Box>
        </form>
      </Paper>
    </motion.div>
  );
};

export default OrderForm;