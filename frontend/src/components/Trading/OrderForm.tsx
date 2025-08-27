/**
 * OrderForm component - Professional order entry interface
 * Handles market orders, limit orders, stop orders with validation
 */

import React, { useState, useCallback, useMemo } from 'react';
import {
  TrendingUp,
  TrendingDown,
  Calculator,
  Shield,
  Zap,
  Clock,
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Separator } from '@/components/ui/separator';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { Switch } from '@/components/ui/switch';
import { cn } from '@/lib/utils';
import { motion } from 'framer-motion';
import { useForm, Controller } from 'react-hook-form';

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
    _event: React.MouseEvent<HTMLElement>,
    newSide: 'buy' | 'sell' | null,
  ) => {
    if (newSide) {
      setSide(newSide);
      setValue('side', newSide);
    }
  }, [setValue]);

  const handleOrderTypeChange = useCallback((
    _event: React.MouseEvent<HTMLElement>,
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
    <TooltipProvider>
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.3 }}
      >
        <Card className="p-6">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-semibold">Place Order</h2>
            <div className="flex items-center space-x-2">
              <Label htmlFor="advanced-mode" className="text-sm">Advanced</Label>
              <Switch
                id="advanced-mode"
                checked={advancedMode}
                onCheckedChange={setAdvancedMode}
              />
            </div>
          </div>

          <form onSubmit={handleSubmit(handleFormSubmit)} className="space-y-6">
            {/* Symbol and Current Price */}
            <div className="flex justify-between items-center">
              <h3 className="text-2xl font-semibold">{symbol}</h3>
              <div className="text-right">
                <p className="text-sm text-muted-foreground">Market Price</p>
                <p className="text-lg font-mono">${currentPrice.toLocaleString()}</p>
              </div>
            </div>

            {/* Side Selection */}
            <div className="space-y-2">
              <Label className="text-sm text-muted-foreground">Order Side</Label>
              <div className="grid grid-cols-2 gap-2">
                <Button
                  type="button"
                  variant={side === 'buy' ? 'default' : 'outline'}
                  onClick={() => handleSideChange({} as any, 'buy')}
                  className={cn(
                    "w-full",
                    side === 'buy' 
                      ? 'bg-green-600 hover:bg-green-700 text-white' 
                      : 'text-green-600 border-green-600 hover:bg-green-600/10'
                  )}
                >
                  <TrendingUp className="w-4 h-4 mr-2" />
                  Buy / Long
                </Button>
                <Button
                  type="button"
                  variant={side === 'sell' ? 'default' : 'outline'}
                  onClick={() => handleSideChange({} as any, 'sell')}
                  className={cn(
                    "w-full",
                    side === 'sell' 
                      ? 'bg-red-600 hover:bg-red-700 text-white' 
                      : 'text-red-600 border-red-600 hover:bg-red-600/10'
                  )}
                >
                  <TrendingDown className="w-4 h-4 mr-2" />
                  Sell / Short
                </Button>
              </div>
            </div>

            {/* Order Type */}
            <div className="space-y-2">
              <Label className="text-sm text-muted-foreground">Order Type</Label>
              <div className="grid grid-cols-4 gap-1">
                <Button
                  type="button"
                  variant={orderType === 'market' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => handleOrderTypeChange({} as any, 'market')}
                  className="flex flex-col items-center p-2 h-auto"
                >
                  <Zap className="w-4 h-4 mb-1" />
                  <span className="text-xs">Market</span>
                </Button>
                <Button
                  type="button"
                  variant={orderType === 'limit' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => handleOrderTypeChange({} as any, 'limit')}
                  className="flex flex-col items-center p-2 h-auto"
                >
                  <Calculator className="w-4 h-4 mb-1" />
                  <span className="text-xs">Limit</span>
                </Button>
                <Button
                  type="button"
                  variant={orderType === 'stop' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => handleOrderTypeChange({} as any, 'stop')}
                  className="flex flex-col items-center p-2 h-auto"
                >
                  <Shield className="w-4 h-4 mb-1" />
                  <span className="text-xs">Stop</span>
                </Button>
                <Button
                  type="button"
                  variant={orderType === 'stop_limit' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => handleOrderTypeChange({} as any, 'stop_limit')}
                  className="flex flex-col items-center p-2 h-auto"
                >
                  <Clock className="w-4 h-4 mb-1" />
                  <span className="text-xs">Stop Limit</span>
                </Button>
              </div>
            </div>

            {/* Price Input (for limit/stop orders) */}
            {(orderType === 'limit' || orderType === 'stop_limit') && (
              <div className="space-y-2">
                <Label htmlFor="price">Limit Price</Label>
                <Controller
                  name="price"
                  control={control}
                  rules={{ required: 'Price is required', min: { value: 0.01, message: 'Price must be positive' } }}
                  render={({ field }) => (
                    <div className="relative">
                      <span className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground">$</span>
                      <Input
                        {...field}
                        id="price"
                        type="number"
                        className="pl-6"
                        placeholder="0.00"
                      />
                      {errors.price && (
                        <p className="text-sm text-red-500 mt-1">{errors.price.message}</p>
                      )}
                    </div>
                  )}
                />
              </div>
            )}

            {/* Stop Price (for stop orders) */}
            {(orderType === 'stop' || orderType === 'stop_limit') && (
              <div className="space-y-2">
                <Label htmlFor="stopPrice">Stop Price</Label>
                <Controller
                  name="stopPrice"
                  control={control}
                  rules={{ required: 'Stop price is required', min: { value: 0.01, message: 'Stop price must be positive' } }}
                  render={({ field }) => (
                    <div className="relative">
                      <span className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground">$</span>
                      <Input
                        {...field}
                        id="stopPrice"
                        type="number"
                        className="pl-6"
                        placeholder="0.00"
                      />
                      {errors.stopPrice && (
                        <p className="text-sm text-red-500 mt-1">{errors.stopPrice.message}</p>
                      )}
                    </div>
                  )}
                />
              </div>
            )}

            {/* Leverage Slider */}
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <Label className="text-sm text-muted-foreground">Leverage</Label>
                <span className={cn(
                  "text-sm font-mono",
                  leverage && leverage > 10 ? "text-yellow-500" : "text-foreground"
                )}>
                  {leverage}x
                </span>
              </div>
              <Controller
                name="leverage"
                control={control}
                render={({ field }) => (
                  <div className="px-2">
                    <input
                      type="range"
                      min={1}
                      max={maxLeverage}
                      value={field.value || 1}
                      onChange={(e) => field.onChange(Number(e.target.value))}
                      className={cn(
                        "w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider",
                        leverage && leverage > 10 ? "accent-yellow-500" : "accent-primary"
                      )}
                    />
                    <div className="flex justify-between text-xs text-muted-foreground mt-1">
                      {leverageMarks.filter(mark => mark.value <= maxLeverage).map(mark => (
                        <span key={mark.value}>{mark.label}</span>
                      ))}
                    </div>
                  </div>
                )}
              />
            </div>

            {/* Quantity Input */}
            <div className="space-y-2">
              <Label htmlFor="quantity">Quantity</Label>
              <Controller
                name="quantity"
                control={control}
                rules={{ 
                  required: 'Quantity is required', 
                  min: { value: 0.00001, message: 'Quantity must be positive' },
                  max: { value: orderCalculations.maxQuantity, message: 'Insufficient balance' }
                }}
                render={({ field }) => (
                  <div className="relative">
                    <Input
                      {...field}
                      id="quantity"
                      type="number"
                      placeholder="0.00"
                      className="pr-10"
                    />
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Button
                          type="button"
                          variant="ghost"
                          size="icon"
                          className="absolute right-1 top-1/2 transform -translate-y-1/2 h-8 w-8"
                          onClick={() => handleSizePercentage(100)}
                        >
                          <Calculator className="h-4 w-4" />
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent>Calculate max quantity</TooltipContent>
                    </Tooltip>
                  </div>
                )}
              />
              {errors.quantity ? (
                <p className="text-sm text-red-500">{errors.quantity.message}</p>
              ) : (
                <p className="text-sm text-muted-foreground">Max: {orderCalculations.maxQuantity.toFixed(6)}</p>
              )}

              {/* Size Percentage Buttons */}
              <div className="flex gap-2 flex-wrap">
                {sizePercentages.map((percentage) => (
                  <Badge
                    key={percentage}
                    variant="outline"
                    className="cursor-pointer hover:bg-muted"
                    onClick={() => handleSizePercentage(percentage)}
                  >
                    {percentage}%
                  </Badge>
                ))}
              </div>
            </div>

            {/* Advanced Options */}
            {advancedMode && (
              <div className="space-y-4">
                <Label className="text-sm text-muted-foreground">Advanced Options</Label>
                
                <Controller
                  name="timeInForce"
                  control={control}
                  render={({ field }) => (
                    <div className="space-y-2">
                      <Label htmlFor="timeInForce">Time in Force</Label>
                      <Select onValueChange={field.onChange} defaultValue={field.value}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select time in force" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="GTC">Good Till Canceled</SelectItem>
                          <SelectItem value="IOC">Immediate or Cancel</SelectItem>
                          <SelectItem value="FOK">Fill or Kill</SelectItem>
                          <SelectItem value="GTD">Good Till Date</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  )}
                />

                <div className="flex gap-6">
                  <Controller
                    name="reduceOnly"
                    control={control}
                    render={({ field }) => (
                      <div className="flex items-center space-x-2">
                        <Switch
                          id="reduceOnly"
                          checked={field.value}
                          onCheckedChange={field.onChange}
                        />
                        <Label htmlFor="reduceOnly" className="text-sm">Reduce Only</Label>
                      </div>
                    )}
                  />
                  
                  <Controller
                    name="postOnly"
                    control={control}
                    render={({ field }) => (
                      <div className="flex items-center space-x-2">
                        <Switch
                          id="postOnly"
                          checked={field.value}
                          onCheckedChange={field.onChange}
                        />
                        <Label htmlFor="postOnly" className="text-sm">Post Only</Label>
                      </div>
                    )}
                  />
                </div>
              </div>
            )}

            <Separator className="my-6" />

            {/* Order Summary */}
            <div className="space-y-3">
              <Label className="text-sm text-muted-foreground">Order Summary</Label>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm">Order Value:</span>
                  <span className="text-sm font-mono">${orderCalculations.orderValue.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm">Margin Required:</span>
                  <span className="text-sm font-mono">${orderCalculations.margin.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm">Est. Fee:</span>
                  <span className="text-sm font-mono">${orderCalculations.fee.toFixed(2)}</span>
                </div>
                <Separator />
                <div className="flex justify-between">
                  <span className="text-sm font-semibold">Total:</span>
                  <span className="text-sm font-mono font-semibold">${orderCalculations.total.toFixed(2)}</span>
                </div>
              </div>
            </div>

            {/* Risk Warning */}
            {leverage && leverage > 10 && (
              <Alert className="mb-4">
                <AlertDescription>
                  High leverage increases both potential profits and losses. 
                  Consider your risk tolerance carefully.
                </AlertDescription>
              </Alert>
            )}

            {/* Error Display */}
            {error && (
              <Alert className="mb-4 border-red-200 bg-red-50">
                <AlertDescription className="text-red-800">
                  {error}
                </AlertDescription>
              </Alert>
            )}

            {/* Submit Button */}
            <Button
              type="submit"
              size="lg"
              disabled={!isFormValid || loading}
              className={cn(
                "w-full py-3 font-semibold",
                side === 'buy' 
                  ? 'bg-green-600 hover:bg-green-700' 
                  : 'bg-red-600 hover:bg-red-700'
              )}
            >
              {loading ? 'Placing Order...' : `${side.toUpperCase()} ${symbol}`}
            </Button>

            {/* Balance Info */}
            <div className="mt-4 text-center">
              <p className="text-xs text-muted-foreground">
                Available Balance: ${availableBalance.toFixed(2)}
              </p>
            </div>
          </form>
        </Card>
      </motion.div>
    </TooltipProvider>
  );
};

export default OrderForm;