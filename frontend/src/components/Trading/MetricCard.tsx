/**
 * MetricCard component - Professional trading metric display
 * Shows financial metrics with proper formatting and color coding
 */

import React from 'react';
import {
  TrendingUp,
  TrendingDown,
  Info,
  RefreshCw,
} from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { Skeleton } from '@/components/ui/skeleton';
import { cn } from '@/lib/utils';
import { motion } from 'framer-motion';

interface MetricCardProps {
  title: string;
  value: string | number;
  change?: number;
  changePercent?: number;
  subtitle?: string;
  loading?: boolean;
  tooltip?: string;
  onClick?: () => void;
  onRefresh?: () => void;
  format?: 'currency' | 'percentage' | 'number' | 'custom';
  precision?: number;
  trend?: 'up' | 'down' | 'neutral';
  size?: 'small' | 'medium' | 'large';
  variant?: 'default' | 'highlighted' | 'minimal';
}

const formatValue = (
  value: string | number,
  format: string,
  precision: number = 2
): string => {
  if (typeof value === 'string') return value;
  
  switch (format) {
    case 'currency':
      return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: precision,
        maximumFractionDigits: precision,
      }).format(value);
    case 'percentage':
      return `${(value * 100).toFixed(precision)}%`;
    case 'number':
      return new Intl.NumberFormat('en-US', {
        minimumFractionDigits: precision,
        maximumFractionDigits: precision,
      }).format(value);
    default:
      return value.toString();
  }
};

const formatChange = (change: number, format: string): string => {
  const prefix = change >= 0 ? '+' : '';
  if (format === 'currency') {
    return `${prefix}${formatValue(change, 'currency')}`;
  }
  return `${prefix}${formatValue(change, format)}`;
};

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  change,
  changePercent,
  subtitle,
  loading = false,
  tooltip,
  onClick,
  onRefresh,
  format = 'number',
  precision = 2,
  trend,
  size = 'medium',
  variant = 'default',
}) => {
  const cardHeight = {
    small: 120,
    medium: 140,
    large: 160,
  }[size];

  const valueSize = {
    small: 'h6',
    medium: 'h5',
    large: 'h4',
  }[size] as any;

  const determineTrend = (): 'up' | 'down' | 'neutral' => {
    if (trend) return trend;
    if (change !== undefined) {
      return change > 0 ? 'up' : change < 0 ? 'down' : 'neutral';
    }
    return 'neutral';
  };

  const currentTrend = determineTrend();
  const changeColor = (change || 0) >= 0 ? 'text-green-500' : 'text-red-500';

  const getCardClasses = () => {
    const base = "transition-all duration-200 hover:shadow-lg";
    switch (variant) {
      case 'highlighted':
        return cn(
          base,
          "bg-gradient-to-br from-blue-500/10 to-purple-500/10 border-blue-500/20 hover:shadow-blue-500/30"
        );
      case 'minimal':
        return cn(base, "bg-transparent border");
      default:
        return base;
    }
  };

  return (
    <TooltipProvider>
      <motion.div
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
        transition={{ duration: 0.2 }}
      >
        <Card
          className={cn(
            getCardClasses(),
            "relative overflow-visible",
            onClick ? "cursor-pointer" : "cursor-default"
          )}
          style={{ height: cardHeight }}
          onClick={onClick}
        >
          <CardContent className="h-full flex flex-col p-4">
            {/* Header with title and actions */}
            <div className="flex justify-between items-start mb-2">
              <h3 className="text-sm text-muted-foreground font-medium tracking-wide uppercase">
                {title}
              </h3>
              
              <div className="flex items-center gap-1">
                {tooltip && (
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button variant="ghost" size="icon" className="h-6 w-6">
                        <Info className="h-4 w-4" />
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>{tooltip}</TooltipContent>
                  </Tooltip>
                )}
                
                {onRefresh && (
                  <Button 
                    variant="ghost" 
                    size="icon" 
                    className="h-6 w-6"
                    onClick={(e) => {
                      e.stopPropagation();
                      onRefresh();
                    }}
                  >
                    <RefreshCw className="h-4 w-4" />
                  </Button>
                )}
              </div>
            </div>

            {/* Main value */}
            <div className="flex-1 flex flex-col justify-center">
              {loading ? (
                <Skeleton className="h-10 w-4/5" />
              ) : (
                <div className={cn(
                  "font-semibold font-mono leading-tight",
                  size === 'small' ? 'text-lg' :
                  size === 'medium' ? 'text-xl' : 'text-2xl'
                )}>
                  {formatValue(value, format, precision)}
                </div>
              )}
            </div>

            {/* Change indicator and subtitle */}
            <div>
              {(change !== undefined || changePercent !== undefined || subtitle) && (
                <div className="flex items-center justify-between">
                  {loading ? (
                    <Skeleton className="h-5 w-3/5" />
                  ) : (
                    <>
                      {(change !== undefined || changePercent !== undefined) && (
                        <div className="flex items-center gap-1">
                          {currentTrend === 'up' && (
                            <TrendingUp className={cn("h-4 w-4", changeColor)} />
                          )}
                          {currentTrend === 'down' && (
                            <TrendingDown className={cn("h-4 w-4", changeColor)} />
                          )}
                          
                          <span className={cn(
                            "text-sm font-medium font-mono",
                            changeColor
                          )}>
                            {change !== undefined && formatChange(change, format)}
                            {change !== undefined && changePercent !== undefined && ' '}
                            {changePercent !== undefined && `(${formatChange(changePercent, 'percentage')})`}
                          </span>
                        </div>
                      )}
                      
                      {subtitle && (
                        <span className="text-xs text-muted-foreground leading-tight">
                          {subtitle}
                        </span>
                      )}
                    </>
                  )}
                </div>
              )}
            </div>
          </CardContent>

          {/* Animated border for highlighted variant */}
          {variant === 'highlighted' && (
            <div className="absolute inset-0 rounded-lg bg-gradient-to-br from-transparent via-blue-500/10 to-transparent pointer-events-none opacity-0 transition-opacity duration-300 group-hover:opacity-100" />
          )}
        </Card>
      </motion.div>
    </TooltipProvider>
  );
};

export default MetricCard;