/**
 * MetricCard component - Professional trading metric display
 * Shows financial metrics with proper formatting and color coding
 */

import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Skeleton,
  Tooltip,
  IconButton,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Info,
  Refresh,
} from '@mui/icons-material';
import { motion } from 'framer-motion';
import { tradingTheme } from '@/theme';

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
  const changeColor = tradingTheme.getPriceChangeColor(change || 0);

  const cardVariants = {
    default: {
      background: 'background.paper',
      border: 'none',
    },
    highlighted: {
      background: 'linear-gradient(135deg, rgba(64, 196, 255, 0.1), rgba(147, 51, 234, 0.1))',
      border: '1px solid rgba(64, 196, 255, 0.2)',
    },
    minimal: {
      background: 'transparent',
      border: '1px solid',
      borderColor: 'divider',
    },
  };

  return (
    <motion.div
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      transition={{ duration: 0.2 }}
    >
      <Card
        sx={{
          height: cardHeight,
          cursor: onClick ? 'pointer' : 'default',
          position: 'relative',
          overflow: 'visible',
          ...cardVariants[variant],
          '&:hover': {
            boxShadow: (theme) => 
              variant === 'highlighted' 
                ? '0 8px 32px rgba(64, 196, 255, 0.3)'
                : theme.shadows[4],
          },
        }}
        onClick={onClick}
      >
        <CardContent sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
          {/* Header with title and actions */}
          <Box 
            display="flex" 
            justifyContent="space-between" 
            alignItems="flex-start"
            mb={1}
          >
            <Typography 
              variant="body2" 
              color="text.secondary"
              sx={{ 
                fontWeight: 500,
                letterSpacing: '0.5px',
                textTransform: 'uppercase',
              }}
            >
              {title}
            </Typography>
            
            <Box display="flex" alignItems="center" gap={0.5}>
              {tooltip && (
                <Tooltip title={tooltip} arrow>
                  <IconButton size="small" sx={{ p: 0.5 }}>
                    <Info sx={{ fontSize: 16 }} />
                  </IconButton>
                </Tooltip>
              )}
              
              {onRefresh && (
                <IconButton 
                  size="small" 
                  onClick={(e) => {
                    e.stopPropagation();
                    onRefresh();
                  }}
                  sx={{ p: 0.5 }}
                >
                  <Refresh sx={{ fontSize: 16 }} />
                </IconButton>
              )}
            </Box>
          </Box>

          {/* Main value */}
          <Box flex={1} display="flex" flexDirection="column" justifyContent="center">
            {loading ? (
              <Skeleton variant="text" width="80%" height={40} />
            ) : (
              <Typography 
                variant={valueSize}
                sx={{ 
                  fontWeight: 600,
                  fontFamily: 'monospace',
                  lineHeight: 1.2,
                }}
              >
                {formatValue(value, format, precision)}
              </Typography>
            )}
          </Box>

          {/* Change indicator and subtitle */}
          <Box>
            {(change !== undefined || changePercent !== undefined || subtitle) && (
              <Box display="flex" alignItems="center" justifyContent="space-between">
                {loading ? (
                  <Skeleton variant="text" width="60%" height={20} />
                ) : (
                  <>
                    {(change !== undefined || changePercent !== undefined) && (
                      <Box display="flex" alignItems="center" gap={0.5}>
                        {currentTrend === 'up' && (
                          <TrendingUp sx={{ fontSize: 16, color: changeColor }} />
                        )}
                        {currentTrend === 'down' && (
                          <TrendingDown sx={{ fontSize: 16, color: changeColor }} />
                        )}
                        
                        <Typography 
                          variant="body2" 
                          sx={{ 
                            color: changeColor,
                            fontWeight: 500,
                            fontFamily: 'monospace',
                          }}
                        >
                          {change !== undefined && formatChange(change, format)}
                          {change !== undefined && changePercent !== undefined && ' '}
                          {changePercent !== undefined && `(${formatChange(changePercent, 'percentage')})`}
                        </Typography>
                      </Box>
                    )}
                    
                    {subtitle && (
                      <Typography 
                        variant="caption" 
                        color="text.secondary"
                        sx={{ 
                          fontWeight: 400,
                          lineHeight: 1.2,
                        }}
                      >
                        {subtitle}
                      </Typography>
                    )}
                  </>
                )}
              </Box>
            )}
          </Box>
        </CardContent>

        {/* Animated border for highlighted variant */}
        {variant === 'highlighted' && (
          <Box
            sx={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              borderRadius: 1,
              background: 'linear-gradient(45deg, transparent, rgba(64, 196, 255, 0.1), transparent)',
              pointerEvents: 'none',
              opacity: 0,
              transition: 'opacity 0.3s ease',
              '.MuiCard-root:hover &': {
                opacity: 1,
              },
            }}
          />
        )}
      </Card>
    </motion.div>
  );
};

export default MetricCard;