/**
 * BotCard component - Trading bot display and control interface
 * Shows bot status, performance, and management controls
 */

import React, { useState, useMemo } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  IconButton,
  Button,
  Switch,
  FormControlLabel,
  Tooltip,
  Alert,
  Divider,
  LinearProgress,
  Menu,
  MenuItem,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
} from '@mui/material';
import {
  PlayArrow,
  Pause,
  Stop,
  Edit,
  Delete,
  Settings,
  Assessment,
  MoreVert,
  TrendingUp,
  TrendingDown,
  Security,
  Speed,
  Timeline,
  Warning,
} from '@mui/icons-material';
import { motion } from 'framer-motion';
import { tradingTheme } from '@/theme';
import { colors } from '@/theme/colors';

export interface Bot {
  id: string;
  name: string;
  strategy: string;
  status: 'running' | 'paused' | 'stopped' | 'error';
  exchange: string;
  symbol: string;
  balance: number;
  pnl: number;
  pnlPercent: number;
  trades: number;
  winRate: number;
  created: number;
  lastActivity: number;
  riskLevel: 'low' | 'medium' | 'high';
  leverage: number;
  maxDrawdown: number;
  sharpeRatio: number;
  description?: string;
  version?: string;
  isActive: boolean;
}

interface BotCardProps {
  bot: Bot;
  onStart?: (botId: string) => void;
  onPause?: (botId: string) => void;
  onStop?: (botId: string) => void;
  onEdit?: (botId: string) => void;
  onDelete?: (botId: string) => void;
  onToggleActive?: (botId: string, active: boolean) => void;
  onViewDetails?: (botId: string) => void;
  compact?: boolean;
  showActions?: boolean;
}

const formatCurrency = (value: number): string => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
};

const formatPercentage = (value: number): string => {
  const prefix = value >= 0 ? '+' : '';
  return `${prefix}${value.toFixed(2)}%`;
};

const getStatusColor = (status: string) => {
  switch (status) {
    case 'running':
      return colors.status.online;
    case 'paused':
      return colors.status.warning;
    case 'stopped':
      return colors.status.offline;
    case 'error':
      return colors.status.error;
    default:
      return colors.status.info;
  }
};

const getRiskColor = (risk: string) => {
  switch (risk) {
    case 'low':
      return colors.financial.profit;
    case 'medium':
      return colors.financial.warning;
    case 'high':
      return colors.financial.loss;
    default:
      return colors.status.info;
  }
};

export const BotCard: React.FC<BotCardProps> = ({
  bot,
  onStart,
  onPause,
  onStop,
  onEdit,
  onDelete,
  onToggleActive,
  onViewDetails,
  compact = false,
  showActions = true,
}) => {
  const [menuAnchor, setMenuAnchor] = useState<null | HTMLElement>(null);
  const [deleteDialog, setDeleteDialog] = useState(false);
  const [editDialog, setEditDialog] = useState(false);
  const [editName, setEditName] = useState(bot.name);

  const {
    id,
    name,
    strategy,
    status,
    exchange,
    symbol,
    balance,
    pnl,
    pnlPercent,
    trades,
    winRate,
    created,
    lastActivity,
    riskLevel,
    leverage,
    maxDrawdown,
    sharpeRatio,
    description,
    version,
    isActive,
  } = bot;

  const pnlColor = tradingTheme.getPriceChangeColor(pnl);
  const statusColor = getStatusColor(status);
  const riskColor = getRiskColor(riskLevel);

  const isRunning = status === 'running';
  const isPaused = status === 'paused';
  const isStopped = status === 'stopped';
  const hasError = status === 'error';

  const timeSinceLastActivity = useMemo(() => {
    const diff = Date.now() - lastActivity;
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);

    if (days > 0) return `${days}d ago`;
    if (hours > 0) return `${hours}h ago`;
    if (minutes > 0) return `${minutes}m ago`;
    return 'Just now';
  }, [lastActivity]);

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setMenuAnchor(event.currentTarget);
  };

  const handleMenuClose = () => {
    setMenuAnchor(null);
  };

  const handleStart = () => {
    if (onStart) onStart(id);
  };

  const handlePause = () => {
    if (onPause) onPause(id);
  };

  const handleStop = () => {
    if (onStop) onStop(id);
  };

  const handleEdit = () => {
    setEditDialog(true);
    handleMenuClose();
  };

  const handleSaveEdit = () => {
    if (onEdit && editName.trim()) {
      onEdit(id);
      setEditDialog(false);
    }
  };

  const handleDelete = () => {
    if (onDelete) onDelete(id);
    setDeleteDialog(false);
    handleMenuClose();
  };

  const handleToggleActive = (checked: boolean) => {
    if (onToggleActive) onToggleActive(id, checked);
  };

  const handleViewDetails = () => {
    if (onViewDetails) onViewDetails(id);
  };

  return (
    <>
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.95 }}
        transition={{ duration: 0.2 }}
        whileHover={{ scale: 1.02 }}
      >
        <Card
          sx={{
            position: 'relative',
            backgroundColor: 'background.paper',
            border: '1px solid',
            borderColor: hasError ? 'error.main' : 'divider',
            borderLeftWidth: '4px',
            borderLeftColor: statusColor,
            opacity: isActive ? 1 : 0.7,
            '&:hover': {
              boxShadow: 3,
            },
          }}
        >
          <CardContent sx={{ p: compact ? 2 : 3 }}>
            {/* Header */}
            <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={2}>
              <Box flex={1}>
                <Box display="flex" alignItems="center" gap={1} mb={1}>
                  <Typography 
                    variant={compact ? "h6" : "h5"} 
                    sx={{ fontWeight: 600, cursor: 'pointer' }}
                    onClick={handleViewDetails}
                  >
                    {name}
                  </Typography>
                  
                  <Chip
                    label={status.toUpperCase()}
                    size="small"
                    sx={{
                      backgroundColor: statusColor,
                      color: 'white',
                      fontWeight: 600,
                      minWidth: 60,
                    }}
                  />
                  
                  <Chip
                    label={riskLevel.toUpperCase()}
                    size="small"
                    variant="outlined"
                    sx={{
                      borderColor: riskColor,
                      color: riskColor,
                    }}
                  />
                </Box>

                <Typography variant="body2" color="text.secondary" gutterBottom>
                  {strategy} • {exchange} • {symbol}
                </Typography>

                {version && (
                  <Typography variant="caption" color="text.secondary">
                    v{version}
                  </Typography>
                )}
              </Box>

              {showActions && (
                <Box display="flex" alignItems="center" gap={0.5}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={isActive}
                        onChange={(e) => handleToggleActive(e.target.checked)}
                        size="small"
                      />
                    }
                    label=""
                    sx={{ m: 0 }}
                  />
                  
                  <IconButton size="small" onClick={handleMenuOpen}>
                    <MoreVert />
                  </IconButton>
                </Box>
              )}
            </Box>

            {/* Error Alert */}
            {hasError && (
              <Alert severity="error" sx={{ mb: 2, fontSize: '0.75rem' }}>
                Bot encountered an error. Check logs for details.
              </Alert>
            )}

            {/* Performance Metrics */}
            <Box display="flex" flexDirection={compact ? 'column' : 'row'} gap={2} mb={2}>
              {/* Left Column */}
              <Box flex={1}>
                <Box display="flex" justifyContent="space-between" mb={1}>
                  <Typography variant="body2" color="text.secondary">
                    Balance
                  </Typography>
                  <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                    {formatCurrency(balance)}
                  </Typography>
                </Box>

                <Box display="flex" justifyContent="space-between" mb={1}>
                  <Typography variant="body2" color="text.secondary">
                    P&L
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
                      {formatCurrency(pnl)}
                    </Typography>
                    <Typography 
                      variant="caption" 
                      sx={{ 
                        color: pnlColor,
                        fontFamily: 'monospace',
                      }}
                    >
                      {formatPercentage(pnlPercent)}
                    </Typography>
                  </Box>
                </Box>

                <Box display="flex" justifyContent="space-between" mb={1}>
                  <Typography variant="body2" color="text.secondary">
                    Trades
                  </Typography>
                  <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                    {trades.toLocaleString()}
                  </Typography>
                </Box>
              </Box>

              {/* Right Column */}
              {!compact && (
                <Box flex={1}>
                  <Box display="flex" justifyContent="space-between" mb={1}>
                    <Typography variant="body2" color="text.secondary">
                      Win Rate
                    </Typography>
                    <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                      {winRate.toFixed(1)}%
                    </Typography>
                  </Box>

                  <Box display="flex" justifyContent="space-between" mb={1}>
                    <Typography variant="body2" color="text.secondary">
                      Leverage
                    </Typography>
                    <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                      {leverage}x
                    </Typography>
                  </Box>

                  <Box display="flex" justifyContent="space-between" mb={1}>
                    <Typography variant="body2" color="text.secondary">
                      Sharpe Ratio
                    </Typography>
                    <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                      {sharpeRatio.toFixed(2)}
                    </Typography>
                  </Box>
                </Box>
              )}
            </Box>

            {/* Activity Indicator */}
            <Box mb={2}>
              <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                <Typography variant="caption" color="text.secondary">
                  Last Activity: {timeSinceLastActivity}
                </Typography>
                {isRunning && (
                  <Box display="flex" alignItems="center" gap={0.5}>
                    <Box
                      sx={{
                        width: 8,
                        height: 8,
                        borderRadius: '50%',
                        backgroundColor: statusColor,
                        animation: 'pulse 2s infinite',
                      }}
                    />
                    <Typography variant="caption" color="text.secondary">
                      Active
                    </Typography>
                  </Box>
                )}
              </Box>

              {isRunning && (
                <LinearProgress 
                  variant="indeterminate" 
                  sx={{ 
                    height: 2,
                    backgroundColor: 'rgba(255, 255, 255, 0.1)',
                    '& .MuiLinearProgress-bar': {
                      backgroundColor: statusColor,
                    },
                  }} 
                />
              )}
            </Box>

            {/* Control Buttons */}
            {showActions && (
              <Box display="flex" gap={1}>
                {isStopped && (
                  <Button
                    size="small"
                    variant="contained"
                    color="success"
                    startIcon={<PlayArrow />}
                    onClick={handleStart}
                    disabled={!isActive}
                  >
                    Start
                  </Button>
                )}

                {isRunning && (
                  <>
                    <Button
                      size="small"
                      variant="outlined"
                      color="warning"
                      startIcon={<Pause />}
                      onClick={handlePause}
                    >
                      Pause
                    </Button>
                    <Button
                      size="small"
                      variant="outlined"
                      color="error"
                      startIcon={<Stop />}
                      onClick={handleStop}
                    >
                      Stop
                    </Button>
                  </>
                )}

                {isPaused && (
                  <>
                    <Button
                      size="small"
                      variant="contained"
                      color="success"
                      startIcon={<PlayArrow />}
                      onClick={handleStart}
                      disabled={!isActive}
                    >
                      Resume
                    </Button>
                    <Button
                      size="small"
                      variant="outlined"
                      color="error"
                      startIcon={<Stop />}
                      onClick={handleStop}
                    >
                      Stop
                    </Button>
                  </>
                )}

                <Button
                  size="small"
                  variant="outlined"
                  startIcon={<Assessment />}
                  onClick={handleViewDetails}
                >
                  Details
                </Button>
              </Box>
            )}
          </CardContent>
        </Card>
      </motion.div>

      {/* Context Menu */}
      <Menu
        anchorEl={menuAnchor}
        open={Boolean(menuAnchor)}
        onClose={handleMenuClose}
      >
        <MenuItem onClick={handleEdit}>
          <Edit sx={{ mr: 1 }} />
          Edit
        </MenuItem>
        <MenuItem onClick={() => setDeleteDialog(true)}>
          <Delete sx={{ mr: 1 }} />
          Delete
        </MenuItem>
        <MenuItem onClick={handleViewDetails}>
          <Assessment sx={{ mr: 1 }} />
          View Details
        </MenuItem>
      </Menu>

      {/* Edit Dialog */}
      <Dialog open={editDialog} onClose={() => setEditDialog(false)}>
        <DialogTitle>Edit Bot</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Bot Name"
            fullWidth
            variant="outlined"
            value={editName}
            onChange={(e) => setEditName(e.target.value)}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditDialog(false)}>Cancel</Button>
          <Button onClick={handleSaveEdit} variant="contained">
            Save
          </Button>
        </DialogActions>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteDialog} onClose={() => setDeleteDialog(false)}>
        <DialogTitle>Delete Bot</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete the bot "{name}"? This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialog(false)}>Cancel</Button>
          <Button onClick={handleDelete} color="error" variant="contained">
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default BotCard;