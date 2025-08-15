/**
 * Trading page component
 * Professional trading interface with charts, order entry, and position management
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Tabs,
  Tab,
  Button,
  IconButton,
  Tooltip,
  Switch,
  FormControlLabel,
  Alert,
  Chip,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
} from '@mui/material';
import {
  Refresh,
  Settings,
  Fullscreen,
  TrendingUp,
  Speed,
  History,
  Assessment,
  Security,
  NotificationsActive,
} from '@mui/icons-material';
import { motion } from 'framer-motion';
import { useAppSelector, useAppDispatch } from '@/store';
import { colors } from '@/theme/colors';

// Import our trading components
import PriceChart from '@/components/Trading/PriceChart';
import OrderForm, { OrderData } from '@/components/Trading/OrderForm';
import OrderBook, { generateMockOrderBookData } from '@/components/Trading/OrderBook';
import PositionCard, { Position } from '@/components/Trading/PositionCard';
import MetricCard from '@/components/Trading/MetricCard';

// Mock data for development
const mockSymbols = [
  { symbol: 'BTC/USD', price: 45000, change: 2.34, volume: 125000000 },
  { symbol: 'ETH/USD', price: 2850, change: -1.23, volume: 95000000 },
  { symbol: 'ADA/USD', price: 0.52, change: 5.67, volume: 45000000 },
  { symbol: 'SOL/USD', price: 98.45, change: 3.21, volume: 32000000 },
  { symbol: 'MATIC/USD', price: 0.89, change: -0.45, volume: 28000000 },
];

const mockPriceData = Array.from({ length: 100 }, (_, i) => {
  const basePrice = 45000;
  const timestamp = Date.now() - (100 - i) * 60000;
  const noise = (Math.random() - 0.5) * 1000;
  const trend = Math.sin(i / 10) * 500;
  const price = basePrice + trend + noise;
  
  return {
    timestamp,
    price,
    close: price,
    open: price - (Math.random() - 0.5) * 100,
    high: price + Math.random() * 200,
    low: price - Math.random() * 200,
    volume: Math.random() * 1000000,
  };
});

const mockPositions: Position[] = [
  {
    id: '1',
    symbol: 'BTC/USD',
    side: 'long',
    size: 0.5,
    entryPrice: 44200,
    currentPrice: 45100,
    stopLoss: 43000,
    takeProfit: 47000,
    unrealizedPnL: 450,
    unrealizedPnLPercent: 2.04,
    margin: 2210,
    leverage: 10,
    timestamp: Date.now() - 3600000,
    strategy: 'Manual',
    exchange: 'Binance',
    status: 'open',
  },
];

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel: React.FC<TabPanelProps> = ({ children, value, index }) => {
  return (
    <div role="tabpanel" hidden={value !== index}>
      {value === index && <Box>{children}</Box>}
    </div>
  );
};

const TradingPage: React.FC = () => {
  const [selectedSymbol, setSelectedSymbol] = useState('BTC/USD');
  const [chartTimeframe, setChartTimeframe] = useState('1h');
  const [chartType, setChartType] = useState<'candlestick' | 'line' | 'area'>('line');
  const [rightPanelTab, setRightPanelTab] = useState(0);
  const [bottomPanelTab, setBottomPanelTab] = useState(0);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [loading, setLoading] = useState(false);
  const [fullscreenChart, setFullscreenChart] = useState(false);
  const [orderBookData, setOrderBookData] = useState(() => 
    generateMockOrderBookData(mockSymbols[0].price)
  );

  // Find current symbol data
  const currentSymbolData = useMemo(() => 
    mockSymbols.find(s => s.symbol === selectedSymbol) || mockSymbols[0],
    [selectedSymbol]
  );

  // Real-time data simulation
  useEffect(() => {
    if (!autoRefresh) return;
    
    const interval = setInterval(() => {
      // Update order book data
      setOrderBookData(generateMockOrderBookData(currentSymbolData.price));
    }, 1000);
    
    return () => clearInterval(interval);
  }, [autoRefresh, currentSymbolData.price]);

  const handleSymbolChange = useCallback((symbol: string) => {
    setSelectedSymbol(symbol);
    setOrderBookData(generateMockOrderBookData(
      mockSymbols.find(s => s.symbol === symbol)?.price || 45000
    ));
  }, []);

  const handleRefresh = useCallback(async () => {
    setLoading(true);
    await new Promise(resolve => setTimeout(resolve, 1000));
    setOrderBookData(generateMockOrderBookData(currentSymbolData.price));
    setLoading(false);
  }, [currentSymbolData.price]);

  const handleOrderSubmit = useCallback(async (orderData: OrderData) => {
    console.log('Submitting order:', orderData);
    // Simulate order submission
    await new Promise(resolve => setTimeout(resolve, 2000));
    // In real app, this would call the trading API
  }, []);

  const handlePriceClick = useCallback((price: number) => {
    console.log('Price clicked:', price);
    // In real app, this would populate the order form with the clicked price
  }, []);

  const handleTimeframeChange = useCallback((timeframe: string) => {
    setChartTimeframe(timeframe);
    // In real app, this would fetch new chart data
  }, []);

  const handleChartTypeChange = useCallback((type: 'candlestick' | 'line' | 'area') => {
    setChartType(type);
  }, []);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
    >
      <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {/* Header */}
        <Box 
          p={2} 
          display="flex" 
          justifyContent="space-between" 
          alignItems="center"
          borderBottom="1px solid"
          borderColor="divider"
          sx={{ backgroundColor: 'background.paper' }}
        >
          <Box display="flex" alignItems="center" gap={3}>
            <Typography variant="h5" sx={{ fontWeight: 600 }}>
              Trading
            </Typography>
            
            {/* Symbol Selector */}
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <InputLabel>Symbol</InputLabel>
              <Select
                value={selectedSymbol}
                onChange={(e) => handleSymbolChange(e.target.value)}
                label="Symbol"
              >
                {mockSymbols.map((symbol) => (\n                  <MenuItem key={symbol.symbol} value={symbol.symbol}>\n                    <Box display=\"flex\" alignItems=\"center\" justifyContent=\"space-between\" width=\"100%\">\n                      <Typography>{symbol.symbol}</Typography>\n                      <Chip \n                        label={`${symbol.change >= 0 ? '+' : ''}${symbol.change.toFixed(2)}%`}\n                        size=\"small\"\n                        sx={{\n                          ml: 1,\n                          backgroundColor: symbol.change >= 0 ? colors.financial.profit : colors.financial.loss,\n                          color: 'white',\n                          fontSize: '0.7rem',\n                        }}\n                      />\n                    </Box>\n                  </MenuItem>\n                ))}\n              </Select>\n            </FormControl>\n            \n            {/* Current Price Display */}\n            <Box>\n              <Typography variant=\"h4\" sx={{ fontFamily: 'monospace', fontWeight: 600 }}>\n                ${currentSymbolData.price.toLocaleString()}\n              </Typography>\n              <Typography \n                variant=\"body2\" \n                sx={{ \n                  color: currentSymbolData.change >= 0 ? colors.financial.profit : colors.financial.loss,\n                  fontFamily: 'monospace',\n                }}\n              >\n                {currentSymbolData.change >= 0 ? '+' : ''}{currentSymbolData.change.toFixed(2)}%\n              </Typography>\n            </Box>\n          </Box>\n          \n          <Box display=\"flex\" alignItems=\"center\" gap={2}>\n            <FormControlLabel\n              control={\n                <Switch\n                  checked={autoRefresh}\n                  onChange={(e) => setAutoRefresh(e.target.checked)}\n                  size=\"small\"\n                />\n              }\n              label=\"Real-time\"\n            />\n            \n            <Tooltip title=\"Refresh data\">\n              <IconButton onClick={handleRefresh} disabled={loading}>\n                <Refresh sx={{ animation: loading ? 'spin 1s linear infinite' : 'none' }} />\n              </IconButton>\n            </Tooltip>\n            \n            <Tooltip title=\"Trading settings\">\n              <IconButton>\n                <Settings />\n              </IconButton>\n            </Tooltip>\n            \n            <Tooltip title=\"Trading alerts\">\n              <IconButton>\n                <NotificationsActive />\n              </IconButton>\n            </Tooltip>\n          </Box>\n        </Box>\n\n        {/* Main Trading Interface */}\n        <Box flex={1} display=\"flex\" overflow=\"hidden\">\n          {/* Left Panel - Chart */}\n          <Box flex={1} display=\"flex\" flexDirection=\"column\">\n            {/* Chart Area */}\n            <Box flex={1} p={2}>\n              <PriceChart\n                data={mockPriceData}\n                symbol={selectedSymbol}\n                timeframe={chartTimeframe}\n                chartType={chartType}\n                height={fullscreenChart ? window.innerHeight - 200 : 600}\n                loading={loading}\n                onRefresh={handleRefresh}\n                onTimeframeChange={handleTimeframeChange}\n                onChartTypeChange={handleChartTypeChange}\n                fullscreen={fullscreenChart}\n                onFullscreenToggle={() => setFullscreenChart(!fullscreenChart)}\n              />\n            </Box>\n            \n            {/* Bottom Panel - Positions & Orders */}\n            {!fullscreenChart && (\n              <Paper sx={{ borderRadius: 0, borderTop: '1px solid', borderColor: 'divider' }}>\n                <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>\n                  <Tabs \n                    value={bottomPanelTab} \n                    onChange={(_, newValue) => setBottomPanelTab(newValue)}\n                    sx={{ px: 2 }}\n                  >\n                    <Tab icon={<TrendingUp />} label=\"Positions\" />\n                    <Tab icon={<Speed />} label=\"Open Orders\" />\n                    <Tab icon={<History />} label=\"Order History\" />\n                    <Tab icon={<Assessment />} label=\"Trade History\" />\n                  </Tabs>\n                </Box>\n                \n                <Box sx={{ height: 300, overflow: 'auto' }}>\n                  <TabPanel value={bottomPanelTab} index={0}>\n                    <Box p={2}>\n                      {mockPositions.length > 0 ? (\n                        <Grid container spacing={2}>\n                          {mockPositions.map((position) => (\n                            <Grid item xs={12} md={6} lg={4} key={position.id}>\n                              <PositionCard\n                                position={position}\n                                compact\n                                onClose={(id) => console.log('Close position:', id)}\n                                onEdit={(id) => console.log('Edit position:', id)}\n                              />\n                            </Grid>\n                          ))}\n                        </Grid>\n                      ) : (\n                        <Box \n                          display=\"flex\" \n                          flexDirection=\"column\" \n                          alignItems=\"center\" \n                          justifyContent=\"center\" \n                          minHeight={200}\n                        >\n                          <TrendingUp sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />\n                          <Typography variant=\"h6\" color=\"text.secondary\" gutterBottom>\n                            No Open Positions\n                          </Typography>\n                          <Typography variant=\"body2\" color=\"text.secondary\">\n                            Place an order to start trading\n                          </Typography>\n                        </Box>\n                      )}\n                    </Box>\n                  </TabPanel>\n                  \n                  <TabPanel value={bottomPanelTab} index={1}>\n                    <Box p={2}>\n                      <Typography color=\"text.secondary\">\n                        Open orders will be displayed here\n                      </Typography>\n                    </Box>\n                  </TabPanel>\n                  \n                  <TabPanel value={bottomPanelTab} index={2}>\n                    <Box p={2}>\n                      <Typography color=\"text.secondary\">\n                        Order history will be displayed here\n                      </Typography>\n                    </Box>\n                  </TabPanel>\n                  \n                  <TabPanel value={bottomPanelTab} index={3}>\n                    <Box p={2}>\n                      <Typography color=\"text.secondary\">\n                        Trade history will be displayed here\n                      </Typography>\n                    </Box>\n                  </TabPanel>\n                </Box>\n              </Paper>\n            )}\n          </Box>\n          \n          {/* Right Panel - Order Book & Order Form */}\n          {!fullscreenChart && (\n            <Box \n              width={400} \n              display=\"flex\" \n              flexDirection=\"column\"\n              borderLeft=\"1px solid\"\n              borderColor=\"divider\"\n            >\n              {/* Right Panel Tabs */}\n              <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>\n                <Tabs \n                  value={rightPanelTab} \n                  onChange={(_, newValue) => setRightPanelTab(newValue)}\n                  variant=\"fullWidth\"\n                >\n                  <Tab icon={<Speed />} label=\"Order\" />\n                  <Tab icon={<Assessment />} label=\"Book\" />\n                </Tabs>\n              </Box>\n              \n              <Box flex={1} overflow=\"auto\">\n                <TabPanel value={rightPanelTab} index={0}>\n                  <Box p={2}>\n                    <OrderForm\n                      symbol={selectedSymbol}\n                      currentPrice={currentSymbolData.price}\n                      availableBalance={5000}\n                      maxLeverage={100}\n                      onSubmit={handleOrderSubmit}\n                      loading={loading}\n                    />\n                  </Box>\n                </TabPanel>\n                \n                <TabPanel value={rightPanelTab} index={1}>\n                  <OrderBook\n                    data={orderBookData}\n                    symbol={selectedSymbol}\n                    currentPrice={currentSymbolData.price}\n                    onPriceClick={handlePriceClick}\n                    onRefresh={handleRefresh}\n                    loading={loading}\n                    precision={2}\n                    maxRows={20}\n                    compact\n                  />\n                </TabPanel>\n              </Box>\n            </Box>\n          )}\n        </Box>\n      </Box>\n    </motion.div>\n  );\n};\n\nexport default TradingPage;"