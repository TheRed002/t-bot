/**
 * Trading page component
 * Professional trading interface with charts, order entry, and position management
 * Built with Shadcn/ui components and resizable panels
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { motion } from 'framer-motion';
import { 
  RefreshCw as Refresh, 
  Settings, 
  Fullscreen, 
  TrendingUp, 
  Activity, 
  History, 
  BarChart3, 
  DollarSign,
  Bell,
  ChevronDown,
  Plus,
  Minus,
  X,
  Eye,
  Wifi,
  WifiOff
} from 'lucide-react';
import { useAppSelector, useAppDispatch } from '@/store';
import { websocketService } from '@/services/websocket';

// Shadcn/ui components
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { PanelGroup, Panel, ResizeHandle } from '@/components/ui/resizable-panels';

// Utils
import { cn, formatPrice, formatPnL, formatPercent, getPnLColor } from '@/lib/utils';

// Trading types
interface TradingSymbol {
  symbol: string;
  price: number;
  change: number;
  volume: number;
  high24h: number;
  low24h: number;
}

interface OrderBookEntry {
  price: number;
  quantity: number;
  total: number;
}

interface OrderBookData {
  bids: OrderBookEntry[];
  asks: OrderBookEntry[];
}

interface Position {
  id: string;
  symbol: string;
  side: 'long' | 'short';
  amount: number;
  entryPrice: number;
  currentPrice: number;
  pnl: number;
  pnlPercent: number;
  margin?: number;
  actions?: string[];
}

interface OpenOrder {
  id: string;
  symbol: string;
  type: string;
  side: 'buy' | 'sell';
  amount: number;
  price: number;
  filled: number;
  status: string;
  time: string;
}

interface Trade {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  amount: number;
  price: number;
  time: string;
  fee: number;
}

// Mock data for development
const mockSymbols: TradingSymbol[] = [
  { symbol: 'BTC/USDT', price: 67450.32, change: 2.34, volume: 125000000, high24h: 68200, low24h: 65800 },
  { symbol: 'ETH/USDT', price: 3425.67, change: -1.23, volume: 95000000, high24h: 3480, low24h: 3350 },
  { symbol: 'BNB/USDT', price: 342.15, change: 1.89, volume: 45000000, high24h: 350, low24h: 335 },
  { symbol: 'SOL/USDT', price: 176.32, change: 3.21, volume: 32000000, high24h: 182, low24h: 168 },
  { symbol: 'ADA/USDT', price: 0.6745, change: -0.67, volume: 28000000, high24h: 0.69, low24h: 0.65 },
];

// Generate mock order book data
const generateOrderBookData = (currentPrice: number): OrderBookData => {
  const bids: OrderBookEntry[] = [];
  const asks: OrderBookEntry[] = [];
  
  // Generate bids (buy orders) below current price
  for (let i = 0; i < 15; i++) {
    const price = currentPrice - (i + 1) * (currentPrice * 0.0001);
    const quantity = Math.random() * 5 + 0.1;
    bids.push({
      price,
      quantity,
      total: quantity * price
    });
  }
  
  // Generate asks (sell orders) above current price
  for (let i = 0; i < 15; i++) {
    const price = currentPrice + (i + 1) * (currentPrice * 0.0001);
    const quantity = Math.random() * 5 + 0.1;
    asks.push({
      price,
      quantity,
      total: quantity * price
    });
  }
  
  return { bids, asks };
};

const mockPositions: Position[] = [
  {
    id: '1',
    symbol: 'BTC/USDT',
    side: 'long',
    amount: 0.25,
    entryPrice: 66800,
    currentPrice: 67450.32,
    pnl: 162.58,
    pnlPercent: 0.97,
    margin: 3340,
  },
  {
    id: '2',
    symbol: 'ETH/USDT',
    side: 'short',
    amount: 2.5,
    entryPrice: 3480,
    currentPrice: 3425.67,
    pnl: 135.83,
    pnlPercent: 1.56,
    margin: 1712.84,
  }
];

const mockOpenOrders: OpenOrder[] = [
  {
    id: '1',
    symbol: 'BTC/USDT',
    type: 'Limit',
    side: 'buy',
    amount: 0.1,
    price: 66500,
    filled: 0,
    status: 'Open',
    time: '2024-01-15 10:30:15'
  },
  {
    id: '2',
    symbol: 'ETH/USDT',
    type: 'Stop-Limit',
    side: 'sell',
    amount: 1.0,
    price: 3400,
    filled: 0,
    status: 'Open',
    time: '2024-01-15 09:45:22'
  }
];

const mockTrades: Trade[] = [
  {
    id: '1',
    symbol: 'BTC/USDT',
    side: 'buy',
    amount: 0.15,
    price: 67320.45,
    time: '2024-01-15 14:22:18',
    fee: 1.01
  },
  {
    id: '2',
    symbol: 'ETH/USDT',
    side: 'sell',
    amount: 0.8,
    price: 3445.67,
    time: '2024-01-15 14:18:05',
    fee: 0.69
  },
  {
    id: '3',
    symbol: 'SOL/USDT',
    side: 'buy',
    amount: 5.2,
    price: 175.23,
    time: '2024-01-15 14:15:32',
    fee: 0.46
  }
];

// Order form component
const OrderForm: React.FC<{ symbol: string; currentPrice: number }> = ({ symbol, currentPrice }) => {
  const [orderType, setOrderType] = useState('market');
  const [side, setSide] = useState<'buy' | 'sell'>('buy');
  const [amount, setAmount] = useState('');
  const [price, setPrice] = useState('');
  const [total, setTotal] = useState('');

  const percentageButtons = [25, 50, 75, 100];

  const handlePercentageClick = (percentage: number) => {
    // Mock available balance
    const availableBalance = 10000; // USDT
    const calculatedAmount = (availableBalance * percentage / 100) / currentPrice;
    setAmount(calculatedAmount.toFixed(6));
    setTotal((calculatedAmount * currentPrice).toFixed(2));
  };

  return (
    <Card className="h-full">
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">Order Entry</CardTitle>
          <Badge variant="outline">{symbol}</Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Order Type Selector */}
        <Tabs value={orderType} onValueChange={setOrderType}>
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="market">Market</TabsTrigger>
            <TabsTrigger value="limit">Limit</TabsTrigger>
            <TabsTrigger value="stop-limit">Stop-Limit</TabsTrigger>
          </TabsList>
        </Tabs>

        {/* Buy/Sell Toggle */}
        <div className="grid grid-cols-2 gap-2">
          <Button 
            variant={side === 'buy' ? 'default' : 'outline'}
            className={cn('h-12', side === 'buy' && 'bg-green-600 hover:bg-green-700')}
            onClick={() => setSide('buy')}
          >
            Buy
          </Button>
          <Button 
            variant={side === 'sell' ? 'default' : 'outline'}
            className={cn('h-12', side === 'sell' && 'bg-red-600 hover:bg-red-700')}
            onClick={() => setSide('sell')}
          >
            Sell
          </Button>
        </div>

        {/* Price Input (for limit orders) */}
        {orderType !== 'market' && (
          <div className="space-y-2">
            <label className="text-sm font-medium">Price (USDT)</label>
            <input 
              className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
              placeholder={currentPrice.toString()}
              value={price}
              onChange={(e) => setPrice(e.target.value)}
            />
          </div>
        )}

        {/* Amount Input */}
        <div className="space-y-2">
          <label className="text-sm font-medium">Amount ({symbol.split('/')[0]})</label>
          <input 
            className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
            placeholder="0.00000000"
            value={amount}
            onChange={(e) => setAmount(e.target.value)}
          />
        </div>

        {/* Percentage Buttons */}
        <div className="grid grid-cols-4 gap-2">
          {percentageButtons.map((percentage) => (
            <Button 
              key={percentage}
              variant="outline" 
              size="sm"
              onClick={() => handlePercentageClick(percentage)}
            >
              {percentage}%
            </Button>
          ))}
        </div>

        {/* Total */}
        <div className="space-y-2">
          <label className="text-sm font-medium">Total (USDT)</label>
          <input 
            className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
            placeholder="0.00"
            value={total}
            onChange={(e) => setTotal(e.target.value)}
            readOnly={orderType === 'market'}
          />
        </div>

        {/* Submit Button */}
        <Button 
          className={cn(
            'w-full h-12 text-lg font-semibold',
            side === 'buy' 
              ? 'bg-green-600 hover:bg-green-700' 
              : 'bg-red-600 hover:bg-red-700'
          )}
        >
          {side === 'buy' ? 'Buy' : 'Sell'} {symbol.split('/')[0]}
        </Button>

        {/* Available Balance */}
        <div className="pt-2 border-t">
          <div className="flex justify-between text-sm text-muted-foreground">
            <span>Available:</span>
            <span>10,000.00 USDT</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

// Order Book Component
const OrderBookComponent: React.FC<{ data: OrderBookData; currentPrice: number }> = ({ data, currentPrice }) => {
  const maxTotal = Math.max(
    ...data.bids.map(b => b.total),
    ...data.asks.map(a => a.total)
  );

  return (
    <Card className="h-full">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">Order Book</CardTitle>
          <Button variant="ghost" size="icon">
            <Refresh className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>
      <CardContent className="p-0">
        {/* Header */}
        <div className="grid grid-cols-3 gap-4 px-4 pb-2 text-xs font-medium text-muted-foreground border-b">
          <span className="text-left">Price (USDT)</span>
          <span className="text-right">Amount</span>
          <span className="text-right">Total</span>
        </div>
        
        {/* Asks (Sell orders) */}
        <div className="max-h-48 overflow-auto">
          {data.asks.slice().reverse().map((ask, index) => (
            <div key={index} className="relative">
              <div 
                className="absolute inset-y-0 right-0 bg-red-500/10"
                style={{ width: `${(ask.total / maxTotal) * 100}%` }}
              />
              <div className="relative grid grid-cols-3 gap-4 px-4 py-1 text-xs hover:bg-muted/50 cursor-pointer">
                <span className="text-red-400 font-mono">{ask.price.toFixed(2)}</span>
                <span className="text-right font-mono">{ask.quantity.toFixed(4)}</span>
                <span className="text-right font-mono">{ask.total.toFixed(2)}</span>
              </div>
            </div>
          ))}
        </div>
        
        {/* Current Price */}
        <div className="flex items-center justify-between px-4 py-3 bg-muted/30 border-y">
          <span className="text-sm font-medium">Current Price</span>
          <span className="text-lg font-bold font-mono text-green-400">
            {formatPrice(currentPrice)}
          </span>
        </div>
        
        {/* Bids (Buy orders) */}
        <div className="max-h-48 overflow-auto">
          {data.bids.map((bid, index) => (
            <div key={index} className="relative">
              <div 
                className="absolute inset-y-0 right-0 bg-green-500/10"
                style={{ width: `${(bid.total / maxTotal) * 100}%` }}
              />
              <div className="relative grid grid-cols-3 gap-4 px-4 py-1 text-xs hover:bg-muted/50 cursor-pointer">
                <span className="text-green-400 font-mono">{bid.price.toFixed(2)}</span>
                <span className="text-right font-mono">{bid.quantity.toFixed(4)}</span>
                <span className="text-right font-mono">{bid.total.toFixed(2)}</span>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};

// Price Chart Placeholder
const PriceChart: React.FC<{ symbol: string; price: number }> = ({ symbol, price }) => {
  const [timeframe, setTimeframe] = useState('1h');
  const timeframes = ['1m', '5m', '15m', '1h', '4h', '1d'];

  return (
    <Card className="h-full">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-xl">{symbol}</CardTitle>
            <p className="text-3xl font-bold font-mono text-green-400">
              {formatPrice(price)}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Select value={timeframe} onValueChange={setTimeframe}>
              <SelectTrigger className="w-20">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {timeframes.map(tf => (
                  <SelectItem key={tf} value={tf}>{tf}</SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button variant="ghost" size="icon">
              <Fullscreen className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="h-96">
        <div className="w-full h-full bg-muted/10 rounded-lg flex items-center justify-center">
          <div className="text-center text-muted-foreground">
            <BarChart3 className="h-16 w-16 mx-auto mb-4" />
            <p>Chart will be rendered here</p>
            <p className="text-sm">TradingView, Chart.js, or Recharts integration</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

const TradingPage: React.FC = () => {
  const [selectedSymbol, setSelectedSymbol] = useState('BTC/USDT');
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [loading, setLoading] = useState(false);
  const [orderBookData, setOrderBookData] = useState<OrderBookData>(() => 
    generateOrderBookData(mockSymbols[0].price)
  );
  const [positions] = useState<Position[]>(mockPositions);
  const [openOrders] = useState<OpenOrder[]>(mockOpenOrders);
  const [trades] = useState<Trade[]>(mockTrades);

  // WebSocket connection status
  const [isConnected, setIsConnected] = useState(false);

  // Find current symbol data
  const currentSymbolData = useMemo(() => 
    mockSymbols.find(s => s.symbol === selectedSymbol) || mockSymbols[0],
    [selectedSymbol]
  );

  // WebSocket connection management
  useEffect(() => {
    const token = localStorage.getItem('token');
    if (token && !websocketService.isConnected()) {
      websocketService.connect(token);
    }

    // Subscribe to market data for selected symbol
    if (websocketService.isConnected()) {
      websocketService.subscribe([
        'market_data',
        'order_book_update',
        'position_update',
        'order_update'
      ], { symbols: [selectedSymbol] });
    }

    const checkConnection = setInterval(() => {
      setIsConnected(websocketService.isConnected());
    }, 1000);

    return () => {
      clearInterval(checkConnection);
      websocketService.unsubscribe([
        'market_data',
        'order_book_update', 
        'position_update',
        'order_update'
      ]);
    };
  }, [selectedSymbol]);

  // Real-time data simulation
  useEffect(() => {
    if (!autoRefresh) return;
    
    const interval = setInterval(() => {
      // Update order book data with small price movements
      const priceVariation = (Math.random() - 0.5) * 10;
      const newPrice = currentSymbolData.price + priceVariation;
      setOrderBookData(generateOrderBookData(newPrice));
    }, 2000);
    
    return () => clearInterval(interval);
  }, [autoRefresh, currentSymbolData.price]);

  const handleSymbolChange = useCallback((symbol: string) => {
    setSelectedSymbol(symbol);
    const symbolData = mockSymbols.find(s => s.symbol === symbol);
    if (symbolData) {
      setOrderBookData(generateOrderBookData(symbolData.price));
      
      // Update WebSocket subscription
      if (websocketService.isConnected()) {
        websocketService.unsubscribe(['market_data', 'order_book_update']);
        websocketService.subscribe(['market_data', 'order_book_update'], { symbols: [symbol] });
      }
    }
  }, []);

  const handleRefresh = useCallback(async () => {
    setLoading(true);
    await new Promise(resolve => setTimeout(resolve, 500));
    setOrderBookData(generateOrderBookData(currentSymbolData.price));
    setLoading(false);
  }, [currentSymbolData.price]);

  const refreshData = useCallback(async () => {
    setLoading(true);
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 500));
      
      // Update order book
      setOrderBookData(generateOrderBookData(currentSymbolData.price));
    } catch (error) {
      console.error('Error refreshing data:', error);
    } finally {
      setLoading(false);
    }
  }, [currentSymbolData.price]);

  const handleCancelOrder = useCallback(async (orderId: string) => {
    try {
      setLoading(true);
      // Simulate API call to cancel order
      await new Promise(resolve => setTimeout(resolve, 500));
      console.log('Order cancelled:', orderId);
      await refreshData();
    } catch (error) {
      console.error('Error cancelling order:', error);
    } finally {
      setLoading(false);
    }
  }, [refreshData]);

  const handleClosePosition = useCallback(async (positionId: string) => {
    try {
      setLoading(true);
      // Simulate API call to close position
      await new Promise(resolve => setTimeout(resolve, 500));
      console.log('Position closed:', positionId);
      await refreshData();
    } catch (error) {
      console.error('Error closing position:', error);
    } finally {
      setLoading(false);
    }
  }, [refreshData]);

  return (
    <TooltipProvider>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.3 }}
        className="h-screen flex flex-col bg-background"
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b bg-card">
          <div className="flex items-center gap-6">
            <h1 className="text-2xl font-bold">Trading</h1>
            
            {/* Symbol Selector */}
            <Select value={selectedSymbol} onValueChange={handleSymbolChange}>
              <SelectTrigger className="w-48">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {mockSymbols.map((symbol) => (
                  <SelectItem key={symbol.symbol} value={symbol.symbol}>
                    <div className="flex items-center justify-between w-full">
                      <span>{symbol.symbol}</span>
                      <Badge 
                        variant={symbol.change >= 0 ? 'default' : 'destructive'}
                        className={cn(
                          'ml-2 text-xs',
                          symbol.change >= 0 ? 'bg-green-600' : 'bg-red-600'
                        )}
                      >
                        {formatPercent(symbol.change)}
                      </Badge>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            
            {/* Current Price Display */}
            <div className="space-y-1">
              <div className="text-3xl font-bold font-mono">
                {formatPrice(currentSymbolData.price)}
              </div>
              <div className={cn(
                'text-sm font-mono font-semibold',
                getPnLColor(currentSymbolData.change)
              )}>
                {formatPercent(currentSymbolData.change)}
              </div>
            </div>

            {/* 24h Stats */}
            <div className="flex items-center gap-4 text-sm text-muted-foreground">
              <div>
                <span className="font-medium">24h High: </span>
                <span className="font-mono">{formatPrice(currentSymbolData.high24h)}</span>
              </div>
              <div>
                <span className="font-medium">24h Low: </span>
                <span className="font-mono">{formatPrice(currentSymbolData.low24h)}</span>
              </div>
              <div>
                <span className="font-medium">Volume: </span>
                <span className="font-mono">{(currentSymbolData.volume / 1000000).toFixed(2)}M</span>
              </div>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            {/* WebSocket Connection Status */}
            <div className="flex items-center gap-2">
              {isConnected ? (
                <Tooltip>
                  <TooltipTrigger>
                    <div className="flex items-center gap-1 text-green-500">
                      <Wifi className="h-4 w-4" />
                      <span className="text-xs font-medium">Live</span>
                    </div>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Connected to live data feed</p>
                  </TooltipContent>
                </Tooltip>
              ) : (
                <Tooltip>
                  <TooltipTrigger>
                    <div className="flex items-center gap-1 text-red-500">
                      <WifiOff className="h-4 w-4" />
                      <span className="text-xs font-medium">Offline</span>
                    </div>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Disconnected from live data feed</p>
                  </TooltipContent>
                </Tooltip>
              )}
            </div>

            <div className="flex items-center gap-2">
              <span className="text-sm font-medium">Real-time</span>
              <Button
                variant={autoRefresh ? "default" : "outline"}
                size="sm"
                onClick={() => setAutoRefresh(!autoRefresh)}
              >
                {autoRefresh ? 'ON' : 'OFF'}
              </Button>
            </div>
            
            <Separator orientation="vertical" className="h-6" />
            
            <Tooltip>
              <TooltipTrigger>
                <Button 
                  variant="ghost" 
                  size="icon"
                  onClick={handleRefresh} 
                  disabled={loading}
                >
                  <Refresh className={cn('h-4 w-4', loading && 'animate-spin')} />
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>Refresh data</p>
              </TooltipContent>
            </Tooltip>
            
            <Tooltip>
              <TooltipTrigger>
                <Button variant="ghost" size="icon">
                  <Settings className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>Trading settings</p>
              </TooltipContent>
            </Tooltip>
            
            <Tooltip>
              <TooltipTrigger>
                <Button variant="ghost" size="icon">
                  <Bell className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>Notifications</p>
              </TooltipContent>
            </Tooltip>
          </div>
        </div>

        {/* Main Trading Interface */}
        <div className="flex-1 overflow-hidden">
          <PanelGroup direction="horizontal" className="h-full">
            {/* Left Panel - Order Book */}
            <Panel defaultSize={25} minSize={20} maxSize={35}>
              <div className="h-full p-2">
                <OrderBookComponent data={orderBookData} currentPrice={currentSymbolData.price} />
              </div>
            </Panel>
            
            <ResizeHandle direction="horizontal" />
            
            {/* Center Panel - Chart and Bottom Tables */}
            <Panel defaultSize={50} minSize={40}>
              <PanelGroup direction="vertical">
                {/* Chart */}
                <Panel defaultSize={65} minSize={50}>
                  <div className="h-full p-2">
                    <PriceChart symbol={selectedSymbol} price={currentSymbolData.price} />
                  </div>
                </Panel>
                
                <ResizeHandle direction="vertical" />
                
                {/* Bottom Tables */}
                <Panel defaultSize={35} minSize={25}>
                  <div className="h-full p-2">
                    <Tabs defaultValue="positions" className="h-full">
                      <TabsList className="grid w-full grid-cols-4">
                        <TabsTrigger value="positions" className="flex items-center gap-2">
                          <TrendingUp className="h-4 w-4" />
                          Positions
                        </TabsTrigger>
                        <TabsTrigger value="orders" className="flex items-center gap-2">
                          <Activity className="h-4 w-4" />
                          Open Orders
                        </TabsTrigger>
                        <TabsTrigger value="history" className="flex items-center gap-2">
                          <History className="h-4 w-4" />
                          Order History
                        </TabsTrigger>
                        <TabsTrigger value="trades" className="flex items-center gap-2">
                          <BarChart3 className="h-4 w-4" />
                          Trade History
                        </TabsTrigger>
                      </TabsList>
                      
                      <TabsContent value="positions" className="h-full mt-2">
                        <Card className="h-full">
                          <CardContent className="p-0">
                            <Table>
                              <TableHeader>
                                <TableRow>
                                  <TableHead>Pair</TableHead>
                                  <TableHead>Side</TableHead>
                                  <TableHead>Amount</TableHead>
                                  <TableHead>Entry Price</TableHead>
                                  <TableHead>Current Price</TableHead>
                                  <TableHead>P&L</TableHead>
                                  <TableHead>Actions</TableHead>
                                </TableRow>
                              </TableHeader>
                              <TableBody>
                                {positions.map((position) => (
                                  <TableRow key={position.id}>
                                    <TableCell className="font-medium">{position.symbol}</TableCell>
                                    <TableCell>
                                      <Badge 
                                        variant={position.side === 'long' ? 'default' : 'destructive'}
                                        className={position.side === 'long' ? 'bg-green-600' : 'bg-red-600'}
                                      >
                                        {position.side.toUpperCase()}
                                      </Badge>
                                    </TableCell>
                                    <TableCell className="font-mono">{position.amount}</TableCell>
                                    <TableCell className="font-mono">{formatPrice(position.entryPrice)}</TableCell>
                                    <TableCell className="font-mono">{formatPrice(position.currentPrice)}</TableCell>
                                    <TableCell className={cn('font-mono', getPnLColor(position.pnl))}>
                                      {formatPnL(position.pnl)}
                                      <br />
                                      <span className="text-xs">{formatPercent(position.pnlPercent)}</span>
                                    </TableCell>
                                    <TableCell>
                                      <div className="flex items-center gap-1">
                                        <Button variant="ghost" size="icon" className="h-8 w-8">
                                          <Eye className="h-3 w-3" />
                                        </Button>
                                        <Button 
                                          variant="ghost" 
                                          size="icon" 
                                          className="h-8 w-8 text-red-500 hover:text-red-600"
                                          onClick={() => handleClosePosition(position.id)}
                                        >
                                          <X className="h-3 w-3" />
                                        </Button>
                                      </div>
                                    </TableCell>
                                  </TableRow>
                                ))}
                              </TableBody>
                            </Table>
                            {positions.length === 0 && (
                              <div className="flex flex-col items-center justify-center h-32 text-muted-foreground">
                                <TrendingUp className="h-8 w-8 mb-2" />
                                <p className="font-medium">No Open Positions</p>
                                <p className="text-sm">Place an order to start trading</p>
                              </div>
                            )}
                          </CardContent>
                        </Card>
                      </TabsContent>
                      
                      <TabsContent value="orders" className="h-full mt-2">
                        <Card className="h-full">
                          <CardContent className="p-0">
                            <Table>
                              <TableHeader>
                                <TableRow>
                                  <TableHead>Pair</TableHead>
                                  <TableHead>Type</TableHead>
                                  <TableHead>Side</TableHead>
                                  <TableHead>Amount</TableHead>
                                  <TableHead>Price</TableHead>
                                  <TableHead>Filled</TableHead>
                                  <TableHead>Status</TableHead>
                                  <TableHead>Time</TableHead>
                                  <TableHead>Actions</TableHead>
                                </TableRow>
                              </TableHeader>
                              <TableBody>
                                {openOrders.map((order) => (
                                  <TableRow key={order.id}>
                                    <TableCell className="font-medium">{order.symbol}</TableCell>
                                    <TableCell>{order.type}</TableCell>
                                    <TableCell>
                                      <Badge 
                                        variant={order.side === 'buy' ? 'default' : 'destructive'}
                                        className={order.side === 'buy' ? 'bg-green-600' : 'bg-red-600'}
                                      >
                                        {order.side.toUpperCase()}
                                      </Badge>
                                    </TableCell>
                                    <TableCell className="font-mono">{order.amount}</TableCell>
                                    <TableCell className="font-mono">{formatPrice(order.price)}</TableCell>
                                    <TableCell className="font-mono">{order.filled.toFixed(4)}</TableCell>
                                    <TableCell>
                                      <Badge variant="outline">{order.status}</Badge>
                                    </TableCell>
                                    <TableCell className="text-sm">{order.time}</TableCell>
                                    <TableCell>
                                      <Button 
                                        variant="ghost" 
                                        size="icon" 
                                        className="h-8 w-8 text-red-500 hover:text-red-600"
                                        onClick={() => handleCancelOrder(order.id)}
                                      >
                                        <X className="h-3 w-3" />
                                      </Button>
                                    </TableCell>
                                  </TableRow>
                                ))}
                              </TableBody>
                            </Table>
                            {openOrders.length === 0 && (
                              <div className="flex flex-col items-center justify-center h-32 text-muted-foreground">
                                <Activity className="h-8 w-8 mb-2" />
                                <p className="font-medium">No Open Orders</p>
                                <p className="text-sm">Your pending orders will appear here</p>
                              </div>
                            )}
                          </CardContent>
                        </Card>
                      </TabsContent>
                      
                      <TabsContent value="history" className="h-full mt-2">
                        <Card className="h-full">
                          <CardContent className="flex items-center justify-center h-32 text-muted-foreground">
                            <div className="text-center">
                              <History className="h-8 w-8 mx-auto mb-2" />
                              <p className="font-medium">Order History</p>
                              <p className="text-sm">Your completed orders will appear here</p>
                            </div>
                          </CardContent>
                        </Card>
                      </TabsContent>
                      
                      <TabsContent value="trades" className="h-full mt-2">
                        <Card className="h-full">
                          <CardContent className="p-0">
                            <Table>
                              <TableHeader>
                                <TableRow>
                                  <TableHead>Pair</TableHead>
                                  <TableHead>Side</TableHead>
                                  <TableHead>Amount</TableHead>
                                  <TableHead>Price</TableHead>
                                  <TableHead>Fee</TableHead>
                                  <TableHead>Time</TableHead>
                                </TableRow>
                              </TableHeader>
                              <TableBody>
                                {trades.map((trade) => (
                                  <TableRow key={trade.id}>
                                    <TableCell className="font-medium">{trade.symbol}</TableCell>
                                    <TableCell>
                                      <Badge 
                                        variant={trade.side === 'buy' ? 'default' : 'destructive'}
                                        className={trade.side === 'buy' ? 'bg-green-600' : 'bg-red-600'}
                                      >
                                        {trade.side.toUpperCase()}
                                      </Badge>
                                    </TableCell>
                                    <TableCell className="font-mono">{trade.amount}</TableCell>
                                    <TableCell className="font-mono">{formatPrice(trade.price)}</TableCell>
                                    <TableCell className="font-mono">{formatPrice(trade.fee)}</TableCell>
                                    <TableCell className="text-sm">{trade.time}</TableCell>
                                  </TableRow>
                                ))}
                              </TableBody>
                            </Table>
                          </CardContent>
                        </Card>
                      </TabsContent>
                    </Tabs>
                  </div>
                </Panel>
              </PanelGroup>
            </Panel>
            
            <ResizeHandle direction="horizontal" />
            
            {/* Right Panel - Order Form */}
            <Panel defaultSize={25} minSize={20} maxSize={35}>
              <div className="h-full p-2">
                <OrderForm symbol={selectedSymbol} currentPrice={currentSymbolData.price} />
              </div>
            </Panel>
          </PanelGroup>
        </div>
      </motion.div>
    </TooltipProvider>
  );
};

export default TradingPage;