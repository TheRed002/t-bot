/**
 * BalancesSection Component - Display account balances across exchanges
 * Multi-exchange balance aggregation with real-time updates
 */

import React, { useState } from 'react';
import { motion } from 'framer-motion';

// Shadcn/ui components
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Skeleton } from '@/components/ui/skeleton';

// Lucide React icons
import {
  Wallet,
  Eye,
  EyeOff,
  Search,
  ArrowLeftRight,
  TrendingUp,
  TrendingDown,
  Info,
} from 'lucide-react';

import { Balance } from '@/types';
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Tooltip as ChartTooltip,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
} from 'recharts';

interface BalancesSectionProps {
  balances?: Balance[];
  isLoading?: boolean;
  onTransfer?: (from: string, to: string, amount: number) => void;
}

const BalancesSection: React.FC<BalancesSectionProps> = ({
  balances = [],
  isLoading = false,
  onTransfer,
}) => {
  const [showZeroBalances, setShowZeroBalances] = useState(false);
  const [viewMode, setViewMode] = useState<'table' | 'cards'>('cards');
  const [searchTerm, setSearchTerm] = useState('');
  const [hideBalances, setHideBalances] = useState(false);

  // Mock balances for demonstration
  const mockBalances: Balance[] = [
    {
      currency: 'USDT',
      exchange: 'Binance',
      free: 5000,
      locked: 500,
      total: 5500,
      usdValue: 5500,
    },
    {
      currency: 'BTC',
      exchange: 'Binance',
      free: 0.25,
      locked: 0.05,
      total: 0.3,
      usdValue: 13050,
    },
    {
      currency: 'ETH',
      exchange: 'Binance',
      free: 2,
      locked: 0.5,
      total: 2.5,
      usdValue: 5375,
    },
    {
      currency: 'USDT',
      exchange: 'Coinbase',
      free: 3000,
      locked: 0,
      total: 3000,
      usdValue: 3000,
    },
    {
      currency: 'SOL',
      exchange: 'Coinbase',
      free: 50,
      locked: 10,
      total: 60,
      usdValue: 6000,
    },
  ];

  const displayBalances = balances.length > 0 ? balances : mockBalances;

  // Filter balances
  const filteredBalances = displayBalances.filter((balance) => {
    const matchesSearch =
      balance.currency.toLowerCase().includes(searchTerm.toLowerCase()) ||
      balance.exchange.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesZeroFilter = showZeroBalances || balance.total > 0;
    return matchesSearch && matchesZeroFilter;
  });

  // Calculate totals by exchange
  const exchangeTotals = filteredBalances.reduce((acc, balance) => {
    if (!acc[balance.exchange]) {
      acc[balance.exchange] = 0;
    }
    acc[balance.exchange] += balance.usdValue;
    return acc;
  }, {} as Record<string, number>);

  // Calculate totals by currency
  const currencyTotals = filteredBalances.reduce((acc, balance) => {
    if (!acc[balance.currency]) {
      acc[balance.currency] = 0;
    }
    acc[balance.currency] += balance.usdValue;
    return acc;
  }, {} as Record<string, number>);

  const totalValue = Object.values(exchangeTotals).reduce((sum, val) => sum + val, 0);

  const formatCurrency = (value: number, decimals = 2) => {
    if (hideBalances) return '****';
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    }).format(value);
  };

  const formatAmount = (value: number, currency: string) => {
    if (hideBalances) return '****';
    if (currency === 'USDT' || currency === 'USDC') {
      return value.toFixed(2);
    }
    return value.toFixed(8);
  };

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D'];

  if (isLoading) {
    return (
      <div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {[...Array(4)].map((_, index) => (
            <Skeleton key={index} className="h-[150px]" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-950/50 dark:to-indigo-950/50">
          <CardContent className="p-4">
            <div className="flex flex-col space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Total Balance</span>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-6 w-6 p-0"
                  onClick={() => setHideBalances(!hideBalances)}
                >
                  {hideBalances ? (
                    <EyeOff className="h-3 w-3" />
                  ) : (
                    <Eye className="h-3 w-3" />
                  )}
                </Button>
              </div>
              <h3 className="text-2xl font-bold">{formatCurrency(totalValue)}</h3>
              <div className="flex space-x-2">
                <Badge variant="outline" className="text-xs">
                  {Object.keys(exchangeTotals).length} exchanges
                </Badge>
                <Badge variant="outline" className="text-xs">
                  {Object.keys(currencyTotals).length} assets
                </Badge>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Exchange Distribution
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {Object.entries(exchangeTotals)
              .sort((a, b) => b[1] - a[1])
              .slice(0, 3)
              .map(([exchange, value]) => (
                <div key={exchange} className="flex items-center justify-between">
                  <span className="text-sm">{exchange}</span>
                  <div className="flex items-center space-x-2">
                    <Progress
                      value={(value / totalValue) * 100}
                      className="w-12 h-2"
                    />
                    <span className="text-sm font-medium">
                      {formatCurrency(value, 0)}
                    </span>
                  </div>
                </div>
              ))}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Top Assets
            </CardTitle>
          </CardHeader>
          <CardContent className="h-20">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={Object.entries(currencyTotals)
                    .sort((a, b) => b[1] - a[1])
                    .slice(0, 5)
                    .map(([currency, value]) => ({ name: currency, value }))}
                  cx="50%"
                  cy="50%"
                  innerRadius={20}
                  outerRadius={35}
                  paddingAngle={2}
                  dataKey="value"
                >
                  {Object.entries(currencyTotals).map((_, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <ChartTooltip
                  contentStyle={{
                    backgroundColor: 'var(--background)',
                    border: '1px solid var(--border)',
                    borderRadius: '8px',
                  }}
                  formatter={(value: any) => formatCurrency(value)}
                />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="text-center space-y-2">
              <Wallet className="h-8 w-8 mx-auto text-primary" />
              <p className="text-sm font-medium">Portfolio Health</p>
              <div className="text-2xl font-bold text-green-600">Excellent</div>
              <p className="text-xs text-muted-foreground">Well diversified</p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Controls */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div className="relative w-full sm:w-80">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
          <Input
            placeholder="Search currency or exchange..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10"
          />
        </div>
        <Tabs value={viewMode} onValueChange={(value: any) => setViewMode(value)}>
          <TabsList>
            <TabsTrigger value="cards">Cards</TabsTrigger>
            <TabsTrigger value="table">Table</TabsTrigger>
          </TabsList>
        </Tabs>
      </div>

      {/* Balances Display */}
      <Tabs value={viewMode} onValueChange={(value: any) => setViewMode(value)}>
        <TabsContent value="cards" className="mt-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {filteredBalances.map((balance, index) => (
              <motion.div
                key={`${balance.exchange}-${balance.currency}`}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: index * 0.05 }}
              >
                <Card className="hover:shadow-md transition-all hover:-translate-y-1">
                  <CardContent className="p-4">
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <div>
                          <h3 className="font-semibold text-lg">{balance.currency}</h3>
                          <Badge variant="outline" className="text-xs">
                            {balance.exchange}
                          </Badge>
                        </div>
                        <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                          <ArrowLeftRight className="h-4 w-4" />
                        </Button>
                      </div>
                      
                      <div>
                        <p className="text-xs text-muted-foreground mb-1">Total Balance</p>
                        <p className="text-xl font-bold">
                          {formatAmount(balance.total, balance.currency)} {balance.currency}
                        </p>
                        <p className="text-sm text-muted-foreground">
                          ≈ {formatCurrency(balance.usdValue)}
                        </p>
                      </div>

                      <div className="space-y-1">
                        <div className="flex justify-between text-xs">
                          <span className="text-muted-foreground">Available</span>
                          <span>{formatAmount(balance.free, balance.currency)}</span>
                        </div>
                        <div className="flex justify-between text-xs">
                          <span className="text-muted-foreground">Locked</span>
                          <span>{formatAmount(balance.locked, balance.currency)}</span>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="table" className="mt-4">
          <Card>
            <Table>
              <TableHeader>
                <TableRow className="bg-muted/50">
                  <TableHead>Currency</TableHead>
                  <TableHead>Exchange</TableHead>
                  <TableHead className="text-right">Available</TableHead>
                  <TableHead className="text-right">Locked</TableHead>
                  <TableHead className="text-right">Total</TableHead>
                  <TableHead className="text-right">USD Value</TableHead>
                  <TableHead className="text-center">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredBalances.map((balance) => (
                  <TableRow
                    key={`${balance.exchange}-${balance.currency}`}
                    className="hover:bg-muted/50"
                  >
                    <TableCell>
                      <span className="font-semibold">{balance.currency}</span>
                    </TableCell>
                    <TableCell>
                      <Badge variant="outline">{balance.exchange}</Badge>
                    </TableCell>
                    <TableCell className="text-right">
                      {formatAmount(balance.free, balance.currency)}
                    </TableCell>
                    <TableCell className="text-right">
                      {formatAmount(balance.locked, balance.currency)}
                    </TableCell>
                    <TableCell className="text-right font-semibold">
                      {formatAmount(balance.total, balance.currency)}
                    </TableCell>
                    <TableCell className="text-right font-semibold">
                      {formatCurrency(balance.usdValue)}
                    </TableCell>
                    <TableCell className="text-center">
                      <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                        <ArrowLeftRight className="h-4 w-4" />
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default BalancesSection;