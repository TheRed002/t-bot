/**
 * PositionsTable Component - Display active trading positions
 * Real-time position tracking with P&L calculations
 */

import React, { useState, useMemo } from 'react';
import { motion } from 'framer-motion';

// Shadcn/ui components
import { Card } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Skeleton } from '@/components/ui/skeleton';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';

// Lucide React icons
import {
  X,
  Edit,
  MoreHorizontal,
  Search,
  Filter,
  TrendingUp,
  TrendingDown,
  Clock,
} from 'lucide-react';

import { Position } from '@/types';

interface PositionsTableProps {
  positions?: Position[];
  isLoading?: boolean;
  onClosePosition?: (positionId: string) => void;
  onEditPosition?: (positionId: string) => void;
}

const PositionsTable: React.FC<PositionsTableProps> = ({
  positions = [],
  isLoading = false,
  onClosePosition,
  onEditPosition,
}) => {
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [searchTerm, setSearchTerm] = useState('');

  // Mock positions for demonstration
  const mockPositions: Position[] = [
    {
      id: '1',
      botId: 'bot1',
      exchange: 'Binance',
      symbol: 'BTC/USDT',
      side: 'long',
      quantity: 0.5,
      entryPrice: 42000,
      currentPrice: 43500,
      unrealizedPnl: 750,
      realizedPnl: 0,
      createdAt: '2024-01-07T10:00:00Z',
      updatedAt: '2024-01-07T12:00:00Z',
    },
    {
      id: '2',
      botId: 'bot2',
      exchange: 'Binance',
      symbol: 'ETH/USDT',
      side: 'long',
      quantity: 5,
      entryPrice: 2200,
      currentPrice: 2150,
      unrealizedPnl: -250,
      realizedPnl: 0,
      createdAt: '2024-01-07T09:00:00Z',
      updatedAt: '2024-01-07T12:00:00Z',
    },
    {
      id: '3',
      botId: 'bot3',
      exchange: 'Coinbase',
      symbol: 'SOL/USDT',
      side: 'short',
      quantity: 10,
      entryPrice: 100,
      currentPrice: 95,
      unrealizedPnl: 50,
      realizedPnl: 0,
      createdAt: '2024-01-06T15:00:00Z',
      updatedAt: '2024-01-07T12:00:00Z',
    },
  ];

  const displayPositions = positions.length > 0 ? positions : mockPositions;

  // Filter positions based on search
  const filteredPositions = useMemo(() => {
    return displayPositions.filter(
      (position) =>
        position.symbol.toLowerCase().includes(searchTerm.toLowerCase()) ||
        position.exchange.toLowerCase().includes(searchTerm.toLowerCase())
    );
  }, [displayPositions, searchTerm]);

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
    }).format(value);
  };

  const formatPercentage = (entry: number, current: number) => {
    const change = ((current - entry) / entry) * 100;
    return `${change >= 0 ? '+' : ''}${change.toFixed(2)}%`;
  };

  const getTimeSince = (date: string) => {
    const now = new Date();
    const created = new Date(date);
    const hours = Math.floor((now.getTime() - created.getTime()) / (1000 * 60 * 60));
    if (hours < 1) return 'Just now';
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    return `${days}d ago`;
  };

  if (isLoading) {
    return (
      <div className="space-y-2">
        {[...Array(5)].map((_, index) => (
          <Skeleton key={index} className="h-16 w-full" />
        ))}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Table Header with Search */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div className="relative w-full sm:w-80">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
          <Input
            placeholder="Search positions..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10"
          />
        </div>
        <div className="flex items-center space-x-2">
          <Button variant="outline" size="sm">
            <Filter className="h-4 w-4 mr-2" />
            Filter
          </Button>
          <Button variant="outline" size="sm" className="text-red-600 hover:text-red-700">
            Close All
          </Button>
        </div>
      </div>

      {/* Positions Table */}
      <Card>
        <Table>
          <TableHeader>
            <TableRow className="bg-muted/50">
              <TableHead>Symbol</TableHead>
              <TableHead>Exchange</TableHead>
              <TableHead>Side</TableHead>
              <TableHead className="text-right">Quantity</TableHead>
              <TableHead className="text-right">Entry Price</TableHead>
              <TableHead className="text-right">Current Price</TableHead>
              <TableHead className="text-right">Unrealized P&L</TableHead>
              <TableHead className="text-right">Change %</TableHead>
              <TableHead>Duration</TableHead>
              <TableHead className="text-center">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filteredPositions
              .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
              .map((position, index) => (
                <TableRow
                  key={position.id}
                  className="hover:bg-muted/50"
                >
                  <TableCell>
                    <div className="flex items-center space-x-2">
                      <span className="font-semibold">{position.symbol}</span>
                    </div>
                  </TableCell>
                  <TableCell>
                    <Badge variant="outline">{position.exchange}</Badge>
                  </TableCell>
                  <TableCell>
                    <Badge
                      variant={position.side === 'long' ? 'default' : 'destructive'}
                      className="flex items-center w-fit"
                    >
                      {position.side === 'long' ? (
                        <TrendingUp className="h-3 w-3 mr-1" />
                      ) : (
                        <TrendingDown className="h-3 w-3 mr-1" />
                      )}
                      {position.side.toUpperCase()}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-right">
                    <span>{position.quantity}</span>
                  </TableCell>
                  <TableCell className="text-right">
                    <span>{formatCurrency(position.entryPrice)}</span>
                  </TableCell>
                  <TableCell className="text-right">
                    <span className="font-semibold">{formatCurrency(position.currentPrice)}</span>
                  </TableCell>
                  <TableCell className="text-right">
                    <span
                      className={`font-semibold ${
                        position.unrealizedPnl >= 0 ? 'text-green-600' : 'text-red-600'
                      }`}
                    >
                      {formatCurrency(position.unrealizedPnl)}
                    </span>
                  </TableCell>
                  <TableCell className="text-right">
                    <Badge
                      variant={position.currentPrice >= position.entryPrice ? 'default' : 'destructive'}
                      className={
                        position.currentPrice >= position.entryPrice 
                          ? 'bg-green-100 text-green-800 hover:bg-green-200'
                          : 'bg-red-100 text-red-800 hover:bg-red-200'
                      }
                    >
                      {formatPercentage(position.entryPrice, position.currentPrice)}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center space-x-1">
                      <Clock className="h-3 w-3 text-muted-foreground" />
                      <span className="text-xs text-muted-foreground">
                        {getTimeSince(position.createdAt)}
                      </span>
                    </div>
                  </TableCell>
                  <TableCell className="text-center">
                    <div className="flex items-center justify-center space-x-1">
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-8 w-8 p-0"
                        onClick={() => onEditPosition?.(position.id)}
                      >
                        <Edit className="h-3 w-3" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-8 w-8 p-0 text-red-600 hover:text-red-700"
                        onClick={() => onClosePosition?.(position.id)}
                      >
                        <X className="h-3 w-3" />
                      </Button>
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                            <MoreHorizontal className="h-3 w-3" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end" className="w-40">
                          <DropdownMenuItem>View Details</DropdownMenuItem>
                          <DropdownMenuItem>Add Stop Loss</DropdownMenuItem>
                          <DropdownMenuItem>Add Take Profit</DropdownMenuItem>
                          <DropdownMenuItem>Duplicate Position</DropdownMenuItem>
                          <DropdownMenuItem className="text-red-600">
                            Force Close
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </div>
                  </TableCell>
                </TableRow>
              ))}
          </TableBody>
        </Table>
      </Card>

      {/* Simple Pagination Info */}
      {filteredPositions.length > rowsPerPage && (
        <div className="flex items-center justify-between">
          <p className="text-sm text-muted-foreground">
            Showing {page * rowsPerPage + 1} to {Math.min((page + 1) * rowsPerPage, filteredPositions.length)} of{' '}
            {filteredPositions.length} positions
          </p>
          <div className="flex items-center space-x-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setPage(Math.max(0, page - 1))}
              disabled={page === 0}
            >
              Previous
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setPage(page + 1)}
              disabled={(page + 1) * rowsPerPage >= filteredPositions.length}
            >
              Next
            </Button>
          </div>
        </div>
      )}
    </div>
  );
};

export default PositionsTable;