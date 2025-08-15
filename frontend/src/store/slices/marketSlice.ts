/**
 * Market data slice for Redux store
 */

import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { MarketState, MarketData, CandlestickData, OrderBook } from '@/types';

const initialState: MarketState = {
  marketData: {},
  candlestickData: {},
  orderBooks: {},
  watchlist: ['BTCUSDT', 'ETHUSDT', 'ADAUSDT'],
  isLoading: false,
  error: null,
};

const marketSlice = createSlice({
  name: 'market',
  initialState,
  reducers: {
    updateMarketData: (state, action: PayloadAction<MarketData>) => {
      const data = action.payload;
      state.marketData[data.symbol] = data;
    },
    updateCandlestickData: (state, action: PayloadAction<{ symbol: string; data: CandlestickData[] }>) => {
      const { symbol, data } = action.payload;
      state.candlestickData[symbol] = data;
    },
    updateOrderBook: (state, action: PayloadAction<OrderBook>) => {
      const orderBook = action.payload;
      state.orderBooks[orderBook.symbol] = orderBook;
    },
    addToWatchlist: (state, action: PayloadAction<string>) => {
      if (!state.watchlist.includes(action.payload)) {
        state.watchlist.push(action.payload);
      }
    },
    removeFromWatchlist: (state, action: PayloadAction<string>) => {
      state.watchlist = state.watchlist.filter(symbol => symbol !== action.payload);
    },
  },
});

export const {
  updateMarketData,
  updateCandlestickData,
  updateOrderBook,
  addToWatchlist,
  removeFromWatchlist,
} = marketSlice.actions;

export default marketSlice.reducer;