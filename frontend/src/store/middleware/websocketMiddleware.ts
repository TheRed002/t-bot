/**
 * WebSocket middleware for real-time data management
 */

import { Middleware } from '@reduxjs/toolkit';
import { connected, disconnect, error, messageReceived } from '../slices/websocketSlice';
import { updateBotStatus, updateBotPerformance } from '../slices/botSlice';
import { updateMarketData, updateOrderBook } from '../slices/marketSlice';
import { updatePositionPrice } from '../slices/portfolioSlice';

let socket: WebSocket | null = null;

export const websocketMiddleware: Middleware = (store) => (next) => (action) => {
  const result = next(action);
  
  // Handle WebSocket connection actions
  if (action.type === 'websocket/connect') {
    if (socket) {
      socket.close();
    }
    
    const token = store.getState().auth.token;
    socket = new WebSocket(`ws://localhost:8000/ws?token=${token}`);
    
    socket.onopen = () => {
      store.dispatch(connected());
    };
    
    socket.onclose = () => {
      store.dispatch(disconnect());
    };
    
    socket.onerror = () => {
      store.dispatch(error('WebSocket connection failed'));
    };
    
    socket.onmessage = (event) => {
      const message = JSON.parse(event.data);
      store.dispatch(messageReceived(message));
      
      // Route messages to appropriate slices
      switch (message.type) {
        case 'bot_status_update':
          store.dispatch(updateBotStatus(message.data));
          break;
        case 'bot_performance_update':
          store.dispatch(updateBotPerformance(message.data));
          break;
        case 'market_data':
          store.dispatch(updateMarketData(message.data));
          break;
        case 'order_book':
          store.dispatch(updateOrderBook(message.data));
          break;
        case 'position_update':
          store.dispatch(updatePositionPrice(message.data));
          break;
      }
    };
  }
  
  if (action.type === 'websocket/disconnect' && socket) {
    socket.close();
    socket = null;
  }
  
  return result;
};