/**
 * WebSocket service for real-time updates
 * Handles connection management, reconnection, and message routing
 */

import { io, Socket } from 'socket.io-client';
import { store } from '@/store';
import { connected, disconnected, disconnect, error, heartbeat, messageReceived } from '@/store/slices/websocketSlice';
import { updateBotStatus, updateBotPerformance } from '@/store/slices/botSlice';
import { updateMarketData, updateOrderBook } from '@/store/slices/marketSlice';
import { updatePositionPrice } from '@/store/slices/portfolioSlice';
import { addAlert } from '@/store/slices/riskSlice';
import { addNotification } from '@/store/slices/uiSlice';

interface WebSocketConfig {
  url: string;
  autoConnect: boolean;
  reconnection: boolean;
  reconnectionAttempts: number;
  reconnectionDelay: number;
  reconnectionDelayMax: number;
  timeout: number;
}

class WebSocketService {
  private socket: Socket | null = null;
  private config: WebSocketConfig;
  private isAuthenticated = false;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private heartbeatInterval: NodeJS.Timeout | null = null;

  constructor(config: Partial<WebSocketConfig> = {}) {
    this.config = {
      url: process.env.REACT_APP_API_URL || 'http://localhost:8000',
      autoConnect: false,
      reconnection: true,
      reconnectionAttempts: 5,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      timeout: 20000,
      ...config,
    };
  }

  // Connect to WebSocket server
  connect(token?: string): void {
    if (this.socket?.connected) {
      console.warn('WebSocket already connected');
      return;
    }

    const authToken = token || this.getStoredToken();
    if (!authToken) {
      console.error('No authentication token available for WebSocket connection');
      return;
    }

    // Disconnect existing socket if any
    if (this.socket) {
      this.socket.disconnect();
    }

    // Create new socket connection
    this.socket = io(this.config.url, {
      auth: {
        token: authToken,
      },
      path: '/socket.io/',
      autoConnect: this.config.autoConnect,
      reconnection: this.config.reconnection,
      reconnectionAttempts: this.config.reconnectionAttempts,
      reconnectionDelay: this.config.reconnectionDelay,
      reconnectionDelayMax: this.config.reconnectionDelayMax,
      timeout: this.config.timeout,
      transports: ['websocket', 'polling'],
      upgrade: true,
    });

    this.setupEventHandlers();
    
    // Manually connect if autoConnect is false
    if (!this.config.autoConnect) {
      this.socket.connect();
    }

    store.dispatch({ type: 'websocket/connect' });
  }

  // Disconnect from WebSocket server
  disconnect(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }

    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }

    this.isAuthenticated = false;
    this.reconnectAttempts = 0;
    store.dispatch(disconnect());
  }

  // Setup event handlers
  private setupEventHandlers(): void {
    if (!this.socket) return;

    // Connection events
    this.socket.on('connect', () => {
      console.log('WebSocket connected');
      this.reconnectAttempts = 0;
      store.dispatch(connected());
      this.startHeartbeat();
      
      // Auto-subscribe to default channels after connection
      this.subscribe(['market_data', 'bot_status', 'portfolio', 'alerts']);
    });

    this.socket.on('disconnect', (reason) => {
      console.log('WebSocket disconnected:', reason);
      this.isAuthenticated = false;
      store.dispatch(disconnected(reason));
      this.stopHeartbeat();

      // Auto-reconnect for recoverable disconnects
      const recoverableReasons = [
        'transport close',
        'transport error',
        'ping timeout'
      ];
      
      if (recoverableReasons.includes(reason)) {
        this.handleReconnection();
      }
    });

    this.socket.on('connect_error', (err) => {
      console.error('WebSocket connection error:', err);
      store.dispatch(error(err.message));
      
      // Only attempt reconnection if under max attempts
      if (this.reconnectAttempts < this.maxReconnectAttempts) {
        this.handleReconnection();
      } else {
        console.error('Max reconnection attempts reached');
        store.dispatch(error('Failed to establish connection after multiple attempts'));
      }
    });

    // Authentication events
    this.socket.on('authenticated', (data) => {
      console.log('WebSocket authenticated:', data);
      this.isAuthenticated = true;
      store.dispatch({ type: 'websocket/authenticated', payload: data });
      
      // Subscribe to channels after authentication
      if (data.status === 'success') {
        this.subscribe(['market_data', 'bot_status', 'portfolio', 'alerts']);
      }
    });

    this.socket.on('auth_error', (err) => {
      console.error('WebSocket authentication failed:', err);
      store.dispatch(error(err.error || 'Authentication failed'));
      this.isAuthenticated = false;
      
      // Clear token and redirect to login
      localStorage.removeItem('token');
      store.dispatch({ type: 'auth/clearAuth' });
      window.location.href = '/login';
    });

    // Heartbeat
    this.socket.on('pong', () => {
      store.dispatch(heartbeat());
    });

    // Trading bot updates
    this.socket.on('bot_status_update', (data) => {
      store.dispatch(updateBotStatus(data));
      store.dispatch(messageReceived({
        type: 'bot_status_update',
        data,
        timestamp: new Date().toISOString(),
      }));
    });

    this.socket.on('bot_performance_update', (data) => {
      store.dispatch(updateBotPerformance(data));
      store.dispatch(messageReceived({
        type: 'bot_performance_update',
        data,
        timestamp: new Date().toISOString(),
      }));
    });

    // Market data updates
    this.socket.on('market_data', (data) => {
      store.dispatch(updateMarketData(data));
      store.dispatch(messageReceived({
        type: 'market_data',
        data,
        timestamp: new Date().toISOString(),
      }));
    });

    this.socket.on('order_book_update', (data) => {
      store.dispatch(updateOrderBook(data));
      store.dispatch(messageReceived({
        type: 'order_book_update',
        data,
        timestamp: new Date().toISOString(),
      }));
    });

    // Portfolio updates
    this.socket.on('position_update', (data) => {
      store.dispatch(updatePositionPrice(data));
      store.dispatch(messageReceived({
        type: 'position_update',
        data,
        timestamp: new Date().toISOString(),
      }));
    });

    // Risk alerts
    this.socket.on('risk_alert', (data) => {
      store.dispatch(addAlert(data));
      store.dispatch(addNotification({
        type: data.severity === 'critical' ? 'error' : 'warning',
        title: 'Risk Alert',
        message: data.message,
        autoHide: false,
      }));
      store.dispatch(messageReceived({
        type: 'risk_alert',
        data,
        timestamp: new Date().toISOString(),
      }));
    });

    // System notifications
    this.socket.on('system_notification', (data) => {
      store.dispatch(addNotification({
        type: data.type || 'info',
        title: data.title || 'System Notification',
        message: data.message,
        autoHide: data.autoHide !== false,
        duration: data.duration,
      }));
    });

    // Error handling
    this.socket.on('error', (err) => {
      console.error('WebSocket error:', err);
      store.dispatch(error(err.message || 'WebSocket error'));
    });
  }

  // Handle reconnection logic
  private handleReconnection(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      store.dispatch(error('Connection failed after multiple attempts'));
      return;
    }

    this.reconnectAttempts++;
    const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
    
    console.log(`Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts})`);
    
    setTimeout(() => {
      if (!this.socket?.connected) {
        const token = this.getStoredToken();
        if (token) {
          this.connect(token);
        }
      }
    }, delay);
  }

  // Start heartbeat to keep connection alive
  private startHeartbeat(): void {
    this.heartbeatInterval = setInterval(() => {
      if (this.socket?.connected) {
        this.socket.emit('ping');
      }
    }, 30000); // 30 seconds
  }

  // Stop heartbeat
  private stopHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }

  // Get stored authentication token
  private getStoredToken(): string | null {
    return localStorage.getItem('token');
  }

  // Send message to server
  emit(event: string, data?: any): void {
    if (this.socket?.connected && this.isAuthenticated) {
      this.socket.emit(event, data);
    } else {
      console.warn('Cannot emit message: WebSocket not connected or not authenticated');
    }
  }

  // Subscribe to specific data updates
  subscribe(channels: string | string[], params?: any): void {
    const channelArray = Array.isArray(channels) ? channels : [channels];
    if (this.socket?.connected) {
      this.socket.emit('subscribe', { channels: channelArray, ...params });
    } else {
      console.warn('Cannot subscribe: WebSocket not connected');
    }
  }

  // Unsubscribe from data updates
  unsubscribe(channels: string | string[]): void {
    const channelArray = Array.isArray(channels) ? channels : [channels];
    if (this.socket?.connected) {
      this.socket.emit('unsubscribe', { channels: channelArray });
    }
  }

  // Get connection status
  isConnected(): boolean {
    return this.socket?.connected || false;
  }

  // Get authentication status
  isAuth(): boolean {
    return this.isAuthenticated;
  }

  // Send order execution request
  sendOrder(orderData: any): void {
    if (this.socket?.connected && this.isAuthenticated) {
      this.socket.emit('execute_order', orderData);
    } else {
      console.error('Cannot send order: Not connected or authenticated');
      store.dispatch(error('Please connect and authenticate before placing orders'));
    }
  }

  // Request portfolio data
  requestPortfolio(): void {
    if (this.socket?.connected && this.isAuthenticated) {
      this.socket.emit('get_portfolio', {});
    }
  }

  // Send ping for connection health check
  ping(): void {
    if (this.socket?.connected) {
      this.socket.emit('ping', { timestamp: new Date().toISOString() });
    }
  }

  // Get connection status
  getConnectionStatus(): boolean {
    return this.socket?.connected || false;
  }

  // Get socket instance (for advanced usage)
  getSocket(): Socket | null {
    return this.socket;
  }
}

// Create singleton instance
export const websocketService = new WebSocketService();

// Export hooks for React components
export const useWebSocket = () => {
  return {
    connect: (token?: string) => websocketService.connect(token),
    disconnect: () => websocketService.disconnect(),
    emit: (event: string, data?: any) => websocketService.emit(event, data),
    subscribe: (channel: string, params?: any) => websocketService.subscribe(channel, params),
    unsubscribe: (channel: string) => websocketService.unsubscribe(channel),
    isConnected: () => websocketService.isConnected(),
    isAuthenticated: () => websocketService.isAuth(),
  };
};

export default websocketService;