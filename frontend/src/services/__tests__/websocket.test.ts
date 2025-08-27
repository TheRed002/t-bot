/**
 * Unit tests for WebSocket service
 */

import { io, Socket } from 'socket.io-client';
import websocketService from '../websocket';
import { store } from '@/store';

// Mock socket.io-client
jest.mock('socket.io-client');
jest.mock('@/store');

describe('WebSocketService', () => {
  let service: typeof websocketService;
  let mockSocket: jest.Mocked<Socket>;
  let mockStore: any;

  beforeEach(() => {
    // Reset mocks
    jest.clearAllMocks();
    
    // Create mock socket
    mockSocket = {
      connected: false,
      connect: jest.fn(),
      disconnect: jest.fn(),
      on: jest.fn(),
      off: jest.fn(),
      emit: jest.fn(),
      removeAllListeners: jest.fn(),
    } as any;

    // Mock io function
    (io as jest.Mock).mockReturnValue(mockSocket);

    // Mock store
    mockStore = {
      dispatch: jest.fn(),
      getState: jest.fn(() => ({
        auth: { token: 'test-token' },
        websocket: { isConnected: false }
      })),
    };
    (store as any).dispatch = mockStore.dispatch;
    (store as any).getState = mockStore.getState;

    // Use the singleton service instance
    service = websocketService;
  });

  afterEach(() => {
    service.disconnect();
  });

  describe('connect', () => {
    it('should create socket connection with token', () => {
      service.connect('test-token');

      expect(io).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          auth: { token: 'test-token' },
          path: '/socket.io/',
          autoConnect: false,
          reconnection: true,
        })
      );
    });

    it('should not connect if already connected', () => {
      mockSocket.connected = true;
      service.connect('test-token');

      expect(io).not.toHaveBeenCalled();
    });

    it('should not connect without token', () => {
      service.connect();

      expect(io).not.toHaveBeenCalled();
    });

    it('should setup event handlers after connection', () => {
      service.connect('test-token');

      expect(mockSocket.on).toHaveBeenCalledWith('connect', expect.any(Function));
      expect(mockSocket.on).toHaveBeenCalledWith('disconnect', expect.any(Function));
      expect(mockSocket.on).toHaveBeenCalledWith('error', expect.any(Function));
      expect(mockSocket.on).toHaveBeenCalledWith('authenticated', expect.any(Function));
    });

    it('should manually connect if autoConnect is false', () => {
      service.connect('test-token');

      expect(mockSocket.connect).toHaveBeenCalled();
    });

    it('should dispatch connect action to store', () => {
      service.connect('test-token');

      expect(mockStore.dispatch).toHaveBeenCalledWith({ type: 'websocket/connect' });
    });
  });

  describe('disconnect', () => {
    it('should disconnect socket if connected', () => {
      service.connect('test-token');
      service.disconnect();

      expect(mockSocket.disconnect).toHaveBeenCalled();
    });

    it('should clear heartbeat interval on disconnect', () => {
      const clearIntervalSpy = jest.spyOn(global, 'clearInterval');
      
      service.connect('test-token');
      // Simulate heartbeat being set
      (service as any).heartbeatInterval = setInterval(() => {}, 1000);
      
      service.disconnect();

      expect(clearIntervalSpy).toHaveBeenCalled();
    });

    it('should dispatch disconnect action to store', () => {
      service.connect('test-token');
      service.disconnect();

      expect(mockStore.dispatch).toHaveBeenCalledWith({ type: 'websocket/disconnect' });
    });
  });

  describe('event handlers', () => {
    beforeEach(() => {
      service.connect('test-token');
    });

    it('should handle connect event', () => {
      const connectHandler = mockSocket.on.mock.calls.find(call => call[0] === 'connect')?.[1];
      connectHandler?.();

      expect(mockStore.dispatch).toHaveBeenCalledWith({ 
        type: 'websocket/connected' 
      });
    });

    it('should handle disconnect event', () => {
      const disconnectHandler = mockSocket.on.mock.calls.find(call => call[0] === 'disconnect')?.[1];
      disconnectHandler?.('transport close');

      expect(mockStore.dispatch).toHaveBeenCalledWith({ 
        type: 'websocket/disconnected',
        payload: 'transport close'
      });
    });

    it('should handle error event', () => {
      const errorHandler = mockSocket.on.mock.calls.find(call => call[0] === 'error')?.[1];
      const error = new Error('Connection failed');
      errorHandler?.(error);

      expect(mockStore.dispatch).toHaveBeenCalledWith({ 
        type: 'websocket/error',
        payload: error.message
      });
    });

    it('should handle authenticated event', () => {
      const authHandler = mockSocket.on.mock.calls.find(call => call[0] === 'authenticated')?.[1];
      authHandler?.({ status: 'success' });

      expect(mockSocket.emit).toHaveBeenCalledWith('subscribe', {
        channels: ['market_data', 'bot_status', 'portfolio', 'alerts']
      });
    });

    it('should handle market_data event', () => {
      const handler = mockSocket.on.mock.calls.find(call => call[0] === 'market_data')?.[1];
      const data = { BTC: 50000, ETH: 3000 };
      handler?.(data);

      expect(mockStore.dispatch).toHaveBeenCalledWith({ 
        type: 'market/updateMarketData',
        payload: data
      });
    });

    it('should handle bot_status event', () => {
      const handler = mockSocket.on.mock.calls.find(call => call[0] === 'bot_status')?.[1];
      const data = { botId: '123', status: 'running' };
      handler?.(data);

      expect(mockStore.dispatch).toHaveBeenCalledWith({ 
        type: 'bot/updateBotStatus',
        payload: data
      });
    });

    it('should handle portfolio_update event', () => {
      const handler = mockSocket.on.mock.calls.find(call => call[0] === 'portfolio_update')?.[1];
      const data = { total_value: 10000 };
      handler?.(data);

      expect(mockStore.dispatch).toHaveBeenCalledWith({ 
        type: 'portfolio/updatePortfolio',
        payload: data
      });
    });
  });

  describe('emit methods', () => {
    beforeEach(() => {
      service.connect('test-token');
    });

    it('should emit subscribe event', () => {
      service.subscribe(['market_data', 'bot_status']);

      expect(mockSocket.emit).toHaveBeenCalledWith('subscribe', {
        channels: ['market_data', 'bot_status']
      });
    });

    it('should emit unsubscribe event', () => {
      service.unsubscribe(['market_data']);

      expect(mockSocket.emit).toHaveBeenCalledWith('unsubscribe', {
        channels: ['market_data']
      });
    });

    it('should emit order event', () => {
      const orderData = {
        symbol: 'BTC/USDT',
        side: 'buy',
        type: 'limit',
        amount: 0.1,
        price: 45000
      };
      
      service.sendOrder(orderData);

      expect(mockSocket.emit).toHaveBeenCalledWith('execute_order', orderData);
    });

    it('should not emit if socket is not connected', () => {
      service.disconnect();
      service.subscribe(['market_data']);

      expect(mockSocket.emit).not.toHaveBeenCalled();
    });
  });

  describe('reconnection', () => {
    it('should handle reconnect event', () => {
      service.connect('test-token');
      const reconnectHandler = mockSocket.on.mock.calls.find(call => call[0] === 'reconnect')?.[1];
      
      reconnectHandler?.(3);

      expect(mockStore.dispatch).toHaveBeenCalledWith({ 
        type: 'websocket/reconnected',
        payload: { attempt: 3 }
      });
    });

    it('should handle reconnect_error event', () => {
      service.connect('test-token');
      const errorHandler = mockSocket.on.mock.calls.find(call => call[0] === 'reconnect_error')?.[1];
      const error = new Error('Reconnection failed');
      
      errorHandler?.(error);

      expect(mockStore.dispatch).toHaveBeenCalledWith({ 
        type: 'websocket/reconnectError',
        payload: error.message
      });
    });

    it('should handle reconnect_failed event', () => {
      service.connect('test-token');
      const failedHandler = mockSocket.on.mock.calls.find(call => call[0] === 'reconnect_failed')?.[1];
      
      failedHandler?.();

      expect(mockStore.dispatch).toHaveBeenCalledWith({ 
        type: 'websocket/reconnectFailed'
      });
    });
  });

  describe('heartbeat', () => {
    jest.useFakeTimers();

    it('should start heartbeat on connection', () => {
      service.connect('test-token');
      const connectHandler = mockSocket.on.mock.calls.find(call => call[0] === 'connect')?.[1];
      
      connectHandler?.();
      
      // Fast-forward time
      jest.advanceTimersByTime(30000);

      expect(mockSocket.emit).toHaveBeenCalledWith('ping', {
        timestamp: expect.any(String)
      });
    });

    it('should clear heartbeat on disconnect', () => {
      service.connect('test-token');
      const connectHandler = mockSocket.on.mock.calls.find(call => call[0] === 'connect')?.[1];
      connectHandler?.();
      
      service.disconnect();
      
      // Fast-forward time
      jest.advanceTimersByTime(30000);

      // Should not emit ping after disconnect
      expect(mockSocket.emit).not.toHaveBeenCalledWith('ping', expect.any(Object));
    });

    jest.useRealTimers();
  });

  describe('getConnectionStatus', () => {
    it('should return connection status', () => {
      expect(service.getConnectionStatus()).toBe(false);
      
      service.connect('test-token');
      mockSocket.connected = true;
      
      expect(service.getConnectionStatus()).toBe(true);
    });
  });

  describe('getSocket', () => {
    it('should return socket instance', () => {
      expect(service.getSocket()).toBeNull();
      
      service.connect('test-token');
      
      expect(service.getSocket()).toBe(mockSocket);
    });
  });
});