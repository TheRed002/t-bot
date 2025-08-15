/**
 * WebSocket slice for Redux store
 */

import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { WebSocketState, WebSocketMessage } from '@/types';

const initialState: WebSocketState = {
  isConnected: false,
  connectionStatus: 'disconnected',
  lastHeartbeat: undefined,
  error: null,
};

const websocketSlice = createSlice({
  name: 'websocket',
  initialState,
  reducers: {
    connect: (state) => {
      state.connectionStatus = 'connecting';
      state.error = null;
    },
    connected: (state) => {
      state.isConnected = true;
      state.connectionStatus = 'connected';
      state.error = null;
    },
    disconnect: (state) => {
      state.isConnected = false;
      state.connectionStatus = 'disconnected';
    },
    error: (state, action: PayloadAction<string>) => {
      state.isConnected = false;
      state.connectionStatus = 'error';
      state.error = action.payload;
    },
    heartbeat: (state) => {
      state.lastHeartbeat = new Date().toISOString();
    },
    messageReceived: (state, action: PayloadAction<WebSocketMessage>) => {
      // Handle different message types here
      // This is where real-time updates would be processed
    },
  },
});

export const { connect, connected, disconnect, error, heartbeat, messageReceived } = websocketSlice.actions;
export default websocketSlice.reducer;