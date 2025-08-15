/**
 * Color palette for T-Bot Trading System
 * Professional dark theme optimized for financial data visualization
 */

export const colors = {
  // Primary brand colors
  primary: {
    50: '#e3f2ff',
    100: '#b8deff',
    200: '#8bcaff',
    300: '#5eb5ff',
    400: '#3da6ff',
    500: '#1c96ff', // Main brand blue
    600: '#1886e5',
    700: '#1374cc',
    800: '#0e63b3',
    900: '#084580',
  },

  // Accent colors for trading interface
  accent: {
    cyan: '#00d4ff',    // Bright accent for highlights
    teal: '#00bcd4',    // Secondary accent
    purple: '#7c4dff',  // Data visualization
    pink: '#ff4081',    // Alerts/warnings
  },

  // Financial status colors
  financial: {
    profit: '#00e676',     // Green for profits
    loss: '#ff5252',       // Red for losses
    warning: '#ffc107',    // Yellow for warnings
    neutral: '#90a4ae',    // Gray for neutral
    buy: '#4caf50',        // Buy orders
    sell: '#f44336',       // Sell orders
  },

  // Background colors (dark theme)
  background: {
    primary: '#0a0e1a',     // Main app background
    secondary: '#0f1419',   // Card backgrounds
    tertiary: '#1a1f2e',    // Input backgrounds
    elevated: '#1e2329',    // Modal/dropdown backgrounds
    surface: '#252932',     // Surface elements
    paper: '#2a2f3a',       // Paper-like surfaces
  },

  // Text colors
  text: {
    primary: '#ffffff',      // Primary text
    secondary: '#b3bcc8',    // Secondary text
    muted: '#8892a0',        // Muted/disabled text
    inverse: '#0a0e1a',      // Text on light backgrounds
    hint: '#6c7584',         // Placeholder text
  },

  // Border colors
  border: {
    primary: '#2a2f3a',      // Primary borders
    secondary: '#1e2329',    // Secondary borders
    focus: '#1c96ff',        // Focus state borders
    hover: '#3a4252',        // Hover state borders
    divider: '#1a1f2e',      // Divider lines
  },

  // Status colors
  status: {
    online: '#00e676',       // Bot online status
    offline: '#ff5252',      // Bot offline status
    warning: '#ffc107',      // Warning status
    pending: '#ff9800',      // Pending status
    error: '#f44336',        // Error status
    info: '#2196f3',         // Info status
  },

  // Chart colors for data visualization
  chart: {
    // Candlestick colors
    bullish: '#00e676',      // Green candles
    bearish: '#ff5252',      // Red candles
    
    // Line chart colors (categorical)
    line1: '#1c96ff',        // Primary blue
    line2: '#00d4ff',        // Cyan
    line3: '#7c4dff',        // Purple
    line4: '#ff4081',        // Pink
    line5: '#00e676',        // Green
    line6: '#ffc107',        // Yellow
    line7: '#ff5722',        // Orange
    line8: '#9c27b0',        // Purple variant
    
    // Grid and axis colors
    grid: '#1a1f2e',         // Chart grid lines
    axis: '#3a4252',         // Chart axes
    crosshair: '#90a4ae',    // Crosshair lines
    
    // Volume colors
    volumeBuy: 'rgba(0, 230, 118, 0.6)',
    volumeSell: 'rgba(255, 82, 82, 0.6)',
  },

  // Shadow colors
  shadow: {
    light: 'rgba(0, 0, 0, 0.1)',
    medium: 'rgba(0, 0, 0, 0.2)',
    heavy: 'rgba(0, 0, 0, 0.3)',
    glow: 'rgba(28, 150, 255, 0.3)',
  },

  // Overlay colors
  overlay: {
    light: 'rgba(10, 14, 26, 0.4)',
    medium: 'rgba(10, 14, 26, 0.6)',
    heavy: 'rgba(10, 14, 26, 0.8)',
    modal: 'rgba(0, 0, 0, 0.7)',
  },
} as const;