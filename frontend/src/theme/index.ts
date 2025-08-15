/**
 * Main theme configuration for T-Bot Trading System
 * Combines all theme modules into a cohesive Material-UI theme
 */

import { createTheme, ThemeOptions } from '@mui/material/styles';
import { colors } from './colors';
import { typography } from './typography';
import { spacing } from './spacing';
import { breakpoints } from './breakpoints';
import { components } from './components';

// Create the base theme configuration
const themeOptions: ThemeOptions = {
  palette: {
    mode: 'dark',
    
    // Primary colors
    primary: {
      main: colors.primary[500],
      light: colors.primary[300],
      dark: colors.primary[700],
      contrastText: colors.text.primary,
    },

    // Secondary colors (using accent cyan)
    secondary: {
      main: colors.accent.cyan,
      light: colors.accent.teal,
      dark: colors.accent.purple,
      contrastText: colors.text.primary,
    },

    // Error colors
    error: {
      main: colors.financial.loss,
      light: '#ff867c',
      dark: '#d32f2f',
      contrastText: colors.text.primary,
    },

    // Warning colors
    warning: {
      main: colors.financial.warning,
      light: '#ffb74d',
      dark: '#f57c00',
      contrastText: colors.background.primary,
    },

    // Info colors
    info: {
      main: colors.status.info,
      light: '#64b5f6',
      dark: '#1976d2',
      contrastText: colors.text.primary,
    },

    // Success colors
    success: {
      main: colors.financial.profit,
      light: '#81c784',
      dark: '#388e3c',
      contrastText: colors.text.primary,
    },

    // Background colors
    background: {
      default: colors.background.primary,
      paper: colors.background.secondary,
    },

    // Text colors
    text: {
      primary: colors.text.primary,
      secondary: colors.text.secondary,
      disabled: colors.text.muted,
    },

    // Divider color
    divider: colors.border.divider,

    // Action colors
    action: {
      active: colors.text.primary,
      hover: colors.background.tertiary,
      selected: colors.background.surface,
      disabled: colors.text.muted,
      disabledBackground: colors.background.tertiary,
    },
  },

  // Typography configuration
  typography: {
    fontFamily: typography.fontFamily.primary,
    
    // Base font sizes
    fontSize: 14,
    htmlFontSize: 16,

    // Typography variants
    h1: typography.variants.h1,
    h2: typography.variants.h2,
    h3: typography.variants.h3,
    h4: typography.variants.h4,
    h5: typography.variants.h5,
    h6: typography.variants.h6,
    body1: typography.variants.body1,
    body2: typography.variants.body2,
    button: typography.variants.button,
    caption: typography.variants.caption,
    overline: typography.variants.label,
  },

  // Spacing configuration
  spacing: spacing.base,

  // Shape configuration
  shape: {
    borderRadius: spacing.borderRadius.md,
  },

  // Breakpoints configuration
  breakpoints: {
    values: breakpoints.values,
  },

  // Z-index configuration
  zIndex: {
    mobileStepper: spacing.zIndex.docked,
    speedDial: spacing.zIndex.sticky,
    appBar: spacing.zIndex.sticky,
    drawer: spacing.zIndex.dropdown,
    modal: spacing.zIndex.modal,
    snackbar: spacing.zIndex.toast,
    tooltip: spacing.zIndex.tooltip,
  },

  // Component style overrides
  components,
};

// Create and export the theme
export const theme = createTheme(themeOptions);

// Export theme modules for direct access
export { colors } from './colors';
export { typography } from './typography';
export { spacing } from './spacing';
export { breakpoints } from './breakpoints';
export { components } from './components';

// Custom theme augmentation for additional properties
declare module '@mui/material/styles' {
  interface Theme {
    custom: {
      colors: typeof colors;
      spacing: typeof spacing;
      typography: typeof typography;
      breakpoints: typeof breakpoints;
    };
  }

  interface ThemeOptions {
    custom?: {
      colors?: typeof colors;
      spacing?: typeof spacing;
      typography?: typeof typography;
      breakpoints?: typeof breakpoints;
    };
  }
}

// Add custom properties to theme
export const extendedTheme = createTheme({
  ...themeOptions,
  custom: {
    colors,
    spacing,
    typography,
    breakpoints,
  },
});

// Trading-specific theme utilities
export const tradingTheme = {
  // Price change colors
  getPriceChangeColor: (change: number) => {
    if (change > 0) return colors.financial.profit;
    if (change < 0) return colors.financial.loss;
    return colors.financial.neutral;
  },

  // Status colors
  getStatusColor: (status: string) => {
    switch (status.toLowerCase()) {
      case 'online':
      case 'active':
      case 'running':
        return colors.status.online;
      case 'offline':
      case 'stopped':
      case 'inactive':
        return colors.status.offline;
      case 'warning':
      case 'degraded':
        return colors.status.warning;
      case 'pending':
      case 'loading':
        return colors.status.pending;
      case 'error':
      case 'failed':
        return colors.status.error;
      default:
        return colors.status.info;
    }
  },

  // Chart colors for data series
  getChartColor: (index: number) => {
    const chartColors = [
      colors.chart.line1,
      colors.chart.line2,
      colors.chart.line3,
      colors.chart.line4,
      colors.chart.line5,
      colors.chart.line6,
      colors.chart.line7,
      colors.chart.line8,
    ];
    return chartColors[index % chartColors.length];
  },

  // Responsive helpers
  isMobile: (theme: any) => theme.breakpoints.down('md'),
  isTablet: (theme: any) => theme.breakpoints.between('md', 'lg'),
  isDesktop: (theme: any) => theme.breakpoints.up('lg'),
};