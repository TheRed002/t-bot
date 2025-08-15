/**
 * Typography system for T-Bot Trading System
 * Optimized for financial data readability and professional appearance
 */

export const typography = {
  // Font families
  fontFamily: {
    primary: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    monospace: "'JetBrains Mono', 'Fira Code', 'Source Code Pro', monospace",
    numeric: "'JetBrains Mono', 'Roboto Mono', monospace", // For numbers in trading
  },

  // Font weights
  fontWeight: {
    light: 300,
    regular: 400,
    medium: 500,
    semibold: 600,
    bold: 700,
  },

  // Font sizes (in rem)
  fontSize: {
    xs: '0.75rem',    // 12px
    sm: '0.875rem',   // 14px
    base: '1rem',     // 16px
    lg: '1.125rem',   // 18px
    xl: '1.25rem',    // 20px
    '2xl': '1.5rem',  // 24px
    '3xl': '1.875rem', // 30px
    '4xl': '2.25rem', // 36px
    '5xl': '3rem',    // 48px
  },

  // Line heights
  lineHeight: {
    tight: 1.2,
    normal: 1.4,
    relaxed: 1.6,
    loose: 1.8,
  },

  // Letter spacing
  letterSpacing: {
    tighter: '-0.05em',
    tight: '-0.025em',
    normal: '0',
    wide: '0.025em',
    wider: '0.05em',
    widest: '0.1em',
  },

  // Text styles for specific use cases
  variants: {
    // Headings
    h1: {
      fontSize: '2.25rem',
      fontWeight: 700,
      lineHeight: 1.2,
      letterSpacing: '-0.025em',
    },
    h2: {
      fontSize: '1.875rem',
      fontWeight: 600,
      lineHeight: 1.3,
      letterSpacing: '-0.025em',
    },
    h3: {
      fontSize: '1.5rem',
      fontWeight: 600,
      lineHeight: 1.3,
    },
    h4: {
      fontSize: '1.25rem',
      fontWeight: 600,
      lineHeight: 1.4,
    },
    h5: {
      fontSize: '1.125rem',
      fontWeight: 600,
      lineHeight: 1.4,
    },
    h6: {
      fontSize: '1rem',
      fontWeight: 600,
      lineHeight: 1.4,
    },

    // Body text
    body1: {
      fontSize: '1rem',
      fontWeight: 400,
      lineHeight: 1.6,
    },
    body2: {
      fontSize: '0.875rem',
      fontWeight: 400,
      lineHeight: 1.5,
    },

    // Financial data specific
    price: {
      fontFamily: "'JetBrains Mono', monospace",
      fontSize: '1rem',
      fontWeight: 600,
      letterSpacing: '0.025em',
      lineHeight: 1.2,
    },
    priceSmall: {
      fontFamily: "'JetBrains Mono', monospace",
      fontSize: '0.875rem',
      fontWeight: 500,
      letterSpacing: '0.025em',
      lineHeight: 1.2,
    },
    priceLarge: {
      fontFamily: "'JetBrains Mono', monospace",
      fontSize: '1.25rem',
      fontWeight: 700,
      letterSpacing: '0.025em',
      lineHeight: 1.2,
    },

    // Percentage changes
    percentage: {
      fontFamily: "'JetBrains Mono', monospace",
      fontSize: '0.875rem',
      fontWeight: 600,
      letterSpacing: '0.025em',
    },

    // Currency amounts
    currency: {
      fontFamily: "'JetBrains Mono', monospace",
      fontSize: '1rem',
      fontWeight: 500,
      letterSpacing: '0.025em',
    },

    // Code and technical data
    code: {
      fontFamily: "'JetBrains Mono', monospace",
      fontSize: '0.875rem',
      fontWeight: 400,
      letterSpacing: '0.025em',
      lineHeight: 1.4,
    },

    // Labels and captions
    label: {
      fontSize: '0.875rem',
      fontWeight: 500,
      lineHeight: 1.4,
      letterSpacing: '0.025em',
    },
    caption: {
      fontSize: '0.75rem',
      fontWeight: 400,
      lineHeight: 1.4,
      letterSpacing: '0.025em',
    },

    // Button text
    button: {
      fontSize: '0.875rem',
      fontWeight: 600,
      lineHeight: 1.2,
      letterSpacing: '0.025em',
      textTransform: 'none' as const,
    },

    // Navigation
    nav: {
      fontSize: '0.875rem',
      fontWeight: 500,
      lineHeight: 1.4,
    },

    // Table headers
    tableHeader: {
      fontSize: '0.75rem',
      fontWeight: 600,
      lineHeight: 1.2,
      letterSpacing: '0.05em',
      textTransform: 'uppercase' as const,
    },

    // Metric values (dashboard)
    metric: {
      fontFamily: "'JetBrains Mono', monospace",
      fontSize: '1.5rem',
      fontWeight: 700,
      lineHeight: 1.2,
      letterSpacing: '-0.025em',
    },
    metricLabel: {
      fontSize: '0.75rem',
      fontWeight: 500,
      lineHeight: 1.2,
      letterSpacing: '0.05em',
      textTransform: 'uppercase' as const,
    },

    // Status indicators
    status: {
      fontSize: '0.75rem',
      fontWeight: 600,
      lineHeight: 1.2,
      letterSpacing: '0.05em',
      textTransform: 'uppercase' as const,
    },
  },
} as const;