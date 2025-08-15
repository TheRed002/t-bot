/**
 * Spacing system for T-Bot Trading System
 * Consistent spacing scale for layouts, padding, and margins
 */

export const spacing = {
  // Base spacing unit (4px)
  base: 4,

  // Spacing scale (in pixels, converted to rem in theme)
  scale: {
    0: 0,
    1: 4,      // 0.25rem
    2: 8,      // 0.5rem
    3: 12,     // 0.75rem
    4: 16,     // 1rem
    5: 20,     // 1.25rem
    6: 24,     // 1.5rem
    8: 32,     // 2rem
    10: 40,    // 2.5rem
    12: 48,    // 3rem
    16: 64,    // 4rem
    20: 80,    // 5rem
    24: 96,    // 6rem
    32: 128,   // 8rem
    40: 160,   // 10rem
    48: 192,   // 12rem
    56: 224,   // 14rem
    64: 256,   // 16rem
  },

  // Semantic spacing for specific use cases
  component: {
    // Card spacing
    cardPadding: 24,
    cardGap: 16,
    
    // Form spacing
    formPadding: 24,
    fieldSpacing: 16,
    fieldPadding: 12,
    
    // Navigation
    navPadding: 16,
    navItemSpacing: 8,
    
    // Table spacing
    tablePadding: 16,
    cellPadding: 12,
    rowSpacing: 8,
    
    // Chart spacing
    chartPadding: 20,
    chartMargin: 16,
    
    // Button spacing
    buttonPadding: 12,
    buttonSpacing: 8,
    
    // Modal spacing
    modalPadding: 32,
    modalMargin: 24,
    
    // Sidebar spacing
    sidebarPadding: 20,
    sidebarItemSpacing: 4,
    
    // Header spacing
    headerPadding: 20,
    headerHeight: 64,
    
    // Content spacing
    contentPadding: 24,
    sectionSpacing: 32,
  },

  // Layout spacing
  layout: {
    // Container widths
    containerMaxWidth: 1440,
    containerPadding: 24,
    
    // Grid spacing
    gridGap: 24,
    gridColumnGap: 20,
    gridRowGap: 24,
    
    // Sidebar dimensions
    sidebarWidth: 280,
    sidebarCollapsedWidth: 64,
    
    // Breakpoint-specific spacing
    mobile: {
      containerPadding: 16,
      gridGap: 16,
      sectionSpacing: 24,
    },
    tablet: {
      containerPadding: 20,
      gridGap: 20,
      sectionSpacing: 28,
    },
    desktop: {
      containerPadding: 24,
      gridGap: 24,
      sectionSpacing: 32,
    },
  },

  // Border radius scale
  borderRadius: {
    none: 0,
    sm: 4,
    base: 6,
    md: 8,
    lg: 12,
    xl: 16,
    '2xl': 20,
    '3xl': 24,
    full: 9999,
  },

  // Z-index scale
  zIndex: {
    hide: -1,
    auto: 'auto',
    base: 0,
    docked: 10,
    dropdown: 1000,
    sticky: 1100,
    banner: 1200,
    overlay: 1300,
    modal: 1400,
    popover: 1500,
    skipLink: 1600,
    toast: 1700,
    tooltip: 1800,
  },
} as const;