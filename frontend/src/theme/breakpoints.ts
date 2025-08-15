/**
 * Responsive breakpoints for T-Bot Trading System
 * Mobile-first responsive design breakpoints
 */

export const breakpoints = {
  // Breakpoint values (in pixels)
  values: {
    xs: 0,      // Extra small devices (phones)
    sm: 576,    // Small devices (large phones)
    md: 768,    // Medium devices (tablets)
    lg: 992,    // Large devices (desktops)
    xl: 1200,   // Extra large devices (large desktops)
    xxl: 1400,  // Extra extra large devices (very large desktops)
  },

  // Media query helpers
  up: (breakpoint: keyof typeof breakpoints.values) => 
    `@media (min-width: ${breakpoints.values[breakpoint]}px)`,
  
  down: (breakpoint: keyof typeof breakpoints.values) => 
    `@media (max-width: ${breakpoints.values[breakpoint] - 1}px)`,
  
  between: (min: keyof typeof breakpoints.values, max: keyof typeof breakpoints.values) =>
    `@media (min-width: ${breakpoints.values[min]}px) and (max-width: ${breakpoints.values[max] - 1}px)`,

  only: (breakpoint: keyof typeof breakpoints.values) => {
    const bps = Object.keys(breakpoints.values) as Array<keyof typeof breakpoints.values>;
    const index = bps.indexOf(breakpoint);
    
    if (index === 0) {
      return breakpoints.down(bps[1]);
    } else if (index === bps.length - 1) {
      return breakpoints.up(breakpoint);
    } else {
      return breakpoints.between(breakpoint, bps[index + 1]);
    }
  },

  // Container max widths for each breakpoint
  container: {
    xs: '100%',
    sm: '540px',
    md: '720px',
    lg: '960px',
    xl: '1140px',
    xxl: '1320px',
  },

  // Grid columns for each breakpoint
  grid: {
    xs: 1,
    sm: 2,
    md: 3,
    lg: 4,
    xl: 6,
    xxl: 8,
  },

  // Dashboard specific breakpoints
  dashboard: {
    // Sidebar behavior
    sidebarCollapse: 'md', // Collapse sidebar below medium screens
    
    // Chart responsiveness
    chartStackVertical: 'sm', // Stack charts vertically on small screens
    
    // Table behavior
    tableScrollHorizontal: 'md', // Enable horizontal scroll below medium
    
    // Card grid
    cardSingle: 'xs',  // 1 card per row
    cardDouble: 'sm',  // 2 cards per row
    cardTriple: 'md',  // 3 cards per row
    cardQuad: 'lg',    // 4 cards per row
    cardSix: 'xl',     // 6 cards per row
  },

  // Trading interface specific
  trading: {
    // Order book layout
    orderBookStack: 'md',     // Stack order book vertically
    
    // Chart and trading controls
    chartMinHeight: 'sm',     // Minimum chart height
    
    // Position table
    positionTableCompact: 'lg', // Use compact table layout
    
    // Quick actions
    quickActionsCollapse: 'sm', // Collapse quick actions menu
  },
} as const;