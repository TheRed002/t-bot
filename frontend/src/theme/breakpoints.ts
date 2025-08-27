/**
 * Responsive breakpoints for T-Bot Trading System
 * Mobile-first responsive design breakpoints
 */

// Breakpoint values (in pixels)
const breakpointValues = {
  xs: 0,      // Extra small devices (phones)
  sm: 576,    // Small devices (large phones)
  md: 768,    // Medium devices (tablets)
  lg: 992,    // Large devices (desktops)
  xl: 1200,   // Extra large devices (large desktops)
  xxl: 1400,  // Extra extra large devices (very large desktops)
} as const;

type BreakpointKey = keyof typeof breakpointValues;

export const breakpoints: {
  values: typeof breakpointValues;
  up: (breakpoint: BreakpointKey) => string;
  down: (breakpoint: BreakpointKey) => string;
  between: (min: BreakpointKey, max: BreakpointKey) => string;
  only: (breakpoint: BreakpointKey) => string;
  container: Record<BreakpointKey, string>;
  grid: Record<BreakpointKey, number>;
  dashboard: {
    sidebarCollapse: string;
    chartStackVertical: string;
    tableScrollHorizontal: string;
    cardSingle: string;
    cardDouble: string;
    cardTriple: string;
    cardQuad: string;
    cardSix: string;
  };
  trading: {
    orderBookStack: string;
    chartMinHeight: string;
    positionTableCompact: string;
    quickActionsCollapse: string;
  };
} = {
  values: breakpointValues,

  // Media query helpers
  up: (breakpoint: BreakpointKey) => 
    `@media (min-width: ${breakpointValues[breakpoint]}px)`,
  
  down: (breakpoint: BreakpointKey) => 
    `@media (max-width: ${breakpointValues[breakpoint] - 1}px)`,
  
  between: (min: BreakpointKey, max: BreakpointKey) =>
    `@media (min-width: ${breakpointValues[min]}px) and (max-width: ${breakpointValues[max] - 1}px)`,

  only: (breakpoint: BreakpointKey) => {
    const bps = Object.keys(breakpointValues) as Array<BreakpointKey>;
    const index = bps.indexOf(breakpoint);
    
    if (index === 0) {
      return `@media (max-width: ${breakpointValues[bps[1]] - 1}px)`;
    } else if (index === bps.length - 1) {
      return `@media (min-width: ${breakpointValues[breakpoint]}px)`;
    } else {
      return `@media (min-width: ${breakpointValues[breakpoint]}px) and (max-width: ${breakpointValues[bps[index + 1]] - 1}px)`;
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