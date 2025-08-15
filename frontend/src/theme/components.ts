/**
 * Component theme overrides for Material-UI
 * Customizes MUI components to match T-Bot design system
 */

import { colors } from './colors';
import { typography } from './typography';
import { spacing } from './spacing';

export const components = {
  // Button component overrides
  MuiButton: {
    styleOverrides: {
      root: {
        ...typography.variants.button,
        borderRadius: spacing.borderRadius.md,
        textTransform: 'none',
        boxShadow: 'none',
        '&:hover': {
          boxShadow: 'none',
        },
      },
      contained: {
        backgroundColor: colors.primary[500],
        color: colors.text.primary,
        '&:hover': {
          backgroundColor: colors.primary[600],
        },
        '&:disabled': {
          backgroundColor: colors.background.tertiary,
          color: colors.text.muted,
        },
      },
      outlined: {
        borderColor: colors.border.primary,
        color: colors.text.primary,
        '&:hover': {
          borderColor: colors.border.hover,
          backgroundColor: colors.background.tertiary,
        },
      },
      text: {
        color: colors.text.primary,
        '&:hover': {
          backgroundColor: colors.background.tertiary,
        },
      },
    },
  },

  // Card component overrides
  MuiCard: {
    styleOverrides: {
      root: {
        backgroundColor: colors.background.secondary,
        borderRadius: spacing.borderRadius.lg,
        border: `1px solid ${colors.border.primary}`,
        boxShadow: `0 4px 12px ${colors.shadow.medium}`,
      },
    },
  },

  // Paper component overrides
  MuiPaper: {
    styleOverrides: {
      root: {
        backgroundColor: colors.background.paper,
        backgroundImage: 'none',
      },
      outlined: {
        border: `1px solid ${colors.border.primary}`,
      },
    },
  },

  // TextField component overrides
  MuiTextField: {
    styleOverrides: {
      root: {
        '& .MuiOutlinedInput-root': {
          backgroundColor: colors.background.tertiary,
          borderRadius: spacing.borderRadius.md,
          '& fieldset': {
            borderColor: colors.border.secondary,
          },
          '&:hover fieldset': {
            borderColor: colors.border.hover,
          },
          '&.Mui-focused fieldset': {
            borderColor: colors.border.focus,
          },
        },
        '& .MuiInputLabel-root': {
          color: colors.text.secondary,
          ...typography.variants.label,
        },
        '& .MuiInputBase-input': {
          color: colors.text.primary,
          ...typography.variants.body2,
        },
      },
    },
  },

  // Chip component overrides
  MuiChip: {
    styleOverrides: {
      root: {
        borderRadius: spacing.borderRadius.base,
        ...typography.variants.caption,
        fontWeight: typography.fontWeight.medium,
      },
      filled: {
        backgroundColor: colors.background.surface,
        color: colors.text.primary,
      },
      outlined: {
        borderColor: colors.border.primary,
        color: colors.text.primary,
      },
    },
  },

  // Table component overrides
  MuiTableContainer: {
    styleOverrides: {
      root: {
        backgroundColor: colors.background.secondary,
        borderRadius: spacing.borderRadius.lg,
        border: `1px solid ${colors.border.primary}`,
      },
    },
  },

  MuiTableHead: {
    styleOverrides: {
      root: {
        backgroundColor: colors.background.surface,
      },
    },
  },

  MuiTableCell: {
    styleOverrides: {
      root: {
        borderColor: colors.border.divider,
        color: colors.text.primary,
        padding: spacing.component.cellPadding,
      },
      head: {
        backgroundColor: colors.background.surface,
        color: colors.text.secondary,
        ...typography.variants.tableHeader,
      },
    },
  },

  // AppBar/Toolbar overrides
  MuiAppBar: {
    styleOverrides: {
      root: {
        backgroundColor: colors.background.elevated,
        borderBottom: `1px solid ${colors.border.primary}`,
        boxShadow: `0 2px 8px ${colors.shadow.light}`,
      },
    },
  },

  MuiToolbar: {
    styleOverrides: {
      root: {
        minHeight: `${spacing.layout.headerHeight}px !important`,
        padding: `0 ${spacing.layout.containerPadding}px`,
      },
    },
  },

  // Drawer overrides
  MuiDrawer: {
    styleOverrides: {
      paper: {
        backgroundColor: colors.background.elevated,
        borderColor: colors.border.primary,
        width: spacing.layout.sidebarWidth,
      },
    },
  },

  // List component overrides
  MuiListItem: {
    styleOverrides: {
      root: {
        borderRadius: spacing.borderRadius.md,
        margin: `${spacing.component.sidebarItemSpacing}px 0`,
        '&:hover': {
          backgroundColor: colors.background.tertiary,
        },
        '&.Mui-selected': {
          backgroundColor: colors.background.surface,
          '&:hover': {
            backgroundColor: colors.background.surface,
          },
        },
      },
    },
  },

  MuiListItemText: {
    styleOverrides: {
      primary: {
        ...typography.variants.nav,
        color: colors.text.primary,
      },
      secondary: {
        ...typography.variants.caption,
        color: colors.text.secondary,
      },
    },
  },

  // Dialog overrides
  MuiDialog: {
    styleOverrides: {
      paper: {
        backgroundColor: colors.background.elevated,
        borderRadius: spacing.borderRadius.lg,
        border: `1px solid ${colors.border.primary}`,
      },
    },
  },

  MuiDialogTitle: {
    styleOverrides: {
      root: {
        ...typography.variants.h4,
        color: colors.text.primary,
        borderBottom: `1px solid ${colors.border.divider}`,
        padding: spacing.component.modalPadding,
      },
    },
  },

  MuiDialogContent: {
    styleOverrides: {
      root: {
        padding: spacing.component.modalPadding,
      },
    },
  },

  // Tooltip overrides
  MuiTooltip: {
    styleOverrides: {
      tooltip: {
        backgroundColor: colors.background.elevated,
        color: colors.text.primary,
        border: `1px solid ${colors.border.primary}`,
        borderRadius: spacing.borderRadius.md,
        ...typography.variants.caption,
        boxShadow: `0 4px 12px ${colors.shadow.medium}`,
      },
      arrow: {
        color: colors.background.elevated,
        '&::before': {
          border: `1px solid ${colors.border.primary}`,
        },
      },
    },
  },

  // Menu overrides
  MuiMenu: {
    styleOverrides: {
      paper: {
        backgroundColor: colors.background.elevated,
        border: `1px solid ${colors.border.primary}`,
        borderRadius: spacing.borderRadius.md,
        boxShadow: `0 8px 24px ${colors.shadow.heavy}`,
      },
    },
  },

  MuiMenuItem: {
    styleOverrides: {
      root: {
        ...typography.variants.body2,
        color: colors.text.primary,
        '&:hover': {
          backgroundColor: colors.background.tertiary,
        },
        '&.Mui-selected': {
          backgroundColor: colors.background.surface,
        },
      },
    },
  },

  // Progress indicators
  MuiLinearProgress: {
    styleOverrides: {
      root: {
        backgroundColor: colors.background.tertiary,
        borderRadius: spacing.borderRadius.base,
      },
      bar: {
        backgroundColor: colors.primary[500],
        borderRadius: spacing.borderRadius.base,
      },
    },
  },

  MuiCircularProgress: {
    styleOverrides: {
      root: {
        color: colors.primary[500],
      },
    },
  },

  // Switch overrides
  MuiSwitch: {
    styleOverrides: {
      root: {
        '& .MuiSwitch-switchBase': {
          '&.Mui-checked': {
            color: colors.primary[500],
            '& + .MuiSwitch-track': {
              backgroundColor: colors.primary[500],
            },
          },
        },
        '& .MuiSwitch-track': {
          backgroundColor: colors.background.tertiary,
        },
      },
    },
  },

  // Tabs overrides
  MuiTabs: {
    styleOverrides: {
      root: {
        borderBottom: `1px solid ${colors.border.primary}`,
      },
      indicator: {
        backgroundColor: colors.primary[500],
      },
    },
  },

  MuiTab: {
    styleOverrides: {
      root: {
        ...typography.variants.nav,
        textTransform: 'none',
        color: colors.text.secondary,
        '&.Mui-selected': {
          color: colors.primary[500],
        },
      },
    },
  },
} as const;