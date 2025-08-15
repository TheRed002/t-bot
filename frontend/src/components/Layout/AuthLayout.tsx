/**
 * Authentication layout component
 * Simple centered layout for login/register pages
 */

import React from 'react';
import { Box, Container, Paper } from '@mui/material';
import { colors } from '@/theme/colors';

interface AuthLayoutProps {
  children: React.ReactNode;
}

const AuthLayout: React.FC<AuthLayoutProps> = ({ children }) => {
  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: '100vh',
        backgroundColor: colors.background.primary,
        backgroundImage: `
          radial-gradient(circle at 25% 25%, ${colors.primary[500]}20 0%, transparent 50%),
          radial-gradient(circle at 75% 75%, ${colors.accent.cyan}15 0%, transparent 50%)
        `,
      }}
    >
      <Container maxWidth="sm">
        <Paper
          elevation={0}
          sx={{
            p: 4,
            backgroundColor: colors.background.secondary,
            border: `1px solid ${colors.border.primary}`,
            borderRadius: 2,
          }}
        >
          {children}
        </Paper>
      </Container>
    </Box>
  );
};

export default AuthLayout;