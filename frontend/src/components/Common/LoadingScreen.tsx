/**
 * Loading screen component for initial app load
 */

import React from 'react';
import { Box, CircularProgress, Typography, keyframes } from '@mui/material';
import { styled } from '@mui/material/styles';
import { colors } from '@/theme/colors';

// Animated logo rotation
const logoSpin = keyframes`
  0% {
    transform: rotate(0deg);
  }
  100% {
    transform: rotate(360deg);
  }
`;

const LoadingContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  height: '100vh',
  backgroundColor: colors.background.primary,
  color: colors.text.primary,
}));

const Logo = styled(Box)(({ theme }) => ({
  width: 64,
  height: 64,
  backgroundColor: colors.primary[500],
  borderRadius: '50%',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  marginBottom: theme.spacing(3),
  animation: `${logoSpin} 2s linear infinite`,
  fontSize: '24px',
  fontWeight: 'bold',
  color: colors.text.primary,
}));

const LoadingText = styled(Typography)(({ theme }) => ({
  color: colors.text.secondary,
  marginBottom: theme.spacing(2),
  fontSize: '14px',
}));

const ProgressContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing(2),
}));

interface LoadingScreenProps {
  message?: string;
  showProgress?: boolean;
}

const LoadingScreen: React.FC<LoadingScreenProps> = ({
  message = 'Loading T-Bot Trading System...',
  showProgress = true,
}) => {
  return (
    <LoadingContainer>
      <Logo>TB</Logo>
      
      <Typography
        variant="h4"
        component="h1"
        sx={{
          color: colors.text.primary,
          fontWeight: 600,
          marginBottom: 1,
        }}
      >
        T-Bot Trading System
      </Typography>
      
      <LoadingText>{message}</LoadingText>
      
      {showProgress && (
        <ProgressContainer>
          <CircularProgress
            size={24}
            sx={{ color: colors.primary[500] }}
          />
        </ProgressContainer>
      )}
    </LoadingContainer>
  );
};

export default LoadingScreen;