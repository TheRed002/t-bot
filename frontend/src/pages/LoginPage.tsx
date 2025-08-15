/**
 * Login page component
 * Handles user authentication
 */

import React, { useState } from 'react';
import {
  Box,
  TextField,
  Button,
  Typography,
  Alert,
  CircularProgress,
} from '@mui/material';
import { useForm, Controller } from 'react-hook-form';
import { useAppDispatch, useAppSelector } from '@/store';
import { loginUser, selectAuthLoading, selectAuthError } from '@/store/slices/authSlice';
import { colors } from '@/theme/colors';

interface LoginFormData {
  email: string;
  password: string;
}

const LoginPage: React.FC = () => {
  const dispatch = useAppDispatch();
  const isLoading = useAppSelector(selectAuthLoading);
  const error = useAppSelector(selectAuthError);

  const {
    control,
    handleSubmit,
    formState: { errors },
  } = useForm<LoginFormData>({
    defaultValues: {
      email: '',
      password: '',
    },
  });

  const onSubmit = async (data: LoginFormData) => {
    try {
      await dispatch(loginUser(data)).unwrap();
    } catch (error) {
      // Error is handled by the slice
      console.error('Login failed:', error);
    }
  };

  return (
    <Box>
      {/* Logo and title */}
      <Box sx={{ textAlign: 'center', mb: 4 }}>
        <Box
          sx={{
            width: 64,
            height: 64,
            backgroundColor: colors.primary[500],
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            margin: '0 auto 16px',
            fontSize: '24px',
            fontWeight: 'bold',
            color: colors.text.primary,
          }}
        >
          TB
        </Box>
        <Typography variant="h4" component="h1" gutterBottom>
          Welcome to T-Bot
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Sign in to access your trading dashboard
        </Typography>
      </Box>

      {/* Error message */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {/* Login form */}
      <form onSubmit={handleSubmit(onSubmit)}>
        <Controller
          name="email"
          control={control}
          rules={{
            required: 'Email is required',
            pattern: {
              value: /^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$/i,
              message: 'Invalid email address',
            },
          }}
          render={({ field }) => (
            <TextField
              {...field}
              fullWidth
              label="Email"
              type="email"
              variant="outlined"
              margin="normal"
              error={!!errors.email}
              helperText={errors.email?.message}
              disabled={isLoading}
            />
          )}
        />

        <Controller
          name="password"
          control={control}
          rules={{
            required: 'Password is required',
            minLength: {
              value: 6,
              message: 'Password must be at least 6 characters',
            },
          }}
          render={({ field }) => (
            <TextField
              {...field}
              fullWidth
              label="Password"
              type="password"
              variant="outlined"
              margin="normal"
              error={!!errors.password}
              helperText={errors.password?.message}
              disabled={isLoading}
            />
          )}
        />

        <Button
          type="submit"
          fullWidth
          variant="contained"
          size="large"
          disabled={isLoading}
          sx={{ mt: 3, mb: 2 }}
        >
          {isLoading ? (
            <CircularProgress size={24} color="inherit" />
          ) : (
            'Sign In'
          )}
        </Button>
      </form>

      {/* Demo credentials for testing */}
      <Box sx={{ mt: 3, p: 2, backgroundColor: colors.background.tertiary, borderRadius: 1 }}>
        <Typography variant="caption" color="text.secondary">
          Demo credentials for testing:
          <br />
          Email: demo@tbot.com
          <br />
          Password: demo123
        </Typography>
      </Box>
    </Box>
  );
};

export default LoginPage;