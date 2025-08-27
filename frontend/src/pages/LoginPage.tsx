/**
 * Login page component
 * Handles user authentication with Shadcn/ui components
 */

import React, { useEffect } from 'react';
import { useForm, Controller } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { useNavigate } from 'react-router-dom';
import { useAppDispatch, useAppSelector } from '@/store';
import { loginUser, mockLogin, selectAuthLoading, selectAuthError, selectIsAuthenticated } from '@/store/slices/authSlice';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Checkbox } from '@/components/ui/checkbox';
import { Loader2, TrendingUp, AlertCircle } from 'lucide-react';

// Validation schema
const loginSchema = z.object({
  username: z
    .string()
    .min(1, 'Username is required')
    .min(3, 'Username must be at least 3 characters')
    .max(50, 'Username must not exceed 50 characters'),
  password: z
    .string()
    .min(1, 'Password is required')
    .min(6, 'Password must be at least 6 characters')
    .max(128, 'Password must not exceed 128 characters'),
  remember_me: z.boolean(),
});

type LoginFormData = z.infer<typeof loginSchema>;

const LoginPage: React.FC = () => {
  const dispatch = useAppDispatch();
  const navigate = useNavigate();
  const isLoading = useAppSelector(selectAuthLoading);
  const error = useAppSelector(selectAuthError);
  const isAuthenticated = useAppSelector(selectIsAuthenticated);

  const {
    control,
    handleSubmit,
    formState: { errors, isSubmitting },
    setError,
    clearErrors,
  } = useForm<LoginFormData>({
    resolver: zodResolver(loginSchema),
    defaultValues: {
      username: '',
      password: '',
      remember_me: false,
    },
    mode: 'onBlur',
  });

  // Redirect to dashboard when authenticated
  useEffect(() => {
    console.log('[LoginPage] isAuthenticated changed:', isAuthenticated);
    if (isAuthenticated) {
      console.log('[LoginPage] User is authenticated, navigating to dashboard...');
      navigate('/dashboard', { replace: true });
    }
  }, [isAuthenticated, navigate]);

  // Clear errors when user starts typing
  useEffect(() => {
    if (error) {
      const timer = setTimeout(() => {
        clearErrors();
      }, 5000); // Clear form errors after 5 seconds
      
      return () => clearTimeout(timer);
    }
    return undefined; // Explicit return for when there's no error
  }, [error, clearErrors]);

  const onSubmit = async (data: LoginFormData) => {
    try {
      clearErrors();
      console.log('[LoginPage] Submitting mock login form:', data.username, 'Remember Me:', data.remember_me);
      // Use mock login for development when backend is not available
      const result = await dispatch(mockLogin(data)).unwrap();
      console.log('[LoginPage] Mock login successful:', result);
      // Navigation will happen via the useEffect above
    } catch (error: any) {
      console.error('[LoginPage] Mock login failed:', error);
      
      // Handle specific error cases with form field errors
      const errorMessage = typeof error === 'string' ? error : error?.message || 'Login failed';
      
      if (errorMessage.includes('Invalid username or password') || errorMessage.includes('Invalid credentials')) {
        setError('username', { message: ' ' }); // Empty message to show red border
        setError('password', { message: 'Invalid username or password' });
      } else if (errorMessage.includes('Account is locked')) {
        setError('username', { message: 'Account is temporarily locked' });
      } else if (errorMessage.includes('Too many login attempts')) {
        setError('username', { message: 'Too many attempts. Please try again later.' });
      } else if (errorMessage.includes('Account is disabled')) {
        setError('username', { message: 'Account is disabled. Contact support.' });
      } else if (errorMessage.includes('Network error')) {
        setError('username', { message: 'Connection failed. Check your internet.' });
      }
      // Redux error will still be displayed in the alert for other cases
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center p-4">
      <Card className="w-full max-w-md">
        <CardHeader className="text-center">
          {/* Logo */}
          <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-primary text-primary-foreground">
            <TrendingUp className="h-8 w-8" />
          </div>
          <CardTitle className="text-2xl font-bold">Welcome to T-Bot</CardTitle>
          <CardDescription>Sign in to access your trading dashboard</CardDescription>
        </CardHeader>
        
        <CardContent>
          {/* Error message */}
          {error && (
            <Alert variant="destructive" className="mb-4">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {/* Login form */}
          <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="username">Username</Label>
              <Controller
                name="username"
                control={control}
                rules={{
                  required: 'Username is required',
                  minLength: {
                    value: 3,
                    message: 'Username must be at least 3 characters',
                  },
                }}
                render={({ field }) => (
                  <>
                    <Input
                      {...field}
                      id="username"
                      type="text"
                      placeholder="Enter your username"
                      disabled={isLoading || isSubmitting}
                      className={errors.username ? 'border-destructive' : ''}
                    />
                    {errors.username && (
                      <p className="text-sm text-destructive">{errors.username.message}</p>
                    )}
                  </>
                )}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="password">Password</Label>
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
                  <>
                    <Input
                      {...field}
                      id="password"
                      type="password"
                      placeholder="Enter your password"
                      disabled={isLoading || isSubmitting}
                      className={errors.password ? 'border-destructive' : ''}
                    />
                    {errors.password && (
                      <p className="text-sm text-destructive">{errors.password.message}</p>
                    )}
                  </>
                )}
              />
            </div>

            <div className="flex items-center space-x-2">
              <Controller
                name="remember_me"
                control={control}
                render={({ field }) => (
                  <>
                    <Checkbox
                      id="remember_me"
                      checked={field.value}
                      onCheckedChange={field.onChange}
                      disabled={isLoading || isSubmitting}
                    />
                    <Label
                      htmlFor="remember_me"
                      className="text-sm font-normal cursor-pointer"
                    >
                      Remember me
                    </Label>
                  </>
                )}
              />
            </div>

            <Button
              type="submit"
              className="w-full"
              size="lg"
              disabled={isLoading || isSubmitting}
            >
              {(isLoading || isSubmitting) ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Signing in...
                </>
              ) : (
                'Sign In'
              )}
            </Button>
          </form>

          {/* Demo credentials info */}
          <div className="mt-6 rounded-lg border bg-blue-50 p-3">
            <div className="flex items-center gap-2 mb-2">
              <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
              <p className="text-xs font-medium text-blue-700">Mock Authentication Enabled</p>
            </div>
            <div className="text-xs text-blue-600 space-y-2">
              <p>Use these demo credentials to access the dashboard:</p>
              <div className="bg-blue-100 rounded p-2 font-mono">
                <div><strong>Username:</strong> demo</div>
                <div><strong>Password:</strong> demo123</div>
              </div>
              <p className="text-blue-500">
                (Any username/password will work - this bypasses backend authentication for development)
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default LoginPage;