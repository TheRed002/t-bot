/**
 * Loading screen component
 * Full-screen loading indicator with animation
 */

import React from 'react';
import { cn } from '@/lib/utils';

interface LoadingScreenProps {
  message?: string;
  className?: string;
}

const LoadingScreen: React.FC<LoadingScreenProps> = ({ 
  message = 'Loading trading system...', 
  className 
}) => {
  return (
    <div className={cn(
      "fixed inset-0 z-50 flex flex-col items-center justify-center bg-background/95 backdrop-blur-sm",
      className
    )}>
      <div className="relative">
        {/* Animated rings */}
        <div className="absolute inset-0 animate-ping">
          <div className="h-24 w-24 rounded-full border-4 border-primary/30" />
        </div>
        <div className="absolute inset-0 animate-ping animation-delay-200">
          <div className="h-24 w-24 rounded-full border-4 border-primary/20" />
        </div>
        <div className="absolute inset-0 animate-ping animation-delay-400">
          <div className="h-24 w-24 rounded-full border-4 border-primary/10" />
        </div>
        
        {/* Center spinner */}
        <div className="relative flex h-24 w-24 items-center justify-center">
          <div className="h-16 w-16 animate-spin rounded-full border-4 border-muted border-t-primary" />
        </div>
      </div>
      
      {/* Loading text */}
      <div className="mt-8 space-y-2 text-center">
        <h2 className="text-xl font-semibold tracking-tight">T-Bot Trading System</h2>
        <p className="text-sm text-muted-foreground animate-pulse">{message}</p>
      </div>
      
      {/* Progress dots */}
      <div className="mt-4 flex space-x-1">
        <span className="h-2 w-2 animate-bounce rounded-full bg-primary animation-delay-0" />
        <span className="h-2 w-2 animate-bounce rounded-full bg-primary animation-delay-200" />
        <span className="h-2 w-2 animate-bounce rounded-full bg-primary animation-delay-400" />
      </div>
    </div>
  );
};

export default LoadingScreen;