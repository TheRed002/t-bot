/**
 * Authentication layout component
 * Simple centered layout for login/register pages
 */

import React from 'react';

interface AuthLayoutProps {
  children: React.ReactNode;
}

const AuthLayout: React.FC<AuthLayoutProps> = ({ children }) => {
  return (
    <div className="min-h-screen bg-background">
      {/* Background pattern */}
      <div className="fixed inset-0 -z-10 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px]" />
      
      {/* Content */}
      <div className="relative z-10">
        {children}
      </div>
      
      {/* Footer */}
      <footer className="fixed bottom-0 w-full border-t bg-background/95 backdrop-blur-sm">
        <div className="mx-auto max-w-7xl px-4 py-3">
          <p className="text-center text-xs text-muted-foreground">
            T-Bot Trading System © {new Date().getFullYear()} - Advanced Cryptocurrency Trading Platform
          </p>
        </div>
      </footer>
    </div>
  );
};

export default AuthLayout;