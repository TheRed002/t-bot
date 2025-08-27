/**
 * Main layout component for authenticated users
 * Modern dashboard layout with Shadcn/ui components
 */

import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import {
  Sheet,
  SheetContent,
  SheetTrigger,
} from '@/components/ui/sheet';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import {
  Menu,
  Bell,
  User,
  LogOut,
  Settings,
  HelpCircle,
  Moon,
  Sun,
  X,
} from 'lucide-react';
import { useAppDispatch, useAppSelector } from '@/store';
import { toggleSidebar, setTheme } from '@/store/slices/uiSlice';
import { logoutUser } from '@/store/slices/authSlice';
import { selectUser } from '@/store/slices/authSlice';
import Navigation from './Navigation';

interface MainLayoutProps {
  children: React.ReactNode;
}

const MainLayout: React.FC<MainLayoutProps> = ({ children }) => {
  const navigate = useNavigate();
  const dispatch = useAppDispatch();
  const user = useAppSelector(selectUser);
  const { sidebar, theme: currentTheme } = useAppSelector((state) => state.ui);
  
  const [isMobile, setIsMobile] = useState(false);
  
  // Check if mobile on mount and resize
  React.useEffect(() => {
    const checkMobile = () => setIsMobile(window.innerWidth < 768);
    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);
  
  const handleLogout = async () => {
    await dispatch(logoutUser());
    navigate('/login');
  };
  
  const handleToggleSidebar = () => {
    dispatch(toggleSidebar());
  };
  
  const handleThemeToggle = () => {
    dispatch(setTheme(currentTheme === 'dark' ? 'light' : 'dark'));
  };
  
  const getUserInitials = () => {
    if (!user) return 'U';
    const names = user.username.split(' ');
    if (names.length >= 2) {
      return `${names[0][0]}${names[1][0]}`.toUpperCase();
    }
    return user.username.substring(0, 2).toUpperCase();
  };

  return (
    <TooltipProvider>
      <div className="flex min-h-screen bg-background">
        {/* Top Bar */}
        <div className="fixed top-0 left-0 right-0 z-50 h-16 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
          <div className="flex h-16 items-center gap-4 px-6">
            {/* Menu Toggle */}
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={handleToggleSidebar}
                  className="hover:bg-accent"
                >
                  {sidebar.isOpen && isMobile ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>Toggle Sidebar</p>
              </TooltipContent>
            </Tooltip>
            
            {/* Logo and Title */}
            <div className="flex items-center gap-3 flex-1">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-red-600 to-red-700 flex items-center justify-center text-white font-bold text-lg">
                  T
                </div>
                {!isMobile && (
                  <h1 className="text-xl font-semibold bg-gradient-to-r from-red-600 to-red-700 bg-clip-text text-transparent">
                    T-Bot Trading System
                  </h1>
                )}
              </div>
              
              {/* Status Badge */}
              {!isMobile && (
                <Badge variant="outline" className="ml-4 border-green-500/20 bg-green-500/10 text-green-500">
                  <div className="w-2 h-2 rounded-full bg-green-500 mr-2 animate-pulse" />
                  Live
                </Badge>
              )}
            </div>
            
            {/* Right side actions */}
            <div className="flex items-center gap-2">
              {/* Theme Toggle */}
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={handleThemeToggle}
                  >
                    {currentTheme === 'dark' ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Switch to {currentTheme === 'dark' ? 'light' : 'dark'} mode</p>
                </TooltipContent>
              </Tooltip>
              
              {/* Notifications */}
              <DropdownMenu>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <DropdownMenuTrigger asChild>
                      <Button variant="ghost" size="icon" className="relative">
                        <Bell className="h-4 w-4" />
                        <Badge className="absolute -top-1 -right-1 h-5 w-5 text-xs" variant="destructive">
                          3
                        </Badge>
                      </Button>
                    </DropdownMenuTrigger>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Notifications</p>
                  </TooltipContent>
                </Tooltip>
                <DropdownMenuContent align="end" className="w-80">
                  <DropdownMenuLabel className="flex justify-between items-center">
                    Notifications
                    <Button variant="ghost" size="sm">Mark all as read</Button>
                  </DropdownMenuLabel>
                  <DropdownMenuSeparator />
                  <div className="p-4 text-center text-muted-foreground">
                    No new notifications
                  </div>
                </DropdownMenuContent>
              </DropdownMenu>
              
              {/* Help */}
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => navigate('/help')}
                  >
                    <HelpCircle className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Help & Documentation</p>
                </TooltipContent>
              </Tooltip>
              
              {/* User Menu */}
              <DropdownMenu>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <DropdownMenuTrigger asChild>
                      <Button variant="ghost" className="relative h-9 w-9 rounded-full">
                        <Avatar className="h-9 w-9">
                          <AvatarFallback className="bg-gradient-to-br from-red-600 to-red-700 text-white font-semibold">
                            {getUserInitials()}
                          </AvatarFallback>
                        </Avatar>
                      </Button>
                    </DropdownMenuTrigger>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Account settings</p>
                  </TooltipContent>
                </Tooltip>
                <DropdownMenuContent align="end" className="w-56">
                  <DropdownMenuLabel>
                    <div>
                      <p className="font-semibold">{user?.username || 'User'}</p>
                      <p className="text-xs text-muted-foreground">{user?.email || 'user@example.com'}</p>
                    </div>
                  </DropdownMenuLabel>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem onClick={() => navigate('/profile')} className="cursor-pointer">
                    <User className="mr-2 h-4 w-4" />
                    Profile
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={() => navigate('/settings')} className="cursor-pointer">
                    <Settings className="mr-2 h-4 w-4" />
                    Settings
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem onClick={handleLogout} className="cursor-pointer text-red-600 focus:text-red-600">
                    <LogOut className="mr-2 h-4 w-4" />
                    Logout
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          </div>
        </div>

        {/* Sidebar */}
        {isMobile ? (
          <Sheet open={sidebar.isOpen} onOpenChange={handleToggleSidebar}>
            <SheetContent side="left" className="p-0 w-80">
              <Navigation />
            </SheetContent>
          </Sheet>
        ) : (
          <div 
            className={cn(
              "fixed left-0 top-16 bottom-0 z-40 border-r bg-background transition-transform duration-300 ease-in-out",
              sidebar.isOpen ? "translate-x-0" : "-translate-x-full",
              "w-80"
            )}
          >
            <Navigation />
          </div>
        )}

        {/* Main Content Area */}
        <main 
          className={cn(
            "flex-1 pt-16 transition-all duration-300 ease-in-out",
            !isMobile && sidebar.isOpen ? "ml-80" : "ml-0"
          )}
        >
          <div className="p-6 min-h-screen bg-background">
            {children}
          </div>
        </main>
      </div>
    </TooltipProvider>
  );
};

export default MainLayout;