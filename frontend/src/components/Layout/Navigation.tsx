/**
 * Modern navigation component with Shadcn/ui
 * Features tooltips, better visual hierarchy, and smooth animations
 */

import React, { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import {
  LayoutDashboard,
  TrendingUp,
  Bot,
  Wallet,
  Brain,
  Shield,
  FlaskConical,
  BarChart3,
  Settings,
  HelpCircle,
  ChevronDown,
  ChevronRight,
  Activity,
} from 'lucide-react';

interface NavigationItem {
  path: string;
  label: string;
  icon: React.ReactNode;
  description: string;
  badge?: number | string;
  status?: 'active' | 'beta' | 'new';
  children?: NavigationItem[];
}

const navigationItems: NavigationItem[] = [
  {
    path: '/',
    label: 'Dashboard',
    icon: <LayoutDashboard className="h-5 w-5" />,
    description: 'Real-time system overview and key performance metrics',
  },
  {
    path: '/trading',
    label: 'Trading',
    icon: <TrendingUp className="h-5 w-5" />,
    description: 'Live trading interface with order management',
    badge: '3',
  },
  {
    path: '/bots',
    label: 'Bots',
    icon: <Bot className="h-5 w-5" />,
    description: 'Create, configure, and monitor trading bots',
    status: 'active',
  },
  {
    path: '/portfolio',
    label: 'Portfolio',
    icon: <Wallet className="h-5 w-5" />,
    description: 'Track positions, balances, and P&L across exchanges',
  },
  {
    path: '/strategies',
    label: 'Strategies',
    icon: <Brain className="h-5 w-5" />,
    description: 'Develop, test, and optimize trading strategies',
    status: 'beta',
  },
  {
    path: '/risk',
    label: 'Risk',
    icon: <Shield className="h-5 w-5" />,
    description: 'Monitor risk metrics and set safety controls',
  },
  {
    path: '/playground',
    label: 'Playground',
    icon: <FlaskConical className="h-5 w-5" />,
    description: 'Experiment with strategies in a safe environment',
    status: 'new',
  },
];

const bottomNavigationItems: NavigationItem[] = [
  {
    path: '/analytics',
    label: 'Analytics',
    icon: <BarChart3 className="h-5 w-5" />,
    description: 'Advanced analytics and reporting',
    children: [
      {
        path: '/analytics/performance',
        label: 'Performance',
        icon: <TrendingUp className="h-4 w-4" />,
        description: 'Detailed performance metrics',
      },
      {
        path: '/analytics/metrics',
        label: 'Metrics',
        icon: <BarChart3 className="h-4 w-4" />,
        description: 'Custom metrics and KPIs',
      },
    ],
  },
  {
    path: '/settings',
    label: 'Settings',
    icon: <Settings className="h-5 w-5" />,
    description: 'System configuration and preferences',
  },
  {
    path: '/help',
    label: 'Help',
    icon: <HelpCircle className="h-5 w-5" />,
    description: 'Documentation, tutorials, and support',
  },
];

const Navigation: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [expandedItems, setExpandedItems] = useState<string[]>([]);

  const handleNavigate = (path: string) => {
    navigate(path);
  };

  const handleToggleExpand = (path: string) => {
    setExpandedItems((prev) =>
      prev.includes(path)
        ? prev.filter((item) => item !== path)
        : [...prev, path]
    );
  };

  const isActive = (path: string) => {
    if (path === '/') {
      return location.pathname === '/' || location.pathname === '/dashboard';
    }
    return location.pathname.startsWith(path);
  };

  const getStatusColor = (status?: string) => {
    switch (status) {
      case 'active':
        return 'bg-green-500/10 text-green-500 border-green-500/20';
      case 'beta':
        return 'bg-yellow-500/10 text-yellow-500 border-yellow-500/20';
      case 'new':
        return 'bg-blue-500/10 text-blue-500 border-blue-500/20';
      default:
        return '';
    }
  };

  const renderNavigationItem = (item: NavigationItem, depth = 0) => {
    const hasChildren = item.children && item.children.length > 0;
    const isExpanded = expandedItems.includes(item.path);
    const active = isActive(item.path);

    return (
      <div key={item.path} className="mb-1">
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              onClick={() => {
                if (hasChildren) {
                  handleToggleExpand(item.path);
                } else {
                  handleNavigate(item.path);
                }
              }}
              className={cn(
                "w-full justify-start h-auto p-3 text-left group relative",
                depth > 0 && "ml-4",
                active && "bg-red-500/10 text-red-600 border-l-2 border-red-600",
                !active && "hover:bg-accent/50 text-muted-foreground hover:text-foreground"
              )}
            >
              <div className="flex items-center gap-3 flex-1">
                <div className={cn("flex-shrink-0", active && "text-red-600")}>
                  {item.badge ? (
                    <div className="relative">
                      {item.icon}
                      <Badge 
                        variant="destructive" 
                        className="absolute -top-2 -right-2 h-5 w-5 text-xs p-0 flex items-center justify-center"
                      >
                        {item.badge}
                      </Badge>
                    </div>
                  ) : (
                    item.icon
                  )}
                </div>
                
                <div className="flex items-center gap-2 flex-1 min-w-0">
                  <span className={cn("font-medium truncate", active && "text-red-600")}>
                    {item.label}
                  </span>
                  {item.status && (
                    <Badge 
                      variant="outline" 
                      className={cn("text-xs px-2 py-0", getStatusColor(item.status))}
                    >
                      {item.status}
                    </Badge>
                  )}
                </div>
                
                {hasChildren && (
                  <div className="flex-shrink-0">
                    {isExpanded ? (
                      <ChevronDown className="h-4 w-4" />
                    ) : (
                      <ChevronRight className="h-4 w-4" />
                    )}
                  </div>
                )}
              </div>
            </Button>
          </TooltipTrigger>
          <TooltipContent side="right" className="max-w-xs">
            <p>{item.description}</p>
          </TooltipContent>
        </Tooltip>
        
        {hasChildren && isExpanded && (
          <div className="ml-6 mt-2 space-y-1">
            {item.children!.map((child) => renderNavigationItem(child, depth + 1))}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="flex flex-col h-full bg-background">
      {/* Header with branding */}
      <div className="p-6 border-b bg-gradient-to-br from-red-500/5 to-red-600/5">
        <div className="flex items-center gap-4">
          <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-red-600 to-red-700 flex items-center justify-center text-white font-bold text-xl shadow-lg shadow-red-500/30">
            T
          </div>
          <div>
            <h2 className="text-lg font-bold bg-gradient-to-r from-red-600 to-red-700 bg-clip-text text-transparent">
              T-Bot
            </h2>
            <p className="text-xs text-muted-foreground font-medium">
              Trading System v1.0
            </p>
          </div>
        </div>
      </div>

      {/* Main Navigation */}
      <div className="flex-1 overflow-y-auto p-4">
        <nav className="space-y-2">
          {navigationItems.map((item) => renderNavigationItem(item))}
        </nav>

        <Separator className="my-6" />

        {/* Bottom Navigation */}
        <nav className="space-y-2">
          {bottomNavigationItems.map((item) => renderNavigationItem(item))}
        </nav>
      </div>

      {/* Status Footer */}
      <div className="p-4 border-t bg-background/50 backdrop-blur">
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-muted-foreground">System Status</span>
            <div className="flex items-center gap-2">
              <Activity className="h-3 w-3 text-green-500 animate-pulse" />
              <span className="text-xs font-semibold text-green-500">Online</span>
            </div>
          </div>
          
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="text-muted-foreground">Active Bots</span>
              <span className="font-semibold">5 / 10</span>
            </div>
            <div className="flex items-center justify-between text-xs">
              <span className="text-muted-foreground">API Calls</span>
              <span className="font-semibold">1,234 / 10K</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Navigation;