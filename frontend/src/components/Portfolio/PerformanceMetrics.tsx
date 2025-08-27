/**
 * PerformanceMetrics Component - Display detailed performance analytics
 * Shows Sharpe ratio, win rate, profit factor, and other key metrics
 */

import React from 'react';
import { motion } from 'framer-motion';

// Shadcn/ui components
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Skeleton } from '@/components/ui/skeleton';

// Lucide React icons
import {
  TrendingUp,
  TrendingDown,
  Activity,
  Gauge,
  LineChart,
  BarChart3,
  Info,
  HelpCircle,
  Trophy,
  AlertTriangle,
} from 'lucide-react';

import {
  ResponsiveContainer,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as ChartTooltip,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from 'recharts';

interface PerformanceMetricsProps {
  summary?: any;
  isLoading?: boolean;
}

const PerformanceMetrics: React.FC<PerformanceMetricsProps> = ({
  summary,
  isLoading = false,
}) => {
  // Mock data for charts
  const performanceData = [
    { month: 'Jan', profit: 1200, trades: 45, winRate: 68 },
    { month: 'Feb', profit: 1800, trades: 52, winRate: 72 },
    { month: 'Mar', profit: -500, trades: 38, winRate: 45 },
    { month: 'Apr', profit: 2200, trades: 61, winRate: 75 },
    { month: 'May', profit: 3100, trades: 73, winRate: 78 },
    { month: 'Jun', profit: 2800, trades: 68, winRate: 71 },
  ];

  const radarData = [
    { metric: 'Win Rate', value: 75, fullMark: 100 },
    { metric: 'Profit Factor', value: 85, fullMark: 100 },
    { metric: 'Risk/Reward', value: 70, fullMark: 100 },
    { metric: 'Consistency', value: 80, fullMark: 100 },
    { metric: 'Recovery', value: 65, fullMark: 100 },
    { metric: 'Efficiency', value: 72, fullMark: 100 },
  ];

  const metrics = [
    {
      label: 'Sharpe Ratio',
      value: summary?.sharpeRatio || 1.85,
      target: 2.0,
      description: 'Risk-adjusted returns metric',
      icon: <Gauge className="h-4 w-4" />,
      color: 'text-blue-600',
      bgColor: 'bg-blue-50',
      format: (v: number) => v.toFixed(2),
    },
    {
      label: 'Win Rate',
      value: summary?.winRate || 68.5,
      target: 70,
      description: 'Percentage of profitable trades',
      icon: <Trophy className="h-4 w-4" />,
      color: 'text-green-600',
      bgColor: 'bg-green-50',
      format: (v: number) => `${v.toFixed(1)}%`,
    },
    {
      label: 'Profit Factor',
      value: 2.35,
      target: 2.0,
      description: 'Gross profit / Gross loss ratio',
      icon: <TrendingUp className="h-4 w-4" />,
      color: 'text-emerald-600',
      bgColor: 'bg-emerald-50',
      format: (v: number) => v.toFixed(2),
    },
    {
      label: 'Max Drawdown',
      value: Math.abs(summary?.maxDrawdown || -8.5),
      target: 10,
      description: 'Maximum peak-to-trough decline',
      icon: <TrendingDown className="h-4 w-4" />,
      color: 'text-orange-600',
      bgColor: 'bg-orange-50',
      format: (v: number) => `-${v.toFixed(1)}%`,
      inverse: true,
    },
    {
      label: 'Avg Win/Loss',
      value: 1.85,
      target: 2.0,
      description: 'Average win / Average loss ratio',
      icon: <Activity className="h-4 w-4" />,
      color: 'text-purple-600',
      bgColor: 'bg-purple-50',
      format: (v: number) => v.toFixed(2),
    },
    {
      label: 'Recovery Factor',
      value: 3.2,
      target: 3.0,
      description: 'Net profit / Max drawdown ratio',
      icon: <LineChart className="h-4 w-4" />,
      color: 'text-indigo-600',
      bgColor: 'bg-indigo-50',
      format: (v: number) => v.toFixed(1),
    },
  ];

  const getProgressColor = (value: number, target: number, inverse = false) => {
    const ratio = value / target;
    if (inverse) {
      if (ratio <= 1) return 'bg-green-500';
      if (ratio <= 1.5) return 'bg-yellow-500';
      return 'bg-red-500';
    }
    if (ratio >= 1) return 'bg-green-500';
    if (ratio >= 0.7) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value);
  };

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[...Array(6)].map((_, index) => (
            <Skeleton key={index} className="h-32" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Key Metrics Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {metrics.map((metric, index) => (
          <motion.div
            key={metric.label}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: index * 0.05 }}
          >
            <Card className="hover:shadow-md transition-all hover:-translate-y-1">
              <CardContent className="p-4">
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <div className={`p-2 rounded-lg ${metric.bgColor} ${metric.color}`}>
                        {metric.icon}
                      </div>
                      <span className="text-sm text-muted-foreground">{metric.label}</span>
                    </div>
                    <HelpCircle className="h-4 w-4 text-muted-foreground" />
                  </div>

                  <div>
                    <div className={`text-2xl font-bold ${metric.color}`}>
                      {metric.format(metric.value)}
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Target: {metric.format(metric.target)}
                    </p>
                  </div>

                  <div className="space-y-1">
                    <Progress
                      value={Math.min((metric.value / metric.target) * 100, 100)}
                      className="h-2"
                    />
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </div>

      {/* Performance Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Monthly P&L Chart */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <BarChart3 className="h-5 w-5" />
              <span>Monthly Performance</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={performanceData}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--muted))" />
                <XAxis dataKey="month" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 12 }} tickFormatter={(value) => `$${value / 1000}k`} />
                <ChartTooltip
                  contentStyle={{
                    backgroundColor: 'hsl(var(--background))',
                    border: '1px solid hsl(var(--border))',
                    borderRadius: '8px',
                  }}
                  formatter={(value: any) => formatCurrency(value)}
                />
                <Bar
                  dataKey="profit"
                  fill="hsl(var(--primary))"
                  radius={[8, 8, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Performance Radar Chart */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Activity className="h-5 w-5" />
              <span>Performance Analysis</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="hsl(var(--muted))" />
                <PolarAngleAxis dataKey="metric" tick={{ fontSize: 11 }} />
                <PolarRadiusAxis
                  angle={90}
                  domain={[0, 100]}
                  tick={{ fontSize: 10 }}
                  tickFormatter={(value) => `${value}%`}
                />
                <Radar
                  name="Performance"
                  dataKey="value"
                  stroke="hsl(var(--primary))"
                  fill="hsl(var(--primary))"
                  fillOpacity={0.3}
                  strokeWidth={2}
                />
                <ChartTooltip
                  contentStyle={{
                    backgroundColor: 'hsl(var(--background))',
                    border: '1px solid hsl(var(--border))',
                    borderRadius: '8px',
                  }}
                />
              </RadarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Statistics Summary */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <LineChart className="h-5 w-5" />
            <span>Trading Statistics</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-6">
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground">Total Trades</p>
              <p className="text-2xl font-semibold">342</p>
              <Badge variant="outline" className="text-xs text-green-600">
                +12% this month
              </Badge>
            </div>
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground">Winning Trades</p>
              <p className="text-2xl font-semibold text-green-600">234</p>
              <p className="text-xs text-muted-foreground">68.4% win rate</p>
            </div>
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground">Average Win</p>
              <p className="text-2xl font-semibold text-green-600">$125.50</p>
              <p className="text-xs text-muted-foreground">Per winning trade</p>
            </div>
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground">Average Loss</p>
              <p className="text-2xl font-semibold text-red-600">-$67.80</p>
              <p className="text-xs text-muted-foreground">Per losing trade</p>
            </div>
          </div>

          <Separator className="my-6" />

          {/* Additional Stats */}
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-6">
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground">Best Trade</p>
              <p className="text-xl font-semibold text-green-600">+$1,250.00</p>
            </div>
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground">Worst Trade</p>
              <p className="text-xl font-semibold text-red-600">-$450.00</p>
            </div>
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground">Avg Trade Duration</p>
              <p className="text-xl font-semibold">4h 23m</p>
            </div>
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground">Consecutive Wins</p>
              <p className="text-xl font-semibold text-green-600">12</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default PerformanceMetrics;