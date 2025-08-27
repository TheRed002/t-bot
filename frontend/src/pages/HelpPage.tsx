/**
 * Comprehensive Help & Documentation page for T-Bot Trading System
 * Provides searchable documentation, tutorials, and user guidance
 */

import React, { useState, useMemo, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Alert } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import {
  Search,
  ChevronDown,
  Play,
  TrendingUp,
  Brain,
  Shield,
  Bot,
  Beaker,
  HelpCircle,
  Printer,
  Share,
  Bookmark,
  ChevronRight,
  MessageSquare,
  Settings,
  Code,
  GraduationCap,
  Bug,
  Copy,
} from 'lucide-react';
import { motion } from 'framer-motion';
import { colors } from '@/theme/colors';
import { cn } from '@/lib/utils';

// Types for help content
interface HelpArticle {
  id: string;
  title: string;
  category: string;
  content: string;
  tags: string[];
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  estimatedTime: string;
  lastUpdated: string;
  codeExamples?: CodeExample[];
  relatedArticles?: string[];
}

interface CodeExample {
  language: string;
  title: string;
  code: string;
  description: string;
}

interface HelpCategory {
  id: string;
  title: string;
  icon: React.ReactNode;
  description: string;
  articles: string[];
  color: string;
}

// Help content data
const helpCategories: HelpCategory[] = [
  {
    id: 'getting-started',
    title: 'Getting Started',
    icon: <Play className="w-6 h-6" />,
    description: 'Learn the basics of T-Bot trading system',
    articles: ['quick-start', 'account-setup', 'first-bot'],
    color: colors.accent.cyan,
  },
  {
    id: 'trading-basics',
    title: 'Trading Basics',
    icon: <TrendingUp className="w-6 h-6" />,
    description: 'Fundamental trading concepts and operations',
    articles: ['trading-interface', 'order-types', 'market-analysis'],
    color: colors.financial.profit,
  },
  {
    id: 'strategy-config',
    title: 'Strategy Configuration',
    icon: <Brain className="w-6 h-6" />,
    description: 'Configure and customize trading strategies',
    articles: ['strategy-basics', 'parameter-tuning', 'backtesting'],
    color: colors.accent.purple,
  },
  {
    id: 'risk-management',
    title: 'Risk Management',
    icon: <Shield className="w-6 h-6" />,
    description: 'Protect your capital with proper risk controls',
    articles: ['risk-settings', 'position-sizing', 'stop-losses'],
    color: colors.financial.warning,
  },
  {
    id: 'bot-management',
    title: 'Bot Management',
    icon: <Bot className="w-6 h-6" />,
    description: 'Create, monitor, and manage trading bots',
    articles: ['bot-creation', 'bot-monitoring', 'bot-optimization'],
    color: colors.accent.teal,
  },
  {
    id: 'playground',
    title: 'Playground Tutorial',
    icon: <Beaker className="w-6 h-6" />,
    description: 'Use the playground for strategy testing',
    articles: ['playground-basics', 'simulation-modes', 'optimization'],
    color: colors.primary[500],
  },
  {
    id: 'api-docs',
    title: 'API Documentation',
    icon: <Code className="w-6 h-6" />,
    description: 'Technical API reference and examples',
    articles: ['api-overview', 'authentication', 'endpoints'],
    color: colors.status.info,
  },
  {
    id: 'troubleshooting',
    title: 'Troubleshooting',
    icon: <Bug className="w-6 h-6" />,
    description: 'Common issues and solutions',
    articles: ['common-errors', 'connection-issues', 'performance'],
    color: colors.financial.loss,
  },
];

const helpArticles: HelpArticle[] = [
  {
    id: 'quick-start',
    title: 'Quick Start Guide',
    category: 'getting-started',
    content: `
# Welcome to T-Bot Trading System

This quick start guide will help you get up and running with T-Bot in just a few minutes.

## Step 1: Account Setup
1. Log in to your account using the credentials provided
2. Navigate to the Dashboard to see your portfolio overview
3. Check the system status indicator in the sidebar

## Step 2: Configure Risk Settings
Before starting any trading activity, it's crucial to set up proper risk management:

1. Go to **Risk Dashboard**
2. Set your maximum position size (recommended: 2-5% per trade)
3. Configure stop-loss parameters
4. Set daily loss limits

## Step 3: Create Your First Bot
1. Navigate to **Bot Management**
2. Click "Create New Bot"
3. Choose a strategy (start with Mean Reversion for beginners)
4. Configure parameters using recommended defaults
5. Set position sizing and risk parameters
6. Run backtests to validate performance

## Step 4: Monitor and Optimize
- Use the Dashboard to monitor real-time performance
- Check bot status regularly
- Review trades in the Portfolio section
- Adjust parameters based on market conditions

## Important Safety Notes
- Always start with small position sizes
- Use sandbox mode for testing
- Never risk more than you can afford to lose
- Monitor market conditions and news events
    `,
    tags: ['beginner', 'setup', 'tutorial'],
    difficulty: 'beginner',
    estimatedTime: '10 minutes',
    lastUpdated: '2024-01-15',
    relatedArticles: ['account-setup', 'first-bot', 'risk-settings'],
  },
  {
    id: 'strategy-basics',
    title: 'Understanding Trading Strategies',
    category: 'strategy-config',
    content: `
# Trading Strategy Fundamentals

T-Bot supports multiple algorithmic trading strategies, each designed for different market conditions.

## Strategy Types

### 1. Mean Reversion
**Best for:** Sideways markets, range-bound assets
**Concept:** Assumes prices will return to their average over time
**Parameters:**
- Bollinger Band periods (default: 20)
- Standard deviation multiplier (default: 2.0)
- RSI oversold/overbought levels (30/70)

### 2. Trend Following
**Best for:** Trending markets with clear direction
**Concept:** Follows the momentum of price movements
**Parameters:**
- Moving average periods (fast: 10, slow: 20)
- Momentum threshold (default: 0.02)
- Trend confirmation period (default: 3)

### 3. Market Making
**Best for:** High liquidity pairs, experienced traders
**Concept:** Provides liquidity by placing buy/sell orders around current price
**Parameters:**
- Spread percentage (default: 0.1%)
- Order refresh rate (default: 30 seconds)
- Maximum position size

### 4. Arbitrage
**Best for:** Multi-exchange setups, advanced users
**Concept:** Exploits price differences between exchanges
**Parameters:**
- Minimum profit threshold (default: 0.05%)
- Maximum execution time (default: 5 seconds)
- Exchange priority settings

## Strategy Selection Guidelines

1. **Market Analysis:** Analyze current market conditions
2. **Risk Tolerance:** Choose strategies matching your risk profile
3. **Capital Requirements:** Ensure sufficient capital for chosen strategy
4. **Time Commitment:** Consider monitoring requirements

## Backtesting Your Strategy

Before deploying any strategy:
1. Run at least 3-6 months of historical data
2. Test on different market conditions (bull, bear, sideways)
3. Analyze key metrics: Sharpe ratio, maximum drawdown, win rate
4. Validate performance across multiple timeframes
    `,
    tags: ['strategy', 'intermediate', 'backtesting'],
    difficulty: 'intermediate',
    estimatedTime: '15 minutes',
    lastUpdated: '2024-01-15',
    codeExamples: [
      {
        language: 'yaml',
        title: 'Mean Reversion Strategy Configuration',
        code: `strategy:
  name: "mean_reversion"
  parameters:
    bollinger_periods: 20
    std_dev_multiplier: 2.0
    rsi_period: 14
    rsi_oversold: 30
    rsi_overbought: 70
    position_size: 0.02  # 2% of portfolio
  risk_management:
    stop_loss: 0.05      # 5% stop loss
    take_profit: 0.10    # 10% take profit
    max_positions: 3`,
        description: 'Example configuration for a mean reversion strategy',
      },
    ],
    relatedArticles: ['parameter-tuning', 'backtesting', 'risk-settings'],
  },
  {
    id: 'playground-basics',
    title: 'Playground Tutorial',
    category: 'playground',
    content: `
# Using the T-Bot Playground

The Playground is your safe environment for testing strategies and optimizing parameters without risking real capital.

## Features Overview

### 1. Strategy Testing
- Test any strategy with historical data
- Compare multiple strategies side-by-side
- Analyze performance metrics in real-time

### 2. Parameter Optimization
- Run automated parameter sweeps
- Use genetic algorithms for optimization
- Visualize parameter sensitivity

### 3. Simulation Modes
- **Historical Replay:** Test on past market data
- **Monte Carlo:** Test with simulated scenarios
- **Paper Trading:** Test with live data, simulated orders

## Getting Started

### Step 1: Select Your Configuration
1. Choose trading symbol (e.g., BTC/USD)
2. Select strategy type
3. Set date range for backtesting
4. Configure initial portfolio size

### Step 2: Parameter Setup
1. Use recommended defaults or customize
2. Set risk management parameters
3. Configure position sizing rules
4. Enable/disable features as needed

### Step 3: Run Simulation
1. Click "Start Simulation"
2. Monitor progress in real-time
3. View preliminary results
4. Stop or adjust as needed

### Step 4: Analyze Results
1. Review performance metrics
2. Examine trade history
3. Analyze risk metrics
4. Compare with benchmarks

## Advanced Features

### Batch Optimization
Run multiple strategy variants simultaneously:
- Parameter ranges: Test min/max values
- Grid search: Systematic parameter combinations
- Genetic algorithms: Evolutionary optimization
- Cross-validation: Prevent overfitting

### Scenario Analysis
Test strategy robustness:
- Market stress tests
- Black swan events
- Different market regimes
- Correlation breakdowns

## Best Practices

1. **Start Simple:** Begin with default parameters
2. **Validate Results:** Use out-of-sample testing
3. **Avoid Overfitting:** Don't over-optimize on historical data
4. **Consider Costs:** Include realistic fees and slippage
5. **Test Robustness:** Verify performance across different periods
    `,
    tags: ['playground', 'testing', 'optimization'],
    difficulty: 'intermediate',
    estimatedTime: '20 minutes',
    lastUpdated: '2024-01-15',
    relatedArticles: ['simulation-modes', 'optimization', 'backtesting'],
  },
  {
    id: 'api-overview',
    title: 'API Reference',
    category: 'api-docs',
    content: `
# T-Bot API Documentation

The T-Bot API provides programmatic access to all trading system functions.

## Authentication

All API requests require authentication using JWT tokens.

### Getting Your API Token
1. Navigate to Settings > API Keys
2. Generate a new API key
3. Store securely - it won't be shown again
4. Use in Authorization header: \`Bearer YOUR_TOKEN\`

## Base URL
\`\`\`
https://api.tbot.trading/v1
\`\`\`

## Core Endpoints

### Portfolio Management
- \`GET /portfolio\` - Get portfolio summary
- \`GET /portfolio/positions\` - List active positions
- \`GET /portfolio/history\` - Get portfolio history

### Bot Management
- \`GET /bots\` - List all bots
- \`POST /bots\` - Create new bot
- \`GET /bots/{id}\` - Get bot details
- \`PUT /bots/{id}\` - Update bot configuration
- \`DELETE /bots/{id}\` - Delete bot
- \`POST /bots/{id}/start\` - Start bot
- \`POST /bots/{id}/stop\` - Stop bot

### Strategy Management
- \`GET /strategies\` - List available strategies
- \`GET /strategies/{id}\` - Get strategy details
- \`POST /strategies/{id}/backtest\` - Run backtest

### Market Data
- \`GET /market/symbols\` - List available symbols
- \`GET /market/ticker/{symbol}\` - Get current price
- \`GET /market/history/{symbol}\` - Get historical data

## Rate Limits
- 1000 requests per hour for standard endpoints
- 100 requests per hour for computationally intensive operations
- Real-time data: 10 requests per second

## Error Handling
All errors return standard HTTP status codes with JSON error details.
    `,
    tags: ['api', 'technical', 'reference'],
    difficulty: 'advanced',
    estimatedTime: '25 minutes',
    lastUpdated: '2024-01-15',
    codeExamples: [
      {
        language: 'javascript',
        title: 'Authentication Example',
        code: `// Initialize API client
const apiClient = axios.create({
  baseURL: 'https://api.tbot.trading/v1',
  headers: {
    'Authorization': 'Bearer YOUR_API_TOKEN',
    'Content-Type': 'application/json'
  }
});

// Get portfolio summary
const getPortfolio = async () => {
  try {
    const response = await apiClient.get('/portfolio');
    return response.data;
  } catch (error) {
    console.error('API Error:', error.response.data);
  }
};`,
        description: 'Basic API authentication and request example',
      },
      {
        language: 'python',
        title: 'Create Bot Example',
        code: `import requests

# Create new trading bot
def create_bot(strategy_config):
    url = "https://api.tbot.trading/v1/bots"
    headers = {
        "Authorization": "Bearer YOUR_API_TOKEN",
        "Content-Type": "application/json"
    }
    
    payload = {
        "name": "My Trading Bot",
        "strategy": "mean_reversion",
        "symbol": "BTC/USD",
        "config": strategy_config,
        "risk_management": {
            "max_position_size": 0.02,
            "stop_loss": 0.05,
            "take_profit": 0.10
        }
    }
    
    response = requests.post(url, json=payload, headers=headers)
    return response.json()`,
        description: 'Example of creating a new trading bot via API',
      },
    ],
    relatedArticles: ['authentication', 'endpoints'],
  },
];

const faqData = [
  {
    question: 'How do I get started with T-Bot?',
    answer: 'Start by reading our Quick Start Guide, then set up your risk management settings before creating your first bot. We recommend beginning with paper trading to familiarize yourself with the system.',
    category: 'getting-started',
  },
  {
    question: 'What is the minimum capital required?',
    answer: 'There is no hard minimum, but we recommend at least $1,000 for effective diversification and risk management. Start with smaller amounts in paper trading mode.',
    category: 'getting-started',
  },
  {
    question: 'How do I choose the right strategy?',
    answer: 'Strategy selection depends on market conditions, your risk tolerance, and investment goals. Mean reversion works well in sideways markets, while trend following excels in trending markets. Use our Playground to test different strategies.',
    category: 'strategy-config',
  },
  {
    question: 'What exchanges are supported?',
    answer: 'T-Bot currently supports Binance, Coinbase Pro, and OKX. We are continuously adding new exchanges based on user demand.',
    category: 'trading-basics',
  },
  {
    question: 'How often should I monitor my bots?',
    answer: 'While bots operate autonomously, we recommend checking them at least daily. Monitor performance metrics, market conditions, and any alerts or notifications.',
    category: 'bot-management',
  },
  {
    question: 'What happens if my bot loses money?',
    answer: 'Losses are part of trading. Ensure proper risk management with stop-losses, position sizing, and daily loss limits. Review and adjust your strategy based on performance analysis.',
    category: 'risk-management',
  },
  {
    question: 'Can I run multiple bots simultaneously?',
    answer: 'Yes, you can run multiple bots with different strategies and symbols. Ensure your total exposure stays within your risk tolerance and account limits.',
    category: 'bot-management',
  },
  {
    question: 'How do I optimize my strategy parameters?',
    answer: 'Use the Playground\'s optimization features to test parameter ranges. Avoid overfitting by using out-of-sample validation and testing across different market conditions.',
    category: 'playground',
  },
];

const keyboardShortcuts = [
  { key: 'Ctrl + /', description: 'Open search' },
  { key: 'Ctrl + H', description: 'Go to help page' },
  { key: 'Ctrl + D', description: 'Go to dashboard' },
  { key: 'Ctrl + T', description: 'Go to trading page' },
  { key: 'Ctrl + B', description: 'Go to bot management' },
  { key: 'Ctrl + P', description: 'Go to playground' },
  { key: 'Ctrl + R', description: 'Refresh current page' },
  { key: 'Escape', description: 'Close modals/dialogs' },
];

const HelpPage: React.FC = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [selectedArticle, setSelectedArticle] = useState<HelpArticle | null>(null);
  const [expandedAccordions, setExpandedAccordions] = useState<string[]>([]);
  const [tabValue, setTabValue] = useState(0);
  const [bookmarkedArticles, setBookmarkedArticles] = useState<string[]>([]);

  // Search functionality
  const filteredArticles = useMemo(() => {
    if (!searchQuery) return helpArticles;
    
    const query = searchQuery.toLowerCase();
    return helpArticles.filter(
      article =>
        article.title.toLowerCase().includes(query) ||
        article.content.toLowerCase().includes(query) ||
        article.tags.some(tag => tag.toLowerCase().includes(query))
    );
  }, [searchQuery]);

  const filteredFAQ = useMemo(() => {
    if (!searchQuery) return faqData;
    
    const query = searchQuery.toLowerCase();
    return faqData.filter(
      faq =>
        faq.question.toLowerCase().includes(query) ||
        faq.answer.toLowerCase().includes(query)
    );
  }, [searchQuery]);

  // Handle accordion expansion
  const handleAccordionChange = (panel: string) => (
    event: React.SyntheticEvent,
    isExpanded: boolean
  ) => {
    setExpandedAccordions(prev =>
      isExpanded
        ? [...prev, panel]
        : prev.filter(p => p !== panel)
    );
  };

  // Handle bookmarking
  const toggleBookmark = (articleId: string) => {
    setBookmarkedArticles(prev =>
      prev.includes(articleId)
        ? prev.filter(id => id !== articleId)
        : [...prev, articleId]
    );
  };

  // Copy code to clipboard
  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    // Could add a toast notification here
  };

  // Breadcrumb navigation
  const getBreadcrumbs = () => {
    const breadcrumbs = [
      { label: 'Help', href: '/help' },
    ];

    if (selectedCategory) {
      const category = helpCategories.find(cat => cat.id === selectedCategory);
      if (category) {
        breadcrumbs.push({ label: category.title, href: `/help/${category.id}` });
      }
    }

    if (selectedArticle) {
      breadcrumbs.push({ label: selectedArticle.title, href: `/help/article/${selectedArticle.id}` });
    }

    return breadcrumbs;
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <div className="h-screen flex flex-col">
        {/* Header */}
        <div className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
          <div className="flex h-16 items-center justify-between px-6">
            <h1 className="text-2xl font-semibold">Help & Documentation</h1>
            
            <div className="flex items-center gap-4">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  placeholder="Search documentation..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-80 pl-10"
                />
              </div>
              
              <Dialog>
                <DialogTrigger asChild>
                  <Button variant="ghost" size="icon" className="h-8 w-8">
                    <Settings className="h-4 w-4" />
                    <span className="sr-only">Keyboard shortcuts</span>
                  </Button>
                </DialogTrigger>
                <DialogContent>
                  <DialogHeader>
                    <DialogTitle>Keyboard Shortcuts</DialogTitle>
                  </DialogHeader>
                  <div className="space-y-2">
                    {keyboardShortcuts.map((shortcut, index) => (
                      <div key={index} className="flex justify-between items-center">
                        <span className="text-sm">{shortcut.description}</span>
                        <Badge variant="outline" className="font-mono text-xs">
                          {shortcut.key}
                        </Badge>
                      </div>
                    ))}
                  </div>
                </DialogContent>
              </Dialog>
              
              <Button
                variant="ghost"
                size="icon"
                onClick={() => window.print()}
                className="h-8 w-8"
              >
                <Printer className="h-4 w-4" />
                <span className="sr-only">Print page</span>
              </Button>
            </div>
          </div>
        </div>

        <div className="flex flex-1 overflow-hidden">
          {/* Sidebar */}
          <div className="w-80 border-r bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 overflow-auto">
            {/* Breadcrumbs */}
            {(selectedCategory || selectedArticle) && (
              <div className="p-4 border-b">
                <nav className="flex items-center space-x-1 text-sm text-muted-foreground">
                  {getBreadcrumbs().map((crumb, index) => (
                    <React.Fragment key={index}>
                      {index > 0 && <ChevronRight className="h-4 w-4" />}
                      <button
                        onClick={(e) => {
                          e.preventDefault();
                          if (index === 0) {
                            setSelectedCategory(null);
                            setSelectedArticle(null);
                          } else if (index === 1) {
                            setSelectedArticle(null);
                          }
                        }}
                        className="hover:text-foreground transition-colors"
                      >
                        {crumb.label}
                      </button>
                    </React.Fragment>
                  ))}
                </nav>
              </div>
            )}

            {/* Navigation */}
            <div className="p-2 space-y-1">
              {!selectedCategory ? (
                // Category list
                helpCategories.map((category) => (
                  <button
                    key={category.id}
                    onClick={() => setSelectedCategory(category.id)}
                    className="w-full flex items-start gap-3 p-3 rounded-lg hover:bg-accent transition-colors text-left"
                  >
                    <div style={{ color: category.color }} className="mt-0.5">
                      {category.icon}
                    </div>
                    <div className="flex-1">
                      <div className="font-medium text-sm">{category.title}</div>
                      <div className="text-xs text-muted-foreground mt-0.5">
                        {category.description}
                      </div>
                    </div>
                  </button>
                ))
              ) : (
                // Article list for selected category
                <>
                  <div className="p-2">
                    <Button
                      variant="ghost"
                      onClick={() => setSelectedCategory(null)}
                      className="mb-2"
                    >
                      <ChevronRight className="h-4 w-4 mr-2 rotate-180" />
                      Back to Categories
                    </Button>
                  </div>
                  <div className="space-y-1">
                    {helpArticles
                      .filter(article => article.category === selectedCategory)
                      .map((article) => (
                        <div
                          key={article.id}
                          className={cn(
                            "mx-2 rounded-lg transition-colors cursor-pointer",
                            selectedArticle?.id === article.id
                              ? "bg-accent"
                              : "hover:bg-accent/50"
                          )}
                          onClick={() => setSelectedArticle(article)}
                        >
                          <div className="flex items-start justify-between p-3">
                            <div className="flex-1">
                              <div className="text-sm font-medium">{article.title}</div>
                              <div className="flex items-center gap-2 mt-1">
                                <Badge
                                  variant={article.difficulty === 'beginner' ? 'default' : 
                                          article.difficulty === 'intermediate' ? 'secondary' : 'destructive'}
                                  className="text-xs"
                                >
                                  {article.difficulty}
                                </Badge>
                                <span className="text-xs text-muted-foreground">
                                  {article.estimatedTime}
                                </span>
                              </div>
                            </div>
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-6 w-6"
                              onClick={(e) => {
                                e.stopPropagation();
                                toggleBookmark(article.id);
                              }}
                            >
                              <Bookmark
                                className={cn(
                                  "h-3 w-3",
                                  bookmarkedArticles.includes(article.id)
                                    ? "fill-current text-primary"
                                    : "text-muted-foreground"
                                )}
                              />
                            </Button>
                          </div>
                        </div>
                      ))}
                  </div>
                </>
              )}
            </div>
          </div>

          {/* Main Content */}
          <div className="flex-1 overflow-auto">
            {selectedArticle ? (
              // Article content
              <div className="p-6 max-w-4xl mx-auto">
                <div className="flex justify-between items-start mb-6">
                  <div>
                    <h1 className="text-3xl font-semibold mb-4">
                      {selectedArticle.title}
                    </h1>
                    
                    <div className="flex items-center gap-4 mb-4">
                      <Badge
                        variant={selectedArticle.difficulty === 'beginner' ? 'default' : 
                                selectedArticle.difficulty === 'intermediate' ? 'secondary' : 'destructive'}
                      >
                        {selectedArticle.difficulty}
                      </Badge>
                      <span className="text-sm text-muted-foreground">
                        Estimated time: {selectedArticle.estimatedTime}
                      </span>
                      <span className="text-sm text-muted-foreground">
                        Updated: {selectedArticle.lastUpdated}
                      </span>
                    </div>

                    <div className="flex gap-2 mb-6">
                      {selectedArticle.tags.map((tag) => (
                        <Badge key={tag} variant="outline">{tag}</Badge>
                      ))}
                    </div>
                  </div>

                  <div className="flex gap-2">
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => toggleBookmark(selectedArticle.id)}
                    >
                      <Bookmark
                        className={cn(
                          "h-4 w-4",
                          bookmarkedArticles.includes(selectedArticle.id)
                            ? "fill-current text-primary"
                            : "text-muted-foreground"
                        )}
                      />
                    </Button>
                    <Button variant="ghost" size="icon">
                      <Share className="h-4 w-4" />
                    </Button>
                  </div>
                </div>

                {/* Article content */}
                <Card className="mb-6">
                  <CardContent className="p-6 prose max-w-none">
                    <div className="space-y-4">
                      {selectedArticle.content.split('\n').map((line, index) => {
                        if (line.startsWith('# ')) {
                          return <h1 key={index} className="text-2xl font-semibold mt-6 mb-4 first:mt-0">{line.slice(2)}</h1>;
                        }
                        if (line.startsWith('## ')) {
                          return <h2 key={index} className="text-xl font-semibold mt-6 mb-3">{line.slice(3)}</h2>;
                        }
                        if (line.startsWith('### ')) {
                          return <h3 key={index} className="text-lg font-semibold mt-4 mb-2">{line.slice(4)}</h3>;
                        }
                        if (line.trim() === '') {
                          return <div key={index} className="h-2" />;
                        }
                        return <p key={index} className="leading-7 text-sm">{line}</p>;
                      })}
                    </div>
                  </CardContent>
                </Card>

                {/* Code Examples */}
                {selectedArticle.codeExamples && selectedArticle.codeExamples.length > 0 && (
                  <div className="mb-6">
                    <h2 className="text-xl font-semibold mb-4">Code Examples</h2>
                    <div className="space-y-4">
                      {selectedArticle.codeExamples.map((example, index) => (
                        <Card key={index}>
                          <CardContent className="p-4">
                            <div className="flex justify-between items-start mb-4">
                              <div>
                                <h3 className="font-semibold text-lg mb-1">{example.title}</h3>
                                <p className="text-sm text-muted-foreground">{example.description}</p>
                              </div>
                              <div className="flex gap-2">
                                <Badge variant="secondary">{example.language}</Badge>
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  className="h-6 w-6"
                                  onClick={() => copyToClipboard(example.code)}
                                >
                                  <Copy className="h-3 w-3" />
                                </Button>
                              </div>
                            </div>
                            <pre className="bg-muted p-4 rounded-lg overflow-auto text-sm">
                              <code>{example.code}</code>
                            </pre>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  </div>
                )}

                {/* Related Articles */}
                {selectedArticle.relatedArticles && selectedArticle.relatedArticles.length > 0 && (
                  <div>
                    <h2 className="text-xl font-semibold mb-4">Related Articles</h2>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {selectedArticle.relatedArticles.map((relatedId) => {
                        const relatedArticle = helpArticles.find(a => a.id === relatedId);
                        if (!relatedArticle) return null;
                        
                        return (
                          <Card
                            key={relatedId}
                            className="cursor-pointer hover:bg-accent transition-colors"
                            onClick={() => setSelectedArticle(relatedArticle)}
                          >
                            <CardContent className="p-4">
                              <h3 className="font-semibold mb-2">{relatedArticle.title}</h3>
                              <div className="flex items-center gap-2">
                                <Badge
                                  variant={relatedArticle.difficulty === 'beginner' ? 'default' : 
                                          relatedArticle.difficulty === 'intermediate' ? 'secondary' : 'destructive'}
                                >
                                  {relatedArticle.difficulty}
                                </Badge>
                                <span className="text-xs text-muted-foreground">
                                  {relatedArticle.estimatedTime}
                                </span>
                              </div>
                            </CardContent>
                          </Card>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              // Main help page content
              <div className="p-6">
                {searchQuery ? (
                  // Search results
                  <div>
                    <h2 className="text-2xl font-semibold mb-6">
                      Search Results for "{searchQuery}"
                    </h2>
                    
                    {filteredArticles.length > 0 && (
                      <div className="mb-8">
                        <h3 className="text-lg font-semibold mb-4">
                          Articles ({filteredArticles.length})
                        </h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                          {filteredArticles.map((article) => (
                            <Card
                              key={article.id}
                              className="cursor-pointer h-full hover:bg-accent transition-colors"
                              onClick={() => {
                                setSelectedCategory(article.category);
                                setSelectedArticle(article);
                              }}
                            >
                              <CardContent className="p-4">
                                <h4 className="font-semibold mb-2">{article.title}</h4>
                                <p className="text-sm text-muted-foreground mb-4 line-clamp-3">
                                  {article.content.slice(0, 100)}...
                                </p>
                                <div className="flex justify-between items-center">
                                  <Badge
                                    variant={article.difficulty === 'beginner' ? 'default' : 
                                            article.difficulty === 'intermediate' ? 'secondary' : 'destructive'}
                                  >
                                    {article.difficulty}
                                  </Badge>
                                  <span className="text-xs text-muted-foreground">
                                    {article.estimatedTime}
                                  </span>
                                </div>
                              </CardContent>
                            </Card>
                          ))}
                        </div>
                      </div>
                    )}

                    {filteredFAQ.length > 0 && (
                      <div>
                        <h3 className="text-lg font-semibold mb-4">
                          FAQ Results ({filteredFAQ.length})
                        </h3>
                        <div className="space-y-2">
                          {filteredFAQ.map((faq, index) => (
                            <Card key={index}>
                              <CardContent className="p-4">
                                <div className="flex items-start gap-3">
                                  <HelpCircle className="h-5 w-5 mt-0.5 text-muted-foreground flex-shrink-0" />
                                  <div>
                                    <h4 className="font-medium mb-2">{faq.question}</h4>
                                    <p className="text-sm text-muted-foreground">{faq.answer}</p>
                                  </div>
                                </div>
                              </CardContent>
                            </Card>
                          ))}
                        </div>
                      </div>
                    )}

                    {filteredArticles.length === 0 && filteredFAQ.length === 0 && (
                      <Alert>
                        <HelpCircle className="h-4 w-4" />
                        <div>
                          No results found for "{searchQuery}". Try different keywords or browse categories below.
                        </div>
                      </Alert>
                    )}
                  </div>
                ) : (
                  // Default help page content
                  <div>
                    {/* Welcome Section */}
                    <div className="text-center mb-12">
                      <h1 className="text-4xl font-semibold mb-4">
                        Welcome to T-Bot Help Center
                      </h1>
                      <p className="text-lg text-muted-foreground mb-8 max-w-2xl mx-auto">
                        Your comprehensive guide to mastering algorithmic trading with T-Bot.
                        Find tutorials, API documentation, and troubleshooting guides.
                      </p>
                      
                      <div className="flex justify-center gap-4">
                        <Button
                          onClick={() => {
                            setSelectedCategory('getting-started');
                            setSelectedArticle(helpArticles.find(a => a.id === 'quick-start') || null);
                          }}
                          className="gap-2"
                        >
                          <Play className="h-4 w-4" />
                          Quick Start Guide
                        </Button>
                        <Button
                          variant="outline"
                          onClick={() => window.open('/playground', '_blank')}
                          className="gap-2"
                        >
                          <Play className="h-4 w-4" />
                          Try Playground
                        </Button>
                      </div>
                    </div>

                    {/* Categories Grid */}
                    <h2 className="text-2xl font-semibold mb-6">
                      Documentation Categories
                    </h2>
                    
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 mb-12">
                      {helpCategories.map((category) => (
                        <Card
                          key={category.id}
                          className="cursor-pointer h-full hover:shadow-lg transition-all duration-200 hover:-translate-y-1"
                          onClick={() => setSelectedCategory(category.id)}
                        >
                          <CardContent className="text-center p-6">
                            <div
                              className="inline-flex p-4 rounded-full mb-4"
                              style={{
                                backgroundColor: `${category.color}20`,
                                color: category.color,
                              }}
                            >
                              {category.icon}
                            </div>
                            <h3 className="text-lg font-semibold mb-2">{category.title}</h3>
                            <p className="text-sm text-muted-foreground mb-4">{category.description}</p>
                            <Badge variant="outline">
                              {category.articles.length} articles
                            </Badge>
                          </CardContent>
                        </Card>
                      ))}
                    </div>

                    {/* Quick Access Tabs */}
                    <Card className="mb-8">
                      <Tabs value={tabValue === 0 ? 'faq' : tabValue === 1 ? 'tutorials' : 'support'} onValueChange={(v) => setTabValue(v === 'faq' ? 0 : v === 'tutorials' ? 1 : 2)}>
                        <TabsList className="grid w-full grid-cols-3">
                          <TabsTrigger value="faq" className="flex items-center gap-2">
                            <MessageSquare className="h-4 w-4" />
                            FAQ
                          </TabsTrigger>
                          <TabsTrigger value="tutorials" className="flex items-center gap-2">
                            <GraduationCap className="h-4 w-4" />
                            Video Tutorials
                          </TabsTrigger>
                          <TabsTrigger value="support" className="flex items-center gap-2">
                            <HelpCircle className="h-4 w-4" />
                            Contact Support
                          </TabsTrigger>
                        </TabsList>

                        <TabsContent value="faq" className="p-6">
                          <h3 className="text-lg font-semibold mb-4">
                            Frequently Asked Questions
                          </h3>
                          <div className="space-y-3">
                            {faqData.slice(0, 6).map((faq, index) => (
                              <Card key={index}>
                                <CardContent className="p-4">
                                  <div className="flex items-start gap-3">
                                    <HelpCircle className="h-5 w-5 mt-0.5 text-muted-foreground flex-shrink-0" />
                                    <div>
                                      <h4 className="font-medium mb-2">{faq.question}</h4>
                                      <p className="text-sm text-muted-foreground">{faq.answer}</p>
                                    </div>
                                  </div>
                                </CardContent>
                              </Card>
                            ))}
                          </div>
                          <div className="text-center mt-6">
                            <Button variant="outline">View All FAQ</Button>
                          </div>
                        </TabsContent>

                        <TabsContent value="tutorials" className="p-6">
                          <h3 className="text-lg font-semibold mb-4">
                            Video Tutorials
                          </h3>
                          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                            {[
                              { title: 'Getting Started with T-Bot', duration: '5:30', level: 'Beginner' },
                              { title: 'Setting Up Your First Bot', duration: '8:45', level: 'Beginner' },
                              { title: 'Advanced Strategy Configuration', duration: '12:20', level: 'Advanced' },
                              { title: 'Risk Management Best Practices', duration: '7:15', level: 'Intermediate' },
                            ].map((video, index) => (
                              <Card key={index} className="overflow-hidden">
                                <div className="h-32 bg-muted flex items-center justify-center cursor-pointer hover:bg-muted/80 transition-colors">
                                  <Play className="h-10 w-10 text-primary" />
                                </div>
                                <CardContent className="p-4">
                                  <h4 className="font-semibold text-sm mb-2">{video.title}</h4>
                                  <div className="flex justify-between items-center">
                                    <span className="text-xs text-muted-foreground">{video.duration}</span>
                                    <Badge
                                      variant={video.level === 'Beginner' ? 'default' : 
                                              video.level === 'Intermediate' ? 'secondary' : 'destructive'}
                                    >
                                      {video.level}
                                    </Badge>
                                  </div>
                                </CardContent>
                              </Card>
                            ))}
                          </div>
                        </TabsContent>

                        <TabsContent value="support" className="p-6">
                          <h3 className="text-lg font-semibold mb-4">
                            Get Help & Support
                          </h3>
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                            <Card>
                              <CardContent className="p-6">
                                <h4 className="text-lg font-semibold mb-2 flex items-center gap-2">
                                  📧 Email Support
                                </h4>
                                <p className="text-sm text-muted-foreground mb-4">
                                  Get help from our technical support team. We typically respond within 24 hours.
                                </p>
                                <Button asChild>
                                  <a href="mailto:support@tbot.trading">Contact Support</a>
                                </Button>
                              </CardContent>
                            </Card>
                            <Card>
                              <CardContent className="p-6">
                                <h4 className="text-lg font-semibold mb-2 flex items-center gap-2">
                                  💬 Community Forum
                                </h4>
                                <p className="text-sm text-muted-foreground mb-4">
                                  Join our community to ask questions and share knowledge with other traders.
                                </p>
                                <Button variant="outline" asChild>
                                  <a href="https://community.tbot.trading" target="_blank">Visit Forum</a>
                                </Button>
                              </CardContent>
                            </Card>
                          </div>
                          <Alert>
                            <HelpCircle className="h-4 w-4" />
                            <div>
                              For urgent technical issues affecting live trading, please contact our emergency support line at +1-555-TBOT-911.
                            </div>
                          </Alert>
                        </TabsContent>
                      </Tabs>
                    </Card>

                    {/* System Requirements */}
                    <Card>
                      <CardContent className="p-6">
                        <h3 className="text-lg font-semibold mb-6">
                          System Requirements
                        </h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                          <div>
                            <h4 className="font-semibold mb-4">Minimum Requirements</h4>
                            <ul className="space-y-2 text-sm">
                              <li>• Modern web browser (Chrome 90+, Firefox 88+, Safari 14+)</li>
                              <li>• Stable internet connection (minimum 1 Mbps)</li>
                              <li>• JavaScript enabled</li>
                              <li>• Minimum screen resolution: 1024x768</li>
                            </ul>
                          </div>
                          <div>
                            <h4 className="font-semibold mb-4">Recommended</h4>
                            <ul className="space-y-2 text-sm">
                              <li>• Latest Chrome or Firefox browser</li>
                              <li>• High-speed internet (10+ Mbps)</li>
                              <li>• Full HD display (1920x1080)</li>
                              <li>• Hardware acceleration enabled</li>
                            </ul>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default HelpPage;