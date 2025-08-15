/**
 * Comprehensive Help & Documentation page for T-Bot Trading System
 * Provides searchable documentation, tutorials, and user guidance
 */

import React, { useState, useMemo, useEffect } from 'react';
import {
  Box,
  Grid,
  Typography,
  Paper,
  TextField,
  InputAdornment,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  Button,
  IconButton,
  Tooltip,
  Card,
  CardContent,
  CardActions,
  Breadcrumbs,
  Link,
  Divider,
  Alert,
  Switch,
  FormControlLabel,
  Tab,
  Tabs,
  AppBar,
  Toolbar,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import {
  Search as SearchIcon,
  ExpandMore as ExpandMoreIcon,
  GetStarted as GetStartedIcon,
  TrendingUp as TradingIcon,
  Psychology as StrategyIcon,
  Security as RiskIcon,
  SmartToy as BotIcon,
  Science as PlaygroundIcon,
  Code as CodeIcon,
  ContactSupport as ContactIcon,
  Print as PrintIcon,
  Share as ShareIcon,
  Bookmark as BookmarkIcon,
  KeyboardArrowRight as ArrowIcon,
  PlayArrow as PlayIcon,
  VideoLibrary as VideoIcon,
  Article as ArticleIcon,
  QuestionAnswer as QAIcon,
  Settings as SettingsIcon,
  Api as ApiIcon,
  School as TutorialIcon,
  BugReport as TroubleshootIcon,
  MenuBook as GlossaryIcon,
  ContentCopy as CopyIcon,
  Home as HomeIcon,
} from '@mui/icons-material';
import { motion } from 'framer-motion';
import { useTheme } from '@mui/material/styles';
import { colors } from '@/theme/colors';

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
    icon: <GetStartedIcon />,
    description: 'Learn the basics of T-Bot trading system',
    articles: ['quick-start', 'account-setup', 'first-bot'],
    color: colors.accent.cyan,
  },
  {
    id: 'trading-basics',
    title: 'Trading Basics',
    icon: <TradingIcon />,
    description: 'Fundamental trading concepts and operations',
    articles: ['trading-interface', 'order-types', 'market-analysis'],
    color: colors.financial.profit,
  },
  {
    id: 'strategy-config',
    title: 'Strategy Configuration',
    icon: <StrategyIcon />,
    description: 'Configure and customize trading strategies',
    articles: ['strategy-basics', 'parameter-tuning', 'backtesting'],
    color: colors.accent.purple,
  },
  {
    id: 'risk-management',
    title: 'Risk Management',
    icon: <RiskIcon />,
    description: 'Protect your capital with proper risk controls',
    articles: ['risk-settings', 'position-sizing', 'stop-losses'],
    color: colors.financial.warning,
  },
  {
    id: 'bot-management',
    title: 'Bot Management',
    icon: <BotIcon />,
    description: 'Create, monitor, and manage trading bots',
    articles: ['bot-creation', 'bot-monitoring', 'bot-optimization'],
    color: colors.accent.teal,
  },
  {
    id: 'playground',
    title: 'Playground Tutorial',
    icon: <PlaygroundIcon />,
    description: 'Use the playground for strategy testing',
    articles: ['playground-basics', 'simulation-modes', 'optimization'],
    color: colors.primary[500],
  },
  {
    id: 'api-docs',
    title: 'API Documentation',
    icon: <ApiIcon />,
    description: 'Technical API reference and examples',
    articles: ['api-overview', 'authentication', 'endpoints'],
    color: colors.status.info,
  },
  {
    id: 'troubleshooting',
    title: 'Troubleshooting',
    icon: <TroubleshootIcon />,
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
  const theme = useTheme();
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [selectedArticle, setSelectedArticle] = useState<HelpArticle | null>(null);
  const [expandedAccordions, setExpandedAccordions] = useState<string[]>([]);
  const [tabValue, setTabValue] = useState(0);
  const [showShortcuts, setShowShortcuts] = useState(false);
  const [darkMode, setDarkMode] = useState(true);
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
      <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
        {/* Header */}
        <AppBar position="static" color="transparent" elevation={0}>
          <Toolbar>
            <Typography variant="h5" component="h1" sx={{ flexGrow: 1, fontWeight: 600 }}>
              Help & Documentation
            </Typography>
            
            <Box display="flex" alignItems="center" gap={2}>
              <TextField
                size="small"
                placeholder="Search documentation..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <SearchIcon />
                    </InputAdornment>
                  ),
                }}
                sx={{ width: 300 }}
              />
              
              <Tooltip title="Keyboard shortcuts">
                <IconButton onClick={() => setShowShortcuts(true)}>
                  <SettingsIcon />
                </IconButton>
              </Tooltip>
              
              <Tooltip title="Print page">
                <IconButton onClick={() => window.print()}>
                  <PrintIcon />
                </IconButton>
              </Tooltip>
            </Box>
          </Toolbar>
        </AppBar>

        <Box sx={{ display: 'flex', flexGrow: 1, overflow: 'hidden' }}>
          {/* Sidebar */}
          <Paper
            sx={{
              width: 320,
              borderRadius: 0,
              borderRight: 1,
              borderColor: 'divider',
              overflow: 'auto',
            }}
          >
            {/* Breadcrumbs */}
            {(selectedCategory || selectedArticle) && (
              <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
                <Breadcrumbs separator={<ArrowIcon fontSize="small" />}>
                  {getBreadcrumbs().map((crumb, index) => (
                    <Link
                      key={index}
                      color="inherit"
                      href={crumb.href}
                      onClick={(e) => {
                        e.preventDefault();
                        if (index === 0) {
                          setSelectedCategory(null);
                          setSelectedArticle(null);
                        } else if (index === 1) {
                          setSelectedArticle(null);
                        }
                      }}
                      sx={{ cursor: 'pointer' }}
                    >
                      {crumb.label}
                    </Link>
                  ))}
                </Breadcrumbs>
              </Box>
            )}

            {/* Navigation */}
            <List sx={{ py: 1 }}>
              {!selectedCategory ? (
                // Category list
                helpCategories.map((category) => (
                  <ListItem key={category.id} disablePadding sx={{ mb: 0.5 }}>
                    <ListItemButton
                      onClick={() => setSelectedCategory(category.id)}
                      sx={{
                        mx: 1,
                        borderRadius: 2,
                        '&:hover': {
                          backgroundColor: 'action.hover',
                        },
                      }}
                    >
                      <ListItemIcon sx={{ color: category.color }}>
                        {category.icon}
                      </ListItemIcon>
                      <ListItemText
                        primary={category.title}
                        secondary={category.description}
                        primaryTypographyProps={{ fontWeight: 500 }}
                        secondaryTypographyProps={{ fontSize: '0.75rem' }}
                      />
                    </ListItemButton>
                  </ListItem>
                ))
              ) : (
                // Article list for selected category
                <>
                  <ListItem>
                    <Button
                      startIcon={<ArrowIcon sx={{ transform: 'rotate(180deg)' }} />}
                      onClick={() => setSelectedCategory(null)}
                      sx={{ mb: 1 }}
                    >
                      Back to Categories
                    </Button>
                  </ListItem>
                  {helpArticles
                    .filter(article => article.category === selectedCategory)
                    .map((article) => (
                      <ListItem key={article.id} disablePadding sx={{ mb: 0.5 }}>
                        <ListItemButton
                          selected={selectedArticle?.id === article.id}
                          onClick={() => setSelectedArticle(article)}
                          sx={{
                            mx: 1,
                            borderRadius: 2,
                          }}
                        >
                          <ListItemText
                            primary={article.title}
                            secondary={
                              <Box display="flex" alignItems="center" gap={1} mt={0.5}>
                                <Chip
                                  label={article.difficulty}
                                  size="small"
                                  color={
                                    article.difficulty === 'beginner' ? 'success' :
                                    article.difficulty === 'intermediate' ? 'warning' : 'error'
                                  }
                                />
                                <Typography variant="caption" color="text.secondary">
                                  {article.estimatedTime}
                                </Typography>
                              </Box>
                            }
                            primaryTypographyProps={{ fontSize: '0.9rem' }}
                          />
                          <IconButton
                            size="small"
                            onClick={(e) => {
                              e.stopPropagation();
                              toggleBookmark(article.id);
                            }}
                          >
                            <BookmarkIcon
                              fontSize="small"
                              color={bookmarkedArticles.includes(article.id) ? 'primary' : 'disabled'}
                            />
                          </IconButton>
                        </ListItemButton>
                      </ListItem>
                    ))}
                </>
              )}
            </List>
          </Paper>

          {/* Main Content */}
          <Box sx={{ flexGrow: 1, overflow: 'auto' }}>
            {selectedArticle ? (
              // Article content
              <Box sx={{ p: 4, maxWidth: 800, mx: 'auto' }}>
                <Box display="flex" justifyContent="between" alignItems="flex-start" mb={3}>
                  <Box>
                    <Typography variant="h4" component="h1" gutterBottom sx={{ fontWeight: 600 }}>
                      {selectedArticle.title}
                    </Typography>
                    
                    <Box display="flex" alignItems="center" gap={2} mb={2}>
                      <Chip
                        label={selectedArticle.difficulty}
                        size="small"
                        color={
                          selectedArticle.difficulty === 'beginner' ? 'success' :
                          selectedArticle.difficulty === 'intermediate' ? 'warning' : 'error'
                        }
                      />
                      <Typography variant="body2" color="text.secondary">
                        Estimated time: {selectedArticle.estimatedTime}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Updated: {selectedArticle.lastUpdated}
                      </Typography>
                    </Box>

                    <Box display="flex" gap={1} mb={3}>
                      {selectedArticle.tags.map((tag) => (
                        <Chip key={tag} label={tag} size="small" variant="outlined" />
                      ))}
                    </Box>
                  </Box>

                  <Box display="flex" gap={1}>
                    <Tooltip title="Bookmark article">
                      <IconButton onClick={() => toggleBookmark(selectedArticle.id)}>
                        <BookmarkIcon
                          color={bookmarkedArticles.includes(selectedArticle.id) ? 'primary' : 'disabled'}
                        />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Share article">
                      <IconButton>
                        <ShareIcon />
                      </IconButton>
                    </Tooltip>
                  </Box>
                </Box>

                {/* Article content */}
                <Paper sx={{ p: 3, mb: 3 }}>
                  <Typography
                    component="div"
                    sx={{
                      '& h1': { fontSize: '1.75rem', fontWeight: 600, mb: 2, mt: 3 },
                      '& h2': { fontSize: '1.5rem', fontWeight: 600, mb: 2, mt: 3 },
                      '& h3': { fontSize: '1.25rem', fontWeight: 600, mb: 1.5, mt: 2 },
                      '& p': { mb: 1.5, lineHeight: 1.7 },
                      '& ul': { pl: 3, mb: 2 },
                      '& li': { mb: 0.5 },
                      '& code': {
                        backgroundColor: 'action.hover',
                        px: 0.5,
                        borderRadius: 0.5,
                        fontFamily: 'monospace',
                      },
                    }}
                  >
                    {selectedArticle.content.split('\n').map((line, index) => {
                      if (line.startsWith('# ')) {
                        return <Typography key={index} variant="h1" component="h1">{line.slice(2)}</Typography>;
                      }
                      if (line.startsWith('## ')) {
                        return <Typography key={index} variant="h2" component="h2">{line.slice(3)}</Typography>;
                      }
                      if (line.startsWith('### ')) {
                        return <Typography key={index} variant="h3" component="h3">{line.slice(4)}</Typography>;
                      }
                      if (line.trim() === '') {
                        return <br key={index} />;
                      }
                      return <Typography key={index} component="p">{line}</Typography>;
                    })}
                  </Typography>
                </Paper>

                {/* Code Examples */}
                {selectedArticle.codeExamples && selectedArticle.codeExamples.length > 0 && (
                  <Box mb={3}>
                    <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                      Code Examples
                    </Typography>
                    {selectedArticle.codeExamples.map((example, index) => (
                      <Card key={index} sx={{ mb: 2 }}>
                        <CardContent>
                          <Box display="flex" justifyContent="between" alignItems="center" mb={2}>
                            <Box>
                              <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                                {example.title}
                              </Typography>
                              <Typography variant="body2" color="text.secondary">
                                {example.description}
                              </Typography>
                            </Box>
                            <Box display="flex" gap={1}>
                              <Chip label={example.language} size="small" />
                              <Tooltip title="Copy code">
                                <IconButton
                                  size="small"
                                  onClick={() => copyToClipboard(example.code)}
                                >
                                  <CopyIcon fontSize="small" />
                                </IconButton>
                              </Tooltip>
                            </Box>
                          </Box>
                          <Box
                            component="pre"
                            sx={{
                              backgroundColor: 'action.hover',
                              p: 2,
                              borderRadius: 1,
                              overflow: 'auto',
                              fontFamily: 'monospace',
                              fontSize: '0.875rem',
                              lineHeight: 1.5,
                            }}
                          >
                            <code>{example.code}</code>
                          </Box>
                        </CardContent>
                      </Card>
                    ))}
                  </Box>
                )}

                {/* Related Articles */}
                {selectedArticle.relatedArticles && selectedArticle.relatedArticles.length > 0 && (
                  <Box>
                    <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                      Related Articles
                    </Typography>
                    <Grid container spacing={2}>
                      {selectedArticle.relatedArticles.map((relatedId) => {
                        const relatedArticle = helpArticles.find(a => a.id === relatedId);
                        if (!relatedArticle) return null;
                        
                        return (
                          <Grid item xs={12} md={6} key={relatedId}>
                            <Card sx={{ cursor: 'pointer' }} onClick={() => setSelectedArticle(relatedArticle)}>
                              <CardContent>
                                <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                                  {relatedArticle.title}
                                </Typography>
                                <Box display="flex" alignItems="center" gap={1} mt={1}>
                                  <Chip
                                    label={relatedArticle.difficulty}
                                    size="small"
                                    color={
                                      relatedArticle.difficulty === 'beginner' ? 'success' :
                                      relatedArticle.difficulty === 'intermediate' ? 'warning' : 'error'
                                    }
                                  />
                                  <Typography variant="caption" color="text.secondary">
                                    {relatedArticle.estimatedTime}
                                  </Typography>
                                </Box>
                              </CardContent>
                            </Card>
                          </Grid>
                        );
                      })}
                    </Grid>
                  </Box>
                )}
              </Box>
            ) : (
              // Main help page content
              <Box sx={{ p: 4 }}>
                {searchQuery ? (
                  // Search results
                  <Box>
                    <Typography variant="h5" gutterBottom sx={{ fontWeight: 600 }}>
                      Search Results for "{searchQuery}"
                    </Typography>
                    
                    {filteredArticles.length > 0 && (
                      <Box mb={4}>
                        <Typography variant="h6" gutterBottom>
                          Articles ({filteredArticles.length})
                        </Typography>
                        <Grid container spacing={2}>
                          {filteredArticles.map((article) => (
                            <Grid item xs={12} md={6} lg={4} key={article.id}>
                              <Card
                                sx={{ cursor: 'pointer', height: '100%' }}
                                onClick={() => {
                                  setSelectedCategory(article.category);
                                  setSelectedArticle(article);
                                }}
                              >
                                <CardContent>
                                  <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                                    {article.title}
                                  </Typography>
                                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                                    {article.content.slice(0, 100)}...
                                  </Typography>
                                  <Box display="flex" justifyContent="between" alignItems="center">
                                    <Chip
                                      label={article.difficulty}
                                      size="small"
                                      color={
                                        article.difficulty === 'beginner' ? 'success' :
                                        article.difficulty === 'intermediate' ? 'warning' : 'error'
                                      }
                                    />
                                    <Typography variant="caption" color="text.secondary">
                                      {article.estimatedTime}
                                    </Typography>
                                  </Box>
                                </CardContent>
                              </Card>
                            </Grid>
                          ))}
                        </Grid>
                      </Box>
                    )}

                    {filteredFAQ.length > 0 && (
                      <Box>
                        <Typography variant="h6" gutterBottom>
                          FAQ Results ({filteredFAQ.length})
                        </Typography>
                        {filteredFAQ.map((faq, index) => (
                          <Accordion key={index}>
                            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                              <Typography sx={{ fontWeight: 500 }}>{faq.question}</Typography>
                            </AccordionSummary>
                            <AccordionDetails>
                              <Typography>{faq.answer}</Typography>
                            </AccordionDetails>
                          </Accordion>
                        ))}
                      </Box>
                    )}

                    {filteredArticles.length === 0 && filteredFAQ.length === 0 && (
                      <Alert severity="info">
                        No results found for "{searchQuery}". Try different keywords or browse categories below.
                      </Alert>
                    )}
                  </Box>
                ) : (
                  // Default help page content
                  <Box>
                    {/* Welcome Section */}
                    <Box textAlign="center" mb={6}>
                      <Typography variant="h3" component="h1" gutterBottom sx={{ fontWeight: 600 }}>
                        Welcome to T-Bot Help Center
                      </Typography>
                      <Typography variant="h6" color="text.secondary" sx={{ mb: 4, maxWidth: 600, mx: 'auto' }}>
                        Your comprehensive guide to mastering algorithmic trading with T-Bot.
                        Find tutorials, API documentation, and troubleshooting guides.
                      </Typography>
                      
                      <Box display="flex" justifyContent="center" gap={2}>
                        <Button
                          variant="contained"
                          startIcon={<GetStartedIcon />}
                          onClick={() => {
                            setSelectedCategory('getting-started');
                            setSelectedArticle(helpArticles.find(a => a.id === 'quick-start') || null);
                          }}
                        >
                          Quick Start Guide
                        </Button>
                        <Button
                          variant="outlined"
                          startIcon={<PlayIcon />}
                          onClick={() => window.open('/playground', '_blank')}
                        >
                          Try Playground
                        </Button>
                      </Box>
                    </Box>

                    {/* Categories Grid */}
                    <Typography variant="h5" gutterBottom sx={{ fontWeight: 600, mb: 3 }}>
                      Documentation Categories
                    </Typography>
                    
                    <Grid container spacing={3} mb={6}>
                      {helpCategories.map((category) => (
                        <Grid item xs={12} sm={6} md={4} key={category.id}>
                          <Card
                            sx={{
                              cursor: 'pointer',
                              height: '100%',
                              transition: 'transform 0.2s, box-shadow 0.2s',
                              '&:hover': {
                                transform: 'translateY(-4px)',
                                boxShadow: 4,
                              },
                            }}
                            onClick={() => setSelectedCategory(category.id)}
                          >
                            <CardContent sx={{ textAlign: 'center', p: 3 }}>
                              <Box
                                sx={{
                                  display: 'inline-flex',
                                  p: 2,
                                  borderRadius: '50%',
                                  backgroundColor: `${category.color}20`,
                                  color: category.color,
                                  mb: 2,
                                }}
                              >
                                {category.icon}
                              </Box>
                              <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                                {category.title}
                              </Typography>
                              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                                {category.description}
                              </Typography>
                              <Chip
                                label={`${category.articles.length} articles`}
                                size="small"
                                variant="outlined"
                              />
                            </CardContent>
                          </Card>
                        </Grid>
                      ))}
                    </Grid>

                    {/* Quick Access Tabs */}
                    <Paper sx={{ mb: 4 }}>
                      <Tabs value={tabValue} onChange={(e, v) => setTabValue(v)}>
                        <Tab icon={<QAIcon />} label="FAQ" />
                        <Tab icon={<TutorialIcon />} label="Video Tutorials" />
                        <Tab icon={<ContactIcon />} label="Contact Support" />
                      </Tabs>

                      {/* FAQ Tab */}
                      {tabValue === 0 && (
                        <Box sx={{ p: 3 }}>
                          <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                            Frequently Asked Questions
                          </Typography>
                          {faqData.slice(0, 6).map((faq, index) => (
                            <Accordion
                              key={index}
                              expanded={expandedAccordions.includes(`faq-${index}`)}
                              onChange={handleAccordionChange(`faq-${index}`)}
                            >
                              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                                <Typography sx={{ fontWeight: 500 }}>{faq.question}</Typography>
                              </AccordionSummary>
                              <AccordionDetails>
                                <Typography>{faq.answer}</Typography>
                              </AccordionDetails>
                            </Accordion>
                          ))}
                          <Box textAlign="center" mt={3}>
                            <Button variant="outlined">View All FAQ</Button>
                          </Box>
                        </Box>
                      )}

                      {/* Video Tutorials Tab */}
                      {tabValue === 1 && (
                        <Box sx={{ p: 3 }}>
                          <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                            Video Tutorials
                          </Typography>
                          <Grid container spacing={3}>
                            {[
                              { title: 'Getting Started with T-Bot', duration: '5:30', level: 'Beginner' },
                              { title: 'Setting Up Your First Bot', duration: '8:45', level: 'Beginner' },
                              { title: 'Advanced Strategy Configuration', duration: '12:20', level: 'Advanced' },
                              { title: 'Risk Management Best Practices', duration: '7:15', level: 'Intermediate' },
                            ].map((video, index) => (
                              <Grid item xs={12} sm={6} md={3} key={index}>
                                <Card>
                                  <Box
                                    sx={{
                                      height: 120,
                                      backgroundColor: 'action.hover',
                                      display: 'flex',
                                      alignItems: 'center',
                                      justifyContent: 'center',
                                      cursor: 'pointer',
                                    }}
                                  >
                                    <PlayIcon sx={{ fontSize: 40, color: 'primary.main' }} />
                                  </Box>
                                  <CardContent>
                                    <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                                      {video.title}
                                    </Typography>
                                    <Box display="flex" justifyContent="between" alignItems="center" mt={1}>
                                      <Typography variant="caption" color="text.secondary">
                                        {video.duration}
                                      </Typography>
                                      <Chip
                                        label={video.level}
                                        size="small"
                                        color={
                                          video.level === 'Beginner' ? 'success' :
                                          video.level === 'Intermediate' ? 'warning' : 'error'
                                        }
                                      />
                                    </Box>
                                  </CardContent>
                                </Card>
                              </Grid>
                            ))}
                          </Grid>
                        </Box>
                      )}

                      {/* Contact Support Tab */}
                      {tabValue === 2 && (
                        <Box sx={{ p: 3 }}>
                          <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                            Get Help & Support
                          </Typography>
                          <Grid container spacing={3}>
                            <Grid item xs={12} md={6}>
                              <Card>
                                <CardContent>
                                  <Typography variant="h6" gutterBottom>
                                    📧 Email Support
                                  </Typography>
                                  <Typography variant="body2" color="text.secondary" paragraph>
                                    Get help from our technical support team. We typically respond within 24 hours.
                                  </Typography>
                                  <Button variant="contained" href="mailto:support@tbot.trading">
                                    Contact Support
                                  </Button>
                                </CardContent>
                              </Card>
                            </Grid>
                            <Grid item xs={12} md={6}>
                              <Card>
                                <CardContent>
                                  <Typography variant="h6" gutterBottom>
                                    💬 Community Forum
                                  </Typography>
                                  <Typography variant="body2" color="text.secondary" paragraph>
                                    Join our community to ask questions and share knowledge with other traders.
                                  </Typography>
                                  <Button variant="outlined" href="https://community.tbot.trading" target="_blank">
                                    Visit Forum
                                  </Button>
                                </CardContent>
                              </Card>
                            </Grid>
                            <Grid item xs={12}>
                              <Alert severity="info">
                                <Typography variant="body2">
                                  For urgent technical issues affecting live trading, please contact our emergency support line at +1-555-TBOT-911.
                                </Typography>
                              </Alert>
                            </Grid>
                          </Grid>
                        </Box>
                      )}
                    </Paper>

                    {/* System Requirements */}
                    <Paper sx={{ p: 3, mb: 4 }}>
                      <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                        System Requirements
                      </Typography>
                      <Grid container spacing={3}>
                        <Grid item xs={12} md={6}>
                          <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 600 }}>
                            Minimum Requirements
                          </Typography>
                          <List dense>
                            <ListItem>
                              <ListItemText primary="Modern web browser (Chrome 90+, Firefox 88+, Safari 14+)" />
                            </ListItem>
                            <ListItem>
                              <ListItemText primary="Stable internet connection (minimum 1 Mbps)" />
                            </ListItem>
                            <ListItem>
                              <ListItemText primary="JavaScript enabled" />
                            </ListItem>
                            <ListItem>
                              <ListItemText primary="Minimum screen resolution: 1024x768" />
                            </ListItem>
                          </List>
                        </Grid>
                        <Grid item xs={12} md={6}>
                          <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 600 }}>
                            Recommended
                          </Typography>
                          <List dense>
                            <ListItem>
                              <ListItemText primary="Latest Chrome or Firefox browser" />
                            </ListItem>
                            <ListItem>
                              <ListItemText primary="High-speed internet (10+ Mbps)" />
                            </ListItem>
                            <ListItem>
                              <ListItemText primary="Full HD display (1920x1080)" />
                            </ListItem>
                            <ListItem>
                              <ListItemText primary="Hardware acceleration enabled" />
                            </ListItem>
                          </List>
                        </Grid>
                      </Grid>
                    </Paper>
                  </Box>
                )}
              </Box>
            )}
          </Box>
        </Box>

        {/* Keyboard Shortcuts Dialog */}
        <Dialog open={showShortcuts} onClose={() => setShowShortcuts(false)} maxWidth="sm" fullWidth>
          <DialogTitle>Keyboard Shortcuts</DialogTitle>
          <DialogContent>
            <List>
              {keyboardShortcuts.map((shortcut, index) => (
                <ListItem key={index}>
                  <Box display="flex" justifyContent="between" width="100%">
                    <Typography variant="body2">{shortcut.description}</Typography>
                    <Chip
                      label={shortcut.key}
                      size="small"
                      variant="outlined"
                      sx={{ fontFamily: 'monospace' }}
                    />
                  </Box>
                </ListItem>
              ))}
            </List>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setShowShortcuts(false)}>Close</Button>
          </DialogActions>
        </Dialog>
      </Box>
    </motion.div>
  );
};

export default HelpPage;