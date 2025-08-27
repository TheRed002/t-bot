/**
 * Modern Bot Creation Wizard Component using Shadcn/ui
 * Enhanced multi-step wizard with comprehensive strategy template integration,
 * dynamic parameter configuration, and professional trading UI
 */

import React, { useState, useCallback, useEffect, useMemo } from "react";
import { useDispatch, useSelector } from "react-redux";
import { motion, AnimatePresence } from "framer-motion";
import { useQuery } from "@tanstack/react-query";

// Redux
import { createBot } from "@/store/slices/botSlice";
import type { RootState } from "@/store";

// API Services
import { cachedStrategyAPI } from "@/services/api/strategyAPI";

// Types
import {
  BotType,
  BotPriority,
  BotConfiguration,
  CreateBotRequest,
  StrategyType,
  StrategyCategory,
  RiskLevel,
  PositionSizeMethod,
  AllocationStrategy,
  StrategyTemplate,
  StrategyParameterDefinition,
  RiskConfiguration,
} from "@/types";

// Shadcn/ui components
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Progress } from "@/components/ui/progress";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

// Lucide React icons
import {
  Bot,
  TrendingUp,
  ArrowLeftRight,
  BarChart3,
  Target,
  DollarSign,
  Shield,
  CheckCircle,
  AlertTriangle,
  ChevronLeft,
  ChevronRight,
  Rocket,
  Settings,
  X,
  Search,
  Filter,
  Zap,
  Brain,
  Activity,
  Layers,
  Database,
  PieChart,
  LineChart,
  CandlestickChart,
  TrendingDown,
  Shuffle,
  GitBranch,
  Compass,
  Lock,
  Unlock,
  AlertCircle,
  Info,
  Star,
  Clock,
  Users,
  Award,
  Gauge,
} from "lucide-react";

// Utils
import { cn } from "@/lib/utils";

interface BotCreationWizardProps {
  open: boolean;
  onClose: () => void;
  onSuccess?: (botId: string) => void;
}

// Bot types with modern styling - updated to match backend BotType enum
const BOT_TYPES = [
  {
    value: BotType.TRADING,
    label: "Trading Bot",
    description: "Single strategy focused trading",
    icon: <TrendingUp className="h-5 w-5" />,
    color: "bg-green-500",
    recommended: true,
  },
  {
    value: BotType.ARBITRAGE,
    label: "Arbitrage Bot",
    description: "Cross-exchange opportunities",
    icon: <ArrowLeftRight className="h-5 w-5" />,
    color: "bg-blue-500",
    recommended: false,
  },
  {
    value: BotType.MARKET_MAKING,
    label: "Market Making Bot",
    description: "Provide liquidity to markets",
    icon: <BarChart3 className="h-5 w-5" />,
    color: "bg-orange-500",
    recommended: false,
  },
];

// Comprehensive Strategy Configuration with 15+ Strategies
interface StrategyInfo {
  type: StrategyType;
  label: string;
  category: StrategyCategory;
  riskLevel: RiskLevel;
  description: string;
  longDescription: string;
  icon: React.ReactNode;
  color: string;
  bgColor: string;
  features: string[];
  minCapital: number;
  avgReturn: string;
  complexity: 'beginner' | 'intermediate' | 'advanced' | 'expert';
  timeframe: string[];
  popularity: number; // 1-5 stars
  isRecommended?: boolean;
}

const COMPREHENSIVE_STRATEGIES: StrategyInfo[] = [
  // Static Strategies
  {
    type: StrategyType.MEAN_REVERSION,
    label: "Mean Reversion",
    category: StrategyCategory.STATIC,
    riskLevel: RiskLevel.CONSERVATIVE,
    description: "Counter-trend strategy exploiting price reversals",
    longDescription: "Identifies overbought/oversold conditions and trades against the current trend, expecting prices to return to their historical mean.",
    icon: <TrendingDown className="h-5 w-5" />,
    color: "text-green-600",
    bgColor: "bg-green-50",
    features: ["RSI signals", "Bollinger Bands", "Statistical analysis", "Risk control"],
    minCapital: 1000,
    avgReturn: "8-12%",
    complexity: 'intermediate',
    timeframe: ['5m', '15m', '1h', '4h'],
    popularity: 4,
    isRecommended: true
  },
  {
    type: StrategyType.MOMENTUM,
    label: "Momentum Trading",
    category: StrategyCategory.STATIC,
    riskLevel: RiskLevel.MODERATE,
    description: "Trend-following with momentum indicators",
    longDescription: "Capitalizes on strong price movements by following trends with confirmation from momentum oscillators and volume analysis.",
    icon: <TrendingUp className="h-5 w-5" />,
    color: "text-blue-600",
    bgColor: "bg-blue-50",
    features: ["MACD signals", "Volume analysis", "Breakout detection", "Trend confirmation"],
    minCapital: 2000,
    avgReturn: "12-18%",
    complexity: 'beginner',
    timeframe: ['15m', '1h', '4h', '1d'],
    popularity: 5,
    isRecommended: true
  },
  {
    type: StrategyType.ARBITRAGE,
    label: "Statistical Arbitrage",
    category: StrategyCategory.STATIC,
    riskLevel: RiskLevel.CONSERVATIVE,
    description: "Risk-free profit from price differences",
    longDescription: "Exploits temporary price inefficiencies across different markets or time periods with statistical models.",
    icon: <ArrowLeftRight className="h-5 w-5" />,
    color: "text-purple-600",
    bgColor: "bg-purple-50",
    features: ["Cross-exchange", "Pair trading", "Statistical models", "Low risk"],
    minCapital: 5000,
    avgReturn: "6-10%",
    complexity: 'advanced',
    timeframe: ['1m', '5m', '15m'],
    popularity: 3
  },
  {
    type: StrategyType.MARKET_MAKING,
    label: "Market Making",
    category: StrategyCategory.STATIC,
    riskLevel: RiskLevel.CONSERVATIVE,
    description: "Provide liquidity and profit from spreads",
    longDescription: "Places both buy and sell orders around current market price to capture bid-ask spreads while providing market liquidity.",
    icon: <BarChart3 className="h-5 w-5" />,
    color: "text-orange-600",
    bgColor: "bg-orange-50",
    features: ["Bid-ask spreads", "Inventory management", "Liquidity provision", "Low volatility"],
    minCapital: 10000,
    avgReturn: "5-8%",
    complexity: 'expert',
    timeframe: ['1m', '5m'],
    popularity: 2
  },
  {
    type: StrategyType.TREND_FOLLOWING,
    label: "Trend Following",
    category: StrategyCategory.STATIC,
    riskLevel: RiskLevel.MODERATE,
    description: "Long-term trend capture system",
    longDescription: "Identifies and follows major market trends using moving averages, trend lines, and momentum indicators for sustained moves.",
    icon: <LineChart className="h-5 w-5" />,
    color: "text-indigo-600",
    bgColor: "bg-indigo-50",
    features: ["Moving averages", "Trend detection", "Position scaling", "Long-term holds"],
    minCapital: 3000,
    avgReturn: "15-25%",
    complexity: 'intermediate',
    timeframe: ['4h', '1d', '1w'],
    popularity: 4
  },
  {
    type: StrategyType.PAIRS_TRADING,
    label: "Pairs Trading",
    category: StrategyCategory.STATIC,
    riskLevel: RiskLevel.CONSERVATIVE,
    description: "Market-neutral strategy using correlated pairs",
    longDescription: "Trades two historically correlated securities by going long one and short the other when their relationship diverges.",
    icon: <Shuffle className="h-5 w-5" />,
    color: "text-teal-600",
    bgColor: "bg-teal-50",
    features: ["Correlation analysis", "Market neutral", "Statistical models", "Risk balanced"],
    minCapital: 8000,
    avgReturn: "8-12%",
    complexity: 'advanced',
    timeframe: ['1h', '4h', '1d'],
    popularity: 3
  },
  {
    type: StrategyType.STATISTICAL_ARBITRAGE,
    label: "Statistical Arbitrage",
    category: StrategyCategory.STATIC,
    riskLevel: RiskLevel.MODERATE,
    description: "Advanced statistical models for price prediction",
    longDescription: "Uses complex statistical models and mean reversion principles to identify and exploit temporary mispricings in related securities.",
    icon: <Database className="h-5 w-5" />,
    color: "text-violet-600",
    bgColor: "bg-violet-50",
    features: ["Statistical models", "Mean reversion", "Multi-asset", "Quantitative analysis"],
    minCapital: 15000,
    avgReturn: "10-15%",
    complexity: 'expert',
    timeframe: ['5m', '15m', '1h'],
    popularity: 2
  },
  {
    type: StrategyType.BREAKOUT,
    label: "Breakout Strategy",
    category: StrategyCategory.STATIC,
    riskLevel: RiskLevel.AGGRESSIVE,
    description: "Capture explosive price movements",
    longDescription: "Identifies key support and resistance levels and trades breakouts with high volume confirmation for explosive moves.",
    icon: <Zap className="h-5 w-5" />,
    color: "text-red-600",
    bgColor: "bg-red-50",
    features: ["Support/resistance", "Volume confirmation", "Momentum capture", "High volatility"],
    minCapital: 2500,
    avgReturn: "20-35%",
    complexity: 'intermediate',
    timeframe: ['5m', '15m', '1h'],
    popularity: 4
  },
  {
    type: StrategyType.CROSS_EXCHANGE_ARBITRAGE,
    label: "Cross-Exchange Arbitrage",
    category: StrategyCategory.STATIC,
    riskLevel: RiskLevel.CONSERVATIVE,
    description: "Profit from price differences across exchanges",
    longDescription: "Simultaneously buys and sells the same asset on different exchanges to profit from price discrepancies.",
    icon: <GitBranch className="h-5 w-5" />,
    color: "text-emerald-600",
    bgColor: "bg-emerald-50",
    features: ["Multi-exchange", "Price arbitrage", "Instant execution", "Risk-free profits"],
    minCapital: 5000,
    avgReturn: "4-8%",
    complexity: 'advanced',
    timeframe: ['1m', '5m'],
    popularity: 3
  },
  {
    type: StrategyType.TRIANGULAR_ARBITRAGE,
    label: "Triangular Arbitrage",
    category: StrategyCategory.STATIC,
    riskLevel: RiskLevel.CONSERVATIVE,
    description: "Currency arbitrage using three trading pairs",
    longDescription: "Exploits pricing inefficiencies between three currency pairs by executing a series of trades that return to the original currency.",
    icon: <Compass className="h-5 w-5" />,
    color: "text-cyan-600",
    bgColor: "bg-cyan-50",
    features: ["Three-way arbitrage", "Currency cycles", "Fast execution", "Mathematical precision"],
    minCapital: 10000,
    avgReturn: "3-6%",
    complexity: 'expert',
    timeframe: ['1m'],
    popularity: 2
  },

  // Dynamic Strategies
  {
    type: StrategyType.VOLATILITY_BREAKOUT,
    label: "Volatility Breakout",
    category: StrategyCategory.DYNAMIC,
    riskLevel: RiskLevel.AGGRESSIVE,
    description: "Dynamic volatility-based entry system",
    longDescription: "Adapts to market volatility conditions and triggers trades when volatility exceeds dynamic thresholds for maximum profit potential.",
    icon: <Activity className="h-5 w-5" />,
    color: "text-amber-600",
    bgColor: "bg-amber-50",
    features: ["Volatility analysis", "Dynamic thresholds", "Adaptive sizing", "Market regime detection"],
    minCapital: 5000,
    avgReturn: "25-40%",
    complexity: 'advanced',
    timeframe: ['15m', '1h', '4h'],
    popularity: 3
  },

  // Hybrid Strategies
  {
    type: StrategyType.ENSEMBLE,
    label: "Ensemble Strategy",
    category: StrategyCategory.HYBRID,
    riskLevel: RiskLevel.MODERATE,
    description: "Combines multiple strategies for optimal performance",
    longDescription: "Integrates signals from multiple trading strategies using weighted voting and machine learning to optimize overall performance.",
    icon: <Layers className="h-5 w-5" />,
    color: "text-slate-600",
    bgColor: "bg-slate-50",
    features: ["Multi-strategy", "Signal aggregation", "Risk diversification", "Performance optimization"],
    minCapital: 7500,
    avgReturn: "12-20%",
    complexity: 'advanced',
    timeframe: ['15m', '1h', '4h'],
    popularity: 4
  },
  {
    type: StrategyType.FALLBACK,
    label: "Fallback Strategy",
    category: StrategyCategory.HYBRID,
    riskLevel: RiskLevel.CONSERVATIVE,
    description: "Backup strategy with risk management focus",
    longDescription: "Provides fallback trading logic with enhanced risk controls when primary strategies encounter adverse market conditions.",
    icon: <Shield className="h-5 w-5" />,
    color: "text-gray-600",
    bgColor: "bg-gray-50",
    features: ["Risk management", "Fallback logic", "Capital preservation", "Market adaptation"],
    minCapital: 3000,
    avgReturn: "5-10%",
    complexity: 'intermediate',
    timeframe: ['1h', '4h', '1d'],
    popularity: 3
  },
  {
    type: StrategyType.RULE_BASED_AI,
    label: "Rule-Based AI",
    category: StrategyCategory.HYBRID,
    riskLevel: RiskLevel.MODERATE,
    description: "AI-enhanced rule-based trading system",
    longDescription: "Combines traditional rule-based trading with AI pattern recognition and adaptive learning for improved decision making.",
    icon: <Brain className="h-5 w-5" />,
    color: "text-pink-600",
    bgColor: "bg-pink-50",
    features: ["AI patterns", "Rule adaptation", "Machine learning", "Pattern recognition"],
    minCapital: 10000,
    avgReturn: "15-28%",
    complexity: 'expert',
    timeframe: ['5m', '15m', '1h'],
    popularity: 4
  },

  // Custom Strategy
  {
    type: StrategyType.CUSTOM,
    label: "Custom Strategy",
    category: StrategyCategory.CUSTOM,
    riskLevel: RiskLevel.EXPERIMENTAL,
    description: "Build your own custom trading strategy",
    longDescription: "Create and configure a completely custom trading strategy with your own parameters, indicators, and trading logic.",
    icon: <Settings className="h-5 w-5" />,
    color: "text-neutral-600",
    bgColor: "bg-neutral-50",
    features: ["Custom logic", "Flexible parameters", "Advanced configuration", "Unlimited possibilities"],
    minCapital: 1000,
    avgReturn: "Variable",
    complexity: 'expert',
    timeframe: ['1m', '5m', '15m', '1h', '4h', '1d'],
    popularity: 2
  }
];

// Strategy Categories with metadata
const STRATEGY_CATEGORIES = [
  {
    category: StrategyCategory.STATIC,
    label: "Static Strategies",
    description: "Traditional, rule-based strategies with fixed parameters",
    icon: <Lock className="h-4 w-4" />,
    color: "text-blue-600",
    count: COMPREHENSIVE_STRATEGIES.filter(s => s.category === StrategyCategory.STATIC).length
  },
  {
    category: StrategyCategory.DYNAMIC,
    label: "Dynamic Strategies", 
    description: "Adaptive strategies that adjust to market conditions",
    icon: <Unlock className="h-4 w-4" />,
    color: "text-green-600",
    count: COMPREHENSIVE_STRATEGIES.filter(s => s.category === StrategyCategory.DYNAMIC).length
  },
  {
    category: StrategyCategory.HYBRID,
    label: "Hybrid Strategies",
    description: "Advanced combinations of multiple approaches",
    icon: <Layers className="h-4 w-4" />,
    color: "text-purple-600",
    count: COMPREHENSIVE_STRATEGIES.filter(s => s.category === StrategyCategory.HYBRID).length
  },
  {
    category: StrategyCategory.CUSTOM,
    label: "Custom Strategies",
    description: "User-defined strategies with custom logic",
    icon: <Settings className="h-4 w-4" />,
    color: "text-orange-600",
    count: COMPREHENSIVE_STRATEGIES.filter(s => s.category === StrategyCategory.CUSTOM).length
  }
];

const EXCHANGES = [
  { value: "binance", label: "Binance", icon: "🟡", fees: "0.1%" },
  { value: "coinbase", label: "Coinbase", icon: "🔵", fees: "0.5%" },
  { value: "okx", label: "OKX", icon: "⚪", fees: "0.1%" },
];

const SYMBOLS = [
  "BTC/USDT",
  "ETH/USDT",
  "BNB/USDT",
  "SOL/USDT",
  "ADA/USDT",
  "DOT/USDT",
  "MATIC/USDT",
  "LINK/USDT",
];

// Enhanced Risk Levels with detailed configuration
const ENHANCED_RISK_LEVELS = [
  {
    value: RiskLevel.CONSERVATIVE,
    label: "Conservative",
    percent: "1-2%",
    color: "text-green-600 bg-green-100",
    description: "Low risk, steady returns",
    maxRiskPerTrade: 0.01,
    maxDrawdown: 0.05,
    features: ["Capital preservation", "Low volatility", "Steady growth"]
  },
  {
    value: RiskLevel.MODERATE,
    label: "Moderate",
    percent: "2-3%",
    color: "text-blue-600 bg-blue-100",
    description: "Balanced risk-reward profile",
    maxRiskPerTrade: 0.02,
    maxDrawdown: 0.1,
    features: ["Balanced approach", "Moderate volatility", "Good diversification"]
  },
  {
    value: RiskLevel.AGGRESSIVE,
    label: "Aggressive",
    percent: "3-5%",
    color: "text-orange-600 bg-orange-100",
    description: "Higher risk for potential higher returns",
    maxRiskPerTrade: 0.05,
    maxDrawdown: 0.2,
    features: ["Growth focused", "Higher volatility", "Active management"]
  },
  {
    value: RiskLevel.EXPERIMENTAL,
    label: "Experimental",
    percent: "5%+",
    color: "text-red-600 bg-red-100",
    description: "Maximum risk for experimental strategies",
    maxRiskPerTrade: 0.1,
    maxDrawdown: 0.3,
    features: ["High risk", "Experimental", "Maximum potential"]
  },
];

// Position sizing methods
const POSITION_SIZE_METHODS = [
  {
    value: PositionSizeMethod.FIXED,
    label: "Fixed Amount",
    description: "Trade with a fixed dollar amount",
    icon: <DollarSign className="h-4 w-4" />
  },
  {
    value: PositionSizeMethod.PERCENTAGE,
    label: "Percentage of Capital",
    description: "Trade with a percentage of total capital",
    icon: <PieChart className="h-4 w-4" />
  },
  {
    value: PositionSizeMethod.KELLY_CRITERION,
    label: "Kelly Criterion",
    description: "Optimal position sizing based on win probability",
    icon: <Brain className="h-4 w-4" />
  },
  {
    value: PositionSizeMethod.RISK_PARITY,
    label: "Risk Parity",
    description: "Equal risk contribution from all positions",
    icon: <BarChart3 className="h-4 w-4" />
  },
  {
    value: PositionSizeMethod.VOLATILITY_ADJUSTED,
    label: "Volatility Adjusted",
    description: "Adjust position size based on market volatility",
    icon: <Activity className="h-4 w-4" />
  }
];

const BotCreationWizard: React.FC<BotCreationWizardProps> = ({
  open,
  onClose,
  onSuccess,
}) => {
  const dispatch = useDispatch();
  const { isLoading, error: reduxError } = useSelector(
    (state: RootState) => state.bots,
  );

  const [currentStep, setCurrentStep] = useState(0);
  const [error, setError] = useState<string | null>(null);

  // Enhanced form data with comprehensive strategy configuration
  const [formData, setFormData] = useState({
    // Step 1: Basic Info
    botName: "",
    botType: BotType.TRADING,
    description: "",

    // Step 2: Strategy Category & Selection
    selectedCategory: null as StrategyCategory | null,
    strategy: "" as StrategyType | "",
    timeframe: "1h",

    // Step 3: Template Selection
    selectedTemplate: null as StrategyTemplate | null,
    templateVariant: "conservative" as 'conservative' | 'moderate' | 'aggressive',
    useTemplate: true,

    // Step 4: Dynamic Parameters (populated from template)
    strategyParameters: {} as Record<string, any>,
    parameterValidation: {} as Record<string, string | null>,

    // Step 5: Markets
    exchanges: [] as string[],
    symbols: [] as string[],

    // Step 6: Enhanced Risk & Capital Configuration
    capital: 10000,
    riskLevel: RiskLevel.MODERATE,
    positionSizeMethod: PositionSizeMethod.PERCENTAGE,
    positionSizeValue: 2, // 2% of capital
    maxRiskPerTrade: 0.02,
    stopLoss: 2,
    takeProfit: 5,
    maxDrawdown: 10,
    correlationLimit: 0.7,
    
    // Circuit Breakers
    enableCircuitBreaker: true,
    dailyLossLimit: 5, // 5% daily loss limit
    consecutiveLossLimit: 3,

    // Position Management
    maxPositions: 5,
    allocationStrategy: AllocationStrategy.EQUAL_WEIGHT,

    // Step 7: Advanced Settings
    autoStart: false,
    priority: BotPriority.NORMAL,
    maxTrades: 100,
    
    // Execution Settings
    maxSlippage: 0.1, // 0.1%
    orderTimeout: 30, // 30 seconds
    retryAttempts: 3,
    fillStrategy: 'smart' as 'market' | 'limit' | 'smart',
  });

  // Filter and search state for strategy selection
  const [strategyFilters, setStrategyFilters] = useState({
    category: null as StrategyCategory | null,
    riskLevel: null as RiskLevel | 'all' | null,
    complexity: null as 'beginner' | 'intermediate' | 'advanced' | 'expert' | 'all' | null,
    search: ""
  });

  // Template loading state
  const [selectedStrategyInfo, setSelectedStrategyInfo] = useState<StrategyInfo | null>(null);
  const [loadingTemplate, setLoadingTemplate] = useState(false);
  const [templateError, setTemplateError] = useState<string | null>(null);

  // Enhanced wizard steps with template integration
  const steps = [
    { title: "Bot Info", icon: Bot, description: "Basic bot configuration" },
    {
      title: "Strategy",
      icon: TrendingUp,
      description: "Select strategy category & type",
    },
    {
      title: "Template",
      icon: Database,
      description: "Choose strategy template",
    },
    {
      title: "Parameters",
      icon: Settings,
      description: "Configure strategy parameters",
    },
    { title: "Markets", icon: Target, description: "Select exchanges & pairs" },
    {
      title: "Risk Setup",
      icon: Shield,
      description: "Configure risk management",
    },
    { title: "Review", icon: CheckCircle, description: "Review and launch" },
  ];

  // Strategy template API integration
  const {
    data: availableTemplates,
    isLoading: templatesLoading,
    error: templatesError
  } = useQuery({
    queryKey: ['strategy-templates', strategyFilters],
    queryFn: () => cachedStrategyAPI.getTemplates({
      category: strategyFilters.category ? [strategyFilters.category] : undefined,
      riskLevel: strategyFilters.riskLevel && strategyFilters.riskLevel !== 'all' ? [strategyFilters.riskLevel] : undefined,
      search: strategyFilters.search || undefined,
      limit: 50
    }),
    enabled: formData.strategy !== "",
    staleTime: 5 * 60 * 1000, // 5 minutes
  });

  // Get specific template when selected
  const {
    data: selectedTemplateData,
    isLoading: templateLoading,
    error: templateLoadError
  } = useQuery({
    queryKey: ['strategy-template', formData.selectedTemplate?.id],
    queryFn: () => formData.selectedTemplate ? 
      cachedStrategyAPI.getTemplate(formData.selectedTemplate.id, true) : 
      Promise.resolve(null),
    enabled: !!formData.selectedTemplate?.id,
  });

  const handleNext = useCallback(() => {
    if (currentStep < steps.length - 1) {
      setCurrentStep((prev) => prev + 1);
    }
  }, [currentStep, steps.length]);

  const handleBack = useCallback(() => {
    if (currentStep > 0) {
      setCurrentStep((prev) => prev - 1);
    }
  }, [currentStep]);

  const handleFieldChange = useCallback((field: string, value: any) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
    
    // Handle strategy selection changes
    if (field === 'strategy') {
      const strategyInfo = COMPREHENSIVE_STRATEGIES.find(s => s.type === value);
      setSelectedStrategyInfo(strategyInfo || null);
      
      // Reset template selection when strategy changes
      if (value !== formData.strategy) {
        setFormData(prevData => ({
          ...prevData,
          selectedTemplate: null,
          strategyParameters: {},
          parameterValidation: {}
        }));
      }
    }
    
    // Handle risk level changes
    if (field === 'riskLevel') {
      const riskConfig = ENHANCED_RISK_LEVELS.find(r => r.value === value);
      if (riskConfig) {
        setFormData(prevData => ({
          ...prevData,
          riskLevel: value,
          maxRiskPerTrade: riskConfig.maxRiskPerTrade,
          maxDrawdown: riskConfig.maxDrawdown * 100 // Convert to percentage
        }));
      }
    }
  }, []);

  // Parameter validation function
  const validateParameter = useCallback((param: StrategyParameterDefinition, value: any): string | null => {
    if (param.required && (value === undefined || value === null || value === '')) {
      return `${param.displayName} is required`;
    }
    
    if (param.validation) {
      const { min, max, pattern, customValidator } = param.validation;
      
      if (typeof value === 'number') {
        if (min !== undefined && value < min) {
          return `${param.displayName} must be at least ${min}`;
        }
        if (max !== undefined && value > max) {
          return `${param.displayName} must be at most ${max}`;
        }
      }
      
      if (typeof value === 'string' && pattern) {
        const regex = new RegExp(pattern);
        if (!regex.test(value)) {
          return `${param.displayName} format is invalid`;
        }
      }
      
      if (customValidator) {
        return customValidator(value);
      }
    }
    
    return null;
  }, []);

  // Handle parameter changes with validation
  const handleParameterChange = useCallback((paramName: string, value: any) => {
    setFormData(prev => ({
      ...prev,
      strategyParameters: {
        ...prev.strategyParameters,
        [paramName]: value
      }
    }));
    
    // Validate parameter if template is loaded
    if (selectedTemplateData?.template) {
      const param = selectedTemplateData.template.parameters.find(p => p.name === paramName);
      if (param) {
        const validationError = validateParameter(param, value);
        setFormData(prev => ({
          ...prev,
          parameterValidation: {
            ...prev.parameterValidation,
            [paramName]: validationError
          }
        }));
      }
    }
  }, [selectedTemplateData, validateParameter]);

  // Filter strategies based on current filters
  const filteredStrategies = useMemo(() => {
    let strategies = COMPREHENSIVE_STRATEGIES;
    
    if (strategyFilters.category) {
      strategies = strategies.filter(s => s.category === strategyFilters.category);
    }
    
    if (strategyFilters.riskLevel && strategyFilters.riskLevel !== 'all') {
      strategies = strategies.filter(s => s.riskLevel === strategyFilters.riskLevel);
    }
    
    if (strategyFilters.complexity && strategyFilters.complexity !== 'all') {
      strategies = strategies.filter(s => s.complexity === strategyFilters.complexity);
    }
    
    if (strategyFilters.search) {
      const search = strategyFilters.search.toLowerCase();
      strategies = strategies.filter(s => 
        s.label.toLowerCase().includes(search) ||
        s.description.toLowerCase().includes(search) ||
        s.features.some(f => f.toLowerCase().includes(search))
      );
    }
    
    return strategies.sort((a, b) => {
      // Sort by recommendation first, then popularity
      if (a.isRecommended && !b.isRecommended) return -1;
      if (!a.isRecommended && b.isRecommended) return 1;
      return b.popularity - a.popularity;
    });
  }, [strategyFilters]);

  const handleSubmit = useCallback(async () => {
    setError(null);

    try {
      // Validate required fields
      if (!formData.botName.trim()) {
        throw new Error("Bot name is required");
      }
      if (!formData.strategy) {
        throw new Error("Strategy selection is required");
      }
      if (formData.exchanges.length === 0) {
        throw new Error("At least one exchange must be selected");
      }
      if (formData.symbols.length === 0) {
        throw new Error("At least one trading pair must be selected");
      }
      if (formData.capital <= 0) {
        throw new Error("Allocated capital must be greater than 0");
      }

      // Create enhanced bot configuration matching backend structure
      const riskConfig = ENHANCED_RISK_LEVELS.find(r => r.value === formData.riskLevel);
      
      const botConfiguration: BotConfiguration = {
        bot_id: "", // Will be set by backend
        bot_name: formData.botName.trim(),
        bot_type: formData.botType,
        strategy_name: formData.strategy as StrategyType,
        exchanges: formData.exchanges,
        symbols: formData.symbols,
        allocated_capital: formData.capital,
        risk_percentage: riskConfig?.maxRiskPerTrade || 0.02,
        priority: formData.priority,
        auto_start: formData.autoStart,
        strategy_config: {
          // Basic configuration
          timeframe: formData.timeframe,
          stop_loss_percentage: formData.stopLoss / 100,
          take_profit_percentage: formData.takeProfit / 100,
          max_daily_trades: formData.maxTrades,
          description: formData.description,
          
          // Template configuration
          template_id: formData.selectedTemplate?.id,
          template_variant: formData.templateVariant,
          use_template: formData.useTemplate,
          
          // Enhanced risk management
          risk_level: formData.riskLevel,
          position_size_method: formData.positionSizeMethod,
          position_size_value: formData.positionSizeValue,
          max_positions: formData.maxPositions,
          max_drawdown_percentage: formData.maxDrawdown / 100,
          correlation_limit: formData.correlationLimit,
          
          // Circuit breakers
          enable_circuit_breaker: formData.enableCircuitBreaker,
          daily_loss_limit: formData.dailyLossLimit / 100,
          consecutive_loss_limit: formData.consecutiveLossLimit,
          
          // Execution settings
          max_slippage: formData.maxSlippage / 100,
          order_timeout: formData.orderTimeout,
          retry_attempts: formData.retryAttempts,
          fill_strategy: formData.fillStrategy,
          
          // Strategy parameters
          strategy_parameters: formData.strategyParameters,
          
          // Allocation strategy
          allocation_strategy: formData.allocationStrategy,
        },
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      };

      // Prepare enhanced create bot request
      const createBotData: CreateBotRequest = {
        bot_name: formData.botName.trim(),
        bot_type: formData.botType,
        strategy_name: formData.strategy as StrategyType,
        exchanges: formData.exchanges,
        symbols: formData.symbols,
        allocated_capital: formData.capital,
        risk_percentage: riskConfig?.maxRiskPerTrade || 0.02,
        priority: formData.priority,
        auto_start: formData.autoStart,
        configuration: botConfiguration.strategy_config,
      };

      // Validate required template parameters if using template
      if (formData.useTemplate && selectedTemplateData?.template) {
        const requiredParams = selectedTemplateData.template.parameters.filter(p => p.required);
        const missingParams = requiredParams.filter(p => 
          formData.strategyParameters[p.name] === undefined || 
          formData.strategyParameters[p.name] === null ||
          formData.strategyParameters[p.name] === ''
        );
        
        if (missingParams.length > 0) {
          throw new Error(`Missing required parameters: ${missingParams.map(p => p.displayName).join(', ')}`);
        }
      }
      
      // Dispatch Redux action - convert to legacy format for Redux slice
      const legacyBotData = {
        bot_name: createBotData.bot_name,
        strategy_name: createBotData.strategy_name,
        exchange: createBotData.exchanges[0], // Primary exchange
        config: botConfiguration,
      };
      
      const result = await dispatch(createBot(legacyBotData) as any);

      if (createBot.fulfilled.match(result)) {
        // Success - bot created
        const createdBot = result.payload;
        if (onSuccess && createdBot.bot_id) {
          onSuccess(createdBot.bot_id);
        }
        onClose();
      } else {
        // Handle Redux rejection
        throw new Error((result.payload as string) || "Failed to create bot");
      }
    } catch (err: any) {
      setError(err.message || "Failed to create bot");
      console.error("Bot creation error:", err);
    }
  }, [formData, onSuccess, onClose, dispatch]);

  const renderStep = () => {
    switch (currentStep) {
      case 0: // Bot Type
        return (
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="space-y-6"
          >
            <div>
              <h3 className="text-lg font-semibold mb-2">Choose Bot Type</h3>
              <p className="text-muted-foreground">
                Select the type of trading bot you want to create
              </p>
            </div>

            <div className="space-y-4">
              <div>
                <Label htmlFor="botName">Bot Name</Label>
                <Input
                  id="botName"
                  placeholder="My Trading Bot"
                  value={formData.botName}
                  onChange={(e) => handleFieldChange("botName", e.target.value)}
                  className="mt-1"
                />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
                {BOT_TYPES.map((type) => (
                  <Card
                    key={type.value}
                    className={cn(
                      "cursor-pointer transition-all hover:shadow-lg border-2",
                      formData.botType === type.value
                        ? "border-primary shadow-lg ring-2 ring-primary/20"
                        : "border-gray-200 hover:border-gray-300",
                    )}
                    onClick={() => handleFieldChange("botType", type.value)}
                  >
                    <CardHeader className="pb-3">
                      <div className="flex items-center justify-between">
                        <div className={cn("p-2 rounded-lg", type.color)}>
                          {type.icon}
                        </div>
                        {type.recommended && (
                          <Badge className="text-xs">Recommended</Badge>
                        )}
                      </div>
                      <CardTitle className="text-base">{type.label}</CardTitle>
                      <CardDescription className="text-sm">
                        {type.description}
                      </CardDescription>
                    </CardHeader>
                  </Card>
                ))}
              </div>

              <div>
                <Label htmlFor="description">Description (Optional)</Label>
                <Textarea
                  id="description"
                  placeholder="Brief description of your bot's purpose..."
                  value={formData.description}
                  onChange={(e) =>
                    handleFieldChange("description", e.target.value)
                  }
                  className="mt-1"
                  rows={3}
                />
              </div>
            </div>
          </motion.div>
        );

      case 1: // Enhanced Strategy Selection
        return (
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="space-y-6"
          >
            <div>
              <h3 className="text-lg font-semibold mb-2">Select Trading Strategy</h3>
              <p className="text-muted-foreground">
                Choose from our comprehensive collection of professional trading strategies
              </p>
            </div>

            {/* Strategy Category Filters */}
            <div className="space-y-4">
              <div className="flex flex-wrap gap-2">
                <Button
                  variant={strategyFilters.category === null ? "default" : "outline"}
                  size="sm"
                  onClick={() => setStrategyFilters(prev => ({ ...prev, category: null }))}
                >
                  All Categories
                </Button>
                {STRATEGY_CATEGORIES.map((cat) => (
                  <Button
                    key={cat.category}
                    variant={strategyFilters.category === cat.category ? "default" : "outline"}
                    size="sm"
                    className={strategyFilters.category === cat.category ? cat.color : ''}
                    onClick={() => setStrategyFilters(prev => ({ 
                      ...prev, 
                      category: prev.category === cat.category ? null : cat.category 
                    }))}
                  >
                    {cat.icon}
                    {cat.label} ({cat.count})
                  </Button>
                ))}
              </div>

              {/* Risk Level and Complexity Filters */}
              <div className="flex flex-wrap gap-2">
                <Select
                  value={strategyFilters.riskLevel || ""}
                  onValueChange={(value) => setStrategyFilters(prev => ({
                    ...prev, 
                    riskLevel: value ? value as RiskLevel : null
                  }))}
                >
                  <SelectTrigger className="w-40">
                    <SelectValue placeholder="Risk Level" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Risk Levels</SelectItem>
                    {ENHANCED_RISK_LEVELS.map((level) => (
                      <SelectItem key={level.value} value={level.value}>
                        {level.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>

                <Select
                  value={strategyFilters.complexity || ""}
                  onValueChange={(value) => setStrategyFilters(prev => ({
                    ...prev,
                    complexity: value ? value as any : null
                  }))}
                >
                  <SelectTrigger className="w-40">
                    <SelectValue placeholder="Complexity" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Levels</SelectItem>
                    <SelectItem value="beginner">Beginner</SelectItem>
                    <SelectItem value="intermediate">Intermediate</SelectItem>
                    <SelectItem value="advanced">Advanced</SelectItem>
                    <SelectItem value="expert">Expert</SelectItem>
                  </SelectContent>
                </Select>

                {/* Search */}
                <div className="relative flex-1 min-w-[200px]">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    placeholder="Search strategies..."
                    value={strategyFilters.search}
                    onChange={(e) => setStrategyFilters(prev => ({ ...prev, search: e.target.value }))}
                    className="pl-10"
                  />
                </div>
              </div>
            </div>

            {/* Strategy Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 max-h-96 overflow-y-auto">
              {filteredStrategies.map((strategy) => (
                <Card
                  key={strategy.type}
                  className={cn(
                    "cursor-pointer transition-all hover:shadow-lg border-2",
                    formData.strategy === strategy.type
                      ? "border-primary shadow-lg ring-2 ring-primary/20"
                      : "border-gray-200 hover:border-gray-300",
                  )}
                  onClick={() => handleFieldChange("strategy", strategy.type)}
                >
                  <CardHeader className="pb-3">
                    <div className="flex items-start justify-between">
                      <div className={cn("p-2 rounded-lg", strategy.bgColor)}>
                        {strategy.icon}
                      </div>
                      <div className="flex flex-col gap-1 items-end">
                        {strategy.isRecommended && (
                          <Badge className="text-xs bg-green-100 text-green-800">
                            <Star className="h-3 w-3 mr-1" />
                            Recommended
                          </Badge>
                        )}
                        <Badge
                          variant={
                            strategy.riskLevel === RiskLevel.CONSERVATIVE
                              ? "default"
                              : strategy.riskLevel === RiskLevel.MODERATE
                                ? "secondary"
                                : "destructive"
                          }
                          className="text-xs"
                        >
                          {strategy.riskLevel}
                        </Badge>
                        <div className="flex items-center gap-1">
                          {[...Array(5)].map((_, i) => (
                            <Star
                              key={i}
                              className={cn(
                                "h-3 w-3",
                                i < strategy.popularity
                                  ? "fill-yellow-400 text-yellow-400"
                                  : "text-gray-300"
                              )}
                            />
                          ))}
                        </div>
                      </div>
                    </div>
                    <CardTitle className="text-base mt-2">{strategy.label}</CardTitle>
                    <CardDescription className="text-sm">
                      {strategy.description}
                    </CardDescription>
                    
                    {/* Strategy Stats */}
                    <div className="grid grid-cols-3 gap-2 mt-3 text-xs">
                      <div className="text-center p-2 bg-gray-50 rounded">
                        <div className="font-semibold">${strategy.minCapital.toLocaleString()}</div>
                        <div className="text-muted-foreground">Min Capital</div>
                      </div>
                      <div className="text-center p-2 bg-gray-50 rounded">
                        <div className="font-semibold">{strategy.avgReturn}</div>
                        <div className="text-muted-foreground">Avg Return</div>
                      </div>
                      <div className="text-center p-2 bg-gray-50 rounded">
                        <div className="font-semibold capitalize">{strategy.complexity}</div>
                        <div className="text-muted-foreground">Complexity</div>
                      </div>
                    </div>

                    {/* Features */}
                    <div className="flex flex-wrap gap-1 mt-2">
                      {strategy.features.slice(0, 3).map((feature, i) => (
                        <Badge key={i} variant="outline" className="text-xs">
                          {feature}
                        </Badge>
                      ))}
                      {strategy.features.length > 3 && (
                        <Badge variant="outline" className="text-xs">
                          +{strategy.features.length - 3} more
                        </Badge>
                      )}
                    </div>
                  </CardHeader>
                </Card>
              ))}
            </div>

            {/* Selected Strategy Details */}
            {selectedStrategyInfo && (
              <Card className="border-primary/50 bg-primary/5">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-base">
                    {selectedStrategyInfo.icon}
                    {selectedStrategyInfo.label} - Configuration
                  </CardTitle>
                  <CardDescription>{selectedStrategyInfo.longDescription}</CardDescription>
                  
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
                    <div>
                      <Label>Recommended Timeframes</Label>
                      <Select
                        value={formData.timeframe}
                        onValueChange={(value) => handleFieldChange("timeframe", value)}
                      >
                        <SelectTrigger className="mt-1">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {selectedStrategyInfo.timeframe.map((tf) => (
                            <SelectItem key={tf} value={tf}>
                              {tf === '1m' ? '1 Minute' :
                               tf === '5m' ? '5 Minutes' :
                               tf === '15m' ? '15 Minutes' :
                               tf === '30m' ? '30 Minutes' :
                               tf === '1h' ? '1 Hour' :
                               tf === '4h' ? '4 Hours' :
                               tf === '1d' ? '1 Day' : tf}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    
                    <div className="text-sm">
                      <Label className="text-muted-foreground">Category</Label>
                      <div className="font-medium capitalize">{selectedStrategyInfo.category}</div>
                    </div>
                    
                    <div className="text-sm">
                      <Label className="text-muted-foreground">Min Capital</Label>
                      <div className="font-medium">${selectedStrategyInfo.minCapital.toLocaleString()}</div>
                    </div>
                    
                    <div className="text-sm">
                      <Label className="text-muted-foreground">Expected Return</Label>
                      <div className="font-medium">{selectedStrategyInfo.avgReturn}</div>
                    </div>
                  </div>
                </CardHeader>
              </Card>
            )}

            {!filteredStrategies.length && (
              <div className="text-center py-8 text-muted-foreground">
                <Database className="h-12 w-12 mx-auto mb-2 opacity-50" />
                <p>No strategies match your current filters</p>
                <Button 
                  variant="outline" 
                  size="sm" 
                  className="mt-2"
                  onClick={() => setStrategyFilters({ category: null, riskLevel: null, complexity: null, search: "" })}
                >
                  Clear Filters
                </Button>
              </div>
            )}
          </motion.div>
        );

      case 2: // Template Selection
        return (
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="space-y-6"
          >
            <div>
              <h3 className="text-lg font-semibold mb-2">Strategy Template</h3>
              <p className="text-muted-foreground">
                Choose a pre-configured template or customize your strategy
              </p>
            </div>

            {/* Template Options */}
            <Tabs value={formData.useTemplate ? "template" : "custom"} onValueChange={(value) => 
              handleFieldChange("useTemplate", value === "template")
            }>
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="template">Use Template</TabsTrigger>
                <TabsTrigger value="custom">Custom Configuration</TabsTrigger>
              </TabsList>
              
              <TabsContent value="template" className="space-y-4">
                {templatesLoading && (
                  <div className="flex items-center justify-center py-8">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
                    <span className="ml-2">Loading templates...</span>
                  </div>
                )}
                
                {templatesError && (
                  <Alert variant="destructive">
                    <AlertTriangle className="h-4 w-4" />
                    <AlertTitle>Error Loading Templates</AlertTitle>
                    <AlertDescription>
                      {templatesError instanceof Error ? templatesError.message : "Failed to load strategy templates"}
                    </AlertDescription>
                  </Alert>
                )}
                
                {availableTemplates && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {availableTemplates.templates.slice(0, 6).map((template) => (
                      <Card
                        key={template.id}
                        className={cn(
                          "cursor-pointer transition-all hover:shadow-lg border-2",
                          formData.selectedTemplate?.id === template.id
                            ? "border-primary shadow-lg ring-2 ring-primary/20"
                            : "border-gray-200 hover:border-gray-300"
                        )}
                        onClick={() => handleFieldChange("selectedTemplate", template)}
                      >
                        <CardHeader className="pb-3">
                          <div className="flex items-start justify-between">
                            <div>
                              <CardTitle className="text-base">{template.displayName}</CardTitle>
                              <Badge 
                                variant={template.riskLevel === RiskLevel.CONSERVATIVE ? "default" : 
                                        template.riskLevel === RiskLevel.MODERATE ? "secondary" : "destructive"}
                                className="text-xs mt-1"
                              >
                                {template.riskLevel}
                              </Badge>
                            </div>
                            {template.isProduction && (
                              <Badge className="text-xs bg-green-100 text-green-800">
                                <Award className="h-3 w-3 mr-1" />
                                Verified
                              </Badge>
                            )}
                          </div>
                          
                          <CardDescription className="text-sm mt-2">
                            {template.description}
                          </CardDescription>
                          
                          <div className="grid grid-cols-2 gap-2 mt-3 text-xs">
                            <div className="text-center p-2 bg-gray-50 rounded">
                              <div className="font-semibold">${template.minimumCapital.toLocaleString()}</div>
                              <div className="text-muted-foreground">Min Capital</div>
                            </div>
                            <div className="text-center p-2 bg-gray-50 rounded">
                              <div className="font-semibold">{template.version}</div>
                              <div className="text-muted-foreground">Version</div>
                            </div>
                          </div>
                          
                          <div className="flex flex-wrap gap-1 mt-2">
                            {template.tags.slice(0, 3).map((tag, i) => (
                              <Badge key={i} variant="outline" className="text-xs">
                                {tag}
                              </Badge>
                            ))}
                          </div>
                        </CardHeader>
                      </Card>
                    ))}
                  </div>
                )}
                
                {/* Template Variants */}
                {formData.selectedTemplate && (
                  <Card className="border-primary/50 bg-primary/5">
                    <CardHeader>
                      <CardTitle className="text-base">Template Variant</CardTitle>
                      <CardDescription>Choose the risk profile for this template</CardDescription>
                      
                      <div className="grid grid-cols-3 gap-3 mt-4">
                        {['conservative', 'moderate', 'aggressive'].map((variant) => (
                          <Card
                            key={variant}
                            className={cn(
                              "cursor-pointer transition-all border-2 p-3",
                              formData.templateVariant === variant
                                ? "border-primary bg-primary/10"
                                : "border-gray-200 hover:border-gray-300"
                            )}
                            onClick={() => handleFieldChange("templateVariant", variant)}
                          >
                            <div className="text-center">
                              <div className="font-medium capitalize">{variant}</div>
                              <div className="text-xs text-muted-foreground mt-1">
                                {variant === 'conservative' ? 'Lower risk, steady returns' :
                                 variant === 'moderate' ? 'Balanced risk-reward' :
                                 'Higher risk, higher potential'}
                              </div>
                            </div>
                          </Card>
                        ))}
                      </div>
                    </CardHeader>
                  </Card>
                )}
              </TabsContent>
              
              <TabsContent value="custom" className="space-y-4">
                <Alert>
                  <Info className="h-4 w-4" />
                  <AlertTitle>Custom Configuration</AlertTitle>
                  <AlertDescription>
                    You'll be able to configure all strategy parameters manually in the next step.
                    This option is recommended for advanced users.
                  </AlertDescription>
                </Alert>
                
                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">Custom Strategy Setup</CardTitle>
                    <CardDescription>
                      Create a custom configuration for {selectedStrategyInfo?.label || 'your selected strategy'}
                    </CardDescription>
                  </CardHeader>
                </Card>
              </TabsContent>
            </Tabs>
          </motion.div>
        );

      case 3: // Dynamic Parameters Configuration
        return (
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="space-y-6"
          >
            <div>
              <h3 className="text-lg font-semibold mb-2">Strategy Parameters</h3>
              <p className="text-muted-foreground">
                Configure the parameters for your selected strategy
              </p>
            </div>

            {templateLoading && (
              <div className="flex items-center justify-center py-8">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
                <span className="ml-2">Loading parameter configuration...</span>
              </div>
            )}

            {templateLoadError && (
              <Alert variant="destructive">
                <AlertTriangle className="h-4 w-4" />
                <AlertTitle>Configuration Error</AlertTitle>
                <AlertDescription>
                  Failed to load strategy parameters. Please try selecting a different template.
                </AlertDescription>
              </Alert>
            )}

            {/* Template-based Parameters */}
            {formData.useTemplate && selectedTemplateData?.template && (
              <div className="space-y-6">
                {/* Group parameters by category */}
                {Object.entries(
                  selectedTemplateData.template.parameters.reduce((groups: Record<string, StrategyParameterDefinition[]>, param) => {
                    const group = param.group || 'General';
                    if (!groups[group]) groups[group] = [];
                    groups[group].push(param);
                    return groups;
                  }, {})
                ).map(([groupName, params]) => (
                  <Card key={groupName}>
                    <CardHeader className="pb-4">
                      <CardTitle className="text-base">{groupName} Parameters</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      {params.map((param) => (
                        <div key={param.name}>
                          <Label htmlFor={param.name} className="flex items-center gap-2">
                            {param.displayName}
                            {param.required && <span className="text-red-500">*</span>}
                            {param.tooltipText && (
                              <div className="group relative">
                                <Info className="h-3 w-3 text-muted-foreground cursor-help" />
                                <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-2 py-1 bg-black text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity duration-200 whitespace-nowrap z-10">
                                  {param.tooltipText}
                                </div>
                              </div>
                            )}
                          </Label>
                          
                          {param.type === 'number' && (
                            <Input
                              id={param.name}
                              type="number"
                              min={param.validation?.min}
                              max={param.validation?.max}
                              step={param.validation?.step || 'any'}
                              value={formData.strategyParameters[param.name] || param.defaultValue}
                              onChange={(e) => handleParameterChange(param.name, parseFloat(e.target.value) || 0)}
                              className={cn(
                                "mt-1",
                                formData.parameterValidation[param.name] && "border-red-500"
                              )}
                            />
                          )}
                          
                          {param.type === 'string' && (
                            <Input
                              id={param.name}
                              type="text"
                              value={formData.strategyParameters[param.name] || param.defaultValue}
                              onChange={(e) => handleParameterChange(param.name, e.target.value)}
                              className={cn(
                                "mt-1",
                                formData.parameterValidation[param.name] && "border-red-500"
                              )}
                            />
                          )}
                          
                          {param.type === 'boolean' && (
                            <div className="flex items-center space-x-2 mt-1">
                              <Switch
                                id={param.name}
                                checked={formData.strategyParameters[param.name] ?? param.defaultValue}
                                onCheckedChange={(value) => handleParameterChange(param.name, value)}
                              />
                              <Label htmlFor={param.name} className="text-sm text-muted-foreground">
                                {param.description}
                              </Label>
                            </div>
                          )}
                          
                          {(param.type === 'select' || param.type === 'multi_select') && param.options && (
                            <Select
                              value={formData.strategyParameters[param.name] || param.defaultValue}
                              onValueChange={(value) => handleParameterChange(param.name, value)}
                            >
                              <SelectTrigger className={cn(
                                "mt-1",
                                formData.parameterValidation[param.name] && "border-red-500"
                              )}>
                                <SelectValue />
                              </SelectTrigger>
                              <SelectContent>
                                {param.options.map((option) => (
                                  <SelectItem 
                                    key={option.value} 
                                    value={option.value}
                                    disabled={option.disabled}
                                  >
                                    {option.label}
                                    {option.description && (
                                      <span className="text-xs text-muted-foreground ml-2">
                                        - {option.description}
                                      </span>
                                    )}
                                  </SelectItem>
                                ))}
                              </SelectContent>
                            </Select>
                          )}
                          
                          {param.description && (
                            <p className="text-xs text-muted-foreground mt-1">{param.description}</p>
                          )}
                          
                          {formData.parameterValidation[param.name] && (
                            <p className="text-xs text-red-500 mt-1">
                              {formData.parameterValidation[param.name]}
                            </p>
                          )}
                        </div>
                      ))}
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}

            {/* Custom Parameters */}
            {!formData.useTemplate && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Custom Strategy Parameters</CardTitle>
                  <CardDescription>
                    Configure basic parameters for your custom {selectedStrategyInfo?.label || 'strategy'}
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {/* Basic custom parameters based on strategy type */}
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="customParam1">Entry Threshold</Label>
                      <Input
                        id="customParam1"
                        type="number"
                        step="0.1"
                        placeholder="1.5"
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label htmlFor="customParam2">Exit Threshold</Label>
                      <Input
                        id="customParam2"
                        type="number"
                        step="0.1"
                        placeholder="2.0"
                        className="mt-1"
                      />
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </motion.div>
        );

      case 4: // Markets
        return (
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="space-y-6"
          >
            <div>
              <h3 className="text-lg font-semibold mb-2">Select Markets</h3>
              <p className="text-muted-foreground">
                Choose exchanges and trading pairs for your strategy
              </p>
            </div>

            <div>
              <Label className="text-sm font-medium">Exchanges</Label>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mt-2">
                {EXCHANGES.map((exchange) => (
                  <Card
                    key={exchange.value}
                    className={cn(
                      "cursor-pointer transition-all hover:shadow-md border-2 p-4",
                      formData.exchanges.includes(exchange.value)
                        ? "border-primary bg-primary/5"
                        : "border-gray-200",
                    )}
                    onClick={() => {
                      const newExchanges = formData.exchanges.includes(
                        exchange.value,
                      )
                        ? formData.exchanges.filter((e) => e !== exchange.value)
                        : [...formData.exchanges, exchange.value];
                      handleFieldChange("exchanges", newExchanges);
                    }}
                  >
                    <div className="flex items-center space-x-3">
                      <span className="text-lg">{exchange.icon}</span>
                      <div>
                        <p className="font-medium text-sm">{exchange.label}</p>
                        <p className="text-xs text-muted-foreground">
                          Fees: {exchange.fees}
                        </p>
                      </div>
                      {formData.exchanges.includes(exchange.value) && (
                        <CheckCircle className="h-4 w-4 text-primary ml-auto" />
                      )}
                    </div>
                  </Card>
                ))}
              </div>
            </div>

            <div>
              <Label className="text-sm font-medium">Trading Pairs</Label>
              <div className="flex flex-wrap gap-2 mt-2">
                {SYMBOLS.map((symbol) => (
                  <Badge
                    key={symbol}
                    variant={
                      formData.symbols.includes(symbol) ? "default" : "outline"
                    }
                    className="cursor-pointer px-3 py-1.5 text-xs"
                    onClick={() => {
                      const newSymbols = formData.symbols.includes(symbol)
                        ? formData.symbols.filter((s) => s !== symbol)
                        : [...formData.symbols, symbol];
                      handleFieldChange("symbols", newSymbols);
                    }}
                  >
                    {symbol}
                    {formData.symbols.includes(symbol) && (
                      <CheckCircle className="ml-1 h-3 w-3" />
                    )}
                  </Badge>
                ))}
              </div>
            </div>

            <Alert>
              <Target className="h-4 w-4" />
              <AlertTitle>Selection Summary</AlertTitle>
              <AlertDescription>
                Selected {formData.exchanges.length} exchange(s) and{" "}
                {formData.symbols.length} trading pair(s)
              </AlertDescription>
            </Alert>
          </motion.div>
        );

      case 5: // Enhanced Risk & Capital Configuration
        return (
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="space-y-6"
          >
            <div>
              <h3 className="text-lg font-semibold mb-2">Risk Management & Capital Allocation</h3>
              <p className="text-muted-foreground">
                Configure comprehensive risk management and position sizing
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <Label htmlFor="capital">Allocated Capital ($)</Label>
                <Input
                  id="capital"
                  type="number"
                  min="100"
                  value={formData.capital}
                  onChange={(e) =>
                    handleFieldChange(
                      "capital",
                      parseFloat(e.target.value) || 0,
                    )
                  }
                  className="mt-1"
                />
              </div>

              <div>
                <Label htmlFor="maxTrades">Max Daily Trades</Label>
                <Input
                  id="maxTrades"
                  type="number"
                  min="1"
                  max="1000"
                  value={formData.maxTrades}
                  onChange={(e) =>
                    handleFieldChange(
                      "maxTrades",
                      parseInt(e.target.value) || 0,
                    )
                  }
                  className="mt-1"
                />
              </div>
            </div>

            {/* Enhanced Risk Level Selection */}
            <div>
              <Label className="text-sm font-medium">Risk Profile</Label>
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mt-2">
                {ENHANCED_RISK_LEVELS.map((level) => (
                  <Card
                    key={level.value}
                    className={cn(
                      "cursor-pointer transition-all hover:shadow-md border-2 p-4",
                      formData.riskLevel === level.value
                        ? "border-primary bg-primary/5"
                        : "border-gray-200",
                    )}
                    onClick={() => handleFieldChange("riskLevel", level.value)}
                  >
                    <div className="text-center">
                      <div
                        className={cn(
                          "px-2 py-1 rounded-full text-xs font-medium mb-2",
                          level.color,
                        )}
                      >
                        {level.percent}
                      </div>
                      <p className="text-sm font-medium">{level.label}</p>
                      <p className="text-xs text-muted-foreground mt-1">
                        {level.description}
                      </p>
                      <div className="mt-2">
                        <div className="flex flex-wrap gap-1">
                          {level.features.map((feature, i) => (
                            <Badge key={i} variant="outline" className="text-xs">
                              {feature}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    </div>
                  </Card>
                ))}
              </div>
            </div>

            {/* Position Sizing Configuration */}
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Position Sizing</CardTitle>
                <CardDescription>Configure how position sizes are calculated</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <Label>Position Sizing Method</Label>
                  <Select
                    value={formData.positionSizeMethod}
                    onValueChange={(value) => handleFieldChange("positionSizeMethod", value)}
                  >
                    <SelectTrigger className="mt-1">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {POSITION_SIZE_METHODS.map((method) => (
                        <SelectItem key={method.value} value={method.value}>
                          <div className="flex items-center gap-2">
                            {method.icon}
                            <div>
                              <div className="font-medium">{method.label}</div>
                              <div className="text-xs text-muted-foreground">
                                {method.description}
                              </div>
                            </div>
                          </div>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label htmlFor="positionSizeValue">
                      {formData.positionSizeMethod === PositionSizeMethod.FIXED ? 
                        'Fixed Amount ($)' :
                        formData.positionSizeMethod === PositionSizeMethod.PERCENTAGE ?
                        'Percentage of Capital (%)' :
                        'Size Parameter'
                      }
                    </Label>
                    <Input
                      id="positionSizeValue"
                      type="number"
                      min={formData.positionSizeMethod === PositionSizeMethod.FIXED ? "100" : "0.1"}
                      max={formData.positionSizeMethod === PositionSizeMethod.PERCENTAGE ? "20" : undefined}
                      step={formData.positionSizeMethod === PositionSizeMethod.FIXED ? "100" : "0.1"}
                      value={formData.positionSizeValue}
                      onChange={(e) => handleFieldChange("positionSizeValue", parseFloat(e.target.value) || 0)}
                      className="mt-1"
                    />
                  </div>

                  <div>
                    <Label htmlFor="maxPositions">Maximum Positions</Label>
                    <Input
                      id="maxPositions"
                      type="number"
                      min="1"
                      max="20"
                      value={formData.maxPositions}
                      onChange={(e) => handleFieldChange("maxPositions", parseInt(e.target.value) || 1)}
                      className="mt-1"
                    />
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Circuit Breakers */}
            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <AlertTriangle className="h-4 w-4" />
                  Circuit Breakers
                </CardTitle>
                <CardDescription>Automatic risk protection mechanisms</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center space-x-2">
                  <Switch
                    id="circuitBreaker"
                    checked={formData.enableCircuitBreaker}
                    onCheckedChange={(value) => handleFieldChange("enableCircuitBreaker", value)}
                  />
                  <Label htmlFor="circuitBreaker">Enable Circuit Breakers</Label>
                </div>

                {formData.enableCircuitBreaker && (
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="dailyLossLimit">Daily Loss Limit (%)</Label>
                      <Input
                        id="dailyLossLimit"
                        type="number"
                        min="1"
                        max="20"
                        step="0.5"
                        value={formData.dailyLossLimit}
                        onChange={(e) => handleFieldChange("dailyLossLimit", parseFloat(e.target.value) || 0)}
                        className="mt-1"
                      />
                    </div>

                    <div>
                      <Label htmlFor="consecutiveLossLimit">Max Consecutive Losses</Label>
                      <Input
                        id="consecutiveLossLimit"
                        type="number"
                        min="2"
                        max="10"
                        value={formData.consecutiveLossLimit}
                        onChange={(e) => handleFieldChange("consecutiveLossLimit", parseInt(e.target.value) || 0)}
                        className="mt-1"
                      />
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Risk Metrics */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="maxDrawdown">Maximum Drawdown (%)</Label>
                <Input
                  id="maxDrawdown"
                  type="number"
                  min="1"
                  max="50"
                  step="0.5"
                  value={formData.maxDrawdown}
                  onChange={(e) => handleFieldChange("maxDrawdown", parseFloat(e.target.value) || 0)}
                  className="mt-1"
                />
              </div>

              <div>
                <Label htmlFor="correlationLimit">Correlation Limit</Label>
                <Input
                  id="correlationLimit"
                  type="number"
                  min="0.1"
                  max="1"
                  step="0.1"
                  value={formData.correlationLimit}
                  onChange={(e) => handleFieldChange("correlationLimit", parseFloat(e.target.value) || 0)}
                  className="mt-1"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="stopLoss">Stop Loss (%)</Label>
                <Input
                  id="stopLoss"
                  type="number"
                  min="0.1"
                  max="20"
                  step="0.1"
                  value={formData.stopLoss}
                  onChange={(e) =>
                    handleFieldChange(
                      "stopLoss",
                      parseFloat(e.target.value) || 0,
                    )
                  }
                  className="mt-1"
                />
              </div>

              <div>
                <Label htmlFor="takeProfit">Take Profit (%)</Label>
                <Input
                  id="takeProfit"
                  type="number"
                  min="0.1"
                  max="50"
                  step="0.1"
                  value={formData.takeProfit}
                  onChange={(e) =>
                    handleFieldChange(
                      "takeProfit",
                      parseFloat(e.target.value) || 0,
                    )
                  }
                  className="mt-1"
                />
              </div>
            </div>
          </motion.div>
        );

      case 6: // Enhanced Review
        return (
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="space-y-6"
          >
            <div>
              <h3 className="text-lg font-semibold mb-2">Review & Launch</h3>
              <p className="text-muted-foreground">
                Review your bot configuration before launching
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm">Basic Configuration</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Bot Name:</span>
                    <span className="font-medium">
                      {formData.botName || "Untitled Bot"}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Type:</span>
                    <span className="font-medium">
                      {
                        BOT_TYPES.find((t) => t.value === formData.botType)
                          ?.label
                      }
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Strategy:</span>
                    <span className="font-medium">
                      {selectedStrategyInfo?.label || 'Not selected'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Template:</span>
                    <span className="font-medium">
                      {formData.useTemplate && formData.selectedTemplate ? 
                        `${formData.selectedTemplate.displayName} (${formData.templateVariant})` : 
                        'Custom Configuration'
                      }
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Timeframe:</span>
                    <span className="font-medium">{formData.timeframe}</span>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm">Risk Management</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Capital:</span>
                    <span className="font-medium">
                      ${formData.capital.toLocaleString()}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Risk Profile:</span>
                    <span className="font-medium capitalize">
                      {formData.riskLevel}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Position Sizing:</span>
                    <span className="font-medium">
                      {POSITION_SIZE_METHODS.find(m => m.value === formData.positionSizeMethod)?.label || 'Not set'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Max Positions:</span>
                    <span className="font-medium">{formData.maxPositions}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Stop Loss:</span>
                    <span className="font-medium">{formData.stopLoss}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Take Profit:</span>
                    <span className="font-medium">{formData.takeProfit}%</span>
                  </div>
                </CardContent>
              </Card>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm">Markets</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Exchanges:</span>
                    <span className="font-medium">
                      {formData.exchanges.join(", ") || "None selected"}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Trading Pairs:</span>
                    <span className="font-medium">
                      {formData.symbols.length} pairs selected
                    </span>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm">Advanced Settings</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Circuit Breakers:</span>
                    <span className="font-medium">
                      {formData.enableCircuitBreaker ? 'Enabled' : 'Disabled'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Max Trades:</span>
                    <span className="font-medium">{formData.maxTrades}/day</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Fill Strategy:</span>
                    <span className="font-medium capitalize">{formData.fillStrategy}</span>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Strategy Parameter Summary */}
            {formData.useTemplate && formData.selectedTemplate && Object.keys(formData.strategyParameters).length > 0 && (
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm">Strategy Parameters</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    {Object.entries(formData.strategyParameters).slice(0, 6).map(([key, value]) => (
                      <div key={key} className="flex justify-between">
                        <span className="text-muted-foreground">{key.replace(/_/g, ' ')}:</span>
                        <span className="font-medium">
                          {typeof value === 'boolean' ? (value ? 'Yes' : 'No') : 
                           typeof value === 'number' ? value.toString() : 
                           value?.toString() || 'Not set'}
                        </span>
                      </div>
                    ))}
                    {Object.keys(formData.strategyParameters).length > 6 && (
                      <div className="col-span-2 text-center text-muted-foreground">
                        +{Object.keys(formData.strategyParameters).length - 6} more parameters
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            )}

            <div className="flex items-center space-x-2">
              <Switch
                id="autoStart"
                checked={formData.autoStart}
                onCheckedChange={(value) =>
                  handleFieldChange("autoStart", value)
                }
              />
              <Label htmlFor="autoStart" className="text-sm">
                Auto-start bot after creation
              </Label>
            </div>

            <div className="flex items-center space-x-2">
              <Label htmlFor="priority" className="text-sm">
                Bot Priority:
              </Label>
              <Select
                value={formData.priority}
                onValueChange={(value) => handleFieldChange("priority", value)}
              >
                <SelectTrigger className="w-48">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value={BotPriority.LOW}>
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full bg-gray-400"></div>
                      Low Priority
                    </div>
                  </SelectItem>
                  <SelectItem value={BotPriority.NORMAL}>
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full bg-blue-400"></div>
                      Normal Priority
                    </div>
                  </SelectItem>
                  <SelectItem value={BotPriority.HIGH}>
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full bg-orange-400"></div>
                      High Priority
                    </div>
                  </SelectItem>
                  <SelectItem value={BotPriority.CRITICAL}>
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full bg-red-400"></div>
                      Critical Priority
                    </div>
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>
          </motion.div>
        );

      default:
        return null;
    }
  };

  if (!open) return null;

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center space-x-2">
            <Bot className="h-5 w-5" />
            <span>Create Trading Bot</span>
          </DialogTitle>
          <DialogDescription>
            Set up your automated trading bot with professional-grade controls
          </DialogDescription>
        </DialogHeader>

        {/* Progress Bar */}
        <div className="py-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium">
              Step {currentStep + 1} of {steps.length} - {steps[currentStep].title}
            </span>
            <span className="text-sm text-muted-foreground">
              {steps[currentStep].description}
            </span>
          </div>
          <Progress
            value={((currentStep + 1) / steps.length) * 100}
            className="h-2"
          />
        </div>

        {/* Steps Navigation */}
        <div className="flex justify-between mb-6">
          {steps.map((step, index) => {
            const Icon = step.icon;
            return (
              <div
                key={step.title}
                className={cn(
                  "flex flex-col items-center space-y-1 flex-1",
                  index <= currentStep
                    ? "text-primary"
                    : "text-muted-foreground",
                )}
              >
                <div
                  className={cn(
                    "w-8 h-8 rounded-full flex items-center justify-center border-2",
                    index < currentStep
                      ? "bg-primary border-primary text-white"
                      : index === currentStep
                        ? "border-primary text-primary"
                        : "border-gray-300",
                  )}
                >
                  {index < currentStep ? (
                    <CheckCircle className="h-4 w-4" />
                  ) : (
                    <Icon className="h-4 w-4" />
                  )}
                </div>
                <span className="text-xs font-medium hidden sm:block">
                  {step.title}
                </span>
              </div>
            );
          })}
        </div>

        {/* Error Message */}
        {(error || reduxError) && (
          <Alert variant="destructive" className="mb-4">
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{error || reduxError}</AlertDescription>
          </Alert>
        )}

        {/* Step Content */}
        <div className="min-h-[400px]">
          <AnimatePresence mode="wait">{renderStep()}</AnimatePresence>
        </div>

        {/* Footer Actions */}
        <DialogFooter className="flex justify-between">
          <Button
            variant="outline"
            onClick={handleBack}
            disabled={currentStep === 0 || isLoading}
          >
            <ChevronLeft className="mr-1 h-4 w-4" />
            Back
          </Button>

          <div className="flex space-x-2">
            <Button variant="ghost" onClick={onClose} disabled={isLoading}>
              Cancel
            </Button>

            {currentStep < steps.length - 1 ? (
              <Button onClick={handleNext} disabled={isLoading}>
                Next
                <ChevronRight className="ml-1 h-4 w-4" />
              </Button>
            ) : (
              <Button
                onClick={handleSubmit}
                disabled={isLoading}
                className="bg-gradient-to-r from-blue-600 to-blue-700"
              >
                {isLoading ? (
                  <>Creating Bot...</>
                ) : (
                  <>
                    <Rocket className="mr-1 h-4 w-4" />
                    {formData.autoStart ? "Launch Bot" : "Create Bot"}
                  </>
                )}
              </Button>
            )}
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default BotCreationWizard;
