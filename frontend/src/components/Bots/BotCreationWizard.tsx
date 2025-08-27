/**
 * Modern Bot Creation Wizard Component using Shadcn/ui
 * Multi-step wizard with form validation and trading-specific UI
 */

import React, { useState, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

// Shadcn/ui components
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { Switch } from '@/components/ui/switch';
import { Separator } from '@/components/ui/separator';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

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
} from 'lucide-react';

// Utils
import { cn } from '@/lib/utils';

// Types
import { 
  StrategyType, 
  StrategyCategory, 
  RiskLevel, 
  BotType,
  BotPriority,
  CreateBotRequest,
  StrategyTemplate 
} from '@/types';

// API Services
import { strategyAPI } from '@/services/api/strategyAPI';
import { botAPI } from '@/services/api/botAPI';

interface BotCreationWizardProps {
  open: boolean;
  onClose: () => void;
  onSuccess?: (botId: string) => void;
}

// Bot types with modern styling
const BOT_TYPES = [
  {
    value: 'strategy',
    label: 'Strategy Bot',
    description: 'Single strategy focused trading',
    icon: <TrendingUp className="h-5 w-5" />,
    color: 'bg-green-500',
    recommended: true,
  },
  {
    value: 'arbitrage',
    label: 'Arbitrage Bot',
    description: 'Cross-exchange opportunities',
    icon: <ArrowLeftRight className="h-5 w-5" />,
    color: 'bg-blue-500',
    recommended: false,
  },
  {
    value: 'market_maker',
    label: 'Market Maker',
    description: 'Provide liquidity to markets',
    icon: <BarChart3 className="h-5 w-5" />,
    color: 'bg-orange-500',
    recommended: false,
  },
];

// Strategy options using the new StrategyType enum
const STRATEGIES = [
  { 
    value: StrategyType.MOMENTUM, 
    label: 'Momentum Trading', 
    risk: RiskLevel.MODERATE, 
    category: StrategyCategory.DYNAMIC,
    description: 'Trend-following strategy that capitalizes on market momentum',
    icon: <TrendingUp className="h-4 w-4" />
  },
  { 
    value: StrategyType.MEAN_REVERSION, 
    label: 'Mean Reversion', 
    risk: RiskLevel.CONSERVATIVE, 
    category: StrategyCategory.STATIC,
    description: 'Counter-trend strategy exploiting price reversion to mean',
    icon: <BarChart3 className="h-4 w-4" />
  },
  { 
    value: StrategyType.ARBITRAGE, 
    label: 'Arbitrage', 
    risk: RiskLevel.CONSERVATIVE, 
    category: StrategyCategory.STATIC,
    description: 'Risk-free profit from price differences',
    icon: <ArrowLeftRight className="h-4 w-4" />
  },
  { 
    value: StrategyType.STATISTICAL_ARBITRAGE, 
    label: 'Statistical Arbitrage', 
    risk: RiskLevel.MODERATE, 
    category: StrategyCategory.STATIC,
    description: 'Statistical price relationship exploitation',
    icon: <Target className="h-4 w-4" />
  },
  { 
    value: StrategyType.TREND_FOLLOWING, 
    label: 'Trend Following', 
    risk: RiskLevel.MODERATE, 
    category: StrategyCategory.STATIC,
    description: 'Long-term trend capture strategy',
    icon: <TrendingUp className="h-4 w-4" />
  },
  { 
    value: StrategyType.MARKET_MAKING, 
    label: 'Market Making', 
    risk: RiskLevel.CONSERVATIVE, 
    category: StrategyCategory.STATIC,
    description: 'Provide liquidity and earn spreads',
    icon: <DollarSign className="h-4 w-4" />
  },
  { 
    value: StrategyType.PAIRS_TRADING, 
    label: 'Pairs Trading', 
    risk: RiskLevel.MODERATE, 
    category: StrategyCategory.STATIC,
    description: 'Trade correlated asset pairs',
    icon: <ArrowLeftRight className="h-4 w-4" />
  },
  { 
    value: StrategyType.VOLATILITY_BREAKOUT, 
    label: 'Volatility Breakout', 
    risk: RiskLevel.AGGRESSIVE, 
    category: StrategyCategory.DYNAMIC,
    description: 'Capitalize on high volatility movements',
    icon: <TrendingUp className="h-4 w-4" />
  },
  { 
    value: StrategyType.BREAKOUT, 
    label: 'Breakout', 
    risk: RiskLevel.AGGRESSIVE, 
    category: StrategyCategory.STATIC,
    description: 'Trade price breakouts from ranges',
    icon: <TrendingUp className="h-4 w-4" />
  },
  { 
    value: StrategyType.CROSS_EXCHANGE_ARBITRAGE, 
    label: 'Cross-Exchange Arbitrage', 
    risk: RiskLevel.CONSERVATIVE, 
    category: StrategyCategory.STATIC,
    description: 'Arbitrage between exchanges',
    icon: <ArrowLeftRight className="h-4 w-4" />
  },
  { 
    value: StrategyType.TRIANGULAR_ARBITRAGE, 
    label: 'Triangular Arbitrage', 
    risk: RiskLevel.CONSERVATIVE, 
    category: StrategyCategory.STATIC,
    description: 'Three-way currency arbitrage',
    icon: <Target className="h-4 w-4" />
  },
  { 
    value: StrategyType.ENSEMBLE, 
    label: 'Ensemble Strategy', 
    risk: RiskLevel.MODERATE, 
    category: StrategyCategory.HYBRID,
    description: 'Combine multiple strategies',
    icon: <Settings className="h-4 w-4" />
  },
  { 
    value: StrategyType.FALLBACK, 
    label: 'Fallback Strategy', 
    risk: RiskLevel.CONSERVATIVE, 
    category: StrategyCategory.HYBRID,
    description: 'Fallback when primary fails',
    icon: <Shield className="h-4 w-4" />
  },
  { 
    value: StrategyType.RULE_BASED_AI, 
    label: 'Rule-Based AI', 
    risk: RiskLevel.EXPERIMENTAL, 
    category: StrategyCategory.HYBRID,
    description: 'AI-driven rule-based trading',
    icon: <Bot className="h-4 w-4" />
  },
  { 
    value: StrategyType.CUSTOM, 
    label: 'Custom Strategy', 
    risk: RiskLevel.EXPERIMENTAL, 
    category: StrategyCategory.CUSTOM,
    description: 'Build your own custom strategy',
    icon: <Settings className="h-4 w-4" />
  }
];

const EXCHANGES = [
  { value: 'binance', label: 'Binance', icon: '🟡', fees: '0.1%' },
  { value: 'coinbase', label: 'Coinbase', icon: '🔵', fees: '0.5%' },
  { value: 'okx', label: 'OKX', icon: '⚪', fees: '0.1%' },
];

const SYMBOLS = [
  'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT',
  'ADA/USDT', 'DOT/USDT', 'MATIC/USDT', 'LINK/USDT'
];

// Risk levels using the new RiskLevel enum
const RISK_LEVELS = [
  { 
    value: RiskLevel.CONSERVATIVE, 
    label: 'Conservative', 
    percent: '1%', 
    numericValue: 1,
    color: 'text-green-600 bg-green-100',
    description: 'Low risk, steady returns'
  },
  { 
    value: RiskLevel.MODERATE, 
    label: 'Moderate', 
    percent: '2-3%', 
    numericValue: 2.5,
    color: 'text-yellow-600 bg-yellow-100',
    description: 'Balanced risk-return profile'
  },
  { 
    value: RiskLevel.AGGRESSIVE, 
    label: 'Aggressive', 
    percent: '5%+', 
    numericValue: 5,
    color: 'text-red-600 bg-red-100',
    description: 'High risk, high potential returns'
  },
  { 
    value: RiskLevel.EXPERIMENTAL, 
    label: 'Experimental', 
    percent: 'Variable', 
    numericValue: 10,
    color: 'text-purple-600 bg-purple-100',
    description: 'Testing new strategies'
  },
];

const BotCreationWizard: React.FC<BotCreationWizardProps> = ({
  open,
  onClose,
  onSuccess,
}) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Strategy template state
  const [strategyTemplates, setStrategyTemplates] = useState<StrategyTemplate[]>([]);
  const [loadingTemplates, setLoadingTemplates] = useState(false);
  const [selectedTemplate, setSelectedTemplate] = useState<StrategyTemplate | null>(null);

  // Form data using new type system
  const [formData, setFormData] = useState({
    // Step 1: Basic Info
    botName: '',
    botType: BotType.TRADING,
    description: '',
    
    // Step 2: Strategy
    strategy: '' as StrategyType | '',
    strategyCategory: '' as StrategyCategory | '',
    timeframe: '1h',
    
    // Step 3: Markets
    exchanges: [] as string[],
    symbols: [] as string[],
    
    // Step 4: Risk & Capital
    capital: 1000,
    riskLevel: RiskLevel.MODERATE,
    stopLoss: 2,
    takeProfit: 5,
    maxRiskPerTrade: 2,
    
    // Step 5: Advanced
    autoStart: false,
    tradingMode: 'paper',
    maxTrades: 50,
    priority: BotPriority.NORMAL,
  });

  const steps = [
    { title: 'Bot Type', icon: Bot, description: 'Choose your bot type' },
    { title: 'Strategy', icon: TrendingUp, description: 'Select trading strategy' },
    { title: 'Markets', icon: Target, description: 'Choose exchanges & pairs' },
    { title: 'Risk Setup', icon: Shield, description: 'Configure risk parameters' },
    { title: 'Review', icon: CheckCircle, description: 'Review and launch' },
  ];

  // Load strategy templates when wizard opens
  useEffect(() => {
    if (open && strategyTemplates.length === 0) {
      loadStrategyTemplates();
    }
  }, [open]);

  const loadStrategyTemplates = async () => {
    setLoadingTemplates(true);
    try {
      const response = await strategyAPI.getTemplates({
        limit: 50,
        sortBy: 'risk_level',
        sortOrder: 'asc'
      });
      setStrategyTemplates(response.templates);
    } catch (error) {
      console.error('Failed to load strategy templates:', error);
      setError('Failed to load strategy templates');
    } finally {
      setLoadingTemplates(false);
    }
  };

  const handleNext = useCallback(() => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(prev => prev + 1);
    }
  }, [currentStep, steps.length]);

  const handleBack = useCallback(() => {
    if (currentStep > 0) {
      setCurrentStep(prev => prev - 1);
    }
  }, [currentStep]);

  const handleFieldChange = useCallback((field: string, value: any) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  }, []);

  const handleStrategyChange = useCallback((strategyType: StrategyType) => {
    // Find the strategy in our static list to get category
    const strategy = STRATEGIES.find(s => s.value === strategyType);
    const template = strategyTemplates.find(t => t.strategyType === strategyType);
    
    setFormData(prev => ({ 
      ...prev, 
      strategy: strategyType,
      strategyCategory: strategy?.category || StrategyCategory.CUSTOM,
      riskLevel: strategy?.risk || RiskLevel.MODERATE
    }));
    
    setSelectedTemplate(template || null);
  }, [strategyTemplates]);

  const handleSubmit = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Validate form data
      if (!formData.botName.trim()) {
        throw new Error('Bot name is required');
      }
      if (!formData.strategy) {
        throw new Error('Strategy selection is required');
      }
      if (formData.exchanges.length === 0) {
        throw new Error('At least one exchange must be selected');
      }
      if (formData.symbols.length === 0) {
        throw new Error('At least one trading pair must be selected');
      }

      const createBotRequest: CreateBotRequest = {
        bot_name: formData.botName,
        bot_type: formData.botType,
        strategy_name: formData.strategy as StrategyType,
        exchanges: formData.exchanges,
        symbols: formData.symbols,
        allocated_capital: formData.capital,
        risk_percentage: formData.maxRiskPerTrade / 100,
        priority: formData.priority,
        auto_start: formData.autoStart,
        configuration: {
          timeframe: formData.timeframe,
          max_position_size: formData.capital * (formData.maxRiskPerTrade / 100),
          max_daily_trades: formData.maxTrades,
          max_concurrent_positions: 5,
          stop_loss_percentage: formData.stopLoss / 100,
          take_profit_percentage: formData.takeProfit / 100,
          trailing_stop: false,
          trading_mode: formData.tradingMode,
          description: formData.description,
          // Add strategy-specific parameters from template
          strategy_parameters: selectedTemplate ? 
            selectedTemplate.parameters.reduce((params, param) => ({
              ...params,
              [param.name]: param.defaultValue
            }), {}) : {}
        },
      };
      
      const response = await botAPI.createBot({
        bot_name: createBotRequest.bot_name,
        strategy_name: createBotRequest.strategy_name,
        exchange: createBotRequest.exchanges[0], // Take first exchange
        config: {
          bot_id: '', // Will be generated
          bot_name: createBotRequest.bot_name,
          bot_type: createBotRequest.bot_type,
          strategy_name: createBotRequest.strategy_name,
          exchanges: createBotRequest.exchanges,
          symbols: createBotRequest.symbols,
          allocated_capital: createBotRequest.allocated_capital,
          risk_percentage: createBotRequest.risk_percentage,
          priority: createBotRequest.priority || BotPriority.NORMAL,
          auto_start: createBotRequest.auto_start || false,
          strategy_config: createBotRequest.configuration || {},
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString()
        }
      });
      
      if (onSuccess) {
        onSuccess(response.data.bot_id);
      }
      onClose();
    } catch (err: any) {
      setError(err.message || 'Failed to create bot');
    } finally {
      setLoading(false);
    }
  }, [formData, selectedTemplate, onSuccess, onClose]);

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
              <p className="text-muted-foreground">Select the type of trading bot you want to create</p>
            </div>
            
            <div className="space-y-4">
              <div>
                <Label htmlFor="botName">Bot Name</Label>
                <Input
                  id="botName"
                  placeholder="My Trading Bot"
                  value={formData.botName}
                  onChange={(e) => handleFieldChange('botName', e.target.value)}
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
                        : "border-gray-200 hover:border-gray-300"
                    )}
                    onClick={() => handleFieldChange('botType', type.value)}
                  >
                    <CardHeader className="pb-3">
                      <div className="flex items-center justify-between">
                        <div className={cn("p-2 rounded-lg text-white", type.color)}>
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
                  onChange={(e) => handleFieldChange('description', e.target.value)}
                  className="mt-1"
                  rows={3}
                />
              </div>
            </div>
          </motion.div>
        );

      case 1: // Strategy
        return (
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="space-y-6"
          >
            <div>
              <h3 className="text-lg font-semibold mb-2">Select Strategy</h3>
              <p className="text-muted-foreground">Choose the trading strategy for your bot</p>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {STRATEGIES.map((strategy) => (
                <Card 
                  key={strategy.value}
                  className={cn(
                    "cursor-pointer transition-all hover:shadow-md border-2",
                    formData.strategy === strategy.value 
                      ? "border-primary shadow-md" 
                      : "border-gray-200"
                  )}
                  onClick={() => handleFieldChange('strategy', strategy.value)}
                >
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-sm font-semibold">
                        {strategy.label}
                      </CardTitle>
                      <Badge 
                        variant={
                          strategy.risk === RiskLevel.CONSERVATIVE ? 'default' :
                          strategy.risk === RiskLevel.MODERATE ? 'secondary' : 'destructive'
                        }
                        className="text-xs"
                      >
                        {strategy.risk} risk
                      </Badge>
                    </div>
                    <CardDescription className="text-xs">
                      {strategy.description}
                    </CardDescription>
                  </CardHeader>
                </Card>
              ))}
            </div>
            
            <div>
              <Label htmlFor="timeframe">Trading Timeframe</Label>
              <Select value={formData.timeframe} onValueChange={(value) => handleFieldChange('timeframe', value)}>
                <SelectTrigger className="mt-1">
                  <SelectValue placeholder="Select timeframe" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="1m">1 Minute</SelectItem>
                  <SelectItem value="5m">5 Minutes</SelectItem>
                  <SelectItem value="15m">15 Minutes</SelectItem>
                  <SelectItem value="30m">30 Minutes</SelectItem>
                  <SelectItem value="1h">1 Hour</SelectItem>
                  <SelectItem value="4h">4 Hours</SelectItem>
                  <SelectItem value="1d">1 Day</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </motion.div>
        );

      case 2: // Markets
        return (
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="space-y-6"
          >
            <div>
              <h3 className="text-lg font-semibold mb-2">Select Markets</h3>
              <p className="text-muted-foreground">Choose exchanges and trading pairs</p>
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
                        : "border-gray-200"
                    )}
                    onClick={() => {
                      const newExchanges = formData.exchanges.includes(exchange.value)
                        ? formData.exchanges.filter(e => e !== exchange.value)
                        : [...formData.exchanges, exchange.value];
                      handleFieldChange('exchanges', newExchanges);
                    }}
                  >
                    <div className="flex items-center space-x-3">
                      <span className="text-lg">{exchange.icon}</span>
                      <div>
                        <p className="font-medium text-sm">{exchange.label}</p>
                        <p className="text-xs text-muted-foreground">Fees: {exchange.fees}</p>
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
                    variant={formData.symbols.includes(symbol) ? "default" : "outline"}
                    className="cursor-pointer px-3 py-1.5 text-xs"
                    onClick={() => {
                      const newSymbols = formData.symbols.includes(symbol)
                        ? formData.symbols.filter(s => s !== symbol)
                        : [...formData.symbols, symbol];
                      handleFieldChange('symbols', newSymbols);
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
                Selected {formData.exchanges.length} exchange(s) and {formData.symbols.length} trading pair(s)
              </AlertDescription>
            </Alert>
          </motion.div>
        );

      case 3: // Risk & Capital
        return (
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="space-y-6"
          >
            <div>
              <h3 className="text-lg font-semibold mb-2">Risk Management</h3>
              <p className="text-muted-foreground">Configure capital allocation and risk parameters</p>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <Label htmlFor="capital">Allocated Capital ($)</Label>
                <Input
                  id="capital"
                  type="number"
                  min="100"
                  value={formData.capital}
                  onChange={(e) => handleFieldChange('capital', parseFloat(e.target.value) || 0)}
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
                  onChange={(e) => handleFieldChange('maxTrades', parseInt(e.target.value) || 0)}
                  className="mt-1"
                />
              </div>
            </div>
            
            <div>
              <Label className="text-sm font-medium">Risk Level</Label>
              <div className="grid grid-cols-3 gap-3 mt-2">
                {RISK_LEVELS.map((level) => (
                  <Card 
                    key={level.value}
                    className={cn(
                      "cursor-pointer transition-all hover:shadow-md border-2 p-4",
                      formData.riskLevel === level.value
                        ? "border-primary bg-primary/5" 
                        : "border-gray-200"
                    )}
                    onClick={() => handleFieldChange('riskLevel', level.value)}
                  >
                    <div className="text-center">
                      <div className={cn("px-2 py-1 rounded-full text-xs font-medium mb-2", level.color)}>
                        {level.percent}
                      </div>
                      <p className="text-sm font-medium">{level.label}</p>
                      <p className="text-xs text-muted-foreground mt-1">
                        ${(formData.capital * level.numericValue / 100).toFixed(2)} per trade
                      </p>
                    </div>
                  </Card>
                ))}
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
                  onChange={(e) => handleFieldChange('stopLoss', parseFloat(e.target.value) || 0)}
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
                  onChange={(e) => handleFieldChange('takeProfit', parseFloat(e.target.value) || 0)}
                  className="mt-1"
                />
              </div>
            </div>
          </motion.div>
        );

      case 4: // Review
        return (
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="space-y-6"
          >
            <div>
              <h3 className="text-lg font-semibold mb-2">Review & Launch</h3>
              <p className="text-muted-foreground">Review your bot configuration before launching</p>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm">Basic Configuration</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Bot Name:</span>
                    <span className="font-medium">{formData.botName || 'Untitled Bot'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Type:</span>
                    <span className="font-medium">{BOT_TYPES.find(t => t.value === formData.botType)?.label}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Strategy:</span>
                    <span className="font-medium">{STRATEGIES.find(s => s.value === formData.strategy)?.label}</span>
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
                    <span className="font-medium">${formData.capital.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Risk Level:</span>
                    <span className="font-medium">
                      {RISK_LEVELS.find(r => r.value === formData.riskLevel)?.percent || formData.riskLevel}
                    </span>
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
            
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm">Markets</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Exchanges:</span>
                  <span className="font-medium">{formData.exchanges.join(', ') || 'None selected'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Trading Pairs:</span>
                  <span className="font-medium">{formData.symbols.length} pairs selected</span>
                </div>
              </CardContent>
            </Card>
            
            <div className="flex items-center space-x-2">
              <Switch 
                id="autoStart"
                checked={formData.autoStart}
                onCheckedChange={(value) => handleFieldChange('autoStart', value)}
              />
              <Label htmlFor="autoStart" className="text-sm">
                Auto-start bot after creation
              </Label>
            </div>
            
            <div className="flex items-center space-x-2">
              <Label htmlFor="tradingMode" className="text-sm">Trading Mode:</Label>
              <Select value={formData.tradingMode} onValueChange={(value) => handleFieldChange('tradingMode', value)}>
                <SelectTrigger className="w-48">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="paper">Paper Trading (Simulated)</SelectItem>
                  <SelectItem value="live">Live Trading (Real Money)</SelectItem>
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
              Step {currentStep + 1} of {steps.length}
            </span>
            <span className="text-sm text-muted-foreground">
              {steps[currentStep].description}
            </span>
          </div>
          <Progress value={((currentStep + 1) / steps.length) * 100} className="h-2" />
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
                  index <= currentStep ? "text-primary" : "text-muted-foreground"
                )}
              >
                <div className={cn(
                  "w-8 h-8 rounded-full flex items-center justify-center border-2",
                  index < currentStep ? "bg-primary border-primary text-white" :
                  index === currentStep ? "border-primary text-primary" :
                  "border-gray-300"
                )}>
                  {index < currentStep ? (
                    <CheckCircle className="h-4 w-4" />
                  ) : (
                    <Icon className="h-4 w-4" />
                  )}
                </div>
                <span className="text-xs font-medium hidden sm:block">{step.title}</span>
              </div>
            );
          })}
        </div>
        
        {/* Error Message */}
        {error && (
          <Alert variant="destructive" className="mb-4">
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}
        
        {/* Step Content */}
        <div className="min-h-[400px]">
          <AnimatePresence mode="wait">
            {renderStep()}
          </AnimatePresence>
        </div>
        
        {/* Footer Actions */}
        <DialogFooter className="flex justify-between">
          <Button
            variant="outline"
            onClick={handleBack}
            disabled={currentStep === 0 || loading}
          >
            <ChevronLeft className="mr-1 h-4 w-4" />
            Back
          </Button>
          
          <div className="flex space-x-2">
            <Button variant="ghost" onClick={onClose} disabled={loading}>
              Cancel
            </Button>
            
            {currentStep < steps.length - 1 ? (
              <Button onClick={handleNext} disabled={loading}>
                Next
                <ChevronRight className="ml-1 h-4 w-4" />
              </Button>
            ) : (
              <Button onClick={handleSubmit} disabled={loading} className="bg-gradient-to-r from-blue-600 to-blue-700">
                {loading ? (
                  <>Creating Bot...</>
                ) : (
                  <>
                    <Rocket className="mr-1 h-4 w-4" />
                    {formData.autoStart ? 'Launch Bot' : 'Create Bot'}
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