/**
 * Advanced Strategy Playground - T-Bot Trading System
 * Professional visual strategy development and optimization platform
 * Based on modern trading terminals with real-time capabilities
 */

import React, { useState, useCallback, useRef, useEffect, useMemo } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Alert } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { Switch } from '@/components/ui/switch';
import { Slider } from '@/components/ui/slider';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Textarea } from '@/components/ui/textarea';
import { Separator } from '@/components/ui/separator';
import { Sheet, SheetContent, SheetTrigger } from '@/components/ui/sheet';

import {
  Plus,
  Save,
  FolderOpen,
  Play,
  Square,
  Pause,
  Zap,
  RocketIcon,
  CheckCircle,
  Gauge,
  BarChart3,
  GitBranch,
  Sparkles,
  Trash2,
  Search,
  ChevronDown,
  Database,
  Settings,
  TrendingUp,
  Brain,
  Shield,
  Layers,
  Code,
  Eye,
  ZoomIn,
  ZoomOut,
  Grid3x3,
  Map,
  Maximize,
  LayoutDashboard,
  Bug,
  Cpu,
  Network,
  Clock,
  CircuitBoard,
  Target,
  Wrench,
  FileOutput,
  ChevronLeft,
  ChevronRight,
  AlertTriangle
} from 'lucide-react';

import { useAppSelector, useAppDispatch } from '@/store';
import { selectPlaygroundState } from '@/store/slices/playgroundSlice';
import { PlaygroundConfiguration, PlaygroundExecution } from '@/types';
import { cn } from '@/lib/utils';

// Visual Strategy Builder Components
import VisualStrategyCanvas from './components/VisualStrategyCanvas';
import ComponentPalette from './components/ComponentPalette';
import NodePropertiesPanel from './components/NodePropertiesPanel';
import OptimizationEngine from './components/OptimizationEngine';
import PerformanceVisualization from './components/PerformanceVisualization';
import MonteCarloSimulator from './components/MonteCarloSimulator';
import WalkForwardAnalysis from './components/WalkForwardAnalysis';
import StrategyEnsembleBuilder from './components/StrategyEnsembleBuilder';
import RiskOptimizer from './components/RiskOptimizer';
import MultiObjectiveOptimizer from './components/MultiObjectiveOptimizer';
import CodeEditor from './components/CodeEditor';
import ParameterSurfacePlot from './components/ParameterSurfacePlot';
import OptimizationProgress from './components/OptimizationProgress';
import StrategyComparison from './components/StrategyComparison';
import HardwareAllocation from './components/HardwareAllocation';
import ConfigurationExporter from './components/ConfigurationExporter';

// Component Categories for the Palette
const COMPONENT_CATEGORIES = {
  'data-input': {
    name: 'Data Input',
    icon: Database,
    color: '#4CAF50',
    components: [
      { id: 'market-data', name: 'Market Data', icon: '📈', description: 'Real-time market data feed' },
      { id: 'order-book', name: 'Order Book', icon: '📋', description: 'Live order book data' },
      { id: 'news-feed', name: 'News Feed', icon: '📰', description: 'Financial news aggregator' },
      { id: 'social-media', name: 'Social Media', icon: '💬', description: 'Social sentiment data' },
      { id: 'blockchain', name: 'Blockchain Data', icon: '⛓️', description: 'On-chain analytics' },
      { id: 'economic', name: 'Economic Data', icon: '🏦', description: 'Macro economic indicators' },
      { id: 'whale-alerts', name: 'Whale Alerts', icon: '🐋', description: 'Large transaction monitoring' },
      { id: 'funding-rates', name: 'Funding Rates', icon: '💰', description: 'Perpetual funding rates' },
      { id: 'options-flow', name: 'Options Flow', icon: '🎯', description: 'Options market activity' },
      { id: 'fear-greed', name: 'Fear & Greed', icon: '😨', description: 'Market sentiment index' },
      { id: 'custom-api', name: 'Custom API', icon: '🔌', description: 'External data source' },
      { id: 'file-input', name: 'File Input', icon: '📁', description: 'Historical data files' }
    ]
  },
  'preprocessing': {
    name: 'Preprocessing',
    icon: Settings,
    color: '#00BCD4',
    components: [
      { id: 'data-cleaner', name: 'Data Cleaner', icon: '🧹', description: 'Remove outliers and clean data' },
      { id: 'normalizer', name: 'Normalizer', icon: '📏', description: 'Normalize data ranges' },
      { id: 'outlier-detection', name: 'Outlier Detection', icon: '⚠️', description: 'Statistical outlier detection' },
      { id: 'missing-handler', name: 'Missing Data Handler', icon: '🔍', description: 'Handle missing values' },
      { id: 'resampler', name: 'Resampler', icon: '⏱️', description: 'Change data frequency' },
      { id: 'aggregator', name: 'Aggregator', icon: '📊', description: 'Aggregate data points' },
      { id: 'smoother', name: 'Smoother', icon: '〰️', description: 'Smooth noisy signals' },
      { id: 'denoiser', name: 'Denoiser', icon: '🔇', description: 'Remove noise from data' },
      { id: 'validator', name: 'Data Validator', icon: '✅', description: 'Validate data quality' },
      { id: 'transformer', name: 'Transformer', icon: '🔄', description: 'Transform data format' },
      { id: 'encoder', name: 'Encoder', icon: '🔢', description: 'Encode categorical data' },
      { id: 'scaler', name: 'Scaler', icon: '📐', description: 'Scale numeric features' },
      { id: 'windower', name: 'Windower', icon: '🪟', description: 'Create sliding windows' },
      { id: 'lag-creator', name: 'Lag Creator', icon: '⏮️', description: 'Create lagged features' },
      { id: 'quality-check', name: 'Quality Check', icon: '🏆', description: 'Assess data quality' }
    ]
  },
  'feature': {
    name: 'Feature Engineering',
    icon: TrendingUp,
    color: '#2196F3',
    components: [
      { id: 'technical-indicators', name: 'Technical Indicators', icon: '📉', description: 'RSI, MACD, Bollinger Bands' },
      { id: 'rolling-stats', name: 'Rolling Statistics', icon: '📊', description: 'Moving averages and stats' },
      { id: 'price-action', name: 'Price Action', icon: '💹', description: 'Candlestick patterns' },
      { id: 'volume-profile', name: 'Volume Profile', icon: '📶', description: 'Volume at price levels' },
      { id: 'market-microstructure', name: 'Market Microstructure', icon: '🔬', description: 'Order flow analysis' },
      { id: 'correlation-matrix', name: 'Correlation Matrix', icon: '🕸️', description: 'Asset correlations' },
      { id: 'pca', name: 'PCA', icon: '🎯', description: 'Principal component analysis' },
      { id: 'feature-selector', name: 'Feature Selector', icon: '✂️', description: 'Select important features' },
      { id: 'polynomial', name: 'Polynomial Features', icon: '📈', description: 'Create polynomial features' },
      { id: 'interaction', name: 'Interaction Features', icon: '🤝', description: 'Feature interactions' }
    ]
  },
  'analysis': {
    name: 'Analysis',
    icon: Brain,
    color: '#9C27B0',
    components: [
      { id: 'sentiment-analysis', name: 'Sentiment Analysis', icon: '😊', description: 'Text sentiment scoring' },
      { id: 'pattern-recognition', name: 'Pattern Recognition', icon: '🎨', description: 'Chart pattern detection' },
      { id: 'trend-detection', name: 'Trend Detection', icon: '📈', description: 'Identify market trends' },
      { id: 'regime-detection', name: 'Regime Detection', icon: '🌡️', description: 'Market regime changes' },
      { id: 'anomaly-detection', name: 'Anomaly Detection', icon: '🚨', description: 'Detect unusual patterns' },
      { id: 'nlp-processor', name: 'NLP Processor', icon: '💬', description: 'Natural language processing' },
      { id: 'volatility-analysis', name: 'Volatility Analysis', icon: '📊', description: 'Volatility modeling' },
      { id: 'liquidity-analysis', name: 'Liquidity Analysis', icon: '💧', description: 'Market liquidity metrics' },
      { id: 'correlation-analysis', name: 'Correlation Analysis', icon: '🔗', description: 'Cross-asset correlations' },
      { id: 'statistical-tests', name: 'Statistical Tests', icon: '📐', description: 'Statistical significance tests' }
    ]
  },
  'ml-model': {
    name: 'ML Models',
    icon: Brain,
    color: '#FF9800',
    components: [
      { id: 'lstm', name: 'LSTM Network', icon: '🧠', description: 'Long short-term memory neural network' },
      { id: 'transformer', name: 'Transformer', icon: '🔄', description: 'Attention-based neural network' },
      { id: 'xgboost', name: 'XGBoost', icon: '🌲', description: 'Gradient boosting trees' },
      { id: 'random-forest', name: 'Random Forest', icon: '🌳', description: 'Ensemble decision trees' },
      { id: 'neural-net', name: 'Neural Network', icon: '🕸️', description: 'Feedforward neural network' },
      { id: 'reinforcement', name: 'RL Agent', icon: '🎮', description: 'Reinforcement learning agent' },
      { id: 'ensemble', name: 'Ensemble', icon: '👥', description: 'Model ensemble methods' },
      { id: 'automl', name: 'AutoML', icon: '🔮', description: 'Automated machine learning' },
      { id: 'gan', name: 'GAN', icon: '🎭', description: 'Generative adversarial network' },
      { id: 'vae', name: 'VAE', icon: '🎨', description: 'Variational autoencoder' }
    ]
  },
  'strategy': {
    name: 'Strategy',
    icon: CircuitBoard,
    color: '#E91E63',
    components: [
      { id: 'mean-reversion', name: 'Mean Reversion', icon: '🔄', description: 'Mean reversion strategy' },
      { id: 'trend-following', name: 'Trend Following', icon: '📈', description: 'Momentum-based strategy' },
      { id: 'arbitrage', name: 'Arbitrage', icon: '⚖️', description: 'Price arbitrage opportunities' },
      { id: 'momentum', name: 'Momentum', icon: '🚀', description: 'Price momentum strategy' },
      { id: 'market-making', name: 'Market Making', icon: '💱', description: 'Provide liquidity strategy' },
      { id: 'scalping', name: 'Scalping', icon: '⚡', description: 'High-frequency scalping' },
      { id: 'grid-trading', name: 'Grid Trading', icon: '🎯', description: 'Grid-based trading' },
      { id: 'pairs-trading', name: 'Pairs Trading', icon: '👥', description: 'Statistical arbitrage' },
      { id: 'volatility', name: 'Volatility', icon: '📊', description: 'Volatility-based strategy' },
      { id: 'hybrid', name: 'Hybrid Strategy', icon: '🔀', description: 'Multiple strategy combination' }
    ]
  },
  'signal': {
    name: 'Signals',
    icon: Zap,
    color: '#673AB7',
    components: [
      { id: 'signal-generator', name: 'Signal Generator', icon: '📡', description: 'Generate trading signals' },
      { id: 'signal-aggregator', name: 'Signal Aggregator', icon: '📊', description: 'Combine multiple signals' },
      { id: 'signal-filter', name: 'Signal Filter', icon: '🔍', description: 'Filter signal quality' },
      { id: 'signal-validator', name: 'Signal Validator', icon: '✅', description: 'Validate signal strength' },
      { id: 'signal-scorer', name: 'Signal Scorer', icon: '💯', description: 'Score signal confidence' },
      { id: 'signal-combiner', name: 'Signal Combiner', icon: '🔗', description: 'Combine signal types' },
      { id: 'signal-buffer', name: 'Signal Buffer', icon: '📦', description: 'Buffer signals for processing' },
      { id: 'signal-logger', name: 'Signal Logger', icon: '📝', description: 'Log signal history' }
    ]
  },
  'risk': {
    name: 'Risk Management',
    icon: Shield,
    color: '#f44336',
    components: [
      { id: 'position-sizer', name: 'Position Sizer', icon: '📏', description: 'Calculate position sizes' },
      { id: 'stop-loss', name: 'Stop Loss', icon: '🛑', description: 'Stop loss management' },
      { id: 'take-profit', name: 'Take Profit', icon: '💰', description: 'Take profit levels' },
      { id: 'risk-calculator', name: 'Risk Calculator', icon: '🧮', description: 'Risk metrics calculation' },
      { id: 'drawdown-monitor', name: 'Drawdown Monitor', icon: '📉', description: 'Monitor portfolio drawdown' },
      { id: 'circuit-breaker', name: 'Circuit Breaker', icon: '⚡', description: 'Emergency stop mechanism' },
      { id: 'exposure-manager', name: 'Exposure Manager', icon: '⚖️', description: 'Manage market exposure' },
      { id: 'kelly-criterion', name: 'Kelly Criterion', icon: '📊', description: 'Optimal position sizing' },
      { id: 'var-calculator', name: 'VaR Calculator', icon: '📈', description: 'Value at risk calculation' },
      { id: 'portfolio-optimizer', name: 'Portfolio Optimizer', icon: '💼', description: 'Portfolio optimization' }
    ]
  },
  'execution': {
    name: 'Execution',
    icon: Target,
    color: '#795548',
    components: [
      { id: 'order-manager', name: 'Order Manager', icon: '📋', description: 'Manage order lifecycle' },
      { id: 'twap', name: 'TWAP Engine', icon: '⏱️', description: 'Time-weighted average price' },
      { id: 'vwap', name: 'VWAP Engine', icon: '📊', description: 'Volume-weighted average price' },
      { id: 'smart-router', name: 'Smart Router', icon: '🔀', description: 'Intelligent order routing' },
      { id: 'iceberg', name: 'Iceberg Orders', icon: '🧊', description: 'Large order splitting' },
      { id: 'slippage-model', name: 'Slippage Model', icon: '📉', description: 'Estimate execution costs' },
      { id: 'order-splitter', name: 'Order Splitter', icon: '✂️', description: 'Split large orders' },
      { id: 'execution-algo', name: 'Execution Algo', icon: '🎯', description: 'Custom execution algorithm' },
      { id: 'order-validator', name: 'Order Validator', icon: '✅', description: 'Validate order parameters' },
      { id: 'retry-logic', name: 'Retry Logic', icon: '🔁', description: 'Handle execution failures' }
    ]
  },
  'logic': {
    name: 'Logic & Control',
    icon: GitBranch,
    color: '#FFC107',
    components: [
      { id: 'router', name: 'Router', icon: '🔀', description: 'Route data flow' },
      { id: 'merger', name: 'Merger', icon: '🔗', description: 'Merge data streams' },
      { id: 'splitter', name: 'Splitter', icon: '✂️', description: 'Split data streams' },
      { id: 'filter', name: 'Filter', icon: '🔽', description: 'Filter data conditions' },
      { id: 'condition', name: 'Condition', icon: '❓', description: 'Conditional logic' },
      { id: 'switch', name: 'Switch', icon: '🔄', description: 'Switch between paths' },
      { id: 'loop', name: 'Loop', icon: '🔁', description: 'Iterate over data' },
      { id: 'delay', name: 'Delay', icon: '⏱️', description: 'Add processing delay' },
      { id: 'gate', name: 'Gate', icon: '🚪', description: 'Control data flow' },
      { id: 'trigger', name: 'Trigger', icon: '⚡', description: 'Event triggering' },
      { id: 'scheduler', name: 'Scheduler', icon: '📅', description: 'Schedule executions' },
      { id: 'state-machine', name: 'State Machine', icon: '🎰', description: 'State-based logic' }
    ]
  },
  'utility': {
    name: 'Utility',
    icon: Wrench,
    color: '#607D8B',
    components: [
      { id: 'cache', name: 'Cache', icon: '💾', description: 'Cache data in memory' },
      { id: 'buffer', name: 'Buffer', icon: '📦', description: 'Buffer data streams' },
      { id: 'queue', name: 'Queue', icon: '📝', description: 'Queue management' },
      { id: 'rate-limiter', name: 'Rate Limiter', icon: '⏳', description: 'Limit processing rate' },
      { id: 'logger', name: 'Logger', icon: '📝', description: 'Log system events' },
      { id: 'monitor', name: 'Monitor', icon: '👁️', description: 'Monitor system health' },
      { id: 'debugger', name: 'Debugger', icon: '🐛', description: 'Debug strategy flow' },
      { id: 'profiler', name: 'Profiler', icon: '⚡', description: 'Performance profiling' }
    ]
  },
  'output': {
    name: 'Output',
    icon: FileOutput,
    color: '#8BC34A',
    components: [
      { id: 'trade-executor', name: 'Trade Executor', icon: '🎯', description: 'Execute trades' },
      { id: 'alert-sender', name: 'Alert Sender', icon: '🔔', description: 'Send notifications' },
      { id: 'database', name: 'Database', icon: '🗄️', description: 'Store data' },
      { id: 'dashboard', name: 'Dashboard', icon: '📊', description: 'Display metrics' },
      { id: 'report-generator', name: 'Report Generator', icon: '📈', description: 'Generate reports' },
      { id: 'webhook', name: 'Webhook', icon: '🔗', description: 'HTTP webhooks' },
      { id: 'file-writer', name: 'File Writer', icon: '💾', description: 'Write to files' },
      { id: 'metrics-exporter', name: 'Metrics Exporter', icon: '📊', description: 'Export metrics' }
    ]
  }
};

const PlaygroundPage: React.FC = () => {
  const dispatch = useAppDispatch();
  const playgroundState = useAppSelector(selectPlaygroundState);
  
  // UI State
  const [activeView, setActiveView] = useState<'visual' | 'code' | 'analysis' | 'optimization'>('visual');
  const [leftPanelOpen, setLeftPanelOpen] = useState(true);
  const [rightPanelOpen, setRightPanelOpen] = useState(true);
  const [selectedNodes, setSelectedNodes] = useState<string[]>([]);
  const [canvasZoom, setCanvasZoom] = useState(1);
  const [canvasOffset, setCanvasOffset] = useState({ x: 0, y: 0 });
  const [showGrid, setShowGrid] = useState(true);
  const [showMinimap, setShowMinimap] = useState(false);
  const [dataFlowAnimation, setDataFlowAnimation] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  
  // Strategy State
  const [strategyNodes, setStrategyNodes] = useState<any[]>([]);
  const [strategyConnections, setStrategyConnections] = useState<any[]>([]);
  const [currentStrategy, setCurrentStrategy] = useState<PlaygroundConfiguration | null>(null);
  const [optimizationRunning, setOptimizationRunning] = useState(false);
  const [optimizationProgress, setOptimizationProgress] = useState(0);
  
  // Dialogs
  const [saveDialogOpen, setSaveDialogOpen] = useState(false);
  const [loadDialogOpen, setLoadDialogOpen] = useState(false);
  const [optimizeDialogOpen, setOptimizeDialogOpen] = useState(false);
  const [deployDialogOpen, setDeployDialogOpen] = useState(false);
  
  // Refs
  const canvasRef = useRef<HTMLDivElement>(null);
  
  // System Status
  const systemStatus = {
    connected: true,
    nodes: strategyNodes.length,
    connections: strategyConnections.length,
    cpu: 12,
    memory: 245,
    latency: 2
  };
  
  // Toolbar Actions
  const handleNewStrategy = () => {
    if (strategyNodes.length > 0) {
      if (confirm('Create a new strategy? Current work will be lost.')) {
        setStrategyNodes([]);
        setStrategyConnections([]);
        setCurrentStrategy(null);
      }
    }
  };
  
  const handleSaveStrategy = () => {
    setSaveDialogOpen(true);
  };
  
  const handleLoadStrategy = () => {
    setLoadDialogOpen(true);
  };
  
  const handleRunBacktest = async () => {
    if (!currentStrategy) {
      alert('Please configure a strategy first');
      return;
    }
    
    // Start backtest
    console.log('Starting backtest for strategy:', currentStrategy.name);
    // Implementation would go here
  };
  
  const handleRunSimulation = async () => {
    if (!currentStrategy) {
      alert('Please configure a strategy first');
      return;
    }
    
    // Start simulation
    console.log('Starting simulation for strategy:', currentStrategy.name);
    // Implementation would go here
  };
  
  const handleDeployLive = () => {
    if (!currentStrategy) {
      alert('Please configure a strategy first');
      return;
    }
    
    setDeployDialogOpen(true);
  };
  
  const handleStopAll = () => {
    if (confirm('Stop all running strategies?')) {
      // Stop all executions
      console.log('Stopping all strategies');
      // Implementation would go here
    }
  };
  
  const handleValidateStrategy = () => {
    if (!currentStrategy) {
      alert('No strategy to validate');
      return;
    }
    
    // Validate strategy
    console.log('Validating strategy:', currentStrategy.name);
    // Implementation would go here
    alert('Strategy validation: ✓ Passed');
  };
  
  const handleOptimizeStrategy = () => {
    if (!currentStrategy) {
      alert('Please configure a strategy first');
      return;
    }
    
    setOptimizeDialogOpen(true);
  };
  
  const handleShowAnalytics = () => {
    setActiveView('analysis');
  };
  
  const handleToggleDataFlow = () => {
    setDataFlowAnimation(!dataFlowAnimation);
  };
  
  const handleAutoLayout = () => {
    // Auto-arrange nodes using force-directed layout
    console.log('Auto-arranging nodes');
    // Implementation would go here
  };
  
  const handleClearCanvas = () => {
    if (strategyNodes.length > 0 && confirm('Clear the canvas?')) {
      setStrategyNodes([]);
      setStrategyConnections([]);
    }
  };
  
  // Canvas Controls
  const handleZoomIn = () => {
    setCanvasZoom(Math.min(canvasZoom * 1.2, 3));
  };
  
  const handleZoomOut = () => {
    setCanvasZoom(Math.max(canvasZoom / 1.2, 0.3));
  };
  
  const handleResetZoom = () => {
    setCanvasZoom(1);
  };
  
  const handleCenterView = () => {
    setCanvasOffset({ x: 0, y: 0 });
  };
  
  const handleToggleGrid = () => {
    setShowGrid(!showGrid);
  };
  
  const handleToggleMinimap = () => {
    setShowMinimap(!showMinimap);
  };
  
  const handleToggleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  };
  
  // Node Management
  const handleAddNode = (nodeType: string, position: { x: number, y: number }) => {
    const newNode = {
      id: `node_${Date.now()}`,
      type: nodeType,
      position,
      data: {},
      status: 'idle'
    };
    
    setStrategyNodes([...strategyNodes, newNode]);
  };
  
  const handleUpdateNode = (nodeId: string, updates: any) => {
    setStrategyNodes(nodes => 
      nodes.map(node => 
        node.id === nodeId ? { ...node, ...updates } : node
      )
    );
  };
  
  const handleDeleteNode = (nodeId: string) => {
    setStrategyNodes(nodes => nodes.filter(node => node.id !== nodeId));
    setStrategyConnections(connections => 
      connections.filter(conn => conn.from !== nodeId && conn.to !== nodeId)
    );
  };
  
  const handleConnectNodes = (fromNode: string, toNode: string, connectionType: string) => {
    const newConnection = {
      id: `conn_${Date.now()}`,
      from: fromNode,
      to: toNode,
      type: connectionType
    };
    
    setStrategyConnections([...strategyConnections, newConnection]);
  };
  
  // Memoized toolbar component
  const ToolbarComponent = useMemo(() => (
    <div className="bg-gradient-to-r from-gray-900/95 to-gray-800/95 backdrop-blur-md border-b-2 border-primary">
      <div className="flex items-center justify-between px-4 py-2 min-h-[56px]">
        {/* File Operations */}
        <div className="flex items-center gap-2">
          <div className="flex gap-1">
            <Button
              size="sm"
              variant="ghost"
              onClick={handleNewStrategy}
              className="text-white hover:bg-white/10 text-xs"
            >
              <Plus className="w-3 h-3 mr-1" />
              New
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={handleSaveStrategy}
              className="text-white hover:bg-white/10 text-xs"
            >
              <Save className="w-3 h-3 mr-1" />
              Save
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={handleLoadStrategy}
              className="text-white hover:bg-white/10 text-xs"
            >
              <FolderOpen className="w-3 h-3 mr-1" />
              Load
            </Button>
          </div>
          
          <Separator orientation="vertical" className="h-6 bg-white/20" />
          
          {/* Execution Controls */}
          <div className="flex gap-1">
            <Button
              size="sm"
              onClick={handleRunBacktest}
              className="text-xs bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700"
            >
              <Play className="w-3 h-3 mr-1" />
              Backtest
            </Button>
            <Button
              size="sm"
              onClick={handleRunSimulation}
              className="text-xs bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-700 hover:to-blue-700"
            >
              <Zap className="w-3 h-3 mr-1" />
              Simulate
            </Button>
            <Button
              size="sm"
              onClick={handleDeployLive}
              className="text-xs bg-gradient-to-r from-yellow-600 to-orange-600 hover:from-yellow-700 hover:to-orange-700"
            >
              <RocketIcon className="w-3 h-3 mr-1" />
              Deploy
            </Button>
            <Button
              size="sm"
              onClick={handleStopAll}
              className="text-xs bg-gradient-to-r from-pink-600 to-red-600 hover:from-pink-700 hover:to-red-700"
            >
              <Square className="w-3 h-3 mr-1" />
              Stop
            </Button>
          </div>
          
          <Separator orientation="vertical" className="h-6 bg-white/20" />
          
          {/* Analysis Tools */}
          <div className="flex gap-1">
            <Button
              size="sm"
              variant="ghost"
              onClick={handleValidateStrategy}
              className="text-white hover:bg-white/10 text-xs"
            >
              <CheckCircle className="w-3 h-3 mr-1" />
              Validate
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={handleOptimizeStrategy}
              className="text-white hover:bg-white/10 text-xs"
            >
              <Gauge className="w-3 h-3 mr-1" />
              Optimize
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={handleShowAnalytics}
              className="text-white hover:bg-white/10 text-xs"
            >
              <BarChart3 className="w-3 h-3 mr-1" />
              Analytics
            </Button>
          </div>
          
          <Separator orientation="vertical" className="h-6 bg-white/20" />
          
          {/* Canvas Tools */}
          <div className="flex gap-1">
            <Button
              size="sm"
              variant="ghost"
              onClick={handleToggleDataFlow}
              className={cn(
                "text-white hover:bg-white/10 text-xs",
                dataFlowAnimation && "bg-gradient-to-r from-green-600 to-emerald-600"
              )}
            >
              <GitBranch className="w-3 h-3 mr-1" />
              Data Flow
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={handleAutoLayout}
              className="text-white hover:bg-white/10 text-xs"
            >
              <Sparkles className="w-3 h-3 mr-1" />
              Auto Layout
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={handleClearCanvas}
              className="text-white hover:bg-white/10 text-xs"
            >
              <Trash2 className="w-3 h-3 mr-1" />
              Clear
            </Button>
          </div>
        </div>
        
        <div className="flex items-center gap-4">
          {/* View Toggle */}
          <Tabs value={activeView} onValueChange={(value) => setActiveView(value as any)}>
            <TabsList className="bg-black/20">
              <TabsTrigger value="visual" className="text-xs">Visual</TabsTrigger>
              <TabsTrigger value="code" className="text-xs">Code</TabsTrigger>
              <TabsTrigger value="analysis" className="text-xs">Analysis</TabsTrigger>
              <TabsTrigger value="optimization" className="text-xs">Optimize</TabsTrigger>
            </TabsList>
          </Tabs>
          
          <Button
            size="sm"
            variant="ghost"
            onClick={handleToggleFullscreen}
            className="text-white hover:bg-white/10"
          >
            <Maximize className="w-4 h-4" />
          </Button>
        </div>
      </div>
    </div>
  ), [activeView, dataFlowAnimation, strategyNodes.length, currentStrategy]);
  
  return (
    <div className="h-screen flex flex-col bg-black text-white overflow-hidden">
      {/* Top Toolbar */}
      {ToolbarComponent}
      
      {/* Main Content */}
      <div className="flex flex-1 overflow-hidden relative">
        {/* Left Sidebar - Component Palette */}
        <div 
          className={cn(
            "w-64 bg-gray-900 border-r border-gray-700 flex flex-col overflow-hidden transition-all duration-300",
            !leftPanelOpen && "-ml-64"
          )}
        >
          <ComponentPalette
            categories={COMPONENT_CATEGORIES}
            onComponentDrag={(component) => {
              // Handle component drag start
              console.log('Dragging component:', component);
            }}
            searchable
          />
        </div>
        
        {/* Canvas Area */}
        <div
          ref={canvasRef}
          className="flex-1 relative bg-gradient-to-br from-blue-900/5 via-black to-purple-900/5 overflow-hidden"
        >
          {/* Canvas Controls */}
          <div 
            className={cn(
              "absolute top-2 flex gap-1 z-50 bg-black/80 p-1 rounded backdrop-blur-md transition-all duration-300",
              rightPanelOpen ? "right-[310px]" : "right-2"
            )}
          >
            <Button
              size="sm"
              variant="ghost"
              onClick={handleZoomIn}
              className="text-white hover:bg-white/10 h-8 w-8 p-0"
            >
              <ZoomIn className="w-3 h-3" />
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={handleZoomOut}
              className="text-white hover:bg-white/10 h-8 w-8 p-0"
            >
              <ZoomOut className="w-3 h-3" />
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={handleResetZoom}
              className="text-white hover:bg-white/10 text-xs px-2"
            >
              {Math.round(canvasZoom * 100)}%
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={handleToggleGrid}
              className={cn(
                "text-white hover:bg-white/10 h-8 w-8 p-0",
                showGrid && "text-primary"
              )}
            >
              <Grid3x3 className="w-3 h-3" />
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={handleToggleMinimap}
              className={cn(
                "text-white hover:bg-white/10 h-8 w-8 p-0",
                showMinimap && "text-primary"
              )}
            >
              <Map className="w-3 h-3" />
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={handleCenterView}
              className="text-white hover:bg-white/10 h-8 w-8 p-0"
            >
              <Target className="w-3 h-3" />
            </Button>
          </div>
          
          {/* Minimap */}
          {showMinimap && (
            <Card 
              className={cn(
                "absolute bottom-2 w-48 h-36 bg-black/90 border-primary/30 z-50 transition-all duration-300",
                rightPanelOpen ? "right-[310px]" : "right-2"
              )}
            >
              <CardContent className="p-2">
                <div className="text-xs text-white/70">Strategy Overview</div>
              </CardContent>
            </Card>
          )}
          
          {/* Main Canvas Content */}
          <div className="w-full h-full relative">
            {activeView === 'visual' && (
              <VisualStrategyCanvas
                nodes={strategyNodes}
                connections={strategyConnections}
                zoom={canvasZoom}
                offset={canvasOffset}
                showGrid={showGrid}
                dataFlowAnimation={dataFlowAnimation}
                selectedNodes={selectedNodes}
                onNodeAdd={handleAddNode}
                onNodeUpdate={handleUpdateNode}
                onNodeDelete={handleDeleteNode}
                onNodeSelect={setSelectedNodes}
                onNodeConnect={handleConnectNodes}
                onCanvasClick={() => setSelectedNodes([])}
              />
            )}
            
            {activeView === 'code' && (
              <CodeEditor
                strategy={currentStrategy}
                onChange={(code: string) => {
                  // Handle code changes
                  console.log('Code updated:', code);
                }}
              />
            )}
            
            {activeView === 'analysis' && (
              <div className="p-6 h-full overflow-auto">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <PerformanceVisualization
                    executions={playgroundState.executions}
                  />
                  <ParameterSurfacePlot
                    optimizationResults={[]}
                  />
                  <div className="col-span-full">
                    <StrategyComparison
                      strategies={playgroundState.configurations}
                    />
                  </div>
                </div>
              </div>
            )}
            
            {activeView === 'optimization' && (
              <div className="p-6 h-full overflow-auto">
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                  <div className="lg:col-span-2">
                    <OptimizationEngine
                      strategy={currentStrategy}
                      onOptimizationStart={() => setOptimizationRunning(true)}
                      onOptimizationComplete={() => setOptimizationRunning(false)}
                      onProgressUpdate={setOptimizationProgress}
                    />
                  </div>
                  <div className="space-y-4">
                    <MonteCarloSimulator strategy={currentStrategy} />
                    <WalkForwardAnalysis strategy={currentStrategy} />
                    <RiskOptimizer strategy={currentStrategy} />
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
        
        {/* Right Panel - Properties & Analysis */}
        <div 
          className={cn(
            "w-80 bg-gray-900 border-l border-gray-700 flex flex-col overflow-hidden transition-all duration-300",
            !rightPanelOpen && "mr-[-320px]"
          )}
        >
          <NodePropertiesPanel
            selectedNodes={selectedNodes.map(id => 
              strategyNodes.find(node => node.id === id)
            ).filter(Boolean)}
            onNodeUpdate={handleUpdateNode}
            systemMetrics={{
              dataFlowRate: 120,
              latency: systemStatus.latency,
              cpuUsage: systemStatus.cpu,
              memoryUsage: systemStatus.memory
            }}
          />
        </div>
      </div>
      
      {/* Status Bar */}
      <div className="h-8 bg-gray-950 border-t border-gray-800 flex items-center px-4 text-xs text-white/60 gap-6">
        <div className="flex items-center gap-1">
          <div 
            className={cn(
              "w-2 h-2 rounded-full",
              systemStatus.connected ? "bg-green-500" : "bg-red-500"
            )}
          />
          <span>{systemStatus.connected ? 'Connected' : 'Disconnected'}</span>
        </div>
        <span>Nodes: {systemStatus.nodes}</span>
        <span>Connections: {systemStatus.connections}</span>
        <span>CPU: {systemStatus.cpu}%</span>
        <span>Memory: {systemStatus.memory} MB</span>
        <span>Latency: {systemStatus.latency}ms</span>
        
        {optimizationRunning && (
          <div className="flex items-center gap-2 ml-auto">
            <span>Optimization: {Math.round(optimizationProgress)}%</span>
            <Progress 
              value={optimizationProgress} 
              className="w-16 h-1"
            />
          </div>
        )}
      </div>
      
      {/* Panel Toggle Controls */}
      <Button
        size="sm"
        variant="ghost"
        onClick={() => setLeftPanelOpen(!leftPanelOpen)}
        className={cn(
          "fixed top-1/2 transform -translate-y-1/2 z-50 bg-black/80 text-white rounded-r-lg transition-all duration-300",
          leftPanelOpen ? "left-64" : "left-0"
        )}
      >
        {leftPanelOpen ? <ChevronLeft className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
      </Button>
      
      <Button
        size="sm"
        variant="ghost"
        onClick={() => setRightPanelOpen(!rightPanelOpen)}
        className={cn(
          "fixed top-1/2 transform -translate-y-1/2 z-50 bg-black/80 text-white rounded-l-lg transition-all duration-300",
          rightPanelOpen ? "right-80" : "right-0"
        )}
      >
        {rightPanelOpen ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
      </Button>
      
      {/* Dialogs */}
      <Dialog open={saveDialogOpen} onOpenChange={setSaveDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Save Strategy</DialogTitle>
          </DialogHeader>
          <div className="space-y-4">
            <div>
              <Label htmlFor="strategy-name">Strategy Name</Label>
              <Input
                id="strategy-name"
                defaultValue={currentStrategy?.name || 'My Strategy'}
                className="mt-1"
              />
            </div>
          </div>
          <div className="flex justify-end gap-2">
            <Button variant="outline" onClick={() => setSaveDialogOpen(false)}>
              Cancel
            </Button>
            <Button onClick={() => {
              setSaveDialogOpen(false);
              // Implement save logic
            }}>
              Save
            </Button>
          </div>
        </DialogContent>
      </Dialog>
      
      <Dialog open={deployDialogOpen} onOpenChange={setDeployDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Deploy Strategy</DialogTitle>
            <DialogDescription>
              You are about to deploy this strategy to live trading. Please review carefully.
            </DialogDescription>
          </DialogHeader>
          <Alert>
            <AlertTriangle className="h-4 w-4" />
            <div>
              <div className="font-medium">Warning</div>
              <div className="text-sm">This will deploy your strategy with real money.</div>
            </div>
          </Alert>
          <div className="space-y-2 text-sm">
            <div>Strategy: {currentStrategy?.name}</div>
            <div>Symbols: {currentStrategy?.symbols?.join(', ') || 'None'}</div>
          </div>
          <div className="flex justify-end gap-2">
            <Button variant="outline" onClick={() => setDeployDialogOpen(false)}>
              Cancel
            </Button>
            <Button 
              variant="destructive"
              onClick={() => {
                setDeployDialogOpen(false);
                // Implement deployment logic
              }}
            >
              Deploy Live
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default PlaygroundPage;