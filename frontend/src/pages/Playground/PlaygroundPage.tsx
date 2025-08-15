/**
 * Main Playground Page for T-Bot Trading System
 * Comprehensive interface for testing strategies and configurations
 */

import React, { useState, useCallback } from 'react';
import {
  Box,
  Grid,
  Paper,
  Tabs,
  Tab,
  Typography,
  Divider,
  useTheme,
  alpha
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Settings as SettingsIcon,
  Timeline as TimelineIcon,
  Analytics as AnalyticsIcon,
  Science as ScienceIcon,
  Speed as SpeedIcon
} from '@mui/icons-material';

import { useAppSelector } from '@/store';
import { selectPlaygroundState } from '@/store/slices/playgroundSlice';
import { PlaygroundConfiguration, PlaygroundExecution } from '@/types';

// Import playground components
import ConfigurationPanel from './components/ConfigurationPanel';
import ExecutionControls from './components/ExecutionControls';
import MonitoringDashboard from './components/MonitoringDashboard';
import ResultsAnalysis from './components/ResultsAnalysis';
import AdvancedFeatures from './components/AdvancedFeatures';
import BatchOptimizer from './components/BatchOptimizer';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`playground-tabpanel-${index}`}
      aria-labelledby={`playground-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

function a11yProps(index: number) {
  return {
    id: `playground-tab-${index}`,
    'aria-controls': `playground-tabpanel-${index}`,
  };
}

const PlaygroundPage: React.FC = () => {
  const theme = useTheme();
  const playgroundState = useAppSelector(selectPlaygroundState);
  const [activeTab, setActiveTab] = useState(0);
  const [currentConfiguration, setCurrentConfiguration] = useState<PlaygroundConfiguration | null>(null);
  const [activeExecution, setActiveExecution] = useState<PlaygroundExecution | null>(null);

  const handleTabChange = useCallback((event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  }, []);

  const handleConfigurationChange = useCallback((config: PlaygroundConfiguration) => {
    setCurrentConfiguration(config);
  }, []);

  const handleExecutionStart = useCallback((execution: PlaygroundExecution) => {
    setActiveExecution(execution);
    // Automatically switch to monitoring tab when execution starts
    if (activeTab === 0) {
      setActiveTab(1);
    }
  }, [activeTab]);

  const isExecutionActive = activeExecution && 
    ['running', 'paused'].includes(activeExecution.status);

  const tabsData = [
    {
      label: 'Configuration',
      icon: <SettingsIcon />,
      description: 'Set up your trading strategy and parameters'
    },
    {
      label: 'Execution & Monitoring',
      icon: <PlayIcon />,
      description: 'Control and monitor your strategy execution'
    },
    {
      label: 'Results & Analysis',
      icon: <AnalyticsIcon />,
      description: 'Analyze performance and optimize parameters'
    },
    {
      label: 'Advanced Features',
      icon: <ScienceIcon />,
      description: 'A/B testing and parameter optimization'
    },
    {
      label: 'Batch Optimizer',
      icon: <SpeedIcon />,
      description: 'Run multiple configurations simultaneously'
    }
  ];

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      {/* Page Header */}
      <Box sx={{ mb: 3 }}>
        <Typography
          variant="h3"
          component="h1"
          gutterBottom
          sx={{
            fontWeight: 'bold',
            background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
            backgroundClip: 'text',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}
        >
          Strategy Playground
        </Typography>
        <Typography
          variant="h6"
          color="text.secondary"
          sx={{ mb: 2 }}
        >
          Test, optimize, and validate your trading strategies with comprehensive backtesting and simulation tools
        </Typography>

        {/* Status Indicators */}
        <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
          {currentConfiguration && (
            <Paper
              elevation={1}
              sx={{
                px: 2,
                py: 1,
                backgroundColor: alpha(theme.palette.success.main, 0.1),
                border: `1px solid ${alpha(theme.palette.success.main, 0.3)}`
              }}
            >
              <Typography variant="body2" color="success.main">
                Configuration: {currentConfiguration.name}
              </Typography>
            </Paper>
          )}
          
          {isExecutionActive && (
            <Paper
              elevation={1}
              sx={{
                px: 2,
                py: 1,
                backgroundColor: alpha(theme.palette.info.main, 0.1),
                border: `1px solid ${alpha(theme.palette.info.main, 0.3)}`,
                animation: 'pulse 2s infinite'
              }}
            >
              <Typography variant="body2" color="info.main">
                Execution: {activeExecution.status.toUpperCase()} ({Math.round(activeExecution.progress)}%)
              </Typography>
            </Paper>
          )}
        </Box>
      </Box>

      <Divider sx={{ mb: 3 }} />

      {/* Main Content */}
      <Paper 
        elevation={2}
        sx={{ 
          overflow: 'hidden',
          borderRadius: 2
        }}
      >
        {/* Navigation Tabs */}
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs
            value={activeTab}
            onChange={handleTabChange}
            aria-label="playground navigation tabs"
            variant="scrollable"
            scrollButtons="auto"
            sx={{
              '& .MuiTab-root': {
                minWidth: 160,
                py: 2,
                textTransform: 'none',
                fontWeight: 500
              }
            }}
          >
            {tabsData.map((tab, index) => (
              <Tab
                key={index}
                icon={tab.icon}
                label={tab.label}
                iconPosition="start"
                {...a11yProps(index)}
                title={tab.description}
              />
            ))}
          </Tabs>
        </Box>

        {/* Tab Panels */}
        <TabPanel value={activeTab} index={0}>
          <ConfigurationPanel
            configuration={currentConfiguration}
            onConfigurationChange={handleConfigurationChange}
            isExecutionActive={isExecutionActive}
          />
        </TabPanel>

        <TabPanel value={activeTab} index={1}>
          <Grid container spacing={3}>
            <Grid item xs={12} lg={4}>
              <ExecutionControls
                configuration={currentConfiguration}
                execution={activeExecution}
                onExecutionStart={handleExecutionStart}
                onExecutionControl={(action) => {
                  // Handle execution control actions (start, pause, stop, etc.)
                  console.log('Execution control:', action);
                }}
              />
            </Grid>
            <Grid item xs={12} lg={8}>
              <MonitoringDashboard
                execution={activeExecution}
                configuration={currentConfiguration}
              />
            </Grid>
          </Grid>
        </TabPanel>

        <TabPanel value={activeTab} index={2}>
          <ResultsAnalysis
            executions={playgroundState.executions}
            comparisonExecutions={playgroundState.comparisonExecutions}
          />
        </TabPanel>

        <TabPanel value={activeTab} index={3}>
          <AdvancedFeatures
            configuration={currentConfiguration}
            executions={playgroundState.executions}
          />
        </TabPanel>

        <TabPanel value={activeTab} index={4}>
          <BatchOptimizer />
        </TabPanel>
      </Paper>

      {/* Global Styles for Animations */}
      <style>
        {`
          @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
          }
        `}
      </style>
    </Box>
  );
};

export default PlaygroundPage;