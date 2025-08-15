/**
 * Playground validation utilities
 * Provides comprehensive validation for playground configurations and parameters
 */

import { PlaygroundConfiguration, PlaygroundExecution } from '@/types';

export interface ValidationError {
  field: string;
  message: string;
  severity: 'error' | 'warning' | 'info';
}

export interface ValidationResult {
  isValid: boolean;
  errors: ValidationError[];
  warnings: ValidationError[];
}

/**
 * Validate playground configuration
 */
export const validatePlaygroundConfiguration = (config: PlaygroundConfiguration): ValidationResult => {
  const errors: ValidationError[] = [];
  const warnings: ValidationError[] = [];

  // Basic information validation
  if (!config.name.trim()) {
    errors.push({
      field: 'name',
      message: 'Configuration name is required',
      severity: 'error'
    });
  }

  if (config.name.length > 100) {
    errors.push({
      field: 'name',
      message: 'Configuration name must be less than 100 characters',
      severity: 'error'
    });
  }

  // Symbol validation
  if (!config.symbols || config.symbols.length === 0) {
    errors.push({
      field: 'symbols',
      message: 'At least one trading symbol must be selected',
      severity: 'error'
    });
  }

  if (config.symbols.length > 10) {
    warnings.push({
      field: 'symbols',
      message: 'Trading more than 10 symbols may impact performance',
      severity: 'warning'
    });
  }

  // Validate symbol format
  config.symbols.forEach((symbol, index) => {
    if (!/^[A-Z]{2,10}\/[A-Z]{2,10}$/.test(symbol)) {
      errors.push({
        field: `symbols[${index}]`,
        message: `Invalid symbol format: ${symbol}. Expected format: BASE/QUOTE (e.g., BTC/USDT)`,
        severity: 'error'
      });
    }
  });

  // Position sizing validation
  if (!config.positionSizing) {
    errors.push({
      field: 'positionSizing',
      message: 'Position sizing configuration is required',
      severity: 'error'
    });
  } else {
    if (config.positionSizing.value <= 0) {
      errors.push({
        field: 'positionSizing.value',
        message: 'Position size must be greater than 0',
        severity: 'error'
      });
    }

    if (config.positionSizing.type === 'percentage' && config.positionSizing.value > 50) {
      warnings.push({
        field: 'positionSizing.value',
        message: 'Position size greater than 50% of portfolio is high risk',
        severity: 'warning'
      });
    }

    if (config.positionSizing.type === 'fixed' && config.positionSizing.value < 10) {
      warnings.push({
        field: 'positionSizing.value',
        message: 'Very small fixed position sizes may not be practical due to fees',
        severity: 'warning'
      });
    }

    if (config.positionSizing.maxPositions < 1) {
      errors.push({
        field: 'positionSizing.maxPositions',
        message: 'Maximum positions must be at least 1',
        severity: 'error'
      });
    }

    if (config.positionSizing.maxPositions > 50) {
      warnings.push({
        field: 'positionSizing.maxPositions',
        message: 'Managing more than 50 positions simultaneously may be complex',
        severity: 'warning'
      });
    }
  }

  // Risk settings validation
  if (!config.riskSettings) {
    errors.push({
      field: 'riskSettings',
      message: 'Risk settings configuration is required',
      severity: 'error'
    });
  } else {
    const { stopLossPercentage, takeProfitPercentage, maxDrawdownPercentage, maxRiskPerTrade } = config.riskSettings;

    // Stop loss validation
    if (stopLossPercentage <= 0) {
      errors.push({
        field: 'riskSettings.stopLossPercentage',
        message: 'Stop loss percentage must be greater than 0',
        severity: 'error'
      });
    }

    if (stopLossPercentage > 50) {
      warnings.push({
        field: 'riskSettings.stopLossPercentage',
        message: 'Stop loss greater than 50% is extremely high',
        severity: 'warning'
      });
    }

    // Take profit validation
    if (takeProfitPercentage <= 0) {
      errors.push({
        field: 'riskSettings.takeProfitPercentage',
        message: 'Take profit percentage must be greater than 0',
        severity: 'error'
      });
    }

    // Risk/reward ratio validation
    if (takeProfitPercentage <= stopLossPercentage) {
      errors.push({
        field: 'riskSettings',
        message: 'Take profit must be greater than stop loss for positive risk/reward ratio',
        severity: 'error'
      });
    }

    const riskRewardRatio = takeProfitPercentage / stopLossPercentage;
    if (riskRewardRatio < 1.5) {
      warnings.push({
        field: 'riskSettings',
        message: `Risk/reward ratio is ${riskRewardRatio.toFixed(2)}. Consider targeting at least 1.5:1`,
        severity: 'warning'
      });
    }

    // Max drawdown validation
    if (maxDrawdownPercentage <= 0 || maxDrawdownPercentage > 100) {
      errors.push({
        field: 'riskSettings.maxDrawdownPercentage',
        message: 'Max drawdown percentage must be between 0 and 100',
        severity: 'error'
      });
    }

    if (maxDrawdownPercentage > 30) {
      warnings.push({
        field: 'riskSettings.maxDrawdownPercentage',
        message: 'Max drawdown greater than 30% is considered high risk',
        severity: 'warning'
      });
    }

    // Max risk per trade validation
    if (maxRiskPerTrade <= 0 || maxRiskPerTrade > 100) {
      errors.push({
        field: 'riskSettings.maxRiskPerTrade',
        message: 'Max risk per trade must be between 0 and 100',
        severity: 'error'
      });
    }

    if (maxRiskPerTrade > 10) {
      warnings.push({
        field: 'riskSettings.maxRiskPerTrade',
        message: 'Risk per trade greater than 10% is extremely aggressive',
        severity: 'warning'
      });
    }
  }

  // Portfolio settings validation
  if (!config.portfolioSettings) {
    errors.push({
      field: 'portfolioSettings',
      message: 'Portfolio settings configuration is required',
      severity: 'error'
    });
  } else {
    if (config.portfolioSettings.maxPositions < 1) {
      errors.push({
        field: 'portfolioSettings.maxPositions',
        message: 'Portfolio max positions must be at least 1',
        severity: 'error'
      });
    }

    if (config.portfolioSettings.maxPositions < config.symbols.length) {
      warnings.push({
        field: 'portfolioSettings.maxPositions',
        message: 'Max positions is less than selected symbols. Some symbols may not be traded.',
        severity: 'warning'
      });
    }
  }

  // Strategy validation
  if (!config.strategy || !config.strategy.type) {
    errors.push({
      field: 'strategy.type',
      message: 'Strategy type is required',
      severity: 'error'
    });
  }

  // Model validation for ML strategies
  if (config.strategy?.type === 'ml_based' && !config.model) {
    warnings.push({
      field: 'model',
      message: 'ML-based strategy selected but no model configured',
      severity: 'warning'
    });
  }

  // Timeframe validation
  const validTimeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w'];
  if (!validTimeframes.includes(config.timeframe)) {
    errors.push({
      field: 'timeframe',
      message: `Invalid timeframe. Must be one of: ${validTimeframes.join(', ')}`,
      severity: 'error'
    });
  }

  return {
    isValid: errors.length === 0,
    errors,
    warnings
  };
};

/**
 * Validate execution settings
 */
export const validateExecutionSettings = (settings: {
  mode: 'historical' | 'live' | 'sandbox' | 'production';
  startDate?: string;
  endDate?: string;
  initialBalance: number;
  commission: number;
  speed: number;
}): ValidationResult => {
  const errors: ValidationError[] = [];
  const warnings: ValidationError[] = [];

  // Initial balance validation
  if (settings.initialBalance <= 0) {
    errors.push({
      field: 'initialBalance',
      message: 'Initial balance must be greater than 0',
      severity: 'error'
    });
  }

  if (settings.initialBalance < 100) {
    warnings.push({
      field: 'initialBalance',
      message: 'Very small initial balance may not be realistic for trading',
      severity: 'warning'
    });
  }

  // Commission validation
  if (settings.commission < 0 || settings.commission > 5) {
    errors.push({
      field: 'commission',
      message: 'Commission must be between 0% and 5%',
      severity: 'error'
    });
  }

  if (settings.commission > 1) {
    warnings.push({
      field: 'commission',
      message: 'Commission greater than 1% is quite high',
      severity: 'warning'
    });
  }

  // Historical mode specific validation
  if (settings.mode === 'historical') {
    if (!settings.startDate) {
      errors.push({
        field: 'startDate',
        message: 'Start date is required for historical backtesting',
        severity: 'error'
      });
    }

    if (!settings.endDate) {
      errors.push({
        field: 'endDate',
        message: 'End date is required for historical backtesting',
        severity: 'error'
      });
    }

    if (settings.startDate && settings.endDate) {
      const start = new Date(settings.startDate);
      const end = new Date(settings.endDate);
      const now = new Date();

      if (start >= end) {
        errors.push({
          field: 'dateRange',
          message: 'End date must be after start date',
          severity: 'error'
        });
      }

      if (end > now) {
        errors.push({
          field: 'endDate',
          message: 'End date cannot be in the future',
          severity: 'error'
        });
      }

      const daysDiff = (end.getTime() - start.getTime()) / (1000 * 60 * 60 * 24);
      if (daysDiff < 1) {
        warnings.push({
          field: 'dateRange',
          message: 'Backtest period is less than 1 day. Results may not be statistically significant.',
          severity: 'warning'
        });
      }

      if (daysDiff > 365 * 2) {
        warnings.push({
          field: 'dateRange',
          message: 'Very long backtest periods may take significant time to complete',
          severity: 'warning'
        });
      }
    }

    // Speed validation for historical mode
    if (settings.speed <= 0 || settings.speed > 1000) {
      errors.push({
        field: 'speed',
        message: 'Execution speed must be between 0.1x and 1000x',
        severity: 'error'
      });
    }

    if (settings.speed > 100) {
      warnings.push({
        field: 'speed',
        message: 'Very high execution speeds may cause instability',
        severity: 'warning'
      });
    }
  }

  // Production mode warnings
  if (settings.mode === 'production') {
    warnings.push({
      field: 'mode',
      message: 'Production mode trades with real money. Ensure thorough testing first.',
      severity: 'warning'
    });

    if (settings.initialBalance > 10000) {
      warnings.push({
        field: 'initialBalance',
        message: 'Large initial balance in production mode carries significant risk',
        severity: 'warning'
      });
    }
  }

  return {
    isValid: errors.length === 0,
    errors,
    warnings
  };
};

/**
 * Validate batch optimization settings
 */
export const validateBatchSettings = (settings: {
  configurationCount: number;
  maxConcurrentJobs: number;
  resourceLimit: number;
  overfittingProtection: boolean;
  crossValidationFolds: number;
  outOfSamplePercentage: number;
}): ValidationResult => {
  const errors: ValidationError[] = [];
  const warnings: ValidationError[] = [];

  // Configuration count validation
  if (settings.configurationCount < 1) {
    errors.push({
      field: 'configurationCount',
      message: 'Configuration count must be at least 1',
      severity: 'error'
    });
  }

  if (settings.configurationCount > 1000) {
    warnings.push({
      field: 'configurationCount',
      message: 'Large number of configurations will take significant time to complete',
      severity: 'warning'
    });
  }

  // Concurrent jobs validation
  if (settings.maxConcurrentJobs < 1 || settings.maxConcurrentJobs > 20) {
    errors.push({
      field: 'maxConcurrentJobs',
      message: 'Max concurrent jobs must be between 1 and 20',
      severity: 'error'
    });
  }

  if (settings.maxConcurrentJobs > 10) {
    warnings.push({
      field: 'maxConcurrentJobs',
      message: 'High concurrent job count may overwhelm system resources',
      severity: 'warning'
    });
  }

  // Resource limit validation
  if (settings.resourceLimit < 10 || settings.resourceLimit > 100) {
    errors.push({
      field: 'resourceLimit',
      message: 'Resource limit must be between 10% and 100%',
      severity: 'error'
    });
  }

  if (settings.resourceLimit > 80) {
    warnings.push({
      field: 'resourceLimit',
      message: 'High resource usage may impact system stability',
      severity: 'warning'
    });
  }

  // Cross-validation validation
  if (settings.crossValidationFolds < 3 || settings.crossValidationFolds > 10) {
    errors.push({
      field: 'crossValidationFolds',
      message: 'Cross-validation folds must be between 3 and 10',
      severity: 'error'
    });
  }

  // Out-of-sample validation
  if (settings.outOfSamplePercentage < 10 || settings.outOfSamplePercentage > 50) {
    errors.push({
      field: 'outOfSamplePercentage',
      message: 'Out-of-sample percentage must be between 10% and 50%',
      severity: 'error'
    });
  }

  if (!settings.overfittingProtection) {
    warnings.push({
      field: 'overfittingProtection',
      message: 'Overfitting protection is recommended for reliable results',
      severity: 'warning'
    });
  }

  return {
    isValid: errors.length === 0,
    errors,
    warnings
  };
};

/**
 * Validate parameter ranges for optimization
 */
export const validateParameterRanges = (ranges: Record<string, {
  min: number;
  max: number;
  step: number;
  enabled: boolean;
}>): ValidationResult => {
  const errors: ValidationError[] = [];
  const warnings: ValidationError[] = [];

  const enabledParams = Object.entries(ranges).filter(([_, range]) => range.enabled);

  if (enabledParams.length === 0) {
    errors.push({
      field: 'parameters',
      message: 'At least one parameter must be enabled for optimization',
      severity: 'error'
    });
  }

  enabledParams.forEach(([paramName, range]) => {
    if (range.min >= range.max) {
      errors.push({
        field: paramName,
        message: `${paramName}: Minimum value must be less than maximum value`,
        severity: 'error'
      });
    }

    if (range.step <= 0) {
      errors.push({
        field: paramName,
        message: `${paramName}: Step size must be greater than 0`,
        severity: 'error'
      });
    }

    if (range.step > (range.max - range.min)) {
      errors.push({
        field: paramName,
        message: `${paramName}: Step size cannot be larger than the range`,
        severity: 'error'
      });
    }

    const steps = (range.max - range.min) / range.step;
    if (steps > 1000) {
      warnings.push({
        field: paramName,
        message: `${paramName}: Very small step size will create many combinations (${Math.ceil(steps)} steps)`,
        severity: 'warning'
      });
    }
  });

  // Calculate total combinations
  const totalCombinations = enabledParams.reduce((total, [_, range]) => {
    const steps = Math.ceil((range.max - range.min) / range.step) + 1;
    return total * steps;
  }, 1);

  if (totalCombinations > 10000) {
    warnings.push({
      field: 'parameters',
      message: `Parameter combinations will create ${totalCombinations.toLocaleString()} configurations. This may take very long to complete.`,
      severity: 'warning'
    });
  }

  return {
    isValid: errors.length === 0,
    errors,
    warnings
  };
};

/**
 * Format validation messages for display
 */
export const formatValidationMessage = (field: string, message: string): string => {
  // Convert camelCase field names to readable format
  const readableField = field
    .replace(/([A-Z])/g, ' $1')
    .replace(/^./, str => str.toUpperCase())
    .replace(/\[(\d+)\]/g, ' #$1');

  return `${readableField}: ${message}`;
};

/**
 * Get validation severity color
 */
export const getValidationSeverityColor = (severity: ValidationError['severity']): string => {
  switch (severity) {
    case 'error':
      return '#f44336'; // Red
    case 'warning':
      return '#ff9800'; // Orange
    case 'info':
      return '#2196f3'; // Blue
    default:
      return '#757575'; // Grey
  }
};