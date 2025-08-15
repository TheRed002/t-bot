/**
 * Tests for playground validation utilities
 */

import {
  validatePlaygroundConfiguration,
  validateExecutionSettings,
  validateBatchSettings,
  validateParameterRanges,
  formatValidationMessage,
  getValidationSeverityColor
} from '../playgroundValidation';
import { PlaygroundConfiguration } from '@/types';

describe('playgroundValidation', () => {
  describe('validatePlaygroundConfiguration', () => {
    const validConfiguration: PlaygroundConfiguration = {
      name: 'Test Configuration',
      description: 'Test description',
      symbols: ['BTC/USDT', 'ETH/USDT'],
      positionSizing: {
        type: 'percentage',
        value: 2,
        maxPositions: 5
      },
      tradingSide: 'both',
      riskSettings: {
        stopLossPercentage: 2,
        takeProfitPercentage: 4,
        maxDrawdownPercentage: 10,
        maxRiskPerTrade: 2
      },
      portfolioSettings: {
        maxPositions: 5,
        allocationStrategy: 'equal_weight',
        rebalanceFrequency: 'daily'
      },
      strategy: {
        type: 'trend_following',
        parameters: {}
      },
      timeframe: '1h'
    };

    it('validates a correct configuration', () => {
      const result = validatePlaygroundConfiguration(validConfiguration);
      expect(result.isValid).toBe(true);
      expect(result.errors).toHaveLength(0);
    });

    it('requires configuration name', () => {
      const config = { ...validConfiguration, name: '' };
      const result = validatePlaygroundConfiguration(config);
      
      expect(result.isValid).toBe(false);
      expect(result.errors).toContain(
        expect.objectContaining({
          field: 'name',
          message: 'Configuration name is required',
          severity: 'error'
        })
      );
    });

    it('validates configuration name length', () => {
      const config = { ...validConfiguration, name: 'a'.repeat(101) };
      const result = validatePlaygroundConfiguration(config);
      
      expect(result.isValid).toBe(false);
      expect(result.errors).toContain(
        expect.objectContaining({
          field: 'name',
          message: 'Configuration name must be less than 100 characters'
        })
      );
    });

    it('requires at least one symbol', () => {
      const config = { ...validConfiguration, symbols: [] };
      const result = validatePlaygroundConfiguration(config);
      
      expect(result.isValid).toBe(false);
      expect(result.errors).toContain(
        expect.objectContaining({
          field: 'symbols',
          message: 'At least one trading symbol must be selected'
        })
      );
    });

    it('validates symbol format', () => {
      const config = { ...validConfiguration, symbols: ['INVALID_SYMBOL', 'BTC/USDT'] };
      const result = validatePlaygroundConfiguration(config);
      
      expect(result.isValid).toBe(false);
      expect(result.errors).toContain(
        expect.objectContaining({
          field: 'symbols[0]',
          message: 'Invalid symbol format: INVALID_SYMBOL. Expected format: BASE/QUOTE (e.g., BTC/USDT)'
        })
      );
    });

    it('warns about too many symbols', () => {
      const symbols = Array.from({ length: 15 }, (_, i) => `SYM${i}/USDT`);
      const config = { ...validConfiguration, symbols };
      const result = validatePlaygroundConfiguration(config);
      
      expect(result.warnings).toContain(
        expect.objectContaining({
          field: 'symbols',
          message: 'Trading more than 10 symbols may impact performance'
        })
      );
    });

    it('validates position sizing value', () => {
      const config = {
        ...validConfiguration,
        positionSizing: { ...validConfiguration.positionSizing, value: 0 }
      };
      const result = validatePlaygroundConfiguration(config);
      
      expect(result.isValid).toBe(false);
      expect(result.errors).toContain(
        expect.objectContaining({
          field: 'positionSizing.value',
          message: 'Position size must be greater than 0'
        })
      );
    });

    it('warns about high percentage position sizing', () => {
      const config = {
        ...validConfiguration,
        positionSizing: { ...validConfiguration.positionSizing, value: 60 }
      };
      const result = validatePlaygroundConfiguration(config);
      
      expect(result.warnings).toContain(
        expect.objectContaining({
          field: 'positionSizing.value',
          message: 'Position size greater than 50% of portfolio is high risk'
        })
      );
    });

    it('validates stop loss percentage', () => {
      const config = {
        ...validConfiguration,
        riskSettings: { ...validConfiguration.riskSettings, stopLossPercentage: 0 }
      };
      const result = validatePlaygroundConfiguration(config);
      
      expect(result.isValid).toBe(false);
      expect(result.errors).toContain(
        expect.objectContaining({
          field: 'riskSettings.stopLossPercentage',
          message: 'Stop loss percentage must be greater than 0'
        })
      );
    });

    it('validates risk/reward ratio', () => {
      const config = {
        ...validConfiguration,
        riskSettings: {
          ...validConfiguration.riskSettings,
          stopLossPercentage: 5,
          takeProfitPercentage: 3
        }
      };
      const result = validatePlaygroundConfiguration(config);
      
      expect(result.isValid).toBe(false);
      expect(result.errors).toContain(
        expect.objectContaining({
          field: 'riskSettings',
          message: 'Take profit must be greater than stop loss for positive risk/reward ratio'
        })
      );
    });

    it('warns about poor risk/reward ratio', () => {
      const config = {
        ...validConfiguration,
        riskSettings: {
          ...validConfiguration.riskSettings,
          stopLossPercentage: 3,
          takeProfitPercentage: 4
        }
      };
      const result = validatePlaygroundConfiguration(config);
      
      expect(result.warnings).toContain(
        expect.objectContaining({
          field: 'riskSettings',
          message: expect.stringContaining('Risk/reward ratio is 1.33. Consider targeting at least 1.5:1')
        })
      );
    });

    it('validates max drawdown range', () => {
      const config = {
        ...validConfiguration,
        riskSettings: { ...validConfiguration.riskSettings, maxDrawdownPercentage: 0 }
      };
      const result = validatePlaygroundConfiguration(config);
      
      expect(result.isValid).toBe(false);
      expect(result.errors).toContain(
        expect.objectContaining({
          field: 'riskSettings.maxDrawdownPercentage',
          message: 'Max drawdown percentage must be between 0 and 100'
        })
      );
    });

    it('requires strategy type', () => {
      const config = {
        ...validConfiguration,
        strategy: { ...validConfiguration.strategy, type: '' }
      };
      const result = validatePlaygroundConfiguration(config);
      
      expect(result.isValid).toBe(false);
      expect(result.errors).toContain(
        expect.objectContaining({
          field: 'strategy.type',
          message: 'Strategy type is required'
        })
      );
    });

    it('validates timeframe', () => {
      const config = { ...validConfiguration, timeframe: 'invalid' as any };
      const result = validatePlaygroundConfiguration(config);
      
      expect(result.isValid).toBe(false);
      expect(result.errors).toContain(
        expect.objectContaining({
          field: 'timeframe',
          message: expect.stringContaining('Invalid timeframe')
        })
      );
    });

    it('warns about ML strategy without model', () => {
      const config = {
        ...validConfiguration,
        strategy: { type: 'ml_based', parameters: {} }
      };
      const result = validatePlaygroundConfiguration(config);
      
      expect(result.warnings).toContain(
        expect.objectContaining({
          field: 'model',
          message: 'ML-based strategy selected but no model configured'
        })
      );
    });
  });

  describe('validateExecutionSettings', () => {
    const validSettings = {
      mode: 'historical' as const,
      startDate: '2023-01-01',
      endDate: '2023-02-01',
      initialBalance: 10000,
      commission: 0.1,
      speed: 1
    };

    it('validates correct execution settings', () => {
      const result = validateExecutionSettings(validSettings);
      expect(result.isValid).toBe(true);
      expect(result.errors).toHaveLength(0);
    });

    it('validates initial balance', () => {
      const settings = { ...validSettings, initialBalance: 0 };
      const result = validateExecutionSettings(settings);
      
      expect(result.isValid).toBe(false);
      expect(result.errors).toContain(
        expect.objectContaining({
          field: 'initialBalance',
          message: 'Initial balance must be greater than 0'
        })
      );
    });

    it('warns about small initial balance', () => {
      const settings = { ...validSettings, initialBalance: 50 };
      const result = validateExecutionSettings(settings);
      
      expect(result.warnings).toContain(
        expect.objectContaining({
          field: 'initialBalance',
          message: 'Very small initial balance may not be realistic for trading'
        })
      );
    });

    it('validates commission range', () => {
      const settings = { ...validSettings, commission: 6 };
      const result = validateExecutionSettings(settings);
      
      expect(result.isValid).toBe(false);
      expect(result.errors).toContain(
        expect.objectContaining({
          field: 'commission',
          message: 'Commission must be between 0% and 5%'
        })
      );
    });

    it('requires dates for historical mode', () => {
      const settings = { ...validSettings, startDate: undefined };
      const result = validateExecutionSettings(settings);
      
      expect(result.isValid).toBe(false);
      expect(result.errors).toContain(
        expect.objectContaining({
          field: 'startDate',
          message: 'Start date is required for historical backtesting'
        })
      );
    });

    it('validates date range', () => {
      const settings = {
        ...validSettings,
        startDate: '2023-02-01',
        endDate: '2023-01-01'
      };
      const result = validateExecutionSettings(settings);
      
      expect(result.isValid).toBe(false);
      expect(result.errors).toContain(
        expect.objectContaining({
          field: 'dateRange',
          message: 'End date must be after start date'
        })
      );
    });

    it('prevents future end dates', () => {
      const futureDate = new Date();
      futureDate.setDate(futureDate.getDate() + 1);
      
      const settings = {
        ...validSettings,
        endDate: futureDate.toISOString().split('T')[0]
      };
      const result = validateExecutionSettings(settings);
      
      expect(result.isValid).toBe(false);
      expect(result.errors).toContain(
        expect.objectContaining({
          field: 'endDate',
          message: 'End date cannot be in the future'
        })
      );
    });

    it('validates execution speed range', () => {
      const settings = { ...validSettings, speed: 0 };
      const result = validateExecutionSettings(settings);
      
      expect(result.isValid).toBe(false);
      expect(result.errors).toContain(
        expect.objectContaining({
          field: 'speed',
          message: 'Execution speed must be between 0.1x and 1000x'
        })
      );
    });

    it('warns about production mode', () => {
      const settings = { ...validSettings, mode: 'production' as const };
      const result = validateExecutionSettings(settings);
      
      expect(result.warnings).toContain(
        expect.objectContaining({
          field: 'mode',
          message: 'Production mode trades with real money. Ensure thorough testing first.'
        })
      );
    });
  });

  describe('validateBatchSettings', () => {
    const validSettings = {
      configurationCount: 10,
      maxConcurrentJobs: 5,
      resourceLimit: 70,
      overfittingProtection: true,
      crossValidationFolds: 5,
      outOfSamplePercentage: 20
    };

    it('validates correct batch settings', () => {
      const result = validateBatchSettings(validSettings);
      expect(result.isValid).toBe(true);
      expect(result.errors).toHaveLength(0);
    });

    it('validates configuration count minimum', () => {
      const settings = { ...validSettings, configurationCount: 0 };
      const result = validateBatchSettings(settings);
      
      expect(result.isValid).toBe(false);
      expect(result.errors).toContain(
        expect.objectContaining({
          field: 'configurationCount',
          message: 'Configuration count must be at least 1'
        })
      );
    });

    it('warns about large configuration counts', () => {
      const settings = { ...validSettings, configurationCount: 1500 };
      const result = validateBatchSettings(settings);
      
      expect(result.warnings).toContain(
        expect.objectContaining({
          field: 'configurationCount',
          message: 'Large number of configurations will take significant time to complete'
        })
      );
    });

    it('validates concurrent jobs range', () => {
      const settings = { ...validSettings, maxConcurrentJobs: 0 };
      const result = validateBatchSettings(settings);
      
      expect(result.isValid).toBe(false);
      expect(result.errors).toContain(
        expect.objectContaining({
          field: 'maxConcurrentJobs',
          message: 'Max concurrent jobs must be between 1 and 20'
        })
      );
    });

    it('validates resource limit range', () => {
      const settings = { ...validSettings, resourceLimit: 5 };
      const result = validateBatchSettings(settings);
      
      expect(result.isValid).toBe(false);
      expect(result.errors).toContain(
        expect.objectContaining({
          field: 'resourceLimit',
          message: 'Resource limit must be between 10% and 100%'
        })
      );
    });

    it('validates cross-validation folds', () => {
      const settings = { ...validSettings, crossValidationFolds: 2 };
      const result = validateBatchSettings(settings);
      
      expect(result.isValid).toBe(false);
      expect(result.errors).toContain(
        expect.objectContaining({
          field: 'crossValidationFolds',
          message: 'Cross-validation folds must be between 3 and 10'
        })
      );
    });

    it('warns about disabled overfitting protection', () => {
      const settings = { ...validSettings, overfittingProtection: false };
      const result = validateBatchSettings(settings);
      
      expect(result.warnings).toContain(
        expect.objectContaining({
          field: 'overfittingProtection',
          message: 'Overfitting protection is recommended for reliable results'
        })
      );
    });
  });

  describe('validateParameterRanges', () => {
    const validRanges = {
      stopLoss: { min: 1, max: 5, step: 0.1, enabled: true },
      takeProfit: { min: 2, max: 10, step: 0.5, enabled: true }
    };

    it('validates correct parameter ranges', () => {
      const result = validateParameterRanges(validRanges);
      expect(result.isValid).toBe(true);
      expect(result.errors).toHaveLength(0);
    });

    it('requires at least one enabled parameter', () => {
      const ranges = {
        stopLoss: { min: 1, max: 5, step: 0.1, enabled: false },
        takeProfit: { min: 2, max: 10, step: 0.5, enabled: false }
      };
      const result = validateParameterRanges(ranges);
      
      expect(result.isValid).toBe(false);
      expect(result.errors).toContain(
        expect.objectContaining({
          field: 'parameters',
          message: 'At least one parameter must be enabled for optimization'
        })
      );
    });

    it('validates min/max relationship', () => {
      const ranges = {
        ...validRanges,
        stopLoss: { min: 5, max: 1, step: 0.1, enabled: true }
      };
      const result = validateParameterRanges(ranges);
      
      expect(result.isValid).toBe(false);
      expect(result.errors).toContain(
        expect.objectContaining({
          field: 'stopLoss',
          message: 'stopLoss: Minimum value must be less than maximum value'
        })
      );
    });

    it('validates step size', () => {
      const ranges = {
        ...validRanges,
        stopLoss: { min: 1, max: 5, step: 0, enabled: true }
      };
      const result = validateParameterRanges(ranges);
      
      expect(result.isValid).toBe(false);
      expect(result.errors).toContain(
        expect.objectContaining({
          field: 'stopLoss',
          message: 'stopLoss: Step size must be greater than 0'
        })
      );
    });

    it('warns about too many combinations', () => {
      const ranges = {
        param1: { min: 0, max: 10, step: 0.01, enabled: true },
        param2: { min: 0, max: 10, step: 0.01, enabled: true }
      };
      const result = validateParameterRanges(ranges);
      
      expect(result.warnings).toContain(
        expect.objectContaining({
          field: 'parameters',
          message: expect.stringContaining('Parameter combinations will create')
        })
      );
    });
  });

  describe('formatValidationMessage', () => {
    it('formats simple field names', () => {
      const result = formatValidationMessage('name', 'is required');
      expect(result).toBe('Name: is required');
    });

    it('formats camelCase field names', () => {
      const result = formatValidationMessage('stopLossPercentage', 'must be positive');
      expect(result).toBe('Stop Loss Percentage: must be positive');
    });

    it('formats array indices', () => {
      const result = formatValidationMessage('symbols[0]', 'is invalid');
      expect(result).toBe('Symbols #0: is invalid');
    });

    it('formats nested field names', () => {
      const result = formatValidationMessage('riskSettings.maxDrawdown', 'is too high');
      expect(result).toBe('Risk Settings. Max Drawdown: is too high');
    });
  });

  describe('getValidationSeverityColor', () => {
    it('returns correct colors for each severity', () => {
      expect(getValidationSeverityColor('error')).toBe('#f44336');
      expect(getValidationSeverityColor('warning')).toBe('#ff9800');
      expect(getValidationSeverityColor('info')).toBe('#2196f3');
    });

    it('returns default color for unknown severity', () => {
      expect(getValidationSeverityColor('unknown' as any)).toBe('#757575');
    });
  });
});