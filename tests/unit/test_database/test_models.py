"""Tests for database models main entry point."""

import pytest
from src.database import models


class TestDatabaseModels:
    """Test database models module imports."""

    def test_import_audit_models(self):
        """Test that audit models can be imported."""
        assert hasattr(models, 'CapitalAuditLog')
        assert hasattr(models, 'ExecutionAuditLog') 
        assert hasattr(models, 'PerformanceAuditLog')
        assert hasattr(models, 'RiskAuditLog')

    def test_import_base_model(self):
        """Test that base model can be imported."""
        assert hasattr(models, 'Base')

    def test_import_bot_models(self):
        """Test that bot models can be imported."""
        assert hasattr(models, 'Bot')
        assert hasattr(models, 'BotLog')
        assert hasattr(models, 'Signal')
        assert hasattr(models, 'Strategy')
        assert hasattr(models, 'BotInstance')

    def test_import_capital_models(self):
        """Test that capital models can be imported."""
        assert hasattr(models, 'CapitalAllocationDB')
        assert hasattr(models, 'CurrencyExposureDB')
        assert hasattr(models, 'ExchangeAllocationDB')
        assert hasattr(models, 'FundFlowDB')

    def test_import_data_models(self):
        """Test that data models can be imported."""
        assert hasattr(models, 'DataPipelineRecord')
        assert hasattr(models, 'DataQualityRecord')
        assert hasattr(models, 'FeatureRecord')

    def test_import_market_data_models(self):
        """Test that market data models can be imported."""
        assert hasattr(models, 'MarketDataRecord')

    def test_import_ml_models(self):
        """Test that ML models can be imported."""
        assert hasattr(models, 'MLModelMetadata')
        assert hasattr(models, 'MLPrediction')
        assert hasattr(models, 'MLTrainingJob')

    def test_import_state_models(self):
        """Test that state models can be imported."""
        assert hasattr(models, 'StateBackup')
        assert hasattr(models, 'StateCheckpoint')
        assert hasattr(models, 'StateHistory')
        assert hasattr(models, 'StateMetadata')
        assert hasattr(models, 'StateSnapshot')

    def test_import_system_models(self):
        """Test that system models can be imported."""
        assert hasattr(models, 'Alert')
        assert hasattr(models, 'AuditLog')
        assert hasattr(models, 'BalanceSnapshot')
        assert hasattr(models, 'PerformanceMetrics')

    def test_import_trading_models(self):
        """Test that trading models can be imported."""
        assert hasattr(models, 'Order')
        assert hasattr(models, 'OrderFill')
        assert hasattr(models, 'Position')
        assert hasattr(models, 'Trade')

    def test_import_user_models(self):
        """Test that user models can be imported."""
        assert hasattr(models, 'User')

    def test_all_exports(self):
        """Test that __all__ contains all expected models."""
        expected_models = {
            'Alert', 'AlertRule', 'AuditLog', 'BalanceSnapshot', 'Base', 'Bot', 'BotInstance', 'BotLog',
            'CapitalAllocationDB', 'CapitalAuditLog', 'CurrencyExposureDB', 'DataPipelineRecord',
            'DataQualityRecord', 'EscalationPolicy', 'ExchangeAllocationDB', 'ExecutionAuditLog', 'FeatureRecord',
            'FundFlowDB', 'MLModelMetadata', 'MLPrediction', 'MLTrainingJob', 'MarketDataRecord',
            'Order', 'OrderFill', 'PerformanceAuditLog', 'PerformanceMetrics', 'Position',
            'RiskAuditLog', 'Signal', 'StateBackup', 'StateCheckpoint', 'StateHistory',
            'StateMetadata', 'StateSnapshot', 'Strategy', 'Trade', 'User',
            'AnalyticsOperationalMetrics', 'AnalyticsPortfolioMetrics', 'AnalyticsPositionMetrics',
            'AnalyticsRiskMetrics', 'AnalyticsStrategyMetrics'
        }
        
        assert set(models.__all__) == expected_models

    def test_model_availability(self):
        """Test that all models in __all__ are available."""
        for model_name in models.__all__:
            assert hasattr(models, model_name), f"Model {model_name} not available in models module"

    def test_model_classes_are_classes(self):
        """Test that imported models are actually classes."""
        # Test a few key models to ensure they're proper classes
        assert isinstance(models.Base, type)
        assert isinstance(models.User, type)
        assert isinstance(models.Bot, type)
        assert isinstance(models.Order, type)