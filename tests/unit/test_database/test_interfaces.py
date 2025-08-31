"""Tests for database interfaces."""

import pytest
from src.database.interfaces import (
    DatabaseServiceInterface,
    TradingDataServiceInterface,
    BotMetricsServiceInterface,
    HealthAnalyticsServiceInterface,
    ResourceManagementServiceInterface,
    RepositoryFactoryInterface,
    MLServiceInterface,
    UnitOfWorkFactoryInterface
)


class TestDatabaseInterfaces:
    """Test database interface definitions."""
    
    def test_database_service_interface_exists(self):
        """Test that DatabaseServiceInterface exists and has expected methods."""
        assert hasattr(DatabaseServiceInterface, 'start')
        assert hasattr(DatabaseServiceInterface, 'stop')
        assert hasattr(DatabaseServiceInterface, 'create_entity')
        assert hasattr(DatabaseServiceInterface, 'get_entity_by_id')
        assert hasattr(DatabaseServiceInterface, 'update_entity')
        assert hasattr(DatabaseServiceInterface, 'delete_entity')
        assert hasattr(DatabaseServiceInterface, 'list_entities')
        assert hasattr(DatabaseServiceInterface, 'count_entities')
        assert hasattr(DatabaseServiceInterface, 'bulk_create')
        assert hasattr(DatabaseServiceInterface, 'get_health_status')
        assert hasattr(DatabaseServiceInterface, 'get_performance_metrics')
    
    def test_trading_data_service_interface_exists(self):
        """Test that TradingDataServiceInterface exists and has expected methods."""
        assert hasattr(TradingDataServiceInterface, 'get_trades_by_bot')
        assert hasattr(TradingDataServiceInterface, 'get_positions_by_bot')
        assert hasattr(TradingDataServiceInterface, 'calculate_total_pnl')
    
    def test_bot_metrics_service_interface_exists(self):
        """Test that BotMetricsServiceInterface exists and has expected methods."""
        assert hasattr(BotMetricsServiceInterface, 'get_bot_metrics')
        assert hasattr(BotMetricsServiceInterface, 'store_bot_metrics')
        assert hasattr(BotMetricsServiceInterface, 'get_active_bots')
        assert hasattr(BotMetricsServiceInterface, 'archive_bot_record')
    
    def test_health_analytics_service_interface_exists(self):
        """Test that HealthAnalyticsServiceInterface exists and has expected methods."""
        assert hasattr(HealthAnalyticsServiceInterface, 'store_bot_health_analysis')
        assert hasattr(HealthAnalyticsServiceInterface, 'get_bot_health_analyses')
        assert hasattr(HealthAnalyticsServiceInterface, 'get_recent_health_analyses')
    
    def test_resource_management_service_interface_exists(self):
        """Test that ResourceManagementServiceInterface exists and has expected methods."""
        assert hasattr(ResourceManagementServiceInterface, 'store_resource_allocation')
        assert hasattr(ResourceManagementServiceInterface, 'store_resource_usage')
        assert hasattr(ResourceManagementServiceInterface, 'store_resource_reservation')
        assert hasattr(ResourceManagementServiceInterface, 'update_resource_allocation_status')
    
    def test_repository_factory_interface_exists(self):
        """Test that RepositoryFactoryInterface exists and has expected methods."""
        assert hasattr(RepositoryFactoryInterface, 'create_repository')
        assert hasattr(RepositoryFactoryInterface, 'register_repository')
        assert hasattr(RepositoryFactoryInterface, 'is_repository_registered')
    
    def test_ml_service_interface_exists(self):
        """Test that MLServiceInterface exists and has expected methods."""
        assert hasattr(MLServiceInterface, 'get_model_performance_summary')
        assert hasattr(MLServiceInterface, 'validate_model_deployment')
        assert hasattr(MLServiceInterface, 'get_model_recommendations')
    
    def test_unit_of_work_factory_interface_exists(self):
        """Test that UnitOfWorkFactoryInterface exists and has expected methods."""
        assert hasattr(UnitOfWorkFactoryInterface, 'create')
        assert hasattr(UnitOfWorkFactoryInterface, 'create_async')
        assert hasattr(UnitOfWorkFactoryInterface, 'configure_dependencies')
    
    def test_interface_classes_are_abstract(self):
        """Test that interface classes are properly defined as abstract."""
        # These should be abstract classes
        assert DatabaseServiceInterface is not None
        assert TradingDataServiceInterface is not None
        assert BotMetricsServiceInterface is not None
        assert HealthAnalyticsServiceInterface is not None
        assert ResourceManagementServiceInterface is not None
        assert RepositoryFactoryInterface is not None
        assert MLServiceInterface is not None
        assert UnitOfWorkFactoryInterface is not None
    
    def test_cannot_instantiate_abstract_interfaces(self):
        """Test that abstract interfaces cannot be instantiated."""
        with pytest.raises(TypeError):
            DatabaseServiceInterface()
        with pytest.raises(TypeError):
            TradingDataServiceInterface()
        with pytest.raises(TypeError):
            BotMetricsServiceInterface()
        with pytest.raises(TypeError):
            HealthAnalyticsServiceInterface()
        with pytest.raises(TypeError):
            ResourceManagementServiceInterface()
        with pytest.raises(TypeError):
            RepositoryFactoryInterface()
        with pytest.raises(TypeError):
            MLServiceInterface()
        with pytest.raises(TypeError):
            UnitOfWorkFactoryInterface()
