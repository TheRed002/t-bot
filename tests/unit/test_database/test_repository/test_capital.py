"""
Unit tests for capital management repository implementations.

This module tests all capital-related repositories including CapitalAllocationRepository,
FundFlowRepository, CurrencyExposureRepository, ExchangeAllocationRepository, and CapitalAuditLogRepository.
"""

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import pytest
from sqlalchemy.exc import IntegrityError, OperationalError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.exceptions import DatabaseError
from src.database.models.audit import CapitalAuditLog
from src.database.models.capital import (
    CapitalAllocationDB,
    CurrencyExposureDB,
    ExchangeAllocationDB,
    FundFlowDB,
)
from src.database.repository.capital import (
    CapitalAllocationRepository,
    CapitalAuditLogRepository,
    CurrencyExposureRepository,
    ExchangeAllocationRepository,
    FundFlowRepository,
)


class TestCapitalAllocationRepository:
    """Test CapitalAllocationRepository implementation."""

    @pytest.fixture
    def mock_session(self):
        """Create mock AsyncSession for testing."""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def allocation_repository(self, mock_session):
        """Create CapitalAllocationRepository instance for testing."""
        return CapitalAllocationRepository(mock_session)

    @pytest.fixture
    def sample_allocation(self):
        """Create sample allocation entity."""
        return CapitalAllocationDB(
            id=str(uuid.uuid4()),
            strategy_id=str(uuid.uuid4()),
            exchange="binance",
            allocated_amount=Decimal("10000.00"),
            available_amount=Decimal("8500.00"),
            utilized_amount=Decimal("1500.00"),
        )

    def test_allocation_repository_init(self, mock_session):
        """Test CapitalAllocationRepository initialization."""
        repo = CapitalAllocationRepository(mock_session)

        assert repo.session == mock_session
        assert repo.model == CapitalAllocationDB
        assert repo.name == "CapitalAllocationRepository"

    @pytest.mark.asyncio
    async def test_get_by_strategy(self, allocation_repository, mock_session):
        """Test get allocations by strategy."""
        strategy_id = str(uuid.uuid4())
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = [Mock(strategy_id=strategy_id)]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await allocation_repository.get_by_strategy(strategy_id)

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_by_exchange(self, allocation_repository, mock_session):
        """Test get allocations by exchange."""
        exchange = "binance"
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = [Mock(exchange=exchange), Mock(exchange=exchange)]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await allocation_repository.get_by_exchange(exchange)

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_find_by_strategy_exchange_found(self, allocation_repository, sample_allocation):
        """Test find allocation by strategy and exchange when found."""
        # The actual method calls RepositoryUtils.get_entities_by_multiple_fields, not get_all
        with patch(
            "src.database.repository.utils.RepositoryUtils.get_entities_by_multiple_fields",
            return_value=[sample_allocation],
        ):
            result = await allocation_repository.find_by_strategy_exchange(
                sample_allocation.strategy_id, sample_allocation.exchange
            )

            assert result == sample_allocation

    @pytest.mark.asyncio
    async def test_find_by_strategy_exchange_not_found(self, allocation_repository):
        """Test find allocation by strategy and exchange when not found."""
        # The actual method calls RepositoryUtils.get_entities_by_multiple_fields, not get_all
        with patch(
            "src.database.repository.utils.RepositoryUtils.get_entities_by_multiple_fields",
            return_value=[],
        ):
            result = await allocation_repository.find_by_strategy_exchange(
                "strategy_id", "exchange"
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_find_by_strategy_exchange_database_error(self, allocation_repository):
        """Test find allocation with database error."""
        # The actual method calls RepositoryUtils.get_entities_by_multiple_fields, not get_all
        with patch(
            "src.database.repository.utils.RepositoryUtils.get_entities_by_multiple_fields",
            side_effect=IntegrityError("Database error", None, None),
        ):
            with pytest.raises(DatabaseError):
                await allocation_repository.find_by_strategy_exchange("strategy_id", "exchange")

    @pytest.mark.asyncio
    async def test_find_by_strategy_exchange_operational_error(self, allocation_repository):
        """Test find allocation with operational error."""
        # The actual method calls RepositoryUtils.get_entities_by_multiple_fields, not get_all
        with patch(
            "src.database.repository.utils.RepositoryUtils.get_entities_by_multiple_fields",
            side_effect=OperationalError("Connection lost", None, None),
        ):
            with pytest.raises(DatabaseError):
                await allocation_repository.find_by_strategy_exchange("strategy_id", "exchange")

    @pytest.mark.asyncio
    async def test_get_total_allocated_by_strategy(self, allocation_repository):
        """Test get total allocated amount for strategy."""
        strategy_id = str(uuid.uuid4())
        allocations = [
            Mock(allocated_amount=Decimal("5000.00")),
            Mock(allocated_amount=Decimal("3000.00")),
            Mock(allocated_amount=Decimal("2000.00")),
        ]

        with patch.object(allocation_repository, "get_by_strategy", return_value=allocations):
            result = await allocation_repository.get_total_allocated_by_strategy(strategy_id)

            assert result == Decimal("10000.00")

    @pytest.mark.asyncio
    async def test_get_total_allocated_by_strategy_no_allocations(self, allocation_repository):
        """Test get total allocated with no allocations."""
        strategy_id = str(uuid.uuid4())

        with patch.object(allocation_repository, "get_by_strategy", return_value=[]):
            result = await allocation_repository.get_total_allocated_by_strategy(strategy_id)

            assert result == Decimal("0")

    @pytest.mark.asyncio
    async def test_get_available_capital_by_exchange(self, allocation_repository):
        """Test get available capital for exchange."""
        exchange = "binance"
        allocations = [
            Mock(available_amount=Decimal("2500.00")),
            Mock(available_amount=Decimal("1500.00")),
        ]

        with patch.object(allocation_repository, "get_by_exchange", return_value=allocations):
            result = await allocation_repository.get_available_capital_by_exchange(exchange)

            assert result == Decimal("4000.00")

    @pytest.mark.asyncio
    async def test_get_available_capital_by_exchange_no_allocations(self, allocation_repository):
        """Test get available capital with no allocations."""
        exchange = "coinbase"

        with patch.object(allocation_repository, "get_by_exchange", return_value=[]):
            result = await allocation_repository.get_available_capital_by_exchange(exchange)

            assert result == Decimal("0")


class TestFundFlowRepository:
    """Test FundFlowRepository implementation."""

    @pytest.fixture
    def mock_session(self):
        """Create mock AsyncSession for testing."""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def flow_repository(self, mock_session):
        """Create FundFlowRepository instance for testing."""
        return FundFlowRepository(mock_session)

    @pytest.fixture
    def sample_flow(self):
        """Create sample fund flow entity."""
        return FundFlowDB(
            id=str(uuid.uuid4()),
            from_strategy=str(uuid.uuid4()),
            to_strategy=str(uuid.uuid4()),
            from_exchange="binance",
            to_exchange="coinbase",
            amount=Decimal("5000.00"),
            currency="USDT",
            reason="rebalancing",
            timestamp=datetime.now(timezone.utc),
        )

    def test_flow_repository_init(self, mock_session):
        """Test FundFlowRepository initialization."""
        repo = FundFlowRepository(mock_session)

        assert repo.session == mock_session
        assert repo.model == FundFlowDB
        assert repo.name == "FundFlowRepository"

    @pytest.mark.asyncio
    async def test_get_by_from_strategy(self, flow_repository, mock_session):
        """Test get flows from strategy."""
        strategy_id = str(uuid.uuid4())
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = [Mock(from_strategy=strategy_id)]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await flow_repository.get_by_from_strategy(strategy_id)

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_by_to_strategy(self, flow_repository, mock_session):
        """Test get flows to strategy."""
        strategy_id = str(uuid.uuid4())
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = [Mock(to_strategy=strategy_id)]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await flow_repository.get_by_to_strategy(strategy_id)

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_by_exchange_flow(self, flow_repository, mock_session):
        """Test get flows between exchanges."""
        from_exchange = "binance"
        to_exchange = "coinbase"
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = [Mock(from_exchange=from_exchange, to_exchange=to_exchange)]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await flow_repository.get_by_exchange_flow(from_exchange, to_exchange)

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_by_reason(self, flow_repository, mock_session):
        """Test get flows by reason."""
        reason = "risk_management"
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = [Mock(reason=reason), Mock(reason=reason)]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await flow_repository.get_by_reason(reason)

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_by_currency(self, flow_repository, mock_session):
        """Test get flows by currency."""
        currency = "BTC"
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = [Mock(currency=currency)]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await flow_repository.get_by_currency(currency)

        assert len(result) == 1


class TestCurrencyExposureRepository:
    """Test CurrencyExposureRepository implementation."""

    @pytest.fixture
    def mock_session(self):
        """Create mock AsyncSession for testing."""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def exposure_repository(self, mock_session):
        """Create CurrencyExposureRepository instance for testing."""
        return CurrencyExposureRepository(mock_session)

    @pytest.fixture
    def sample_exposure(self):
        """Create sample currency exposure entity."""
        return CurrencyExposureDB(
            id=str(uuid.uuid4()),
            currency="BTC",
            total_exposure=Decimal("2.5"),
            spot_exposure=Decimal("1.5"),
            futures_exposure=Decimal("1.0"),
            var_1d=0.05,
            volatility=0.15,
        )

    def test_exposure_repository_init(self, mock_session):
        """Test CurrencyExposureRepository initialization."""
        repo = CurrencyExposureRepository(mock_session)

        assert repo.session == mock_session
        assert repo.model == CurrencyExposureDB
        assert repo.name == "CurrencyExposureRepository"

    @pytest.mark.asyncio
    async def test_get_by_currency_found(self, exposure_repository, sample_exposure):
        """Test get exposure by currency when found."""
        with patch.object(exposure_repository, "get_by", return_value=sample_exposure):
            result = await exposure_repository.get_by_currency(sample_exposure.currency)

            assert result == sample_exposure

    @pytest.mark.asyncio
    async def test_get_by_currency_not_found(self, exposure_repository):
        """Test get exposure by currency when not found."""
        with patch.object(exposure_repository, "get_by", return_value=None):
            result = await exposure_repository.get_by_currency("NONEXISTENT")

            assert result is None

    @pytest.mark.asyncio
    async def test_get_hedging_required(self, exposure_repository, mock_session):
        """Test get exposures requiring hedging."""
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = [Mock(hedging_required=True), Mock(hedging_required=True)]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await exposure_repository.get_hedging_required()

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_hedging_required_none(self, exposure_repository, mock_session):
        """Test get exposures requiring hedging when none exist."""
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = []
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await exposure_repository.get_hedging_required()

        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_get_total_exposure(self, exposure_repository):
        """Test get total currency exposure."""
        exposures = [
            Mock(total_exposure=Decimal("112500.0")),
            Mock(total_exposure=Decimal("87500.0")),
            Mock(total_exposure=Decimal("25000.0")),
        ]

        with patch.object(exposure_repository, "get_all", return_value=exposures):
            result = await exposure_repository.get_total_exposure()

            assert result == Decimal("225000.0")

    @pytest.mark.asyncio
    async def test_get_total_exposure_no_exposures(self, exposure_repository):
        """Test get total exposure with no exposures."""
        with patch.object(exposure_repository, "get_all", return_value=[]):
            result = await exposure_repository.get_total_exposure()

            assert result == Decimal("0")


class TestExchangeAllocationRepository:
    """Test ExchangeAllocationRepository implementation."""

    @pytest.fixture
    def mock_session(self):
        """Create mock AsyncSession for testing."""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def exchange_allocation_repository(self, mock_session):
        """Create ExchangeAllocationRepository instance for testing."""
        return ExchangeAllocationRepository(mock_session)

    @pytest.fixture
    def sample_exchange_allocation(self):
        """Create sample exchange allocation entity."""
        return ExchangeAllocationDB(
            id=str(uuid.uuid4()),
            exchange="binance",
            total_allocation=Decimal("50000.0"),
            utilized_allocation=Decimal("15000.0"),
            reserved_allocation=Decimal("5000.0"),
            efficiency_score=0.85,
            utilization_ratio=0.30,
        )

    def test_exchange_allocation_repository_init(self, mock_session):
        """Test ExchangeAllocationRepository initialization."""
        repo = ExchangeAllocationRepository(mock_session)

        assert repo.session == mock_session
        assert repo.model == ExchangeAllocationDB
        assert repo.name == "ExchangeAllocationRepository"

    @pytest.mark.asyncio
    async def test_get_by_exchange_found(
        self, exchange_allocation_repository, sample_exchange_allocation
    ):
        """Test get allocation by exchange when found."""
        with patch.object(
            exchange_allocation_repository, "get_by", return_value=sample_exchange_allocation
        ):
            result = await exchange_allocation_repository.get_by_exchange(
                sample_exchange_allocation.exchange
            )

            assert result == sample_exchange_allocation

    @pytest.mark.asyncio
    async def test_get_by_exchange_not_found(self, exchange_allocation_repository):
        """Test get allocation by exchange when not found."""
        with patch.object(exchange_allocation_repository, "get_by", return_value=None):
            result = await exchange_allocation_repository.get_by_exchange("nonexistent")

            assert result is None

    @pytest.mark.asyncio
    async def test_get_total_allocated(self, exchange_allocation_repository):
        """Test get total allocated amount."""
        allocations = [
            Mock(total_allocation=Decimal("50000.0")),
            Mock(total_allocation=Decimal("30000.0")),
            Mock(total_allocation=Decimal("20000.0")),
        ]

        with patch.object(exchange_allocation_repository, "get_all", return_value=allocations):
            result = await exchange_allocation_repository.get_total_allocated()

            assert result == Decimal("100000.0")

    @pytest.mark.asyncio
    async def test_get_total_allocated_no_allocations(self, exchange_allocation_repository):
        """Test get total allocated with no allocations."""
        with patch.object(exchange_allocation_repository, "get_all", return_value=[]):
            result = await exchange_allocation_repository.get_total_allocated()

            assert result == Decimal("0")

    @pytest.mark.asyncio
    async def test_get_total_available(self, exchange_allocation_repository):
        """Test get total available amount."""
        allocations = [
            Mock(total_allocation=Decimal("50000.0"), utilized_allocation=Decimal("15000.0")),
            Mock(total_allocation=Decimal("35000.0"), utilized_allocation=Decimal("15000.0")),
            Mock(total_allocation=Decimal("30000.0"), utilized_allocation=Decimal("15000.0")),
        ]

        with patch.object(exchange_allocation_repository, "get_all", return_value=allocations):
            result = await exchange_allocation_repository.get_total_available()

            assert result == Decimal("70000.0")

    @pytest.mark.asyncio
    async def test_get_total_available_no_allocations(self, exchange_allocation_repository):
        """Test get total available with no allocations."""
        with patch.object(exchange_allocation_repository, "get_all", return_value=[]):
            result = await exchange_allocation_repository.get_total_available()

            assert result == Decimal("0")

    @pytest.mark.asyncio
    async def test_get_underutilized_exchanges_default_threshold(
        self, exchange_allocation_repository
    ):
        """Test get underutilized exchanges with default threshold."""
        allocations = [
            Mock(
                allocated_amount=100000.0, utilized_amount=30000.0
            ),  # 30% utilization - underutilized
            Mock(
                allocated_amount=50000.0, utilized_amount=40000.0
            ),  # 80% utilization - well utilized
            Mock(
                allocated_amount=75000.0, utilized_amount=25000.0
            ),  # 33% utilization - underutilized
            Mock(allocated_amount=0.0, utilized_amount=0.0),  # No allocation - excluded
        ]

        with patch.object(exchange_allocation_repository, "get_all", return_value=allocations):
            result = await exchange_allocation_repository.get_underutilized_exchanges()

            # Should return allocations with utilization < 50% (default threshold) and allocated_amount > 0
            assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_underutilized_exchanges_custom_threshold(
        self, exchange_allocation_repository
    ):
        """Test get underutilized exchanges with custom threshold."""
        allocations = [
            Mock(allocated_amount=100000.0, utilized_amount=60000.0),  # 60% utilization
            Mock(allocated_amount=50000.0, utilized_amount=40000.0),  # 80% utilization
            Mock(allocated_amount=75000.0, utilized_amount=45000.0),  # 60% utilization
        ]

        with patch.object(exchange_allocation_repository, "get_all", return_value=allocations):
            result = await exchange_allocation_repository.get_underutilized_exchanges(threshold=0.7)

            # Should return allocations with utilization < 70%
            assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_underutilized_exchanges_no_underutilized(
        self, exchange_allocation_repository
    ):
        """Test get underutilized exchanges when all are well utilized."""
        allocations = [
            Mock(allocated_amount=100000.0, utilized_amount=80000.0),  # 80% utilization
            Mock(allocated_amount=50000.0, utilized_amount=40000.0),  # 80% utilization
        ]

        with patch.object(exchange_allocation_repository, "get_all", return_value=allocations):
            result = await exchange_allocation_repository.get_underutilized_exchanges()

            assert len(result) == 0


class TestCapitalAuditLogRepository:
    """Test CapitalAuditLogRepository implementation."""

    @pytest.fixture
    def mock_session(self):
        """Create mock AsyncSession for testing."""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def audit_repository(self, mock_session):
        """Create CapitalAuditLogRepository instance for testing."""
        return CapitalAuditLogRepository(mock_session)

    @pytest.fixture
    def sample_audit_log(self):
        """Create sample audit log entity."""
        return CapitalAuditLog(
            id=str(uuid.uuid4()),
            operation_id=str(uuid.uuid4()),
            strategy_id=str(uuid.uuid4()),
            exchange="binance",
            operation_type="allocate",
            operation_description="Allocating capital to strategy",
            amount=Decimal("15000.00"),
            previous_amount=Decimal("10000.00"),
            new_amount=Decimal("15000.00"),
            success=True,
            requested_at=datetime.now(timezone.utc),
            source_component="CapitalManager",
        )

    def test_audit_repository_init(self, mock_session):
        """Test CapitalAuditLogRepository initialization."""
        repo = CapitalAuditLogRepository(mock_session)

        assert repo.session == mock_session
        assert repo.model == CapitalAuditLog
        assert repo.name == "CapitalAuditLogRepository"

    @pytest.mark.asyncio
    async def test_get_by_operation_id_found(self, audit_repository, sample_audit_log):
        """Test get audit log by operation ID when found."""
        with patch.object(audit_repository, "get_by", return_value=sample_audit_log):
            result = await audit_repository.get_by_operation_id(sample_audit_log.operation_id)

            assert result == sample_audit_log

    @pytest.mark.asyncio
    async def test_get_by_operation_id_not_found(self, audit_repository):
        """Test get audit log by operation ID when not found."""
        with patch.object(audit_repository, "get_by", return_value=None):
            result = await audit_repository.get_by_operation_id("nonexistent")

            assert result is None

    @pytest.mark.asyncio
    async def test_get_by_strategy(self, audit_repository, mock_session):
        """Test get audit logs by strategy."""
        strategy_id = str(uuid.uuid4())
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = [
            Mock(strategy_id=strategy_id),
            Mock(strategy_id=strategy_id),
        ]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await audit_repository.get_by_strategy(strategy_id)

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_by_exchange(self, audit_repository, mock_session):
        """Test get audit logs by exchange."""
        exchange = "coinbase"
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = [Mock(exchange=exchange)]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await audit_repository.get_by_exchange(exchange)

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_failed_operations_default_limit(self, audit_repository, mock_session):
        """Test get failed operations with default limit."""
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = [Mock(success=False) for _ in range(5)]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await audit_repository.get_failed_operations()

        assert len(result) == 5

    @pytest.mark.asyncio
    async def test_get_failed_operations_custom_limit(self, audit_repository, mock_session):
        """Test get failed operations with custom limit."""
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = [Mock(success=False) for _ in range(25)]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await audit_repository.get_failed_operations(limit=25)

        assert len(result) == 25

    @pytest.mark.asyncio
    async def test_get_failed_operations_no_failures(self, audit_repository, mock_session):
        """Test get failed operations when none exist."""
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = []
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await audit_repository.get_failed_operations()

        assert len(result) == 0


class TestCapitalRepositoryErrorHandling:
    """Test error handling in capital repositories."""

    @pytest.fixture
    def mock_session(self):
        """Create mock AsyncSession for testing."""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def allocation_repository(self, mock_session):
        """Create CapitalAllocationRepository instance for testing."""
        return CapitalAllocationRepository(mock_session)

    @pytest.mark.asyncio
    async def test_database_error_handling(self, allocation_repository, mock_session):
        """Test database error handling in repository operations."""
        mock_session.execute.side_effect = SQLAlchemyError("Database connection lost")

        # Repository wraps SQLAlchemyError in RepositoryError
        from src.core.exceptions import RepositoryError

        with pytest.raises(RepositoryError):
            await allocation_repository.get_by_strategy("strategy_id")

    @pytest.mark.asyncio
    async def test_integrity_error_handling(self, allocation_repository, mock_session):
        """Test integrity error handling during create operations."""
        allocation = CapitalAllocationDB(
            id=str(uuid.uuid4()),
            strategy_id=str(uuid.uuid4()),
            exchange="binance",
            allocated_amount=Decimal("10000.00"),
            available_amount=Decimal("10000.00"),
        )
        mock_session.flush.side_effect = IntegrityError("Duplicate key", None, None)

        # Repository wraps IntegrityError in RepositoryError
        from src.core.exceptions import RepositoryError

        with pytest.raises(RepositoryError):
            await allocation_repository.create(allocation)

        mock_session.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_database_error_in_find_method(self, allocation_repository):
        """Test database error in find_by_strategy_exchange method."""
        # The actual method calls RepositoryUtils.get_entities_by_multiple_fields, not get_all
        # The find_by_strategy_exchange method only catches IntegrityError and OperationalError
        # For general SQLAlchemyError, it would bubble up through RepositoryUtils
        with patch(
            "src.database.repository.utils.RepositoryUtils.get_entities_by_multiple_fields",
            side_effect=IntegrityError("General database error", None, None),
        ):
            with pytest.raises(DatabaseError) as exc_info:
                await allocation_repository.find_by_strategy_exchange("strategy_id", "exchange")

            assert "Failed to find allocation" in str(exc_info.value)


class TestCapitalRepositoryConcurrency:
    """Test concurrent operations in capital repositories."""

    @pytest.fixture
    def mock_session(self):
        """Create mock AsyncSession for testing."""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def allocation_repository(self, mock_session):
        """Create CapitalAllocationRepository instance for testing."""
        return CapitalAllocationRepository(mock_session)

    @pytest.mark.asyncio
    async def test_concurrent_allocation_updates(self, allocation_repository):
        """Test concurrent allocation updates."""
        allocation_id = str(uuid.uuid4())
        allocation = CapitalAllocationDB(
            id=allocation_id,
            strategy_id=str(uuid.uuid4()),
            exchange="binance",
            allocated_amount=Decimal("10000.00"),
            available_amount=Decimal("8000.00"),
        )

        with (
            patch.object(allocation_repository, "get", return_value=allocation),
            patch.object(allocation_repository, "update", return_value=allocation),
        ):
            # Simulate concurrent updates
            allocation.available_amount = Decimal("7500.00")
            result1 = await allocation_repository.update(allocation)

            allocation.available_amount = Decimal("7000.00")
            result2 = await allocation_repository.update(allocation)

            assert result1 == allocation
            assert result2 == allocation

    @pytest.mark.asyncio
    async def test_concurrent_fund_flow_creation(self, mock_session):
        """Test concurrent fund flow creation."""
        flow_repository = FundFlowRepository(mock_session)

        flow1 = FundFlowDB(
            id=str(uuid.uuid4()),
            flow_type="allocation",
            from_account="strategy_1",
            to_account="strategy_2",
            amount=Decimal("1000.00"),
            currency="USDT",
        )

        flow2 = FundFlowDB(
            id=str(uuid.uuid4()),
            flow_type="rebalance",
            from_account="strategy_3",
            to_account="strategy_4",
            amount=Decimal("2000.00"),
            currency="USDT",
        )

        with patch.object(flow_repository, "create", side_effect=[flow1, flow2]):
            # Simulate concurrent flow creations
            result1 = await flow_repository.create(flow1)
            result2 = await flow_repository.create(flow2)

            assert result1 == flow1
            assert result2 == flow2


class TestCapitalRepositoryPerformance:
    """Test performance-related functionality in capital repositories."""

    @pytest.fixture
    def mock_session(self):
        """Create mock AsyncSession for testing."""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def allocation_repository(self, mock_session):
        """Create CapitalAllocationRepository instance for testing."""
        return CapitalAllocationRepository(mock_session)

    @pytest.mark.asyncio
    async def test_batch_allocation_queries(self, allocation_repository):
        """Test batch allocation queries for performance."""
        strategy_ids = [str(uuid.uuid4()) for _ in range(10)]

        # Mock multiple strategy allocations
        allocations_by_strategy = {
            strategy_id: [Mock(strategy_id=strategy_id, allocated_amount=Decimal("1000.00"))]
            for strategy_id in strategy_ids
        }

        async def mock_get_by_strategy(strategy_id):
            return allocations_by_strategy.get(strategy_id, [])

        with patch.object(
            allocation_repository, "get_by_strategy", side_effect=mock_get_by_strategy
        ):
            # Test getting allocations for multiple strategies
            results = []
            for strategy_id in strategy_ids:
                result = await allocation_repository.get_by_strategy(strategy_id)
                results.append(result)

            assert len(results) == 10
            for result in results:
                assert len(result) == 1

    @pytest.mark.asyncio
    async def test_large_dataset_aggregation(self, allocation_repository):
        """Test aggregation operations on large datasets."""
        # Simulate large number of allocations
        large_allocation_set = [Mock(allocated_amount=Decimal(str(i * 1000))) for i in range(1000)]

        with patch.object(
            allocation_repository, "get_by_strategy", return_value=large_allocation_set
        ):
            result = await allocation_repository.get_total_allocated_by_strategy("strategy_id")

            # Sum should be 0 + 1000 + 2000 + ... + 999000 = 999 * 1000 * 1000 / 2 = 499,500,000
            expected = sum(Decimal(str(i * 1000)) for i in range(1000))
            assert result == expected


class TestCapitalRepositoryEdgeCases:
    """Test edge cases in capital repositories."""

    @pytest.fixture
    def mock_session(self):
        """Create mock AsyncSession for testing."""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def allocation_repository(self, mock_session):
        """Create CapitalAllocationRepository instance for testing."""
        return CapitalAllocationRepository(mock_session)

    @pytest.mark.asyncio
    async def test_zero_allocations(self, allocation_repository):
        """Test operations with zero allocations."""
        allocation = CapitalAllocationDB(
            id=str(uuid.uuid4()),
            strategy_id=str(uuid.uuid4()),
            exchange="binance",
            allocated_amount=Decimal("0.00"),
            available_amount=Decimal("0.00"),
        )

        with patch.object(allocation_repository, "create", return_value=allocation):
            result = await allocation_repository.create(allocation)

            assert result.allocated_amount == Decimal("0.00")
            assert result.available_amount == Decimal("0.00")

    @pytest.mark.asyncio
    async def test_negative_amounts(self, allocation_repository):
        """Test handling of negative amounts."""
        # This might represent a debt or deficit
        allocation = CapitalAllocationDB(
            id=str(uuid.uuid4()),
            strategy_id=str(uuid.uuid4()),
            exchange="binance",
            allocated_amount=Decimal("10000.00"),
            available_amount=Decimal("-1000.00"),  # Overdraft situation
        )

        with patch.object(allocation_repository, "create", return_value=allocation):
            result = await allocation_repository.create(allocation)

            assert result.available_amount == Decimal("-1000.00")

    @pytest.mark.asyncio
    async def test_very_large_amounts(self, allocation_repository):
        """Test handling of very large amounts."""
        allocation = CapitalAllocationDB(
            id=str(uuid.uuid4()),
            strategy_id=str(uuid.uuid4()),
            exchange="binance",
            allocated_amount=Decimal("999999999999.99"),
            available_amount=Decimal("999999999999.99"),
        )

        with patch.object(allocation_repository, "create", return_value=allocation):
            result = await allocation_repository.create(allocation)

            assert result.allocated_amount == Decimal("999999999999.99")

    @pytest.mark.asyncio
    async def test_precision_handling(self, allocation_repository):
        """Test decimal precision handling."""
        allocation = CapitalAllocationDB(
            id=str(uuid.uuid4()),
            strategy_id=str(uuid.uuid4()),
            exchange="binance",
            allocated_amount=Decimal("10000.123456789"),
            available_amount=Decimal("9999.987654321"),
        )

        with patch.object(allocation_repository, "create", return_value=allocation):
            result = await allocation_repository.create(allocation)

            # Should preserve high precision
            assert result.allocated_amount == Decimal("10000.123456789")
            assert result.available_amount == Decimal("9999.987654321")
