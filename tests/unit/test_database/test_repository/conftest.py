"""Shared fixtures for repository tests."""

import pytest
from unittest.mock import AsyncMock, Mock
from sqlalchemy.ext.asyncio import AsyncSession


@pytest.fixture
def mock_session():
    """Create properly configured mock AsyncSession for repository testing."""
    session = AsyncMock(spec=AsyncSession)
    
    # Configure sync methods as regular mocks to avoid warnings
    session.delete = Mock()
    session.add = Mock()
    session.expunge = Mock()
    session.expunge_all = Mock()
    
    # Configure async methods as AsyncMocks
    session.merge = AsyncMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.execute = AsyncMock()
    session.close = AsyncMock()
    session.refresh = AsyncMock()
    
    # Configure query result mocks
    session.scalar = AsyncMock()
    session.scalars = AsyncMock()
    
    return session