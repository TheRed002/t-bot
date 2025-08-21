"""
Core type definitions for the trading bot framework.

This module re-exports all types from the types submodules for backward compatibility.
All new code should import from src.core.types directly, which will use the organized submodules.
"""

# Re-export everything from the organized submodules
from src.core.types.base import *  # noqa: F403
from src.core.types.bot import *  # noqa: F403
from src.core.types.data import *  # noqa: F403
from src.core.types.execution import *  # noqa: F403
from src.core.types.market import *  # noqa: F403
from src.core.types.risk import *  # noqa: F403
from src.core.types.strategy import *  # noqa: F403
from src.core.types.trading import *  # noqa: F403

# This ensures backward compatibility for any code still importing from src.core.types
# module directly
