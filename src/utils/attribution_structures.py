"""
Attribution Analysis Structure Utilities.

Provides standard data structures and empty templates for performance attribution analysis
to ensure consistency across attribution components.
"""

from typing import Any


def create_empty_attribution_structure() -> dict[str, Any]:
    """
    Create an empty attribution structure with all standard sections.

    Returns:
        Dictionary with empty attribution analysis structure
    """
    return {
        "symbol_attribution": {},
        "timing_attribution": {},
        "factor_attribution": {},
        "cost_attribution": {},
        "summary": {
            "total_return": 0.0,
            "alpha": 0.0,
            "beta": 0.0,
        },
    }


def create_empty_service_attribution_structure() -> dict[str, Any]:
    """
    Create an empty attribution structure for service layer interfaces.

    Returns:
        Dictionary with empty service attribution analysis structure
    """
    return {
        "asset_allocation": {},
        "security_selection": {},
        "timing_effects": {},
        "factor_attribution": {},
        "regime_analysis": {},
    }


def create_symbol_attribution_summary(
    total_pnl: float,
    total_trades: int,
    top_contributor: str | None = None,
    worst_performer: str | None = None,
) -> dict[str, Any]:
    """
    Create symbol attribution summary structure.

    Args:
        total_pnl: Total profit and loss across all symbols
        total_trades: Total number of trades
        top_contributor: Symbol with highest contribution (optional)
        worst_performer: Symbol with worst performance (optional)

    Returns:
        Dictionary with symbol attribution summary
    """
    return {
        "total_pnl": total_pnl,
        "total_trades": total_trades,
        "top_contributor": top_contributor,
        "worst_performer": worst_performer,
    }


def create_attribution_summary(
    total_return: float = 0.0,
    alpha: float = 0.0,
    beta: float = 1.0,
    sharpe_ratio: float = 0.0,
    information_ratio: float = 0.0,
) -> dict[str, Any]:
    """
    Create attribution summary structure with key metrics.

    Args:
        total_return: Total strategy return
        alpha: Alpha (excess return)
        beta: Beta (market exposure)
        sharpe_ratio: Risk-adjusted return ratio
        information_ratio: Information ratio

    Returns:
        Dictionary with attribution summary
    """
    return {
        "total_return": total_return,
        "alpha": alpha,
        "beta": beta,
        "sharpe_ratio": sharpe_ratio,
        "information_ratio": information_ratio,
    }
