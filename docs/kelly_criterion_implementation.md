# Kelly Criterion Implementation Documentation

## Overview

The Kelly Criterion is a mathematical formula used to determine optimal position sizing for maximizing long-term growth while minimizing the risk of ruin. This document describes the improved implementation in the T-Bot trading system.

## Formula

The Kelly Criterion formula implemented:

```
f = (p * b - q) / b
```

Where:
- `f` = fraction of capital to wager
- `p` = probability of winning
- `q` = probability of losing (1 - p)
- `b` = win/loss ratio (average win / average loss)

## Key Improvements

### 1. Half-Kelly Implementation
- **Safety Factor**: The system uses Half-Kelly (f * 0.5) instead of full Kelly
- **Rationale**: Full Kelly can be too aggressive for real trading, leading to large drawdowns
- **Implementation**: After calculating the Kelly fraction, we multiply by 0.5 for conservative sizing

### 2. Proper Bounds Enforcement
- **Minimum Position Size**: 1% of portfolio (increased from 0.1%)
- **Maximum Position Size**: 25% absolute maximum (enforced even if config allows higher)
- **Rationale**: Prevents both negligible positions and excessive concentration

### 3. Decimal Precision Throughout
- **All calculations use Python's Decimal type**
- **Prevents floating-point precision errors in financial calculations**
- **Critical for accurate position sizing with large portfolio values**

### 4. Win/Loss Probability Calculation
- **Correctly calculates win probability from historical returns**
- **Separates winning and losing trades**
- **Calculates average win and average loss magnitudes**
- **Computes win/loss ratio accurately**

## Edge Cases Handled

### Insufficient Data
- Falls back to fixed percentage sizing when less than 30 days of data
- Configurable lookback period via `kelly_lookback_days`

### All Winning or All Losing Trades
- Falls back to fixed percentage when no losing trades (can't calculate ratio)
- Falls back to fixed percentage when no winning trades

### Negative Edge
- When Kelly fraction is negative (losing strategy), uses minimum position size
- Prevents the system from suggesting short positions when not intended

### Near-Zero Variance
- Handles cases where average loss is very small (< 0.0001)
- Prevents division by zero errors

### Extreme Kelly Values
- Caps maximum position at 25% regardless of Kelly calculation
- Ensures minimum position of 1% for viable trades

## Configuration Parameters

```python
# In RiskConfig
kelly_lookback_days: int = 30  # Days of history required
kelly_max_fraction: float = 0.25  # Maximum Kelly fraction (before Half-Kelly)
```

## Usage Example

```python
from decimal import Decimal
from src.risk_management.position_sizing import PositionSizer
from src.core.types import PositionSizeMethod, Signal

# Initialize position sizer
position_sizer = PositionSizer(config)

# Update historical returns
await position_sizer.update_price_history("BTCUSDT", 50000.0)

# Calculate position size using Kelly
position_size = await position_sizer.calculate_position_size(
    signal=signal,
    portfolio_value=Decimal("10000"),
    method=PositionSizeMethod.KELLY_CRITERION
)
```

## Mathematical Example

Given:
- 60% win rate
- Average win: 2%
- Average loss: 1%
- Win/loss ratio: 2.0
- Signal confidence: 0.8

Calculation:
1. Full Kelly: (0.6 * 2 - 0.4) / 2 = 0.4 (40%)
2. Half Kelly: 0.4 * 0.5 = 0.2 (20%)
3. With confidence: 0.2 * 0.8 = 0.16 (16%)
4. Final position: 16% of portfolio (or capped at config max)

## Testing Coverage

The implementation includes comprehensive tests for:
- Positive and negative edge scenarios
- Boundary conditions (min/max bounds)
- Decimal precision preservation
- Win/loss ratio accuracy
- Confidence adjustments
- Edge cases (all wins, all losses, insufficient data)

## Safety Considerations

1. **Conservative Sizing**: Half-Kelly reduces risk of large drawdowns
2. **Hard Limits**: 25% maximum prevents excessive concentration
3. **Minimum Viable**: 1% minimum ensures trades are worth transaction costs
4. **Fallback Logic**: Multiple fallback paths to fixed percentage sizing
5. **Error Handling**: Comprehensive exception handling with logging

## Performance Impact

- Calculations are optimized using Decimal for precision
- Historical data is maintained in memory with size limits
- Lookback period is configurable to balance accuracy vs. computation

## Future Enhancements

Potential improvements for consideration:
1. Dynamic Half-Kelly adjustment based on market volatility
2. Sector-specific Kelly parameters
3. Time-weighted returns for more recent data emphasis
4. Confidence intervals for Kelly estimates
5. Monte Carlo validation of Kelly parameters