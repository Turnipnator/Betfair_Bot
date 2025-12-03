"""Trading strategies module."""

from src.strategies.base import BaseStrategy
from src.strategies.value_betting import (
    FootballValueStrategy,
    HorseRacingValueStrategy,
    ValueBettingStrategy,
)
from src.strategies.lay_the_draw import (
    LayTheDrawStrategy,
    LTDPosition,
    LTDState,
)
from src.strategies.arbitrage import (
    ArbOpportunity,
    ArbType,
    ArbitrageStrategy,
    ScalpingStrategy,
)

__all__ = [
    # Base
    "BaseStrategy",
    # Value betting
    "FootballValueStrategy",
    "HorseRacingValueStrategy",
    "ValueBettingStrategy",
    # Lay the Draw
    "LayTheDrawStrategy",
    "LTDPosition",
    "LTDState",
    # Arbitrage
    "ArbOpportunity",
    "ArbType",
    "ArbitrageStrategy",
    "ScalpingStrategy",
]
