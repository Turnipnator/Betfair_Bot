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
from src.strategies.lay_the_server import (
    LayTheServerStrategy,
    LTSPosition,
    LTSState,
)
from src.strategies.arbitrage import (
    ArbOpportunity,
    ArbType,
    ArbitrageStrategy,
    ScalpingStrategy,
)
from src.strategies.horse_racing import (
    NagsBackStrategy,
    NagsLayFavStrategy,
)

__all__ = [
    # Base
    "BaseStrategy",
    # Value betting
    "FootballValueStrategy",
    "HorseRacingValueStrategy",
    "ValueBettingStrategy",
    # Lay the Draw (Football)
    "LayTheDrawStrategy",
    "LTDPosition",
    "LTDState",
    # Lay the Server (Tennis)
    "LayTheServerStrategy",
    "LTSPosition",
    "LTSState",
    # Arbitrage
    "ArbOpportunity",
    "ArbType",
    "ArbitrageStrategy",
    "ScalpingStrategy",
    # Horse racing (Nags)
    "NagsBackStrategy",
    "NagsLayFavStrategy",
]
