"""Data services for fetching external statistics."""

from src.data.football_data import (
    FootballDataService,
    LeagueStats,
    TeamStats,
    football_data_service,
    LEAGUE_URLS,
    LEAGUE_NAME_MAP,
)

__all__ = [
    "FootballDataService",
    "LeagueStats",
    "TeamStats",
    "football_data_service",
    "LEAGUE_URLS",
    "LEAGUE_NAME_MAP",
]
