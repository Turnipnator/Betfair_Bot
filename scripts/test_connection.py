#!/usr/bin/env python3
"""Quick test of Betfair API connection."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
from config import settings
from src.betfair import betfair_client
from src.models import MarketFilter, Sport


async def test():
    print("Testing Betfair connection...")
    print(f"  Username: {settings.betfair.username}")
    print(f"  App Key: {settings.betfair.app_key[:8]}...")
    print(f"  Cert: {settings.betfair.cert_path}")
    print()

    # Test login
    success = await betfair_client.login()

    if not success:
        print("FAILED to login to Betfair")
        return False

    print("SUCCESS - Logged in to Betfair!")
    print()

    # Try fetching some markets
    print("Fetching upcoming markets...")

    market_filter = MarketFilter(
        sports=[Sport.HORSE_RACING],
        market_types=["WIN"],
        countries=["GB", "IE"],
        from_hours=0,
        to_hours=24,
        max_results=5,
    )

    markets = await betfair_client.get_markets(market_filter)

    print(f"Found {len(markets)} markets")

    for market in markets[:5]:
        print(f"  - {market.event_name}: {market.market_name}")
        print(f"    Start: {market.start_time}")
        print(f"    Runners: {len(market.runners)}")

    # Logout
    await betfair_client.logout()
    print()
    print("Connection test complete!")
    return True


if __name__ == "__main__":
    asyncio.run(test())
