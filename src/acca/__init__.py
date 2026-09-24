"""
Advisory football accumulator engine.

Runs in its own container (acca-advisor) and never places a bet: it prices
football selections from Betfair Exchange books, proposes accumulators whose
legs have shortened on the Exchange, and sends them to Telegram for manual
placement at a bookmaker. Ledger: data/acca.db.
"""
