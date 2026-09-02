"""CLV is a pre-off measurement (2 Sep 2026).

The capture job used to keep refreshing a bet's "close" through the match and
after settlement, so the stored price was the final in-play price: value
betting recorded -49% CLV on a winner whose close was 1.02. Now nothing is
recorded once the market is in-play or closed, settled bets are never queued,
and readings taken after kick-off are purged at startup.

Also exercises the persisted strategy funnel (strategy_evaluations) end to
end against a throwaway SQLite file.

Run in the betfair-bot container:
  docker compose exec -T -e PYTHONPATH=/app betfair-bot python tests/test_clv_preoff.py
"""
import asyncio
import json
import os
import pathlib
import tempfile
from datetime import datetime, timedelta, timezone

TMP = tempfile.mkdtemp()
os.environ["DATABASE_TYPE"] = "sqlite"
os.environ["DATABASE_URL"] = f"sqlite:///{TMP}/clv_test.db"

from sqlalchemy import select  # noqa: E402

from src.database import BetRepository, EvaluationRepository, db  # noqa: E402
from src.database.schema import BetRecord, MarketRecord, StrategyEvaluationRecord  # noqa: E402
from src.models import Market, MarketStatus, Sport  # noqa: E402
from src.utils.clv import closing_line_capturable  # noqa: E402

PASS = FAIL = 0


def check(label, got, want):
    global PASS, FAIL
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {label}: got {got!r} want {want!r}")
    PASS += ok
    FAIL += not ok


def mk(in_play=False, status=MarketStatus.OPEN):
    return Market(market_id="1", market_name="Match Odds", event_name="A v B", sport=Sport.FOOTBALL,
                  market_type="MATCH_ODDS", start_time=datetime.now(timezone.utc),
                  in_play=in_play, status=status)


print("closing_line_capturable")
check("open, pre-off: capture", closing_line_capturable(mk()), True)
check("in-play: never", closing_line_capturable(mk(in_play=True)), False)
check("closed: never", closing_line_capturable(mk(status=MarketStatus.CLOSED)), False)
check("suspended: wait", closing_line_capturable(mk(status=MarketStatus.SUSPENDED)), False)
check("no market: never", closing_line_capturable(None), False)


async def db_checks():
    await db.initialize()
    ko = datetime(2026, 8, 23, 17, 0, 0)

    print("purge_post_off_clv")

    def bet(i, recorded, price=None, clv=None):
        return BetRecord(
            id=i, market_id="1.100", selection_id=1, selection_name="Monaco",
            strategy="value_betting", bet_type="BACK", requested_odds=2.0, matched_odds=2.0,
            stake=10, potential_profit=10, potential_loss=10, status="SETTLED", is_paper=False,
            result="WON", placed_at=ko - timedelta(hours=14),
            close_price=price, clv_percent=clv, close_recorded_at=recorded,
        )

    async with db.session() as session:
        session.add(MarketRecord(id="1.100", event_name="Le Havre v Monaco", market_name="Match Odds",
                                 sport="football", market_type="MATCH_ODDS", start_time=ko))
        session.add(bet(1, ko - timedelta(minutes=5), 2.1, 5.0))   # pre-off: a real closing line
        session.add(bet(2, ko + timedelta(hours=3), 1.02, -49.0))  # post-off: the result in disguise
        session.add(bet(3, None))                                  # never captured

    async with db.session() as session:
        cleared = await BetRepository(session).purge_post_off_clv()
    check("one post-off reading cleared", cleared, 1)

    async with db.session() as session:
        b1 = await session.get(BetRecord, 1)
        b2 = await session.get(BetRecord, 2)
        check("pre-off reading kept", (b1.close_price, b1.clv_percent), (2.1, 5.0))
        check("post-off reading cleared", (b2.close_price, b2.clv_percent, b2.close_recorded_at), (None, None, None))

    async with db.session() as session:
        check("idempotent", await BetRepository(session).purge_post_off_clv(), 0)

    print("strategy_evaluations")
    m = Market(market_id="1.200", market_name="Match Odds", event_name="Arsenal v Everton",
               sport=Sport.FOOTBALL, market_type="MATCH_ODDS",
               start_time=datetime(2026, 9, 5, 14, 0, tzinfo=timezone.utc),
               competition="Premier League", country_code="GB", event_id=555)

    async with db.session() as session:
        await EvaluationRepository(session).upsert(
            strategy="lay_the_draw", market=m, stage="prematch",
            outcome="rejected", reason="draw_odds_range", detail={"draw_odds": 6.2})
    async with db.session() as session:
        await EvaluationRepository(session).upsert(
            strategy="lay_the_draw", market=m, stage="prematch",
            outcome="candidate", reason="stored", detail={"draw_odds": 3.4})

    async with db.session() as session:
        rows = list((await session.execute(select(StrategyEvaluationRecord))).scalars().all())
        check("one row per strategy/market/stage", len(rows), 1)
        r = rows[0]
        check("latest verdict wins", (r.outcome, r.reason), ("candidate", "stored"))
        check("evaluations counted", r.evaluations, 2)
        check("start_time stored as naive UTC", r.start_time, datetime(2026, 9, 5, 14, 0))
        check("competition and country carried", (r.competition, r.country_code), ("Premier League", "GB"))
        check("detail is JSON", json.loads(r.detail)["draw_odds"], 3.4)
        rid = r.id

    async with db.session() as session:
        repo = EvaluationRepository(session)
        check("20 min after KO: nothing due",
              await repo.get_pending_scores(datetime(2026, 9, 5, 14, 20)), [])
        pend = await repo.get_pending_scores(datetime(2026, 9, 5, 14, 50))
        check("50 min after KO: HT due", [p.id for p in pend], [rid])
        await repo.record_scores(rid, ht=(0, 0), checked_at=datetime(2026, 9, 5, 14, 50))

    async with db.session() as session:
        repo = EvaluationRepository(session)
        check("HT known, 65 min: FT not yet due",
              await repo.get_pending_scores(datetime(2026, 9, 5, 15, 5)), [])
        pend = await repo.get_pending_scores(datetime(2026, 9, 5, 15, 40))
        check("100 min after KO: FT due", [p.id for p in pend], [rid])
        await repo.record_scores(rid, ft=(2, 1), checked_at=datetime(2026, 9, 5, 15, 40))

    async with db.session() as session:
        repo = EvaluationRepository(session)
        check("both scores known: done", await repo.get_pending_scores(datetime(2026, 9, 5, 16, 0)), [])
        r = await session.get(StrategyEvaluationRecord, rid)
        check("scores stored", (r.ht_home, r.ht_away, r.ft_home, r.ft_away), (0, 0, 2, 1))
        check("a day later it has aged out",
              await repo.get_pending_scores(datetime(2026, 9, 7, 16, 0)), [])

    await db.close()


asyncio.run(db_checks())

print("engine and repository wiring")
engine_src = pathlib.Path("scripts/run_paper_trading.py").read_text()
repo_src = pathlib.Path("src/database/repositories.py").read_text()
check("capture job guards on pre-off", "if not closing_line_capturable(market):" in engine_src, True)
check("purge runs at startup", "purge_post_off_clv()" in engine_src, True)
check("settled bets no longer queued for CLV",
      "& BetRecord.close_recorded_at.is_(None)" in repo_src, False)
check("only open bets queued", ".where(BetRecord.status == BetStatus.MATCHED.value)" in repo_src, True)
check("competition column migrated", '"competition"' in pathlib.Path("src/database/connection.py").read_text(), True)

print(f"\nRESULT: {PASS}/{PASS + FAIL} passed")
raise SystemExit(1 if FAIL else 0)
