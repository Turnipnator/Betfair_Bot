"""
Acca advisor engine: scan, qualify legs, propose accas, settle.

Advisory only. Nothing here places a bet anywhere; the only outputs are
Telegram messages and rows in data/acca.db.
"""

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Protocol

from config.logging_config import get_logger
from src.acca.builder import Leg, choose_legs, propose
from src.acca.messages import UK, format_alert, uk_time
from src.acca.pricing import MARKET_LABELS, FairSelection, devig
from src.acca.scanner import AccaScanner
from src.acca.settlement import (
    LOST,
    UNKNOWN,
    VOID,
    acca_result,
    acca_return,
    notional_return,
    selection_result,
)
from src.acca.signal import Quote, measure_move, qualifies
from src.acca.store import AccaStore, event_key, naive_utc
from src.models import Market

logger = get_logger(__name__)

# A quote is persisted when the fair probability moves this much, the market
# trades this much more, or this long has passed; otherwise the reading only
# lives in memory. Keeps the quote table to what the signal and the closing
# line actually need.
STORE_PROB_STEP = 0.001
STORE_MATCHED_STEP = 500.0
STORE_HEARTBEAT = timedelta(minutes=30)
# How far back claimed legs are remembered (longer than any window).
CLAIM_MEMORY = timedelta(days=4)
STUCK_REALERT = timedelta(hours=24)


class Notifier(Protocol):
    async def send(self, text: str) -> Optional[int]: ...
    async def send_alert(self, text: str, acca_id: int) -> Optional[int]: ...


@dataclass
class _Stored:
    at: datetime
    fair_prob: float
    matched: float


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def uk_day_start(now: datetime) -> datetime:
    local = now.replace(tzinfo=timezone.utc).astimezone(UK)
    start = local.replace(hour=0, minute=0, second=0, microsecond=0)
    return start.astimezone(timezone.utc).replace(tzinfo=None)


def uk_week_start(now: datetime) -> datetime:
    local = now.replace(tzinfo=timezone.utc).astimezone(UK)
    start = (local - timedelta(days=local.weekday())).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return start.astimezone(timezone.utc).replace(tzinfo=None)


class AccaEngine:
    """Owns the in-memory price history and runs the scheduled jobs."""

    def __init__(self, cfg, store: AccaStore, scanner: AccaScanner, notifier: Optional[Notifier]):
        self.cfg = cfg
        self.store = store
        self.scanner = scanner
        self.notifier = notifier
        self.paused = False
        self._history: dict[str, list[Quote]] = {}
        self._last_stored: dict[str, _Stored] = {}
        # When each selection's current run of qualifying scans began. The
        # leg TTL counts from here, so a fresh move next day is a fresh leg.
        # In memory only: after a restart a still-qualifying leg gets a new TTL.
        self._qualifying_since: dict[str, datetime] = {}
        self.last_pool: list[Leg] = []

    @property
    def sending(self) -> bool:
        return self.cfg.alerts_enabled and self.notifier is not None

    async def warm_start(self) -> None:
        """Reload recent history so a restart does not blind the signal for hours."""
        since = utcnow() - timedelta(hours=self.cfg.signal_lookback_hours + 1)
        self._history = await self.store.load_history(since)
        for key, quotes in self._history.items():
            q = quotes[-1]
            self._last_stored[key] = _Stored(q.at, q.fair_prob, q.market_matched)
        logger.info("Acca history loaded", selections=len(self._history))

    # ------------------------------------------------------------------ scan

    def _should_store(self, key: str, sel: FairSelection, matched: float, now: datetime) -> bool:
        last = self._last_stored.get(key)
        if (
            last is None
            or abs(sel.fair_prob - last.fair_prob) >= STORE_PROB_STEP
            or matched - last.matched >= STORE_MATCHED_STEP
            or now - last.at >= STORE_HEARTBEAT
        ):
            self._last_stored[key] = _Stored(now, sel.fair_prob, matched)
            return True
        return False

    def _remember(self, key: str, quote: Quote) -> list[Quote]:
        cutoff = quote.at - timedelta(hours=self.cfg.signal_lookback_hours + 1)
        hist = [q for q in self._history.get(key, []) if q.at >= cutoff]
        hist.append(quote)
        self._history[key] = hist
        return hist

    async def scan(self) -> None:
        """One scan: price, record, qualify, maybe propose."""
        now = utcnow()
        cfg = self.cfg
        extra = await self.store.open_leg_market_ids(now)
        markets = await self.scanner.fetch(cfg, extra)

        rejects: Counter = Counter()
        pairs: list[tuple[Market, FairSelection]] = []
        store_keys: set[str] = set()
        candidates: list[Leg] = []

        for market in markets:
            priced = devig(market, cfg)
            if not priced.trusted:
                rejects[priced.reject_reason] += 1
                continue
            for sel in priced.selections:
                pairs.append((market, sel))
                if self._should_store(sel.key, sel, market.total_matched, now):
                    store_keys.add(sel.key)
                hist = self._remember(sel.key, Quote(now, sel.fair_prob, market.total_matched))
                move = measure_move(hist, now, cfg.signal_lookback_hours)
                if qualifies(move, cfg):
                    candidates.append(
                        Leg(
                            key=sel.key,
                            event_id=event_key(market),
                            event_name=market.event_name,
                            competition=market.competition or "",
                            kickoff=naive_utc(market.start_time),
                            market_type=sel.market_type,
                            label=sel.label,
                            fair_odds=sel.fair_odds,
                            shortening=move.shortening,
                            move_volume=move.volume,
                        )
                    )
                    # A qualifying reading is always kept: it is the candidate log.
                    store_keys.add(sel.key)

        await self.store.record_prices(pairs, now, store_keys)
        await self.store.mark_qualified((c.key for c in candidates), now)

        # Drop legs whose signal has gone stale, and matches already offered.
        live = {c.key for c in candidates}
        self._qualifying_since = {
            k: self._qualifying_since.get(k, now) for k in live
        }
        claimed = await self.store.claimed_events(now - CLAIM_MEMORY)
        ttl = timedelta(minutes=cfg.leg_ttl_minutes)
        pool = [
            c for c in candidates
            if c.event_id not in claimed and now - self._qualifying_since[c.key] <= ttl
        ]
        self.last_pool = pool

        trusted = len(markets) - sum(rejects.values())
        await self.store.record_scan(now, len(markets), trusted, dict(rejects), len(candidates), len(pool))
        logger.info(
            "Acca scan",
            markets=len(markets),
            trusted=trusted,
            qualified=len(candidates),
            pool=len(pool),
            rejects=dict(rejects),
        )

        if self.paused:
            return
        if await self.store.alerts_since(uk_day_start(now)) >= cfg.max_alerts_per_day:
            return
        chosen = choose_legs(pool, now, cfg)
        if chosen:
            await self._propose(chosen, now)

    async def _propose(self, legs: list[Leg], now: datetime) -> None:
        staked_today = await self.store.staked_since(uk_day_start(now))
        staked_week = await self.store.staked_since(uk_week_start(now))
        proposal = propose(legs, staked_today, staked_week, self.cfg)
        status = "alerted" if self.sending else "dry_run"
        acca_id = await self.store.save_acca(proposal, status, now)
        logger.info(
            "Acca proposed",
            acca_id=acca_id,
            status=status,
            legs=proposal.n_legs,
            fair_odds=round(proposal.combined_fair_odds, 2),
            min_odds=round(proposal.min_combined_odds, 2),
            stake=proposal.suggested_stake,
        )
        if status == "alerted":
            msg_id = await self.notifier.send_alert(format_alert(acca_id, proposal), acca_id)
            if msg_id:
                await self.store.update_acca(acca_id, telegram_message_id=msg_id)

    # ------------------------------------------------------------ settlement

    async def settle(self) -> None:
        """Settle selections from Betfair, then every acca they complete."""
        now = utcnow()
        cfg = self.cfg

        for sel in await self.store.selections_postponed(now, timedelta(hours=cfg.postpone_void_hours)):
            await self.store.settle_selection(sel.key, VOID, "postponed", now, None, None)
            await self._say(
                f"⏸ Postponed, leg void (1.00): {sel.event_name}, {sel.label}\n"
                f"Kick-off moved {uk_time(sel.kickoff_original)} → {uk_time(sel.kickoff)}. "
                "Check your bookmaker's rule; /acca_result overrides."
            )

        due = await self.store.selections_to_settle(now, timedelta(minutes=cfg.settle_after_minutes))
        if due:
            results = await self.scanner.results(sorted({s.market_id for s in due}))
            for sel in due:
                market_status, statuses = results.get(sel.market_id, ("", {}))
                res = selection_result(
                    json.loads(sel.winning_ids), json.loads(sel.void_ids), statuses, market_status
                )
                if res in (None, UNKNOWN):
                    await self._maybe_stuck(sel, now, market_status, res)
                    continue
                close = await self.store.closing_quote(sel.key, sel.kickoff)
                await self.store.settle_selection(
                    sel.key, res, "betfair", now,
                    (1.0 / close.fair_prob) if close else None,
                    close.captured_at if close else None,
                )

        await self.store.refresh_leg_outcomes()
        await self._settle_accas(now)

    async def _maybe_stuck(self, sel, now: datetime, market_status: str, res: Optional[str]) -> None:
        age = now - sel.kickoff_original
        if age < timedelta(hours=self.cfg.stuck_alert_hours):
            return
        if sel.stuck_alerted_at and now - sel.stuck_alerted_at < STUCK_REALERT:
            return
        why = "closed without a single winner" if res == UNKNOWN else f"market {market_status or 'not returned'}"
        await self._say(
            f"❓ Unsettled leg: {sel.event_name}, {sel.label} ({why}, "
            f"{age.total_seconds() / 3600:.0f}h after kick-off).\n"
            "Settle by hand: /acca_result &lt;acca&gt; &lt;leg&gt; won|lost|void"
        )
        await self.store.set_stuck_alerted(sel.key, now)

    async def _settle_accas(self, now: datetime) -> None:
        for acca in await self.store.unsettled_accas():
            legs = await self.store.get_legs(acca.id)
            results = [sel.result for _, sel in legs]
            outcome = acca_result(results)
            if outcome is None:
                continue
            mins = [leg.min_odds for leg, _ in legs]
            # A lost acca may still have legs to play; count them as unknown.
            filled = [r if r in ("WON", LOST, VOID) else LOST for r in results]
            values: dict = {
                "result": outcome,
                "settled_at": now,
                "notional_return": notional_return(outcome, filled, mins),
            }
            if acca.status == "placed" and acca.taken_stake:
                taken = [leg.taken_odds for leg, _ in legs]
                gross, estimated = acca_return(
                    outcome, acca.taken_stake, acca.taken_odds, filled,
                    taken if all(taken) else None, mins,
                )
                values.update(gross_return=gross, pnl=gross - acca.taken_stake, return_estimated=estimated)
                await self._say(self._settled_text(acca, outcome, gross, estimated, legs))
            await self.store.update_acca(acca.id, **values)

    @staticmethod
    def _settled_text(acca, outcome: str, gross: float, estimated: bool, legs) -> str:
        icon = {"WON": "✅", "LOST": "❌", "VOID": "↩️"}[outcome]
        pnl = gross - acca.taken_stake
        lines = [f"{icon} Acca #{acca.id} {outcome}: £{pnl:+.2f} (stake £{acca.taken_stake:.2f})"]
        for i, (_, sel) in enumerate(legs, start=1):
            market = MARKET_LABELS.get(sel.market_type, sel.market_type)
            lines.append(f"  {i}. {sel.event_name}, {market}: {sel.label} → {sel.result or 'pending'}")
        if estimated:
            lines.append(
                "Return estimated (a leg voided and only combined odds were given). "
                f"Correct with /acca_payout {acca.id} &lt;amount returned&gt;"
            )
        return "\n".join(lines)

    async def _say(self, text: str) -> None:
        logger.info("Acca notice", text=text.replace("\n", " | "))
        if self.notifier:
            await self.notifier.send(text)

    # ------------------------------------------------------------ housekeeping

    async def prune(self) -> None:
        before = utcnow() - timedelta(hours=self.cfg.quote_retention_hours)
        n = await self.store.prune_quotes(before)
        # In-memory history of fixtures that have stopped being quoted.
        stale = utcnow() - timedelta(hours=self.cfg.signal_lookback_hours + 1)
        for key in [k for k, h in self._history.items() if not h or h[-1].at < stale]:
            self._history.pop(key, None)
            self._last_stored.pop(key, None)
        logger.info("Acca quotes pruned", deleted=n, in_memory=len(self._history))

    async def daily_summary(self) -> None:
        now = utcnow()
        accas = await self.store.accas_since(uk_day_start(now))
        scan = await self.store.last_scan()
        mode = "LIVE alerts" if self.sending else "DRY RUN (ACCA_ALERTS_ENABLED=false)"
        lines = [
            f"📋 <b>Acca advisor today</b> · {mode}" + (" · PAUSED" if self.paused else ""),
            f"Accas proposed: {len(accas)} "
            f"(placed {sum(a.status == 'placed' for a in accas)}, "
            f"skipped {sum(a.status == 'skipped' for a in accas)})",
        ]
        if scan:
            lines.append(
                f"Last scan: {scan.markets} markets, {scan.trusted} trusted, "
                f"{scan.qualified} qualifying legs, {scan.pool} in the pool"
            )
        await self._say("\n".join(lines))
