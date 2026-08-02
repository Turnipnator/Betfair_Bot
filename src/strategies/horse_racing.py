"""
Horse racing strategies that consume Nags daily picks.

Two strategies share a Nags SQLite reader and a combined daily cap:

* ``NagsBackStrategy`` (``nags_back``) — back Nags' daily picks
  (NAP / Next Best / per-race selection) on Betfair Exchange. Exchange
  prices are typically a few percent better than Bet365 even after the
  5% commission, so re-using Paul's existing pick stream is free alpha.

* ``NagsLayFavStrategy`` (``nags_lay_fav``) — B1a refinement: lay the
  Betfair favourite when Nags has picked a *different* horse in that
  race. Why the 2.0 floor? Nags' own rules block any pick at evens or
  shorter (the "sub-evens block"), so without the floor we'd lay every
  sub-evens favourite by default and that's a false signal. The 4.0
  ceiling caps liability and keeps us out of races where the
  "favourite" is barely the favourite.

Reads ``/app/nags-data/racing.db`` (the Nags container's SQLite
mounted read-only into the Betfair bot — see docker-compose.yml).
"""

from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from config.logging_config import get_logger
from src.models import Bet, BetSignal, BetType, Market, Runner, Sport
from src.strategies.base import BaseStrategy

logger = get_logger(__name__)


# Nags DB path inside the Betfair container. Mounted read-only from
# /root/horse-racing-bot/data on the VPS (see docker-compose.yml).
# Override via env var for local development.
NAGS_DB_PATH = Path(os.environ.get("NAGS_DB_PATH", "/app/nags-data/racing.db"))

# Shared risk caps for both Nags strategies.
DAILY_HORSE_RACING_BET_CAP = 6  # combined nags_back + nags_lay_fav
BACK_FLAT_STAKE = 5.0
LAY_LIABILITY_CAP = 5.0
MIN_SECONDS_TO_OFF = 300  # don't bet inside the last 5 minutes

# nags_place (the each-way place leg) filters — the actual CLAUDE.md EW rule:
# E/W when (8+ runners AND 3/1+ odds), and HANDICAPS are always E/W regardless
# of field size / odds. The win price is read from the Nags odds_guide (the
# morning price the rubric was applied to), NOT the live place-market price —
# a PLACE market prices the place, so it can't reveal the horse's WIN odds.
#
# WIDENED 27 Jul 2026 (was 6.0 / 5-1-only, non-handicap-blind): the old floor
# missed the SHORTER-priced picks, which is exactly where most of our places
# come from (measured 21.8% place rate). odds are decimal (3/1 == 4.0).
EACH_WAY_MIN_WIN_ODDS = 4.0        # 3/1 decimal — non-handicap odds floor
EACH_WAY_MIN_RUNNERS_NONHCAP = 8   # non-handicap field-size floor
PLACE_FLAT_STAKE = 2.0  # cut 5.0->2.0 on 28 Jul to bleed slower while the
# place-only edge is unproven. £2 is the Betfair Exchange GBP minimum back
# stake — watch the first live place bet is MATCHED, not rejected as
# below-minimum (raise back toward 3-5 if the exchange bounces it).
# Betfair offers no place pool below 5 runners — also the floor for a
# handicap's "always E/W" clause (no place market, no place leg).
PLACE_MIN_RUNNERS = 5
# nags_place gets its OWN daily cap. It is paper-only and must never consume
# a slot that the live nags_back would otherwise use.
DAILY_PLACE_BET_CAP = 6

# Every horse-racing strategy, live or paper. Used to scope the durable
# Nags results settlement fallback. Kept separate from the live/paper gate
# below: a strategy going live must not silently lose its settlement path.
HORSE_RACING_STRATEGIES: frozenset[str] = frozenset({
    "nags_back",
    "nags_lay_fav",
    "nags_place",
})

# Strategies in this set are forced to paper mode even when the bot is
# running LIVE — used during the observation window for new strategies
# while the rest of the book (football, value, LTD, arb) trades live.
#
# nags_back went LIVE 2026-07-09 (flat £5 WIN) after an 8-week paper run:
# +£99.15 over 65 decided bets. Note the edge is unproven — strip the two
# biggest winners (Priapos 15.5, Bearish 8.2) and it is -£3.93. Live bets
# settle via reconcile_with_betfair(), not the Nags results fallback.
#
# nags_lay_fav stays paper: -£1.35 on only 5 decided bets, no evidence base.
#
# nags_back reverted to PAPER 2026-07-28: it bled -£88.37 over 34 live bets
# (5 wins, -52% ROI). The Nags system is breakeven-at-best at BOG and the
# exchange gives no BOG, so flat-£5 win-only betting is a structural slow loss
# + variance. Monitoring in paper; only re-enable live on a PROVEN edge.
#
# nags_place (the EW place leg) STAYS LIVE at Paul's direction (he believes the
# place market is where the edge is). NB: with nags_back now on paper, this is
# effectively a PLACE-ONLY real-money bet on the picks (win side is paper). Its
# live sample is still tiny (-£6.38 over 4, one place win Abduction +£8.62), so
# it's a belief not yet data — WATCH it. Trigger = CLAUDE.md EW rule (3/1+ AND
# 8+ runners; handicaps always). Independent of nags_back (own DAILY_PLACE_BET_CAP,
# own PLACE market-type gate), so nags_back going paper does not affect it.
#
# nags_lay_fav stays paper (no evidence base).
FORCE_PAPER_STRATEGIES: frozenset[str] = frozenset({
    "nags_lay_fav",
    "nags_back",
})

# nags_lay_fav (B1a) filters.
LAY_FAV_MIN_ODDS = 2.0
LAY_FAV_MAX_ODDS = 4.0
# Skip races where Nags' only pick is `race_nb` — those are
# lower-confidence backup picks and shouldn't drive a lay signal.
EXCLUDE_RACE_NB_ONLY = True

# Selection types Nags writes into its `selections.selection_type` column.
SELECTION_TYPE_NAP = "nap"
SELECTION_TYPE_NEXT_BEST = "next_best"
SELECTION_TYPE_SELECTION = "selection"
SELECTION_TYPE_RACE_NB = "race_nb"


@dataclass(frozen=True)
class NagsPick:
    """One Nags selection enriched with course/race-time for matching."""

    course: str
    race_time: str  # "HH:MM" local UK time as written by Nags
    horse: str
    selection_type: str
    odds_guide: Optional[str]
    score: Optional[float]
    # Full Nags race_name ("<Course> - <Race>"). Retained so the EW place leg
    # can detect handicaps ("handicap" in the name) — CLAUDE.md makes handicaps
    # always each-way. Defaulted for backward compatibility with any caller
    # that predates this field.
    race_name: Optional[str] = None


class NagsReader:
    """Loads today's Nags picks from the shared SQLite file.

    Nags writes selections with ``meeting_id=NULL`` and packs the course
    name into ``race_name`` as ``"<Course> - <Race>"``. We don't JOIN
    against ``meetings``; we just filter by ``date(created_at)`` and
    parse the course back out of ``race_name``.

    ``superseded_at IS NULL`` (added 1 Aug 2026) keeps the exchange off a
    REPLACED card. Nags' daily cap was enforced per ``/run``, so a second run
    used to append a whole second card at full stakes -- 1 Aug 2026 issued two
    NAPs across Thirsk and Goodwood. A later run now supersedes the earlier
    picks (they are marked, never deleted, so the ledger keeps them for audit);
    without this filter we would still back the withdrawn card.

    ``source = 'bot'`` (added 2 Aug 2026) keeps the exchange off MANUALLY logged
    picks. Claude's own cards are now written to the same table so they can be
    settled and measured, but no strategy produced them and they must never be
    staked. ``source IS NULL`` is tolerated so an un-migrated DB still bets the
    bot's own picks rather than silently going quiet.

    Requires ``selections.superseded_at``, added by Nags' ``init_db`` migration
    -- deploy Nags FIRST. If the column is missing the query raises, the
    ``sqlite3.Error`` handler below logs a warning and returns no picks, so the
    failure is loud-ish and fail-closed (no bets) rather than wrong bets.
    """

    def __init__(self, db_path: Path = NAGS_DB_PATH) -> None:
        self._db_path = db_path

    def load_today(self) -> list[NagsPick]:
        """Return all of today's Nags picks. Empty list if DB missing."""
        if not self._db_path.exists():
            logger.debug("Nags DB not found, skipping", path=str(self._db_path))
            return []

        try:
            # Read-only connection. uri=True needed for the ?mode=ro flag.
            uri = f"file:{self._db_path}?mode=ro"
            with sqlite3.connect(uri, uri=True) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT race_time, race_name, horse, selection_type,
                           odds_guide, score
                    FROM selections
                    WHERE date(created_at) = date('now')
                      AND superseded_at IS NULL
                      AND (source IS NULL OR source = 'bot')
                    """
                ).fetchall()
        except sqlite3.Error as e:
            logger.warning("Failed to read Nags DB", error=str(e))
            return []

        picks: list[NagsPick] = []
        for row in rows:
            course = _course_from_race_name(row["race_name"])
            if not course:
                # No "Course - " prefix, can't safely match to Betfair, so the
                # pick is dropped before any strategy sees it. Raised from
                # debug to warning 19 Jul 2026 — this is the earliest and
                # quietest place a real bet can vanish.
                # Deduped: picks are reloaded on a 120s TTL across three
                # strategy instances, so an undeduped warning here would fire
                # ~90x/hour for one bad row.
                if _tracker.warn_once(f"noCourse:{row['horse']}:{row['race_name']}"):
                    logger.warning(
                        "Nags pick dropped — unparseable race_name, cannot "
                        "resolve course (bet SKIPPED)",
                        race_name=row["race_name"],
                        horse=row["horse"],
                        selection_type=row["selection_type"],
                    )
                continue
            picks.append(
                NagsPick(
                    course=course,
                    race_time=row["race_time"],
                    horse=row["horse"],
                    selection_type=row["selection_type"],
                    odds_guide=row["odds_guide"],
                    score=row["score"],
                    race_name=row["race_name"],
                )
            )
        return picks


def _course_from_race_name(race_name: Optional[str]) -> Optional[str]:
    """Extract the course from Nags' ``"Course - Race"`` race_name."""
    if not race_name:
        return None
    head, sep, _ = race_name.partition(" - ")
    if not sep:
        return None
    course = head.strip()
    return course or None


# Cap groups. nags_back + nags_lay_fav share one daily budget (they are
# complementary bets on the same race). nags_place is paper-only and gets its
# own budget: a paper place leg must never exhaust the cap and lock the LIVE
# nags_back out of a real bet.
_CAP_GROUPS: dict[str, tuple[str, int]] = {
    "nags_back": ("primary", DAILY_HORSE_RACING_BET_CAP),
    "nags_lay_fav": ("primary", DAILY_HORSE_RACING_BET_CAP),
    "nags_place": ("place", DAILY_PLACE_BET_CAP),
}


class _NagsDailyTracker:
    """In-memory daily caps, per cap-group, across the Nags strategies."""

    def __init__(self) -> None:
        self._date = date.today()
        self._bets_today: dict[str, int] = {}
        # Keyed by (strategy, market_id), not market_id alone. nags_back and
        # nags_lay_fav are complementary on the same race (back our pick AND
        # lay the favourite — both win if the favourite underperforms), so one
        # firing must NOT lock the other out of the market. A global market set
        # blocked nags_lay_fav 100% of the time, since nags_back is evaluated
        # first and claims essentially every pick-race.
        self._markets_bet: set[tuple[str, str]] = set()
        # Keys already warned about today. evaluate() runs once per market per
        # scan cycle, so an unresolvable pick would otherwise emit the same
        # warning hundreds of times a day and train us to ignore it.
        self._warned: set[str] = set()

    @staticmethod
    def _group(strategy: str) -> tuple[str, int]:
        return _CAP_GROUPS.get(strategy, ("primary", DAILY_HORSE_RACING_BET_CAP))

    def _maybe_reset(self) -> None:
        today = date.today()
        if self._date != today:
            self._date = today
            self._bets_today.clear()
            self._markets_bet.clear()
            self._warned.clear()
            logger.info("Nags daily counters reset")

    def warn_once(self, key: str) -> bool:
        """True the first time `key` is seen today, False thereafter.

        Used to surface a silent bet-skip exactly once per day instead of
        once per scan cycle.
        """
        self._maybe_reset()
        if key in self._warned:
            return False
        self._warned.add(key)
        return True

    def can_bet(self, strategy: str, market_id: str) -> bool:
        self._maybe_reset()
        group, cap = self._group(strategy)
        if self._bets_today.get(group, 0) >= cap:
            return False
        if (strategy, market_id) in self._markets_bet:
            return False
        return True

    def record_bet(self, strategy: str, market_id: str) -> None:
        self._maybe_reset()
        group, cap = self._group(strategy)
        self._bets_today[group] = self._bets_today.get(group, 0) + 1
        self._markets_bet.add((strategy, market_id))
        logger.info(
            "Nags bet recorded",
            strategy=strategy,
            cap_group=group,
            bets_today=self._bets_today[group],
            cap=cap,
        )


# Module-level singleton — both strategies share the same cap.
_tracker = _NagsDailyTracker()


# Course name normalisation: Betfair uses "Newcastle (AW)" etc.,
# while Nags writes a plain "Newcastle". Strip trailing parentheticals
# and lowercase before comparing.
_PAREN_TAG_RE = re.compile(r"\s*\([^)]*\)\s*$")


def _normalise_course(name: str) -> str:
    return _PAREN_TAG_RE.sub("", name).strip().lower()


# Nags writes race_time as UK local "HH:MM" (BST in summer, GMT in winter),
# so we need to convert Betfair's UTC start_time to Europe/London before
# comparing.
_UK_TZ = ZoneInfo("Europe/London")

# Fallback regex for the rare case where market.start_time is missing.
# Real Betfair markets don't include the off-time in event_name/market_name
# (event_name is just "York 14th May", market_name "1m Hcap") — start_time
# is the authoritative source.
_TIME_ANYWHERE_RE = re.compile(r"\b(\d{1,2}:\d{2})\b")


def _parse_race_time(market: Market) -> Optional[str]:
    """Return scheduled off-time as 'HH:MM' UK local — to match Nags."""
    if market.start_time is not None:
        local = market.start_time.astimezone(_UK_TZ)
        return f"{local.hour:02d}:{local.minute:02d}"
    for text in (market.event_name, market.market_name):
        if not text:
            continue
        m = _TIME_ANYWHERE_RE.search(text)
        if m:
            hh, mm = m.group(1).split(":")
            return f"{int(hh):02d}:{mm}"
    return None


def _normalise_horse_name(name: str) -> str:
    """Strip punctuation and lowercase for fuzzy horse-name matching."""
    return re.sub(r"[^a-z0-9 ]+", "", name.lower()).strip()


def _index_picks_by_race(
    picks: list[NagsPick],
) -> dict[tuple[str, str], list[NagsPick]]:
    """Group picks by (normalised course, HH:MM)."""
    out: dict[tuple[str, str], list[NagsPick]] = {}
    for p in picks:
        key = (_normalise_course(p.course), p.race_time)
        out.setdefault(key, []).append(p)
    return out


def _picks_for_market(
    market: Market,
    picks_by_race: dict[tuple[str, str], list[NagsPick]],
) -> list[NagsPick]:
    if not market.venue:
        return []
    race_time = _parse_race_time(market)
    if not race_time:
        return []
    return picks_by_race.get((_normalise_course(market.venue), race_time), [])


def _match_runner_to_pick(
    market: Market, pick: NagsPick
) -> Optional[Runner]:
    """Match a Nags pick to a Betfair runner with apostrophe tolerance."""
    runner = market.get_runner_by_name(pick.horse)
    if runner is not None:
        return runner
    # Fallback: normalise both sides (strips apostrophes, hyphens, etc.).
    target = _normalise_horse_name(pick.horse)
    if not target:
        return None
    for r in market.runners:
        if _normalise_horse_name(r.name) == target:
            return r
    return None


def _parse_odds_guide(odds_guide: Optional[str]) -> Optional[float]:
    """Best-effort parse of Nags' free-text odds guide to decimal.

    Handles "5/2", "11/4", "Evens", "2.5", "9-2". Returns None on
    anything weird — we only use this for the B1a longer-than-fav
    check, so falling back to runner price is fine.
    """
    if not odds_guide:
        return None
    s = odds_guide.strip().lower()
    if s in ("evens", "evs", "even"):
        return 2.0
    # Decimal form like "2.5" or "3.0".
    try:
        return float(s)
    except ValueError:
        pass
    # Fractional "a/b" or "a-b".
    m = re.match(r"^(\d+)\s*[/\-]\s*(\d+)$", s)
    if m:
        num, den = int(m.group(1)), int(m.group(2))
        if den > 0:
            return 1.0 + num / den
    return None


def _ew_place_eligible(
    is_handicap: bool, num_active: int, win_odds: Optional[float]
) -> bool:
    """Is this pick each-way eligible per the CLAUDE.md rule?

    HANDICAPS are always E/W regardless of field size or odds. Otherwise E/W
    only when the field is 8+ runners AND the win price is 3/1 or bigger
    (win_odds is DECIMAL; EACH_WAY_MIN_WIN_ODDS == 4.0 == 3/1). A None win
    price fails the non-handicap test (can't confirm 3/1) but never blocks a
    handicap. Assumes a place market exists — num_active >= PLACE_MIN_RUNNERS
    is enforced separately by the caller.
    """
    if is_handicap:
        return True
    if win_odds is None:
        return False
    return (
        num_active >= EACH_WAY_MIN_RUNNERS_NONHCAP
        and win_odds >= EACH_WAY_MIN_WIN_ODDS
    )


class _NagsStrategyBase(BaseStrategy):
    """Shared scaffolding: HR-only, pre-play, daily cap, off-time cutoff."""

    supported_sports: list[Sport] = [Sport.HORSE_RACING]
    requires_inplay: bool = False

    # SAFETY: BaseStrategy.supports_market() only checks sport and in-play, so
    # once PLACE markets are added to the scan every horse-racing strategy sees
    # them. Without this gate the LIVE nags_back would back its pick in the
    # "To Be Placed" market at place odds — a real-money bet nobody asked for.
    # Each strategy declares exactly the market type it understands.
    supported_market_types: frozenset[str] = frozenset({"WIN"})

    # Re-read Nags DB at most this often. Short enough that picks
    # written mid-day by the Nags bot appear within a couple of
    # scan cycles; long enough that we're not hammering SQLite for
    # every runner of every horse racing market.
    _PICKS_TTL_SECONDS = 120.0

    def __init__(self, reader: Optional[NagsReader] = None) -> None:
        super().__init__()
        self._reader = reader or NagsReader()
        self._picks_cache: list[NagsPick] = []
        self._picks_cache_date: Optional[date] = None
        self._picks_cache_loaded_at: float = 0.0

    def _todays_picks(self) -> list[NagsPick]:
        import time
        today = date.today()
        now = time.monotonic()
        stale = (now - self._picks_cache_loaded_at) >= self._PICKS_TTL_SECONDS
        if self._picks_cache_date != today or stale:
            self._picks_cache = self._reader.load_today()
            self._picks_cache_date = today
            self._picks_cache_loaded_at = now
            logger.info(
                "Loaded Nags picks for today",
                count=len(self._picks_cache),
                date=today.isoformat(),
            )
        return self._picks_cache

    def supports_market(self, market: Market) -> bool:
        if not super().supports_market(market):
            return False
        # See supported_market_types above — this is the guard that keeps a
        # WIN strategy out of PLACE markets and vice versa.
        return (market.market_type or "") in self.supported_market_types

    def pre_evaluate(self, market: Market) -> bool:
        if not super().pre_evaluate(market):
            return False
        # Belt and braces: pre_evaluate() is reachable without supports_market()
        # in some call paths, and this gate protects real money.
        if (market.market_type or "") not in self.supported_market_types:
            return False
        # Off-time cutoff — 5 minutes is enough margin to clear the
        # 60s pre-close lockout and avoid late price thrash.
        if market.seconds_to_start < MIN_SECONDS_TO_OFF:
            return False
        if not _tracker.can_bet(self.name, market.market_id):
            return False
        return True

    def manage_position(
        self, market: Market, open_bet: Bet
    ) -> Optional[BetSignal]:
        """Nags strategies hold to settlement; no in-play management."""
        return None

    def record_bet_placed(self) -> None:
        """No-op — the tracker is updated inside evaluate() when a
        signal is generated. Over-counting on the rare reject path is
        the safer side to err on for a 6/day cap."""
        return None


class NagsBackStrategy(_NagsStrategyBase):
    """Back Nags' daily picks on Betfair Exchange."""

    name: str = "nags_back"

    async def evaluate(self, market: Market) -> Optional[BetSignal]:
        if not self.pre_evaluate(market):
            return None

        picks = _picks_for_market(market, _index_picks_by_race(self._todays_picks()))
        if not picks:
            return None

        # Prefer NAP > next_best > selection > race_nb.
        priority = {
            SELECTION_TYPE_NAP: 0,
            SELECTION_TYPE_NEXT_BEST: 1,
            SELECTION_TYPE_SELECTION: 2,
            SELECTION_TYPE_RACE_NB: 3,
        }
        picks_sorted = sorted(
            picks, key=lambda p: priority.get(p.selection_type, 99)
        )

        for pick in picks_sorted:
            runner = _match_runner_to_pick(market, pick)
            if runner is None:
                # A pick reached this market on (course, race_time) but its
                # horse is not in the field. Two causes: a genuine non-runner
                # (benign), or a Nags race-integrity fault writing a pick under
                # the wrong race — which silently DROPS A REAL BET. Was
                # logger.debug and therefore invisible at the live log level;
                # raised to warning 19 Jul 2026 after a cross-race next_best
                # was found on the Nags card (Illinois filed under Stratford
                # 15:58 while running Curragh 16:25). Nags CHECK 0b now blocks
                # that at source; this is the downstream tripwire.
                if _tracker.warn_once(
                    f"nomatch:{self.name}:{market.market_id}:{pick.horse}"
                ):
                    logger.warning(
                        "Nags pick has no matching runner — bet SKIPPED "
                        "(non-runner, or pick filed under the wrong race)",
                        horse=pick.horse,
                        selection_type=pick.selection_type,
                        pick_course=pick.course,
                        pick_race_time=pick.race_time,
                        market=market.market_name,
                        venue=market.venue,
                    )
                continue
            if runner.status != "ACTIVE":
                continue
            if not runner.best_back_price:
                continue

            odds = runner.best_back_price
            _tracker.record_bet(self.name, market.market_id)
            return BetSignal(
                market_id=market.market_id,
                selection_id=runner.selection_id,
                selection_name=runner.name,
                bet_type=BetType.BACK,
                odds=odds,
                stake=BACK_FLAT_STAKE,
                strategy=self.name,
                sport=Sport.HORSE_RACING,
                market_name=market.market_name,
                event_name=market.event_name,
                reason=(
                    f"Nags {pick.selection_type} (score "
                    f"{pick.score:.1f})" if pick.score is not None
                    else f"Nags {pick.selection_type}"
                ),
                market_start_time=market.start_time,
                market_type=market.market_type,
            )
        return None


class NagsLayFavStrategy(_NagsStrategyBase):
    """Lay the favourite when Nags backs a different (longer) horse.

    B1a — only fires when:
      * favourite back price is in ``[LAY_FAV_MIN_ODDS, LAY_FAV_MAX_ODDS]``
      * Nags has at least one pick in this race
      * Nags' pick is NOT the favourite
      * Nags' pick is at longer odds than the favourite
      * (optional) Nags has a pick other than ``race_nb``
    """

    name: str = "nags_lay_fav"

    async def evaluate(self, market: Market) -> Optional[BetSignal]:
        if not self.pre_evaluate(market):
            return None

        favourite = market.get_favourite()
        if favourite is None or favourite.best_back_price is None:
            return None

        fav_price = favourite.best_back_price
        if not (LAY_FAV_MIN_ODDS <= fav_price <= LAY_FAV_MAX_ODDS):
            return None

        picks = _picks_for_market(
            market, _index_picks_by_race(self._todays_picks())
        )
        if not picks:
            return None

        if EXCLUDE_RACE_NB_ONLY and all(
            p.selection_type == SELECTION_TYPE_RACE_NB for p in picks
        ):
            return None

        # The pick we care about is the highest-priority non-race_nb
        # one (NAP > NB > selection). If we got here, at least one
        # exists (the EXCLUDE_RACE_NB_ONLY guard above filtered out
        # the all-race_nb case).
        priority = {
            SELECTION_TYPE_NAP: 0,
            SELECTION_TYPE_NEXT_BEST: 1,
            SELECTION_TYPE_SELECTION: 2,
            SELECTION_TYPE_RACE_NB: 3,
        }
        ranked = sorted(picks, key=lambda p: priority.get(p.selection_type, 99))
        primary = ranked[0]

        pick_runner = _match_runner_to_pick(market, primary)
        if pick_runner is None:
            # Same tripwire as NagsBackStrategy above — see that comment.
            if _tracker.warn_once(
                f"nomatch:{self.name}:{market.market_id}:{primary.horse}"
            ):
                logger.warning(
                    "Nags pick has no matching runner — lay SKIPPED "
                    "(non-runner, or pick filed under the wrong race)",
                    horse=primary.horse,
                    selection_type=primary.selection_type,
                    pick_course=primary.course,
                    pick_race_time=primary.race_time,
                    market=market.market_name,
                    venue=market.venue,
                )
            return None

        # Nags picked the favourite — no disagreement, no edge.
        if pick_runner.selection_id == favourite.selection_id:
            return None

        # Confirm Nags' horse is at longer odds than the fav. Prefer
        # live exchange price; fall back to the odds_guide string only
        # if exchange has nothing.
        pick_price = pick_runner.best_back_price or _parse_odds_guide(
            primary.odds_guide
        )
        if pick_price is None or pick_price <= fav_price:
            return None

        # Stake sizing: liability is stake * (odds - 1). Cap at £10.
        lay_price = favourite.best_lay_price or fav_price
        liability_per_unit = lay_price - 1.0
        if liability_per_unit <= 0:
            return None
        stake = round(LAY_LIABILITY_CAP / liability_per_unit, 2)
        if stake < 2.0:  # Betfair £2 minimum
            return None

        _tracker.record_bet(self.name, market.market_id)
        return BetSignal(
            market_id=market.market_id,
            selection_id=favourite.selection_id,
            selection_name=favourite.name,
            bet_type=BetType.LAY,
            odds=lay_price,
            stake=stake,
            strategy=self.name,
            sport=Sport.HORSE_RACING,
            market_name=market.market_name,
            event_name=market.event_name,
            reason=(
                f"Nags backs {primary.horse} ({primary.selection_type}) "
                f"@~{pick_price:.2f}; laying fav @{lay_price:.2f}"
            ),
            market_start_time=market.start_time,
            market_type=market.market_type,
        )


class NagsPlaceStrategy(_NagsStrategyBase):
    """Back the Nags pick in the "To Be Placed" market — the each-way place leg.

    Betfair's exchange has no each-way bet. A bookmaker EW is one stake on the
    WIN and one on the PLACE, so here EW is emulated as two independent bets:
    ``nags_back`` on the WIN market and ``nags_place`` on the PLACE market, on
    the same horse, at the same flat stake.

    Fires only when the pick's WIN odds are ``EACH_WAY_MIN_WIN_ODDS`` (5/1) or
    longer, matching the CLAUDE.md rule. Note the *win* price is taken from the
    Nags ``odds_guide``, not from this market: a PLACE market prices the place,
    so it cannot tell us whether the horse is a 5/1-plus shot.

    Paper-only (see FORCE_PAPER_STRATEGIES) so the EW variant builds its own
    record alongside the live win-only leg before it ever risks money. It also
    draws on its own daily cap, so it cannot starve live ``nags_back``.
    """

    name: str = "nags_place"
    supported_market_types: frozenset[str] = frozenset({"PLACE"})

    async def evaluate(self, market: Market) -> Optional[BetSignal]:
        if not self.pre_evaluate(market):
            return None

        # No bookmaker place pool below 5 runners; Betfair mirrors this by not
        # framing the market, but guard anyway in case one is listed early.
        active = [r for r in market.runners if r.status == "ACTIVE"]
        if len(active) < PLACE_MIN_RUNNERS:
            return None

        # Places paid. Without it the bet cannot be settled later, so skip
        # rather than guess — a wrong place count silently fakes the P&L.
        if not market.number_of_winners:
            logger.debug(
                "Place market has no number_of_winners, skipping",
                market=market.market_id,
            )
            return None

        picks = _picks_for_market(market, _index_picks_by_race(self._todays_picks()))
        if not picks:
            return None

        priority = {
            SELECTION_TYPE_NAP: 0,
            SELECTION_TYPE_NEXT_BEST: 1,
            SELECTION_TYPE_SELECTION: 2,
            SELECTION_TYPE_RACE_NB: 3,
        }
        picks_sorted = sorted(picks, key=lambda p: priority.get(p.selection_type, 99))

        # Resolve THE pick nags_back would back in this race: the first, in
        # priority order, that maps to an active priced runner. Mirrors the
        # skip conditions in NagsBackStrategy.evaluate().
        pick = None
        runner = None
        for candidate in picks_sorted:
            r = _match_runner_to_pick(market, candidate)
            if r is None or r.status != "ACTIVE" or not r.best_back_price:
                continue  # non-runner / unpriced: nags_back would skip it too
            pick, runner = candidate, r
            break

        if pick is None:
            return None

        # From here every failure is TERMINAL — never fall through to a
        # lower-priority pick. This is the each-way leg of the win bet, so it
        # must be the SAME horse nags_back backed or it is not an EW leg at
        # all. (9 Jul 2026: a `continue` here skipped Thunder Call at 7/2 and
        # placed on Calico Blue, a race_nb nags_back never touched.)
        # CLAUDE.md each-way rule (see _ew_place_eligible). `active`
        # (>= PLACE_MIN_RUNNERS, guarded above) is the field on this market.
        is_handicap = "handicap" in (pick.race_name or "").lower()
        num_active = len(active)
        win_odds = _parse_odds_guide(pick.odds_guide)
        if not _ew_place_eligible(is_handicap, num_active, win_odds):
            # Non-handicap with <8 runners or shorter than 3/1 (or a
            # non-handicap with no parseable win price) -> win-only, no leg.
            logger.debug(
                "Nags pick not EW-eligible, no place leg",
                horse=pick.horse,
                win_odds=win_odds,
                num_active=num_active,
                is_handicap=is_handicap,
            )
            return None

        place_odds = runner.best_back_price
        _tracker.record_bet(self.name, market.market_id)
        return BetSignal(
            market_id=market.market_id,
            selection_id=runner.selection_id,
            selection_name=runner.name,
            bet_type=BetType.BACK,
            odds=place_odds,
            stake=PLACE_FLAT_STAKE,
            strategy=self.name,
            sport=Sport.HORSE_RACING,
            market_name=market.market_name,
            event_name=market.event_name,
            reason=(
                f"EW place leg: Nags {pick.selection_type} @ {pick.odds_guide} "
                f"({'handicap' if is_handicap else f'{num_active}r win {win_odds:.2f}'}); "
                f"{market.number_of_winners} places"
            ),
            market_start_time=market.start_time,
            market_type=market.market_type,
            number_of_winners=market.number_of_winners,
        )
