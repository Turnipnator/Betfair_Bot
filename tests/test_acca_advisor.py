"""Acca advisor (src/acca): pricing, signal, builder, settlement, ledger.

Advisory only; nothing here touches the live bot's ledger. The end-to-end
section drives the engine against a throwaway acca.db with a fake scanner:
two scans in which prices shorten, a dry-run acca proposed from them, the
closing line captured, and settlement with a void leg.

Run locally:  .venv/bin/python tests/test_acca_advisor.py
In the container:
  docker compose exec -T -e PYTHONPATH=/app acca-advisor python tests/test_acca_advisor.py
"""
import asyncio
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.acca import engine as engine_module  # noqa: E402
from src.acca.builder import (  # noqa: E402
    Leg,
    choose_legs,
    eligible_legs,
    propose,
    suggest_stake,
)
from src.acca.config import AccaSettings  # noqa: E402
from src.acca.engine import AccaEngine  # noqa: E402
from src.acca.messages import Placement, format_alert, parse_placement  # noqa: E402
from src.acca.pricing import devig  # noqa: E402
from src.acca.settlement import acca_result, acca_return, selection_result  # noqa: E402
from src.acca.signal import Quote, measure_move, qualifies  # noqa: E402
from src.acca.store import AccaStore  # noqa: E402
from src.models import Market, PriceSize, Runner, Sport  # noqa: E402

PASS = FAIL = 0


def check(label, got, want):
    global PASS, FAIL
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {label}: got {got!r} want {want!r}")
    PASS += ok
    FAIL += not ok


def close(label, got, want, tol=1e-6):
    global PASS, FAIL
    ok = got is not None and abs(got - want) <= tol
    print(f"  {'PASS' if ok else 'FAIL'}  {label}: got {got!r} want {want!r}")
    PASS += ok
    FAIL += not ok


def cfg(**over) -> AccaSettings:
    base = dict(_env_file=None, alerts_enabled=False)
    base.update(over)
    return AccaSettings(**base)


def runner(sid, name, back, lay, prio=0, ltp=None):
    return Runner(
        selection_id=sid, name=name, sort_priority=prio, last_price_traded=ltp,
        back_prices=[PriceSize(back, 100)] if back else [],
        lay_prices=[PriceSize(lay, 100)] if lay else [],
    )


def mo_market(mid="1.1", event="Arsenal v Chelsea", eid=10, prices=((2.0, 2.02), (4.0, 4.1), (3.5, 3.6)),
              matched=20000.0, start=None, competition="English Premier League"):
    (hb, hl), (ab, al), (db, dl) = prices
    return Market(
        market_id=mid, market_name="Match Odds", event_name=event, sport=Sport.FOOTBALL,
        market_type="MATCH_ODDS", start_time=start or datetime.utcnow() + timedelta(hours=5),
        competition=competition, event_id=eid, total_matched=matched,
        runners=[
            runner(1, event.split(" v ")[0], hb, hl, 1),
            runner(2, event.split(" v ")[1], ab, al, 2),
            runner(3, "The Draw", db, dl, 3),
        ],
    )


C = cfg()

print("devig: midpoint in probability space, proportional normalisation")
pm = devig(mo_market(), C)
check("trusted", pm.trusted, True)
mids = [(1 / 2.0 + 1 / 2.02) / 2, (1 / 4.0 + 1 / 4.1) / 2, (1 / 3.5 + 1 / 3.6) / 2]
home = next(s for s in pm.selections if s.key == "1.1:1")
close("home fair prob", home.fair_prob, mids[0] / sum(mids))
close("match odds sum to 1", sum(s.fair_prob for s in pm.selections if s.market_type == "MATCH_ODDS"), 1.0)
check("low matched rejected", devig(mo_market(matched=4999), C).reject_reason, "low_matched")
check("wide spread rejected",
      devig(mo_market(prices=((2.0, 2.3), (4.0, 4.1), (3.5, 3.6))), C).reject_reason, "wide_spread")
check("one-sided rejected",
      devig(mo_market(prices=((2.0, None), (4.0, 4.1), (3.5, 3.6))), C).reject_reason, "one_sided")
m = mo_market()
m.in_play = True
check("in-play rejected", devig(m, C).reject_reason, "not_pre_off")
m = mo_market()
m.runners[0].last_price_traded = 2.6
check("last traded far from mid rejected", devig(m, C).reject_reason, "ltp_divergent")

print("derived legs")
p = {s.winning_ids[0]: s.fair_prob for s in pm.selections if s.market_type == "MATCH_ODDS"}
dc = next(s for s in pm.selections if s.key == "1.1:DC:HD")
close("double chance home-or-draw = p(H)+p(D)", dc.fair_prob, p[1] + p[3])
check("DC label", dc.label, "Arsenal or Draw")
dnb = next(s for s in pm.selections if s.key == "1.1:DNB:H")
close("DNB fair odds = 1/P(home | not draw)", dnb.fair_odds, (p[1] + p[2]) / p[1])
close("DNB is EV-neutral at fair odds (draw returns stake)", p[1] * dnb.fair_odds + p[3], 1.0)

print("signal: shortening on volume")
t0 = datetime(2026, 9, 26, 10, 0)
hist = [Quote(t0, 1 / 2.2, 10000), Quote(t0 + timedelta(hours=2), 1 / 2.05, 13000)]
mv = measure_move(hist, t0 + timedelta(hours=2), 6)
close("shortening 2.20 -> 2.05", mv.shortening, 2.2 / 2.05 - 1)
check("volume during move", mv.volume, 3000)
check("qualifies", qualifies(mv, C), True)
thin = [Quote(t0, 1 / 2.2, 10000), Quote(t0 + timedelta(hours=2), 1 / 2.05, 10500)]
check("price move on no money does not", qualifies(measure_move(thin, t0 + timedelta(hours=2), 6), C), False)
check("outside lookback ignored", measure_move(hist, t0 + timedelta(hours=9), 6), None)
small = [Quote(t0, 1 / 2.1, 10000), Quote(t0 + timedelta(hours=1), 1 / 2.05, 15000)]
check("2.4% move too small", qualifies(measure_move(small, t0 + timedelta(hours=1), 6), C), False)

print("builder")
now = datetime(2026, 9, 26, 10, 0)


def leg(key, event, odds, short, hours=5.0, comp="EPL", eid=None):
    return Leg(key=key, event_id=eid or event, event_name=event, competition=comp,
               kickoff=now + timedelta(hours=hours), market_type="MATCH_ODDS", label=key,
               fair_odds=odds, shortening=short, move_volume=5000)


pool = [
    leg("a", "A v B", 1.8, 0.08),
    leg("a2", "A v B", 1.5, 0.06),  # same match, weaker: dropped
    leg("c", "C v D", 2.0, 0.07),
    leg("e", "E v F", 1.6, 0.10, hours=0.5),  # inside the 60 min lead
    leg("g", "G v H", 6.0, 0.20),  # above max leg odds
    leg("i", "I v J", 1.7, 0.05, hours=60),  # outside 48h window
]
el = eligible_legs(pool, now, C)
check("one leg per match, in window, odds range", sorted(x.key for x in el), ["a", "c"])
check("two legs, not closing: waits for a third", choose_legs(pool, now, C), None)
soon = [leg("a", "A v B", 1.8, 0.08, hours=1.1), leg("c", "C v D", 2.0, 0.07)]
check("two legs, first about to close: fires", [x.key for x in choose_legs(soon, now, C)], ["a", "c"])
three = [leg("a", "A v B", 1.8, 0.08), leg("c", "C v D", 2.0, 0.07), leg("k", "K v L", 1.5, 0.09)]
chosen = choose_legs(three, now, C)
check("target reached: strongest first", [x.key for x in chosen], ["k", "a", "c"])

prop = propose(chosen, 0.0, 0.0, C)
fair = 1.5 * 1.8 * 2.0
close("combined fair odds", prop.combined_fair_odds, fair)
close("combined fair prob", prop.combined_fair_prob, 1 / fair)
close("min combined = fair x 1.05^n", prop.min_combined_odds, fair * 1.05 ** 3)
check("each leg min = fair x 1.05", [round(x.min_odds, 4) for x in chosen], [1.575, 1.89, 2.1])
check("weakest leg = longest fair odds", prop.weakest_leg_key, "c")
# quarter Kelly at the minimum: f = (1.05^3 - 1) / (min - 1)
kelly = (1.05 ** 3 - 1) / (fair * 1.05 ** 3 - 1) * 0.25 * 100
check("stake = quarter Kelly, 10p floor", prop.suggested_stake, int(kelly * 10) / 10)
check("stake capped at 1% of bank", suggest_stake(0.6, 2.0, 0, 0, C), (1.0, "capped at 1% of bank"))
check("cut to daily headroom", suggest_stake(0.6, 2.0, 2.5, 2.5, C), (0.5, "cut to daily/weekly limit headroom"))
check("weekly limit reached", suggest_stake(0.6, 2.0, 0, 10.0, C), (0.0, "limit reached, no stake"))

print("correlation flags")
flagged = propose([leg("x", "Arsenal v Spurs", 1.8, 0.1, eid="1"),
                   leg("y", "Brighton v Arsenal", 1.8, 0.1, hours=30, eid="2")], 0, 0, C)
check("same team in two fixtures flagged", any("Arsenal" in f for f in flagged.legs[0].flags), True)
sim = propose([leg("x", "A v B", 1.8, 0.1), leg("y", "C v D", 1.8, 0.1)], 0, 0, C)
check("same competition + kick-off flagged", bool(sim.legs[1].flags), True)
text = format_alert(7, prop)
check("alert says take at X or better", "take at 1.58 or better" in text, True)
check("no likely-winner language", any(w in text.lower() for w in ("likely", "banker", "certain")), False)

print("settlement")
check("won", selection_result([1], [], {1: "WINNER", 2: "LOSER", 3: "LOSER"}, "CLOSED"), "WON")
check("lost", selection_result([1], [], {1: "LOSER", 2: "WINNER", 3: "LOSER"}, "CLOSED"), "LOST")
check("DC won on the draw", selection_result([1, 3], [], {1: "LOSER", 2: "LOSER", 3: "WINNER"}, "CLOSED"), "WON")
check("DNB void on the draw", selection_result([1], [3], {1: "LOSER", 2: "LOSER", 3: "WINNER"}, "CLOSED"), "VOID")
check("voided market", selection_result([1], [], {1: "REMOVED", 2: "REMOVED", 3: "REMOVED"}, "CLOSED"), "VOID")
check("closed, no single winner: never guessed",
      selection_result([1], [], {1: "LOSER", 2: "LOSER", 3: "LOSER"}, "CLOSED"), "UNKNOWN")
check("not closed: pending", selection_result([1], [], {1: "ACTIVE"}, "SUSPENDED"), None)
check("any leg lost settles at once", acca_result(["WON", None, "LOST"]), "LOST")
check("waits for pending legs", acca_result(["WON", None]), None)
check("void leg: acca continues", acca_result(["WON", "VOID", "WON"]), "WON")
check("all void returns stake", acca_result(["VOID", "VOID"]), "VOID")
check("per-leg odds: void counts 1.00", acca_return("WON", 2.0, 0, ["WON", "VOID", "WON"], [2.0, 3.0, 1.5], [1, 1, 1]),
      (6.0, False))
check("combined only, no void: exact", acca_return("WON", 2.0, 10.0, ["WON", "WON"], None, [2.0, 3.0]), (20.0, False))
gross, est = acca_return("WON", 1.0, 8.4, ["WON", "VOID"], None, [2.0, 4.0])
check("combined only with a void: estimated and flagged", (round(gross, 4), est), (round(8.4 / (4.0 * (8.4 / 8) ** 0.5), 4), True))

print("placement reply")
check("odds stake bookie", parse_placement("14.2 1.00 bet365", 3), Placement(14.2, 1.0, None, "bet365"))
check("pound sign", parse_placement("14.2 £1", 3), Placement(14.2, 1.0, None, None))
pp = parse_placement("2 2.5 3 / 0.5 sky bet", 3)
check("per-leg odds", (round(pp.combined_odds, 6), pp.leg_odds, pp.stake, pp.bookmaker), (15.0, [2.0, 2.5, 3.0], 0.5, "sky bet"))
check("wrong leg count", parse_placement("2 2.5 / 1", 3), "Expected 3 leg prices before the /.")
check("garbage", isinstance(parse_placement("yes", 3), str), True)


# ---------------------------------------------------------------- end to end


class FakeScanner:
    def __init__(self):
        self.markets = []
        self.results_map = {}

    async def fetch(self, cfg, extra):
        return self.markets

    async def results(self, ids):
        return {i: self.results_map[i] for i in ids if i in self.results_map}


def fixtures(ko, home_back):
    """Three matches in different competitions; home price given per scan."""
    out = []
    for i, (name, comp) in enumerate([("A v B", "L1"), ("C v D", "L2"), ("E v F", "L3")]):
        hb = home_back[i]
        # The other two drift as the home side shortens, keeping the back
        # book near 101% as a real market does.
        rest = 1.01 - 1 / hb
        ab, db = round(1 / (rest * 0.45), 2), round(1 / (rest * 0.55), 2)
        out.append(mo_market(mid=f"1.{i}", event=name, eid=100 + i, competition=comp, start=ko,
                             prices=((hb, round(hb + 0.02, 2)), (ab, round(ab + 0.1, 2)),
                                     (db, round(db + 0.1, 2)))))
    return out


async def e2e():
    tmp = tempfile.mkdtemp()
    store = AccaStore(os.path.join(tmp, "acca.db"))
    await store.initialize()
    scanner = FakeScanner()
    c = cfg(min_move_volume=1000)
    eng = AccaEngine(c, store, scanner, None)
    t = datetime(2026, 9, 26, 9, 0)
    ko = t + timedelta(hours=6)

    engine_module.utcnow = lambda: t
    scanner.markets = fixtures(ko, [2.2, 2.3, 2.1])
    await eng.scan()
    check("first scan: no move yet, no acca", len(await store.accas_since(t - timedelta(days=1))), 0)

    t = t + timedelta(hours=1)
    engine_module.utcnow = lambda: t
    scanner.markets = fixtures(ko, [1.9, 2.0, 1.8])
    for mk in scanner.markets:
        mk.total_matched += 3000
    await eng.scan()
    accas = await store.accas_since(t - timedelta(days=1))
    check("second scan: one dry-run acca", [(a.status, a.n_legs) for a in accas], [("dry_run", 3)])
    legs = await store.get_legs(accas[0].id)
    check("legs are the shortened home sides", sorted(s.key for _, s in legs), ["1.0:1", "1.1:1", "1.2:1"])

    await eng.scan()
    check("a match is not offered twice (not even as DC/DNB)", len(await store.accas_since(t - timedelta(days=1))), 1)

    # Placed with per-leg odds, then a pre-off price as the closing line.
    acca_id = accas[0].id
    await store.update_acca(acca_id, status="placed", decided_at=t, taken_odds=10.0, taken_stake=1.0)
    taken = {"1.0:1": 2.2, "1.1:1": 2.3, "1.2:1": 2.1}  # A, C, E
    await store.set_leg_taken_odds(acca_id, [taken[s.key] for _, s in legs])
    t = ko - timedelta(minutes=10)
    engine_module.utcnow = lambda: t
    scanner.markets = fixtures(ko, [1.85, 1.95, 1.75])
    await eng.scan()

    t = ko + timedelta(hours=3)
    engine_module.utcnow = lambda: t
    scanner.results_map = {
        "1.0": ("CLOSED", {1: "WINNER", 2: "LOSER", 3: "LOSER"}),
        "1.1": ("CLOSED", {1: "REMOVED", 2: "REMOVED", 3: "REMOVED"}),  # abandoned
        "1.2": ("CLOSED", {1: "WINNER", 2: "LOSER", 3: "LOSER"}),
    }
    await eng.settle()
    acca = await store.get_acca(acca_id)
    check("acca won with a void leg", acca.result, "WON")
    by_key = {s.key: leg for leg, s in await store.get_legs(acca_id)}
    close("void leg (C) counts 1.00: return = 2.2 x 2.1", acca.gross_return, 2.2 * 2.1)
    check("exact, not estimated", acca.return_estimated, False)
    leg0 = by_key["1.0:1"]
    check("closing line captured pre-off", leg0.close_fair_odds is not None, True)
    close("taken CLV = taken / close - 1",
          leg0.clv_taken, (2.2 / leg0.close_fair_odds - 1) * 100)
    check("alert CLV positive (kept shortening)", leg0.clv_alert > 0, True)
    check("void leg result", by_key["1.1:1"].result, "VOID")
    await store.close()


asyncio.run(e2e())

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
