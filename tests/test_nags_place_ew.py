"""Widened each-way place-leg eligibility + live switch (27 Jul 2026).

nags_place used to fire only on 5/1+ picks (missing the shorter-priced picks
where most of our places come from) and ran paper-only. It now uses the real
CLAUDE.md rule -- handicaps always E/W; non-handicaps E/W at 8+ runners AND
3/1+ -- and is LIVE. These lock that contract.

Run in the betfair-bot container:
  docker compose exec -T -e PYTHONPATH=/app betfair-bot python tests/test_nags_place_ew.py
"""
from src.strategies.horse_racing import (
    _ew_place_eligible,
    FORCE_PAPER_STRATEGIES,
    EACH_WAY_MIN_WIN_ODDS,
    EACH_WAY_MIN_RUNNERS_NONHCAP,
)

PASS = FAIL = 0


def check(label, got, want):
    global PASS, FAIL
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {label}: got {got!r} want {want!r}")
    PASS += ok
    FAIL += not ok


print("constants == CLAUDE.md rule")
check("odds floor 3/1 == 4.0 decimal", EACH_WAY_MIN_WIN_ODDS, 4.0)
check("non-handicap field floor == 8", EACH_WAY_MIN_RUNNERS_NONHCAP, 8)

print("handicaps: ALWAYS each-way (any field, any odds)")
check("handicap 5r @ 6/4 (2.5)", _ew_place_eligible(True, 5, 2.5), True)
check("handicap 6r no parseable odds", _ew_place_eligible(True, 6, None), True)
check("handicap 12r @ 9/1", _ew_place_eligible(True, 12, 10.0), True)
check("handicap 9r @ 2/1 (was NOT eligible under 5/1 rule)", _ew_place_eligible(True, 9, 3.0), True)

print("non-handicap: needs 8+ runners AND 3/1+")
check("nonhcap 8r @ 3/1 (4.0) -> yes", _ew_place_eligible(False, 8, 4.0), True)
check("nonhcap 10r @ 3/1 -> yes", _ew_place_eligible(False, 10, 4.0), True)
check("nonhcap 8r @ 11/4 (3.75) -> no (<3/1)", _ew_place_eligible(False, 8, 3.75), False)
check("nonhcap 8r @ 5/2 (3.5) -> no", _ew_place_eligible(False, 8, 3.5), False)
check("nonhcap 7r @ 5/1 -> no (field<8)", _ew_place_eligible(False, 7, 6.0), False)
check("nonhcap 8r no odds -> no (can't confirm 3/1)", _ew_place_eligible(False, 8, None), False)
check("nonhcap 20r @ 5/1 -> yes", _ew_place_eligible(False, 20, 6.0), True)

print("live/paper switch")
check("nags_place is LIVE (not forced-paper)", "nags_place" in FORCE_PAPER_STRATEGIES, False)
check("nags_lay_fav still forced-paper", "nags_lay_fav" in FORCE_PAPER_STRATEGIES, True)

print(f"\nRESULT: {PASS}/{PASS + FAIL} passed")
raise SystemExit(1 if FAIL else 0)
