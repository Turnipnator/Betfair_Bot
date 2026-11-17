"""
How often do Lay the Draw's 0-0 half-time candidates finish drawn?

Research script (read-only, touches no bot state). LTD lays the draw at
about 3.1 at half-time, which breaks even only if fewer than ~32% of those
games finish level. This replays the pre-match filters over past seasons of
football-data.co.uk and counts.

Three populations, all 0-0 at half-time, tier 1/2 leagues with half-time
scores (the 14 mmz4281 divisions; Denmark's file has none):
  A  every game
  B  the odds profile only (favourite <= 2.0, draw 3.0-5.0 at the close)
  C  the full LTD filter: B plus the blended goals filters, replayed with
     only the results known before each kick-off

There are no historical half-time prices. As a stand-in, a Poisson pair is
fitted to each game's closing odds and the second-half share of goals gives
the draw probability from 0-0 at the break. If the filters find games the
market misprices, the actual draw rate will sit below that figure.

Not replayed: the £15k liquidity floor, UEFA ties (no stats, no file), and
the exchange-vs-bookmaker price difference (Pinnacle close is used).

Usage: python3 scripts/research/ltd_base_rate.py <cache_dir>
"""

import csv
import io
import math
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import requests

FD_BASE = "https://football-data.co.uk/mmz4281"
TIERS = {
    "E0": 1, "SP1": 1, "D1": 1, "I1": 1, "F1": 1, "P1": 1, "N1": 1, "SC0": 1,
    "E1": 2, "SP2": 2, "D2": 2, "I2": 2, "F2": 2, "SC1": 2,
}
FIRST_SEASON, LAST_SEASON = 2014, 2026  # start years; the first is prior-only

# The live strategy's thresholds (src/strategies/lay_the_draw.py).
MIN_HOME_GOALS_AVG = 1.5
MIN_AWAY_GOALS_AVG = 0.9
MAX_HOME_CONCEDED_AVG = 1.25
MAX_FAVOURITE_ODDS = 2.0
MIN_PREMATCH_DRAW_ODDS, MAX_PREMATCH_DRAW_ODDS = 3.0, 5.0
MIN_TEAM_HOME_AWAY_GAMES = 3
BLEND_FULL_GAMES = 10
MIN_HT_DRAW_ODDS, MAX_HT_DRAW_ODDS = 1.9, 3.2

COMMISSION = 0.02
STAKE = 10.0
ODDS_COLUMNS = (("PSCH", "PSCD", "PSCA"), ("AvgCH", "AvgCD", "AvgCA"), ("PSH", "PSD", "PSA"),
                ("B365H", "B365D", "B365A"), ("AvgH", "AvgD", "AvgA"), ("BbAvH", "BbAvD", "BbAvA"))

# The bot's own half-time entries (funnel, 5 Sep - 15 Nov 2026):
# (pre-match draw, pre-match favourite, half-time draw price matched).
FUNNEL_ENTRIES = [
    (4.2, 1.69, 2.78), (4.7, 1.57, 2.78), (4.0, 1.71, 2.8), (4.1, 1.65, 2.74), (5.0, 1.44, 3.15),
    (3.6, 1.83, 2.82), (3.55, 2.0, 2.58), (4.8, 1.55, 3.1), (4.1, 1.91, 2.96), (4.2, 1.94, 3.15),
    (4.2, 1.77, 3.1), (5.0, 1.44, 3.2), (4.1, 2.0, 3.1), (3.95, 2.0, 2.86), (4.5, 1.67, 3.1),
    (4.2, 2.0, 3.2), (4.1, 1.98, 3.2), (4.1, 1.99, 3.1), (4.3, 1.63, 3.0), (4.9, 1.61, 3.15),
    (4.8, 1.68, 3.2), (4.4, 1.98, 3.2), (4.8, 1.54, 3.2), (3.4, 1.9, 2.52), (4.7, 1.61, 3.2),
    (5.0, 1.51, 3.2), (4.2, 1.92, 3.05), (4.0, 1.69, 3.1), (4.5, 1.77, 3.1), (4.4, 1.68, 3.2),
    (3.8, 1.96, 3.2), (4.3, 1.62, 3.2), (3.7, 1.91, 2.9), (4.7, 1.55, 3.2),
]


def season_code(start_year: int) -> str:
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def load_season(code: str, start_year: int, cache: Path) -> list[dict]:
    path = cache / f"{code}_{season_code(start_year)}.csv"
    if not path.exists():
        resp = requests.get(f"{FD_BASE}/{season_code(start_year)}/{code}.csv", timeout=60)
        path.write_bytes(resp.content if resp.status_code == 200 else b"")
        time.sleep(0.2)
    rows = []
    text = path.read_bytes().decode("latin-1").lstrip("﻿").lstrip("ï»¿")
    for r in csv.DictReader(io.StringIO(text)):
        try:
            d = r["Date"]
            when = datetime.strptime(d, "%d/%m/%Y" if len(d) == 10 else "%d/%m/%y")
            row = {
                "date": when, "home": r["HomeTeam"], "away": r["AwayTeam"],
                "fthg": int(r["FTHG"]), "ftag": int(r["FTAG"]),
                "hthg": int(r["HTHG"]), "htag": int(r["HTAG"]),
                "league": code, "season": start_year,
            }
        except (KeyError, ValueError, TypeError):
            continue
        row["odds"] = None
        for h, d_, a in ODDS_COLUMNS:
            try:
                odds = (float(r[h]), float(r[d_]), float(r[a]))
            except (KeyError, ValueError, TypeError):
                continue
            if min(odds) > 1.0:
                row["odds"] = odds
                break
        rows.append(row)
    rows.sort(key=lambda x: x["date"])
    return rows


def team_totals(rows: list[dict]) -> dict:
    """Per-team home/away played, scored, conceded over a set of matches."""
    t = defaultdict(lambda: [0.0] * 6)  # hp, hs, hc, ap, as, ac
    for r in rows:
        h, a = t[r["home"]], t[r["away"]]
        h[0] += 1; h[1] += r["fthg"]; h[2] += r["ftag"]
        a[3] += 1; a[4] += r["ftag"]; a[5] += r["fthg"]
    return t


def classify(row: dict, cur: dict, prior: dict) -> str:
    """The pre-match verdict the live strategy would have reached, using the
    season blend from src/data/football_data.py (prior weight 1 - games/10,
    home and away weighted separately; no prior for a promoted side)."""
    def blended(team: str, offset: int) -> tuple[float, float, float]:
        c = cur.get(team, [0.0] * 6)
        p = prior.get(team)
        played, scored, conceded = c[offset], c[offset + 1], c[offset + 2]
        if p is not None:
            w = max(0.0, 1.0 - played / BLEND_FULL_GAMES)
            played += w * p[offset]; scored += w * p[offset + 1]; conceded += w * p[offset + 2]
        return played, scored, conceded

    hp, hs, hc = blended(row["home"], 0)
    ap, as_, _ = blended(row["away"], 3)
    if hp < MIN_TEAM_HOME_AWAY_GAMES or ap < MIN_TEAM_HOME_AWAY_GAMES:
        return "insufficient_games"
    if hs / hp < MIN_HOME_GOALS_AVG:
        return "home_goals"
    if as_ / ap < MIN_AWAY_GOALS_AVG:
        return "away_goals"
    if hc / hp > MAX_HOME_CONCEDED_AVG:
        return "home_conceded"
    return "pass"


def odds_profile(odds: tuple[float, float, float]) -> bool:
    h, d, a = odds
    return MIN_PREMATCH_DRAW_ODDS <= d <= MAX_PREMATCH_DRAW_ODDS and min(h, a) <= MAX_FAVOURITE_ODDS


class PoissonFit:
    """Goal rates from 1X2 odds by nearest match on a (home, away) rate grid."""

    def __init__(self, second_half_share: float, max_goals: int = 10) -> None:
        self.share = second_half_share
        lam = np.arange(0.15, 4.5, 0.02)
        k = np.arange(max_goals + 1)
        fact = np.array([math.factorial(int(i)) for i in k], dtype=float)
        pmf = np.exp(-lam[:, None]) * lam[:, None] ** k / fact  # (L, K)
        cdf_below = np.cumsum(pmf, axis=1) - pmf                # P(X < k)
        # P(home win) = sum_k P(H = k) P(A < k);  P(draw) = sum_k P(H=k) P(A=k)
        self.p_home = (pmf @ cdf_below.T).ravel()
        self.p_draw = (pmf @ pmf.T).ravel()
        self.lam_h = np.repeat(lam, len(lam))
        self.lam_a = np.tile(lam, len(lam))
        self.k, self.fact = k, fact

    def ht_draw_prob(self, odds: np.ndarray) -> np.ndarray:
        """P(finishes drawn | 0-0 at half-time) for each row of 1X2 odds."""
        implied = 1.0 / odds
        implied /= implied.sum(axis=1, keepdims=True)
        out = np.empty(len(odds))
        for i in range(0, len(odds), 500):
            ph, pd = implied[i:i + 500, 0:1], implied[i:i + 500, 1:2]
            idx = np.argmin((self.p_home - ph) ** 2 + (self.p_draw - pd) ** 2, axis=1)
            lh, la = self.lam_h[idx] * self.share, self.lam_a[idx] * self.share
            ph2 = np.exp(-lh[:, None]) * lh[:, None] ** self.k / self.fact
            pa2 = np.exp(-la[:, None]) * la[:, None] ** self.k / self.fact
            out[i:i + 500] = (ph2 * pa2).sum(axis=1)
        return out


def wilson(k: int, n: int) -> tuple[float, float]:
    z, p = 1.96, k / n
    mid = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return mid - half, mid + half


def breakeven(lay_odds: float) -> float:
    """Draw rate at which a lay at `lay_odds` nets zero after commission."""
    return (1 - COMMISSION) / (lay_odds - 1 + (1 - COMMISSION))


def line(label: str, rows: list[dict]) -> str:
    n = len(rows)
    if n == 0:
        return f"{label:<34} n=0"
    k = sum(r["draw"] for r in rows)
    lo, hi = wilson(k, n)
    fair = float(np.mean([r["fair"] for r in rows]))
    nil = sum(r["nilnil"] for r in rows) / n
    return (f"{label:<34} n={n:5d}  drawn {k / n:5.1%} (CI {lo:.1%}-{hi:.1%})  "
            f"odds-implied {fair:5.1%}  0-0 FT {nil:5.1%}")


def lay_pnl(rows: list[dict], markup: float) -> tuple[int, float, float]:
    """Lay each game at its odds-implied half-time price times `markup`,
    entering only inside the bot's 1.9-3.2 window. (bets, P&L, avg price)."""
    n, pnl, prices = 0, 0.0, []
    for r in rows:
        price = markup / r["fair"]
        if not MIN_HT_DRAW_ODDS <= price <= MAX_HT_DRAW_ODDS:
            continue
        n += 1
        prices.append(price)
        pnl += -STAKE * (price - 1) if r["draw"] else STAKE * (1 - COMMISSION)
    return n, pnl, (sum(prices) / n if n else 0.0)


def main() -> None:
    cache = Path(sys.argv[1])
    cache.mkdir(parents=True, exist_ok=True)

    games, h2_goals, all_goals, verdicts = [], 0, 0, defaultdict(int)
    for code in TIERS:
        prior_rows: list[dict] = []
        for year in range(FIRST_SEASON, LAST_SEASON + 1):
            rows = load_season(code, year, cache)
            if year > FIRST_SEASON and rows:
                prior = team_totals(prior_rows) if prior_rows else {}
                cur: dict = defaultdict(lambda: [0.0] * 6)
                for r in rows:
                    h2_goals += r["fthg"] + r["ftag"] - r["hthg"] - r["htag"]
                    all_goals += r["fthg"] + r["ftag"]
                    if r["odds"] is not None:
                        verdict = classify(r, cur, prior)
                        if verdict == "pass" and not odds_profile(r["odds"]):
                            verdict = "odds_profile"
                        verdicts[verdict] += 1
                        if r["hthg"] == 0 and r["htag"] == 0:
                            games.append({
                                **r, "verdict": verdict, "profile": odds_profile(r["odds"]),
                                "draw": int(r["fthg"] == r["ftag"]),
                                "nilnil": int(r["fthg"] + r["ftag"] == 0),
                            })
                    # Only now does this result become known to later games.
                    h, a = cur[r["home"]], cur[r["away"]]
                    h[0] += 1; h[1] += r["fthg"]; h[2] += r["ftag"]
                    a[3] += 1; a[4] += r["ftag"]; a[5] += r["fthg"]
            prior_rows = rows

    share = h2_goals / all_goals
    fit = PoissonFit(share)
    fair = fit.ht_draw_prob(np.array([g["odds"] for g in games]))
    for g, f in zip(games, fair):
        g["fair"] = float(f)

    total = sum(verdicts.values())
    print(f"games with odds {total}; second-half share of goals {share:.3f}")
    print("pre-match verdicts: " + ", ".join(f"{k} {v}" for k, v in sorted(verdicts.items(), key=lambda x: -x[1])))
    print(f"break-even draw rate: lay 2.8 -> {breakeven(2.8):.1%}, 3.0 -> {breakeven(3.0):.1%}, "
          f"3.1 -> {breakeven(3.1):.1%}, 3.2 -> {breakeven(3.2):.1%}")

    a = games
    b = [g for g in games if g["profile"]]
    c = [g for g in games if g["verdict"] == "pass"]
    print("\n--- 0-0 at half-time, seasons 2015/16 to date ---")
    print(line("A all tier 1/2 games", a))
    print(line("B odds profile only", b))
    print(line("C full LTD filter", c))
    print(line("  B minus C (profile, fails goals)", [g for g in b if g["verdict"] != "pass"]))

    print("\n--- C by tier ---")
    for tier in (1, 2):
        print(line(f"tier {tier}", [g for g in c if TIERS[g["league"]] == tier]))
    print("\n--- C by pre-match favourite price ---")
    for lo, hi in ((1.0, 1.4), (1.4, 1.6), (1.6, 1.8), (1.8, 2.0001)):
        print(line(f"favourite {lo:.1f}-{min(hi, 2.0):.1f}", [g for g in c if lo <= min(g["odds"][0], g["odds"][2]) < hi]))
    print("\n--- C by odds-implied half-time draw price ---")
    for lo, hi in ((1.0, 1.9), (1.9, 2.6), (2.6, 2.9), (2.9, 3.2), (3.2, 99)):
        print(line(f"price {lo:.1f}-{hi:.1f}", [g for g in c if lo <= 1 / g["fair"] < hi]))
    print("\n--- C by season ---")
    for year in range(FIRST_SEASON + 1, LAST_SEASON + 1):
        print(line(f"{year}/{(year + 1) % 100:02d}", [g for g in c if g["season"] == year]))

    # How far is the Poisson stand-in from prices the bot was really matched at?
    probe = np.array([[fav, d, 1 / (1 - 1 / fav - 1 / d)] for d, fav, _ in FUNNEL_ENTRIES])
    model = 1 / fit.ht_draw_prob(probe)
    seen = np.array([e[2] for e in FUNNEL_ENTRIES])
    ratio = float(np.mean(seen / model))
    print(f"\nbot's {len(seen)} real HT entries: matched avg {seen.mean():.2f}, "
          f"odds-implied avg {model.mean():.2f}, ratio {ratio:.3f}")

    print("\n--- C: £10 lays inside the 1.9-3.2 window, 2% commission ---")
    for label, markup in (("at the odds-implied price", 1.0), (f"at {ratio:.3f}x (the bot's real fills)", ratio),
                          ("at 1.05x", 1.05), ("at 1.10x", 1.10)):
        n, pnl, avg = lay_pnl(c, markup)
        if n:
            print(f"{label:<34} bets {n:5d}  avg price {avg:.2f}  P&L £{pnl:+9.2f}  per bet £{pnl / n:+.2f}")


if __name__ == "__main__":
    main()
