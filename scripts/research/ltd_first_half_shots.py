"""
Do first-half shots separate the 0-0 half-time games that stay drawn?

Research script (read-only, touches no bot state). Lay the Draw enters at
0-0 at half-time; 6 of its last 10 losing draws finished 0-0. This asks
whether first-half shot volume would have told those games apart.

Data: StatsBomb open data, 2015/16, the four leagues it publishes in full
(Premier League, La Liga, Serie A, Ligue 1), joined to football-data.co.uk
for the half-time score and closing prices. Understat was the first choice
(more seasons) but was serving a maintenance page on 17 Nov 2026.

The cuts are fixed here, before looking at any result, to limit data
snooping: three measures (total shots, shots on target, non-penalty xG,
both teams, first half), each in terciles, on two populations (every 0-0
half-time game; the LTD candidate profile: favourite <= 2.0 and draw
3.0-5.0 at the close).

Usage: python3 scripts/research/ltd_first_half_shots.py <cache_dir>
"""

import csv
import io
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path

import requests

SB_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
FD_BASE = "https://football-data.co.uk/mmz4281/1516"
SB_SEASON_2015_16 = 27
# StatsBomb competition id -> football-data.co.uk division code. The
# Bundesliga is left out: the open data holds 34 of its 306 matches.
LEAGUES = {2: "E0", 11: "SP1", 12: "I1", 7: "F1"}

ON_TARGET_OUTCOMES = {"Goal", "Saved", "Saved to Post"}
MAX_FAVOURITE_ODDS = 2.0
MIN_DRAW_ODDS, MAX_DRAW_ODDS = 3.0, 5.0
DOWNLOAD_WORKERS = 6


def fetch(url: str, path: Path) -> bytes:
    """Download once, then serve from the cache directory."""
    if path.exists() and path.stat().st_size > 0:
        return path.read_bytes()
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return resp.content


def load_fd(code: str, cache: Path) -> list[dict]:
    """football-data.co.uk rows with parsed date, scores and closing prices."""
    raw = fetch(f"{FD_BASE}/{code}.csv", cache / f"fd_{code}.csv")
    rows = []
    for r in csv.DictReader(io.StringIO(raw.decode("latin-1"))):
        if not r.get("Date") or not r.get("HomeTeam"):
            continue
        fmt = "%d/%m/%Y" if len(r["Date"]) == 10 else "%d/%m/%y"
        try:
            rows.append({
                "date": datetime.strptime(r["Date"], fmt).date(),
                "home": r["HomeTeam"], "away": r["AwayTeam"],
                "fthg": int(r["FTHG"]), "ftag": int(r["FTAG"]),
                "hthg": int(r["HTHG"]), "htag": int(r["HTAG"]),
                # Pinnacle close where present, Bet365 otherwise.
                "h": float(r.get("PSCH") or r.get("B365H") or 0),
                "d": float(r.get("PSCD") or r.get("B365D") or 0),
                "a": float(r.get("PSCA") or r.get("B365A") or 0),
            })
        except ValueError:
            continue
    return rows


def name_score(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def join(sb_matches: list[dict], fd_rows: list[dict]) -> tuple[list[tuple[dict, dict]], int]:
    """Pair each StatsBomb match with its football-data row: same score,
    date within a day, best team-name similarity."""
    pairs, unmatched, used = [], 0, set()
    for m in sb_matches:
        day = datetime.strptime(m["match_date"], "%Y-%m-%d").date()
        best, best_score = None, 0.0
        for i, r in enumerate(fd_rows):
            if i in used or abs(r["date"] - day) > timedelta(days=1):
                continue
            if (r["fthg"], r["ftag"]) != (m["home_score"], m["away_score"]):
                continue
            s = name_score(m["home_team"]["home_team_name"], r["home"]) + name_score(
                m["away_team"]["away_team_name"], r["away"]
            )
            if s > best_score:
                best, best_score = i, s
        if best is None:
            unmatched += 1
            continue
        used.add(best)
        pairs.append((m, fd_rows[best]))
    return pairs, unmatched


def first_half_shots(match_id: int, cache: Path) -> dict:
    """First-half shot counts for both teams combined, penalties excluded."""
    path = cache / f"shots_{match_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    resp = requests.get(f"{SB_BASE}/events/{match_id}.json", timeout=120)
    resp.raise_for_status()
    out = {"shots": 0, "sot": 0, "xg": 0.0, "goals_h1": 0, "max_team_shots": 0}
    per_team: dict[str, int] = {}
    for e in resp.json():
        if e.get("period") != 1:
            continue
        if e["type"]["name"] == "Own Goal For":
            out["goals_h1"] += 1
        if e["type"]["name"] != "Shot":
            continue
        shot = e["shot"]
        outcome = shot["outcome"]["name"]
        if outcome == "Goal":
            out["goals_h1"] += 1
        if shot["type"]["name"] == "Penalty":
            continue
        out["shots"] += 1
        out["sot"] += outcome in ON_TARGET_OUTCOMES
        out["xg"] += shot.get("statsbomb_xg", 0.0)
        team = e["team"]["name"]
        per_team[team] = per_team.get(team, 0) + 1
    out["max_team_shots"] = max(per_team.values(), default=0)
    # Only the summary is cached; the event files are ~3 MB each.
    path.write_text(json.dumps(out))
    return out


def wilson(k: int, n: int) -> tuple[float, float]:
    """95% Wilson interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    z, p = 1.96, k / n
    mid = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (mid - half, mid + half)


def trend_p(xs: list[float], ys: list[int]) -> tuple[float, float]:
    """Correlation of the raw measure with 'finished drawn' and its
    two-sided p-value (normal approximation; n is in the hundreds)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return (0.0, 1.0)
    r = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sxx * syy)
    t = r * math.sqrt((n - 2) / max(1e-12, 1 - r * r))
    return (r, math.erfc(abs(t) / math.sqrt(2)))


def report(title: str, rows: list[dict]) -> None:
    n = len(rows)
    draws = sum(r["draw"] for r in rows)
    nilnil = sum(r["nilnil"] for r in rows)
    print(f"\n=== {title}: {n} games 0-0 at HT ===")
    if n == 0:
        return
    lo, hi = wilson(draws, n)
    print(f"finished drawn {draws} ({draws / n:.1%}, 95% CI {lo:.0%}-{hi:.0%}); "
          f"finished 0-0 {nilnil} ({nilnil / n:.1%})")
    for key, label in (("shots", "total shots"), ("sot", "shots on target"), ("xg", "xG")):
        ordered = sorted(rows, key=lambda r: r[key])
        r_val, p_val = trend_p([r[key] for r in rows], [r["draw"] for r in rows])
        print(f"\n  first-half {label}  (correlation with a draw r={r_val:+.3f}, p={p_val:.2f})")
        third = n // 3
        for name, chunk in (("low ", ordered[:third]), ("mid ", ordered[third:n - third]),
                            ("high", ordered[n - third:])):
            k = sum(r["draw"] for r in chunk)
            z = sum(r["nilnil"] for r in chunk)
            lo, hi = wilson(k, len(chunk))
            rng = f"{chunk[0][key]:.2f}-{chunk[-1][key]:.2f}" if key == "xg" else f"{chunk[0][key]}-{chunk[-1][key]}"
            print(f"    {name} {rng:>10}  n={len(chunk):3d}  drawn {k / len(chunk):5.1%} "
                  f"(CI {lo:.0%}-{hi:.0%})  0-0 {z / len(chunk):5.1%}")


def main() -> None:
    cache = Path(sys.argv[1])
    cache.mkdir(parents=True, exist_ok=True)

    todo, total, unmatched_total = [], 0, 0
    for comp, code in LEAGUES.items():
        sb = json.loads(fetch(f"{SB_BASE}/matches/{comp}/{SB_SEASON_2015_16}.json",
                              cache / f"sb_matches_{comp}.json"))
        pairs, unmatched = join(sb, load_fd(code, cache))
        total += len(sb)
        unmatched_total += unmatched
        todo += [(m, r, code) for m, r in pairs if r["hthg"] == 0 and r["htag"] == 0]
    print(f"matches {total}, unjoined {unmatched_total}, 0-0 at HT {len(todo)}")

    with ThreadPoolExecutor(DOWNLOAD_WORKERS) as pool:
        shots = list(pool.map(lambda t: first_half_shots(t[0]["match_id"], cache), todo))

    rows, disagreements = [], 0
    for (m, r, code), s in zip(todo, shots):
        # The two sources must agree the half ended 0-0, or the join is wrong.
        if s["goals_h1"] != 0:
            disagreements += 1
            continue
        fav = min(r["h"], r["a"])
        rows.append({
            **s, "league": code,
            "draw": int(r["fthg"] == r["ftag"]),
            "nilnil": int(r["fthg"] == 0 and r["ftag"] == 0),
            "candidate": 0 < fav <= MAX_FAVOURITE_ODDS and MIN_DRAW_ODDS <= r["d"] <= MAX_DRAW_ODDS,
        })
    print(f"dropped {disagreements} where the sources disagree on the HT score")

    report("All games", rows)
    report("LTD candidate profile (fav <= 2.0, draw 3.0-5.0)", [r for r in rows if r["candidate"]])


if __name__ == "__main__":
    main()
