"""Season-aware stats and prior-season blending (2 Sep 2026).

Until this fix the football-data.co.uk URLs were hard-coded to 2025/26, the
Danish file was filtered to 2024/25, and Understat was pinned to 2024/25 — so
every LTD goals filter and every Poisson input was a full-season average from
a season that had ended, and it never rolled over. The season is now derived
from the date, and a young season is blended with the one before it.

Run in the betfair-bot container:
  docker compose exec -T -e PYTHONPATH=/app betfair-bot python tests/test_season_blend.py
"""
import pathlib
from datetime import date

from src.data.football_data import (
    LEAGUE_FILES,
    LEAGUE_URLS,
    LeagueStats,
    TeamStats,
    blend_league_stats,
    blend_team_stats,
    league_url,
    parse_league_csv,
    prior_season_weight,
    season_code,
    season_labels,
    season_start_year,
)
from src.data.understat_data import (
    LeagueXGStats,
    TeamXGStats,
    blend_league_xg,
    understat_season,
)

PASS = FAIL = 0


def check(label, got, want):
    global PASS, FAIL
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {label}: got {got!r} want {want!r}")
    PASS += ok
    FAIL += not ok


def approx(label, got, want, tol=1e-6):
    """Float comparison: report the wanted value when within tolerance."""
    check(label, want if abs(got - want) < tol else got, want)


print("season derivation")
check("September 2026 is the 2026/27 season", season_start_year(date(2026, 9, 2)), 2026)
check("June 2026 is still 2025/26", season_start_year(date(2026, 6, 30)), 2025)
check("1 July rolls over", season_start_year(date(2026, 7, 1)), 2026)
check("season_code 2026 -> 2627", season_code(2026), "2627")
check("season_code 1999 -> 9900", season_code(1999), "9900")
check("Understat uses the start year", understat_season(date(2026, 9, 2)), 2026)
check("Understat, spring of the same season", understat_season(date(2027, 3, 1)), 2026)

print("urls and labels")
check("mmz URL carries the season code", league_url("E0", 2026).endswith("/mmz4281/2627/E0.csv"), True)
check("new-format URL is season-less", league_url("DNK", 2026).endswith("/new/DNK.csv"), True)
check("LEAGUE_URLS covers every league file", set(LEAGUE_URLS) == set(LEAGUE_FILES), True)
check("LEAGUE_URLS is this season's, not a constant",
      season_code(season_start_year()) in LEAGUE_URLS["E0"], True)
check("current labels: split-year and calendar-year",
      season_labels(2026, date(2026, 9, 2)), frozenset({"2026/2027", "2026"}))
check("prior labels", season_labels(2025, date(2026, 9, 2)), frozenset({"2025/2026", "2025"}))
check("calendar label follows the calendar in spring",
      season_labels(2026, date(2027, 3, 1)), frozenset({"2026/2027", "2027"}))

print("parsing")
MMZ = (
    "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
    "E0,15/08/2026,Arsenal,Everton,2,0,H\n"
    "E0,16/08/2026,Everton,Chelsea,1,1,D\n"
    "E0,23/08/2026,Chelsea,Arsenal,,,\n"  # fixture, not yet played
)
cur = parse_league_csv(MMZ, "E0")
check("played matches counted", cur.total_matches, 2)
check("unplayed fixture rows are skipped", len(cur.match_results), 2)
check("home goals total", cur.total_home_goals, 3)
check("Arsenal home record", (cur.teams["Arsenal"].home_played, cur.teams["Arsenal"].home_goals_for), (1, 2))
check("Everton played home and away", cur.teams["Everton"].matches_played, 2)

NEW = (
    "Country,League,Season,Date,Time,Home,Away,HG,AG,Res\n"
    "Denmark,Superliga,2025/2026,20/07/2025,17:00,Midtjylland,AGF,3,1,H\n"
    "Denmark,Superliga,2026/2027,19/07/2026,17:00,AGF,Midtjylland,0,2,A\n"
    "Denmark,Superliga,2026/2027,26/07/2026,17:00,Midtjylland,Brondby,1,1,D\n"
)
dnk_cur = parse_league_csv(NEW, "DNK", season_labels(2026, date(2026, 9, 2)))
dnk_prior = parse_league_csv(NEW, "DNK", season_labels(2025, date(2026, 9, 2)))
check("new format: current season rows only", dnk_cur.total_matches, 2)
check("new format: prior season rows only", dnk_prior.total_matches, 1)
check("new format Home/Away columns parsed", "Brondby" in dnk_cur.teams, True)

print("weights")
check("no games: full prior", prior_season_weight(0), 1.0)
check("half way", prior_season_weight(5), 0.5)
check("full games: no prior", prior_season_weight(10), 0.0)
check("never negative", prior_season_weight(15), 0.0)

print("team blending")
current = TeamStats(team_name="Arsenal", home_played=2, home_goals_for=4, home_goals_against=1,
                    away_played=0, matches_played=2)
prior = TeamStats(team_name="Arsenal", home_played=19, home_goals_for=38, home_goals_against=19,
                  home_wins=12, away_played=19, away_goals_for=30, away_goals_against=20, away_wins=9,
                  matches_played=38)
b = blend_team_stats(current, prior, full_games=10)
approx("home games: 2 + 0.8*19", b.home_played, 17.2)
approx("home goals: 4 + 0.8*38", b.home_goals_for, 34.4)
approx("home scored avg holds at 2.0", b.home_scored_avg, 2.0)
approx("away with no games: full prior", b.away_played, 19.0)
approx("away scored avg is last season's", b.away_scored_avg, 30 / 19)
approx("matches_played is the blended sum", b.matches_played, 17.2 + 19.0)
approx("prior weight reported", b.prior_weight, 1.0)
check("no prior: current returned untouched", blend_team_stats(current, None).home_played, 2)
check("no prior: weight 0", blend_team_stats(current, None).prior_weight, 0.0)
full = TeamStats(team_name="X", home_played=12, home_goals_for=12, away_played=11, away_goals_for=11)
approx("12 home games: prior ignored", blend_team_stats(full, prior).home_played, 12.0)
approx("11 away games: prior ignored", blend_team_stats(full, prior).away_played, 11.0)

print("league blending")
cur_l = LeagueStats(league_code="E0", total_matches=1, total_home_goals=2, total_away_goals=0,
                    teams={"Arsenal": TeamStats("Arsenal", home_played=1, home_goals_for=2, matches_played=1),
                           "Everton": TeamStats("Everton", away_played=1, away_goals_against=2, matches_played=1)})
cur_l.match_results = ["this-season-result"]
pri_l = LeagueStats(league_code="E0", total_matches=380, total_home_goals=600, total_away_goals=450,
                    teams={"Arsenal": TeamStats("Arsenal", home_played=19, home_goals_for=40, away_played=19, away_goals_for=30),
                           "Everton": TeamStats("Everton", home_played=19, home_goals_for=20, away_played=19, away_goals_for=15),
                           "Leicester": TeamStats("Leicester", home_played=19, away_played=19)})
pri_l.match_results = ["last-season-result"]
bl = blend_league_stats(cur_l, pri_l, full_games=10)
check("team list is this season's (relegated side gone)", sorted(bl.teams), ["Arsenal", "Everton"])
check("results are this season's only", bl.match_results, ["this-season-result"])
# 2 teams, 1 match => 1 game per team => weight 0.9
approx("league totals blended at 0.9", bl.total_matches, 1 + 0.9 * 380)
approx("league prior weight", bl.prior_weight, 0.9)
approx("Everton home: nothing played, full prior", bl.teams["Everton"].home_goals_for, 20.0)
empty = LeagueStats(league_code="E0")
pre = blend_league_stats(empty, pri_l, full_games=10)
check("pre-season: last season's table used whole", sorted(pre.teams), ["Arsenal", "Everton", "Leicester"])
approx("pre-season: weight 1", pre.prior_weight, 1.0)
check("prior only (current file missing)", blend_league_stats(None, pri_l).total_matches, 380)
check("current only (no prior file)", blend_league_stats(cur_l, None) is cur_l, True)

print("xg blending")
cx = LeagueXGStats(league_code="E0", total_matches=10, total_home_xg=15.0, total_away_xg=12.0,
                   teams={"Arsenal": TeamXGStats("Arsenal", matches_played=1, xg_for=2.0, home_played=1, home_xg_for=2.0)})
px = LeagueXGStats(league_code="E0", total_matches=380, total_home_xg=570.0, total_away_xg=440.0,
                   teams={"Arsenal": TeamXGStats("Arsenal", matches_played=38, xg_for=76.0, home_played=19, home_xg_for=40.0,
                                                 away_played=19, away_xg_for=36.0),
                          "Leicester": TeamXGStats("Leicester", matches_played=38)})
bx = blend_league_xg(cx, px, full_games=10)
check("xg team list is this season's", sorted(bx.teams), ["Arsenal"])
approx("xg home: 2 + 0.9*40", bx.teams["Arsenal"].home_xg_for, 2 + 0.9 * 40)
approx("xg away: full prior", bx.teams["Arsenal"].away_xg_for, 36.0)
approx("xg overall: 2 + 0.9*76", bx.teams["Arsenal"].xg_for, 2 + 0.9 * 76)

print("no season constants left behind")
fd_src = pathlib.Path("src/data/football_data.py").read_text()
us_src = pathlib.Path("src/data/understat_data.py").read_text()
check("no hard-coded 2526 path", "/2526/" in fd_src, False)
check("no hard-coded new-format season", '"2024/2025"' in fd_src, False)
check("no CURRENT_SEASON constant", "CURRENT_SEASON" in us_src, False)

print(f"\nRESULT: {PASS}/{PASS + FAIL} passed")
raise SystemExit(1 if FAIL else 0)
