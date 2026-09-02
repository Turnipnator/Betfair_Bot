# Research Notes

Per RESEARCH.md: Question → Hypotheses → Evidence → Confidence → Next Steps.
Newest entry first.

---

## 2026-09-02 — Can LTD be enhanced? Enough leagues? Missing data? Automate BTTS?

### Sub-questions
1. Where does LTD's edge actually sit (version, odds band, competition type)?
2. What does the LTD filter funnel reject, and can we tell whether the filters add value?
3. Which leagues does the scan reach, and where do candidates die for lack of statistics?
4. Which data sources are wired in, how fresh are they, and what is available but unused?
5. Does the Poisson model already produce a BTTS probability, and is there evidence it beats the market?
6. Are Betfair BTTS markets liquid enough to trade at £10?

### Hypotheses
- H1 LTD edge is real and filter-limited: loosening filters adds volume at similar ROI.
- H2 LTD edge is concentrated in a subset; the stats filters are not what carries it.
- H3 The stats feeding the filters are stale/mis-seasoned, so the filters are not doing what the design says.
- H4 Recent break-even months are variance on a small sample.
- H5 (BTTS) The Poisson BTTS probability has edge in the exchange BTTS market because that market is thinner and less sharp than Match Odds.
- H6 (BTTS) The model has no demonstrated edge against Betfair on 1X2, so there is no reason to expect it on BTTS without evidence.

### Evidence

**LTD record (DB, 2 Sep 2026)**

| Version | Bets | Won | Strike | Avg lay odds | Break-even | P&L | ROI |
|---|---|---|---|---|---|---|---|
| v1 pre-match (to 6 Mar) | 142 | 107 | 75.4% | 3.64 | ~71% | +£95.17 | 7.5% |
| v2 HT 0-0 entry (7 Mar on) | 35 | 25 | 71.4% | 2.73 | ~65% | +£73.93 | 21.5% |

v2 by month: Mar +£39.6 (7), Apr +£2.6 (3), May +£32.3 (6), Jun 0 bets, Jul −£4.3 (5), Aug +£3.7 (14).
Jul–Aug combined: 19 bets, 12 won (63%), −£0.57 — at break-even.

v2 split by competition type (from event names):
- European (UCL/UEL/UECL incl. July qualifiers, which **bypass every stats filter**): 18 bets, 14 won (78%), ≈ +£66.
- Domestic (stats-filtered): 17 bets, 11 won (65%), ≈ +£8. That is at the break-even line.
- Confidence: MEDIUM that the split is real, LOW that it is causal. 17 vs 18 bets cannot separate "filters hurt" from variance. Self-critique: July qualifiers are often mismatches, and a 0-0 HT in a mismatch may be structurally more likely to break; that would favour European ties without saying anything about the filters.

v2 by entry odds: 2.2–2.5 band 3 bets 1 won (−£19.6); 2.5+ band 32 bets 24 won (+£93.5). Too few low-band bets to conclude anything. LOW.

**The stats are the wrong season (HIGH — code + live URLs)**
- `src/data/football_data.py` hardcodes `mmz4281/2526/*.csv` (2025/26, a season that ended May 2026). The 2026/27 files exist (`2627/E0.csv`, 20 matches so far on 2 Sep).
- The "new format" leagues (DNK) are filtered to `current_season = "2024/2025"`, a season that ended May 2025. The file contains 2025/2026 and 2026/2027 rows.
- `src/data/understat_data.py` has `CURRENT_SEASON = 2024` (2024/25). Understat serves 2025 and 2026.
- Consequence: every LTD goals/conceded filter and every value-betting Poisson input is last season's full-season average (or two seasons old for Denmark and xG). `home_played >= 3` is always satisfied, so the season-start guard never fires. Promoted/relegated teams get looked up in the wrong division's file.
- This is not automatically bad in August (last season's 38 games beat this season's 2) but it never rolls over, so it decays all season. H3 is confirmed for the mechanism; its P&L effect is unmeasured.

**The funnel cannot be measured (HIGH)**
- Logs rotate at 5 × 10 MB ≈ 2 days. The current window (31 Aug–2 Sep, a quiet midweek) contains 4 distinct LTD fixtures with rejection reasons. Nothing can be inferred about which filter binds.
- `markets.country_code` is empty for every row; `markets.total_matched` is captured when the market is first seen, hours before kick-off, so every LTD bet shows "15–25k". Competition name is fetched from Betfair (`COMPETITION` projection, `Market.competition`) but not persisted.
- Net: there is no stored record of what LTD evaluated and why it passed or failed, so "should we loosen filter X" has been answered from 2-day log windows.

**League coverage (HIGH)**
- Scan: MATCH_ODDS in GB, ES, DE, IT, FR, PT, NL, DK, 0.5–12h ahead, plus a country-less UEFA fetch filtered by keyword.
- Stats: football-data.co.uk E0 E1 SC0 SC1 SP1 SP2 D1 D2 I1 I2 F1 F2 P1 N1 DNK. Tier 1 = Big 5 + P1 N1 SC0 DNK, tier 2 = second divisions.
- "No statistics found" in the current window is dominated by League One/Two, Serie C, Portuguese U23, women's, and domestic cup ties between covered-league teams (Leicester v Plymouth, Sheff Utd v Bolton, Torino v Monza, Parma v Cremonese). Cup ties fail because league detection needs both teams in one league file. By design, not a bug.
- June has zero LTD bets. Summer leagues (Sweden, Norway, Finland, Ireland) have football-data.co.uk "new" files (verified 200 OK, SWE current to 31 Aug 2026) and Betfair markets, but are neither scanned nor in the stats map.

**Data sources available but unused**
- football-data.co.uk "new" files: SWE NOR FIN IRL (+ AUT POL SWZ etc.). Verified live. HIGH availability, MEDIUM value (summer volume for LTD).
- Understat 2025 and 2026 seasons. Verified live. HIGH.
- ClubElo API (Elo for all European divisions, daily CSV): connection failed from this machine (000). Unverified today. LOW until checked from the VPS.
- Betfair's own correlated markets (OVER_UNDER_25, CORRECT_SCORE, BOTH_TEAMS_TO_SCORE) as a market-implied goal model. Needs no external data; not fetched today (DB has only ever seen MATCH_ODDS, WIN, PLACE). MEDIUM.

**BTTS**
- `FootballPoissonModel.predict_match` already returns `btts_prob` and `over_25_prob` from the score matrix. No modelling work needed to get a number. HIGH.
- Plain Poisson assumes home and away goals independent. BTTS is the market most exposed to that assumption (it lives on the 0-0 / 1-0 / 0-1 / 1-1 cells). Dixon-Coles adds one parameter (rho) to correct exactly those cells. Direction of the bias for BTTS depends on the fixture; must be fitted, not assumed. MEDIUM.
- Evidence the model beats Betfair on 1X2: none usable. value_betting all-time 17 bets, 8 won, −£11.28. CLV exists for only 2 bets and both readings (−38.8%, −49.0%, both on winners with close prices of 1.12 and 1.02) are in-play contamination: `record_closing_lines` snapshots `last_price_traded` on **settled** bets, which for football is the final in-play price, not the pre-kick-off close. So the "leading indicator of edge" for value_betting is broken the same way it was for LTD. MEDIUM-HIGH.
- Betfair BTTS liquidity at £10: unknown. Not queried (would need a second Betfair session). GAP.
- Paul's manual BTTS record: not available to this analysis. GAP — it is the only evidence of a BTTS edge and it has not been seen.

### What was ruled out
- "LTD needs more leagues" as the first move: the stats-filtered domestic subsample is at break-even, so adding leagues to the same filters adds volume at ~0 ROI (LOW-MEDIUM; small sample).
- Retuning LTD's thresholds now: impossible to evaluate without a persisted funnel, and the inputs are last season's numbers anyway.
- Automating BTTS on the Poisson probability as it stands: the model's inputs are stale and there is no measurement showing it beats the market on any football market.

### Most supported hypothesis
H3 (stale/mis-seasoned stats) is confirmed as a mechanism. H2 is suggested (edge concentrated in European ties that skip the filters) but the sample is too small to act on. H6 stands until CLV is fixed and shows otherwise.

### Progress (2 Sep 2026, same day)
Steps 1–3 below are built and tested (163 checks across six scripts), not yet
deployed: season-aware blended stats (`football_data.py`, `understat_data.py`),
the persisted funnel (`strategy_evaluations` + `enrich_evaluations`), and the
pre-off CLV rule with startup purge. Deploy = scp `src config scripts tests`
+ rebuild. First useful funnel read-out needs ~2 weeks of fixtures.

### Next steps (ordered)
1. **Season-aware stats loading** in `football_data.py` and `understat_data.py`: derive the season from today's date; blend prior season into current season with a weight that decays as games accumulate (e.g. prior weight = max(0, 1 − games_played/10)). Keep `home_played >= 3` but count blended games. Paper-only impact on LTD until the record shows the change.
2. **Persist the LTD funnel**: a `strategy_evaluations` table (market, competition, reason, key values, HT score, FT score). Also store `competition` on `markets` and snapshot `total_matched` at bet time. Then answer "loosen the favourite filter?" from data. Two weeks of rows is enough to start.
3. **Fix CLV capture**: snapshot the closing price when the market turns in-play (or at T−60s), never after settlement. Without this there is no model-vs-market evidence for any football strategy.
4. **BTTS, in order**: (a) get Paul's manual BTTS bets (date, fixture, price, result) and compare his picks to `btts_prob`; (b) one-off catalogue query for BOTH_TEAMS_TO_SCORE liquidity in the covered leagues; (c) backtest `btts_prob` calibration on 2023/24–2025/26 football-data.co.uk results with and without a Dixon-Coles rho; (d) only then a paper `btts_value` strategy with CLV from a fixed capture.
5. Summer coverage: add SE/NO/FI/IE to the scan and SWE/NOR/FIN/IRL to the stats map, tier 2, paper first. Fills June–July.
6. Verify ClubElo from the VPS; if reachable, it is the cheapest cross-division strength signal (cup ties, promoted teams, "no clear favourite" filter).

### Open questions
- Is the European-vs-domestic split real? Needs ~50 more domestic v2 bets or the funnel table.
- What does Paul's BTTS success look like in numbers?
- Is exchange BTTS liquid enough outside the Big 5?
