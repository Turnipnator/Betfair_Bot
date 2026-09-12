# Research Notes

Per RESEARCH.md: Question → Hypotheses → Evidence → Confidence → Next Steps.
Newest entry first.

---

## 2026-09-12 — Why does LTD v2 open so few positions, and what would add volume?

First read of the persisted funnel (`strategy_evaluations`, 2–12 Sep 2026, 672 markets) plus a replay of the
stats lookup inside the VPS container. Diagnostic script kept as `scripts/research/ltd_nostats_replay.py`
(monkey-patches httpx to follow redirects; run via `docker cp` + `docker exec -w /app`). Read-only, no bets.

### Sub-questions
1. Where in the funnel do fixtures die, and which rejections are large enough to matter?
2. Of the candidates, why do so few become entries?
3. Is `no_stats` (the largest rejection) mostly uncovered leagues, or covered leagues failing to resolve?
4. Do the counterfactual HT/FT scores say any filter should be loosened?
5. Is the summer gap (June = 0 bets) a coverage problem?

### Hypotheses
- H1 Filter-limited: the stats filters reject good candidates; loosen them.
- H2 Coverage-limited: too few fixtures reach the stats path (leagues, name resolution).
- H3 Execution-limited: candidates reach HT 0-0 but the entry rule misses them.
- H4 The HT 0-0 requirement caps volume inherently and nothing cheap changes that.

### Evidence

**Funnel, 2–12 Sep (HIGH — it is the persisted record)**

| Stage / reason | n | share |
|---|---|---|
| prematch rejected `no_stats` | 447 | 66 % |
| prematch rejected `home_goals` | 100 | 15 % |
| prematch rejected `youth_reserve` | 49 | 7 % |
| prematch **candidate** | 22 | 3.3 % |
| prematch rejected `no_clear_favourite` / `home_conceded` / `draw_odds_range` / `away_goals` / `liquidity` | 15 / 13 / 11 / 10 / 5 | 8 % combined |
| halftime dropped `goal_before_ht` | 18 of 22 | |
| halftime **entered** | 4 of 22 | |

Weekly v2 volume since 7 Mar: 1–3 bets/week, 0 in June, 5 in the best week (W31).

**Q2 — the 2.8 HT cap is binding and it delays entry into the second half (HIGH for the mechanism, MEDIUM for the cost)**
- All 35 v2 entries by matched odds: 2.5 ×3, 2.6 ×4, **2.7 ×23, 2.8 ×7**, 3.8 ×1. Thirty of 35 sit in the top 0.1 of a
  1.9–2.8 window. The four entries this window were at 2.78, 2.78, 2.80, 2.74 at match minutes 44, 52, 49, 56.
- A 0-0 HT draw for the candidate profile (favourite ≤ 2.0) trades at ~2.9–3.2 at the whistle. The rule allows
  entry while `status == HalfTime` or 40–65′, so the bot sits through HT rejecting on `ht_odds_range` and enters
  when the price drifts under 2.8 around 50–56′. (The funnel keeps only the latest verdict per stage, so the
  `ht_odds_range` → `ht_entry` progression is invisible in the table; the odds histogram is the evidence.)
- Cost this window: 7 candidates were 0-0 at HT; 4 entered; **3 were lost to a goal at 46′, 52′ and 52′**
  (Dortmund v Villarreal 3-2, Porto v Man City 0-2, Rennes v Marseille 1-0). All three finished non-draw — the delay
  filters out early-second-half goals, which are exactly the outcome the lay wants. Adverse selection, not price
  improvement: the market prices 0-0 at 52′ higher for the draw than 0-0 at 45′ for the same reason.
- Trade-off: laying at ~3.0 instead of ~2.75 raises liability per £10 from £17.50 to £20 and the break-even
  non-draw rate from 64.8 % to 67.8 %. All-time v2 strike is 71.4 % (25/35). Volume +75 % in this window.
  Self-critique: n = 7 for the cost estimate; the mechanism is certain, the size is not.

**Q3a — a live data outage (HIGH)**
- football-data.co.uk now answers `www.football-data.co.uk/...` with `302 → football-data.co.uk/...` (verified with
  curl for both `/mmz4281/2627/E0.csv` and `/new/DNK.csv`; the bare domain returns 200). `FOOTBALL_DATA_BASE` is
  the www host and `httpx.AsyncClient(timeout=10.0)` does not follow redirects, so every download fails.
  First 302 in the current log: 12 Sep 16:02. On 2 Sep the www URLs returned 200 (see the 2 Sep entry).
- In the running process the damage is partial: `_prior_cache` still holds last season from the startup fetch, so
  `fetch_league_data` returns last season whole (`matches_this_season=0 prior_weight=1.0` at 22:03). Current-season
  data is gone. On the next restart nothing is cached and **every domestic fixture becomes `no_stats`** — the
  fresh-process replay loaded 0 of 15 leagues. value_betting uses the same service.
- Second defect: `get_league_stats` returns `None` when the refresh fails even though a stale entry is in `_cache`.
  A failed refresh should serve the stale copy and warn.

**Q3b — team-name gaps in covered leagues (HIGH)**
- 75 `no_stats` fixtures in 10 days were in tier 1/2 competitions the bot already covers (Championship 9,
  Eredivisie 10, Serie B 9, Bundesliga 2 9, Primeira 8, Danish Superliga 8, Segunda 7, Premier League 4, …).
- Replayed with redirects followed: **18 resolve** (redirect victims), **57 do not** — 46 distinct Betfair names
  with no entry in `_normalize_team_name`. Nearly all have an obvious football-data twin: Nottm Forest → Nott'm
  Forest, Sheff Utd → Sheffield United, Paris St-G → Paris SG, Mgladbach → M'gladbach, Hamburger SV → Hamburg,
  AC Monza → Monza, Dundee Utd → Dundee United, Inverness CT → Inverness C, Racing Santander → Santander,
  Espanyol → Espanol, Deportivo → La Coruna, Sporting Lisbon → Sp Lisbon, Braga → Sp Braga, Estoril Praia →
  Estoril, Academico de Viseu → Academico Viseu, FC Andorra → Andorra, CD Castellon → Castellon, AD Ceuta FC →
  Ceuta, Celta Vigo B → Celta B, Sporting Gijon → Sp Gijon, plus the Dutch (SC Telstar, PEC Zwolle, NEC Nijmegen,
  Cambuur Leeuwarden, ADO Den Haag, Fortuna Sittard), German (Arminia Bielefeld, SV Darmstadt, Dynamo Dresden,
  FC Heidenheim, VfL Osnabruck, FC Magdeburg), Italian (Entella, LR Vicenza Virtus, US Cremonese, Calcio Avellino
  SSD), Danish (AGF, OB, FC Nordsjaelland, AC Horsens, Randers) and French (ESTAC Troyes, Pau) forms.
  Full list in the replay output. A prefix-stripping fallback (FC/SC/AC/CD/AD/SV/US/VfL/LR/SSD) would catch most.
- Yield estimate: domestic candidates are ~7 % of domestic stats-resolved fixtures (12 of ~166; the other 10
  candidates were European ties that bypass stats). 75 recovered fixtures ≈ 5 more candidates per 10 days
  (+40 %). MEDIUM.

**Q4 — the counterfactual says keep the filters (MEDIUM, n = 32)**
Of `home_goals` rejections that were 0-0 at HT, **14 of 32 (44 %)** finished non-draw, against 7 of 8 for stored
candidates and a 59 % base rate across every evaluated fixture. Break-even at the current entry price is ~65 %.
That filter is removing exactly the games that stay 0-0. `no_stats` fixtures that were 0-0 at HT: 59 of 96
(61 %) non-draw — below break-even even before knowing what odds they would have offered. `no_clear_favourite`,
`draw_odds_range`, `liquidity` are 31 fixtures combined: loosening them cannot move volume. **H1 rejected.**

**Q5 — summer (HIGH)** June has 0 bets because no scanned league plays. Unchanged from the 2 Sep entry: add
SE/NO/FI/IE to the scan and SWE/NOR/FIN/IRL to the stats map (football-data "new" files exist).

### Most supported hypothesis
H2 + H3 together. Volume is lost *before* the strategy gets to judge (name resolution, and now a redirect outage)
and *after* it has judged correctly (the 2.8 cap converts a HT rule into a 52′ rule and drops early-goal winners).
H4 is true in the narrow sense that 0-0 at HT happens in ~24 % of fixtures (155 of 649), but the levers below
roughly double-to-triple entries without touching that.

### Ruled out
- Loosening `home_goals`, `no_clear_favourite`, `draw_odds_range` or `liquidity` (H1).
- Adding tier-3 leagues on the current evidence (61 % non-draw from 0-0 HT in that pool).

### Rough combined effect (MEDIUM)
22 candidates / 10 days → ~27 with name fixes; ~36 % reach HT 0-0 → ~10; cap raised so all enter → ~10 entries
per 10 days against 4 now. That is 2–3 bets a week becoming 6–7, at the same selection rule.

### Progress (12 Sep 2026, 23:37 UK — deployed)
Steps 1–3 built, tested (290 checks across seven scripts, all passing locally and in the
container) and deployed with a full rebuild. Post-deploy: all 15 leagues fetch from the bare
domain with this season's matches present (E1 69, P1 44, E0 30 ...), zero redirect failures,
`scripts/research/ltd_nostats_replay.py` resolves 75 of 75 previously-no_stats fixtures and
the wider 268-name check resolves every covered-league team seen since 2 Sep with no
within-league collisions. `MAX_HT_DRAW_ODDS` is 3.2 and the `ht_entry` / `ht_odds_range`
funnel rows carry `status`. First read-out of the cap change needs ~30 entries; compare
strike and ROI for `status = HalfTime` against the pre-12-Sep record (71.4 % at 2.73 avg).
Not committed to git at the time of writing.

### Next steps (ordered)
1. ~~**Fix the downloader now**~~ **Done 12 Sep.**: `FOOTBALL_DATA_BASE` to the bare domain *and* `follow_redirects=True`; make
   `get_league_stats` serve the stale cache on a failed refresh. Deploy = scp + rebuild. Until then do not
   restart the container without expecting zero domestic candidates.
2. ~~**Name aliases**~~ **Done 12 Sep.** Add the 46 names above to `_normalize_team_name`, plus a generic prefix-strip fallback, with a
   test that every current Betfair name in tier 1/2 resolves. Re-run `ltd_nostats_replay.py` to confirm 0 unresolved.
3. ~~**HT entry**~~ **Done 12 Sep.** Raise `MAX_HT_DRAW_ODDS` to ~3.2 and record the draw price *at the HalfTime status* in the funnel
   (`ht_draw_odds`) so the cost of entering at 45′ vs 52′ is measurable. Review after 30 entries.
4. Summer leagues as per 2 Sep.
5. Funnel semantics: keep the first halftime rejection alongside the final verdict (or a `first_reason` column).

### Open questions
- Does the edge hold at ~3.0 entry? Only the forward sample answers it; all 35 historical entries were ≤ 2.8.
- How much of the 7 % candidate yield survives once current-season stats are back (they have been last-season
  numbers since the redirect began)?

---

## 2026-09-12 — Is there arbitrage the bot can capture, and would it make money?

Trigger: Paul asked whether we could "hit something dynamically" with arbitrage. The `arbitrage`
strategy has been in `ENABLED_STRATEGIES` on the VPS for months. Answered by measurement, not
theory: two read-only snapshots run from inside the VPS container on Sat 12 Sep, 22:45–22:52 UK,
no bets placed. Scripts kept in `scripts/research/arb_snapshot.py` (Betfair intra-exchange) and
`scripts/research/xvenue_snapshot.py` (Betfair vs Matchbook vs Smarkets). Run with
`docker cp <script> betfair-bot:/tmp/ && docker exec -w /app betfair-bot python /tmp/<script>`.

### Sub-questions
1. Does the existing `arbitrage` strategy detect anything, and can it in principle?
2. How far do Betfair's own books sit from an intra-exchange arb (dutch / back-lay cross)?
3. Is there a cross-market arb inside Betfair (Correct Score vs Match Odds, CS vs Over/Under, WIN vs PLACE)?
4. Is there an exchange-vs-exchange arb (Betfair / Matchbook / Smarkets) on the fixtures the bot scans?
5. What commission does Paul actually pay, since it is the largest single parameter?
6. Given the bankroll, what could any of this be worth in pounds?

### Hypotheses
- H1 Intra-exchange arbs exist but the 60 s poll misses them.
- H2 Intra-exchange arbs do not exist at any cadence we can execute at; the books are always over-round.
- H3 Cross-market inconsistencies inside Betfair (CS vs MO) are large enough to exploit.
- H4 Exchange-vs-exchange gaps exist and net positive after commission at meaningful size.
- H5 Only bookmaker-vs-exchange gaps are large enough to matter, and those are a manual, account-limited grind.

### Evidence

**Q1 — the existing strategy is inert (HIGH)**
- Enabled on the VPS (`ENABLED_STRATEGIES=value_betting,lay_the_draw,arbitrage,nags_place`), scans pre-off
  MATCH_ODDS / WIN / PLACE every 60 s. Zero "Arbitrage opportunity detected" lines in the current log window;
  no `arbitrage` rows in `bets` or `strategy_evaluations`. It does not record to the funnel.
- Check 1 (`best_back >= best_lay` on the same runner) cannot occur on a matching engine: crossing offers match
  instantly. Confirmed empirically: 0 crosses across 74 MO, 73 CS, 73 OU, 48 HR markets.
- Check 2 (back book < 100 %) needs a dutch under-round. See Q2.

**Q2 — Betfair books never touched an arb (HIGH, single snapshot)**
Liquid = ≥ £1k matched. Football markets were ~15 h from kick-off; HR markets were tomorrow's cards.

| Market | n liquid | back overround min / median / p90 | lay underround median | under-100 books | crosses |
|---|---|---|---|---|---|
| MATCH_ODDS | 28 | 100.15 / 100.91 / 101.56 | 99.09 | 0 | 0 |
| OVER_UNDER_25 | 7 | 100.46 / 101.12 / 101.69 | 99.15 | 0 | 0 |
| CORRECT_SCORE | 3 | 102.34 / 104.73 / 114.76 | 97.50 | 0 | 0 |
| HR WIN | 13 | 109.5 / 126.9 / 163.5 | 81.3 | 0 | 0 |
| HR PLACE | 0 | (median £2 matched the night before) | | | |

MO back-to-lay spread on liquid markets: p10 0.82 %, median 2.13 %. The spread alone exceeds any dutch
under-round that could plausibly appear. Self-critique: a 60 s poll cannot see sub-second under-rounds; but even
if one existed, three sequential REST placements at ~100–300 ms each would not fill it, and the 5 % commission
on the winning leg still applies. H1 is unfalsified for millisecond bots and irrelevant for this one; H2 holds
for anything this architecture can execute.

**Q3 — no intra-Betfair cross-market arb (HIGH for this snapshot)**
Riskless construction: dutch-back the CS cells of an outcome, lay that outcome on MO; arb iff
Σ(1/back_cs) ≤ 1/lay_mo, i.e. ratio < 1.
- CS vs MO, 39 outcome checks: ratio min 1.024, median 1.071, p90 1.55. Closest: Man Utd v Man City away, 2.4 % short.
- CS vs O2.5, 13 checks: min 1.024, median 1.135.
- WIN vs PLACE dominance (back PLACE at P ≥ lay WIN at W is riskless): 0 pairs evaluated because every PLACE
  market had < £100 matched the night before. **GAP** — re-run `arb_snapshot.py` within 60 min of a race.
The CS market's own overround (102–115 %) is the reason: the thin market absorbs the divergence. H3 rejected pre-off.

**Q4 — exchange-vs-exchange: exists on paper, worth pennies (HIGH, single snapshot)**
Matchbook (`api.matchbook.com/edge/rest/events`, no auth) and Smarkets (`api.smarkets.com/v3`, no auth, prices are
10000/price, `bids` = best lay, `offers` = best back) both serve public prices. 56 of 74 Betfair fixtures matched on
2+ venues by kick-off time and team name (unmatched are Serie C, WSL and Danish sides, not a liquidity concern).
739 back-venue/lay-venue/outcome comparisons, commission 5 % Betfair, 2 % Matchbook, 2 % Smarkets:

| | value |
|---|---|
| gross gap p50 / p90 / p99 / max | −2.56 % / −0.76 % / 0.00 % / +3.64 % |
| gross-positive pairs | 6 of 739 |
| net-positive after commission | **1 of 739** |
| that one | Mantova v Sampdoria H: back Smarkets 2.28, lay Matchbook 2.20, £1.34 per £100 backed, **£2.00 available on the lay** |
| best on a Betfair market ≥ £5k matched | −£0.88 per £100 (Brest v PSG away, Matchbook back = Smarkets lay) |

Matchbook liquid MO back overround: median 100.71 %, min 100.15 % — the same shape as Betfair. The three exchanges
are quoted off each other. Self-critique: this is one Saturday-night snapshot ~15 h pre-off. Gaps are known to widen
in the last minutes before kick-off and in-play (excluded here by design, as CLV notes explain). The size column
is the real killer: the only net-positive gap had £2 behind it. H4 rejected at meaningful size; unfalsified for
near-off / in-play windows (GAP).

**Q5 — commission is 5.0 % (HIGH)**
77 live winners with a recorded commission (25 Jan – 12 Sep 2026): £28.39 commission on £567.97 gross = 5.00 %.
A web result claiming Betfair raised MBR to 6 % in June 2026 does not apply to this account. The `commission` column is
populated on only 77 of 165 live winners (Betfair reports it per market, not per order); P&L is unaffected because live
settlement uses Betfair's net `profit`.

**Q6 — capital arithmetic (HIGH)**
Available Betfair balance at 13:33 UK today: £199.61. A perfect, riskless 1 %-per-day machine on the whole bankroll
is £2/day. Nothing measured above gets near 1 % once, let alone daily. Arbitrage returns scale with capital and
venue access, not with the quality of the detector.

**Bookmaker-vs-exchange (H5) — not measured (GAP)**
This is the only form of arb that routinely shows 2–5 % gaps at retail scale. Measuring it needs an odds feed with
bookmaker prices (e.g. The Odds API, free tier 500 requests/month, enough for a 48 h sampler on one sport). The
known end-state is stake restriction on the bookmaker accounts within weeks to months, and the bookmaker leg cannot
be automated. The bot's role would be detection plus the Betfair lay; Paul places the back by hand. Confidence that
it exists: HIGH (well documented industry). Confidence that it is worth building here: LOW until sampled.

### Most supported hypothesis
H2 + H5. Inside Betfair, and between Betfair and the other exchanges pre-off, there is no arbitrage this bot can
execute at any size worth the code. The `arbitrage` strategy has been running for months and cannot fire.

### Ruled out
- Intra-exchange back/lay cross (impossible by construction) and dutch under-round (never observed, spread-dominated).
- Intra-Betfair cross-market (CS vs MO/OU): closest 2.4 % short, before commission, and CS depth is thin.
- Exchange-vs-exchange pre-off as a P&L source: 1 net-positive in 739 with £2 behind it.

### Open questions
- Near-off (last 10 min) and in-play cross-exchange gaps: not sampled. In-play "arb" is really latency trading.
- WIN vs PLACE dominance near the off: not sampled (PLACE empty the night before).
- Bookmaker-vs-exchange gap size and frequency: needs a key.

### Suggested next actions
1. Remove `arbitrage` from `ENABLED_STRATEGIES` on the VPS (`.env` change, `docker compose down && up -d`). It costs
   one `evaluate()` per market per minute and can never alert. Not done in this session.
2. If Paul wants to see bookmaker-vs-exchange numbers: get a free The Odds API key; extend `xvenue_snapshot.py` into a
   48 h sampler (one request per 90 min keeps inside the free tier) before any execution code.
3. Use the free Matchbook/Smarkets public feeds as a **consensus closing line** for CLV and as an outlier check on
   value_betting entries. Cheap and strictly additive to the 2 Sep CLV fix.
4. The measured edge in this bot is LTD v2 (+21.5 % ROI on 35 bets, sample still growing under the persisted
   funnel). Time spent widening that sample is worth more than any arb path above.

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
