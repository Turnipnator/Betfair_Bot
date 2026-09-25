---
name: healthcheck
description: Run a comprehensive health check on the Betfair trading bot
---

# Betfair Trading Bot Health Check

Run a comprehensive health check on the betfair-bot. Work through each section systematically and provide a summary dashboard at the end.

## VPS Details
- Server: 149.102.144.190
- SSH Key: ~/.ssh/id_ed25519_vps
- Container: betfair-bot (SQLite at /app/data/betfair_bot.db — there is NO separate betfair-db container)
- Container: acca-advisor (since 24 Sep 2026; advisory football accas, **places no bets**).
  Own ledger `/opt/betfair-bot/data/acca.db` (WAL, safe to query on the host directly with
  `sqlite3`), own log `data/logs/acca.log`, own Telegram bot. Checked in section 9.
- Path: /opt/betfair-bot (NOT a git checkout — files are scp'd; see section 6 for drift detection)
- Note: `sqlite3` is not installed inside the container. To query the DB, copy it out first:
  `docker cp betfair-bot:/app/data/betfair_bot.db /tmp/bf.db && sqlite3 /tmp/bf.db "<query>"`
- `docker logs` only covers the *current* container and is capped at 3×10MB by compose
  (about six hours). Every deploy is `compose down/up`, so after a deploy the docker log is
  minutes old. The full history is on disk at `/opt/betfair-bot/data/logs/bot.log` (rotates
  to `bot.log.1..10`, ~10MB each, roughly a week at INFO since the 13 Sep 2026 volume cut).
  `bot.log` itself can be minutes old, so grep **all** rotated files, oldest first:
  `cat $(ls -r /opt/betfair-bot/data/logs/bot.log.* ) /opt/betfair-bot/data/logs/bot.log`.
  Strip ANSI with `sed -E 's/\x1b\[[0-9;]*m//g'` before grepping.
- Nags DB (read-only mount): `/root/horse-racing-bot/data/racing.db`. Query on the host with
  `sqlite3 'file:/root/horse-racing-bot/data/racing.db?mode=ro' "<query>"`.
- `bets` columns: `bet_ref` (not `betfair_bet_id`), `status` (SETTLED/…), `result`
  (**WON/LOST/VOID** — not WIN/LOSS). `markets` is keyed by `id`, not `market_id`.

## 1. PROCESS STATUS
- Are `betfair-bot` and `acca-advisor` running and healthy? How long, and when did each start?
  They deploy independently (`docker compose up -d --build acca-advisor` leaves the live bot
  alone), so different start times are normal.
- `RestartCount` > 0 means Docker restarted it after a crash. `RestartCount` = 0 with a recent
  start means a deploy — confirm against local `git log` and the image build time.

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "docker ps --format '{{.Names}}\t{{.Status}}\t{{.RunningFor}}' | grep -E 'betfair|acca' && docker inspect -f '{{.Name}} started={{.State.StartedAt}} restarts={{.RestartCount}}' betfair-bot acca-advisor && docker image inspect \$(docker inspect betfair-bot --format '{{.Image}}') --format 'image_built={{.Created}}'"
```

## 2. LOG ANALYSIS
- Check the last 100 lines of the docker log, then the **whole day** from the on-disk log
  (the docker log is empty after a deploy).
- Identify recurring error patterns. `Stream thread: start() returned normally` at the moment
  of a `compose down` is benign shutdown noise.

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "docker compose -f /opt/betfair-bot/docker-compose.yml logs --tail 100 betfair-bot 2>&1"
# Whole day, deduplicated, from disk (all rotated files — bot.log alone may be minutes deep)
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "cd /opt/betfair-bot/data/logs && cat \$(ls -r bot.log.* 2>/dev/null) bot.log | grep -h \"\$(date -u +%F)\" | sed -E 's/\x1b\[[0-9;]*m//g' | grep -E '\[(warning|error|critical)' | grep -v 'HR parse' | cut -c1-300 | awk '{\$1=\"\"; print}' | sort | uniq -c | sort -rn | head -40"
# How far back the on-disk history reaches (should be days, not hours)
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "cd /opt/betfair-bot/data/logs && sed -E 's/\x1b\[[0-9;]*m//g' \$(ls -r bot.log.* | head -1) | grep -m1 -oE '^20[0-9-]+T[0-9:]+'"
```

## 3. SIGNAL GENERATION
- Is the bot actively monitoring markets? (scan_markets job should run every minute)
- What was the last bet placed and when?
- **A quiet day is not automatically a fault.** The Nags strategies only bet when the Nags bot
  has written picks, and Nags runs in cherry-pick mode (`Auto-schedule DISABLED`, picks only
  when Paul sends `/run`). Always check the Nags picks count for today before reading "no bets"
  as a Betfair-bot problem. Football volume is low by design.

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "docker logs betfair-bot --since 1h 2>&1 | grep -c 'scan_markets.*executed successfully'"
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "docker cp betfair-bot:/app/data/betfair_bot.db /tmp/bf.db && sqlite3 /tmp/bf.db 'SELECT id, strategy, selection_name, matched_odds, stake, result, profit_loss, placed_at FROM bets ORDER BY placed_at DESC LIMIT 10;'"
# Bets per day, last 10 days
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "sqlite3 /tmp/bf.db \"SELECT date(placed_at), COUNT(*), GROUP_CONCAT(DISTINCT strategy) FROM bets WHERE placed_at > datetime('now','-10 days') GROUP BY 1 ORDER BY 1;\""
# Nags picks per day (explains Nags-strategy quiet days) and whether Nags is on manual /run
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "sqlite3 'file:/root/horse-racing-bot/data/racing.db?mode=ro' \"SELECT date(created_at), source, COUNT(*) FROM selections WHERE created_at > datetime('now','-8 days') AND superseded_at IS NULL GROUP BY 1,2;\"; docker logs horse-racing-bot 2>&1 | grep -m1 -iE 'Auto-schedule'"
```

## 4. PERFORMANCE METRICS
- Recent P&L from the database (last 14 days), then strike rate vs break-even per strategy.
  Break-even strike % depends on the side: **BACK = 100 / avg odds; LAY = 100 × (1 − 1 / avg odds)**.
  A lay at 2.70 needs to win 63% of the time, not 37% — the 13 Sep 2026 check nearly credited
  LTD with a 25-point edge it does not have. The query below picks the formula from `bet_type`.
  A strategy whose strike rate sits below its break-even is losing regardless of what the
  headline P&L says this week.

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "sqlite3 /tmp/bf.db 'SELECT strategy, status, result, COUNT(*), ROUND(SUM(profit_loss),2) FROM bets WHERE placed_at > datetime(\"now\",\"-14 days\") GROUP BY strategy, status, result;'"
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "sqlite3 -header /tmp/bf.db \"SELECT strategy, bet_type, CASE WHEN bet_ref LIKE 'PAPER-%' THEN 'paper' ELSE 'live' END mode, COUNT(*) n, ROUND(AVG(matched_odds),2) avg_odds, ROUND(100.0*SUM(result='WON')/COUNT(*),1) strike_pct, ROUND(CASE WHEN bet_type='LAY' THEN 100.0*(1-1.0/AVG(matched_odds)) ELSE 100.0/AVG(matched_odds) END,1) breakeven_pct, ROUND(SUM(profit_loss),2) pnl, ROUND(100*SUM(profit_loss)/SUM(stake),1) roi_pct FROM bets WHERE status='SETTLED' AND result IN ('WON','LOST') AND placed_at > datetime('now','-30 days') GROUP BY 1,2,3;\""
# Live money only, since the 27 Feb 2026 bankroll reset (compare with the Betfair balance in section 7e)
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "sqlite3 -header /tmp/bf.db \"SELECT strategy, COUNT(*) n, ROUND(SUM(COALESCE(profit_loss,0)),2) live_pnl FROM bets WHERE bet_ref NOT LIKE 'PAPER-%' AND placed_at > '2026-02-27' GROUP BY 1;\""
```

## 5. SYSTEM RESOURCES
- RAM usage, disk space, CPU usage

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "free -h && echo '---' && df -h / && echo '---' && top -bn1 | head -12 && echo '---' && docker stats --no-stream --format '{{.Name}} {{.MemUsage}} {{.CPUPerc}}' betfair-bot acca-advisor && ls -lh /opt/betfair-bot/data/acca.db*"
```

## 6. CONFIGURATION REVIEW
- Check key environment variables. **Redact secrets before they land in the transcript.**
- Check the deployed code matches the local checkout. `/opt/betfair-bot` is not a git repo,
  so hash-diff the Python files. On 1 Sep 2026 this found a 5-week-old VPS-only hotfix
  (`RedactSecretsFilter` in `config/logging_config.py`) that had never been committed —
  any file that differs is either an undeployed local change or an uncommitted VPS change,
  and both are bad.

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "grep -E '^(ENABLED_STRATEGIES|TRADING_MODE|LOG_LEVEL|.*STAKE.*|.*EXPOSURE.*|STREAMING.*|MARKET_SCAN_INTERVAL|MIN_TIME_TO_START)=' /opt/betfair-bot/.env | sed -E 's/(KEY|TOKEN|PASSWORD|SECRET)=.*/\1=<redacted>/I'"
# Deployed vs local drift (run from the repo root)
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "cd /opt/betfair-bot && find src config scripts -name '*.py' -exec md5sum {} +" | awk '{print $2, $1}' | sort > /tmp/vps_md5.txt; find src config scripts -name '*.py' -exec md5 -r {} + | awk '{print $2, $1}' | sort > /tmp/local_md5.txt; diff /tmp/local_md5.txt /tmp/vps_md5.txt | grep '^[<>]' | awk '{print $2}' | sort -u
```

## 7. BETFAIR SESSION HEALTH (CRITICAL)

The bot can appear "healthy" in `docker ps` while its Betfair session has silently died.
The container HEALTHCHECK (`scripts/healthcheck.py`) only checks log freshness and that the DB
file exists — it says nothing about the session. Symptoms: container up, scheduler running,
no exceptions — but no bets being placed.

**Run these checks every time:**

```bash
# (a) Session failures in the last 24h, both bots, from the on-disk logs (survive a deploy).
# Expect none. Case-insensitive on purpose: the client logs "Not logged in to Betfair" with a
# capital N, and until 24 Sep 2026 this check grepped lower-case and could never see it.
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "cd /opt/betfair-bot/data/logs && cat \$(ls -r bot.log.* acca.log.* 2>/dev/null) bot.log acca.log | sed -E 's/\x1b\[[0-9;]*m//g' | grep -E \"^(\$(date -u +%F)|\$(date -u -d yesterday +%F))\" | grep -ioE 'not logged in to Betfair|Keep-alive failed|attempting re-login|Betfair login failed' | sort | uniq -c | grep . || echo 'no session failures in 24h'"

# (b) Proof the session works NOW: age of the last successful Betfair call, per bot.
# Keep-alives log at DEBUG, so they can't be the evidence. The live bot logs "Fetched markets"
# after every successful catalogue call, every minute, even when the count is 0 (unlike
# "Fetched market prices", which stops overnight when nothing is in the window). The advisor
# logs "Acca scan" every 5 min. VPS local time is CEST, hence the explicit Z.
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "cd /opt/betfair-bot/data/logs && for f in bot acca; do [ \$f = bot ] && pat='Fetched markets ' || pat='Acca scan'; last=\$(sed -E 's/\x1b\[[0-9;]*m//g' \$f.log | grep \"\$pat\" | tail -1 | grep -oE '^[0-9T:-]+'); login=\$(cat \$(ls -r \$f.log.* 2>/dev/null) \$f.log | sed -E 's/\x1b\[[0-9;]*m//g' | grep 'Successfully logged into Betfair' | tail -1 | grep -oE '^[0-9T:-]+'); echo \"\$f: last successful call \$last UTC (\$(( \$(date +%s) - \$(date -d \"\${last}Z\" +%s) ))s ago), last login \${login:-not in logs}\"; done"

# (c) Days since last bet placed — flag if bot is up but no bets for >48h AND Nags had picks (section 3)
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "sqlite3 /tmp/bf.db \"SELECT MAX(placed_at), CAST((julianday('now') - julianday(MAX(placed_at))) AS INTEGER) AS days_ago FROM bets;\""

# (d) Cert expiry — Betfair cert auth dies silently when this lapses. Flag inside 30 days.
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "openssl x509 -in /opt/betfair-bot/certs/client-2048.crt -noout -enddate"

# (e) Live bankroll sync. Logs 'Synced bankroll with Betfair' at INFO only when the balance moved;
# a 'Failed to sync balance' warning means the engine is running on a computed bankroll.
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "sed -E 's/\x1b\[[0-9;]*m//g' /opt/betfair-bot/data/logs/bot.log | grep -E 'Synced bankroll|Failed to sync balance' | tail -3"
```

If the Betfair balance and £259.84 + the section 4 live P&L disagree by more than pennies, run
`scripts/research/bankroll_gap.py` in the container (dry run; see CLAUDE.local.md #21). It
lists every live bet whose DB figures differ from Betfair's cleared order net of 2% commission.

**Interpretation:**
- 🔴 (b) last successful call older than 5 min (live bot) or 15 min (advisor) = session or scheduler dead
- 🔴 `Not logged in` in (a) recurring AND no `attempting re-login` = auto-recovery broken, container restart needed
- 🟡 `attempting re-login` / `Keep-alive failed` in (a) but (b) is fresh = dropped and recovered (working as designed)
- 🔴 Container uptime >> days since last bet, with Nags picks present = trading effectively stopped

## 8. DATA INTEGRITY
- Unsettled bets older than 24h are stranded (see CLAUDE.md, "A missing result is not a non-runner").
- A run of VOIDs on the Nags paper legs is the results-cache bug signature — compare against the
  live `nags_place` leg on the same horse, which settles from Betfair.
- A long losing run on a Nags strategy should be **cross-checked against Nags's own results table**
  before it is blamed on settlement. If Nags agrees the horses lost, it is selection, not the bot.

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "sqlite3 -header /tmp/bf.db \"SELECT id, strategy, selection_name, status, placed_at FROM bets WHERE status!='SETTLED' AND placed_at < datetime('now','-1 day');\"; sqlite3 /tmp/bf.db \"SELECT strategy, COUNT(*) FROM bets WHERE result='VOID' AND placed_at > datetime('now','-14 days') GROUP BY 1;\""
# Any catalogue call that came back full (14 Sep 2026 build) — a fetch is silently truncated
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "cd /opt/betfair-bot/data/logs && cat \$(ls -r bot.log.* 2>/dev/null) bot.log | sed -E 's/\x1b\[[0-9;]*m//g' | grep -c 'Market catalogue hit the result cap'"
# LTD half-time: entries vs drops, and how many entries followed a phantom first-half reading
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "sqlite3 -header /tmp/bf.db \"SELECT outcome, reason, COUNT(*) n, SUM(json_extract(detail,'$.pre_ht_goal_reading') IS NOT NULL) after_phantom, SUM(ht_home=0 AND ht_away=0) ht00 FROM strategy_evaluations WHERE strategy='lay_the_draw' AND stage='halftime' AND start_time > datetime('now','-7 days') GROUP BY 1,2;\""
# Value betting funnel (14 Sep 2026 build): the binding filter per fixture. `no_stats` dominates by design:
# `no_stats` is checked before `league_tier`, and only tier 1/2 leagues are loaded (LEAGUE_FILES), so League 1/2,
# National League, cups and youth leagues all land there. Only a covered league in `no_stats` is a fault — use the
# alias query in CLAUDE.local.md #15 (match the competition names exactly; LIKE '%Ligue%' catches Ligue 3 and women's leagues).
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "sqlite3 -header /tmp/bf.db \"SELECT reason, COUNT(*) n, SUM(ft_home IS NOT NULL) scored FROM strategy_evaluations WHERE strategy='value_betting' AND start_time > datetime('now','-7 days') GROUP BY 1 ORDER BY n DESC;\""
# LTD funnel is being written and scored (table exists from the 2 Sep 2026 build)
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "sqlite3 /tmp/bf.db \"SELECT stage, outcome, COUNT(*) n, SUM(ft_home IS NOT NULL) scored FROM strategy_evaluations WHERE strategy='lay_the_draw' AND start_time > datetime('now','-7 days') GROUP BY 1,2;\""
# nags_place funnel (13 Sep 2026 build): one verdict per race Nags had a pick in. A Nags primary
# pick with NO row here was never matched to a PLACE market (or the daily cap was hit) — that is
# the case to chase. `not_ew_eligible` with the numbers is the rule working as designed.
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "sqlite3 -header /tmp/bf.db \"SELECT date(start_time) d, time(start_time) t, event_name, outcome, reason, detail FROM strategy_evaluations WHERE strategy='nags_place' AND start_time > datetime('now','-7 days') ORDER BY start_time;\""
# Nags's own view of its recent picks (nags_place takes the first of nap > next_best > selection > race_nb per race).
# Only source='bot' rows reach the Betfair bot. Paul also logs his own card as source='manual' most days, so
# without the filter every horse shows twice (the 25 Sep 2026 check misread that as a supersede bug).
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "sqlite3 -header 'file:/root/horse-racing-bot/data/racing.db?mode=ro' \"SELECT date(s.created_at) d, s.horse, s.selection_type, r.result, r.finish_position FROM selections s LEFT JOIN results r ON r.selection_id=s.id WHERE s.created_at > datetime('now','-14 days') AND s.superseded_at IS NULL AND s.source='bot' ORDER BY s.created_at;\""
```

## 9. ACCA ADVISOR (acca-advisor container)

Advisory only: it prices football from the Exchange and sends accas to its own Telegram bot
for Paul to place by hand. See CLAUDE.md, "Acca advisor". It holds its own Betfair session,
so check it separately from section 7. **Read the mode first**: with `ACCA_ALERTS_ENABLED`
unset or false it is a dry run (accas logged as `dry_run`, nothing sent but the 21:30
summary), and with no `ACCA_TELEGRAM_BOT_TOKEN` it sends nothing at all. Neither is a fault.

```bash
# (a) Mode and limits (token redacted)
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "grep -E '^ACCA_' /opt/betfair-bot/.env | sed -E 's/(TOKEN)=.+/\1=<set>/' | grep . || echo 'no ACCA_ settings: dry run, no Telegram'"

# (b) Scanning: one row per scan, every 5 min (expect ~12/hour). Latest scan should be < 10 min old.
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "sqlite3 -header /opt/betfair-bot/data/acca.db \"SELECT COUNT(*) scans_last_hour, MAX(at) latest FROM acca_scans WHERE at > datetime('now','-1 hour'); SELECT at, markets, trusted, qualified, pool, rejects FROM acca_scans ORDER BY id DESC LIMIT 5;\""

# (c) Daily funnel: markets seen, trusted, legs qualifying, accas proposed
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "sqlite3 -header /opt/betfair-bot/data/acca.db \"SELECT date(at) d, COUNT(*) scans, MAX(markets) max_mkts, ROUND(AVG(trusted)) avg_trusted, MAX(qualified) max_qual, MAX(pool) max_pool FROM acca_scans WHERE at > datetime('now','-7 days') GROUP BY 1; SELECT date(created_at) d, status, COUNT(*) n, ROUND(AVG(n_legs),1) legs, ROUND(AVG(combined_fair_odds),2) fair, ROUND(SUM(suggested_stake),2) suggested FROM acca_accas WHERE created_at > datetime('now','-7 days') GROUP BY 1,2;\""

# (d) Errors and warnings today, and its own Betfair session
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "cd /opt/betfair-bot/data/logs && cat \$(ls -r acca.log.* 2>/dev/null) acca.log | grep -h \"\$(date -u +%F)\" | sed -E 's/\x1b\[[0-9;]*m//g' | grep -E '\[(warning|error|critical)|Acca job failed' | cut -c1-250 | awk '{\$1=\"\"; print}' | sort | uniq -c | sort -rn | head -20; sed -E 's/\x1b\[[0-9;]*m//g' acca.log | grep -iE 'logged into Betfair|re-login|not logged in' | tail -3"

# (e) Stuck legs: acca selections unsettled 72h+ after kick-off (Telegram should have asked for /acca_result)
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "sqlite3 -header /opt/betfair-bot/data/acca.db \"SELECT key, event_name, label, kickoff_original, kickoff, stuck_alerted_at FROM acca_selections WHERE result IS NULL AND key IN (SELECT selection_key FROM acca_legs) AND kickoff_original < datetime('now','-3 days');\""

# (f) Health metric: CLV per leg (alert = fair at alert vs close; taken = your price vs close), and placed P&L
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "sqlite3 -header /opt/betfair-bot/data/acca.db \"SELECT COUNT(clv_alert) legs, ROUND(AVG(clv_alert),2) avg_alert_clv, ROUND(100.0*SUM(clv_alert>0)/COUNT(clv_alert),0) pct_pos, COUNT(clv_taken) taken_legs, ROUND(AVG(clv_taken),2) avg_taken_clv FROM acca_legs WHERE clv_alert IS NOT NULL; SELECT status, COUNT(*) n, SUM(result='WON') won, ROUND(SUM(taken_stake),2) staked, ROUND(SUM(pnl),2) pnl FROM acca_accas WHERE result IS NOT NULL GROUP BY 1;\""
```

**Interpretation:**
- 🔴 No scan row in 15+ min, or `markets=0` on every scan (session dead: look for `not logged in` in (d))
- 🔴 Stuck legs in (e) with no `stuck_alerted_at` = the settlement job is not running
- `ACCA_TELEGRAM_BOT_TOKEN not set` once per start is expected until the bot is created.
- 🟡 `Acca catalogue page full above the volume floor` in (d) = a market type needs splitting
- 🟡 Low `trusted` is **not** a fault on international breaks or midweek: on 24 Sep 2026 (Nations
  League week) only 17 match-odds markets worldwide had £5k matched inside 48h. Compare weekends
  with weekends.
- `qualified` stays 0 for up to 6h after a restart while price history rebuilds (warm start
  reloads persisted quotes, so usually much less).
- CLV (f) is the verdict, not P&L. Alert CLV persistently ≤ 0 over 30+ legs means the shortening
  signal is noise. Tuning goes through RESEARCH.md; the health check reports, it does not retune.

## 10. STRATEGY EDGE ASSESSMENT
- Strike rate vs break-even from section 4; 30-day and all-time.
- Which strategies are live vs paper: read `FORCE_PAPER_STRATEGIES` in
  `src/strategies/horse_racing.py`, not the CLAUDE.md table.
- Is the strategy performing as expected? Any parameter tweaks recommended? Strategy changes
  follow RESEARCH.md — the health check reports, it does not retune.

## 11. SECURITY POSTURE (quick)
Baseline from the 1 Sep 2026 review of `../IG/security_report.md` against this bot, corrected
on 13 Sep 2026 when the check found the first two items had never actually been true. Re-check
that none of these have regressed; anything new gets its own line in recommendations.
- Cert/key/.env file mode on the VPS — **600, owner uid 1000** for `certs/*` (the container runs
  as uid 1000 and mounts them read-only), 600 root for `.env`. Fixed 13 Sep 2026. **Any `scp` of
  the certs from the Mac resets them to 644 uid 501**, so re-run the chown/chmod after one.
- `.env` and `certs/` are excluded by `.dockerignore` (added 13 Sep 2026 — before that every
  image, including the `rollback-*` tag, carried the live credentials and private key). Runtime
  gets both from `env_file` and the certs volume, so nothing needs them in the image.
- `docker-compose.yml` still uses `network_mode: host` (shares localhost with ib-gateway etc.).
- `config/logging_config.py` on the VPS still carries `RedactSecretsFilter` (Telegram token redaction).

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "ls -ln /opt/betfair-bot/certs/ /opt/betfair-bot/.env; grep -nE '^(\.env|certs/)$' /opt/betfair-bot/.dockerignore; grep -n network_mode /opt/betfair-bot/docker-compose.yml; grep -c RedactSecretsFilter /opt/betfair-bot/config/logging_config.py"
# Neither built image may contain either (expect four 'No such file' lines)
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "for img in betfair-bot-betfair-bot betfair-bot-acca-advisor; do docker run --rm --network none --entrypoint sh \$img:latest -c 'ls /app/.env /app/certs/client-2048.key' 2>&1; done"
```

## 12. RUNNING THE TESTS
Every file in `tests/` is currently a script that `raise SystemExit`s at import (pytest collects nothing
and aborts). Run pytest with those ignored (so any future real modules still run), then run the scripts with
`PYTHONPATH=.`. On the VPS use a throwaway container from the built image (no volumes, no network).
`tests/test_acca_advisor.py` needs `src/acca`, which is only in the `betfair-bot-acca-advisor`
image until the live bot is next rebuilt, so run it from that image:

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "docker run --rm --network none --entrypoint sh betfair-bot-betfair-bot:latest -c 'S=\$(grep -l \"raise SystemExit\" tests/*.py); python -m pytest -q -p no:cacheprovider \$(echo \"\$S\" | sed \"s/^/--ignore=/\") tests/ | tail -3; for f in \$S; do PYTHONPATH=/app python \$f | tail -1; done'"
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "docker run --rm --network none --entrypoint sh betfair-bot-acca-advisor:latest -c 'PYTHONPATH=/app LOG_LEVEL=WARNING python tests/test_acca_advisor.py | tail -1'"
```

## 13. RECOMMENDATIONS
Provide prioritised recommendations:
- P1 (Critical): Issues that need immediate attention
- P2 (Important): Should be addressed soon
- P3 (Nice to have): Optimisations for later

## 14. SUMMARY DASHBOARD
Present a quick status summary table:

| Check | Status | Notes |
|-------|--------|-------|
| Bot Running | ?/? | uptime, restarts, deploy correlated with git |
| Database OK | ?/? | |
| Logs Healthy | ?/?/? | docker + on-disk day log |
| Markets Active | ?/? | scans/hour |
| Nags Picks Today | ?/? | count; cherry-pick mode noted |
| Resources OK | ?/?/? | |
| Session Valid | ?/? | login/keep-alive, cert expiry |
| Deployed = Repo | ?/? | md5 drift list |
| Data Integrity | ?/? | stranded bets, VOID run, nags_place funnel rows |
| Acca Advisor | ?/?/? | mode, scans/hour, trusted markets, stuck legs, leg CLV |
| Strategy Edge | ?/?/? | strike vs break-even (side-aware) |
| Security | ?/? | file modes, image free of secrets, redaction filter |
| Tests | ?/? | script assertions passed in a throwaway container |

Traffic light summary: 🟢 All good / 🟡 Minor issues / 🔴 Needs attention
