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
- Path: /opt/betfair-bot (NOT a git checkout — files are scp'd; see section 6 for drift detection)
- Note: `sqlite3` is not installed inside the container. To query the DB, copy it out first:
  `docker cp betfair-bot:/app/data/betfair_bot.db /tmp/bf.db && sqlite3 /tmp/bf.db "<query>"`
- `docker logs` only covers the *current* container. Every deploy is `compose down/up`, so
  after a deploy the docker log is minutes old. The full history is on disk at
  `/opt/betfair-bot/data/logs/bot.log` (rotates to `bot.log.1..5`, ~10MB each). Strip ANSI
  with `sed -E 's/\x1b\[[0-9;]*m//g'` before grepping.
- Nags DB (read-only mount): `/root/horse-racing-bot/data/racing.db`. Query on the host with
  `sqlite3 'file:/root/horse-racing-bot/data/racing.db?mode=ro' "<query>"`.
- `bets` columns: `bet_ref` (not `betfair_bet_id`), `status` (SETTLED/…), `result`
  (**WON/LOST/VOID** — not WIN/LOSS). `markets` is keyed by `id`, not `market_id`.

## 1. PROCESS STATUS
- Is `betfair-bot` running and healthy? How long, and when did it start?
- `RestartCount` > 0 means Docker restarted it after a crash. `RestartCount` = 0 with a recent
  start means a deploy — confirm against local `git log` and the image build time.

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "docker ps --format '{{.Names}}\t{{.Status}}\t{{.RunningFor}}' | grep betfair && docker inspect -f 'started={{.State.StartedAt}} restarts={{.RestartCount}}' betfair-bot && docker image inspect \$(docker inspect betfair-bot --format '{{.Image}}') --format 'image_built={{.Created}}'"
```

## 2. LOG ANALYSIS
- Check the last 100 lines of the docker log, then the **whole day** from the on-disk log
  (the docker log is empty after a deploy).
- Identify recurring error patterns. `Stream thread: start() returned normally` at the moment
  of a `compose down` is benign shutdown noise.

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "docker compose -f /opt/betfair-bot/docker-compose.yml logs --tail 100 betfair-bot 2>&1"
# Whole day, deduplicated, from disk
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "grep -h \"\$(date -u +%F)\" /opt/betfair-bot/data/logs/bot.log | sed -E 's/\x1b\[[0-9;]*m//g' | grep -iE 'warning|error|critical' | grep -v 'HR parse' | cut -c1-300 | awk '{\$1=\"\"; print}' | sort | uniq -c | sort -rn | head -40"
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
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "sqlite3 'file:/root/horse-racing-bot/data/racing.db?mode=ro' \"SELECT date(created_at), COUNT(*) FROM selections WHERE created_at > datetime('now','-8 days') AND superseded_at IS NULL GROUP BY 1;\"; docker logs horse-racing-bot 2>&1 | grep -m1 -iE 'Auto-schedule'"
```

## 4. PERFORMANCE METRICS
- Recent P&L from the database (last 14 days), then strike rate vs break-even per strategy.
  Break-even strike % = 100 / average matched odds (for BACK bets). A strategy whose strike
  rate sits below its break-even is losing regardless of what the headline P&L says this week.

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "sqlite3 /tmp/bf.db 'SELECT strategy, status, result, COUNT(*), ROUND(SUM(profit_loss),2) FROM bets WHERE placed_at > datetime(\"now\",\"-14 days\") GROUP BY strategy, status, result;'"
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "sqlite3 -header /tmp/bf.db \"SELECT strategy, COUNT(*) n, ROUND(AVG(matched_odds),2) avg_odds, ROUND(100.0*SUM(result='WON')/COUNT(*),1) strike_pct, ROUND(100.0/AVG(matched_odds),1) breakeven_pct, ROUND(SUM(profit_loss),2) pnl, ROUND(100*SUM(profit_loss)/SUM(stake),1) roi_pct FROM bets WHERE status='SETTLED' AND result IN ('WON','LOST') AND placed_at > datetime('now','-30 days') GROUP BY strategy;\""
```

## 5. SYSTEM RESOURCES
- RAM usage, disk space, CPU usage

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "free -h && echo '---' && df -h / && echo '---' && top -bn1 | head -12"
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
# (a) Count recent "not logged in" warnings. Should be 0 since 2026-04-20 fix
# (re-login now auto-recovers within 15 mins). >3 in the last hour = investigate.
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "docker logs betfair-bot --since 1h 2>&1 | grep -c 'not logged in to Betfair'"

# (b) Confirm a recent successful login or keep-alive
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "docker logs betfair-bot --since 2h 2>&1 | grep -iE 'Successfully logged into Betfair|Session keep-alive successful|attempting re-login' | tail -10"

# (c) Days since last bet placed — flag if bot is up but no bets for >48h AND Nags had picks (section 3)
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "sqlite3 /tmp/bf.db \"SELECT MAX(placed_at), CAST((julianday('now') - julianday(MAX(placed_at))) AS INTEGER) AS days_ago FROM bets;\""

# (d) Cert expiry — Betfair cert auth dies silently when this lapses. Flag inside 30 days.
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "openssl x509 -in /opt/betfair-bot/certs/client-2048.crt -noout -enddate"

# (e) Live bankroll sync. Logs 'Synced bankroll with Betfair' at INFO only when the balance moved;
# a 'Failed to sync balance' warning means the engine is running on a computed bankroll.
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "sed -E 's/\x1b\[[0-9;]*m//g' /opt/betfair-bot/data/logs/bot.log | grep -E 'Synced bankroll|Failed to sync balance' | tail -3"
```

**Interpretation:**
- 🔴 `not logged in` warnings recurring AND no `attempting re-login` messages = auto-recovery broken, container restart needed
- 🟡 `attempting re-login` messages present = session dropped but recovered (working as designed)
- 🔴 Container uptime >> days since last bet, with Nags picks present = trading effectively stopped

## 8. DATA INTEGRITY
- Unsettled bets older than 24h are stranded (see CLAUDE.md, "A missing result is not a non-runner").
- A run of VOIDs on the Nags paper legs is the results-cache bug signature — compare against the
  live `nags_place` leg on the same horse, which settles from Betfair.
- A long losing run on a Nags strategy should be **cross-checked against Nags's own results table**
  before it is blamed on settlement. If Nags agrees the horses lost, it is selection, not the bot.

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "sqlite3 -header /tmp/bf.db \"SELECT id, strategy, selection_name, status, placed_at FROM bets WHERE status!='SETTLED' AND placed_at < datetime('now','-1 day');\"; sqlite3 /tmp/bf.db \"SELECT strategy, COUNT(*) FROM bets WHERE result='VOID' AND placed_at > datetime('now','-14 days') GROUP BY 1;\""
# LTD funnel is being written and scored (table exists from the 2 Sep 2026 build)
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "sqlite3 /tmp/bf.db \"SELECT stage, outcome, COUNT(*) n, SUM(ft_home IS NOT NULL) scored FROM strategy_evaluations WHERE start_time > datetime('now','-7 days') GROUP BY 1,2;\""
# Nags's own view of its recent picks (nags_place takes the first of nap > next_best > selection > race_nb per race)
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "sqlite3 -header 'file:/root/horse-racing-bot/data/racing.db?mode=ro' \"SELECT date(s.created_at) d, s.horse, s.selection_type, r.result, r.finish_position FROM selections s LEFT JOIN results r ON r.selection_id=s.id WHERE s.created_at > datetime('now','-14 days') AND s.superseded_at IS NULL ORDER BY s.created_at;\""
```

## 9. STRATEGY EDGE ASSESSMENT
- Strike rate vs break-even from section 4; 30-day and all-time.
- Which strategies are live vs paper: read `FORCE_PAPER_STRATEGIES` in
  `src/strategies/horse_racing.py`, not the CLAUDE.md table.
- Is the strategy performing as expected? Any parameter tweaks recommended? Strategy changes
  follow RESEARCH.md — the health check reports, it does not retune.

## 10. SECURITY POSTURE (quick)
Baseline from the 1 Sep 2026 review of `../IG/security_report.md` against this bot. Re-check
that none of these have regressed; anything new gets its own line in recommendations.
- Cert/key file mode on the VPS (`ls -l /opt/betfair-bot/certs/`) — target 600, owner uid 1000.
- `.env` and `certs/` are excluded by `.dockerignore` (otherwise `COPY . .` bakes them into the image).
- `docker-compose.yml` still uses `network_mode: host` (shares localhost with ib-gateway etc.).
- `config/logging_config.py` on the VPS still carries `RedactSecretsFilter` (Telegram token redaction).

## 11. RUNNING THE TESTS
Every file in `tests/` is currently a script that `raise SystemExit`s at import (pytest collects nothing
and aborts). Run pytest with those ignored (so any future real modules still run), then run the scripts with
`PYTHONPATH=.`. On the VPS use a throwaway container from the built image (no volumes, no network):

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "docker run --rm --network none --entrypoint sh betfair-bot-betfair-bot:latest -c 'S=\$(grep -l \"raise SystemExit\" tests/*.py); python -m pytest -q -p no:cacheprovider \$(echo \"\$S\" | sed \"s/^/--ignore=/\") tests/ | tail -3; for f in \$S; do PYTHONPATH=/app python \$f | tail -1; done'"
```

## 12. RECOMMENDATIONS
Provide prioritised recommendations:
- P1 (Critical): Issues that need immediate attention
- P2 (Important): Should be addressed soon
- P3 (Nice to have): Optimisations for later

## 13. SUMMARY DASHBOARD
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
| Data Integrity | ?/? | stranded bets, VOID run |
| Strategy Edge | ?/?/? | strike vs break-even |

Traffic light summary: 🟢 All good / 🟡 Minor issues / 🔴 Needs attention
