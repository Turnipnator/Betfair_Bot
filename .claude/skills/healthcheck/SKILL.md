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
- Path: /opt/betfair-bot
- Note: `sqlite3` is not installed inside the container. To query the DB, copy it out first:
  `docker cp betfair-bot:/app/data/betfair_bot.db /tmp/bf.db && sqlite3 /tmp/bf.db "<query>"`

## 1. PROCESS STATUS
- Is `betfair-bot` running and healthy?
- How long has it been running (uptime) — and when did it start?
- Any recent restarts or crashes?

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "docker ps --format '{{.Names}}\t{{.Status}}\t{{.RunningFor}}' | grep betfair && docker inspect -f '{{.State.StartedAt}}' betfair-bot"
```

## 2. LOG ANALYSIS
- Check the last 100 lines of logs for errors, warnings, or anomalies
- Identify any recurring error patterns
- **Specifically check for session health** — see Section 7 below

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "docker compose -f /opt/betfair-bot/docker-compose.yml logs --tail 100 betfair-bot 2>&1"
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "docker logs betfair-bot 2>&1 | grep -iE 'error|warn|fail|session|expired' | tail -30"
```

## 3. SIGNAL GENERATION
- Is the bot actively monitoring markets? (scan_markets job should run every minute)
- What was the last bet placed and when? (compare to container uptime — gap of several days with bot still "up" is a red flag)

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "docker logs betfair-bot --since 1h 2>&1 | grep -c 'scan_markets.*executed successfully'"
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "docker cp betfair-bot:/app/data/betfair_bot.db /tmp/bf.db && sqlite3 /tmp/bf.db 'SELECT id, strategy, selection_name, matched_odds, stake, result, profit_loss, placed_at FROM bets ORDER BY placed_at DESC LIMIT 10;'"
```

## 4. PERFORMANCE METRICS
- Review recent betting P&L from database (last 14 days)

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "docker cp betfair-bot:/app/data/betfair_bot.db /tmp/bf.db && sqlite3 /tmp/bf.db 'SELECT strategy, status, result, COUNT(*), ROUND(SUM(profit_loss),2) FROM bets WHERE placed_at > datetime(\"now\",\"-14 days\") GROUP BY strategy, status, result;'"
```

## 5. SYSTEM RESOURCES
- RAM usage, disk space, CPU usage

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "free -h && echo '---' && df -h / && echo '---' && top -bn1 | head -12"
```

## 6. CONFIGURATION REVIEW
- Check key environment variables are set correctly

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "grep -E 'ENABLE_|MODE|STRATEGY|STAKE|STREAMING' /opt/betfair-bot/.env 2>/dev/null | head -20"
```

## 7. BETFAIR SESSION HEALTH (CRITICAL)

The bot can appear "healthy" in `docker ps` while its Betfair session has silently died.
Symptoms: container up, scheduler running, no exceptions — but no bets being placed.

**Run these checks every time:**

```bash
# (a) Count recent "not logged in" warnings. Should be 0 since 2026-04-20 fix
# (re-login now auto-recovers within 15 mins). >3 in the last hour = investigate.
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "docker logs betfair-bot --since 1h 2>&1 | grep -c 'not logged in to Betfair'"

# (b) Confirm a recent successful login or keep-alive
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "docker logs betfair-bot --since 2h 2>&1 | grep -iE 'Successfully logged into Betfair|Session keep-alive successful|attempting re-login' | tail -10"

# (c) Days since last bet placed — flag if bot is up but no bets for >48h
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "docker cp betfair-bot:/app/data/betfair_bot.db /tmp/bf.db && sqlite3 /tmp/bf.db \"SELECT MAX(placed_at), CAST((julianday('now') - julianday(MAX(placed_at))) AS INTEGER) AS days_ago FROM bets;\""
```

**Interpretation:**
- 🔴 `not logged in` warnings recurring AND no `attempting re-login` messages = auto-recovery broken, container restart needed
- 🟡 `attempting re-login` messages present = session dropped but recovered (working as designed)
- 🔴 Container uptime >> days since last bet = trading effectively stopped

## 8. STRATEGY EDGE ASSESSMENT
- Calculate win rate from database
- Is the strategy performing as expected?
- Any parameter tweaks recommended?

## 9. RECOMMENDATIONS
Provide prioritised recommendations:
- P1 (Critical): Issues that need immediate attention
- P2 (Important): Should be addressed soon
- P3 (Nice to have): Optimisations for later

## 10. SUMMARY DASHBOARD
Present a quick status summary table:

| Check | Status | Notes |
|-------|--------|-------|
| Bot Running | ?/? | |
| Database Running | ?/? | |
| Logs Healthy | ?/?/? | |
| Markets Active | ?/? | |
| Resources OK | ?/?/? | |
| Session Valid | ?/? | |
| Strategy Edge | ?/?/? | |

Traffic light summary: ? All good / ? Minor issues / ? Needs attention
