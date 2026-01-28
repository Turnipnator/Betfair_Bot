---
name: healthcheck
description: Run a comprehensive health check on the Betfair trading bot
---

# Betfair Trading Bot Health Check

Run a comprehensive health check on the betfair-bot. Work through each section systematically and provide a summary dashboard at the end.

## VPS Details
- Server: 149.102.144.190
- SSH Key: ~/.ssh/id_ed25519_vps
- Container: betfair-bot (also betfair-db for database)
- Path: /opt/betfair-bot

## 1. PROCESS STATUS
- Are both containers running? (betfair-bot AND betfair-db)
- How long have they been running (uptime)?
- Any recent restarts or crashes?

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "docker ps --format '{{.Names}}\t{{.Status}}\t{{.RunningFor}}' | grep betfair"
```

## 2. LOG ANALYSIS
- Check the last 100 lines of logs for errors, warnings, or anomalies
- Identify any recurring error patterns
- Check Betfair API session status

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "docker compose -f /opt/betfair-bot/docker-compose.yml logs --tail 100 betfair-bot 2>&1"
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "docker logs betfair-bot 2>&1 | grep -iE 'error|warn|fail|session|expired' | tail -20"
```

## 3. SIGNAL GENERATION
- Is the bot actively monitoring markets?
- What was the last bet placed and when?
- Check database for recent activity

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "ls -la /opt/betfair-bot/data/"
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "docker exec betfair-bot cat /app/data/state.json 2>/dev/null || echo 'No state file'"
```

## 4. PERFORMANCE METRICS
- Check current market subscriptions
- Review recent betting P&L from database
- Check today's results

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "docker exec betfair-db psql -U betfair -d betfair -c 'SELECT COUNT(*) as bets, SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END) as wins, SUM(profit) as total_profit FROM bets WHERE created_at > NOW() - INTERVAL '7 days';' 2>/dev/null || echo 'No DB access'"
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "docker exec betfair-bot cat /app/data/daily_stats.json 2>/dev/null || echo 'No daily stats'"
```

## 5. SYSTEM RESOURCES
- RAM usage, disk space, CPU usage

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "free -h && echo '---' && df -h / && echo '---' && top -bn1 | head -12"
```

## 6. CONFIGURATION REVIEW
- Check key environment variables are set correctly

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "grep -E 'ENABLE_|MODE|STRATEGY|STAKE' /opt/betfair-bot/.env 2>/dev/null | head -15"
```

## 7. BETFAIR-SPECIFIC CHECKS
- Betfair API session validity (check for auth errors in logs)
- Current market subscriptions active
- Betting P&L from logs/database
- Any market suspension handling issues

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
