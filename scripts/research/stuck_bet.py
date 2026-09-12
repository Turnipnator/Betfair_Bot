"""Ask Betfair what became of the open live bet(s) in the DB. Read-only."""
import asyncio, sys, sqlite3, json
sys.path.insert(0, "/app")
from src.betfair.client import betfair_client
def dump(o):
    return {k: (str(v) if not isinstance(v,(int,float,str,type(None))) else v) for k,v in vars(o).items() if not k.startswith("_") and k not in ("elapsed_time","streaming_unique_id","streaming_update","publish_time","publish_time_epoch")}
async def main():
    con = sqlite3.connect("/app/data/betfair_bot.db"); con.row_factory = sqlite3.Row
    bets = [dict(r) for r in con.execute("SELECT * FROM bets WHERE status='MATCHED' AND is_paper=0")]
    cols = [r[1] for r in con.execute("PRAGMA table_info(markets)")]
    print("MARKET COLS", cols)
    for b in bets:
        print("DB BET", json.dumps(b, default=str))
        mk = [dict(r) for r in con.execute("SELECT * FROM markets WHERE id=?", (b["market_id"],))]
        print("DB MARKET", json.dumps(mk, default=str))
    ok = await betfair_client.login()
    if not ok: print("LOGIN FAILED"); return
    api = betfair_client._client
    loop = asyncio.get_event_loop()
    for b in bets:
        ref = b["bet_ref"]; mid = b["market_id"]
        for status in ("SETTLED", "VOIDED", "LAPSED", "CANCELLED"):
            try:
                r = await loop.run_in_executor(None, lambda s=status: api.betting.list_cleared_orders(bet_status=s, bet_ids=[ref]))
                print(f"CLEARED {status}: {[dump(o) for o in r.orders]}")
            except Exception as e:
                print(f"CLEARED {status}: ERR {e}")
        try:
            r = await loop.run_in_executor(None, lambda: api.betting.list_current_orders(bet_ids=[ref]))
            print("CURRENT:", [dump(o) for o in r.orders])
        except Exception as e:
            print("CURRENT: ERR", e)
        try:
            r = await loop.run_in_executor(None, lambda: api.betting.list_cleared_orders(bet_status="SETTLED", market_ids=[mid]))
            print("ALL SETTLED ON MARKET:", [dump(o) for o in r.orders])
        except Exception as e:
            print("MARKET SETTLED: ERR", e)
        try:
            books = await loop.run_in_executor(None, lambda: api.betting.list_market_book(market_ids=[mid]))
            for bk in books:
                print("BOOK:", {"status": bk.status, "inplay": bk.inplay, "number_of_winners": bk.number_of_winners, "total_matched": bk.total_matched,
                               "runners": [(r.selection_id, r.status) for r in bk.runners]})
        except Exception as e:
            print("BOOK: ERR", e)
        try:
            cat = await loop.run_in_executor(None, lambda: api.betting.list_market_catalogue(filter={"marketIds":[mid]}, market_projection=["EVENT","RUNNER_DESCRIPTION","MARKET_START_TIME"], max_results=1))
            for c in cat: print("CATALOGUE:", c.market_name, c.event.name if c.event else None, c.market_start_time, [(r.selection_id, r.runner_name) for r in c.runners])
        except Exception as e:
            print("CATALOGUE: ERR", e)
    await betfair_client.logout()
asyncio.run(main())
