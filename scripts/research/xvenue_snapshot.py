"""Read-only cross-venue snapshot: Betfair vs Matchbook vs Smarkets, Match Odds, next 24h. No bets placed."""
import asyncio, json, re, sys, time, statistics as st, unicodedata, urllib.request, urllib.parse
from datetime import datetime, timezone, timedelta
from difflib import SequenceMatcher
sys.path.insert(0, "/app")
from src.betfair.client import betfair_client
from src.models import MarketFilter, Sport

COUNTRIES = ["GB","ES","DE","IT","FR","PT","NL","DK"]
COMM = {"betfair": 0.05, "matchbook": 0.02, "smarkets": 0.02}
ALIAS = {"man utd":"manchester united","man city":"manchester city","spurs":"tottenham","tottenham hotspur":"tottenham","wolves":"wolverhampton","wolverhampton wanderers":"wolverhampton","nottm forest":"nottingham forest","sheff utd":"sheffield united","sheff wed":"sheffield wednesday","west brom":"west bromwich albion","qpr":"queens park rangers","psg":"paris saint germain","paris sg":"paris saint germain","inter":"inter milan","internazionale":"inter milan","atletico madrid":"atletico de madrid","athletic bilbao":"athletic club","bayern munich":"bayern munchen","fc bayern munchen":"bayern munchen","borussia dortmund":"dortmund","bor dortmund":"dortmund","borussia mgladbach":"monchengladbach","borussia monchengladbach":"monchengladbach","mgladbach":"monchengladbach","gladbach":"monchengladbach","hamburger sv":"hamburg","hsv":"hamburg","ac monza":"monza","ac milan":"milan","napoli":"napoli","ssc napoli":"napoli","real betis":"betis","celta vigo":"celta","fc porto":"porto","sporting cp":"sporting lisbon","sporting":"sporting lisbon","psv eindhoven":"psv","ajax amsterdam":"ajax","az alkmaar":"az","fc kobenhavn":"copenhagen","fc copenhagen":"copenhagen","brondby if":"brondby","st mirren":"st mirren","hearts":"heart of midlothian","hibs":"hibernian"}
def norm(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii","ignore").decode().lower()
    s = re.sub(r"\(.*?\)","",s); s = re.sub(r"[^a-z0-9 ]"," ",s)
    s = re.sub(r"\b(fc|cf|sc|afc|ac|as|ss|us|rc|cd|ud|sd|fk|sk|bk|if|sv|vfb|vfl|tsg|fsv|1899|1\.|club|de|women|w)\b"," ",s)
    s = re.sub(r"\s+"," ",s).strip()
    return ALIAS.get(s, s)
def sim(a,b):
    a,b=norm(a),norm(b)
    if not a or not b: return 0
    if a==b or a in b or b in a: return 1.0
    ta,tb=set(a.split()),set(b.split())
    if any(len(t)>=4 for t in ta&tb): return 0.9
    return SequenceMatcher(None,a,b).ratio()
def get(url, params=None):
    if params: url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"accept":"application/json","user-agent":"Mozilla/5.0 (research snapshot)"})
    with urllib.request.urlopen(req, timeout=20) as r: return json.load(r)

def matchbook(t0, t1):
    out=[]; off=0
    while True:
        d = get("https://api.matchbook.com/edge/rest/events", {"sport-ids":15,"per-page":200,"offset":off,"odds-type":"DECIMAL","include-prices":"true","market-types":"one_x_two","states":"open","exchange-type":"back-lay","after":int(t0.timestamp()),"before":int(t1.timestamp())})
        for ev in d.get("events",[]):
            if ev.get("in-running-flag"): continue
            for m in ev.get("markets",[]):
                if m.get("market-type")!="one_x_two": continue
                runners={}
                for r in m.get("runners",[]):
                    backs=[p for p in r.get("prices",[]) if p["side"]=="back"]; lays=[p for p in r.get("prices",[]) if p["side"]=="lay"]
                    bb=max(backs,key=lambda p:p["odds"]) if backs else None; bl=min(lays,key=lambda p:p["odds"]) if lays else None
                    runners[r["name"]]={"back":bb["odds"] if bb else None,"back_size":bb["available-amount"] if bb else 0,"lay":bl["odds"] if bl else None,"lay_size":bl["available-amount"] if bl else 0}
                out.append({"venue":"matchbook","name":ev["name"],"start":datetime.fromisoformat(ev["start"].replace("Z","+00:00")),"runners":runners,"volume":m.get("volume",0),"back_over":m.get("back-overround"),"lay_over":m.get("lay-overround")})
        off += 200
        if off >= d.get("total",0) or not d.get("events"): break
    return out

def smarkets(t0, t1):
    evs=[]; url="https://api.smarkets.com/v3/events/"; params={"state":"upcoming","type":"football_match","limit":100,"sort":"start_datetime,id","start_datetime_min":t0.strftime("%Y-%m-%dT%H:%M:%SZ"),"start_datetime_max":t1.strftime("%Y-%m-%dT%H:%M:%SZ")}
    for _ in range(8):
        d=get(url, params); evs+=d.get("events",[])
        nxt=(d.get("pagination") or {}).get("next_page")
        if not nxt: break
        url=("https://api.smarkets.com/v3/events/"+nxt) if nxt.startswith("?") else ("https://api.smarkets.com"+nxt if nxt.startswith("/") else nxt); params=None
    return evs
def smarkets_prices(evs):
    """Return per-event dict of runner name -> back/lay decimal. Batches contracts+quotes."""
    out=[]
    ids=[e["id"] for e in evs]
    mk_by_ev={}
    for i in range(0,len(ids),20):
        d=get(f"https://api.smarkets.com/v3/events/{','.join(ids[i:i+20])}/markets/")
        for m in d.get("markets",[]):
            if (m.get("market_type") or {}).get("name")=="WINNER_3_WAY" and m.get("state")=="open": mk_by_ev.setdefault(m["event_id"], m["id"])
    mids=list(mk_by_ev.values())
    contracts={}; quotes={}
    for i in range(0,len(mids),20):
        d=get(f"https://api.smarkets.com/v3/markets/{','.join(mids[i:i+20])}/contracts/")
        for c in d.get("contracts",[]): contracts.setdefault(c["market_id"],[]).append(c)
        quotes.update(get(f"https://api.smarkets.com/v3/markets/{','.join(mids[i:i+20])}/quotes/"))
    ev_by_id={e["id"]:e for e in evs}
    for evid, mid in mk_by_ev.items():
        runners={}
        for c in contracts.get(mid,[]):
            q=quotes.get(c["id"],{}); bids=q.get("bids",[]); offers=q.get("offers",[])
            bo=min(offers,key=lambda x:x["price"]) if offers else None; bb=max(bids,key=lambda x:x["price"]) if bids else None
            runners[c["name"]]={"back":round(10000/bo["price"],3) if bo else None,"back_size_raw":bo["quantity"] if bo else 0,"lay":round(10000/bb["price"],3) if bb else None,"lay_size_raw":bb["quantity"] if bb else 0}
        e=ev_by_id[evid]
        out.append({"venue":"smarkets","name":e["name"],"start":datetime.fromisoformat(e["start_datetime"].replace("Z","+00:00")),"runners":runners,"volume":None})
    return out

def hda(entry, home, away):
    """Map a venue's runner dict to H/D/A using team names."""
    res={}
    for rname, px in entry["runners"].items():
        if norm(rname) in ("draw","the draw","tie"): res["D"]=px; continue
        sh, sa = sim(rname,home), sim(rname,away)
        if max(sh,sa)>=0.6: res["H" if sh>=sa else "A"]=px
    return res

def arb(a, cA, b, cB, s=100.0):
    """Back s at a (venue comm cA), lay at b (venue comm cB); equal-profit lay stake. Returns (net_profit, liability)."""
    l = s*((a-1)*(1-cA)+1)/(b-cB)
    return -s + l*(1-cB), s + l*(b-1)

async def main():
    now=datetime.now(timezone.utc); t0=now+timedelta(minutes=30); t1=now+timedelta(hours=24)
    ok=await betfair_client.login()
    if not ok: print("LOGIN FAILED"); return
    bf=[]
    for countries, mx in ((COUNTRIES,150),([],60)):
        ms=await betfair_client.get_markets(MarketFilter(sports=[Sport.FOOTBALL], market_types=["MATCH_ODDS"], countries=countries, from_hours=0.5, to_hours=24, max_results=mx))
        if not ms: continue
        if not countries: ms=[m for m in ms if any(k in (m.competition or "").lower() for k in ("champions league","europa league","conference league"))]
        priced=await betfair_client.get_market_prices([m.market_id for m in ms])
        for m in priced.values():
            if m.in_play or any(x["market_id"]==m.market_id for x in bf): continue
            rs=sorted(m.runners,key=lambda r:r.sort_priority)
            if len(rs)!=3: continue
            home,away=rs[0].name,rs[1].name
            runners={}
            for key,r in zip("HAD",rs):
                runners[key]={"back":r.best_back_price,"back_size":r.back_prices[0].size if r.back_prices else 0,"lay":r.best_lay_price,"lay_size":r.lay_prices[0].size if r.lay_prices else 0}
            bf.append({"market_id":m.market_id,"name":m.event_name,"home":home,"away":away,"start":m.start_time if m.start_time.tzinfo else m.start_time.replace(tzinfo=timezone.utc),"comp":m.competition,"volume":m.total_matched,"hda":runners})
    await betfair_client.logout()
    mb=matchbook(t0,t1); sm_ev=smarkets(t0,t1); sm=smarkets_prices(sm_ev)
    out={"betfair_fixtures":len(bf),"matchbook_markets":len(mb),"smarkets_events":len(sm_ev),"smarkets_priced":len(sm),"snapshot_utc":now.isoformat(timespec="seconds")}
    # Matchbook overround for comparison
    mbo=[m["back_over"] for m in mb if m.get("back_over") and (m.get("volume") or 0)>=1000]
    out["matchbook_back_overround_liquid"]={"n":len(mbo),"median":round(st.median(mbo),3) if mbo else None,"min":round(min(mbo),3) if mbo else None}
    rows=[]; matched=0; unmatched=[]
    for f in bf:
        venues={"betfair":f["hda"]}
        for pool in (mb,sm):
            best=None; bs=0
            for e in pool:
                if abs((e["start"]-f["start"]).total_seconds())>15*60: continue
                s=sim(e["name"].split(" vs ")[0] if " vs " in e["name"] else e["name"], f["home"]) + sim(e["name"].split(" vs ")[-1] if " vs " in e["name"] else e["name"], f["away"])
                if s>bs: bs, best = s, e
            if best and bs>=1.4:
                h=hda(best, f["home"], f["away"])
                if len(h)==3: venues[best["venue"]]=h
        if len(venues)==1: unmatched.append(f["name"]); continue
        matched+=1
        for k in "HDA":
            for va,pa in venues.items():
                for vb,pb in venues.items():
                    if va==vb: continue
                    a=pa.get(k,{}).get("back"); b=pb.get(k,{}).get("lay")
                    if not a or not b: continue
                    gross=a/b-1; net,liab=arb(a,COMM[va],b,COMM[vb])
                    rows.append({"fixture":f["name"],"comp":f["comp"],"start":f["start"].strftime("%a %H:%M"),"outcome":k,"back_venue":va,"back":a,"back_size":pa[k].get("back_size",pa[k].get("back_size_raw")),"lay_venue":vb,"lay":b,"lay_size":pb[k].get("lay_size",pb[k].get("lay_size_raw")),"gross_pct":round(gross*100,2),"net_per_100":round(net,2),"roi_on_liab_pct":round(net/liab*100,2),"bf_volume":f["volume"]})
    out["fixtures_matched_on_2plus_venues"]=matched; out["unmatched_sample"]=unmatched[:12]; out["unmatched_count"]=len(unmatched)
    out["pairs"]=len(rows)
    out["gross_gap_pct_dist"]=None
    g=sorted(r["gross_pct"] for r in rows)
    if g: out["gross_gap_pct_dist"]={"min":g[0],"p50":g[len(g)//2],"p90":g[9*len(g)//10],"p99":g[int(0.99*len(g))],"max":g[-1]}
    out["gross_positive"]=sum(1 for r in rows if r["gross_pct"]>0); out["net_positive"]=sum(1 for r in rows if r["net_per_100"]>0)
    out["net_positive_at_bf_6pct"]=sum(1 for r in rows if arb(r["back"], 0.06 if r["back_venue"]=="betfair" else COMM[r["back_venue"]], r["lay"], 0.06 if r["lay_venue"]=="betfair" else COMM[r["lay_venue"]])[0]>0)
    out["top10"]=sorted(rows,key=lambda r:-r["net_per_100"])[:10]
    bf_liquid=[r for r in rows if r["bf_volume"]>=5000]
    out["top5_bf_liquid_ge_5k"]=sorted(bf_liquid,key=lambda r:-r["net_per_100"])[:5]
    print("===JSON==="); print(json.dumps(out, default=str, indent=1))
asyncio.run(main())
