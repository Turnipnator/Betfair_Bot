"""Read-only Betfair snapshot: how close are books to an arb? No bets placed."""
import asyncio, json, re, sys, statistics as st
from collections import defaultdict
sys.path.insert(0, "/app")
from src.betfair.client import betfair_client
from src.models import MarketFilter, Sport

COUNTRIES = ["GB","ES","DE","IT","FR","PT","NL","DK"]

def book(m, side):
    tot = 0.0
    for r in m.runners:
        if r.status != "ACTIVE": continue
        p = r.best_back_price if side=="back" else r.best_lay_price
        if not p: return None
        tot += 1.0/p
    return tot

def cross(m):
    return sum(1 for r in m.runners if r.status=="ACTIVE" and r.best_back_price and r.best_lay_price and r.best_back_price >= r.best_lay_price)

def summ(xs):
    xs=[x for x in xs if x is not None]
    if not xs: return {}
    xs=sorted(xs)
    return {"n":len(xs),"min":round(xs[0],4),"p10":round(xs[len(xs)//10],4),"median":round(st.median(xs),4),"p90":round(xs[9*len(xs)//10],4),"max":round(xs[-1],4)}

async def fetch(sport, types, countries, fh, th, mx):
    ms = await betfair_client.get_markets(MarketFilter(sports=[sport], market_types=types, countries=countries, from_hours=fh, to_hours=th, max_results=mx))
    if not ms: return []
    priced = await betfair_client.get_market_prices([m.market_id for m in ms])
    return list(priced.values())

async def main():
    ok = await betfair_client.login()
    if not ok: print("LOGIN FAILED"); return
    out = {}
    mo = await fetch(Sport.FOOTBALL, ["MATCH_ODDS"], COUNTRIES, 0.5, 24, 120)
    cs = await fetch(Sport.FOOTBALL, ["CORRECT_SCORE"], COUNTRIES, 0.5, 24, 120)
    ou = await fetch(Sport.FOOTBALL, ["OVER_UNDER_25"], COUNTRIES, 0.5, 24, 120)
    hr = await fetch(Sport.HORSE_RACING, ["WIN","PLACE"], ["GB","IE"], 0.0, 20, 300)
    mo=[m for m in mo if not m.in_play]; cs=[m for m in cs if not m.in_play]; ou=[m for m in ou if not m.in_play]; hr=[m for m in hr if not m.in_play]
    out["counts"]={"MO":len(mo),"CS":len(cs),"OU25":len(ou),"HR":len(hr)}

    # 1. Single-market books
    for label, ms in (("MATCH_ODDS",mo),("CORRECT_SCORE",cs),("OVER_UNDER_25",ou),("HR_WIN",[m for m in hr if m.market_type=="WIN"]),("HR_PLACE",[m for m in hr if m.market_type=="PLACE"])):
        liquid=[m for m in ms if m.total_matched>=1000]
        out[label]={"markets":len(ms),"liquid_ge_1k":len(liquid),
            "back_overround_pct":summ([(book(m,"back") or 0)*100 for m in liquid if book(m,"back")]),
            "lay_underround_pct":summ([(book(m,"lay") or 0)*100 for m in liquid if book(m,"lay")]),
            "markets_with_back_ge_lay_cross":sum(1 for m in ms if cross(m)>0),
            "markets_back_book_under_100":sum(1 for m in liquid if book(m,"back") and book(m,"back")<1.0),
            "total_matched_median":round(st.median([m.total_matched for m in ms]),0) if ms else None}

    # 2. MO vs CORRECT_SCORE / OU25 : gross arb iff sum(1/back_cs_cells) <= 1/lay_MO  (ratio<1 = arb before commission)
    cs_by_ev={m.event_id:m for m in cs}; ou_by_ev={m.event_id:m for m in ou}
    ratios=defaultdict(list); worst=[]
    def cells(m, pred):
        tot=0.0
        for r in m.runners:
            if r.status!="ACTIVE" or not r.best_back_price: return None
            mm=re.match(r"(\d+)\s*-\s*(\d+)", r.name)
            if mm: h,a=int(mm.group(1)),int(mm.group(2)); kind="H" if h>a else "A" if a>h else "D"; goals=h+a
            elif "Home" in r.name: kind,goals="H",99
            elif "Away" in r.name: kind,goals="A",99
            elif "Draw" in r.name: kind,goals="D",99
            else: return None
            if pred(kind,goals): tot+=1.0/r.best_back_price
        return tot
    for m in mo:
        c=cs_by_ev.get(m.event_id)
        if c and c.total_matched>=200:
            runners={("H" if i==0 else "A" if i==1 else "D"):r for i,r in enumerate(sorted(m.runners,key=lambda r:r.sort_priority))}
            for k in "HDA":
                r=runners.get(k); s=cells(c, lambda kind,g,k=k: kind==k)
                if r and r.best_lay_price and s:
                    ratio=s/(1.0/r.best_lay_price); ratios["CS_vs_MO"].append(ratio); worst.append((ratio,m.event_name,k,round(s*100,1),r.best_lay_price,c.total_matched))
        o=ou_by_ev.get(m.event_id)
        if c and o and c.total_matched>=200 and o.total_matched>=200:
            over=[r for r in o.runners if r.name.lower().startswith("over")]
            s=cells(c, lambda kind,g: g>=3)
            if over and over[0].best_lay_price and s:
                ratio=s/(1.0/over[0].best_lay_price); ratios["CS_vs_OU25"].append(ratio); worst.append((ratio,m.event_name,"O2.5",round(s*100,1),over[0].best_lay_price,c.total_matched))
    out["cross_market"]={k:summ(v) for k,v in ratios.items()}
    out["cross_market_closest_5"]=sorted(worst)[:5]

    # 3. HR WIN vs PLACE dominance: back PLACE at P, lay WIN at W is riskless iff P >= W
    wins={m.event_id:m for m in hr if m.market_type=="WIN"}; pl=[m for m in hr if m.market_type=="PLACE"]
    dom=[]; viol=[]
    for p in pl:
        w=wins.get(p.event_id)
        if not w or p.total_matched<100: continue
        wr={r.selection_id:r for r in w.runners}
        for r in p.runners:
            wrr=wr.get(r.selection_id)
            if r.status=="ACTIVE" and wrr and r.best_back_price and wrr.best_lay_price:
                ratio=r.best_back_price/wrr.best_lay_price; dom.append(ratio)
                if ratio>=1: viol.append((round(ratio,3),p.event_name,r.name,r.best_back_price,wrr.best_lay_price,p.total_matched))
    out["win_vs_place"]={"pairs":len(dom),"place_back_over_win_lay_ratio":summ(dom),"riskless_violations":len(viol),"examples":sorted(viol,reverse=True)[:5]}

    # 4. Spread cost on MO favourites (what an entry+exit costs)
    spreads=[]
    for m in mo:
        if m.total_matched<1000: continue
        for r in m.runners:
            if r.best_back_price and r.best_lay_price: spreads.append((r.best_lay_price-r.best_back_price)/r.best_back_price*100)
    out["MO_spread_pct"]=summ(spreads)
    print("===JSON==="); print(json.dumps(out, default=str, indent=1))
    await betfair_client.logout()

asyncio.run(main())
