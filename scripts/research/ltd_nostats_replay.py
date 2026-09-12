import asyncio, sys, difflib, json, os
sys.path.insert(0, "/app")
import httpx
_orig = httpx.AsyncClient.__init__
def _patched(self, *a, **kw):
    kw.setdefault("follow_redirects", True); _orig(self, *a, **kw)
httpx.AsyncClient.__init__ = _patched
from src.data.football_data import football_data_service, LEAGUE_TIERS
FIX = """Danish Superliga|AGF v Midtjylland
Danish Superliga|FC Copenhagen v FC Nordsjaelland
Danish Superliga|SonderjyskE v AC Horsens
Danish Superliga|OB v FC Copenhagen
Danish Superliga|Brondby v Randers
Dutch Eredivisie|Sparta Rotterdam v PEC Zwolle
Dutch Eredivisie|NEC Nijmegen v Feyenoord
Dutch Eredivisie|SC Telstar v Cambuur Leeuwarden
Dutch Eredivisie|ADO Den Haag v Fortuna Sittard
Dutch Eredivisie|FC Twente v SC Telstar
Dutch Eredivisie|AZ Alkmaar v Willem II
Dutch Eredivisie|Fortuna Sittard v Ajax
English Premier League|Nottm Forest v Tottenham
English Premier League|Crystal Palace v Ipswich
English Premier League|Aston Villa v Nottm Forest
English Premier League|Chelsea v Hull
English Sky Bet Championship|Sheff Utd v Norwich
English Sky Bet Championship|Wrexham v Burnley
English Sky Bet Championship|Blackburn v Sheff Utd
English Sky Bet Championship|Cardiff v Stoke
English Sky Bet Championship|Bolton v West Ham
English Sky Bet Championship|Swansea v Burnley
English Sky Bet Championship|Preston v Lincoln
French Ligue 1|Paris St-G v Monaco
French Ligue 1|ESTAC Troyes v Strasbourg
French Ligue 2|Pau v Sochaux
French Ligue 2|Dijon v Laval
French Ligue 2|Montpellier v Pau
French Ligue 2|Sochaux v Nantes
German Bundesliga|Mgladbach v Elversberg
German Bundesliga|Hamburger SV v Mainz
German Bundesliga|Union Berlin v Schalke 04
German Bundesliga|Dortmund v Paderborn
German Bundesliga 2|Arminia Bielefeld v St Pauli
German Bundesliga 2|Kaiserslautern v SV Darmstadt
German Bundesliga 2|Dynamo Dresden v Bochum
German Bundesliga 2|Greuther Furth v FC Heidenheim
German Bundesliga 2|VfL Osnabruck v Braunschweig
German Bundesliga 2|Hertha Berlin v FC Magdeburg
Italian Serie A|Parma v AC Monza
Italian Serie A|Venezia v Fiorentina
Italian Serie A|Genoa v Frosinone
Italian Serie B|Modena v Calcio Avellino SSD
Italian Serie B|Entella v LR Vicenza Virtus
Italian Serie B|US Cremonese v Padova
Italian Serie B|Empoli v Arezzo
Italian Serie B|Pisa v Entella
Italian Serie B|Benevento v Verona
Italian Serie B|LR Vicenza Virtus v Juve Stabia
Italian Serie B|Cesena v US Cremonese
Portuguese Primeira Liga|Club Football Estrela v Famalicao
Portuguese Primeira Liga|Alverca v Braga
Portuguese Primeira Liga|Sporting Lisbon v CD Nacional Funchal
Portuguese Primeira Liga|Gil Vicente v Academico de Viseu
Portuguese Primeira Liga|Estoril Praia v Arouca
Portuguese Primeira Liga|Academico de Viseu v Guimaraes
Scottish Championship|Inverness CT v Raith
Scottish Championship|Morton v Livingston
Scottish Championship|Raith v Arbroath
Scottish Championship|Ayr v Inverness CT
Scottish Premiership|Motherwell v Dundee Utd
Scottish Premiership|Dundee Utd v Falkirk
Scottish Premiership|St Johnstone v Celtic
Spanish La Liga|Rayo Vallecano v Racing Santander
Spanish La Liga|Villarreal v Deportivo
Spanish La Liga|Espanyol v Sevilla
Spanish La Liga|Racing Santander v Alaves
Spanish La Liga|Osasuna v Espanyol
Spanish Segunda Division|Sporting Gijon v Girona
Spanish Segunda Division|AD Ceuta FC v Celta Vigo B
Spanish Segunda Division|Valladolid v FC Andorra
Spanish Segunda Division|CD Castellon v Albacete
Spanish Segunda Division|Burgos v AD Ceuta FC
Spanish Segunda Division|FC Andorra v Sociedad B
Spanish Segunda Division|Girona v CD Castellon"""
async def main():
    loaded={}
    for code in LEAGUE_TIERS:
        ls = await football_data_service.get_league_stats(code)
        loaded[code] = (len(ls.teams), round(ls.prior_weight,2), int(ls.total_matches)) if ls else None
    print("LOADED", json.dumps(loaded))
    norm = football_data_service._normalize_team_name
    allnames={}
    for code, ls in football_data_service._cache.items():
        for n in ls.teams: allnames.setdefault(norm(n), []).append((n, code))
    ok=0; miss={}; rows=[l.split("|") for l in FIX.splitlines()]
    for comp, ev in rows:
        h,a=[x.strip() for x in ev.split(" v ",1)]
        if await football_data_service.get_match_stats(h,a): ok+=1; continue
        for t in (h,a):
            nt=norm(t)
            if nt in allnames: continue
            close=difflib.get_close_matches(nt, list(allnames), n=2, cutoff=0.5)
            d=miss.setdefault(t, {"comp":comp,"close":[(c, allnames[c][0]) for c in close],"n":0}); d["n"]+=1
    print(f"REPLAYED {len(rows)}; resolve with redirects followed: {ok}; still-unresolved names: {len(miss)}")
    for t,d in sorted(miss.items(), key=lambda kv:(kv[1]['comp'], -kv[1]['n'])):
        print(f"{d['comp'][:26]:26} | {t:26} | close: {d['close']} | x{d['n']}")
    await football_data_service.close()
asyncio.run(main())
