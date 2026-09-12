"""Betfair -> football-data.co.uk team-name resolution (12 Sep 2026).

The LTD funnel showed 57 tier-1/2 fixtures in ten days rejected as
``no_stats`` purely because Betfair spells the club differently from the
stats file (Nottm Forest / Nott'm Forest, Sheff Utd / Sheffield United,
Paris St-G / Paris SG ...). `_normalize_team_name` now carries those aliases
and strips club-type affixes (FC, SC, AD, SV, ... / FC, BK, IF ...) from both
sides. These lock every pair seen, and that no two clubs in one league file
collapse to the same key.

Run in the betfair-bot container:
  docker compose exec -T -e PYTHONPATH=/app betfair-bot python tests/test_team_aliases.py
"""
from src.data.football_data import football_data_service

PASS = FAIL = 0


def check(label, got, want):
    global PASS, FAIL
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {label}: got {got!r} want {want!r}")
    PASS += ok
    FAIL += not ok


norm = football_data_service._normalize_team_name

# (Betfair name, football-data.co.uk name) as seen 2–12 Sep 2026.
PAIRS = [
    # England / Scotland
    ("Nottm Forest", "Nott'm Forest"), ("Sheff Utd", "Sheffield United"),
    ("Man Utd", "Man United"), ("Man City", "Man City"), ("Dundee Utd", "Dundee United"),
    ("Inverness CT", "Inverness C"), ("Raith", "Raith Rvs"), ("Burnley", "Burnley"),
    ("Wolves", "Wolves"), ("QPR", "QPR"),
    # Spain
    ("Espanyol", "Espanol"), ("Deportivo", "La Coruna"), ("Racing Santander", "Santander"),
    ("Sporting Gijon", "Sp Gijon"), ("Celta Vigo B", "Celta B"), ("AD Ceuta FC", "Ceuta"),
    ("CD Castellon", "Castellon"), ("FC Andorra", "Andorra"), ("Sociedad B", "Sociedad B"),
    ("Atletico Madrid", "Ath Madrid"), ("Athletic Bilbao", "Ath Bilbao"), ("Celta Vigo", "Celta"),
    # Germany
    ("Mgladbach", "M'gladbach"), ("Hamburger SV", "Hamburg"), ("Arminia Bielefeld", "Bielefeld"),
    ("Dynamo Dresden", "Dresden"), ("FC Heidenheim", "Heidenheim"), ("FC Magdeburg", "Magdeburg"),
    ("SV Darmstadt", "Darmstadt"), ("VfL Osnabruck", "Osnabruck"), ("FC Cologne", "FC Koln"),
    ("Schalke 04", "Schalke 04"), ("Union Berlin", "Union Berlin"), ("St Pauli", "St Pauli"),
    ("Greuther Furth", "Greuther Furth"), ("RB Leipzig", "RB Leipzig"), ("Bayern Munich", "Bayern Munich"),
    # Italy
    ("AC Monza", "Monza"), ("Entella", "Virtus Entella"), ("LR Vicenza Virtus", "Vicenza"),
    ("US Cremonese", "Cremonese"), ("Calcio Avellino SSD", "Avellino"), ("Juve Stabia", "Juve Stabia"),
    ("Inter", "Inter"), ("AC Milan", "Milan"),
    # France
    ("Paris St-G", "Paris SG"), ("Paris FC", "Paris FC"), ("ESTAC Troyes", "Troyes"),
    ("Pau", "Pau FC"), ("Red Star", "Red Star"), ("St Etienne", "St Etienne"),
    # Portugal
    ("Sporting Lisbon", "Sp Lisbon"), ("Braga", "Sp Braga"), ("Estoril Praia", "Estoril"),
    ("Academico de Viseu", "Academico Viseu"), ("CD Nacional Funchal", "Nacional"),
    ("Club Football Estrela", "Estrela"), ("Gil Vicente", "Gil Vicente"), ("Casa Pia", "Casa Pia"),
    # Netherlands
    ("NEC Nijmegen", "Nijmegen"), ("Fortuna Sittard", "For Sittard"), ("ADO Den Haag", "Den Haag"),
    ("Cambuur Leeuwarden", "Cambuur"), ("PEC Zwolle", "Zwolle"), ("SC Telstar", "Telstar"),
    ("AZ Alkmaar", "AZ Alkmaar"), ("PSV", "PSV Eindhoven"), ("Sparta Rotterdam", "Sparta Rotterdam"),
    ("Go Ahead Eagles", "Go Ahead Eagles"), ("Willem II", "Willem II"), ("FC Twente", "Twente"),
    # Denmark
    ("AGF", "Aarhus"), ("OB", "Odense"), ("FC Nordsjaelland", "Nordsjaelland"),
    ("AC Horsens", "Horsens"), ("Randers", "Randers FC"), ("FC Copenhagen", "FC Copenhagen"),
    ("SonderjyskE", "Sonderjyske"), ("Brondby", "Brondby"),
]

print("Betfair name resolves to the football-data name")
for bf, fd in PAIRS:
    check(f"{bf} == {fd}", norm(bf), norm(fd))

print("normalisation is idempotent and case/space-insensitive")
for bf, _ in PAIRS[:10]:
    check(f"idempotent: {bf}", norm(norm(bf)), norm(bf))
check("case and whitespace", norm("  NOTTM   forest "), norm("Nott'm Forest"))

print("affix stripping never empties a name and leaves plain names alone")
check("lone affix survives", norm("FC"), "fc")
check("Paris SG untouched", norm("Paris SG"), "paris sg")
check("Real Madrid untouched", norm("Real Madrid"), "real madrid")
check("Club Brugge keeps 'club'", norm("Club Brugge"), "club brugge")

# football-data.co.uk team lists per league file, 2026/27 as of 12 Sep 2026.
# No two clubs in a file may share a key, or the wrong club's stats get used.
LEAGUE_FILES_SNAPSHOT = {
    "E0": ['Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton', 'Chelsea', 'Coventry', 'Crystal Palace', 'Everton', 'Fulham', 'Hull', 'Ipswich', 'Leeds', 'Liverpool', 'Man City', 'Man United', 'Newcastle', "Nott'm Forest", 'Sunderland', 'Tottenham'],
    "SP1": ['Alaves', 'Ath Bilbao', 'Ath Madrid', 'Barcelona', 'Betis', 'Celta', 'Elche', 'Espanol', 'Getafe', 'La Coruna', 'Levante', 'Malaga', 'Osasuna', 'Real Madrid', 'Santander', 'Sevilla', 'Sociedad', 'Valencia', 'Vallecano', 'Villarreal'],
    "D1": ['Augsburg', 'Bayern Munich', 'Dortmund', 'Ein Frankfurt', 'Elversberg', 'FC Koln', 'Freiburg', 'Hamburg', 'Hoffenheim', 'Leverkusen', "M'gladbach", 'Mainz', 'Paderborn', 'RB Leipzig', 'Schalke 04', 'Stuttgart', 'Union Berlin', 'Werder Bremen'],
    "I1": ['Atalanta', 'Bologna', 'Cagliari', 'Como', 'Fiorentina', 'Frosinone', 'Genoa', 'Inter', 'Juventus', 'Lazio', 'Lecce', 'Milan', 'Monza', 'Napoli', 'Parma', 'Roma', 'Sassuolo', 'Torino', 'Udinese', 'Venezia'],
    "F1": ['Angers', 'Auxerre', 'Brest', 'Le Havre', 'Le Mans', 'Lens', 'Lille', 'Lorient', 'Lyon', 'Marseille', 'Monaco', 'Nice', 'Paris FC', 'Paris SG', 'Rennes', 'Strasbourg', 'Toulouse', 'Troyes'],
    "P1": ['Academico Viseu', 'Alverca', 'Arouca', 'Benfica', 'Casa Pia', 'Estoril', 'Estrela', 'Famalicao', 'Gil Vicente', 'Guimaraes', 'Maritimo', 'Moreirense', 'Nacional', 'Porto', 'Rio Ave', 'Santa Clara', 'Sp Braga', 'Sp Lisbon'],
    "N1": ['AZ Alkmaar', 'Ajax', 'Cambuur', 'Den Haag', 'Excelsior', 'Feyenoord', 'For Sittard', 'Go Ahead Eagles', 'Groningen', 'Heerenveen', 'Nijmegen', 'PSV Eindhoven', 'Sparta Rotterdam', 'Telstar', 'Twente', 'Utrecht', 'Willem II', 'Zwolle'],
    "SC0": ['Aberdeen', 'Celtic', 'Dundee', 'Dundee United', 'Falkirk', 'Hearts', 'Hibernian', 'Kilmarnock', 'Motherwell', 'Rangers', 'St Johnstone', 'St Mirren'],
    "DNK": ['Aarhus', 'Brondby', 'FC Copenhagen', 'Horsens', 'Lyngby', 'Midtjylland', 'Nordsjaelland', 'Odense', 'Randers FC', 'Silkeborg', 'Sonderjyske', 'Viborg'],
    "E1": ['Birmingham', 'Blackburn', 'Bolton', 'Bristol City', 'Burnley', 'Cardiff', 'Charlton', 'Derby', 'Lincoln', 'Middlesbrough', 'Millwall', 'Norwich', 'Portsmouth', 'Preston', 'QPR', 'Sheffield United', 'Southampton', 'Stoke', 'Swansea', 'Watford', 'West Brom', 'West Ham', 'Wolves', 'Wrexham'],
    "SP2": ['Albacete', 'Almeria', 'Andorra', 'Burgos', 'Cadiz', 'Castellon', 'Celta B', 'Ceuta', 'Cordoba', 'Eibar', 'Eldense', 'Girona', 'Granada', 'Las Palmas', 'Leganes', 'Mallorca', 'Oviedo', 'Sabadell', 'Sociedad B', 'Sp Gijon', 'Tenerife', 'Valladolid'],
    "D2": ['Bielefeld', 'Bochum', 'Braunschweig', 'Cottbus', 'Darmstadt', 'Dresden', 'Greuther Furth', 'Hannover', 'Heidenheim', 'Hertha', 'Holstein Kiel', 'Kaiserslautern', 'Karlsruhe', 'Magdeburg', 'Nurnberg', 'Osnabruck', 'St Pauli', 'Wolfsburg'],
    "I2": ['Arezzo', 'Ascoli', 'Avellino', 'Benevento', 'Carrarese', 'Catanzaro', 'Cesena', 'Cremonese', 'Empoli', 'Juve Stabia', 'Mantova', 'Modena', 'Padova', 'Palermo', 'Pisa', 'Sampdoria', 'Sudtirol', 'Verona', 'Vicenza', 'Virtus Entella'],
    "F2": ['Annecy', 'Boulogne', 'Clermont', 'Dijon', 'Dunkerque', 'Grenoble', 'Guingamp', 'Laval', 'Metz', 'Montpellier', 'Nancy', 'Nantes', 'Pau FC', 'Red Star', 'Reims', 'Rodez', 'Sochaux', 'St Etienne'],
    "SC1": ['Arbroath', 'Ayr', 'Dunfermline', 'Inverness C', 'Livingston', 'Morton', 'Partick', 'Queens Park', 'Raith Rvs', 'Stenhousemuir'],
}

print("no two clubs in one league file share a key")
for code, names in LEAGUE_FILES_SNAPSHOT.items():
    keys = [norm(n) for n in names]
    check(f"{code}: {len(names)} clubs, {len(set(keys))} keys", len(set(keys)), len(names))

print(f"\nRESULT: {PASS}/{PASS + FAIL} passed")
raise SystemExit(1 if FAIL else 0)
