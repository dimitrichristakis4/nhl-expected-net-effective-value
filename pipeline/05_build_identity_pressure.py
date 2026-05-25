import pandas as pd
import numpy as np
import os
import unicodedata
import re
import warnings

warnings.filterwarnings('ignore')

# Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR  = os.path.dirname(CURRENT_DIR) 

# Source data in current folder
BDB_CSV_PATH = os.path.join(CURRENT_DIR, '2022_2023_NHL_PbP_Logs.csv')
SHOTS_PATH   = os.path.join(CURRENT_DIR, 'shots_2022.csv')

# Reference data in root
BRIDGE_PATH  = os.path.join(SCRIPT_DIR, 'game_id_bridge.csv')

# Output to current folder for accessibility
OUT_PATH     = os.path.join(CURRENT_DIR, 'pbp_identity_pressure_lookup.csv')

# Constants
REG_SEASON_END = pd.Timestamp('2023-04-15')
SHOT_TYPES     = {'Shot', 'Goal', 'Missed Shot'}
POSSESSION_LOSS_TYPES = {'Giveaway', 'Takeaway'}

TEAM_MAP = {
    'Anaheim Ducks':        'ANA', 'Arizona Coyotes':      'ARI',
    'Boston Bruins':        'BOS', 'Buffalo Sabres':        'BUF',
    'Calgary Flames':       'CGY', 'Carolina Hurricanes':   'CAR',
    'Chicago Blackhawks':   'CHI', 'Colorado Avalanche':    'COL',
    'Columbus Blue Jackets':'CBJ', 'Dallas Stars':          'DAL',
    'Detroit Red Wings':    'DET', 'Edmonton Oilers':       'EDM',
    'Florida Panthers':     'FLA', 'Los Angeles Kings':     'LAK',
    'Minnesota Wild':       'MIN', 'Montreal Canadiens':    'MTL',
    'Nashville Predators':  'NSH', 'New Jersey Devils':     'NJD',
    'New York Islanders':   'NYI', 'New York Rangers':      'NYR',
    'Ottawa Senators':      'OTT', 'Philadelphia Flyers':   'PHI',
    'Pittsburgh Penguins':  'PIT', 'San Jose Sharks':       'SJS',
    'Seattle Kraken':       'SEA', 'St. Louis Blues':       'STL',
    'Tampa Bay Lightning':  'TBL', 'Toronto Maple Leafs':   'TOR',
    'Vancouver Canucks':    'VAN', 'Vegas Golden Knights':  'VGK',
    'Washington Capitals':  'WSH', 'Winnipeg Jets':         'WPG',
}

PLAYER_JERSEY_MAP = {
    "PIT": {"CASEY DESMITH": 1, "CHAD RUHWEDEL": 2, "TAYLOR FEDUN": 4, "DREW OCONNOR": 10, "ALEXANDER NYLANDER": 11, "JOSH ARCHIBALD": 15, "JASON ZUCKER": 16, "BRYAN RUST": 17, "DRAKE CAGGIULA": 18, "MARCUS PETTERSSON": 28, "SAMUEL POULIN": 22, "RYAN POEHLING": 25, "JEFF PETRY": 26, "DANTON HEINEN": 43, "JAN RUTTA": 44, "JONATHAN GRUDEN": 45, "MARK FRIEDMAN": 52, "TEDDY BLUEGER": 53, "KRIS LETANG": 58, "JAKE GUENTZEL": 59, "RICKARD RAKELL": 67, "FILIP HALLANDER": 36, "PIERRE-OLIVIER JOSEPH": 73, "JEFF CARTER": 77, "SIDNEY CROSBY": 87, "EVGENI MALKIN": 71, "DUSTIN TOKARSKI": 40, "TRISTAN JARRY": 35},
    "ANA": {"LUKAS DOSTAL": 1, "JOHN KLINGBERG": 3, "CAM FOWLER": 4, "URHO VAAKANAINEN": 5, "JAMIE DRYSDALE": 6, "JAYSON MEGNA": 7, "TREVOR ZEGRAS": 11, "ADAM HENRIQUE": 14, "RYAN STROME": 16, "TROY TERRY": 19, "BRETT LEASON": 20, "ISAC LUNDESTROM": 21, "KEVIN SHATTENKIRK": 22, "BROCK MCGINN": 23, "NATHAN BEAULIEU": 28, "DMITRY KULIKOV": 29, "OLLE ERIKSSON EK": 31, "SAM CARRICK": 39, "PAVOL REGENDA": 40, "ANTHONY STOLARZ": 41, "GLENN GAWDIN": 42, "DREW HELLESON": 43, "MAX COMTOIS": 44, "COLTON WHITE": 45, "BENOIT-OLIVIER GROULX": 50, "JUSTIN KIRKLAND": 54, "JACKSON LACOMBE": 60, "NIKITA NESTERENKO": 62, "MASON MCTAVISH": 37, "DEREK GRANT": 38, "JAKOB SILFVERBERG": 33, "FRANK VATRANO": 77, "JOHN GIBSON": 36},
    "TBL": {"BRIAN ELLIOTT": 1, "PHILIPPE MYERS": 5, "HAYDN FLEURY": 7, "COREY PERRY": 10, "ALEX BARRE-BOULET": 12, "PAT MAROON": 14, "ZACH BOGOSIAN": 24, "IAN COLE": 28, "NICKLAUS PERBIX": 48, "DARREN RADDYSH": 43, "COLE KOEPKE": 45, "ANTHONY CIRELLI": 71, "VICTOR HEDMAN": 77, "ROSS COLTON": 79, "ERIK CERNAK": 81, "GABRIEL FORTIER": 82, "NIKITA KUCHEROV": 86, "ANDREI VASILEVSKIY": 88, "STEVEN STAMKOS": 91, "MIKHAIL SERGACHEV": 98, "BRANDON HAGEL": 38, "BRAYDEN POINT": 21, "NICK PAUL": 20},
    "OTT": {"DYLAN FERGUSON": 1, "ARTEM ZUB": 2, "NICK HOLDEN": 5, "JAKOB CHYCHRUN": 6, "BRADY TKACHUK": 7, "ALEX DEBRINCAT": 12, "MARK KASTELIC": 12, "BO HORVAT": 14, "TYLER MOTTE": 14, "TIM STUTZLE": 18, "DRAKE BATHERSON": 19, "MATHIEU JOSEPH": 21, "NIKITA ZAITSEV": 22, "TRAVIS HAMONIC": 23, "JACOB BERNARD-DOCKER": 24, "ERIK BRANNSTROM": 26, "DYLAN GAMBRELL": 27, "PARKER KELLY": 27, "CLAUDE GIROUX": 28, "DILLON HEATHERINGTON": 29, "ANTON FORSBERG": 31, "JACOB LARSSON": 32, "CAM TALBOT": 33, "LEEVI MERILAINEN": 35, "JAKE LUCCHINI": 36, "THOMAS CHABOT": 72, "MADS SOGAARD": 40, "DERICK BRASSARD": 61, "ROURKE CHARTIER": 67, "LASSI THOMSON": 60, "SHANE PINTO": 57, "JAKE SANDERSON": 85, "AUSTIN WATSON": 16, "TYLER KLEVEN": 43, "EGOR SOKOLOV": 75},
    "STL": {"THOMAS GREISS": 1, "NICK LEDDY": 4, "MARCO SCANDELLA": 6, "TYLER PITLICK": 9, "BRAYDEN SCHENN": 10, "ALEXEY TOROPCHENKO": 13, "JAKUB VRANA": 15, "JOSH LEIVO": 17, "ROBERT THOMAS": 18, "BRANDON SAAD": 20, "JORDAN KYROU": 25, "NATHAN WALKER": 26, "SAMMY BLAIS": 79, "NIKITA ALEXANDROV": 59, "LOGAN BROWN": 22, "NIKITA SOSHNIKOV": 41, "WILLIAM BITTEN": 42, "CALLE ROSEN": 43, "DMITRI SAMORUKOV": 37, "STEVEN SANTINI": 36, "TYLER TUCKER": 75, "JUSTIN FAULK": 72, "COLTON PARAYKO": 55, "PAVEL BUCHNEVICH": 89, "JORDAN BINNINGTON": 50, "JOEL HOFER": 30, "VLADIMIR TARASENKO": 91, "RYAN OREILLY": 90},
    "BUF": {"UKKO-PEKKA LUUKKONEN": 1, "JEREMY DAVIES": 4, "HENRI JOKIHARJU": 10, "JORDAN GREENWAY": 12, "LUKAS ROUSEK": 13, "ANDERS BJORK": 15, "PEYTON KREBS": 19, "TYSON JOST": 17, "KYLE OKPOSO": 21, "JACK QUINN": 22, "MATTIAS SAMUELSSON": 23, "DYLAN COZENS": 24, "OWEN POWER": 25, "RASMUS DAHLIN": 26, "ZEMGUS GIRGENSONS": 28, "VINNIE HINOSTROZA": 29, "ERIC COMRIE": 31, "LAWRENCE PILUT": 20, "JJ PETERKA": 77, "VICTOR OLOFSSON": 71, "TAGE THOMPSON": 72, "JEFF SKINNER": 53, "CASEY MITTELSTADT": 37, "KALE CLAGUE": 38, "ALEX TUCH": 89, "CRAIG ANDERSON": 41},
    "BOS": {"JEREMY SWAYMAN": 1, "JAKE MCCABE": 2, "MIKE REILLY": 6, "AJ GREER": 10, "TRENT FREDERIC": 11, "CRAIG SMITH": 12, "CHARLIE COYLE": 13, "CHRIS WAGNER": 14, "NICK FOLIGNO": 17, "PAVEL ZACHA": 18, "DEREK FORBORT": 28, "HAMPUS LINDHOLM": 27, "LINUS ULLMARK": 35, "PATRICE BERGERON": 37, "JOONA KOPPANEN": 45, "DAVID KREJCI": 46, "OSKAR STEEN": 62, "BRAD MARCHAND": 63, "CHARLIE MCAVOY": 73, "DAVID PASTRNAK": 88, "TYLER BERTUZZI": 59, "DMITRY ORLOV": 81, "KEITH KINKAID": 30, "ANTON STRALMAN": 86, "VINNI LETTIERI": 95, "JAKUB LAUKO": 94},
    "MIN": {"CALEN ADDISON": 2, "JONATHON MERRILL": 4, "JACOB MIDDLETON": 5, "DAKOTA MERMIS": 6, "BROCK FABER": 7, "TYSON JOST": 10, "MATT BOLDY": 12, "SAM STEEL": 13, "JOEL ERIKSSON EK": 14, "MASON SHAW": 15, "MARCUS FOLIGNO": 17, "JORDAN GREENWAY": 18, "NIC PETAN": 19, "BRANDON DUHAIME": 21, "KEVIN ROONEY": 22, "MARCO ROSSI": 23, "MATHEW DUMBA": 24, "JONAS BRODIN": 25, "CONNOR DEWAR": 26, "STEVEN FOGARTY": 28, "MARC-ANDRE FLEURY": 29, "MATS ZUCCARELLO": 36, "RYAN HARTMAN": 38, "GUSTAV NYQUIST": 28, "JARED SPURGEON": 46, "MARCUS JOHANSSON": 90, "ADAM BECKMAN": 53, "JOSEPH CRAMAROSSA": 56, "DAMIEN GIROUX": 68, "KIRILL KAPRIZOV": 97, "FILIP GUSTAVSSON": 32, "FREDERICK GAUDREAU": 89},
}

def clean_id(x):
    if pd.isna(x): return "UNKNOWN"
    try:
        # Convert to float then int to handle scientific notation
        return str(int(float(x)))
    except:
        return str(x).strip()

def find_col(kw, cols):
    kw_clean = "".join(filter(str.isalnum, kw.upper()))
    for c in cols:
        c_clean = "".join(filter(str.isalnum, str(c).upper()))
        if kw_clean in c_clean:
            return c
    return None

def normalize_name(s):
    if pd.isna(s): return "UNKNOWN"
    s = str(s).upper()
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')
    s = s.replace('.', '')
    aliases = {
        "ALEX BARR BOULET": "ALEX BARRE BOULET",
        "ALEXIS LAFRENIRE": "ALEXIS LAFRENIERE",
        "EVGENII DADONOV":  "EVGENY DADONOV",
        "JANI HAKANP":      "JANI HAKANPEE",
        "TIM STTZLE":       "TIM STUTZLE",
        "TONY DEANGELO":    "ANTHONY DEANGELO",
        "CHRIS TANEV":      "CHRISTOPHER TANEV",
    }
    s = aliases.get(s, s)
    s = s.replace('-', ' ')
    s = re.sub(r'\s+(JR|SR|II|III|IV)$', '', s)
    s = " ".join(s.split())
    return s


def time_to_sec(t):
    if pd.isna(t):
        return None
    if isinstance(t, str):
        parts = t.strip().split(':')
        try:
            if len(parts) == 3: # HH:MM:SS (Seen in CSV)
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            if len(parts) == 2: # MM:SS (Seen in Excel)
                return int(parts[0]) * 60 + int(parts[1])
        except ValueError:
            return None
        return None
    try:
        return t.hour * 3600 + t.minute * 60 + t.second
    except AttributeError:
        return None


def build_identity_lookup():
    print("=" * 65)
    print("Step 1: Loading Datasets (Volume Anchor: MoneyPuck)")
    print("=" * 65)

    # 1. Load MoneyPuck Baseline Anchor
    # This represents the "100% volume" target of 85,537 shots.
    shots_raw = pd.read_csv(SHOTS_PATH)
    bridge = pd.read_csv(BRIDGE_PATH)
    
    # Clean Bridge IDs as integers
    bridge['game_id'] = bridge['game_id'].apply(lambda x: int(float(x)) if pd.notna(x) else 0)
    bridge['BDB_Game_Id'] = bridge['BDB_Game_Id'].apply(clean_id)
    id_map = dict(zip(bridge['game_id'], bridge['BDB_Game_Id']))
    
    shots_raw['BDB_Game_Id'] = (shots_raw['game_id'] % 100000).astype(int).map(id_map).fillna("UNKNOWN")
    
    # Fix 6: Investigate and document missing games
    expected_games = 1312
    actual_games = shots_raw[shots_raw['BDB_Game_Id'] != "UNKNOWN"]['BDB_Game_Id'].nunique()
    print(f"  Games matched via bridge: {actual_games} / {expected_games}")
    print(f"  Missing: {expected_games - actual_games}")
    
    missing_mask = shots_raw['BDB_Game_Id'] == "UNKNOWN"
    missing_games = shots_raw[missing_mask]['game_id'].unique()
    print(f"  MoneyPuck game_ids with no bridge match: {len(missing_games)}")
    if len(missing_games) > 0:
        print(f"  First 20 missing IDs: {missing_games[:20]}")

    mp_5v5 = shots_raw[
        (shots_raw['isPlayoffGame']    == 0) &
        (shots_raw['awaySkatersOnIce'] == 5) &
        (shots_raw['homeSkatersOnIce'] == 5) &
        (shots_raw['shotOnEmptyNet']   == 0) &
        (shots_raw['awayEmptyNet']     == 0) &
        (shots_raw['homeEmptyNet']     == 0) 
        # Include rebounds so total volume matches MoneyPuck anchor
    ].copy()
    
    mp_5v5['Shooter_Norm'] = mp_5v5['shooterName'].apply(normalize_name)
    
    # Keep shotRush and time for validation and matching
    mp_cols = ['BDB_Game_Id', 'period', 'shooterName', 'Shooter_Norm', 'shotID', 'teamCode', 'shotRush', 'time']
    mp_5v5 = mp_5v5[[c for c in mp_cols if c in mp_5v5.columns]].copy()
    
    # Map jersey numbers to MoneyPuck anchor
    def map_jersey(row):
        return PLAYER_JERSEY_MAP.get(row['teamCode'], {}).get(row['Shooter_Norm'], 0)
    
    mp_5v5['Shooter_Number'] = mp_5v5.apply(map_jersey, axis=1)
    
    mp_5v5 = mp_5v5.sort_values(['BDB_Game_Id', 'period', 'shotID'])
    mp_5v5['Shot_Order_Identity'] = mp_5v5.groupby(['BDB_Game_Id', 'period', 'Shooter_Norm']).cumcount() + 1
    
    print(f"  MoneyPuck 5v5 Anchor: {len(mp_5v5):,} shots.")

    # 2. Load BDB Enrichment Source
    # Use low_memory=False to avoid type warnings
    bdb_pbp = pd.read_csv(BDB_CSV_PATH, low_memory=False)
    print(f"  BDB Raw Rows Loaded: {len(bdb_pbp):,}")
    
    # Clean headers defensively: remove \n, strip whitespace, collapse internal spaces
    clean_cols = [" ".join(str(c).replace('\n', ' ').strip().split()) for c in bdb_pbp.columns]
    bdb_pbp.columns = clean_cols

    # Find columns by keyword to be immune to header name drift
    raw_cols = bdb_pbp.columns
    col_zone    = find_col('PLAY ZONE Defensive', raw_cols)
    col_team    = find_col('TEAM MAKING THE PLAY Faceoff Won', raw_cols)
    col_game    = find_col('BIGDATABALL GAME ID', raw_cols)
    col_time    = find_col('ELAPSED TIME', raw_cols)
    col_type    = find_col('PLAY TYPE', raw_cols)
    col_shooter = find_col('PLAYER IN THE EVENT #1', raw_cols)
    col_home    = find_col('HOME TEAM', raw_cols)
    col_away    = find_col('AWAY TEAM', raw_cols)
    col_desc    = find_col('PLAY DESCRIPTION', raw_cols)
    col_h_goalie = find_col('HOME GOALIE', raw_cols)
    col_a_goalie = find_col('AWAY GOALIE', raw_cols)
    col_xc       = find_col('X COORDINATE', raw_cols)
    col_yc       = find_col('Y COORDINATE', raw_cols)

    # Debug: Check ELAPSED TIME column
    print(f"  Target col_time: '{col_time}'")
    print(f"  First 5 values of '{col_time}':\n{bdb_pbp[col_time].head().tolist()}")
    print(f"  Dtype of '{col_time}': {bdb_pbp[col_time].dtype}")

    # Ensure critical columns exist
    for cname in [col_zone, col_team, col_game, col_time, col_type, col_shooter, col_home, col_away]:
        if cname not in bdb_pbp.columns:
            print(f"  ERROR: Critical BDB column '{cname}' not found!")
            print(f"  Cleaned columns available: {clean_cols}")
            raise KeyError(f"Critical BDB column '{cname}' not found!")

    # Rename roster columns: "AWAY TEAM'S PLAYER ON THE ICE #1" -> awayPlayer1
    original_roster_cols = [c for c in clean_cols if 'PLAYER ON THE ICE' in c and '#' in c]
    new_roster_names = {}
    for c in original_roster_cols:
        side = 'away' if 'AWAY' in c else 'home'
        num  = c.split('#')[-1].strip()
        new_roster_names[c] = f"{side}Player{num}"

    bdb_pbp = bdb_pbp.rename(columns=new_roster_names)
    roster_cols = list(new_roster_names.values())

    # Strip STRENGTH defensively
    if 'STRENGTH' in bdb_pbp.columns:
        bdb_pbp['STRENGTH'] = bdb_pbp['STRENGTH'].astype(str).str.strip()
    
    bdb_pbp['Seconds_Elapsed'] = bdb_pbp[col_time].apply(time_to_sec)
    print(f"  Rows with Seconds_Elapsed: {bdb_pbp['Seconds_Elapsed'].notna().sum():,}")
    
    bdb_pbp = bdb_pbp.dropna(subset=['Seconds_Elapsed'])
    print(f"  After dropping NaN elapsed: {len(bdb_pbp):,}")
    
    bdb_pbp['Shooter_Norm'] = bdb_pbp[col_shooter].apply(normalize_name)
    bdb_pbp['BDB_Game_Id'] = bdb_pbp[col_game].astype(str)

    # Filter BDB to 5v5 and Goalie Present for faster iteration
    bdb_pbp = bdb_pbp[bdb_pbp['STRENGTH'] == '5-on-5'].copy()
    if col_h_goalie and col_a_goalie:
        bdb_pbp = bdb_pbp[(bdb_pbp[col_h_goalie] != '-') & (bdb_pbp[col_a_goalie] != '-')].copy()
    
    if len(bdb_pbp) == 0:
        # Debugging info
        print("  WARNING: 0 rows found for '5-on-5' strength.")
        if 'STRENGTH' in bdb_pbp.columns:
            print(f"  Available strengths: {bdb_pbp['STRENGTH'].unique()}")
        raise ValueError("BDB 5v5 source is empty after filtering!")

    print(f"  BDB 5v5 Source rows: {len(bdb_pbp):,}")

    # For Fix 3 & 4A output
    zone_entry_records = []

    # Step 2: Build OZ Pressure Timer
    print("\nStep 2: Building OZ Pressure Timer...")
    
    pressure_rows = []
    game_groups = bdb_pbp.groupby('BDB_Game_Id')
    total_games = len(game_groups)
    game_count = 0
    games_with_default_sign = []
    
    for g_id, g_df in game_groups:
        game_count += 1
        if game_count % 100 == 0:
            print(f"  Processing game {game_count}/{total_games}...")
            
        # Direction detection: majority vote on period 1 goals
        home_team = g_df.iloc[0][col_home]
        g_goals = g_df[g_df[col_type] == 'Goal']
        
        # Calibration: Majority vote of Period 1 goals
        p1_goals = g_goals[g_goals['PERIOD'] == 1]
        signs = []
        for _, fg in p1_goals.iterrows():
            if not pd.isna(fg.get(col_xc)):
                raw = 1 if fg[col_xc] > 0 else -1
                if fg[col_team] != home_team:
                    raw *= -1
                signs.append(raw)
        
        if signs and sum(signs) != 0:
            h_sign = int(np.sign(sum(signs)))
        else:
            # Audit Fallback: First goal in any period (flipped accordingly)
            found_fallback = False
            for _, fg in g_goals.iterrows():
                if not pd.isna(fg.get(col_xc)):
                    raw = 1 if fg[col_xc] > 0 else -1
                    if fg[col_team] != home_team: raw *= -1
                    f_period = int(fg.get('PERIOD', 1))
                    h_sign = raw if (f_period % 2 == 1) else -raw
                    found_fallback = True
                    break
            if not found_fallback:
                h_sign = 1
                games_with_default_sign.append(g_id)
        
        for period, p_df in g_df.groupby('PERIOD'):
            # Skip OT and shootouts
            if period >= 5: continue 
            
            period_sign = h_sign if period % 2 == 1 else -h_sign
            
            p_df = p_df.sort_values('Seconds_Elapsed')
            events = p_df.to_dict('records')
            num_events = len(events)
            
            teams_in_game = p_df[[col_team, col_home, col_away]].melt().value.unique()
            team_codes = [TEAM_MAP.get(str(t).strip()) for t in teams_in_game if t in TEAM_MAP]
            team_codes = [tc for tc in team_codes if tc]
            
            if len(team_codes) < 2: continue
            
            # Metadata for team mapping
            hc_full = str(home_team).strip()
            ac_full = str(p_df.iloc[0][col_away]).strip()
            team_full_to_code = {hc_full: TEAM_MAP.get(hc_full), ac_full: TEAM_MAP.get(ac_full)}

            oz_start: dict[str, any] = {tc: None for tc in team_codes}
            pending_shots: dict[str, list] = {tc: [] for tc in team_codes}
            last_team_xc: dict[str, float] = {tc: None for tc in team_codes}
            last_team_time: dict[str, float] = {tc: None for tc in team_codes}
            team_shot_counts: dict[str, dict] = {tc: {} for tc in team_codes}

            zone_entry_counts = {tc: 0 for tc in team_codes}
            oz_segment_totals = {tc: 0.0 for tc in team_codes}

            def get_interpolated_transition(t_prev, x_prev, t_curr, x_curr, threshold=25):
                if t_curr == t_prev: return t_curr
                v = (x_curr - x_prev) / (t_curr - t_prev)
                if abs(v) < 1e-6: return t_curr
                t_cross = t_prev + (threshold - x_prev) / v
                return max(t_prev, min(t_curr, t_cross))

            def close_segment(team, current_time, current_xc=None):
                if oz_start.get(team) is not None:
                    exit_time = current_time
                    if current_xc is not None and last_team_xc.get(team) is not None:
                        x_prev = last_team_xc[team]
                        t_prev = last_team_time[team]
                        if x_prev > 25 and current_xc <= 25:
                            exit_time = get_interpolated_transition(t_prev, x_prev, current_time, current_xc)
                    
                    seg_duration = exit_time - oz_start[team]
                    seg_duration = max(0, min(300, seg_duration))
                    oz_segment_totals[team] += seg_duration
                    
                    for shot_idx in pending_shots[team]:
                        shot_time = pressure_rows[shot_idx]['time']
                        pressure_rows[shot_idx]['time_to_exit'] = max(0, exit_time - shot_time)
                    
                    oz_start[team] = None
                    pending_shots[team] = []

            for i in range(num_events):
                row = events[i]
                t    = int(row['Seconds_Elapsed'])
                play = str(row[col_type]).strip()
                actor_raw = str(row[col_team]).strip()
                team = TEAM_MAP.get(actor_raw)
                shooter = row['Shooter_Norm']
                
                ev_xc = row.get(col_xc)
                if pd.isna(ev_xc):
                    zone_str = str(row[col_zone])
                    if 'Offensive' in zone_str and team: p_xc_home_rel = 50 
                    elif 'Defensive' in zone_str and team: p_xc_home_rel = -50
                    else: p_xc_home_rel = 0 
                else:
                    p_xc_home_rel = ev_xc * period_sign

                p_zone = 0
                if p_xc_home_rel > 25: p_zone = 1   # Home OZ
                elif p_xc_home_rel < -25: p_zone = -1 # Away OZ

                hc = TEAM_MAP.get(hc_full)
                ac = TEAM_MAP.get(ac_full)
                
                if hc and ac:
                    if p_zone == 1:
                        if oz_start.get(hc) is None:
                            entry_time = t
                            if last_team_xc.get(hc) is not None:
                                x_p = last_team_xc[hc]
                                t_p = last_team_time[hc]
                                if x_p <= 25: entry_time = get_interpolated_transition(t_p, x_p, t, p_xc_home_rel)
                            oz_start[hc] = entry_time
                            zone_entry_counts[hc] += 1
                        close_segment(ac, t, -p_xc_home_rel) 
                        if team == ac and play in ['Takeaway', 'Giveaway']: close_segment(hc, t, p_xc_home_rel)
                    elif p_zone == -1:
                        if oz_start.get(ac) is None:
                            entry_time = t
                            if last_team_xc.get(ac) is not None:
                                x_p = last_team_xc[ac]
                                t_p = last_team_time[ac]
                                if x_p <= 25: entry_time = get_interpolated_transition(t_p, x_p, t, -p_xc_home_rel)
                            oz_start[ac] = entry_time
                            zone_entry_counts[ac] += 1
                        close_segment(hc, t, p_xc_home_rel)
                        if team == hc and play in ['Takeaway', 'Giveaway']: close_segment(ac, t, -p_xc_home_rel)
                    else:
                        close_segment(hc, t, p_xc_home_rel)
                        close_segment(ac, t, -p_xc_home_rel)

                if hc:
                    last_team_xc[hc] = p_xc_home_rel
                    last_team_time[hc] = t
                if ac:
                    last_team_xc[ac] = -p_xc_home_rel
                    last_team_time[ac] = t

                if play in SHOT_TYPES and team and shooter != "UNKNOWN":
                    start_t = oz_start.get(team)
                    p_val = 0
                    imputed = False
                    
                    if start_t is not None:
                        p_val = max(0, min(300, t - start_t))
                    else:
                        if last_team_xc.get(team) is not None:
                            x_p = last_team_xc[team]
                            t_p = last_team_time[team]
                            e_time = get_interpolated_transition(t_p, x_p, t, 25.1)
                            p_val = max(0, t - e_time)
                            imputed = True
                    
                    team_shot_counts[team][shooter] = team_shot_counts[team].get(shooter, 0) + 1
                    jersey = PLAYER_JERSEY_MAP.get(team, {}).get(shooter, 0)

                    entry = {
                        'BDB_Game_Id': g_id, 'Period': period, 'Shooter_Norm': shooter,
                        'Shooter_Number': jersey, 'Shot_Order_Identity': team_shot_counts[team][shooter],
                        'Shooter_Team': team,
                        'time': t, 'oz_pressure': p_val, 'oz_pressure_imputed': imputed,
                        'time_to_exit': 0.0
                    }
                    for rc in roster_cols: entry[rc] = row.get(rc, "UNKNOWN")
                    pressure_rows.append(entry)
                    if not imputed or p_val > 0: pending_shots[team].append(len(pressure_rows) - 1)

            for tc in team_codes:
                close_segment(tc, t)
                zone_entry_records.append({'BDB_Game_Id': g_id, 'Period': period, 'team_name': tc, 'zone_entries': zone_entry_counts[tc], 'total_oz_seconds': oz_segment_totals[tc]})

    print(f"  Audit: Games requiring fallback h_sign: {len(games_with_default_sign)}")

    pressure_df = pd.DataFrame(pressure_rows)
    if pressure_df.empty:
        raise ValueError("No pressure rows generated.")

    # Step 3: Creating Enriched Lookup (MoneyPuck Anchor)
    print("\nStep 3: Creating Enriched Lookup (MoneyPuck Anchor)...")
    
    # Layer 1: Strict Match
    lookup = mp_5v5.merge(
        pressure_df,
        left_on=['BDB_Game_Id', 'period', 'Shooter_Norm', 'Shot_Order_Identity'],
        right_on=['BDB_Game_Id', 'Period', 'Shooter_Norm', 'Shot_Order_Identity'],
        how='left'
    )
    
    # Layer 1.5: patch unmatched shots by time proximity + player name
    unmatched_mask = lookup['oz_pressure'].isna()
    l15_count = 0
    if unmatched_mask.any():
        print(f"  Layer 1.5: Patching Drift for {unmatched_mask.sum():,} shots...")
        # Fix: Ensure types match for set check
        claimed_bdb_keys = set()
        l1_matched_rows = lookup[lookup['oz_pressure'].notna()]
        for _, r in l1_matched_rows.iterrows():
            claimed_bdb_keys.add((str(r['BDB_Game_Id']), int(r['Period']), str(r['Shooter_Norm']), int(r['Shot_Order_Identity'])))
        
        for idx in lookup[unmatched_mask].index:
            row = lookup.loc[idx]
            g_id = str(row['BDB_Game_Id']); per = int(row['period']); name = str(row['Shooter_Norm']); t_mp = row['time_x']
            
            # Find candidate in this game/period for this player
            cands = pressure_df[(pressure_df['BDB_Game_Id'] == g_id) & (pressure_df['Period'] == per) & (pressure_df['Shooter_Norm'] == name)].copy()
            cands = cands[~cands.apply(lambda r: (g_id, per, r['Shooter_Norm'], r['Shot_Order_Identity']) in claimed_bdb_keys, axis=1)]
            
            if not cands.empty:
                cands['t_diff'] = abs(cands['time'] - t_mp)
                best = cands.sort_values('t_diff').iloc[0]
                if best['t_diff'] <= 4:
                    for col in ['oz_pressure', 'time_to_exit', 'oz_pressure_imputed', 'Shooter_Team'] + roster_cols:
                        lookup.at[idx, col] = best[col]
                    lookup.at[idx, 'time_y'] = best['time']
                    claimed_bdb_keys.add((g_id, per, name, best['Shot_Order_Identity']))
                    l15_count += 1
    
    print(f"  Layer 1.5 Matched: {l15_count:,} shots.")

    l1_matched = lookup['oz_pressure'].notna().sum()
    print(f"  Final Layer 1/1.5 Match: {l1_matched:,} / {len(lookup):,}")

    # Layer 2+: Aggressive 'Vulture' matching (with team constraint)
    unmatched_indices = lookup[lookup['oz_pressure'].isna()].index.tolist()
    if unmatched_indices:
        print(f"  Layer 2+: Starting Aggressive 'Vulture' Matching for {len(unmatched_indices):,} shots...")
        claimed_bdb_keys = set(lookup[lookup['oz_pressure'].notna()][['BDB_Game_Id', 'Period', 'Shooter_Norm', 'Shot_Order_Identity']].itertuples(index=False, name=None))

        for idx in unmatched_indices:
            row = lookup.loc[idx]
            g_id = row['BDB_Game_Id']; per = row['period']; name = row['Shooter_Norm']; num = row['Shooter_Number_x']; t_mp = row['time_x']; team = row['teamCode']

            # Vulture only pulls candidates from the same team
            for layer in [2, 3, 4]:
                cands = pd.DataFrame()
                if layer == 2: # Name + Number
                    cands = pressure_df[(pressure_df['BDB_Game_Id'] == g_id) & (pressure_df['Period'] == per) & (pressure_df['Shooter_Norm'] == name) & (pressure_df['Shooter_Number'] == num)] if num > 0 else pd.DataFrame()
                elif layer == 3: # Number Only
                    cands = pressure_df[(pressure_df['BDB_Game_Id'] == g_id) & (pressure_df['Period'] == per) & (pressure_df['Shooter_Number'] == num)] if num > 0 else pd.DataFrame()
                elif layer == 4: # Proximity Only (Now with Team Constraint)
                    cands = pressure_df[(pressure_df['BDB_Game_Id'] == g_id) & (pressure_df['Period'] == per) & (pressure_df['Shooter_Team'] == team)]
                
                if not cands.empty:
                    cands = cands[~cands.apply(lambda r: (g_id, per, r['Shooter_Norm'], r['Shot_Order_Identity']) in claimed_bdb_keys, axis=1)]
                    if not cands.empty:
                        cands['t_diff'] = abs(cands['time'] - t_mp)
                        best = cands.sort_values('t_diff').iloc[0]
                        if best['t_diff'] <= (60 if layer < 4 else 5):
                            for col in ['oz_pressure', 'time_to_exit', 'oz_pressure_imputed', 'Shooter_Team'] + roster_cols:
                                lookup.at[idx, col] = best[col]
                            lookup.at[idx, 'time_y'] = best['time']
                            claimed_bdb_keys.add((g_id, per, best['Shooter_Norm'], best['Shot_Order_Identity']))
                            break

    final_matched_count = lookup['oz_pressure'].notna().sum()
    print(f"  Final Clean Organic Match: {final_matched_count:,} / {len(lookup):,}")

    # Desynchronized names report
    print("\n" + "="*40)
    print("Desynchronized names report (BDB vs MoneyPuck)")
    print("="*40)
    
    mp_names = set(mp_5v5['Shooter_Norm'].unique())
    bdb_names = set(pressure_df['Shooter_Norm'].unique())
    
    missing_in_bdb = mp_names - bdb_names
    missing_in_mp  = bdb_names - mp_names
    
    print(f"\nNames in MoneyPuck NOT found in BigBall ({len(missing_in_bdb)}):")
    for name in sorted(list(missing_in_bdb))[:50]:
        print(f"  - {name}")
    if len(missing_in_bdb) > 50: print(f"  ... and {len(missing_in_bdb)-50} more")

    print(f"\nNames in BigBall NOT found in MoneyPuck ({len(missing_in_mp)}):")
    for name in sorted(list(missing_in_mp))[:50]:
        print(f"  - {name}")
    if len(missing_in_mp) > 50: print(f"  ... and {len(missing_in_mp)-50} more")
    print("="*40 + "\n")

    # Drop unmatched instead of imputing (Absolute user request)
    unmatched_count = lookup['oz_pressure'].isna().sum()
    if unmatched_count > 0:
        print(f"  Dropping {unmatched_count:,} unmatched shots (synthetic imputation disabled).")
        lookup = lookup.dropna(subset=['oz_pressure'])

    # Rename for compatibility with analysis script
    lookup = lookup.rename(columns={'time_x': 'time'})

    # Final cleanup: ensure all Period values are integers
    lookup['Period'] = lookup['Period'].fillna(lookup['period']).astype(int)
    
    # Filter to final output columns
    final_cols = ['BDB_Game_Id', 'Period', 'Shooter_Norm', 'Shot_Order_Identity', 
                  'time', 'oz_pressure', 'time_to_exit', 'oz_pressure_imputed', 'shotRush'] + roster_cols
    
    lookup[final_cols].to_csv(OUT_PATH, index=False)
    
    # Save Fix 3 & 4A Output CSV
    entry_out_path = os.path.join(CURRENT_DIR, 'pbp_zone_entry_counts.csv')
    pd.DataFrame(zone_entry_records).to_csv(entry_out_path, index=False)
    print(f"Zone entry counts saved: {entry_out_path}")

    print(f"\nEnriched lookup saved: {OUT_PATH}")
    print(f"  Final Valid Volume: {len(lookup):,} shots")
    print(f"  Success Rate: {len(lookup) / len(mp_5v5) * 100:.2f}%")
    print("=" * 65)


if __name__ == "__main__":
    build_identity_lookup()
