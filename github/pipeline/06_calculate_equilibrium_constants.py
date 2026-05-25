import pandas as pd
import numpy as np
import os
import re
import unicodedata

# Paths
CURRENT_DIR = "/Users/dimitrichristakis/Desktop/OUSAC ABstract 2/Currently Relavent"
SCRIPT_DIR  = os.path.dirname(CURRENT_DIR) # Source data in current folder
BDB_LOGS_PATH = os.path.join(CURRENT_DIR, '2022_2023_NHL_PbP_Logs.csv')
SHOTS_PATH    = os.path.join(CURRENT_DIR, 'shots_2022.csv')
LOOKUP_PATH   = os.path.join(CURRENT_DIR, 'pbp_identity_pressure_lookup.csv')
BRIDGE_PATH   = os.path.join(SCRIPT_DIR, 'game_id_bridge.csv')
BASELINES_PATH = os.path.join(CURRENT_DIR, 'xNEV_Shot_Baselines.csv')

# Helpers
def clean_id(x):
    if pd.isna(x): return "UNKNOWN"
    s = str(x).strip()
    if s.endswith('.0'): s = s[:-2]
    return s

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

BIN_EDGES  = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]
def get_dist_bin(d):
    if pd.isna(d): return None
    for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        if lo <= d < hi:
            return f"{lo}-{hi} ft"
    return "72-99 ft"

def time_to_sec(t):
    if pd.isna(t): return None
    if isinstance(t, str):
        parts = t.strip().split(':')
        try:
            # BDB often uses 00:MM:SS or MM:SS
            if len(parts) == 3: 
                # If the first part is '00', ignore it to stay period-relative
                if parts[0] == '00': return int(parts[1]) * 60 + int(parts[2])
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            if len(parts) == 2: return int(parts[0]) * 60 + int(parts[1])
        except ValueError: return None
    return None

def clean_id(x):
    if pd.isna(x): return "UNKNOWN"
    s = str(x).strip()
    if s.endswith('.0'): s = s[:-2]
    return s

def calculate_equilibrium_constants():
    print("=" * 65)
    print("EQUILIBRIUM DISCOVERY: IDENTITY-LINKED ROBUST AUDIT")
    print("=" * 65)

    if not all(os.path.exists(p) for p in [BDB_LOGS_PATH, SHOTS_PATH, BRIDGE_PATH, LOOKUP_PATH, BASELINES_PATH]):
        print(f"Error: Required data files not found.")
        return

    # 1. Load and sync identity bridge
    print("Step 1: Synchronizing Identity Bridge...")
    bridge = pd.read_csv(BRIDGE_PATH)
    shots_raw = pd.read_csv(SHOTS_PATH)
    shots_raw = shots_raw.merge(bridge[['game_id', 'BDB_Game_Id']], on='game_id', how='left')
    shots_raw['BDB_Game_Id'] = shots_raw['BDB_Game_Id'].apply(clean_id)
    shots_raw = shots_raw.dropna(subset=['BDB_Game_Id']).copy()
    
    df = shots_raw[
        (shots_raw['isPlayoffGame']    == 0) &
        (shots_raw['awaySkatersOnIce'] == 5) &
        (shots_raw['homeSkatersOnIce'] == 5) &
        (shots_raw['shotOnEmptyNet']   == 0) &
        (shots_raw['awayEmptyNet']     == 0) &
        (shots_raw['homeEmptyNet']     == 0) &
        (shots_raw['shotRebound']      == 0)
    ].copy()
    
    df['Shooter_Norm'] = df['shooterName'].apply(normalize_name)
    df = df.sort_values(['BDB_Game_Id', 'period', 'shotID'])
    df['Shot_Order_Identity'] = (df.groupby(['BDB_Game_Id', 'period', 'Shooter_Norm']).cumcount() + 1)
    
    # Coerce for merge
    df['period'] = df['period'].astype(int)
    df['Shot_Order_Identity'] = df['Shot_Order_Identity'].astype(int)
    
    pbp_lookup = pd.read_csv(LOOKUP_PATH)
    pbp_lookup['BDB_Game_Id'] = pbp_lookup['BDB_Game_Id'].apply(clean_id)
    pbp_lookup['Shooter_Norm'] = pbp_lookup['Shooter_Norm'].apply(normalize_name)
    pbp_lookup['Period'] = pbp_lookup['Period'].astype(float).astype(int)
    pbp_lookup['Shot_Order_Identity'] = pbp_lookup['Shot_Order_Identity'].astype(float).astype(int)
    
    merged = df.merge(
        pbp_lookup[['BDB_Game_Id', 'Period', 'Shooter_Norm', 'Shot_Order_Identity', 'time']], 
        left_on  = ['BDB_Game_Id', 'period',  'Shooter_Norm', 'Shot_Order_Identity'],
        right_on = ['BDB_Game_Id', 'Period',  'Shooter_Norm', 'Shot_Order_Identity'],
        how='left',
        suffixes=('_mp', '_bdb')
    )
    
    total_shots = len(merged)
    matched_mask = merged['time_bdb'].notna()
    print(f"  Identity Bridge Coverage: {matched_mask.sum():,} / {total_shots:,} ({matched_mask.sum()/total_shots:.1%})")
    merged = merged[matched_mask].copy()

    # 2. Load BDB play-by-play
    print("\nStep 2: Loading High-Fidelity BDB PBP...")
    pbp = pd.read_csv(BDB_LOGS_PATH, low_memory=False)
    clean_cols = [" ".join(str(c).replace('\n', ' ').strip().split()) for c in pbp.columns]
    pbp.columns = clean_cols

    def find_col(kw):
        matches = [c for c in clean_cols if kw.upper() in c.upper()]
        return matches[0] if matches else None

    col_game    = find_col('BIGDATABALL GAME ID')
    col_time    = find_col('ELAPSED TIME')
    col_type    = find_col('PLAY TYPE')
    col_team    = find_col('TEAM MAKING THE PLAY')
    col_zone    = find_col('PLAY ZONE')
    col_xc      = find_col('X COORDINATE')
    col_desc    = find_col('PLAY DESCRIPTION')
    col_p1      = find_col('PLAYER IN THE EVENT #1')
    col_id      = find_col('PLAY ID')

    pbp['BDB_Game_Id'] = pbp[col_game].apply(clean_id)
    pbp['Seconds_Elapsed'] = pbp[col_time].apply(time_to_sec)
    pbp = pbp.dropna(subset=['Seconds_Elapsed']).copy()
    pbp['Seconds_Elapsed'] = pbp['Seconds_Elapsed'].astype(int)
    pbp['xC'] = pd.to_numeric(pbp[col_xc], errors='coerce')
    pbp['Ev_Team'] = pbp[col_team].astype(str).str.strip()
    pbp['Event'] = pbp[col_type].astype(str).str.strip()
    print("\n--- BDB Event Value Counts ---")
    print(pbp['Event'].value_counts().head(30))
    print("-----------------------------\n")
    pbp['PERIOD'] = pbp['PERIOD'].astype(int)
    pbp['Shooter_Norm'] = pbp[col_p1].apply(normalize_name)
    pbp['is_5v5'] = pbp['STRENGTH'].str.contains('5-on-5', na=False) if 'STRENGTH' in pbp.columns else True

    # Ordinal ranking for BDB shots (Filter shots to 5v5 for denominator)
    pbp_shots_only = pbp[(pbp['Event'].str.contains('Shot|Goal|Miss', na=False, case=False)) & (pbp['is_5v5'])].copy()
    pbp_shots_only = pbp_shots_only.sort_values(['BDB_Game_Id', 'PERIOD', 'Seconds_Elapsed', col_id])
    pbp_shots_only['Shot_Order_Identity'] = pbp_shots_only.groupby(['BDB_Game_Id', 'PERIOD', 'Shooter_Norm']).cumcount() + 1
    
    pbp = pbp.merge(pbp_shots_only[['BDB_Game_Id', 'PERIOD', col_id, 'Shot_Order_Identity']], 
                   on=['BDB_Game_Id', 'PERIOD', col_id], how='left')
    
    print(f"  BDB PBP Loaded: {len(pbp):,} rows.")

    # 2.5 Link BDB native teams and coords via identity-ordinal join
    print("Step 2.5: Linking BDB Native Teams & Coords (Identity-Ordinal)...")
    
    shot_ref = pbp_shots_only[['BDB_Game_Id', 'PERIOD', 'Shooter_Norm', 'Shot_Order_Identity', 'Ev_Team', 'xC', col_id]].rename(
        columns={'xC': 'xC_bdb', 'PERIOD': 'period', 'Ev_Team': 'Ev_Team_bdb', col_id: 'bdb_play_id'}
    )
    
    # Robust Identity Join (Matches 93% goal regardless of time drift)
    merged = merged.merge(shot_ref, on=['BDB_Game_Id', 'period', 'Shooter_Norm', 'Shot_Order_Identity'], how='left')
    
    print(f"  Shots with BDB Identity Link: {merged['Ev_Team_bdb'].notna().sum():,} / {len(merged):,}")

    # 3. Link xG values to PBP timeline via play ID
    print("Step 3: Linking xG to PBP Timeline...")
    shot_xg_ref = merged.dropna(subset=['bdb_play_id']).groupby(['BDB_Game_Id', 'period', 'bdb_play_id'])['xGoal'].max().reset_index()
    shot_xg_ref = shot_xg_ref.rename(columns={'period': 'PERIOD', 'bdb_play_id': col_id, 'xGoal': 'xG_final'})

    pbp = pbp.merge(
        shot_xg_ref[['BDB_Game_Id', 'PERIOD', col_id, 'xG_final']],
        on=['BDB_Game_Id', 'PERIOD', col_id],
        how='left'
    )
    pbp['xGoal_linked'] = pbp['xG_final'].fillna(0)
    print(f"  Linked xG Sum: {pbp['xGoal_linked'].sum():.4f}")

    # 4. League-wide xG constants
    print("Step 4: Calculating League-Wide xG Constants...")
    
    # Entry Value
    pbp['in_ozone'] = pbp[col_zone].str.contains('Offensive', na=False)
    controlled_events = ['Faceoff', 'Shot On Goal', 'Missed Shot', 'Goal Scored', 'Giveaway', 'Takeaway']
    pbp['is_controlled'] = pbp['Event'].isin(controlled_events)
    
    control_pbp = pbp[pbp['is_controlled']].copy()
    control_pbp['poss_change'] = (control_pbp['Ev_Team'] != control_pbp['Ev_Team'].shift(1)) | \
                                 (control_pbp['BDB_Game_Id'] != control_pbp['BDB_Game_Id'].shift(1))
    control_pbp['poss_id'] = control_pbp['poss_change'].cumsum()
    
    pbp = pbp.merge(control_pbp[['BDB_Game_Id', 'PERIOD', 'Seconds_Elapsed', 'Event', 'poss_id']], 
                    on=['BDB_Game_Id', 'PERIOD', 'Seconds_Elapsed', 'Event'], how='left')
    pbp['poss_id'] = pbp['poss_id'].ffill()
    
    # Entry value constant
    global_entry_value = 0.08774750833975138
    print(f"  Using ENTRY_VALUE Constant: {global_entry_value:.6f}")

    # Faceoff equity constant
    global_fo_equity = 0.023070373332688025
    print(f"  Using FO_EQUITY Constant: {global_fo_equity:.6f}")

    # Exit penalty = half the entry value
    global_exit_penalty = global_entry_value * 0.5
    print(f"  Using EXIT_PENALTY Constant: {global_exit_penalty:.6f}")

    # Load rebound yield baselines (team-specific)
    print("Step 4.5: Loading Team-Specific Rebound Yields (Averaging to 6ft bins)...")
    baselines = pd.read_csv(BASELINES_PATH)
    
    team_code_map = {
        'ANA': 'Anaheim Ducks', 'ARI': 'Arizona Coyotes', 'BOS': 'Boston Bruins', 'BUF': 'Buffalo Sabres',
        'CGY': 'Calgary Flames', 'CAR': 'Carolina Hurricanes', 'CHI': 'Chicago Blackhawks', 'COL': 'Colorado Avalanche',
        'CBJ': 'Columbus Blue Jackets', 'DAL': 'Dallas Stars', 'DET': 'Detroit Red Wings', 'EDM': 'Edmonton Oilers',
        'FLA': 'Florida Panthers', 'LAK': 'Los Angeles Kings', 'MIN': 'Minnesota Wild', 'MTL': 'Montreal Canadiens',
        'NSH': 'Nashville Predators', 'NJD': 'New Jersey Devils', 'NYI': 'New York Islanders', 'NYR': 'New York Rangers',
        'OTT': 'Ottawa Senators', 'PHI': 'Philadelphia Flyers', 'PIT': 'Pittsburgh Penguins', 'SJS': 'San Jose Sharks',
        'SEA': 'Seattle Kraken', 'STL': 'St. Louis Blues', 'TBL': 'Tampa Bay Lightning', 'TOR': 'Toronto Maple Leafs',
        'VAN': 'Vancouver Canucks', 'VGK': 'Vegas Golden Knights', 'WSH': 'Washington Capitals', 'WPG': 'Winnipeg Jets',
        'LEAGUE': 'LEAGUE'
    }
    
    # Build Lookup: (team_name, bin_6ft) -> secondary_xg_yield
    team_bin_yields = {}
    
    for team_code, group in baselines.groupby('teamCode'):
        team_name = team_code_map.get(team_code, "UNKNOWN")
        if team_name == "UNKNOWN": continue
        
        group_lookup = group.set_index('dist_bin')['secondary_xg_yield'].to_dict()
        
        # 00-06
        team_bin_yields[(team_name, '00-06')] = group_lookup.get('0-6 ft', 0.24)
        
        # 06-72
    for i in range(6, 72, 6):
        lo, mid, hi = i, i+3, i+6
        bin_6 = f"{lo}-{hi} ft"
        y1 = group_lookup.get(f"{lo}-{mid} ft", 0.09)
        y2 = group_lookup.get(f"{mid}-{hi} ft", 0.09)
        team_bin_yields[(team_name, bin_6)] = (y1 + y2) / 2.0

    default_reb_yield = 0.09

    # 5. BDB-native tactical outcome audit
    print("Step 5: Auditing BDB-Native Tactical Outcomes...")
    
    # A. Define Primary Shots strictly in BDB (Denominator)
    reg_season_gids = set(merged['BDB_Game_Id'].unique())
    col_aw_goalie = find_col('AWAY TEAM\'S GOALIE')
    col_hm_goalie = find_col('HOME TEAM\'S GOALIE')
    col_date = find_col('DATE')
    
    # Temporal Tagging: 1st Half vs 2nd Half
    game_dates = pbp[pbp['BDB_Game_Id'].isin(reg_season_gids)].groupby('BDB_Game_Id')[col_date].first().sort_values()
    game_ordinals = {gid: i+1 for i, gid in enumerate(game_dates.index)}
    # Assign half: 1st half is first 50% of games, 2nd half is second 50%
    # (Since games are not evenly distributed globally, we'll use a globally median game index or just simple half-split)
    midpoint = len(game_dates) // 2
    game_halves = {gid: ('1st' if i < midpoint else '2nd') for i, gid in enumerate(game_dates.index)}

    # Filter for all shots/misses, restricted to 5v5 Regular Season
    bdb_all_shots = pbp[
        (pbp['Event'].isin(['Shot', 'Missed Shot', 'Goal'])) & 
        (pbp['STRENGTH'].str.strip() == '5-on-5') &
        (pbp['BDB_Game_Id'].isin(reg_season_gids)) &
        (pbp[col_aw_goalie].notna() & (pbp[col_aw_goalie] != '-')) &
        (pbp[col_hm_goalie].notna() & (pbp[col_hm_goalie] != '-'))
    ].copy()
    
    bdb_all_shots['half'] = bdb_all_shots['BDB_Game_Id'].map(game_halves)
    
    print(f"  DIAGNOSTIC: Reg Season Games: {len(reg_season_gids)}")
    print(f"  DIAGNOSTIC: BDB 5v5 Unblocked Shots (Reg Season): {len(bdb_all_shots)}")
    
    bdb_all_shots = bdb_all_shots.sort_values(['BDB_Game_Id', 'PERIOD', col_id])
    bdb_all_shots['prev_sec'] = bdb_all_shots.groupby(['BDB_Game_Id', 'PERIOD'])['Seconds_Elapsed'].shift(1)
    bdb_all_shots['gap'] = bdb_all_shots['Seconds_Elapsed'] - bdb_all_shots['prev_sec']
    primary_bdb = bdb_all_shots[bdb_all_shots['gap'].fillna(999) > 3.0].copy()
    
    pbp_by_game = {gid: gdf.sort_values(col_id) for gid, gdf in pbp.groupby('BDB_Game_Id')}
    OUTCOME_DEFINERS = ['Faceoff', 'Goal', 'Penalty', 'Giveaway', 'Takeaway', 'Shot', 'Missed Shot', 'Blocked Shot', 'Hit', 'Goalie Stopped']
    
    def native_audit(shot):
        gid = shot['BDB_Game_Id']
        if gid not in pbp_by_game: return 'skip', None
        game_pbp = pbp_by_game[gid]
        
        s_id = int(shot[col_id])
        s_team = shot['Ev_Team']
        s_per = shot['PERIOD']
        s_xc = shot['xC']
        s_time = shot['Seconds_Elapsed']
        azone_sign = 1 if s_xc > 0 else -1
        
        # If the shot itself is a Goal, it's a Goal outcome immediately
        if shot['Event'] == 'Goal': return 'goal', None

        future = game_pbp[(game_pbp[col_id] > s_id) & (game_pbp['PERIOD'] == s_per)]
        
        for i, event in future.iterrows():
            ev_type = event['Event']
            ev_team = event['Ev_Team']
            ev_xc = event['xC']
            ev_time = event['Seconds_Elapsed']
            ev_id = int(event[col_id])
            
            # REBOUND: Check if very next unblocked shot is within 3s
            if ev_type in ['Shot', 'Missed Shot', 'Goal'] and (ev_time - s_time <= 3.0):
                # It's a rebound. RECORD THE DISTANCE BIN of this rebound shot.
                reb_dist = event[find_col('DISTANCE')]
                reb_bin = get_dist_bin(reb_dist)
                return 'rebound', reb_bin
                
            if (ev_time - s_time > 30): return 'stoppage', None
            if ev_type not in OUTCOME_DEFINERS: continue
            
            # Skip any event with NaN coordinates when deciding zone
            if pd.isna(ev_xc):
                # If it's a terminal event (Faceoff/Goal), we still count it regardless of xC
                if ev_type in ['Faceoff', 'Goalie Stopped', 'Goal']:
                    pass # Proceed to handle below
                else:
                    continue # Skip this Hit/Block/etc and look for next valid coordinate record
            
            is_ozone = (ev_xc * azone_sign > 25) if not pd.isna(ev_xc) else False
            
            # 1. Freeze
            if ev_type in ['Faceoff', 'Goalie Stopped']: return 'freeze', None
            
            # 2. Goal (Secondary goal in same possession)
            if ev_type == 'Goal': return 'goal', None
            
            # 3. physical Disruptors
            if ev_type in ['Hit', 'Blocked Shot']:
                lookahead = future[(future[col_id] > ev_id) & (future['Seconds_Elapsed'] <= ev_time + 5)]
                if lookahead.empty:
                    return ('recovery' if (ev_team != s_team and is_ozone) else 'exit'), None
                    
                for _, next_ev in lookahead.iterrows():
                    n_type = next_ev['Event']
                    if n_type not in OUTCOME_DEFINERS: continue
                    n_team = next_ev['Ev_Team']
                    n_xc = next_ev['xC']
                    n_is_ozone = (n_xc * azone_sign > 25) if not pd.isna(n_xc) else True
                    
                    if n_type in ['Faceoff', 'Goalie Stopped']: return 'freeze', None
                    if n_team == s_team and n_is_ozone: return 'recovery', None
                    return 'exit', None
                return 'recovery', None
                
            # 4. Clean Possession
            if ev_team == s_team and is_ozone: return 'recovery', None
            
            # Everything else (turnover, exit, etc)
            return 'exit', None
            
        return 'stoppage', None

    print(f"  Auditing {len(primary_bdb):,} BDB-Native primary shots (5-bucket)...")
    results = primary_bdb.apply(lambda row: native_audit(row), axis=1)
    primary_bdb['outcome'] = [r[0] for r in results]
    primary_bdb['reb_bin'] = [r[1] for r in results]
    primary_bdb['dist_bin'] = primary_bdb[find_col('DISTANCE')].apply(get_dist_bin)
    
    # DIAGNOSTIC: Save raw outcomes
    diag_file = os.path.join(CURRENT_DIR, 'diag_outcomes.csv')
    primary_bdb[['BDB_Game_Id', 'PERIOD', 'Seconds_Elapsed', col_id, 'outcome', 'reb_bin']].to_csv(diag_file, index=False)
    print(f"  DIAGNOSTIC: Saved raw outcomes to {diag_file}")
    print(f"  Outcome Counts:\n{primary_bdb['outcome'].value_counts()}")
    
    # B. Seasonal Splits Calculation (Whole, 1st, 2nd)
    primary_bdb['team_name'] = primary_bdb[col_team].str.strip()
    
    # Create copies for "Whole" season to group easily
    p_bdb_whole = primary_bdb.copy()
    p_bdb_whole['half'] = 'Whole'
    
    # Combined data for aggregation (Whole + 1st + 2nd)
    p_bdb_combined = pd.concat([primary_bdb, p_bdb_whole], ignore_index=True)
    
    # C. Team Grouping
    team_stats = p_bdb_combined.groupby(['team_name', 'dist_bin', 'half']).agg(
        vol=('outcome', 'size'),
        goal_c=('outcome', lambda x: (x == 'goal').sum()),
        reb_c=('outcome', lambda x: (x == 'rebound').sum()),
        recov_c=('outcome', lambda x: (x == 'recovery').sum()),
        exit_c=('outcome', lambda x: (x == 'exit').sum()),
        freeze_c=('outcome', lambda x: (x == 'freeze').sum()),
        xG_sum=('xGoal_linked', 'sum')
    ).reset_index()

    # D. League Grouping (Treat as its own "Team")
    league_stats = p_bdb_combined.groupby(['dist_bin', 'half']).agg(
        vol=('outcome', 'size'),
        goal_c=('outcome', lambda x: (x == 'goal').sum()),
        reb_c=('outcome', lambda x: (x == 'rebound').sum()),
        recov_c=('outcome', lambda x: (x == 'recovery').sum()),
        exit_c=('outcome', lambda x: (x == 'exit').sum()),
        freeze_c=('outcome', lambda x: (x == 'freeze').sum()),
        xG_sum=('xGoal_linked', 'sum')
    ).reset_index()
    league_stats['team_name'] = 'LEAGUE'
    
    # E. Combine & Calculate Rates
    final_stats = pd.concat([team_stats, league_stats], ignore_index=True)
    
    for c in ['goal', 'reb', 'recov', 'exit', 'freeze']:
        final_stats[f'{c}_rate'] = final_stats[f'{c}_c'] / final_stats['vol'].replace(0, np.nan)
    final_stats['xG_rate'] = final_stats['xG_sum'] / final_stats['vol'].replace(0, np.nan)
    
    # Neighborhood Prior Calculation
    def get_neighborhood(bin_name):
        if not bin_name: return 'NONE'
        try:
            # Handle bin names like '00-06'
            dist = int(bin_name.split('-')[0])
            if dist < 18: return 'SLOT'
            if dist < 36: return 'MID'
            return 'OUT'
        except: return 'OUT'

    p_bdb_combined['neighbor'] = p_bdb_combined['dist_bin'].apply(get_neighborhood)
    
    neighbor_stats = p_bdb_combined.groupby(['team_name', 'half', 'neighbor']).agg(
        vol=('outcome', 'size'),
        goal_c=('outcome', lambda x: (x == 'goal').sum()),
        reb_c=('outcome', lambda x: (x == 'rebound').sum()),
        recov_c=('outcome', lambda x: (x == 'recovery').sum()),
        exit_c=('outcome', lambda x: (x == 'exit').sum()),
        freeze_c=('outcome', lambda x: (x == 'freeze').sum()),
        xG_sum=('xGoal_linked', 'sum')
    ).reset_index()

    for c in ['goal', 'reb', 'recov', 'exit', 'freeze']:
        neighbor_stats[f'{c}_rate'] = neighbor_stats[f'{c}_c'] / neighbor_stats['vol'].replace(0, np.nan)
    neighbor_stats['xG_rate'] = neighbor_stats['xG_sum'] / neighbor_stats['vol'].replace(0, np.nan)
    
    # Map back to neighborhood lookup
    neighbor_lookup = neighbor_stats.set_index(['team_name', 'half', 'neighbor'])
    league_lookup = neighbor_stats[neighbor_stats['team_name'] == 'LEAGUE'].set_index(['half', 'neighbor'])

    # F. Bayesian Blending (Linear Ramp to 30)
    final_stats['is_imputed'] = 0
    
    def apply_blend(row):
        team = row['team_name']
        half = row['half']
        bin_x = row['dist_bin']
        neigh = get_neighborhood(bin_x)
        vol = row['vol']
        
        # LEAGUE row is and always was the ultimate baseline
        if team == 'LEAGUE':
            for c in ['goal', 'reb', 'recov', 'exit', 'freeze']:
                row[f'{c}_rate'] = row[f'{c}_c'] / max(vol, 1)
            return row

        # Weight local data (saturated at 30 shots)
        w_local = min(vol, 30) / 30.0
        
        # Get Prior (Neighborhood Average)
        try:
            prior_row = neighbor_lookup.loc[(team, half, neigh)]
            if pd.isna(prior_row['goal_rate']): # If team has zero shots in the whole neighborhood
                prior_row = league_lookup.loc[(half, neigh)]
        except:
            prior_row = league_lookup.loc[(half, neigh)]

        # Apply Blend across all outcome rates
        for c in ['goal', 'reb', 'recov', 'exit', 'freeze']:
            local_rate = row[f'{c}_c'] / max(vol, 1)
            neighbor_rate = prior_row[f'{c}_rate']
            row[f'{c}_rate'] = (local_rate * w_local) + (neighbor_rate * (1.0 - w_local))

        # Blend xG_rate too
        local_xg = row['xG_sum'] / max(vol, 1)
        neighbor_xg = prior_row['xG_rate']
        row['xG_rate'] = (local_xg * w_local) + (neighbor_xg * (1.0 - w_local))

        # Flag as imputed if local volume is under our identity threshold (15)
        # Note: We must return the row to ensure values are captured in the dataframe
        if vol < 15:
            row['is_imputed'] = 1
        else:
            row['is_imputed'] = 0
            
        return row

    final_stats = final_stats.apply(apply_blend, axis=1)
        
    # F. Rebound Scatter Matrix (League-Wide, using Whole season as baseline for stability)
    rebounds_only = p_bdb_combined[(p_bdb_combined['half'] == 'Whole') & (p_bdb_combined['outcome'] == 'rebound')].dropna(subset=['reb_bin'])
    scatter_matrix = rebounds_only.groupby(['dist_bin', 'reb_bin']).size().reset_index(name='count')
    scatter_totals = rebounds_only.groupby(['dist_bin']).size().reset_index(name='total')
    scatter_matrix = scatter_matrix.merge(scatter_totals, on=['dist_bin'])
    scatter_matrix['scatter_prob'] = scatter_matrix['count'] / scatter_matrix['total']
    
    # G. Scaling the Rebound Scatter (Applying to Team-Bin-Half)
    final_scatter_rows = []
    for _, row in final_stats.iterrows():
        team = row['team_name']
        bin_x = row['dist_bin']
        half = row['half']
        team_reb_rate = row['reb_rate']
        
        # Get league scatter distribution for this bin
        distrib = scatter_matrix[scatter_matrix['dist_bin'] == bin_x]
        for _, s_row in distrib.iterrows():
            bin_y = s_row['reb_bin']
            scaled_prob = team_reb_rate * s_row['scatter_prob']
            final_scatter_rows.append({
                'team_name': team,
                'shooter_bin': bin_x,
                'rebound_bin': bin_y,
                'half': half,
                'scaled_rebound_prob': scaled_prob,
                'is_imputed': row['is_imputed']
            })
            
    final_stats['Entry_Value_Constant'] = global_entry_value
    final_stats['FO_Equity_Constant'] = global_fo_equity
    final_stats['Exit_Penalty_Constant'] = global_exit_penalty
    
    # Map Spatial Rebound Yields (Team Specific)
    final_stats['reb_yield_constant'] = final_stats.apply(
        lambda r: team_bin_yields.get((r['team_name'], r['dist_bin']),
                  team_bin_yields.get(('LEAGUE', r['dist_bin']), default_reb_yield)),
        axis=1
    )
    
    # Save Main Results
    out_csv = os.path.join(CURRENT_DIR, 'Team_Zonal_Outcome_Equilibrium_Splits.csv')
    final_stats.to_csv(out_csv, index=False)
    
    # Save Scatter Results
    scatter_csv = os.path.join(CURRENT_DIR, 'Team_Zonal_Rebound_Scatter_Scaled.csv')
    pd.DataFrame(final_scatter_rows).to_csv(scatter_csv, index=False)
    
    print(f"  Main results saved: {out_csv}")
    print(f"  Scaled scatter saved: {scatter_csv}")

    print("\nEquilibrium discovery complete.")

if __name__ == "__main__":
    calculate_equilibrium_constants()
