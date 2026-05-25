import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

SCRIPT_DIR = '/Users/dimitrichristakis/Desktop/OUSAC ABstract 2/'
XLSX_PATH = os.path.join(SCRIPT_DIR, '2022_2023_NHL_PbP_Logs.xlsx')
OUT_CSV = os.path.join(SCRIPT_DIR, 'pbp_2022_full.csv')

TEAM_MAP = {
    'Anaheim Ducks': 'ANA', 'Arizona Coyotes': 'ARI', 'Boston Bruins': 'BOS', 'Buffalo Sabres': 'BUF',
    'Calgary Flames': 'CGY', 'Carolina Hurricanes': 'CAR', 'Chicago Blackhawks': 'CHI', 'Colorado Avalanche': 'COL',
    'Columbus Blue Jackets': 'CBJ', 'Dallas Stars': 'DAL', 'Detroit Red Wings': 'DET', 'Edmonton Oilers': 'EDM',
    'Florida Panthers': 'FLA', 'Los Angeles Kings': 'LAK', 'Minnesota Wild': 'MIN', 'Montreal Canadiens': 'MTL',
    'Nashville Predators': 'NSH', 'New Jersey Devils': 'NJD', 'New York Islanders': 'NYI', 'New York Rangers': 'NYR',
    'Ottawa Senators': 'OTT', 'Philadelphia Flyers': 'PHI', 'Pittsburgh Penguins': 'PIT', 'San Jose Sharks': 'SJS',
    'Seattle Kraken': 'SEA', 'St. Louis Blues': 'STL', 'Tampa Bay Lightning': 'TBL', 'Toronto Maple Leafs': 'TOR',
    'Vancouver Canucks': 'VAN', 'Vegas Golden Knights': 'VGK', 'Washington Capitals': 'WSH', 'Winnipeg Jets': 'WPG'
}

def process():
    print("Loading BigDataBall Excel file (this may take a minute)...")
    df = pd.read_excel(XLSX_PATH, header=1)

    # Clean Column Names
    df.columns = [" ".join(str(c).split()) for c in df.columns]

    col_mapping = {
        'Date': 'DATE', 'Home': 'HOME TEAM', 'Away': 'AWAY TEAM', 'Period': 'PERIOD',
        'Time': 'ELAPSED TIME', 'Strength': 'STRENGTH', 'PlayType': 'PLAY TYPE',
        'GameId': 'BIGDATABALL GAME ID', 'Zone': 'PLAY ZONE Defensive Offensive Neutral',
        'ShooterName': 'PLAYER IN THE EVENT #1 Faceoff Goal Scored Shot On Goal Missed Shot Blocker Penalty On Giveaway Takeaway Hitter -',
        'ShootingTeamRaw': 'TEAM MAKING THE PLAY Faceoff Won Goal Scored Shot On Goal Missed Shot Get Blocked Penalty Giveaway Takeaway Hitter -'
    }

    print("Pre-processing events...")
    df['Date_Str'] = pd.to_datetime(df[col_mapping['Date']]).dt.strftime('%Y-%m-%d')
    df['Home_Abbr'] = df[col_mapping['Home']].map(TEAM_MAP)
    df['Away_Abbr'] = df[col_mapping['Away']].map(TEAM_MAP)
    df['ShootingTeam_Abbr'] = df[col_mapping['ShootingTeamRaw']].map(TEAM_MAP)

    def time_to_sec(t):
        if pd.isna(t): return 0
        if isinstance(t, str):
            parts = t.split(':')
            if len(parts) == 2: return int(parts[0])*60 + int(parts[1])
            return 0
        try: return t.hour * 3600 + t.minute * 60 + t.second
        except: return 0
    df['Seconds_Elapsed'] = df[col_mapping['Time']].apply(time_to_sec)

    # Sort for sequence analysis
    df = df.sort_values([col_mapping['GameId'], col_mapping['Period'], 'Seconds_Elapsed', 'PLAY ID'])

    print("Calculating refined oZ Pressure...")
    # Refined oZ pressure: Time since last oZ faceoff, reset if event outside Offensive zone
    oz_pressure_vals = []
    last_game = None
    last_period = None
    ozone_faceoff_time = -1

    play_types = df[col_mapping['PlayType']].values
    zones = df[col_mapping['Zone']].values
    times = df['Seconds_Elapsed'].values
    games = df[col_mapping['GameId']].values
    periods = df[col_mapping['Period']].values

    for i in range(len(df)):
        current_game = games[i]
        current_period = periods[i]

        # Reset on game/period change
        if current_game != last_game or current_period != last_period:
            ozone_faceoff_time = -1
            last_game = current_game
            last_period = current_period

        current_play = play_types[i]
        current_zone = str(zones[i])

        # Start timer on Offensive Zone Faceoff
        if current_play == 'Faceoff' and 'Offensive' in current_zone:
            ozone_faceoff_time = times[i]

        # Reset timer if event is in Neutral or Defensive zone
        elif 'Neutral' in current_zone or 'Defensive' in current_zone:
            ozone_faceoff_time = -1

        # Calculate pressure for shots
        shot_markers = ['Shot', 'Goal', 'Missed Shot', 'Goalie Stopped']
        if current_play in shot_markers:
            if ozone_faceoff_time != -1:
                oz_pressure_vals.append(times[i] - ozone_faceoff_time)
            else:
                oz_pressure_vals.append(0)
        else:
            oz_pressure_vals.append(0)

    df['refined_oz_pressure'] = oz_pressure_vals

    # Filter to Even Strength ONLY (but ALL events for flow context)
    ev_markers = ['5-on-5', '4-on-4', '3-on-3']
    mask = df[col_mapping['Strength']].isin(ev_markers)
    df_flow = df[mask].copy()

    # Identify players
    away_players = [c for c in df.columns if "AWAY TEAM'S PLAYER ON THE ICE" in c]
    home_players = [c for c in df.columns if "HOME TEAM'S PLAYER ON THE ICE" in c]

    final_cols = ['Date_Str', col_mapping['GameId'], col_mapping['Period'], 'Seconds_Elapsed',
                  'Home_Abbr', 'Away_Abbr', 'ShootingTeam_Abbr', col_mapping['ShooterName'], 'refined_oz_pressure'] + away_players + home_players

    print(f"Saving {len(df_flow)} Even Strength events (Flow Context) to {OUT_CSV}...")
    df_flow[final_cols].to_csv(OUT_CSV, index=False)
    print("Success.")

if __name__ == "__main__":
    process()
