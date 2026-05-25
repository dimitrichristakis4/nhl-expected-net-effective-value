import pandas as pd
import os

SCRIPT_DIR = '/Users/dimitrichristakis/Desktop/OUSAC ABstract 2/'
SHOTS_PATH = os.path.join(SCRIPT_DIR, 'shots_2022.csv')
PBP_PATH = os.path.join(SCRIPT_DIR, 'pbp_2022_full.csv')
OUT_MAP = os.path.join(SCRIPT_DIR, 'game_id_bridge.csv')

def create_bridge():
    print("Loading datasets...")
    shots = pd.read_csv(SHOTS_PATH, usecols=['game_id', 'period', 'time', 'homeTeamCode', 'awayTeamCode', 'shooterName'])
    pbp = pd.read_csv(PBP_PATH, usecols=['BIGDATABALL GAME ID', 'PERIOD', 'Seconds_Elapsed', 'Home_Abbr', 'Away_Abbr', 'PLAYER IN THE EVENT #1 Faceoff Goal Scored Shot On Goal Missed Shot Blocker Penalty On Giveaway Takeaway Hitter -'])

    # Standardize names
    shots['shooterName'] = shots['shooterName'].str.upper()
    pbp['Shooter_Name'] = pbp['PLAYER IN THE EVENT #1 Faceoff Goal Scored Shot On Goal Missed Shot Blocker Penalty On Giveaway Takeaway Hitter -'].str.upper()

    print("Creating game signatures...")
    # Signature = (Home, Away, Period, Time, Shooter)
    # Note: Using +/- 1s for time to handle clock drift
    potential_matches = []

    # Take a larger sample to bridge all games
    sample_shots = shots.sample(min(15000, len(shots)), random_state=42)

    print("Matching signatures...")
    # Join on Home, Away, Period, Shooter. Then check Time difference.
    # This avoids massive cartesian joins.
    merged = sample_shots.merge(pbp, left_on=['homeTeamCode', 'awayTeamCode', 'period', 'shooterName'],
                                right_on=['Home_Abbr', 'Away_Abbr', 'PERIOD', 'Shooter_Name'], how='inner')

    # Filter for time within 5 seconds
    merged['time_diff'] = np.abs(merged['time'] - merged['Seconds_Elapsed'])
    confirmed = merged[merged['time_diff'] <= 5].copy()

    print(f"Matched {len(confirmed)} shot samples.")

    # Majority vote per game_id
    bridge = confirmed.groupby('game_id')['BIGDATABALL GAME ID'].agg(lambda x: x.value_counts().index[0]).reset_index()

    print(f"Mapped {len(bridge)} game IDs out of {shots['game_id'].nunique()}.")

    # Save mapping
    bridge.columns = ['game_id', 'BDB_Game_Id']
    bridge.to_csv(OUT_MAP, index=False)
    print(f"Bridge saved to {OUT_MAP}")

if __name__ == "__main__":
    import numpy as np
    create_bridge()
