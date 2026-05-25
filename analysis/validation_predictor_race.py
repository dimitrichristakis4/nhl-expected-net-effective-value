import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')

# Config and paths
DATA_DIR = "/Users/dimitrichristakis/Desktop/OUSAC ABstract 2/Currently Relavent"
SHOTS_FILE = os.path.join(DATA_DIR, "shots_2022.csv")
SPLITS_FILE = os.path.join(DATA_DIR, "Team_Zonal_Outcome_Equilibrium_Splits.csv")
GAMES_2_FILE = os.path.join(DATA_DIR, "games-2.csv")
OUTPUT_DIR = os.path.join(DATA_DIR, "Outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Constants from Pipeline
ENTRY_VALUE    = 0.052072
EXIT_PENALTY   = 0.026036
FACEOFF_EQUITY = 0.023070

TEAM_CODE_MAP = {
    'ANA':'Anaheim Ducks', 'ARI':'Arizona Coyotes', 'BOS':'Boston Bruins',
    'BUF':'Buffalo Sabres', 'CGY':'Calgary Flames', 'CAR':'Carolina Hurricanes',
    'CHI':'Chicago Blackhawks', 'COL':'Colorado Avalanche', 'CBJ':'Columbus Blue Jackets',
    'DAL':'Dallas Stars', 'DET':'Detroit Red Wings', 'EDM':'Edmonton Oilers',
    'FLA':'Florida Panthers', 'LAK':'Los Angeles Kings', 'MIN':'Minnesota Wild',
    'MTL':'Montreal Canadiens', 'NSH':'Nashville Predators', 'NJD':'New Jersey Devils',
    'NYI':'New York Islanders', 'NYR':'New York Rangers', 'OTT':'Ottawa Senators',
    'PHI':'Philadelphia Flyers', 'PIT':'Pittsburgh Penguins', 'SJS':'San Jose Sharks',
    'SEA':'Seattle Kraken', 'STL':'St. Louis Blues', 'TBL':'Tampa Bay Lightning',
    'TOR':'Toronto Maple Leafs', 'VAN':'Vancouver Canucks', 'VGK':'Vegas Golden Knights',
    'WSH':'Washington Capitals', 'WPG':'Winnipeg Jets'
}

def get_dist_bin_str(dist):
    if dist < 6: return '0-6 ft'
    if dist < 12: return '6-12 ft'
    if dist < 18: return '12-18 ft'
    if dist < 24: return '18-24 ft'
    if dist < 30: return '24-30 ft'
    if dist < 36: return '30-36 ft'
    if dist < 42: return '36-42 ft'
    if dist < 48: return '42-48 ft'
    if dist < 54: return '48-54 ft'
    if dist < 60: return '54-60 ft'
    return '60-66 ft'

def run_predictor_race():
    print("Predictor race: xNEV vs xG vs Corsi")
    print("Comparing xNEV vs xG vs Corsi % for Actual 5v5 Goal Prediction")
    
    # 1. Load Shots & Calculate xNEV
    print(" [1/3] Calculating xNEV Totals...")
    shots = pd.read_csv(SHOTS_FILE)
    splits = pd.read_csv(SPLITS_FILE)

    df = shots[
        (shots['isPlayoffGame'] == 0) &
        (shots['shotOnEmptyNet'] == 0) &
        (shots['shotRebound'] == 0) &
        (shots['awaySkatersOnIce'] == 5) &
        (shots['homeSkatersOnIce'] == 5)
    ].copy()

    df['dist_calc'] = df['arenaAdjustedShotDistance'].fillna(df['shotDistance'])
    df = df[df['dist_calc'] <= 66]
    df['dist_bin'] = df['dist_calc'].apply(get_dist_bin_str)
    df['team_name'] = df['teamCode'].map(TEAM_CODE_MAP)

    zonal = splits[splits['half'] == 'Whole'].copy()
    df = df.merge(zonal[['dist_bin', 'team_name', 'reb_rate', 'recov_rate', 'exit_rate', 'freeze_rate', 'reb_yield_constant']], 
                 on=['dist_bin', 'team_name'], how='left')

    for col in ['reb_rate', 'recov_rate', 'exit_rate', 'freeze_rate', 'reb_yield_constant']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df['xnev'] = (df['xGoal'].fillna(0.01) + (df['reb_rate'] * df['reb_yield_constant']) + 
                 (df['recov_rate'] * ENTRY_VALUE) + (df['freeze_rate'] * FACEOFF_EQUITY) - 
                 (df['exit_rate'] * EXIT_PENALTY))

    team_xnev = df.groupby('teamCode')['xnev'].sum().reset_index()
    team_xnev['Team'] = team_xnev['teamCode'].map(TEAM_CODE_MAP)

    # 2. Load Metric Benchmarks (Corsi, xG, GF)
    print(" [2/3] Loading Benchmarks from games-2.csv...")
    try:
        games = pd.read_csv(GAMES_2_FILE)
        games['Team_Norm'] = games['Team'].str.upper().str.strip()
        team_xnev['Team_Norm'] = team_xnev['Team'].str.upper().str.strip()

        # GF = Goals For, xGF = Expected Goals, CF% = Corsi For %
        final = team_xnev.merge(games[['Team_Norm', 'GF', 'xGF', 'CF%']], on='Team_Norm', how='inner')
    except Exception as e:
        print(f" ERROR: {e}")
        return

    # 3. Predictor Race Dashboard
    print(" [3/3] Generating Predictor Comparison Plot...")
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    comparisons = [
        ('xnev', 'xNEV Model', '#00ff00'),
        ('xGF', 'Public xG', '#ff7f0e'),
        ('CF%', 'Corsi For %', '#1f77b4')
    ]
    
    for i, (col, label, color) in enumerate(comparisons):
        ax = axes[i]
        x_data = final[col]
        y_data = final['GF']
        
        ax.scatter(x_data, y_data, color=color, s=120, alpha=0.7, edgecolors='white', linewidth=1)
        
        # Regression
        z = np.polyfit(x_data, y_data, 1)
        p = np.poly1d(z)
        r_sq = np.corrcoef(x_data, y_data)[0,1]**2
        ax.plot(x_data, p(x_data), "r--", alpha=0.6)
        
        print(f" REPORT: {label} R² = {r_sq:.4f}")
        
        ax.set_title(f"{label} vs Goals\n(R² = {r_sq:.3f})", fontsize=18, fontweight='bold', color=color)
        ax.set_xlabel(label, fontsize=14)
        ax.set_ylabel("Actual 5v5 Goals", fontsize=14)
        ax.grid(alpha=0.15)
        
        # Team Labels (Only for top/bottom to avoid clutter)
        for _, row in final.iterrows():
            ax.annotate(row['teamCode'], (row[col], row['GF']), xytext=(4, 4), 
                        textcoords='offset points', fontsize=9, alpha=0.8)

    fig.suptitle("The Predictor Race: Which Metric Dominates Goal Prediction?", 
                 fontsize=28, fontweight='bold', y=1.05, color='#00ff00')
    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "approach26_predictor_race.png")
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f"\n SUCCESS! Predictor Race saved to: {out_path}")

if __name__ == "__main__":
    run_predictor_race()
