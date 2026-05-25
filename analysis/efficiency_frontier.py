import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')

CURRENT_DIR = "/Users/dimitrichristakis/Desktop/OUSAC ABstract 2/Currently Relavent"
BDB_BRIDGE_PATH = "/Users/dimitrichristakis/Desktop/OUSAC ABstract 2/game_id_bridge.csv"

# xNEV formula constants
ENTRY_VALUE    = 0.052072
EXIT_PENALTY   = 0.026036
FACEOFF_EQUITY = 0.023070

def get_dist_bin_6ft(d):
    if pd.isna(d): return "72-99 ft"
    if 0 <= d < 6: return "0-6 ft"
    if 6 <= d < 12: return "6-12 ft"
    if 12 <= d < 18: return "12-18 ft"
    if 18 <= d < 24: return "18-24 ft"
    if 24 <= d < 30: return "24-30 ft"
    if 30 <= d < 36: return "30-36 ft"
    if 36 <= d < 42: return "36-42 ft"
    if 42 <= d < 48: return "42-48 ft"
    if 48 <= d < 54: return "48-54 ft"
    if 54 <= d < 60: return "54-60 ft"
    if 60 <= d < 66: return "60-66 ft"
    if 66 <= d < 72: return "66-72 ft"
    return "72-99 ft"

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
TEAM_NAME_TO_CODE = {v.upper().replace(' ', ''): k for k, v in TEAM_CODE_MAP.items()}

def run_league_efficacy_scatter():
    print("League-wide efficacy scatter")

    print("Step 1: Loading Databases...")
    shots_raw = pd.read_csv(os.path.join(CURRENT_DIR, 'shots_2022.csv'))
    splits = pd.read_csv(os.path.join(CURRENT_DIR, 'Team_Zonal_Outcome_Equilibrium_Splits.csv'))
    scatter_df = pd.read_csv(os.path.join(CURRENT_DIR, 'Team_Zonal_Rebound_Scatter_Scaled.csv'))

    # Calculate league-wide rebound yield per shooter bin
    league_xg = (splits[(splits['team_name']=='LEAGUE') & (splits['half']=='Whole')]
                 .set_index('dist_bin')['xG_rate'].to_dict())
    lg_scatter = scatter_df[(scatter_df['team_name']=='LEAGUE') & (scatter_df['half']=='Whole')].copy()
    lg_scatter['reb_xg'] = lg_scatter['rebound_bin'].map(league_xg).fillna(0.09)
    lg_scatter['wv']     = lg_scatter['scaled_rebound_prob'] * lg_scatter['reb_xg']
    y_agg = lg_scatter.groupby('shooter_bin').agg(wv=('wv','sum'), rp=('scaled_rebound_prob','sum'))
    y_agg['yield'] = y_agg['wv'] / y_agg['rp']
    yield_lookup = y_agg['yield'].to_dict()
    league_avg_yield = y_agg['yield'].mean()

    # 5v5 primary shots only
    df = shots_raw[
        (shots_raw['isPlayoffGame'] == 0) & 
        (shots_raw['awaySkatersOnIce'] == 5) & 
        (shots_raw['homeSkatersOnIce'] == 5) & 
        (shots_raw['shotOnEmptyNet'] == 0) &
        (shots_raw['shotRebound'] == 0)
    ].copy()

    df['dist_bin'] = df['arenaAdjustedShotDistance'].fillna(df['shotDistance']).apply(get_dist_bin_6ft)
    df['teamCode'] = df['teamCode'].str.upper().str.replace(' ', '').map(TEAM_NAME_TO_CODE).fillna(df['teamCode'])
    df['team_name'] = df['teamCode'].map(TEAM_CODE_MAP)

    print("Step 2: Calculating literal xNEV per shot...")
    zonal_pure = splits[splits['half'] == 'Whole'].copy()
    df = df.merge(zonal_pure[['dist_bin', 'team_name', 'reb_rate', 'recov_rate', 'exit_rate', 'freeze_rate']], 
                        on=['dist_bin', 'team_name'], how='left')
    
    for col in ['reb_rate', 'recov_rate', 'exit_rate', 'freeze_rate']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
    df['yield_val'] = df['dist_bin'].map(yield_lookup).fillna(league_avg_yield)
    df['shot_xnev'] = (df['xGoal'].fillna(0.02) + (df['reb_rate'] * df['yield_val']) + 
                       (df['recov_rate'] * ENTRY_VALUE) + (df['freeze_rate'] * FACEOFF_EQUITY) - 
                       (df['exit_rate'] * EXIT_PENALTY))

    df = df.dropna(subset=['shot_xnev'])

    print("Step 3: Aggregating team metrics...")
    team_results = df.groupby('teamCode').agg(
        total_shots=('shot_xnev', 'count'),
        total_xnev=('shot_xnev', 'sum')
    ).reset_index()
    team_results['avg_xnev'] = team_results['total_xnev'] / team_results['total_shots']

    # Wealth vs efficacy scatter
    fig, ax = plt.subplots(figsize=(16, 10))
    x = team_results['total_xnev']
    y = team_results['avg_xnev']
    ax.set_facecolor('#0d1117')
    scatter = ax.scatter(x, y, s=150, c=y, cmap='RdYlGn', alpha=0.8, edgecolor='white', linewidth=1)
    
    # Crosshairs (League Averages)
    ax.axvline(x.mean(), color='white', linestyle='--', alpha=0.5)
    ax.axhline(y.mean(), color='white', linestyle='--', alpha=0.5)

    for i, row in team_results.iterrows():
        ax.annotate(row['teamCode'], (row['total_xnev'], row['avg_xnev']), 
                     xytext=(0, 10), textcoords='offset points', ha='center', 
                     fontsize=9, fontweight='bold', color='white')

    ax.set_xlabel('Total Season Tactical Value (Cumulative xNEV)', fontsize=14, fontweight='bold', color='white')
    ax.set_ylabel('Systemic xEfficacy per Shot (Avg xNEV)', fontsize=14, fontweight='bold', color='white')
    ax.set_title('The Efficiency Frontier: Wealth vs. Selection', fontsize=22, fontweight='bold', pad=30, color='white')
    ax.grid(True, linestyle=':', alpha=0.3)

    # Tick visibility on dark background
    ax.tick_params(axis='both', colors='white', labelsize=11)
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_color('#0d1117')
    ax.spines['right'].set_color('#0d1117')

    # Format x-axis as clean integers (total xNEV values are in the hundreds)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f'{int(val)}'))

    # Format y-axis to 4 decimal places (avg xNEV values are small floats like 0.0671)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f'{val:.4f}'))

    # Add colorbar scale so the RdYlGn gradient is explained
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Avg xNEV per Shot', color='white', fontsize=11)
    cbar.ax.yaxis.set_tick_params(color='white', labelsize=9)
    # Correct way to set label colors in newer matplotlib
    cbar.ax.yaxis.set_tick_params(labelcolor='white')
    
    out_png = os.path.join(CURRENT_DIR, 'approach7_wealth_vs_efficacy.png')
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, facecolor='black', bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_png}")

    print(f"\nRECONCILIATION VERIFICATION (Pitt):")
    pitt = team_results[team_results['teamCode'] == 'PIT']
    if not pitt.empty:
        print(f"Pittsburgh Total xNEV: {pitt['total_xnev'].values[0]:.2f}")

if __name__ == "__main__":
    run_league_efficacy_scatter()
