# Rebound xG Analysis by Distance Zone
# Reads shots_2022.csv, bins rebound shots by distance, and computes
# average xG per zone for each team and the league average.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Same bins as the NEV analysis (0-6 ft first, then 3-ft bins)
BIN_EDGES = [0, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69]
BIN_LABELS = [f"{lo}-{hi} ft" for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:])]


def load_shots():
    """Load shots_2022.csv with relevant columns."""
    path = os.path.join(SCRIPT_DIR, 'shots_2022.csv')
    cols = ['shotRebound', 'shotGeneratedRebound', 'arenaAdjustedShotDistance',
            'xGoal', 'goal', 'teamCode', 'game_id', 'xRebound']
    df = pd.read_csv(path, usecols=cols)
    print(f"  Loaded {len(df):,} total shots")
    print(f"  Rebound shots: {df['shotRebound'].sum():,}")
    return df


def bin_distance(dist):
    """Assign a distance to one of the standard bins."""
    for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        if lo <= dist < hi:
            return f"{lo}-{hi} ft"
    return None


def analyze_rebounds(df):
    """Compute avg xG of rebound shots by distance bin, per team + league."""
    
    # Filter to rebound shots only
    rebounds = df[df['shotRebound'] == 1].copy()
    rebounds['dist_bin'] = rebounds['arenaAdjustedShotDistance'].apply(bin_distance)
    rebounds = rebounds.dropna(subset=['dist_bin'])
    
    # Also compute for shots that GENERATED rebounds (the initial shot)
    generators = df[df['shotGeneratedRebound'] == 1].copy()
    
    print(f"\n  Rebound shots with valid distance: {len(rebounds):,}")
    print(f"  Shots that generated rebounds: {len(generators):,}")
    
    # League average
    league = rebounds.groupby('dist_bin').agg(
        count=('xGoal', 'size'),
        avg_xG=('xGoal', 'mean'),
        goal_rate=('goal', 'mean'),
        avg_dist=('arenaAdjustedShotDistance', 'mean')
    ).reset_index()
    
    # Sort by bin order
    league['bin_order'] = league['dist_bin'].apply(
        lambda x: BIN_EDGES[BIN_LABELS.index(x)] if x in BIN_LABELS else 99)
    league = league.sort_values('bin_order')
    
    print(f"\n{'='*80}")
    print(f"  LEAGUE AVERAGE: Rebound Shot xG by Distance Zone")
    print(f"  (Where do rebounds land, and how dangerous are they?)")
    print(f"{'='*80}")
    print(f"  {'Zone':<15} {'Count':>6} {'Avg xG':>8} {'Goal%':>8} {'Avg Dist':>10}")
    print(f"  {'-'*15} {'-'*6} {'-'*8} {'-'*8} {'-'*10}")
    
    for _, row in league.iterrows():
        print(f"  {row['dist_bin']:<15} {row['count']:>6} {row['avg_xG']:>8.4f} "
              f"{row['goal_rate']:>8.3f} {row['avg_dist']:>10.1f}")
    
    total_reb = rebounds['xGoal'].count()
    total_xG = rebounds['xGoal'].mean()
    total_goal = rebounds['goal'].mean()
    print(f"  {'-'*15} {'-'*6} {'-'*8} {'-'*8} {'-'*10}")
    print(f"  {'TOTAL':<15} {total_reb:>6} {total_xG:>8.4f} {total_goal:>8.3f}")
    
    # Per-team analysis
    teams = sorted(rebounds['teamCode'].unique())
    
    team_summaries = []
    for team in teams:
        team_reb = rebounds[rebounds['teamCode'] == team]
        team_gen = generators[generators['teamCode'] == team]
        
        team_summary = {
            'team': team,
            'total_rebounds': len(team_reb),
            'total_generated': len(team_gen),
            'avg_rebound_xG': team_reb['xGoal'].mean(),
            'rebound_goal_rate': team_reb['goal'].mean(),
            'gen_rate': len(team_gen) / len(df[df['teamCode'] == team]) if len(df[df['teamCode'] == team]) > 0 else 0,
        }
        
        # Per-zone breakdown
        for _, lrow in league.iterrows():
            zone = lrow['dist_bin']
            zone_reb = team_reb[team_reb['dist_bin'] == zone]
            team_summary[f'count_{zone}'] = len(zone_reb)
            team_summary[f'xG_{zone}'] = zone_reb['xGoal'].mean() if len(zone_reb) > 0 else np.nan
        
        team_summaries.append(team_summary)
    
    team_df = pd.DataFrame(team_summaries)
    team_df = team_df.sort_values('avg_rebound_xG', ascending=False)
    
    print(f"\n\n{'='*80}")
    print(f"  TEAM RANKINGS: Rebound Shot Quality")
    print(f"  (Sorted by average xG on rebound shots)")
    print(f"{'='*80}")
    print(f"  {'Rank':<5} {'Team':<6} {'Rebounds':>8} {'Generated':>9} {'AvgxG':>8} {'Goal%':>8} {'GenRate':>8}")
    print(f"  {'-'*5} {'-'*6} {'-'*8} {'-'*9} {'-'*8} {'-'*8} {'-'*8}")
    
    for i, (_, row) in enumerate(team_df.iterrows()):
        print(f"  {i+1:<5} {row['team']:<6} {row['total_rebounds']:>8} {row['total_generated']:>9} "
              f"{row['avg_rebound_xG']:>8.4f} {row['rebound_goal_rate']:>8.3f} "
              f"{row['gen_rate']:>8.3f}")
    
    # Save plot and CSVs before entering interactive loop
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    zones = league['dist_bin'].values
    x_pos = range(len(zones))
    
    colors = ['#e74c3c' if xg > 0.15 else '#e67e22' if xg > 0.08 else '#3498db' 
              for xg in league['avg_xG']]
    
    bars = ax1.bar(x_pos, league['avg_xG'].values, color=colors, alpha=0.8, edgecolor='white')
    
    for i, (cnt, xg) in enumerate(zip(league['count'].values, league['avg_xG'].values)):
        ax1.text(i, xg + 0.005, f'n={cnt}', ha='center', va='bottom', fontsize=7, color='gray')
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(zones, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Average xG of Rebound Shot', fontsize=12, fontweight='bold')
    ax1.set_title('League Average: xG of Rebound Shots by Distance Zone\n'
                  '(How dangerous are rebounds from each area?)',
                  fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    ax2.bar(x_pos, league['count'].values, color='#2ecc71', alpha=0.7, edgecolor='white')
    
    for i, cnt in enumerate(league['count'].values):
        pct = cnt / total_reb * 100
        ax2.text(i, cnt + 10, f'{pct:.1f}%', ha='center', va='bottom', fontsize=7, color='gray')
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(zones, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Number of Rebound Shots', fontsize=12, fontweight='bold')
    ax2.set_title('Where Do Rebounds Land?\n(Distribution of rebound shot locations)',
                  fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    out_path = os.path.join(SCRIPT_DIR, 'Rebound_xG_by_Zone.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"\n  Saved: {out_path}")
    
    league.to_csv(os.path.join(SCRIPT_DIR, 'rebound_xg_league.csv'), index=False)
    team_df.to_csv(os.path.join(SCRIPT_DIR, 'rebound_xg_teams.csv'), index=False)
    print(f"  Saved: rebound_xg_league.csv")
    print(f"  Saved: rebound_xg_teams.csv")
    
    plt.show()
    
    # Interactive zone breakdown
    print(f"\n\n  Select a team to see zone-by-zone rebound xG breakdown.")
    print(f"  Type a team abbreviation, 'all', or 'quit'.\n")
    
    while True:
        choice = input("  Team (or 'all'/'quit'): ").strip().upper()
        if choice == 'QUIT':
            break
        
        show_teams = teams if choice == 'ALL' else [choice] if choice in teams else []
        if not show_teams:
            print(f"  '{choice}' not found.")
            continue
        
        for team in show_teams:
            team_reb = rebounds[rebounds['teamCode'] == team]
            team_by_zone = team_reb.groupby('dist_bin').agg(
                count=('xGoal', 'size'),
                avg_xG=('xGoal', 'mean'),
                goal_rate=('goal', 'mean')
            ).reset_index()
            team_by_zone['bin_order'] = team_by_zone['dist_bin'].apply(
                lambda x: BIN_EDGES[BIN_LABELS.index(x)] if x in BIN_LABELS else 99)
            team_by_zone = team_by_zone.sort_values('bin_order')
            
            print(f"\n  {'='*65}")
            print(f"  {team}: Rebound xG by Zone  ({len(team_reb)} rebound shots)")
            print(f"  {'='*65}")
            print(f"  {'Zone':<15} {'Count':>6} {'AvgxG':>8} {'Goal%':>8} {'vsLeague':>10}")
            print(f"  {'-'*15} {'-'*6} {'-'*8} {'-'*8} {'-'*10}")
            
            for _, zrow in team_by_zone.iterrows():
                league_xg = league[league['dist_bin'] == zrow['dist_bin']]['avg_xG'].values
                diff = zrow['avg_xG'] - league_xg[0] if len(league_xg) > 0 else 0
                marker = '+' if diff > 0 else '-'
                print(f"  {zrow['dist_bin']:<15} {zrow['count']:>6} {zrow['avg_xG']:>8.4f} "
                      f"{zrow['goal_rate']:>8.3f} {diff:>+10.4f} {marker}")
        
        if choice != 'ALL':
            continue
        break


if __name__ == '__main__':
    shots = load_shots()
    analyze_rebounds(shots)
