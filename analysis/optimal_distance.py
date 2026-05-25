import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import os

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
SPLITS_PATH = os.path.join(REPO_ROOT, 'outputs', 'Team_Zonal_Outcome_Equilibrium_Splits.csv')
SCATTER_PATH = os.path.join(REPO_ROOT, 'outputs', 'Team_Zonal_Rebound_Scatter_Scaled.csv')
OUTPUT_PATH = os.path.join(REPO_ROOT, 'figures', 'optimal_distance.png')

# xNEV formula constants
ENTRY_VALUE = 0.052072
EXIT_PENALTY = 0.026036
FACEOFF_EQUITY = 0.023070

def run_dist_efficacy_analysis():
    print("Top 5 efficacy differentiators by distance zone")

    splits = pd.read_csv(SPLITS_PATH)
    zonal = splits[splits['half'] == 'Whole'].copy()
    scatter_df = pd.read_csv(SCATTER_PATH)

    # Calculate league-wide rebound yield per shooter bin
    league_xg = zonal[zonal['team_name'] == 'LEAGUE'].set_index('dist_bin')['xG_rate'].to_dict()
    lg_scatter = scatter_df[(scatter_df['team_name'] == 'LEAGUE') & (scatter_df['half'] == 'Whole')].copy()
    lg_scatter['reb_xg'] = lg_scatter['rebound_bin'].map(league_xg).fillna(0.09)
    lg_scatter['wv'] = lg_scatter['scaled_rebound_prob'] * lg_scatter['reb_xg']
    y_agg = lg_scatter.groupby('shooter_bin').agg(wv=('wv', 'sum'), rp=('scaled_rebound_prob', 'sum'))
    y_agg['yield'] = y_agg['wv'] / y_agg['rp']
    league_avg_yield = y_agg['yield'].mean()
    yield_lookup = y_agg['yield'].to_dict()

    # xNEV per zone per team
    zonal['yield_val'] = zonal['dist_bin'].map(yield_lookup).fillna(league_avg_yield)
    zonal['shot_xnev'] = (zonal['xG_rate'] +
                          (zonal['reb_rate'] * zonal['yield_val']) +
                          (zonal['recov_rate'] * ENTRY_VALUE) +
                          (zonal['freeze_rate'] * FACEOFF_EQUITY) -
                          (zonal['exit_rate'] * EXIT_PENALTY))

    # Identify top 5 most efficient teams by avg xNEV per zone
    team_overall = zonal[zonal['team_name'] != 'LEAGUE'].groupby('team_name')['shot_xnev'].mean().sort_values(ascending=False)
    top_5_teams = team_overall.head(5).index.tolist()
    print(f"  Top 5: {top_5_teams}")

    dist_order = ["0-6 ft", "6-12 ft", "12-18 ft", "18-24 ft", "24-30 ft", "30-36 ft",
                  "36-42 ft", "42-48 ft", "48-54 ft", "54-60 ft", "60+ ft"]
    zonal['dist_idx'] = zonal['dist_bin'].apply(lambda x: dist_order.index(x) if x in dist_order else 99)
    zonal = zonal[zonal['dist_idx'] < 99].sort_values(['team_name', 'dist_idx'])

    lg_baseline = zonal[zonal['team_name'] == 'LEAGUE'].set_index('dist_idx')['shot_xnev'].to_dict()

    plt.figure(figsize=(16, 10), facecolor='black')
    ax = plt.gca()
    ax.set_facecolor('black')

    # Elite average delta vs league
    elite_agg = zonal[zonal['team_name'].isin(top_5_teams)].groupby('dist_idx')['shot_xnev'].mean()
    elite_delta = elite_agg - pd.Series(lg_baseline)

    ax.plot(elite_delta.index, elite_delta.values, label='Top 5 elite average (advantage)',
            color='#FF4C00', linewidth=8, marker='o', markersize=12,
            path_effects=[patheffects.withStroke(linewidth=12, foreground='#FF4C00', alpha=0.3)],
            zorder=10)

    ax.fill_between(elite_delta.index, 0, elite_delta.values, color='#FF4C00', alpha=0.1)
    ax.axhline(0, color='white', linestyle='--', alpha=0.6, linewidth=2, label='NHL average baseline')

    # Highlight 12-30ft zone
    ax.axvspan(2, 4, color='#00FF00', alpha=0.1, label='Systemic differentiator zone (12-30ft)')

    # Annotate absolute peak
    peak_idx = elite_delta.idxmax()
    peak_val = elite_delta[peak_idx]
    peak_dist = dist_order[peak_idx]
    ax.annotate(f'Absolute peak\n{peak_dist} (+{peak_val:.4f})',
                (peak_idx, peak_val),
                xytext=(-40, 40), textcoords='offset points', color='#00FFFF', fontsize=12, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#00FFFF', lw=2),
                bbox=dict(boxstyle='round,pad=0.3', fc='black', ec='#00FFFF', alpha=0.8))

    # Annotate peak within the 12-30ft window
    zone_slice = elite_delta.iloc[2:5]
    zone_peak_idx = zone_slice.idxmax()
    zone_peak_val = zone_slice[zone_peak_idx]
    zone_peak_dist = dist_order[zone_peak_idx]

    ax.annotate(f'Systemic peak\n{zone_peak_dist} (+{zone_peak_val:.4f} xNEV)',
                (zone_peak_idx, zone_peak_val),
                xytext=(40, -60), textcoords='offset points', color='white', fontsize=15, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#00FF00', lw=4),
                bbox=dict(boxstyle='round,pad=0.5', fc='#009900', alpha=0.7))

    ax.text(3, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05,
            "Elite system differentiator range\n(12-30 ft)",
            color='#00FF00', fontsize=14, fontweight='bold', ha='center', va='bottom', alpha=0.8)

    ax.set_title('Where the elite pull away: top 5 average advantage by distance', color='white', fontsize=24, fontweight='bold', pad=30)
    ax.set_ylabel('Advantage over NHL average (xNEV delta)', color='white', fontsize=16, labelpad=15)
    ax.set_xlabel('Shot distance', color='white', fontsize=16, labelpad=15)

    ax.set_xticks(range(len(dist_order)))
    ax.set_xticklabels(dist_order, rotation=45, color='white', fontsize=12)
    ax.tick_params(axis='y', colors='white', labelsize=13)

    for spine in ax.spines.values():
        spine.set_color('#444444')

    ax.grid(True, linestyle='--', alpha=0.15, color='#888888')
    ax.legend(facecolor='black', edgecolor='#666666', labelcolor='white', fontsize=14, loc='upper right')

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=180, facecolor='black')
    plt.close()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_dist_efficacy_analysis()
