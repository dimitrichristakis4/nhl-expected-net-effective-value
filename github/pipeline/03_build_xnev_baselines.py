import pandas as pd
import numpy as np
import os

# Paths
SCRIPT_DIR = '/Users/dimitrichristakis/Desktop/OUSAC ABstract 2/'
DATA_DIR = os.path.join(SCRIPT_DIR, 'Not right now ')

# Constants
BIN_EDGES = [0, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69]
BIN_LABELS = [f"{lo}-{hi} ft" for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:])]

def bin_distance(dist):
    for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        if lo <= dist < hi:
            return f"{lo}-{hi} ft"
    return None

def build_xnev_baselines():
    print("Step 1: Loading Raw Source Data (xNEV Baseline)...")

    # We ONLY need the Moneypuck shots file
    shots_raw = pd.read_csv(os.path.join(SCRIPT_DIR, 'shots_2022.csv'))
    print(f"  -> Total Raw Shots Loaded: {len(shots_raw)}")

    # Step 2: The Perfect 5v5 Filter (Using built-in MP columns)
    # 1. No Playoffs
    # 2. Exactly 5 skaters for both teams (eliminates 6v5 delayed penalties)
    # 3. No Empty Nets (Goalies are in the crease)
    print("Step 2: Applying True 5v5 Filter...")

    v5v5 = shots_raw[
        (shots_raw['isPlayoffGame'] == 0) &
        (shots_raw['awaySkatersOnIce'] == 5) &
        (shots_raw['homeSkatersOnIce'] == 5) &
        (shots_raw['awayEmptyNet'] == 0) &
        (shots_raw['homeEmptyNet'] == 0)
    ].copy()

    v5v5['dist_bin'] = v5v5['arenaAdjustedShotDistance'].apply(bin_distance)
    v5v5 = v5v5.dropna(subset=['dist_bin'])
    print(f"  -> Verified 5v5 Shots Isolated: {len(v5v5)}")

    # Step 3: Calculate league averages
    print("Step 3: Calculating League-Wide 5v5 Averages...")

    league_primary = v5v5[v5v5['shotRebound'] == 0]

    # GLOBAL Rebound Probability on ANY given shot
    global_reb_prob = league_primary['shotGeneratedRebound'].mean()
    print(f"  -> Global probability of a rebound on any 5v5 shot: {global_reb_prob:.4f}")

    # Bin-Specific League Goal Prob
    league_goal_probs = league_primary.groupby('dist_bin')['goal'].mean().to_dict()

    # Bin-Specific League Rebound Creation Prob
    league_reb_probs = league_primary.groupby('dist_bin')['shotGeneratedRebound'].mean().to_dict()

    # League Rebound Vectoring and Yield
    rebounds = v5v5[v5v5['shotRebound'] == 1].copy()
    generators = v5v5[v5v5['shotGeneratedRebound'] == 1].copy()

    league_links = []
    reb_game_groups = rebounds.groupby(['game_id', 'teamCode'])
    for i, gen_row in generators.iterrows():
        try:
            game_reb = reb_game_groups.get_group((gen_row['game_id'], gen_row['teamCode']))
            follow_up = game_reb[(game_reb['time'] > gen_row['time']) &
                                 (game_reb['time'] <= gen_row['time'] + 4)].head(1)
            if not follow_up.empty:
                league_links.append({
                    'source_zone': gen_row['dist_bin'],
                    'target_zone': follow_up.iloc[0]['dist_bin'],
                    'rebound_xG': follow_up.iloc[0]['xGoal']
                })
        except KeyError:
            continue

    ll_df = pd.DataFrame(league_links)

    # League Destination Prob (Source -> Target %)
    league_dest = ll_df.groupby(['source_zone', 'target_zone']).size().reset_index(name='count')
    league_dest['dest_prob'] = league_dest.groupby('source_zone')['count'].transform(lambda x: x / x.sum())

    # Pre-build a dictionary of destination probabilities for each source zone.
    # Modified so they add up to the total rebound creation probability instead of 1.0
    dest_prob_dict = {}
    for source in BIN_LABELS:
        dest_prob_dict[source] = {f"reb_to_{target}": 0.0 for target in BIN_LABELS}
        zld = league_dest[league_dest['source_zone'] == source]
        source_reb_prob = league_reb_probs.get(source, 0)

        for _, r in zld.iterrows():
            # Multiply the conditional dest_prob by the overall chance of creating a rebound from that source
            dest_prob_dict[source][f"reb_to_{r['target_zone']}"] = r['dest_prob'] * source_reb_prob

    # League Rebound Yield (xG in Target Zone)
    league_yield = ll_df.groupby('target_zone')['rebound_xG'].mean().to_dict()

    # Step 4: Calculate team-specific metrics
    print("Step 4: Calculating Team-Specific 5v5 Metrics...")

    teams = sorted(v5v5['teamCode'].unique())
    master_rows = []

    for team in teams:
        team_v5v5 = v5v5[v5v5['teamCode'] == team]
        team_primary = team_v5v5[team_v5v5['shotRebound'] == 0]
        team_generators = team_v5v5[team_v5v5['shotGeneratedRebound'] == 1]
        team_rebounds = team_v5v5[team_v5v5['shotRebound'] == 1]

        # Link Team Rebounds to get their specific xG yield in target areas
        team_links = []
        for i, gen_row in team_generators.iterrows():
            follow_up = team_rebounds[(team_rebounds['game_id'] == gen_row['game_id']) &
                                     (team_rebounds['time'] > gen_row['time']) &
                                     (team_rebounds['time'] <= gen_row['time'] + 4)].head(1)
            if not follow_up.empty:
                team_links.append({
                    'target_zone': follow_up.iloc[0]['dist_bin'],
                    'rebound_xG': follow_up.iloc[0]['xGoal']
                })
        tl_df = pd.DataFrame(team_links)

        # Calculate the team's average xG yield for rebounds landing in each target zone
        team_yield_in_zone = {}
        if not tl_df.empty:
            team_yield_in_zone = tl_df.groupby('target_zone')['rebound_xG'].mean().to_dict()

        for zone in BIN_LABELS:
            zone_primary = team_primary[team_primary['dist_bin'] == zone]

            # A. Goal Prob (Team Specific w/ Imputation for low sample size)
            if len(zone_primary) > 5:
                goal_prob = zone_primary['goal'].mean()
            else:
                goal_prob = league_goal_probs.get(zone, 0)

            # B. Rebound Creation Prob (STRICTLY LEAGUE WIDE)
            reb_prob = league_reb_probs.get(zone, 0)

            # C & D. Vectoring (LEAGUE WIDE) & Yield (TEAM SPECIFIC)
            zone_league_dest = league_dest[league_dest['source_zone'] == zone]

            secondary_val = 0
            for _, row in zone_league_dest.iterrows():
                target = row['target_zone']
                dest_prob = row['dest_prob'] # We keep this as conditional for the expected yield math

                # Yield is team-specific; if they haven't generated a rebound to this zone, fallback to league avg
                y = team_yield_in_zone.get(target, league_yield.get(target, 0))
                secondary_val += dest_prob * y

            row_data = {
                'teamCode': team,
                'dist_bin': zone,
                'primary_goal_prob': goal_prob,
                'rebound_creation_prob': reb_prob,
                'secondary_xg_yield': secondary_val,
                'xNEV_Baseline': goal_prob + (reb_prob * secondary_val)
            }
            # Append all absolute destination probabilities for this source zone
            row_data.update(dest_prob_dict[zone])
            master_rows.append(row_data)

    # Step 5: Append league average as a pseudo-team row
    for zone in BIN_LABELS:
        gp = league_goal_probs.get(zone, 0)
        rp = league_reb_probs.get(zone, 0)

        zone_league_dest = league_dest[league_dest['source_zone'] == zone]
        sec_val = sum([r['dest_prob'] * league_yield.get(r['target_zone'], 0) for _, r in zone_league_dest.iterrows()])

        row_data = {
            'teamCode': 'LEAGUE_AVG',
            'dist_bin': zone,
            'primary_goal_prob': gp,
            'rebound_creation_prob': rp,
            'secondary_xg_yield': sec_val,
            'xNEV_Baseline': gp + (rp * sec_val)
        }
        row_data.update(dest_prob_dict[zone])
        master_rows.append(row_data)

    # Final Output Generation
    master_df = pd.DataFrame(master_rows)

    # Optional: Ensure directory exists
    os.makedirs(DATA_DIR, exist_ok=True)

    out_path = os.path.join(DATA_DIR, 'xNEV_Shot_Baselines.csv')
    master_df.to_csv(out_path, index=False)

    print("\n" + "="*50)
    print(f"xNEV Baselines Generated: {out_path}")
    print(f"Total Verified 5v5 Shots Analyzed: {len(v5v5)}")
    print("="*50 + "\n")

if __name__ == "__main__":
    build_xnev_baselines()
