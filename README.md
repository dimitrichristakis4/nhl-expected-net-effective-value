# nhl-expected-net-effective-value
# xNEV

A shot quality stat that extends NHL expected goals with what happens after the shot is taken.

## What it measures

Standard xG scores the shot itself. xNEV adds four outcomes that affect expected value from a possession:

- **Rebound** - a follow-up shot within 4 seconds, using the league-wide scatter distribution and team-specific yield
- **Recovery** - shooting team regains possession in the offensive zone (~0.052 xG)
- **Freeze** - puck is covered, resulting in a faceoff (~0.023 xG)
- **Exit** - defending team clears the zone (-0.026 xG)

```
xNEV = xG + (reb_rate x reb_yield) + (recov_rate x 0.052) + (freeze_rate x 0.023) - (exit_rate x 0.026)
```

These constants were derived from BigDataBall play-by-play and validated across the full 2022-23 regular season.

## Validation

Predicting actual 5v5 goals scored across all 32 teams:

| Metric | R2 |
|--------|----|
| xNEV | **0.484** |
| Public xG | 0.443 |
| Corsi % | 0.402 |

## Key finding

The 12-18 ft zone is the clearest differentiator between high-volume and high-efficiency offenses. Carolina drives shot volume from distance. Edmonton and Toronto concentrate attempts in that slot range. xNEV surfaces this distinction more clearly than raw xG because it accounts for what rebounds from that zone are actually worth.

## Data sources

- **MoneyPuck** shots data (shots_2022.csv) - free at moneypuck.com
- **BigDataBall** 2022-23 NHL PbP logs (paid) - required for pressure timing and zone context
- **games-2.csv** - team-level season stats (GF, xGF, CF%) used for validation

See `data/README.md` for setup instructions.

## Pipeline

Scripts must run in order:

1. `pipeline/01_process_bigdataball.py` - parses BigDataBall Excel, extracts oz_pressure timing
2. `pipeline/02_create_game_bridge.py` - bridges MoneyPuck game IDs to BigDataBall game IDs
3. `pipeline/03_build_xnev_baselines.py` - builds per-zone xNEV baselines with rebound yield
4. `pipeline/04_rebound_xg_analysis.py` - diagnostic: rebound xG by distance zone
5. `pipeline/05_build_identity_pressure.py` - merges MoneyPuck shots with BDB pressure timing
6. `pipeline/06_calculate_equilibrium_constants.py` - derives tactical outcome rates, applies Bayesian blending, outputs team-zonal splits

Analysis:

- `analysis/validation_predictor_race.py` - xNEV vs xG vs Corsi predictor comparison
- `analysis/efficiency_frontier.py` - total xNEV vs avg xNEV per shot scatter

## Outputs

- `outputs/xNEV_Shot_Baselines.csv` - per-team, per-zone xNEV baselines
- `outputs/Team_Zonal_Outcome_Equilibrium_Splits.csv` - tactical outcome rates by team, zone, and half-season
- `outputs/Team_Zonal_Rebound_Scatter_Scaled.csv` - rebound destination distribution by source zone

## Status

Presented at OUSAC 2023. Post-conference cuts: possession sustainability metric (uptick too minor to publish), SMI index (circular definition), and related downstream analysis.
