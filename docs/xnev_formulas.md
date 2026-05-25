# xNEV Formula Reference

## Core formula

```
xNEV = xG + (reb_rate * reb_yield) + (recov_rate * ENTRY_VALUE) + (freeze_rate * FO_EQUITY) - (exit_rate * EXIT_PENALTY)
```

Applied per shot, then summed to team totals.

## Constants

Derived from BDB play-by-play, 2022-23 regular season 5v5.

| Constant | Value |
|----------|-------|
| ENTRY_VALUE | 0.052072 |
| FO_EQUITY | 0.023070 |
| EXIT_PENALTY | 0.026036 |

See `pipeline/06_calculate_equilibrium_constants.py` for how these are derived. In brief: ENTRY_VALUE is the average xG generated per offensive zone entry across all BDB events. FO_EQUITY is the net xG from an offensive zone faceoff until the next dead play. EXIT_PENALTY is half of ENTRY_VALUE - a zone exit by the defense is not as dangerous as a clean entry because the opponent still has to carry through the neutral zone and isn't guaranteed to get in.

## Outcome definitions

These are the five buckets assigned to each primary shot in the BDB native audit (`06_calculate_equilibrium_constants.py`):

- **goal** - the shot itself scores
- **rebound** - a follow-up shot occurs within 3 seconds
- **recovery** - shooting team regains possession in the offensive zone
- **freeze** - puck covered (Faceoff or Goalie Stopped event follows)
- **exit** - defending team clears; puck leaves the offensive zone

Each shot gets exactly one outcome. Rates are calculated per team per distance bin.

## Rebound yield

Rebound contribution is two-part: how often a shot generates a rebound (reb_rate) and how dangerous that rebound is (reb_yield).

```
reb_yield = sum over all target zones: scatter_prob(source -> target) * xG_in_target_zone
```

`scatter_prob` is the league-wide probability that a rebound from a given source zone lands in a given target zone. `xG_in_target_zone` is the team-specific average xG of shots taken from that zone; falls back to league average if sample is small.

## Bayesian blending

Applied to all outcome rates when a team has fewer than 30 shots in a zone.

```
blended_rate = (local_rate * w_local) + (neighbor_rate * (1 - w_local))
w_local = min(vol, 30) / 30.0
```

Neighborhood priors:
- **SLOT**: zones under 18 ft
- **MID**: 18-36 ft
- **OUT**: 36 ft and beyond

If a team has zero shots in an entire neighborhood, the prior falls back to the league average for that neighborhood.

## xNEV Baseline

Used in `03_build_xnev_baselines.py` to build zone-level shot value before applying BDB tactical rates.

```
xNEV_Baseline = primary_goal_prob + (rebound_creation_prob * secondary_xg_yield)
```

- `primary_goal_prob`: team-specific goal rate from that zone (or league average if fewer than 5 shots)
- `rebound_creation_prob`: league-wide rate of generating a rebound from that zone
- `secondary_xg_yield`: expected xG of the follow-up rebound, using league scatter and team-specific yield

## Distance bins

Shots are binned by `arenaAdjustedShotDistance` in feet. The first bin is 0-6 ft (the slot), then 3 ft bins from there:

`0-6, 6-9, 9-12, 12-15, 15-18, 21-24, ... up to 66-69 ft`

In `06_calculate_equilibrium_constants.py` the BDB audit uses 6 ft bins:

`0-6, 6-12, 12-18, 18-24, 24-30, 30-36, 36-42, 42-48, 48-54, 54-60, 60-66`
