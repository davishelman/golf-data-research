# v2.5 Point-Cloud Similarity — Real Batch Findings

Results from running the v2.5 surface-aware Chamfer model over the **actual
course dataset** for all four shipped scoring presets. These are real generated
findings; the underlying CSV/manifest artifacts stay **gitignored** (under
`courses/_index/pointcloud_similarity/`) and are fully regenerable from the
configs, so only this curated summary lives in the repo.

> Scope: this validates **model behavior and ranking stability**, not player
> performance prediction. "Similar" here means similar normalized hole
> geometry/surface structure under the current v2.5 model — not a claim about
> scoring outcomes.

## Run status

All four full batches completed (exit code 0). Each run: **540 targets, 12,055
ranked rows.**

| Config | Status |
|---|---|
| `baseline` | ✅ |
| `hazard_heavy` | ✅ |
| `green_heavy` | ✅ |
| `fairway_heavy` | ✅ |

Regenerate with:

```powershell
python -m pipeline.modeling.pointcloud.sweep `
  --configs configs/similarity/pointcloud_chamfer_v1.yaml `
            configs/similarity/pointcloud_chamfer_hazard_heavy.yaml `
            configs/similarity/pointcloud_chamfer_green_heavy.yaml `
            configs/similarity/pointcloud_chamfer_fairway_heavy.yaml --top-n 25
```

## Validation — `augusta_national:13` (par 5), top-10 across all 4 configs

**Headline:** **Quail Hollow 15** is the **#1** match for Augusta 13 in
`baseline`, `fairway_heavy`, and `green_heavy`, and **#3** in `hazard_heavy`.

**Stable all-config core set** (top matches under *every* preset):

- Quail Hollow 15
- Doral Blue Monster 2
- Pebble Beach 18
- TPC Southwind 16

These are robust geometry/surface plays-alike candidates for Augusta National 13
under the current v2.5 model: they survive changes in surface weighting rather
than appearing only for one preset.

### Config overlap (Jaccard of top-10 candidate sets)

| Pair | Jaccard |
|---|---:|
| baseline ~ fairway_heavy | 0.82 |
| baseline ~ green_heavy | 0.67 |
| fairway_heavy ~ green_heavy | 0.67 |
| baseline ~ hazard_heavy | 0.33 |
| green_heavy ~ hazard_heavy | 0.33 |
| fairway_heavy ~ hazard_heavy | 0.25 |

`baseline`, `fairway_heavy`, and `green_heavy` largely agree; `hazard_heavy` is
the outlier. That is expected and useful: weighting bunkers/water more heavily
reshuffles candidates the most, so `hazard_heavy` acts as an opinionated "hazard
similarity" lens rather than a replacement for the baseline ranking.

## Calibration — all 4 configs

| config | n_pairs | miss_dom | pen_dom | elev_corr | flags |
|---|---:|---:|---:|---:|---|
| baseline | 12055 | 0.034 | 0.0003 | 0.255 | none |
| hazard_heavy | 12055 | 0.196 | 0.0003 | 0.148 | none |
| green_heavy | 12055 | 0.009 | 0.0002 | 0.339 | none |
| fairway_heavy | 12055 | 0.001 | 0.0002 | 0.202 | none |

- **Penalty dominance ≈ 0** everywhere → yardage/elevation penalties are not
  swamping the geometry scores.
- **Missing-surface dominance** is low except `hazard_heavy` (19.6%), which is
  expected because that preset weights bunker/water heavily (and penalizes their
  absence); it is still below the 25% flag threshold.
- **Elevation↔surface correlation** is 0.15–0.34, below the 0.5 double-count flag.
- **No config trips a calibration flag.**

**Net:** the model appears reasonably calibrated across all four presets on real
data. `hazard_heavy` is the most *opinionated* preset, not a broken one — the raw
surface geometry still drives the scores.

## Interpretation

- Ranking stability across presets (same core matches for Augusta 13) indicates
  the model is responding to genuine structural similarity, not noise from weight
  changes.
- The single opinionated preset (`hazard_heavy`) gives an analytically distinct
  "when hazards matter more" view while the others stay stable.
- Calibration shows the scorer is geometry-driven, not penalty-driven.

## Limitations

- This is **model-behavior validation**, not predictive validity for play/scoring.
- "Similarity" reflects OSM-derived surfaces + DEM elevation in the normalized
  tee-relative frame; OSM tagging is inconsistent and elevations are derived (see
  `docs/pointcloud_similarity_v2_5.md`).
- Findings are for the shipped configs at `top_n=25`; other weightings/cutoffs may
  shift the long tail (the robust core set is what's emphasized here).
- Raw batch artifacts are gitignored demonstration data and are regenerable from
  the configs; only this summary is committed.
