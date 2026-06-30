# v2.5 Point-Cloud Similarity — Final QA & Merge Plan

Status of the v2.5 backlog (issues #3–#13) and how the branches fit together.

## Branch / PR structure

The work is split into five reviewable, **stacked** branches off the original
`feature/v2-5-pointcloud-batch-export` (the committed v2.5 batch-export
foundation). Each branch is based on the previous one, so review/merge in order.

| Order | Branch | Issues | Adds |
|---|---|---|---|
| 0 | `feature/v2-5-pointcloud-batch-export` | (foundation) | scorer + batch export |
| 1 | `feature/v2-5-validation-reports` | #3 #4 #5 | `validate_similarity.py` + reports |
| 2 | `feature/v2-5-visual-calibration` | #6 #7 | `visualize_similarity.py`, `calibrate.py`, calibrated config |
| 3 | `feature/v2-5-loader-sweep` | #8 #9 | `surface_loader.py`, `sweep.py` |
| 4 | `feature/v2-5-artifact-demo` | #10 #11 | `artifact_export.py`, `demo.py` |
| 5 | `feature/v2-5-docs-final-qa` | #12 #13 | this doc + architecture doc |

Each branch is stacked on the one above it. **Recommended merge order:** base
(0) → 1 → 2 → 3 → 4 → 5, each into the branch below it or, once the base lands on
`main`, retarget each PR to `main` and merge in sequence. Squash-merging is fine;
the commits are already one-per-group.

## Final test matrix

Run the v2.5 suite (no geo toolchain needed):

```powershell
python -m pytest `
  tests/test_pointcloud_config.py `
  tests/test_pointcloud_candidate_filter.py `
  tests/test_pointcloud_score.py `
  tests/test_pointcloud_export_similarity.py `
  tests/test_pointcloud_validate_similarity.py `
  tests/test_pointcloud_calibrate.py `
  tests/test_pointcloud_visualize.py `
  tests/test_pointcloud_surface_loader.py `
  tests/test_pointcloud_sweep.py `
  tests/test_pointcloud_artifact_demo.py -q
```

| Suite | Result |
|---|---|
| v2.5 point-cloud tests (10 files) | **89 passed** |
| Full light-stack suite (excl. 6 rasterio modules) | **passing** (v2 untouched) |
| Geo modules (`test_assignment/holes/integration/labels/ref_parsing/slope`) | collection errors — **pre-existing**, `rasterio` not installed; unrelated to v2.5 |

The 6 rasterio collection errors predate this work and are an environment gap
(the geo toolchain is not installed locally), not a regression.

## QA checklist

- [x] All v2.5 tests pass (89).
- [x] v2 similarity pipeline untouched (no edits to `pipeline/modeling/similarity.py`
      or its outputs).
- [x] v2.5 is additive — new package, new configs, new tests only; the one edit to
      shared code is an additive property on `IndexPaths` (`pointcloud_similarity_dir`).
- [x] No raw point-cloud geometry duplicated in any result file.
- [x] Deterministic output ordering (`total_score`, then `candidate_hole_id`).
- [x] Generated artifacts stay under the gitignored `courses/` tree; exported to
      the HF bundle only via `artifact_export` (with the secret guard).
- [x] No secrets: `.env` is gitignored, untracked, never staged; the artifact
      exporters refuse secret-like files.
- [x] Commits use the local git identity only, no AI-attribution footers.
- [x] Each branch diff is focused to its issue group.

## Known limitations (carried forward)

* Batch export is O(N²) and ~30 min per config for the full 540-hole set.
* Chamfer is unweighted (`point_weight` reserved, not yet used).
* Tee/green elevations are derived rather than measured absolutely.
* Surface weights are hand-tuned; the calibration report is the basis for fitting
  them to human plays-alike judgments later.

## After merge

1. Regenerate batch outputs for the shipping configs (sweep runner).
2. Run `artifact_export` to fold v2.5 results into the next HF artifact build.
3. Add the documented v2.5 view to `app.py` (a few lines over `demo.py`).
```
