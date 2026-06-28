# Hugging Face artifact workflow

The code lives on GitHub; the **generated data** lives on a Hugging Face
Dataset. This page explains why, how to build the upload folder, how to upload
it, and how to pull it back down to reproduce the notebook.

## Why the data isn't in GitHub

The pipeline generates a lot of data:

- ~**54.7M** labeled 3D points across **540** holes,
- `all_hole_points.parquet` alone is ~**1 GB**,
- plus per-hole point clouds (`*.jsonl`, `*.parquet`, `*_compact.json`).

That is far too large (and too binary) for a code repo. So `.gitignore` excludes
`courses/`, `*.parquet`, `*.duckdb`, and rasters. GitHub keeps the **code, docs,
notebook, and tests**; Hugging Face keeps the **data needed to reproduce the
notebook/modeling outputs**.

## Recommended dataset repo name

**Use `davishelman/golf-data-research-artifacts`.**

The artifact contains more than point clouds — engineered features, similarity
tables, clusters, manifests, schema, and a feature dictionary. The
`...-artifacts` name (a) describes the whole bundle accurately, and (b) mirrors
the GitHub repo name `golf-data-research`, so the code↔data relationship is
obvious to anyone browsing your portfolio.

`davishelman/golf-hole-point-clouds` is a reasonable alternative if you want the
headline *point-cloud* product front-and-center for discoverability; the tabular
files still sit tidily under `data/` either way. Pick one and keep it stable —
the exporter records the chosen name in `dataset_manifest.json` /
`metadata/provenance.json` (default `golf-data-research-artifacts`).

## Two-tier strategy (Option C)

| Tier | Contents | Size (approx) | Use |
|---|---|---|---|
| **lite** | aggregate tables (features/clusters/similarity) + `all_holes` + manifests + a *curated* set of compact point clouds + 1–2 visual checks + metadata | ~**35 MB** | quick review; reproduce the notebook's tabular/figure outputs |
| **full** | everything in lite **+ all 540 compact point clouds**; optional per-hole point Parquet (`--include-point-parquet`) and/or `all_hole_points.parquet` (`--include-all-points`, ~1 GB) | ~**2 GB** compact-only; ~**3–4 GB** with point Parquet | the complete data product |

The lite tier's curated point clouds are chosen deterministically: the anchor
hole (`augusta_national__01`), its top length-aware (v2) neighbors, plus one
par-3, one water-heavy hole, and one terrain-heavy hole.

## Build the artifact (nothing is uploaded)

Module CLI:

```powershell
python -m pipeline.modeling hf-export --tier lite --output hf_artifact_lite
python -m pipeline.modeling hf-export --tier full --output hf_artifact_full
python -m pipeline.modeling hf-export --tier full --output hf_artifact_full --include-point-parquet
python -m pipeline.modeling hf-export --tier full --output hf_artifact_full --include-all-points
```

Equivalent standalone script:

```powershell
python scripts/build_hf_artifact.py --tier lite --output hf_artifact_lite
python scripts/build_hf_artifact.py --tier full --output hf_artifact_full --include-point-parquet
```

Both print a size summary and write a local folder only — they never call git or
Hugging Face. The output folders (`hf_artifact*/`) are git-ignored.

Prerequisite: build the modeling artifacts first so the inputs exist:

```powershell
python -m pipeline.modeling all          # -> courses/_index/hole_features.parquet, etc.
```

### Artifact folder layout

```
hf_artifact_<tier>/
├── README.md                 # Hugging Face dataset card (YAML front matter + body)
├── dataset_card.md           # same body, no front matter
├── dataset_manifest.json     # counts, coordinate frame, labels, size, per-file checksums
├── data/
│   ├── hole_features.parquet / .csv
│   ├── hole_clusters.parquet
│   ├── hole_similarity_v2.csv
│   ├── hole_similarity_examples.csv
│   ├── all_holes.parquet
│   ├── all_courses_manifest.json
│   └── visual_checks/*.png
├── point_clouds/
│   ├── compact/<course_slug>__<hole_number>.json     # [x_aligned_m, y_aligned_m, z_rel_m, label_id]
│   └── parquet/<course_slug>__<hole_number>.parquet  # full tier + --include-point-parquet
└── metadata/
    ├── label_map.json
    ├── schema.json
    ├── feature_dictionary.json
    └── provenance.json
```

The exporter copies from an explicit allow-list — it never globs the repo, and a
secret guard refuses to copy `.env`/key files and re-scans the output, so secrets
can't leak into an upload.

## Upload manually (you run this; the tooling does not)

Install and authenticate (use your own token — it is **not** stored in the repo):

```powershell
pip install -U huggingface_hub
```

With the current unified `hf` CLI (`huggingface_hub` >= 0.34):

```powershell
hf auth login
hf repo create golf-data-research-artifacts --repo-type dataset
# upload a built folder's contents to the repo root:
hf upload davishelman/golf-data-research-artifacts hf_artifact_full . --repo-type dataset
```

The older `huggingface-cli` still works if that's what you have installed:

```powershell
huggingface-cli login
huggingface-cli repo create golf-data-research-artifacts --type dataset
huggingface-cli upload davishelman/golf-data-research-artifacts hf_artifact_full . --repo-type dataset
```

> `upload <repo_id> <local_path> <path_in_repo>` — the trailing `.` puts the
> folder's contents at the dataset root.

Recommended: upload the **full** tier as the canonical dataset (it is a superset
of lite). Keep the **lite** tier as a fast local rebuild for review/notebook use;
if you also want it on the Hub, upload it under a `lite/` path
(`hf upload <repo> hf_artifact_lite lite --repo-type dataset`) or as a separate
`...-lite` dataset repo.

## Download / load later

Download the whole dataset into a local folder with the `hf` CLI:

```powershell
hf download davishelman/golf-data-research-artifacts --repo-type dataset --local-dir golf-data-research-artifacts
```

Or fetch a single file / snapshot from Python:

```python
import pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download

path = hf_hub_download("davishelman/golf-data-research-artifacts",
                       "data/hole_features.parquet", repo_type="dataset")
features = pd.read_parquet(path)

local = snapshot_download("davishelman/golf-data-research-artifacts", repo_type="dataset")
print(local)  # folder containing data/, point_clouds/, metadata/
```

## Run the notebook: local data *or* a downloaded artifact

`notebooks/hole_similarity_research.ipynb` no longer hard-codes `courses/_index`.
It loads through `pipeline.modeling.artifact_loader.load_modeling_artifacts`,
which understands **both** layouts:

| Source | Layout | How to select |
|---|---|---|
| Local pipeline output | `courses/_index/hole_features.parquet` (tables in the root) | `ARTIFACT_ROOT = Path("..")/"courses"/"_index"` |
| Downloaded HF artifact | `<root>/data/hole_features.parquet` (tables under `data/`) | `ARTIFACT_ROOT = Path("golf-data-research-artifacts")` |

The first code cell of the notebook is the config cell:

```python
from pathlib import Path
ARTIFACT_ROOT = None   # auto-detect; or set one of the lines below
# ARTIFACT_ROOT = Path("..") / "courses" / "_index"
# ARTIFACT_ROOT = Path("golf-data-research-artifacts")
```

`None` auto-detects: it prefers local `courses/_index`, then common downloaded
folders (`golf-data-research-artifacts`, `hf_artifact_lite`, and their `../`
forms). The notebook prints which source it used:

```text
Using artifact source: local ../courses/_index
Using artifact source: Hugging Face artifact at golf-data-research-artifacts
```

You can also load the tables yourself:

```python
from pipeline.modeling.artifact_loader import load_modeling_artifacts
art = load_modeling_artifacts("golf-data-research-artifacts")  # or None to auto-detect
art["features"]        # hole_features DataFrame (required)
art["clusters"]        # cluster assignments + PCA coords (or None)
art["similarity_v1"]   # hole_similarity_examples.csv (or None)
art["similarity_v2"]   # hole_similarity_v2.csv (or None)
art["manifest"], art["schema"], art["feature_dictionary"]   # optional metadata (or None)
art["compact_dir"], art["courses_root"]                      # where visual checks read points
```

### What works in each mode

The **tabular, cluster, and similarity** sections reproduce identically from any
source — they only need `hole_features` (+ clusters). Visual side-by-sides need
each hole's compact point cloud:

- **Local pipeline / full-tier artifact** — every hole is available.
- **Lite-tier artifact** — only the *curated* subset ships (the anchor hole
  `augusta_national__01` and its v2 neighbours, plus a par-3 / water / terrain
  sample). The notebook's `viz_compare` helper checks availability and **skips
  missing holes with a friendly message** instead of crashing, so the default
  query still renders. For arbitrary-hole visuals, use the full local `courses/`
  tree or download the full-tier artifact.

To switch back to local mode, set `ARTIFACT_ROOT = Path("..")/"courses"/"_index"`
(or just delete/rename the downloaded folder and leave `ARTIFACT_ROOT = None`).

## Notes & concerns

- **Size:** lite ≈ 35 MB (fine to upload anywhere). Full compact-only ≈ 2 GB;
  adding per-hole point Parquet pushes toward ~3–4 GB; adding
  `all_hole_points.parquet` adds ~1 GB. The exporter prints a section-by-section
  size summary before you upload, and records it in `dataset_manifest.json`.
- **Licensing:** geometry is derived from OpenStreetMap (**ODbL 1.0** — keep
  attribution and share-alike for redistribution); elevation is USGS 3DEP (public
  domain) and Copernicus GLO-30 (free/open). See `metadata/provenance.json`.
- **Not official data:** this is an OSM/DEM-derived reconstruction, not official
  PGA Tour / course data. The dataset card states this prominently.
```
