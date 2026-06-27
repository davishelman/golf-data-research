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

Single file:

```python
import pandas as pd
from huggingface_hub import hf_hub_download

path = hf_hub_download("davishelman/golf-data-research-artifacts",
                       "data/hole_features.parquet", repo_type="dataset")
features = pd.read_parquet(path)
```

Whole dataset:

```python
from huggingface_hub import snapshot_download
local = snapshot_download("davishelman/golf-data-research-artifacts", repo_type="dataset")
print(local)  # folder containing data/, point_clouds/, metadata/
```

## Run the notebook: local data vs downloaded HF data

`notebooks/hole_similarity_research.ipynb` reads from `courses/_index/`.

- **Local (you ran the pipeline):** the files already exist under
  `courses/_index/` — just run the notebook.
- **From the Hub (fresh clone, no pipeline run):** download the dataset and place
  the tables where the notebook expects them:

```python
from pathlib import Path
import shutil
from huggingface_hub import snapshot_download

src = Path(snapshot_download("davishelman/golf-data-research-artifacts", repo_type="dataset"))
dst = Path("courses/_index"); dst.mkdir(parents=True, exist_ok=True)
for name in ["hole_features.parquet", "hole_features.csv", "hole_clusters.parquet",
             "hole_similarity_v2.csv", "hole_similarity_examples.csv",
             "all_holes.parquet", "all_courses_manifest.json"]:
    f = src / "data" / name
    if f.exists():
        shutil.copy2(f, dst / name)

# compact point clouds (for visual checks), if present:
for j in (src / "point_clouds" / "compact").glob("*.json"):
    slug, num = j.stem.rsplit("__", 1)
    out = Path("courses") / slug / "holes" / f"hole_{int(num):02d}" / "features"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(j, out / "hole_points_compact.json")
```

The lite tier is enough to reproduce the notebook's **tabular and similarity**
sections. Visual side-by-sides only work for holes whose compact point cloud is
present (all of them in the full tier; the curated subset in lite).

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
