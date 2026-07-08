# Ingesting real historical per-hole scores (#70, #71)

How to turn a **private** raw per-hole score export into the canonical
player-course advantage history schema — safely, locally, and without committing
any real data.

> No real data is bundled in this repo. This is the *adapter*; you bring the data.
> Everything you generate here (raw input, aliases, normalized output) stays
> **private and gitignored** — see the guard rules at the bottom.

## The pipeline

```
private raw CSV → normalize → map course/hole ids → compute field_avg_score
→ validate_hole_score_history → canonical history CSV (private path)
```

## 1. Prepare your raw CSV

A bring-your-own export with (at least) these columns:

| column | required? | notes |
|--------|-----------|-------|
| `source_event_id` | yes | your event id → `tournament_id` |
| `event_name` | no | → `tournament_name` |
| `season` | yes | → `year` |
| `round` | yes | 1–4 (extra tolerated) |
| `course_name` | yes* | mapped to `course_slug` via aliases |
| `course_slug` | no | use this instead of `course_name` if you already have canonical slugs |
| `hole_number` | yes | 1–18 |
| `player_id` | yes | any stable key |
| `player_name` | no | → `player_name` |
| `score` | yes | strokes → `player_score` (must be ≥ 1) |
| `par` | no** | required by the schema; filled per-hole if partially present, else you must supply it |
| `field_avg_score` | no | computed per event/season/course/round/hole if absent |

\* either `course_name` or `course_slug`.  \*\* if `par` is missing for a hole with
no par anywhere in the file, ingestion fails and asks you to provide it.

## 2. Add course aliases (without editing raw input)

Course labels rarely match the canonical slugs (`"Augusta National Golf Club"` vs
`augusta_national`). Keep a **separate** alias CSV — never edit your raw export:

```csv
course_name,course_slug
Augusta National Golf Club,augusta_national
Pebble Beach Golf Links,pebble_beach_golf_links
```

Resolution order per row: an existing canonical slug → the alias table →
a `slugify` fallback. A label that resolves outside the known slug set is reported
as **unmapped** (fix it by adding an alias) rather than silently guessed. The
alias target slugs form the "known" universe, so a typo'd alias surfaces as an
unmapped course.

## 3. Run the normalizer

```bash
python scripts/normalize_player_hole_scores.py \
    --input   path/to/private/raw_hole_scores.csv \
    --output  data/player_course_advantage/private/normalized_history.csv \
    --course-aliases path/to/private/course_aliases.csv
```

It writes **only** to `--output`, prints a mapping/data-health summary and a
validation summary, and exits non-zero with a clear message on:

- missing required columns,
- unmapped courses (lists the most frequent unmapped labels) or holes,
- duplicate `player/event/round/course/hole` grain,
- invalid scores (`< 1`) or inconsistent ids.

## 4. Mapping diagnostics (programmatic)

```python
from pipeline.modeling.player_course_advantage import (
    load_course_aliases, build_mapping_report, normalize_and_validate,
)
aliases = load_course_aliases("course_aliases.csv")
report = build_mapping_report(raw, aliases, known_slugs=my_known_slugs,
                              known_hole_ids=my_v25_hole_ids)
# report: total_rows, mapped_rows, unmapped_course_rows, unmapped_hole_rows,
#         duplicate_source_identifier_count, ambiguous_course_label_count,
#         most_frequent_unmapped_course_labels

canonical, validation, mapping = normalize_and_validate(raw, aliases=aliases,
                                                        known_slugs=my_known_slugs)
```

Pass `known_slugs` / `known_hole_ids` from your v2.5 similar-hole sets so
"unmapped" is measured against the **real** hole universe you can actually score
against.

## Guard rules (do not commit real data)

- Canonical output belongs under `data/player_course_advantage/private/`
  (gitignored). Raw exports, alias files, and `normalized_history*.csv` are also
  gitignored by pattern.
- Never commit `.env`, `courses/`, raw/proprietary player data, or generated
  normalized data.
- This adapter only *transforms* what you supply — it downloads nothing and makes
  no predictive claim. Predictive evaluation still requires a real, leakage-free
  history + actual event outcomes (see the
  [data sourcing plan](player_course_advantage_data_sourcing.md), #46).
