# Player-course advantage — historical data sourcing plan (#46)

**Status: BLOCKER (planning + data contract).** The player-course advantage model
(spec: [`player_course_advantage.md`](player_course_advantage.md)) needs a
**historical player-by-hole scoring table**. No such data exists in this repo
today — it holds only course *geometry* (`courses/`, point clouds, DEM) and the
derived v2/v2.5 similarity. Until a licensed per-hole source is wired and
normalized, every advantage/backtest number is **synthetic** and carries no
predictive claim. This doc defines what to source, how to map it, and what blocks
real evaluation.

## 1. What the model needs

The contract is enforced by
`pipeline.modeling.player_course_advantage.schema.validate_hole_score_history`
(see [spec §9](player_course_advantage.md#9-historical-hole-score-input-schema-issue-31)).
One row per **player / tournament / year / round / course / hole**, with required
columns:

`player_id, tournament_id, year, round, hole_number, course_slug, hole_id_v25,
par, player_score, field_avg_score`

`field_avg_score` (the field's mean strokes on that hole/round) is **required** —
it is what makes the outcome field-relative. A source without it, or without
enough of the field to compute it, is a hard blocker for that event.

## 2. Candidate sources

| Source | Granularity | Field-avg computable? | Years | Licensing / limits | Feasibility |
|--------|-------------|-----------------------|-------|--------------------|-------------|
| **PGA Tour ShotLink** | shot-level (every stroke) | Yes (aggregate to hole/round) | ~2003→ | Academic/commercial agreement; **no public redistribution** | High fidelity, **access-gated** |
| **Data Golf API** | round + some hole-level | Partial (hole-level coverage varies) | ~2017→ hole-level | Paid API, ToS forbids redistribution | Feasible if subscribed |
| **Public leaderboards** (PGA Tour / ESPN hole-by-hole scorecards) | per-hole scorecard | Yes, if all players scraped per round | recent seasons | Site ToS + rate limits; fragile HTML | Medium; ETL + upkeep |
| **Majors archives / Wikipedia** | sometimes per-hole for leaders only | No (partial field only) | varies | open-ish | Low — partial fields break `field_avg_score` |
| **Manual scorecards** | per-hole | Only for entered players | any | n/a | Tiny samples only |

**At least one feasible source:** ShotLink (highest fidelity) or the Data Golf API
(lowest friction) can both satisfy the contract. Both are **access-gated** and
**non-redistributable**, which is the core blocker: we can *use* them locally but
cannot commit the raw data.

## 3. Can field average be computed?

Yes — whenever we have **every (non-WD) player's** per-hole score for a
(course, year, round, hole), `field_avg_score = mean(player_score)` over that
group. Caveats to encode in the ETL:

- Exclude WDs/DQs from the field mean (or the mean is biased by blow-up holes).
- Post-cut rounds (3–4) have a smaller field — compute the mean over the players
  who actually teed off that round, not the full starting field.
- If a source only exposes leaders/partial fields, `field_avg_score` is **not**
  trustworthy → mark the event unusable rather than fabricate it.

## 4. ID mapping (source → v2.5 / v2)

Course identity is the join into similarity. Our slugs match the `courses/`
directory (e.g. `augusta_national`, `pebble_beach_golf_links`,
`harbour_town_golf_links`). Derived ids follow the existing contract:

- `hole_id_v25 = "{course_slug}:{hole_number}"` — join into v2.5 similar-hole sets.
- `hole_id_v2  = "{course_slug}__{hole_number:02d}"` — optional v2 feature id.

The one piece of **manual** work: a **course-name → `course_slug` crosswalk**
(source venue strings like "Augusta National Golf Club" → `augusta_national`),
kept as a small maintained mapping. Hole numbering must be checked per venue
(nines rotated / renumbered in some events). A player-name → stable `player_id`
crosswalk is likewise needed if the source lacks stable ids.

## 5. Years realistically available

- ShotLink: 2003→ (deep history), gated.
- Data Golf hole-level: roughly 2017→.
- Public scorecards: reliably only the last several seasons; older pages decay.

A realistic first cut targets **recent seasons on the ~30 courses already in
`courses/`** (so v2.5 similarity exists for the target holes) — a few seasons is
enough to stand up the backtest without lookahead.

## 6. Licensing / API limits

- **ShotLink**: requires an executed data agreement; redistribution prohibited.
- **Data Golf**: paid API key; ToS forbids republishing raw data.
- **Scraping**: subject to each site's ToS and rate limits; brittle to markup
  changes; must be polite (caching, backoff).
- **Repo rule**: raw/large/proprietary data is **never committed**. Generated
  tables live under gitignored paths (`data/player_course_advantage/`, and the raw
  landing zone should be added to `.gitignore` when the ETL lands). Only tiny
  **synthetic** fixtures are committed.

## 7. Missing fields — handling

| Field | Required? | If missing |
|-------|-----------|-----------|
| `field_avg_score` | **Yes** | event unusable (drop) — do not fabricate |
| `par` | Yes | fill from course metadata (`courses/`), else drop |
| `hole_id_v25` | Yes | derive from `course_slug` + `hole_number` |
| `player_name` | No | leave null; ids still work |
| `hole_id_v2` | No | derive only if v2 cross-ref is wanted |
| `player_strokes_gained_hole` | No | leave null; model uses field-adjusted outcome |
| `made_cut`, `yardage` | No | pass through when present |

## 8. Normalization path into the schema

1. **Extract** raw per-hole (or per-shot) scores from the licensed source.
2. **Aggregate** shot-level → one row per (player, tournament, year, round,
   course, hole) with `player_score`.
3. **Compute** `field_avg_score` per (course, year, round, hole) over the eligible
   field; join back onto each player row.
4. **Map** venue → `course_slug` (crosswalk) and derive `hole_id_v25` /
   optional `hole_id_v2`; attach `par` from course metadata.
5. **Validate** with `validate_hole_score_history(df)` — it fails loudly on nulls,
   duplicate grain, bad ranges, id inconsistencies, and a mismatched cached
   `field_adjusted_score`. Fix upstream until it passes.
6. **Land** the validated table in a gitignored path; feed the scorer / batch /
   backtest.

Column crosswalk (source → schema): `event → tournament_id`,
`season → year`, `rd → round`, `hole → hole_number`, `venue → course_slug`
(via crosswalk), `strokes → player_score`, computed `field_avg_score`.

## 9. Sample fixture (safe, synthetic)

A tiny, obviously-fake example of the *target* shape lives at
[`tests/fixtures/player_course_advantage/sample_hole_scores.csv`](../tests/fixtures/player_course_advantage/sample_hole_scores.csv)
and is checked against `validate_hole_score_history` by
`tests/test_player_course_advantage_data_sourcing.py`. It documents the exact
normalization target — **not** real data.

## 10. Blockers / what remains manual

- **Blocker:** no licensed per-hole source is wired; real evaluation cannot begin
  until one (ShotLink or Data Golf) is provisioned under an agreement that permits
  our use.
- **Manual:** the course-name → `course_slug` crosswalk, the player-name → id
  crosswalk, and per-venue hole-numbering checks.
- **Not built:** the extract/aggregate/field-avg ETL, and the raw landing-zone
  `.gitignore` entries (add when the ETL lands).

Until then the model, batch ranking, backtest, baselines, and sweep all run on
**synthetic** data and make **no** predictive-validity claim.
