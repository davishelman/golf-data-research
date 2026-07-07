# player_course_advantage

A downstream model over **v2.5** similar-hole sets: given an upcoming course `C`
and a player `p`, estimate `p`'s per-hole and per-course **advantage** from how
`p` historically scored on holes that look like each of `C`'s 18 holes.

- **Not** a similarity model. It *reads* v2.5 similarity (`total_score` / rank)
  and never mutates v2 or v2.5 outputs.
- **Positive-is-good** primary outcome: `field_avg_score - player_score`
  (adjusts for hole difficulty and field/day conditions).

## Status

This first PR (issues **#30**, **#31**) ships the **spec** and the historical
hole-score **input schema + validation** only. The scorer is deferred.

- Full spec, notation, formulas, defaults, leakage warnings:
  [`docs/player_course_advantage.md`](../../../docs/player_course_advantage.md)
- Input contract + validation: [`schema.py`](schema.py)
  (`validate_hole_score_history`, `AdvantageParams`, `DEFAULT_PARAMS`,
  `field_adjusted_advantage`).

## Experimental defaults

`n = 10` similar holes/target · `W = 5`-year lookback · `m = 0.85` recency decay.
Placeholders from the sketch — not yet calibrated. See `AdvantageParams`.
