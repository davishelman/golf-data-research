"""Player-course advantage — a downstream model over v2.5 similar-hole sets.

This package answers a *different* question from v2/v2.5 similarity. v2 and v2.5
say **"which holes look alike?"**. This layer says **"given those look-alike
holes, is player p likely to have an edge on upcoming course C?"** — by looking
up how p historically scored on holes similar to each of C's 18 holes.

It is strictly additive and read-only with respect to v2/v2.5: it *consumes*
v2.5 similarity outputs (target hole -> ranked similar holes + ``total_score``)
and a historical player-by-hole scoring table, and never mutates either. See
:doc:`the spec </docs/player_course_advantage>` (``docs/player_course_advantage.md``)
for the full problem statement, notation, and formulas.

Layout (the scorer and backtest remain deferred):

* :mod:`.schema` — historical hole-score input contract, the positive-is-good
  field-adjusted outcome convention, default parameters (``n``, ``W``, ``m``),
  and lightweight validation (:func:`.schema.validate_hole_score_history`).
* :mod:`.similar_holes` — reads v2.5 similarity result CSVs into normalized
  per-target-hole similar-hole sets (:func:`.similar_holes.load_similar_hole_sets`).
* :mod:`.scorer` — the recency-weighted player-hole and player-course advantage
  scorer (:func:`.scorer.score_player_holes`, :func:`.scorer.score_player_course`).
* :mod:`.diagnostics` — explanation outputs reconciling with the scorer
  (:func:`.diagnostics.explain_player_course`, :func:`.diagnostics.contribution_rows`).
* :mod:`.batch` — rank a whole tournament field for a course
  (:func:`.batch.score_tournament_field`), Python API + CLI.
* :mod:`.artifact_export` — persist/load a run (rankings, hole details,
  diagnostics, manifest) in a gitignored artifact layout
  (:func:`.artifact_export.export_advantage_run`, :func:`.artifact_export.load_advantage_run`).
* :mod:`.backtest` — retrospective, leakage-guarded evaluation over historical
  events (:func:`.backtest.run_backtest`).
* :mod:`.baselines` — simple comparison baselines with the model's ranker shape
  (:func:`.baselines.compare_baselines`).
* :mod:`.sweep` — reproducible parameter sweep around the backtest
  (:func:`.sweep.run_sweep`, :class:`.sweep.SweepGrid`).

Nothing here depends on real PGA data, streamlit, or the geometry/DEM stack.
"""

from __future__ import annotations

#: Model family version tag for this downstream advantage layer. Bumped when the
#: input contract or aggregation semantics change in a breaking way.
MODEL_VERSION: str = "player_course_advantage_v0"

from .schema import (  # noqa: E402
    DEFAULT_PARAMS,
    KEY_COLUMNS,
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
    AdvantageParams,
    SchemaError,
    ValidationReport,
    add_field_adjusted_advantage,
    field_adjusted_advantage,
    validate_hole_score_history,
)
from .similar_holes import (  # noqa: E402
    WEIGHT_METHODS,
    SimilarHoleLoaderError,
    add_similarity_weights,
    available_target_hole_numbers,
    load_similar_hole_sets,
    missing_target_hole_numbers,
    parse_v25_hole_id,
)
from .scorer import (  # noqa: E402
    AGGREGATIONS,
    HOLE_OUTPUT_COLUMNS,
    AdvantageScorerError,
    PlayerCourseAdvantage,
    PlayerHoleAdvantage,
    filter_history_for_prediction_window,
    recency_weight,
    score_player_course,
    score_player_holes,
)
from .diagnostics import (  # noqa: E402
    CONTRIBUTION_COLUMNS,
    PlayerCourseExplanation,
    contribution_rows,
    explain_player_course,
)
from .batch import (  # noqa: E402
    FIELD_RANKING_COLUMNS,
    export_field_ranking,
    rank_field,
    score_tournament_field,
)
from .artifact_export import (  # noqa: E402
    ARTIFACT_SUBDIR,
    AdvantageRun,
    ArtifactExportError,
    assemble_field_outputs,
    export_advantage_run,
    load_advantage_run,
    make_run_id,
)
from .backtest import (  # noqa: E402
    BacktestError,
    BacktestResult,
    pearson_corr,
    run_backtest,
    spearman_corr,
    top_k_hit_rate,
    top_k_lift,
)
from .baselines import (  # noqa: E402
    BASELINES,
    compare_baselines,
    model_beats_baselines,
)
from .sweep import (  # noqa: E402
    SweepGrid,
    SweepResult,
    export_sweep,
    iter_param_sets,
    run_sweep,
)
from .data_health import (  # noqa: E402
    build_data_health_report,
    data_health_to_frames,
    summarize_backtest_coverage,
    summarize_history_quality,
    summarize_similarity_coverage,
)
from .ablation import (  # noqa: E402
    ablation_effects,
    baseline_lift_summary,
    baseline_lift_table,
    rank_ablation,
)
from .evaluation import (  # noqa: E402
    build_evaluation_summary,
    export_evaluation_report,
    make_metrics_manifest,
    render_evaluation_markdown,
    summarize_per_event_metrics,
)
from .calibration import (  # noqa: E402
    bucket_predictions,
    calibration_slope,
    calibration_table,
    monotonicity_score,
    reliability_by_coverage,
    render_calibration_summary,
)
from .benchmarks import (  # noqa: E402
    benchmark_components,
    export_benchmarks,
    run_benchmarks,
)
from .identity import (  # noqa: E402
    IdentityError,
    build_mapping_report,
    load_course_aliases,
    resolve_course_slug,
    slugify,
)
from .ingestion import (  # noqa: E402
    IngestionError,
    compute_field_avg_scores,
    normalize_and_validate,
    normalize_hole_scores,
)
from .real_analysis import run_real_analysis  # noqa: E402
from .optimization import (  # noqa: E402
    OptimizationError,
    OptimizationResult,
    export_optimization,
    optimize_parameters,
)
from .variants import (  # noqa: E402
    blend,
    compare_variants,
    default_variants,
    render_variant_summary,
    select_blend_alpha,
    shrink_by_count,
    shrink_toward_zero,
)
from .error_analysis import (  # noqa: E402
    build_error_analysis,
    identify_failure_modes,
    performance_by,
    render_failure_modes,
)
from .insight_report import (  # noqa: E402
    insight_verdict,
    load_and_render,
    render_insight_report,
)
from .course_targets import (  # noqa: E402
    COURSE_TARGET_COLUMNS,
    CourseTargetError,
    build_targets_from_assets,
    load_course_targets,
    supported_targets,
    targets_by_status,
)
from .acquisition import (  # noqa: E402
    SOURCE_MODES,
    AcquisitionError,
    AcquisitionResult,
    find_course_raw_file,
    import_annual_courses,
    resolve_sources,
)
from .annual_analysis import run_annual_analysis  # noqa: E402

__all__ = [
    "MODEL_VERSION",
    "REQUIRED_COLUMNS",
    "OPTIONAL_COLUMNS",
    "KEY_COLUMNS",
    "DEFAULT_PARAMS",
    "AdvantageParams",
    "SchemaError",
    "ValidationReport",
    "field_adjusted_advantage",
    "add_field_adjusted_advantage",
    "validate_hole_score_history",
    # similar-hole loader (#32)
    "WEIGHT_METHODS",
    "SimilarHoleLoaderError",
    "parse_v25_hole_id",
    "add_similarity_weights",
    "load_similar_hole_sets",
    "available_target_hole_numbers",
    "missing_target_hole_numbers",
    # advantage scorer (#33)
    "AGGREGATIONS",
    "HOLE_OUTPUT_COLUMNS",
    "AdvantageScorerError",
    "PlayerHoleAdvantage",
    "PlayerCourseAdvantage",
    "recency_weight",
    "filter_history_for_prediction_window",
    "score_player_holes",
    "score_player_course",
    # diagnostics / explanation (#39)
    "CONTRIBUTION_COLUMNS",
    "PlayerCourseExplanation",
    "contribution_rows",
    "explain_player_course",
    # batch field ranking (#34)
    "FIELD_RANKING_COLUMNS",
    "score_tournament_field",
    "rank_field",
    "export_field_ranking",
    # artifact export + load (#44)
    "ARTIFACT_SUBDIR",
    "AdvantageRun",
    "ArtifactExportError",
    "make_run_id",
    "assemble_field_outputs",
    "export_advantage_run",
    "load_advantage_run",
    # backtest framework (#35)
    "BacktestError",
    "BacktestResult",
    "run_backtest",
    "pearson_corr",
    "spearman_corr",
    "top_k_hit_rate",
    "top_k_lift",
    # baseline comparisons (#38)
    "BASELINES",
    "compare_baselines",
    "model_beats_baselines",
    # parameter sweep (#36)
    "SweepGrid",
    "SweepResult",
    "iter_param_sets",
    "run_sweep",
    "export_sweep",
    # data health report (#62)
    "summarize_history_quality",
    "summarize_similarity_coverage",
    "summarize_backtest_coverage",
    "build_data_health_report",
    "data_health_to_frames",
    # baseline lift & ablation (#60)
    "baseline_lift_table",
    "baseline_lift_summary",
    "rank_ablation",
    "ablation_effects",
    # evaluation report generator (#59)
    "build_evaluation_summary",
    "summarize_per_event_metrics",
    "make_metrics_manifest",
    "render_evaluation_markdown",
    "export_evaluation_report",
    # calibration & reliability (#61)
    "bucket_predictions",
    "calibration_table",
    "monotonicity_score",
    "calibration_slope",
    "reliability_by_coverage",
    "render_calibration_summary",
    # runtime / scalability benchmarks (#63)
    "benchmark_components",
    "run_benchmarks",
    "export_benchmarks",
    # course/hole identity mapping (#71)
    "IdentityError",
    "slugify",
    "load_course_aliases",
    "resolve_course_slug",
    "build_mapping_report",
    # real-data ingestion adapter (#70)
    "IngestionError",
    "compute_field_avg_scores",
    "normalize_hole_scores",
    "normalize_and_validate",
    # real-data analysis runner (#72)
    "run_real_analysis",
    # validation-split optimizer (#73)
    "OptimizationError",
    "OptimizationResult",
    "optimize_parameters",
    "export_optimization",
    # shrinkage / ensemble variants (#74)
    "shrink_toward_zero",
    "shrink_by_count",
    "blend",
    "default_variants",
    "compare_variants",
    "select_blend_alpha",
    "render_variant_summary",
    # error analysis (#75)
    "performance_by",
    "identify_failure_modes",
    "build_error_analysis",
    "render_failure_modes",
    # insight report (#76)
    "insight_verdict",
    "render_insight_report",
    "load_and_render",
    # annual course target manifest (#80)
    "COURSE_TARGET_COLUMNS",
    "CourseTargetError",
    "load_course_targets",
    "supported_targets",
    "targets_by_status",
    "build_targets_from_assets",
    # annual course acquisition / import (#81)
    "SOURCE_MODES",
    "AcquisitionError",
    "AcquisitionResult",
    "resolve_sources",
    "find_course_raw_file",
    "import_annual_courses",
    # batch annual real-analysis runner (#82)
    "run_annual_analysis",
]
