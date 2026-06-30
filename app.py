"""Golf Hole Similarity Explorer — a lightweight Streamlit demo over the HF artifact.

Run it after downloading the dataset artifact:

    pip install streamlit
    hf download davishelman/golf-data-research-artifacts --repo-type dataset \
        --local-dir golf-data-research-artifacts
    streamlit run app.py

It needs no internet at runtime, no Hugging Face token, and no local ``courses/``
tree — only the downloaded artifact. All non-UI logic lives in
``pipeline.modeling.demo_utils`` (v2) and ``pipeline.modeling.pointcloud.demo``
(v2.5) so it stays testable without streamlit.

Two model families are surfaced side by side:

* **v2** — engineered feature-vector similarity (presented plays-like + facets).
* **v2.5** — surface-aware point-cloud Chamfer similarity (per-config presets).
  Reads the batch outputs from the artifact's ``data/pointcloud_similarity/`` or
  a local ``courses/_index/pointcloud_similarity/`` tree; the section hides itself
  when neither is present.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))  # import pipeline from repo root
from pipeline.modeling import demo_utils as du
from pipeline.modeling.pointcloud import demo as pcdemo
from pipeline.modeling.visual_compare import available_compact_ids, plot_hole_comparison

st.set_page_config(page_title="Golf Hole Similarity Explorer", layout="wide")
st.title("Golf Hole Similarity Explorer")
st.caption("Explore golfer-facing plays-like recommendations and raw similarity "
           "facets (v2), plus surface-aware point-cloud similarity (v2.5), from the "
           "Hugging Face artifact.")


@st.cache_resource(show_spinner="Loading artifact…")
def _load(root_str: str) -> dict:
    return du.load_artifact(root_str)


# --------------------------------------------------------------------------- #
# Sidebar: artifact root
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Artifact")
    _detected = du.discover_artifact_root()
    root_str = st.text_input("Artifact root", value=str(_detected) if _detected else "",
                             help="Folder downloaded with `hf download ... --local-dir`.")
    if st.button("Reload artifact"):
        st.cache_resource.clear()
        st.rerun()

if not root_str:
    st.warning("No artifact found. Download it, then set the root in the sidebar:")
    st.code(du.DOWNLOAD_CMD, language="powershell")
    st.stop()

try:
    art = _load(root_str)
except (FileNotFoundError, ValueError) as exc:
    st.error(f"Could not load an artifact at `{root_str}`:\n\n{exc}")
    st.code(du.DOWNLOAD_CMD, language="powershell")
    st.stop()

features = art["features"]
fi = features.set_index("hole_id")

# v2.5 point-cloud results live in the artifact (data/pointcloud_similarity) or a
# local index (courses/_index/pointcloud_similarity). Resolve once, here.
_pc_root = pcdemo.discover_pointcloud_root([
    art.get("root"), art.get("courses_root"), root_str,
    "courses/_index", "../courses/_index",
])
_pc_configs = pcdemo.list_pointcloud_configs(_pc_root) if _pc_root else []


@st.cache_data(show_spinner="Loading v2.5 point-cloud results…")
def _load_pc_results(root_str: str) -> dict:
    return pcdemo.load_pointcloud_results(root_str)


# --------------------------------------------------------------------------- #
# Sidebar: controls
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Query hole")
    slugs = du.course_slugs(features)
    default_course = "augusta_national" if "augusta_national" in slugs else slugs[0]
    course = st.selectbox("Course", slugs, index=slugs.index(default_course))

    holes = du.holes_for_course(features, course)
    hole_ids = holes["hole_id"].tolist()
    labels = {r["hole_id"]: du.hole_label(r) for _, r in holes.iterrows()}
    query_id = st.selectbox("Hole", hole_ids, format_func=lambda h: labels[h])

    st.header("Similarity view")
    view = st.selectbox("View", list(du.VIEW_OPTIONS), index=0)
    top_n = st.radio("Top N", [5, 10, 15], index=1, horizontal=True)

    st.header("Options")
    _is_presented = du.VIEW_OPTIONS[view][0] == "presented"
    show_same_course = st.checkbox("Show same-course matches", value=not _is_presented)
    show_raw_cols = st.checkbox("Show raw diagnostic columns", value=False)
    show_visual = st.checkbox("Show point-cloud visual (if available)", value=True)

    if _pc_configs:
        st.header("v2.5 point-cloud")
        pc_config = st.selectbox("Scoring preset", _pc_configs,
                                 index=_pc_configs.index("baseline")
                                 if "baseline" in _pc_configs else 0)
    else:
        pc_config = None

# v2.5 id for the selected hole (course_slug:hole_number).
q = fi.loc[query_id]
query_pc_id = f"{q['course_slug']}:{int(q['hole_number'])}"

# --------------------------------------------------------------------------- #
# 1. Selected hole summary
# --------------------------------------------------------------------------- #
st.subheader("1. Selected hole")
st.markdown(f"**{course}** · {du.hole_label(q)} · `{query_id}`  ·  v2.5 id `{query_pc_id}`")
summary_cols = [c for c in du.HOLE_SUMMARY_COLS if c in fi.columns]
st.dataframe(q[summary_cols].rename("value").to_frame().T, width="stretch")
st.caption("Note: `sand_pct` (natural / waste sand) is not shown as a headline "
           "hazard — real golf bunkers are `bunker_pct`.")

# --------------------------------------------------------------------------- #
# 2. Similarity results (v2)
# --------------------------------------------------------------------------- #
st.subheader("2. Similar holes — v2 (feature-vector)")
table, kind, source = du.similarity_results(art, view, query_id, top_n, show_same_course)

if kind == "presented":
    st.markdown("**Presented plays-like** — golfer-facing recommendations "
                "(same-par, cross-course, plausibility-filtered). "
                f"Source: *{source}*.")
else:
    st.info("Facet / raw modes compare **one aspect** of the hole and are useful "
            "diagnostics. They are **not** guaranteed to be full plays-like "
            "recommendations (they can include par 5s, short holes, or same-course "
            "neighbours).")

if table is None or table.empty:
    st.warning("No matches available for this hole in this view.")
    selected_match = None
else:
    cols = du.display_columns(table, kind, show_raw_cols)
    st.dataframe(table[cols].reset_index(drop=True), width="stretch")
    match_ids = table["similar_hole_id"].tolist()
    selected_match = st.selectbox(
        "Inspect a match", match_ids,
        format_func=lambda h: f"{h} — {du.hole_label(fi.loc[h])}" if h in fi.index else h)

# --------------------------------------------------------------------------- #
# 3. Match explanation (feature comparison)
# --------------------------------------------------------------------------- #
st.subheader("3. Why are these holes similar?")
if selected_match is None:
    st.caption("Select a match above to see a side-by-side feature comparison.")
else:
    st.markdown(f"**{query_id}**  vs  **{selected_match}**")
    cmp = du.compare_features(features, query_id, selected_match)
    st.dataframe(cmp, width="stretch", hide_index=True)

# --------------------------------------------------------------------------- #
# 4. Visual comparison
# --------------------------------------------------------------------------- #
st.subheader("4. Point-cloud comparison")
compact_dir = art.get("compact_dir")
if not show_visual:
    st.caption("Enable “Show point-cloud visual” in the sidebar to render this.")
elif selected_match is None:
    st.caption("Select a match above to compare point clouds.")
elif compact_dir is None:
    st.info("This source has no compact point clouds (point-cloud visuals need the "
            "artifact's `point_clouds/compact/` folder or the local `courses/` tree).")
else:
    avail = available_compact_ids(compact_dir)
    if query_id in avail and selected_match in avail:
        fig = plot_hole_comparison(
            None, [query_id, selected_match],
            titles=[f"{query_id} (query)", selected_match],
            color_by="label", max_points=30000, compact_dir=compact_dir)
        st.pyplot(fig)
    else:
        st.info("Point-cloud visual not available in this lite artifact for one or "
                "both holes. The lite artifact includes full tabular data but only "
                "curated compact point clouds. Use the full artifact or local "
                "`courses/` tree for arbitrary-hole visuals.")

# --------------------------------------------------------------------------- #
# 5. v2.5 — surface-aware point-cloud similarity
# --------------------------------------------------------------------------- #
st.subheader("5. Similar holes — v2.5 (surface-aware point cloud)")
if not _pc_configs or pc_config is None:
    st.info(
        "v2.5 point-cloud similarity results were not found for this source. "
        "They are a **separate model family** from the v2 results above — holes are "
        "compared by their normalized per-surface point clouds (fairway / green / "
        "bunker / water / tee) using a symmetric **Chamfer** distance. Generate them "
        "with:")
    st.code(
        "python -m pipeline.modeling.pointcloud.export_similarity \\\n"
        "    --config configs/similarity/pointcloud_chamfer_v1.yaml --all --top-n 25",
        language="powershell")
else:
    st.markdown(
        f"**v2.5 · `{pc_config}` preset.** Lower **total_score** = more similar. "
        "This is a *different model* from v2 above: it scores the actual hole "
        "geometry per surface, then adds yardage/elevation penalties. Candidates are "
        "same-par and within a yardage window by construction.")

    pc_results = _load_pc_results(str(_pc_root)).get(pc_config)
    if pc_results is None:
        st.warning(f"Could not load results for preset `{pc_config}`.")
    else:
        top = pcdemo.top_matches_for_hole(pc_results, query_pc_id, top_n,
                                          config_name=pc_config)
        if top.empty:
            n_targets = len(pcdemo.available_target_holes(pc_results))
            st.warning(
                f"No v2.5 matches for `{query_pc_id}` in the `{pc_config}` preset "
                f"(this preset has results for {n_targets} target holes). The hole "
                "may lack a required surface (tee/green/fairway) or have no same-par "
                "candidate in its yardage window.")
        else:
            # Friendly labels for the candidate column.
            show = top.copy()

            def _cand_label(pc_id: str) -> str:
                fid = pcdemo.feature_id_for_pc_hole(pc_id)
                if fid in fi.index:
                    return f"{pc_id} — {du.hole_label(fi.loc[fid])}"
                return pc_id

            show.insert(2, "candidate", show["candidate_hole_id"].map(_cand_label))
            st.dataframe(show, width="stretch", hide_index=True)

            st.caption(
                "Columns: per-surface Chamfer distances (`*_score`), plus "
                "`yardage_penalty`, `elevation_penalty`, and `missing_surface_penalty`. "
                "A row dominated by `missing_surface_penalty` means the candidate is "
                "*missing a surface the target has* rather than being geometrically "
                "different.")

            # Optional v2.5 point-cloud visual: query vs its #1 v2.5 match.
            if show_visual and compact_dir is not None:
                best_pc = top.iloc[0]["candidate_hole_id"]
                best_fid = pcdemo.feature_id_for_pc_hole(best_pc)
                avail = available_compact_ids(compact_dir)
                if query_id in avail and best_fid in avail:
                    st.markdown(f"**Point clouds — `{query_pc_id}` vs #1 match `{best_pc}`**")
                    fig = plot_hole_comparison(
                        None, [query_id, best_fid],
                        titles=[f"{query_pc_id} (query)", f"{best_pc} (#1 v2.5)"],
                        color_by="label", max_points=30000, compact_dir=compact_dir)
                    st.pyplot(fig)

# --------------------------------------------------------------------------- #
# 6. v2.5 — how the presets compare for this hole
# --------------------------------------------------------------------------- #
if _pc_configs and len(_pc_configs) > 1:
    st.subheader("6. v2.5 — how the presets compare for this hole")
    pc_all = _load_pc_results(str(_pc_root))
    cmp = pcdemo.compare_configs_for_hole(pc_all, query_pc_id, top_n)

    best = cmp["best_per_config"]
    best_rows = [
        {"preset": name,
         "top_match": (v[0] if v else "—"),
         "total_score": (round(v[1], 3) if v else None)}
        for name, v in sorted(best.items())
    ]
    st.markdown("**#1 match per preset**")
    st.dataframe(pd.DataFrame(best_rows), width="stretch", hide_index=True)

    shared = cmp["shared_candidates"]
    if shared:
        st.markdown(
            "**Robust matches** — in the top-"
            f"{top_n} of *every* preset (similarity that survives re-weighting):")
        st.write(", ".join(f"`{c}`" for c in shared))
    else:
        st.caption("No candidate appears in the top-N of every preset for this hole.")

    rc = cmp["rank_comparison"]
    if not rc.empty:
        st.markdown("**Per-candidate rank across presets** (lower = better)")
        st.dataframe(rc, width="stretch", hide_index=True)

    ov = cmp["overlap"]
    if not ov.empty:
        st.markdown("**Preset agreement** — Jaccard overlap of top-N candidate sets")
        st.dataframe(ov[["config_a", "config_b", "overlap_count",
                         "union_count", "jaccard_similarity"]],
                     width="stretch", hide_index=True)
    st.caption("A preset that disagrees a lot (low Jaccard) is an *opinionated* lens "
               "— e.g. hazard-heavy re-weights bunkers/water — not a broken one.")

# --------------------------------------------------------------------------- #
# 7. Dataset summary
# --------------------------------------------------------------------------- #
with st.expander("Dataset summary"):
    s = du.dataset_summary(art)
    c1, c2, c3 = st.columns(3)
    c1.metric("Courses", s["courses"])
    c1.metric("Holes", s["holes"])
    c2.metric("Feature columns", s["feature_columns"])
    c2.metric("Presented similarity rows", s["presented_similarity_rows"])
    c3.metric("Similarity modes (v2)", len(s["similarity_modes"]))
    c3.metric("v2.5 point-cloud presets", len(_pc_configs))
    st.caption("Source: " + art.get("label", root_str)
               + (f"  ·  v2.5 results: {_pc_root}" if _pc_root else "  ·  no v2.5 results"))

# --------------------------------------------------------------------------- #
# 8. Interpretation notes
# --------------------------------------------------------------------------- #
with st.expander("How to read this"):
    st.markdown(
        "- **v2 vs v2.5 are different model families.** v2 (sections 2–4) compares "
        "~90 engineered features per hole; v2.5 (section 5) compares the normalized "
        "per-surface point clouds directly with a Chamfer distance.\n"
        "- **Presented plays-like (v2)** is the safest golfer-facing recommendation: "
        "**same-par, cross-course by default, and plausibility-filtered**.\n"
        "- **Raw overall_v2 / facet modes (v2)** are diagnostics for a single aspect "
        "(off the tee, approach, green complex, hazard, terrain, shot shape).\n"
        "- **v2.5 presets** answer 'similar full hole shape', and the hazard-heavy "
        "preset re-weights bunkers/water for a 'when hazards matter more' view. "
        "Lower v2.5 `total_score` = more similar.\n"
        "- The **lite** artifact includes **all tables for all holes**, but only a "
        "**curated** set of compact point clouds, so some visuals are skipped.\n"
        "- `sand_pct` is natural/waste sand (rarely tagged); golf bunkers are "
        "`bunker_pct`.")
