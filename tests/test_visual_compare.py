"""Offline tests for the visual comparison layer (synthetic compact points)."""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")  # headless; no display needed

import pytest

from pipeline.constants import LABEL_NAMES
from pipeline.modeling.visual_compare import (
    compact_points_to_df,
    compute_shared_limits,
    downsample_points,
    load_compact_hole_points,
    plot_hole_comparison,
    save_hole_comparison,
)


def _payload(slug: str, num: int, points: list) -> dict:
    return {
        "schema_version": "1.0.0",
        "hole_id": f"{slug}__{num:02d}",
        "course_slug": slug,
        "hole_number": num,
        "coordinate_system": "tee_relative_aligned_meters",
        "origin": {},
        "alignment": {"enabled": True, "axis": "+Y_toward_green", "rotation_degrees": 0.0},
        "label_map": {str(i): n for i, n in LABEL_NAMES.items()},
        "points": points,
    }


def _write_hole(courses_root, slug, num, points) -> None:
    d = courses_root / slug / "holes" / f"hole_{num:02d}" / "features"
    d.mkdir(parents=True, exist_ok=True)
    (d / "hole_points_compact.json").write_text(json.dumps(_payload(slug, num, points)),
                                                encoding="utf-8")


def _grid(label_id, x0, x1, y0, y1, step=2):
    pts = []
    x = x0
    while x <= x1:
        y = y0
        while y <= y1:
            pts.append([float(x), float(y), 0.0, int(label_id)])
            y += step
        x += step
    return pts


@pytest.fixture
def courses_root(tmp_path):
    root = tmp_path / "courses"
    a = _grid(3, -10, 10, 0, 100) + _grid(2, -3, 3, 95, 100) + _grid(5, 12, 16, 80, 90)
    _write_hole(root, "alpha", 1, a)
    b = _grid(3, -8, 8, 0, 120) + _grid(2, -3, 3, 115, 120)
    _write_hole(root, "beta", 1, b)
    return root


def test_load_and_label_mapping(courses_root):
    df = load_compact_hole_points(courses_root, "alpha__01")
    assert list(df.columns) == ["x", "y", "z", "label_id", "label"]
    assert (df[df["label_id"] == 3]["label"] == "fairway").all()
    assert {"green", "bunker"}.issubset(set(df["label"]))


def test_compact_points_to_df_empty():
    df = compact_points_to_df({"points": [], "label_map": {}})
    assert df.empty
    assert "label" in df.columns


def test_downsample():
    df = compact_points_to_df(_payload("alpha", 1, _grid(3, -20, 20, 0, 200)))
    assert len(df) > 100
    assert len(downsample_points(df, 10)) == 10
    assert len(downsample_points(df, None)) == len(df)       # None -> no downsample
    assert len(downsample_points(df, 10_000_000)) == len(df)  # cap above size -> unchanged


def test_shared_limits_covers_all(courses_root):
    da = load_compact_hole_points(courses_root, "alpha__01")
    db = load_compact_hole_points(courses_root, "beta__01")
    (xlo, xhi), (ylo, yhi) = compute_shared_limits([da, db])
    assert xlo <= min(da["x"].min(), db["x"].min())
    assert xhi >= max(da["x"].max(), db["x"].max())
    assert ylo <= min(da["y"].min(), db["y"].min())
    assert yhi >= max(da["y"].max(), db["y"].max())


def test_comparison_uses_shared_axes(courses_root):
    import matplotlib.pyplot as plt
    fig = plot_hole_comparison(courses_root, ["alpha__01", "beta__01"], color_by="label")
    ax0, ax1 = fig.axes[0], fig.axes[1]
    assert ax0.get_xlim() == ax1.get_xlim()
    assert ax0.get_ylim() == ax1.get_ylim()
    plt.close(fig)


def test_save_creates_image(courses_root, tmp_path):
    out = tmp_path / "vc" / "cmp.png"
    p = save_hole_comparison(courses_root, ["alpha__01", "beta__01"], out,
                             color_by="label", max_points=5000)
    assert p.exists() and p.stat().st_size > 0


def test_missing_hole_raises(courses_root):
    with pytest.raises(FileNotFoundError):
        load_compact_hole_points(courses_root, "nope__09")
