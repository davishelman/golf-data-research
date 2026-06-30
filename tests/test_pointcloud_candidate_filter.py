"""Tests for v2.5 candidate filtering (par / surface / yardage gates)."""

from __future__ import annotations

from pipeline.modeling.pointcloud.candidate_filter import (
    REASON_DIFFERENT_PAR,
    REASON_MISSING_REQUIRED_SURFACE,
    REASON_PASS,
    REASON_YARDAGE_TOO_DIFFERENT,
    filter_candidate,
)
from pipeline.modeling.pointcloud.config import config_from_dict
from pipeline.modeling.pointcloud.schemas import HoleMetadata, make_pc_hole_id
from tests.test_pointcloud_config import _valid_payload


def _config():
    return config_from_dict(_valid_payload())


def _hole(slug="course_a", number=1, par=4, yards=440.0, **flags):
    base = dict(has_tee=True, has_green=True, has_fairway=True,
                has_bunker=True, has_water=True)
    base.update(flags)
    return HoleMetadata(
        hole_id=make_pc_hole_id(slug, number),
        course_slug=slug, hole_number=number, par=par, yards=yards, **base,
    )


# --- 3. different par fails ------------------------------------------------ #

def test_different_par_fails():
    target = _hole(par=4, yards=440)
    candidate = _hole(slug="course_b", par=5, yards=440)
    result = filter_candidate(target, candidate, _config())
    assert result.passed is False
    assert result.reason == REASON_DIFFERENT_PAR


# --- 4. yardage too different fails ---------------------------------------- #

def test_yardage_too_different_fails():
    # par-4 window: max(40, 440 * 0.10) = 44 yards. 500 - 440 = 60 > 44.
    target = _hole(par=4, yards=440)
    candidate = _hole(slug="course_b", par=4, yards=500)
    result = filter_candidate(target, candidate, _config())
    assert result.passed is False
    assert result.reason == REASON_YARDAGE_TOO_DIFFERENT


# --- 5. same par + valid yardage passes ------------------------------------ #

def test_same_par_valid_yardage_passes():
    target = _hole(par=4, yards=440)
    candidate = _hole(slug="course_b", par=4, yards=470)  # diff 30 <= 44
    result = filter_candidate(target, candidate, _config())
    assert result.passed is True
    assert result.reason == REASON_PASS


def test_yardage_uses_more_permissive_of_abs_or_pct():
    # Short par 3: abs window (25) dominates the percentage window (0.12 * 180 = 21.6).
    target = _hole(par=3, yards=180)
    near = _hole(slug="course_b", par=3, yards=204)   # diff 24 <= 25 -> pass
    far = _hole(slug="course_b", par=3, yards=210)    # diff 30 > 25 -> fail
    cfg = _config()
    assert filter_candidate(target, near, cfg).passed is True
    assert filter_candidate(target, far, cfg).passed is False


# --- 6. missing required surface fails ------------------------------------- #

def test_missing_required_surface_on_candidate_fails():
    target = _hole(par=4, yards=440)
    candidate = _hole(slug="course_b", par=4, yards=445, has_fairway=False)
    result = filter_candidate(target, candidate, _config())
    assert result.passed is False
    assert result.reason == REASON_MISSING_REQUIRED_SURFACE


def test_missing_required_surface_on_target_fails():
    target = _hole(par=4, yards=440, has_green=False)
    candidate = _hole(slug="course_b", par=4, yards=445)
    result = filter_candidate(target, candidate, _config())
    assert result.passed is False
    assert result.reason == REASON_MISSING_REQUIRED_SURFACE


def test_missing_non_required_surface_still_passes():
    # bunker/water are NOT in required_surfaces -> their absence does not fail.
    target = _hole(par=4, yards=440)
    candidate = _hole(slug="course_b", par=4, yards=445, has_bunker=False, has_water=False)
    result = filter_candidate(target, candidate, _config())
    assert result.passed is True
