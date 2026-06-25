from __future__ import annotations

import math

import pytest

from pipeline.osm.holes import parse_hole_number, parse_ref_tokens


@pytest.mark.parametrize("value,expected", [
    ("1", [1]),
    ("9;10", [9, 10]),
    ("9, 10", [9, 10]),
    ("18", [18]),
    (7, [7]),
    (None, None),
    ("None", None),
    ("", None),
    ("abc", None),
])
def test_parse_ref_tokens(value, expected):
    assert parse_ref_tokens(value) == expected


def test_parse_ref_tokens_nan():
    assert parse_ref_tokens(float("nan")) is None


def test_parse_hole_number_first_token():
    assert parse_hole_number("9;10") == 9
    assert parse_hole_number(None) is None
