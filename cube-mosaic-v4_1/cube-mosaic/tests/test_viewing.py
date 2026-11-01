"""test_viewing.py  --  checks the viewing-distance maths against hand calculations.

Run:  python tests/test_viewing.py

Hand calculation used below: a 3x3 mosaic 2.0 m x 1.5 m (diagonal 2.5 m), pitch 19 mm.
  best distance : 2.5 / (2 * tan(20 deg)) = 3.434 m   (diagonal covers 40 deg)
  near limit    : 2.5 / (2 * tan(30 deg)) = 2.165 m   (60 deg)
  far limit     : 2.5 / (2 * tan(10 deg)) = 7.088 m   (20 deg)
  blend distance: 0.019 / tan(4 arcmin)   = 16.33 m
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cubemosaic.viewing import check_viewing, estimate_viewing


def _est():
    return estimate_viewing(2000.0, 1500.0, 19.0)


def test_estimate_matches_hand_calc():
    e = _est()
    assert abs(e.best_m - 3.434) < 0.005, e.best_m
    assert abs(e.near_m - 2.165) < 0.005, e.near_m
    assert abs(e.far_m - 7.088) < 0.005, e.far_m
    assert abs(e.blend_m - 16.33) < 0.05, e.blend_m
    assert e.near_m < e.best_m < e.far_m        # the three distances must be ordered


def test_levels():
    e = _est()
    assert check_viewing(e, 19.0, None).level == "info"      # no distance given
    assert check_viewing(e, 19.0, 4.0).level == "ok"         # inside 2.2 - 7.1 m
    assert check_viewing(e, 19.0, 1.0).level == "clash"      # too close
    assert check_viewing(e, 19.0, 12.0).level == "clash"     # too far


def test_clash_text_names_the_recommendation():
    e = _est()
    r = check_viewing(e, 19.0, 1.0)
    assert "3.4 m" in r.lines[0], r.lines[0]                 # the note must recommend our best distance
    assert r.recommended_m == e.best_m


def test_blend_state():
    e = _est()
    # one 19 mm cell from 4 m = atan(0.019/4) = 16.3 arcmin -> ratio 4.1 -> grainy
    assert check_viewing(e, 19.0, 4.0).blend_state == "grainy"
    # from 20 m = 3.3 arcmin -> ratio 0.8 -> blended
    assert check_viewing(e, 19.0, 20.0).blend_state == "blended"


def test_bad_distance_rejected():
    try:
        check_viewing(_est(), 19.0, 0.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


if __name__ == "__main__":
    test_estimate_matches_hand_calc()
    test_levels()
    test_clash_text_names_the_recommendation()
    test_blend_state()
    test_bad_distance_rejected()
    print("viewing tests passed")
