"""test_zoom.py  --  checks the preview zoom/pan maths (cubemosaic/zoom.py). Run: python tests/test_zoom.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cubemosaic.zoom import ViewTransform

WW, WH, PW, PH = 800, 600, 3000, 2000      # a 800x600 widget showing a 3:2 picture


def _close(a, b, tol=1e-6):
    return abs(a - b) < tol


def test_fit_is_centered():
    v = ViewTransform()
    left, top, w, h = v.rect(WW, WH, PW, PH)
    assert _close(w, 800) and _close(h, 800 * 2 / 3)                 # width-limited fit
    assert _close(left, 0) and _close(top, (600 - h) / 2)            # centered vertically


def test_zoom_keeps_point_under_pointer():
    v = ViewTransform()
    ax, ay = 250.0, 300.0
    before = v.fraction_at(ax, ay, WW, WH, PW, PH, clamp=False)
    v.zoom_at(4.0, ax, ay, WW, WH, PW, PH)
    after = v.fraction_at(ax, ay, WW, WH, PW, PH, clamp=False)
    assert _close(v.zoom, 4.0)
    assert _close(before[0], after[0]) and _close(before[1], after[1])   # the same picture point is under the pointer
    v.zoom_at(1.5, 600, 200, WW, WH, PW, PH)                              # and again, somewhere else
    assert _close(v.zoom, 6.0)


def test_zoom_limits_and_reset():
    v = ViewTransform()
    v.zoom_at(0.1, 100, 100, WW, WH, PW, PH)                          # cannot zoom out past fit
    assert v.zoom == 1.0 and v.cx == 0.5 and v.cy == 0.5
    v.zoom_at(1000, 400, 300, WW, WH, PW, PH)
    assert v.zoom == ViewTransform.MAX_ZOOM
    v.reset()
    assert (v.zoom, v.cx, v.cy) == (1.0, 0.5, 0.5)


def test_pan_moves_and_clamps():
    v = ViewTransform()
    v.zoom_at(4.0, 400, 300, WW, WH, PW, PH)
    left0 = v.rect(WW, WH, PW, PH)[0]
    v.pan_by(100, 0, WW, WH, PW, PH)                                  # drag right by 100 px: picture moves right 100 px
    assert _close(v.rect(WW, WH, PW, PH)[0], left0 + 100)
    v.pan_by(-10 ** 7, -10 ** 7, WW, WH, PW, PH)                      # absurd drag: center stays inside the picture
    assert 0.0 <= v.cx <= 1.0 and 0.0 <= v.cy <= 1.0
    v.reset()
    v.pan_by(50, 50, WW, WH, PW, PH)                                  # at fit zoom there is nothing to pan
    assert (v.cx, v.cy) == (0.5, 0.5)


def test_sticker_lookup():
    v = ViewTransform()                                               # 105 x 70 mosaic, picture fitted
    left, top, w, h = v.rect(WW, WH, PW, PH)
    assert v.sticker_at(left + 1, top + 1, WW, WH, PW, PH, 105, 70) == (0, 0)
    assert v.sticker_at(left + w - 1, top + h - 1, WW, WH, PW, PH, 105, 70) == (104, 69)
    assert v.sticker_at(left - 5, top + 10, WW, WH, PW, PH, 105, 70) is None      # outside the picture
    v.zoom_at(10.0, 400, 300, WW, WH, PW, PH)                                      # zoomed: lookup still follows the pointer
    fx, fy = v.fraction_at(400, 300, WW, WH, PW, PH)
    assert v.sticker_at(400, 300, WW, WH, PW, PH, 105, 70) == (int(fx * 105), int(fy * 70))


if __name__ == "__main__":
    test_fit_is_centered()
    test_zoom_keeps_point_under_pointer()
    test_zoom_limits_and_reset()
    test_pan_moves_and_clamps()
    test_sticker_lookup()
    print("zoom tests passed")
