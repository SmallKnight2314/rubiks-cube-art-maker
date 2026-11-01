"""test_geometry.py  --  checks the size/grid maths against numbers worked out by hand.

Run:  python tests/test_geometry.py
Prints "geometry tests passed" when everything is fine; otherwise Python shows
which `assert` failed and with what values.
"""
import os
import sys

# Make "import cubemosaic" work when this file is run directly from any folder:
# add the project root (one level above tests/) to Python's search path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cubemosaic.geometry import CubeSpec, compute_grid


def test_3x3_hand_calc():
    """A standard 3x3, worked out on paper."""
    cube = CubeSpec(3, 57.0, 15.0)
    assert abs(cube.pitch_mm - 19.0) < 1e-9                      # 57 / 3
    assert abs(cube.padding_mm - 4.0) < 1e-9                     # 19 - 15
    assert abs(cube.fill_ratio - (15 / 19) ** 2) < 1e-9          # sticker area share, about 0.62
    # 2.0 m / 57 mm = 35.08 -> 35 cubes across ; 1.5 m / 57 mm = 26.3 -> 26 cubes down.
    # A 4:3 image has exactly the shape 35 x 26.25, so 35 x 26 should win.
    g = compute_grid(4000, 3000, cube, 2.0, 1.5)
    assert (g.cubes_x, g.cubes_y) == (35, 26), (g.cubes_x, g.cubes_y)
    assert (g.px_x, g.px_y) == (105, 78)                         # 35*3 by 26*3 stickers
    assert g.width_mm <= 2000 and g.height_mm <= 1500            # inside the limits


def test_limits_never_exceeded():
    """For every cube type and several image shapes the mosaic must fit the box."""
    for n, mm in ((2, 50), (4, 62), (5, 63), (6, 68), (7, 70)):
        cube = CubeSpec(n, mm, mm / n * 0.8)
        for w, h in ((4000, 3000), (3000, 4000), (1000, 1000), (6000, 1500)):   # landscape, portrait, square, panorama
            g = compute_grid(w, h, cube, 1.2, 0.9)
            assert g.width_mm <= 1200 + 1e-6 and g.height_mm <= 900 + 1e-6, (n, w, h, g)


def test_sticker_larger_than_pitch_rejected():
    """A sticker wider than its cell is impossible and must raise an error."""
    try:
        CubeSpec(3, 57.0, 20.0)      # pitch is only 19 mm
    except ValueError:
        return                       # good: the error we wanted
    raise AssertionError("expected ValueError")


if __name__ == "__main__":
    test_3x3_hand_calc()
    test_limits_never_exceeded()
    test_sticker_larger_than_pitch_rejected()
    print("geometry tests passed")
