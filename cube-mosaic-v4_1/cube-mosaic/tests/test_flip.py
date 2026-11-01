"""test_flip.py  --  checks the exact color flip, the shared pipeline and the region maths (no screen needed).

Run:  python tests/test_flip.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from PIL import Image

from cubemosaic.flip import FlipRule, apply_flips, parse_flip
from cubemosaic.palette import COLOR_NAMES
from cubemosaic.pipeline import (SELECTABLE_VIEWS, VIEWS, Settings, region_from_fractions, render_view,
                                 run_pipeline, save_outputs, summary_lines)

W_, Y_, R_, O_, B_, G_ = range(6)


def _grid():
    """A 4 x 6 test mosaic: row 0 all B, row 1 all G, row 2 all R, rows 3-5 mixed W/Y."""
    a = np.zeros((6, 4), dtype=np.uint8)
    a[0], a[1], a[2] = B_, G_, R_
    a[3:] = np.array([[W_, Y_, W_, Y_]] * 3)
    return a


def test_parse():
    assert parse_flip("B=G") == FlipRule(B_, G_, None)
    assert parse_flip("g=b@1,2,3,4") == FlipRule(G_, B_, (1, 2, 3, 4))
    assert parse_flip("B=G@1,2,3,4").to_text() == "B=G@1,2,3,4"
    assert parse_flip("*=G@1,2,3,4") == FlipRule(None, G_, (1, 2, 3, 4))                 # paint rule
    assert parse_flip("any=R") == FlipRule(None, R_, None)
    assert parse_flip("*=G@1,2,3,4").to_text() == "*=G@1,2,3,4"
    assert "any" in parse_flip("*=G").label()
    for bad in ("B", "B=Q", "*=*", "B=G@1,2,3", "B=G@3,0,1,5", "B=G@a,b,c,d"):
        try:
            parse_flip(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} should be rejected")


def test_exact_flip_and_input_untouched():
    a = _grid()
    before = a.copy()
    out = apply_flips(a, [FlipRule(B_, G_)])
    assert (out[0] == G_).all() and (out[1] == G_).all()           # blue row is now green, exactly
    assert (out[2] == R_).all() and np.array_equal(out[3:], a[3:])  # nothing else moved
    assert np.array_equal(a, before)                                # the input was not modified


def test_swap_does_not_chain():
    out = apply_flips(_grid(), [FlipRule(B_, G_), FlipRule(G_, B_)])
    assert (out[0] == G_).all() and (out[1] == B_).all()            # B and G traded places
    out = apply_flips(_grid(), [FlipRule(B_, G_), FlipRule(G_, R_)])
    assert (out[0] == G_).all(), "B->G followed by G->R must not turn blue into red"
    assert (out[1] == R_).all()


def test_region_disabled_and_later_wins():
    a = _grid()
    out = apply_flips(a, [FlipRule(B_, G_, (1, 0, 3, 1))])          # columns 1 and 2 of row 0 only
    assert list(out[0]) == [B_, G_, G_, B_]
    assert np.array_equal(apply_flips(a, [FlipRule(B_, G_, enabled=False)]), a)
    out = apply_flips(a, [FlipRule(B_, G_), FlipRule(B_, R_)])      # both claim the blue row; later wins
    assert (out[0] == R_).all()
    assert np.array_equal(apply_flips(a, [FlipRule(B_, G_, (50, 50, 60, 60))]), a)   # outside = nothing
    out = apply_flips(a, [FlipRule(B_, G_, (-5, -5, 2, 2))])        # sticking out is clipped, not an error
    assert list(out[0]) == [G_, G_, B_, B_]


def test_paint_any_color():
    a = _grid()
    out = apply_flips(a, [FlipRule(None, O_, (1, 3, 2, 4))])        # paint ONE sticker (column 1, row 3) orange
    assert out[3, 1] == O_ and int((out != a).sum()) == 1            # exactly that one sticker changed
    out = apply_flips(a, [FlipRule(None, Y_, (0, 0, 4, 2))])        # paint the B row and G row yellow
    assert (out[:2] == Y_).all() and np.array_equal(out[2:], a[2:])
    out = apply_flips(a, [FlipRule(None, R_), FlipRule(None, W_, (0, 0, 1, 1))])   # later rule wins where both apply
    assert out[0, 0] == W_ and out[0, 1] == R_
    out = apply_flips(a, [FlipRule(None, G_), FlipRule(B_, W_)])    # rules still read the ORIGINAL, never each other
    assert out[0, 0] == W_ and out[1, 0] == G_


def test_region_from_fractions():
    assert region_from_fractions(0.0, 0.0, 1.0, 1.0, 105, 78) == (0, 0, 105, 78)
    assert region_from_fractions(0.5, 0.5, 0.0, 0.0, 10, 10) == (0, 0, 5, 5)          # corners in any order
    assert region_from_fractions(0.31, 0.0, 0.52, 1.0, 100, 50) == (31, 0, 52, 50)
    x0, y0, x1, y1 = region_from_fractions(0.42, 0.42, 0.42, 0.42, 10, 10)             # a click, no drag
    assert (x1 - x0, y1 - y0) == (1, 1)                                                 # still one sticker
    assert region_from_fractions(-1, -1, 2, 2, 10, 10) == (0, 0, 10, 10)               # clamped to the picture


def _scene(tmp):
    """Left half blue-ish, right half green-ish photo, so both colors surely appear."""
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    img[:, :200] = (0, 70, 173)
    img[:, 200:] = (0, 155, 72)
    path = os.path.join(tmp, "halves.png")
    Image.fromarray(img).save(path)
    return path


def test_pipeline_flip_and_outputs():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        s = Settings(image_path=_scene(tmp), max_w_m=0.6, max_h_m=0.45)
        r = run_pipeline(s)
        n_b, n_g = int((r.indices == B_).sum()), int((r.indices == G_).sum())
        assert n_b > 0 and n_g > 0 and np.array_equal(r.indices, r.matched)

        swapped = r.with_flips([FlipRule(B_, G_), FlipRule(G_, B_)])                    # instant, no new run
        assert int((swapped.indices == B_).sum()) == n_g and int((swapped.indices == G_).sum()) == n_b
        assert np.array_equal(swapped.matched, r.matched)                               # matched part untouched
        assert any(line.startswith("Flips:") for line in summary_lines(swapped))
        assert "Stickers: " in "\n".join(summary_lines(swapped))

        names, _ = save_outputs(swapped, os.path.join(tmp, "out"), compare=True)
        assert {"preview.png", "indices.npy", "flips.txt", "indices_before_flips.npy", "compare.png"} <= set(names)
        saved = np.load(os.path.join(tmp, "out", "indices.npy"))
        assert np.array_equal(saved, swapped.indices)                                   # files hold the FLIPPED result
        with open(os.path.join(tmp, "out", "flips.txt")) as fh:
            assert fh.read().split() == ["B=G", "G=B"]

        for view in VIEWS:                                                              # every view draws something
            img = render_view(swapped, view)
            assert img.width > 0 and img.height > 0, view
        for view in SELECTABLE_VIEWS:
            assert render_view(swapped, view).size[0] > 0


if __name__ == "__main__":
    test_parse()
    test_exact_flip_and_input_untouched()
    test_swap_does_not_chain()
    test_region_disabled_and_later_wins()
    test_paint_any_color()
    test_region_from_fractions()
    test_pipeline_flip_and_outputs()
    print("flip tests passed")
