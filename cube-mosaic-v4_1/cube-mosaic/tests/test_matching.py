"""test_matching.py  --  checks the v3 matching controls and dithering.

Run:  python tests/test_matching.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from cubemosaic.color import oklab_to_linear, linear_to_oklab
from cubemosaic.dither import dither_floyd_steinberg
from cubemosaic.geometry import CubeSpec
from cubemosaic.palette import COLOR_NAMES, DEFAULT_COLORS, FrameSpec, build_palette
from cubemosaic.process import Matcher, MatchSettings
from cubemosaic.viewing import ViewingReport, dither_strength

PAL = build_palette(DEFAULT_COLORS, CubeSpec(3, 57.0, 15.0), FrameSpec())
IDX = {n: i for i, n in enumerate(COLOR_NAMES)}


def _lab(lightness, chroma, hue_deg):
    h = np.radians(hue_deg)
    return np.array([lightness, chroma * np.cos(h), chroma * np.sin(h)])


def test_defaults_equal_plain_nearest():
    """With no settings the matcher is exactly the v2 rule: smallest plain OKLab distance."""
    rng = np.random.default_rng(1)
    target = rng.random((20, 30, 3))
    got = Matcher(PAL).match_array(target)
    lab = linear_to_oklab(target)
    d = ((lab[:, :, None, :] - PAL.perceived_lab) ** 2).sum(-1)
    assert np.array_equal(got, np.argmin(d, axis=-1))


def test_hue_guard_blocks_green_for_blue_targets():
    """A mid-light blue (hue 250, lightness 0.55) is given green by plain matching; the guard stops that."""
    lab = _lab(0.55, 0.08, 250)
    plain = int(np.argmin(Matcher(PAL).distances(lab)))
    guarded = int(np.argmin(Matcher(PAL, MatchSettings(hue_guard_deg=70)).distances(lab)))
    assert plain == IDX["G"], COLOR_NAMES[plain]            # the bug being fixed
    assert guarded == IDX["B"], COLOR_NAMES[guarded]


def test_hue_guard_ignores_grays_and_allows_white():
    """Grays have no hue so the guard leaves them alone; white is always allowed."""
    gray = np.array([0.5, 0.0, 0.0])
    assert np.isfinite(Matcher(PAL, MatchSettings(hue_guard_deg=30)).distances(gray)).all()
    pale_blue = _lab(0.9, 0.05, 250)                          # very light blue
    k = int(np.argmin(Matcher(PAL, MatchSettings(hue_guard_deg=70)).distances(pale_blue)))
    assert k == IDX["W"]


def test_bias_and_neutral_bias():
    gray = np.array([0.5, 0.0, 0.0])
    assert int(np.argmin(Matcher(PAL).distances(gray))) == IDX["G"]                   # plain: mid-gray -> green
    g_pen = (0, 0, 0, 0, 0, 0.05)
    assert int(np.argmin(Matcher(PAL, MatchSettings(bias=g_pen)).distances(gray))) != IDX["G"]
    assert int(np.argmin(Matcher(PAL, MatchSettings(neutral_bias=g_pen)).distances(gray))) != IDX["G"]
    # The neutral bias must not touch colorful targets, while the normal bias must.
    # Compared on the raw costs (not the winner): a real green is so close to the green sticker
    # that even a 0.05 penalty does not change who wins, but the cost itself must differ.
    green = _lab(0.5, 0.14, 150)
    base = Matcher(PAL).distances(green)
    with_neutral = Matcher(PAL, MatchSettings(neutral_bias=g_pen)).distances(green)
    with_bias = Matcher(PAL, MatchSettings(bias=g_pen)).distances(green)
    assert np.allclose(with_neutral, base)                      # colorful target: gray-only bias ignored
    assert abs((with_bias - base)[IDX["G"]] - 0.05) < 1e-12     # normal bias added exactly 0.05 to G


def test_dither_strength_zero_is_plain_and_valid():
    rng = np.random.default_rng(2)
    target = rng.random((16, 24, 3))
    m = Matcher(PAL)
    assert np.array_equal(dither_floyd_steinberg(target, m, 0.0), m.match_array(target))
    out = dither_floyd_steinberg(target, m, 1.0)
    assert out.shape == (16, 24) and out.max() < 6                                    # only the six colors exist


def test_dither_preserves_average_tone():
    """A flat mid color should dither to a mix whose average is closer to it than any single sticker is."""
    wanted = _lab(0.7, 0.0, 0.0)                              # a light gray the palette cannot show exactly
    target = np.broadcast_to(oklab_to_linear(wanted), (40, 40, 3)).copy()
    m = Matcher(PAL)
    plain = m.match_array(target)
    mixed = dither_floyd_steinberg(target, m, 1.0)
    avg_lin = lambda idx: PAL.perceived_linear[idx].mean(axis=(0, 1))
    err_plain = np.abs(avg_lin(plain) - target[0, 0]).sum()
    err_mixed = np.abs(avg_lin(mixed) - target[0, 0]).sum()
    assert err_mixed < err_plain * 0.5, (err_plain, err_mixed)


def test_dither_hue_guard_uses_original_color():
    """Tan sand (hue about 60) must never receive green, even when dithering error accumulates."""
    tan = oklab_to_linear(_lab(0.72, 0.06, 65))
    target = np.broadcast_to(tan, (60, 60, 3)).copy()
    m = Matcher(PAL, MatchSettings(hue_guard_deg=70))
    out = dither_floyd_steinberg(target, m, 1.0)
    assert not (out == IDX["G"]).any(), "green leaked into a tan area"


def test_dither_strength_scale():
    def rep(arcmin):
        return ViewingReport("ok", 4.0, 3.4, arcmin, "x", (), None)
    assert dither_strength(rep(2.0)) == 1.0                   # well below the 4 arcmin threshold
    assert dither_strength(rep(4.0)) == 1.0                   # exactly at it
    assert abs(dither_strength(rep(8.0)) - 0.5) < 1e-9        # ratio 2 -> halfway
    assert dither_strength(rep(12.0)) == 0.0                  # ratio 3
    assert dither_strength(rep(40.0)) == 0.0


if __name__ == "__main__":
    test_defaults_equal_plain_nearest()
    test_hue_guard_blocks_green_for_blue_targets()
    test_hue_guard_ignores_grays_and_allows_white()
    test_bias_and_neutral_bias()
    test_dither_strength_zero_is_plain_and_valid()
    test_dither_preserves_average_tone()
    test_dither_hue_guard_uses_original_color()
    test_dither_strength_scale()
    print("matching tests passed")
