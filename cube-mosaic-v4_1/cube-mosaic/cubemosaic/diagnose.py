"""diagnose.py  (v3)  --  "which sticker will each kind of color get?"

Prints, for YOUR palette, frame and matching settings, a table of what the matcher picks for
targets of every hue and lightness. Use it to understand and tune the blue/green behaviour BEFORE
running a photo:

    python -m cubemosaic.diagnose                         # plain v2 behaviour
    python -m cubemosaic.diagnose --hue-guard-deg 70      # see what the hue guard changes
    python -m cubemosaic.diagnose --lightness-weight 0.5 --bias G=0.004

How to read the grids
---------------------
Rows are lightness (0.25 dark ... 0.85 light). Columns are hue in degrees (rough guide:
105 yellow, 150 green, 195 cyan, 260 blue, 315 magenta-pink). Each cell shows the sticker letter
that target color receives (W Y R O B G).
  "."  = that color cannot occur in an ordinary photo, so it is skipped.
  "*"  after a letter = the result DIFFERS from plain v2 matching (so you can see exactly
         where your settings change things).
The "gray ramp" row shows what pure grays (no color at all) turn into.
"""
import argparse
import math

import numpy as np

from .cli import add_color_options, add_cube_options, add_frame_options, add_match_options, make_match_settings, make_specs
from .color import linear_to_srgb, oklab_to_lch, oklab_to_linear
from .palette import COLOR_NAMES, build_palette
from .process import Matcher

HUES = list(range(105, 316, 15))                      # columns of the grid, degrees
LIGHTNESSES = (0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85)   # rows of the grid
CHROMAS = (0.04, 0.08, 0.12)                          # one grid per colorfulness level


def pick(matcher: Matcher, lab: np.ndarray) -> int:
    """Index the matcher gives to one OKLab target."""
    return int(np.argmin(matcher.distances(lab)))


def main(argv=None):
    p = argparse.ArgumentParser(description="Show which sticker each hue/lightness gets")
    add_cube_options(p)
    add_color_options(p)
    add_frame_options(p)
    add_match_options(p)
    a = p.parse_args(argv)

    cube, frame, colors = make_specs(a)
    palette = build_palette(colors, cube, frame)
    mine = Matcher(palette, make_match_settings(a))        # your settings
    plain = Matcher(palette)                                # v2 behaviour, for the "*" marks

    print("The six colors as the eye sees them (frame blended in):")
    lightness, chroma, hue = oklab_to_lch(palette.perceived_lab)
    srgb = np.round(linear_to_srgb(palette.perceived_linear) * 255).astype(int)
    for i, name in enumerate(COLOR_NAMES):
        print(f"  {name}: lightness {lightness[i]:.2f}  chroma {chroma[i]:.2f}  hue {hue[i]:5.1f}  rgb {tuple(srgb[i])}")

    for c0 in CHROMAS:
        print(f"\nTargets with colorfulness (chroma) {c0}. Columns = hue in degrees")
        print("  light " + "".join(f"{h:5d}" for h in HUES))
        for lt in LIGHTNESSES:
            row = []
            for h in HUES:
                lab = np.array([lt, c0 * math.cos(math.radians(h)), c0 * math.sin(math.radians(h))])
                lin = oklab_to_linear(lab)
                if (lin < 0).any() or (lin > 1).any():
                    row.append("    .")                  # impossible in a photo
                    continue
                k, k_plain = pick(mine, lab), pick(plain, lab)
                row.append(f"    {COLOR_NAMES[k]}" if k == k_plain else f"   {COLOR_NAMES[k]}*")
            print(f"  {lt:.2f}  " + "".join(row))

    print("\nGray ramp (no color at all), light to dark:")
    cells = []
    for lt in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2):
        lab = np.array([lt, 0.0, 0.0])
        k, k_plain = pick(mine, lab), pick(plain, lab)
        cells.append(f"{lt:.1f}:{COLOR_NAMES[k]}{'' if k == k_plain else '*'}")
    print("  " + "  ".join(cells))


if __name__ == "__main__":
    main()
