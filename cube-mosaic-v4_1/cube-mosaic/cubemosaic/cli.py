"""cli.py  (v0/v1, viewing v2, matching controls + dithering + compare v3)  --  command-line front end.

This file only wires the other modules together, in this order:

    read options -> load photo -> grid -> palette -> viewing-distance check
    -> crop + shrink the photo -> match colors (plain or dithered) -> draw pictures -> summary

Nothing clever happens here; all real work is in the other files. When the GUI (v4) arrives it
will call the same functions in the same order.

Example:
    python -m cubemosaic.cli photo.jpg --n 3 --max-w-m 2.0 --max-h-m 1.5 --view-m 4 --out-dir out

Trying to fix blue/green mix-ups (see NOTES.md) and comparing against plain matching:
    python -m cubemosaic.cli photo.jpg --max-w-m 2 --max-h-m 1.5 --hue-guard-deg 70 --compare
"""
import argparse
import os

import numpy as np

from .compare import describe_transitions, render_changed_cells, side_by_side, transition_counts
from .dither import dither_floyd_steinberg
from .geometry import CUBE_PRESETS_MM, CubeSpec, compute_grid, default_sticker_mm
from .palette import (COLOR_NAMES, DEFAULT_COLORS, FRAME_FRAMELESS, FRAME_OPAQUE,
                      FRAME_TRANSLUCENT, FrameSpec, build_palette)
from .process import Matcher, MatchSettings, crop_to_aspect, load_image, target_linear
from .render import add_banner, render_perceived, render_preview
from .viewing import check_viewing, dither_strength, estimate_viewing
from .flip import parse_flip
from .pipeline import Settings, run_pipeline, save_outputs, summary_lines


# --- option groups -------------------------------------------------------------------------
# They are separate functions so diagnose.py can reuse exactly the same options for the cube,
# colors, frame and matching without copying them.

def add_cube_options(p):
    """Cube type and sizes."""
    p.add_argument("--n", type=int, default=3, choices=sorted(CUBE_PRESETS_MM), help="cube type (n x n x n)")
    p.add_argument("--cube-mm", type=float, help="cube edge in mm (default: preset for the cube type)")
    p.add_argument("--sticker-mm", type=float, help="sticker edge in mm (default: about 79%% of the pitch)")


def add_color_options(p):
    """The six sticker colors. Creates --color-w, --color-y, --color-r, --color-o, --color-b, --color-g."""
    for name in COLOR_NAMES:
        p.add_argument(f"--color-{name.lower()}", default=DEFAULT_COLORS[name], help=f"hex for {name}")


def add_frame_options(p):
    """Frame plastic around the stickers."""
    p.add_argument("--frame-mode", choices=[FRAME_OPAQUE, FRAME_TRANSLUCENT, FRAME_FRAMELESS], default=FRAME_OPAQUE)
    p.add_argument("--frame-hex", default="#000000")
    p.add_argument("--frame-opacity", type=float, default=1.0, help="translucent mode: 0..1")
    p.add_argument("--backdrop-hex", default="#000000", help="translucent mode: color behind the frame")
    p.add_argument("--seam-mm", type=float, default=0.6, help="frameless mode: visible gap between pieces")
    p.add_argument("--seam-hex", default="#000000")


def add_match_options(p):
    """Controls for which color wins (all default to off = the v2 result)."""
    p.add_argument("--lightness-weight", type=float, default=1.0,
                   help="how much lightness differences count vs color differences (1 = v2; try 0.4 to 0.7)")
    p.add_argument("--hue-guard-deg", type=float, default=0.0,
                   help="a colorful target may only get a sticker within this many degrees of hue "
                        "(0 = off; try 60 to 75). Stops blue targets getting green and the reverse")
    p.add_argument("--guard-min-chroma", type=float, default=0.04,
                   help="the hue guard ignores targets less colorful than this (near-grays)")
    p.add_argument("--bias", action="append", default=[], metavar="COLOR=VALUE",
                   help="make a color less (positive) or more (negative) likely, e.g. --bias G=0.004. "
                        "Can be repeated. 0.002 is gentle, 0.01 strong")
    p.add_argument("--neutral-bias", action="append", default=[], metavar="COLOR=VALUE",
                   help="same as --bias but only for near-gray targets, e.g. --neutral-bias G=0.02 "
                        "stops gray mid-tones turning green (they go to the next best color instead)")


def make_specs(a):
    """Turn parsed options into the cube, frame and color objects the other modules use."""
    # "x or y" = use x unless it is missing/zero, then use y.
    cube_mm = a.cube_mm or CUBE_PRESETS_MM[a.n]
    sticker_mm = a.sticker_mm or default_sticker_mm(a.n, cube_mm)
    cube = CubeSpec(a.n, cube_mm, sticker_mm)       # validates the sizes (raises a clear error if impossible)
    frame = FrameSpec(a.frame_mode, a.frame_hex, a.frame_opacity, a.backdrop_hex, a.seam_mm, a.seam_hex)
    # Collect the six color options into {"W": "#FFFFFF", "Y": ..., ...}.
    colors = {name: getattr(a, f"color_{name.lower()}") for name in COLOR_NAMES}
    return cube, frame, colors


def _parse_bias(items, flag: str) -> tuple:
    """Turn ["G=0.004", "R=-0.002"] into the 6-number tuple (W, Y, R, O, B, G order)."""
    bias = [0.0] * 6
    for item in items:
        name, _, value = item.partition("=")                       # "G=0.004" -> "G", "=", "0.004"
        name = name.strip().upper()
        if name not in COLOR_NAMES or not value:
            raise SystemExit(f"{flag} expects COLOR=VALUE with COLOR one of {', '.join(COLOR_NAMES)}; got {item!r}")
        bias[COLOR_NAMES.index(name)] += float(value)              # repeated entries for one color add up
    return tuple(bias)


def make_match_settings(a) -> MatchSettings:
    """Turn the matching options into a MatchSettings."""
    return MatchSettings(a.lightness_weight, a.hue_guard_deg, a.guard_min_chroma,
                         _parse_bias(a.bias, "--bias"), _parse_bias(a.neutral_bias, "--neutral-bias"))


def parse_args(argv=None):
    """Declare every command-line option and read what the user typed."""
    p = argparse.ArgumentParser(description="Rubik's cube mosaic generator (v4: flips on top of v3 matching + dithering)")
    p.add_argument("image", help="path to the photo (png, jpg, jpeg, ...)")
    add_cube_options(p)
    p.add_argument("--max-w-m", type=float, required=True, help="maximum mosaic width in meters")
    p.add_argument("--max-h-m", type=float, required=True, help="maximum mosaic height in meters")
    add_color_options(p)
    add_frame_options(p)
    add_match_options(p)

    # --- image placement --------------------------------------------------------
    p.add_argument("--crop-x", type=float, default=0.5, help="crop window position 0..1 (horizontal)")
    p.add_argument("--crop-y", type=float, default=0.5, help="crop window position 0..1 (vertical)")

    # --- viewing distance (v2) ----------------------------------------------------
    p.add_argument("--view-m", type=float, default=None,
                   help="planned viewing distance in meters; checked against the recommended range "
                        "(omit it to just use the recommendation)")

    # --- dithering (v3) -----------------------------------------------------------
    p.add_argument("--dither", choices=["auto", "off", "on"], default="auto",
                   help="auto = strength from the viewing distance (none when stickers are clearly "
                        "visible, full when they merge); off = plain nearest color; on = full strength")
    p.add_argument("--dither-strength", type=float, default=None,
                   help="exact dithering strength 0..1 (overrides --dither)")

    # --- comparison (v3) ----------------------------------------------------------
    p.add_argument("--flip", action="append", default=[], metavar="X=Y[@x0,y0,x1,y1]",
                   help="exact color flip after matching: every X sticker becomes Y, optionally only inside a "
                        "rectangle (sticker coordinates, x1/y1 exclusive). Repeatable. B=G together with G=B swaps "
                        "the two colors. Example: --flip B=G --flip G=R@0,0,40,20")
    p.add_argument("--compare", action="store_true",
                   help="also write compare.png: plain v2 result vs your settings vs a map of what changed")

    p.add_argument("--out-dir", default="out", help="folder for the output files (created if missing)")
    return p.parse_args(argv)


def main(argv=None):
    """Read the options, hand them to pipeline.run_pipeline (the same code the GUI uses), save, print."""
    a = parse_args(argv)
    cube, frame, colors = make_specs(a)                          # validated cube, frame and color objects
    try:
        flips = [parse_flip(t) for t in a.flip]                  # "B=G" -> FlipRule
    except ValueError as err:
        raise SystemExit(str(err))
    settings = Settings(
        image_path=a.image, n=a.n, cube_mm=cube.cube_mm, sticker_mm=cube.sticker_mm,
        max_w_m=a.max_w_m, max_h_m=a.max_h_m, colors=colors, frame=frame, match=make_match_settings(a),
        crop_x=a.crop_x, crop_y=a.crop_y, view_m=a.view_m, dither=a.dither,
        dither_strength=a.dither_strength, flips=flips)

    result = run_pipeline(settings)
    names, comparison = save_outputs(result, a.out_dir, a.compare)

    for line in summary_lines(result):
        print(line)
    if comparison:
        print("Compared with plain v2 matching:")
        for line in comparison:
            print("  " + line)
    print(f"Wrote {a.out_dir}/" + ", ".join(names))


if __name__ == "__main__":
    main()
