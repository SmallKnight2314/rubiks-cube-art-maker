"""pipeline.py  (v4)  --  the ONE place where a run is put together. Used by the command line AND the GUI.

    Settings  --run_pipeline-->  Result  --render_view / save_outputs-->  pictures and files

Nothing in this file knows about Qt or argparse, so it can be tested without a screen.

Settings   everything a run needs (image, cube, colors, frame, matching rules, viewing, flips).
Result     everything a run produced. `matched` is the result BEFORE flips, `indices` AFTER flips;
           the preview, the counts and (v5) the blueprint must always use `indices`.
Changing only the flip rules does not need a new run: Result.with_flips() re-applies them instantly.
"""
from dataclasses import dataclass, field, replace
import os

import numpy as np
from PIL import Image

from .compare import describe_transitions, render_changed_cells, side_by_side, transition_counts
from .dither import dither_floyd_steinberg
from .flip import FlipRule, apply_flips
from .geometry import CUBE_PRESETS_MM, CubeSpec, MosaicGrid, compute_grid, default_sticker_mm
from .palette import COLOR_NAMES, DEFAULT_COLORS, FrameSpec, Palette, build_palette
from .process import Matcher, MatchSettings, crop_to_aspect, load_image, target_linear
from .render import add_banner, render_perceived, render_preview
from .viewing import ViewingReport, check_viewing, dither_strength, estimate_viewing


@dataclass
class Settings:
    """Every input of a run. Defaults reproduce the plain v2 behaviour."""
    image_path: str
    n: int = 3                                   # cube type (2..7)
    cube_mm: float = None                        # None = preset for the cube type
    sticker_mm: float = None                     # None = about 79% of the pitch
    max_w_m: float = 2.0
    max_h_m: float = 1.5
    colors: dict = field(default_factory=lambda: dict(DEFAULT_COLORS))   # {"W": "#FFFFFF", ...}
    frame: FrameSpec = field(default_factory=FrameSpec)
    match: MatchSettings = field(default_factory=MatchSettings)
    crop_x: float = 0.5
    crop_y: float = 0.5
    view_m: float = None                         # None = use the recommended distance
    dither: str = "auto"                         # "auto" | "off" | "on"
    dither_strength: float = None                # explicit 0..1 overrides `dither`
    flips: list = field(default_factory=list)    # list of FlipRule

    def without_flips(self) -> "Settings":
        """Same settings with no flips: used to decide whether a cached Result is still valid."""
        return replace(self, flips=[])


@dataclass
class Result:
    settings: Settings
    cube: CubeSpec
    grid: MosaicGrid
    palette: Palette
    report: ViewingReport
    strength: float                              # dithering strength actually used
    why: str                                     # one line explaining where that strength came from
    cropped: Image.Image                         # the photo after cropping to the grid's shape
    matched: np.ndarray                          # (H, W) 0..5 BEFORE flips
    baseline: np.ndarray                         # plain v2 matching, for the comparison pictures
    flips: list                                  # FlipRule list in use
    indices: np.ndarray                          # (H, W) 0..5 AFTER flips: the real answer

    def with_flips(self, rules) -> "Result":
        """A copy with `rules` applied to the matched result. Cheap: no image processing."""
        rules = list(rules)
        return replace(self, flips=rules, indices=apply_flips(self.matched, rules))

    @property
    def frame(self) -> FrameSpec:
        return self.settings.frame


def resolve_dither(s: Settings, report: ViewingReport):
    """Dithering strength and a one-line reason: explicit value > on/off > automatic from viewing."""
    if s.dither_strength is not None:
        return min(1.0, max(0.0, s.dither_strength)), "set by --dither-strength"
    if s.dither == "on":
        return 1.0, "dither on"
    if s.dither == "off":
        return 0.0, "dither off"
    return dither_strength(report), f"auto, stickers are '{report.blend_state}' from {report.user_m:.1f} m"


def run_pipeline(s: Settings) -> Result:
    """Photo in, mosaic out. Raises ValueError with a readable message for impossible input."""
    cube_mm = s.cube_mm or CUBE_PRESETS_MM[s.n]
    sticker_mm = s.sticker_mm or default_sticker_mm(s.n, cube_mm)
    cube = CubeSpec(s.n, cube_mm, sticker_mm)                      # validates the sizes

    img = load_image(s.image_path)
    grid = compute_grid(img.width, img.height, cube, s.max_w_m, s.max_h_m)
    palette = build_palette(s.colors, cube, s.frame)

    est = estimate_viewing(grid.width_mm, grid.height_mm, cube.pitch_mm)
    report = check_viewing(est, cube.pitch_mm, s.view_m)
    strength, why = resolve_dither(s, report)

    cropped = crop_to_aspect(img, grid.px_x / grid.px_y, s.crop_x, s.crop_y)
    target = target_linear(cropped, grid.px_x, grid.px_y)
    matched = dither_floyd_steinberg(target, Matcher(palette, s.match), strength)
    baseline = Matcher(palette).match_array(target)                # default rules, no dithering = v2
    assert matched.shape == (grid.px_y, grid.px_x) and matched.max() < 6   # only the six colors exist

    result = Result(s, cube, grid, palette, report, strength, why, cropped, matched, baseline, [], matched)
    return result.with_flips(s.flips)


# --- text summary --------------------------------------------------------------------------

def summary_lines(r: Result) -> list:
    """The run's facts as plain text lines (command line prints them, the GUI shows some of them)."""
    g, c, m = r.grid, r.cube, r.settings.match
    counts = np.bincount(r.indices.ravel(), minlength=6)           # stickers of each color, AFTER flips
    lines = [
        f"Cube: {c.n}x{c.n}, {c.cube_mm:g} mm, pitch {c.pitch_mm:.2f} mm, sticker {c.sticker_mm:.2f} mm, "
        f"padding {c.padding_mm:.2f} mm, fill {r.palette.fill_ratio:.2f}",
        f"Grid: {g.cubes_x} x {g.cubes_y} cubes = {g.cube_count} cubes ({g.px_x} x {g.px_y} stickers)",
        f"Size: {g.width_mm / 1000:.3f} m x {g.height_mm / 1000:.3f} m "
        f"(limit {r.settings.max_w_m:g} x {r.settings.max_h_m:g} m), crop loss {g.crop_loss * 100:.1f}%",
        "Stickers: " + ", ".join(f"{n} {k}" for n, k in zip(COLOR_NAMES, counts)),
        f"Matching: lightness weight {m.lightness_weight:g}, hue guard {m.hue_guard_deg:g} deg, "
        f"bias {dict(zip(COLOR_NAMES, m.bias))}, gray-only bias {dict(zip(COLOR_NAMES, m.neutral_bias))}",
        f"Dithering: strength {r.strength:.2f} ({r.why})",
    ]
    active = [f for f in r.flips if f.enabled and f.src != f.dst]
    if active:
        changed = int((r.indices != r.matched).sum())
        lines.append(f"Flips: {changed} stickers changed by {len(active)} rule(s): "
                     + "; ".join(f.label() for f in active))
    lines.append(f"Viewing [{r.report.level}]:")
    lines += ["  " + t for t in r.report.lines]
    return lines


# --- region helper for the GUI (kept here so it can be tested without Qt) ---------------------

def region_from_fractions(fx0, fy0, fx1, fy1, width_px: int, height_px: int):
    """A dragged rectangle (as 0..1 fractions of the picture) -> sticker coordinates (x0, y0, x1, y1).

    The corners may arrive in any order. Floor the start and ceil the end, so every sticker the
    rectangle touches is included, and make it at least one sticker big.
    """
    fx0, fx1 = sorted((min(max(fx0, 0.0), 1.0), min(max(fx1, 0.0), 1.0)))
    fy0, fy1 = sorted((min(max(fy0, 0.0), 1.0), min(max(fy1, 0.0), 1.0)))
    x0, x1 = int(np.floor(fx0 * width_px)), int(np.ceil(fx1 * width_px))
    y0, y1 = int(np.floor(fy0 * height_px)), int(np.ceil(fy1 * height_px))
    x1, y1 = max(x1, x0 + 1), max(y1, y0 + 1)
    return (min(x0, width_px - 1), min(y0, height_px - 1), min(x1, width_px), min(y1, height_px))


# --- pictures ----------------------------------------------------------------------------------

VIEWS = ("Build view", "From far", "Original photo", "Original | From far", "Flip changes")
# Views in which the picture covers exactly the whole mosaic, so a dragged rectangle maps to stickers.
SELECTABLE_VIEWS = ("Build view", "From far", "Original photo", "Flip changes")


def render_view(r: Result, name: str) -> Image.Image:
    """The picture for one named view of the result (always uses the flipped `indices`)."""
    if name == "Build view":
        return render_preview(r.indices, r.palette, r.cube, r.frame)        # real stickers and frame
    far = render_perceived(r.indices, r.palette)                             # blended colors, as from afar
    if name == "From far":
        return far
    original = r.cropped.resize(far.size, Image.LANCZOS)                     # the photo, same size as `far`
    if name == "Original photo":
        return original
    if name == "Original | From far":
        return side_by_side([("Original", original), ("Mosaic from far", far)])
    if name == "Flip changes":
        return render_changed_cells(r.matched, r.indices, r.palette)         # only flipped stickers bright
    raise ValueError(f"unknown view {name!r}")


# --- files ---------------------------------------------------------------------------------------

def save_outputs(r: Result, out_dir: str, compare: bool = False):
    """Write the result to disk. Returns (list of file names written, comparison text lines or None)."""
    os.makedirs(out_dir, exist_ok=True)
    clean = render_preview(r.indices, r.palette, r.cube, r.frame)
    clean.save(os.path.join(out_dir, "preview_clean.png"))                   # the mosaic alone
    add_banner(clean, r.report.lines, r.report.level).save(os.path.join(out_dir, "preview.png"))
    render_perceived(r.indices, r.palette).save(os.path.join(out_dir, "perceived.png"))
    np.save(os.path.join(out_dir, "indices.npy"), r.indices)                 # the answer, flips included
    names = ["preview.png", "preview_clean.png", "perceived.png", "indices.npy"]

    if r.flips:
        np.save(os.path.join(out_dir, "indices_before_flips.npy"), r.matched)
        with open(os.path.join(out_dir, "flips.txt"), "w") as fh:            # one rule per line, CLI syntax
            for rule in r.flips:
                fh.write(("" if rule.enabled else "# disabled: ") + rule.to_text() + "\n")
        names += ["indices_before_flips.npy", "flips.txt"]

    lines = None
    if compare:
        lines = describe_transitions(transition_counts(r.baseline, r.indices), COLOR_NAMES)
        side_by_side([
            ("Plain v2 (from far)", render_perceived(r.baseline, r.palette)),
            ("Your settings (from far)", render_perceived(r.indices, r.palette)),
            ("Changed stickers", render_changed_cells(r.baseline, r.indices, r.palette)),
        ]).save(os.path.join(out_dir, "compare.png"))
        names.append("compare.png")
    return names, lines
