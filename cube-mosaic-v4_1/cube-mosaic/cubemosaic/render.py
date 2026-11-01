"""render.py  (v1, banner added in v2)  --  turn the answer into pictures.

Two pictures are produced from the same grid of color indices (0..5):

  render_preview    "what you will actually build": real sticker squares with
                    real-proportion gaps on the frame color. Use this to judge
                    the result and to approve it.
  render_perceived  "what it roughly looks like from far away": every cell
                    painted with its single blended (sticker + frame) color.

add_banner (v2) attaches a text strip above a picture, used to show the
viewing-distance verdict on the preview.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .color import linear_to_srgb
from .geometry import CubeSpec
from .palette import Palette, FrameSpec, FRAME_FRAMELESS


def _sticker_ratio(cube: CubeSpec, frame: FrameSpec) -> float:
    """Sticker EDGE / pitch (a length ratio, e.g. 15/19), used to size the squares.

    Not the same as fill_ratio, which is an AREA ratio (the square of this).
    Frameless cubes: the colored piece is the pitch minus the thin seam.
    """
    if frame.mode == FRAME_FRAMELESS:
        return max(cube.pitch_mm - frame.seam_mm, 0.0) / cube.pitch_mm
    return cube.sticker_mm / cube.pitch_mm


def render_preview(indices: np.ndarray, palette: Palette, cube: CubeSpec,
                   frame: FrameSpec, max_px: int = 3000) -> Image.Image:
    """Draw the mosaic at true sticker/frame proportions.

    indices  (H, W) array of 0..5 (position in COLOR_NAMES) per sticker.
    max_px   the longer side of the picture will be about this many pixels.

    Method: split the output into one square "cell" per sticker position, fill
    everything with the frame color, then paint a centered sticker-sized square
    in each cell with that cell's color.
    """
    h, w = indices.shape
    cell = max(3, max_px // max(w, h))               # pixels per cell edge, at least 3 so a gap stays visible
    ratio = _sticker_ratio(cube, frame)
    side = int(round(ratio * cell))                  # sticker edge in output pixels
    side = min(max(side, 1), cell)                   # clamp to 1..cell so it always shows and never overflows
    inset = (cell - side) // 2                       # gap on the top/left so the square sits in the middle

    # A single cell-sized True/False stencil: True where the sticker is, False for frame.
    mask = np.zeros((cell, cell), dtype=bool)
    mask[inset:inset + side, inset:inset + side] = True
    mask_full = np.tile(mask, (h, w))                # repeat the stencil for every cell of the mosaic

    # Colors converted to 0..255 sRGB for the PNG.
    raw_srgb = np.round(linear_to_srgb(palette.raw_linear) * 255).astype(np.uint8)    # (6, 3) sticker colors
    frame_srgb = np.round(linear_to_srgb(palette.frame_linear) * 255).astype(np.uint8)  # (3,) frame color

    # Blow each index up to a cell x cell block so it lines up with the stencil.
    idx_up = np.repeat(np.repeat(indices, cell, axis=0), cell, axis=1)
    canvas = np.empty((h * cell, w * cell, 3), dtype=np.uint8)
    canvas[:] = frame_srgb                                    # start with all frame color
    canvas[mask_full] = raw_srgb[idx_up[mask_full]]          # paint sticker pixels where the stencil is True
    return Image.fromarray(canvas, mode="RGB")


def render_perceived(indices: np.ndarray, palette: Palette, cell: int = 8) -> Image.Image:
    """Each sticker position as ONE flat block of its blended (perceived) color.

    This is a rough simulation of the mosaic seen from far enough away that the
    sticker and its frame have merged. `cell` = output pixels per sticker.
    """
    srgb = np.round(linear_to_srgb(palette.perceived_linear) * 255).astype(np.uint8)   # (6, 3)
    small = srgb[indices]                                     # (H, W, 3): look up each cell's color
    up = np.repeat(np.repeat(small, cell, axis=0), cell, axis=1)   # enlarge so it is viewable
    return Image.fromarray(up, mode="RGB")


# Banner background colors by report level (see viewing.ViewingReport.level).
_BANNER_COLORS = {
    "ok": (28, 92, 52),       # green: distance is fine
    "clash": (150, 55, 20),   # burnt orange: distance conflicts with the estimate
    "info": (45, 60, 85),     # slate: no distance given, just informing
}


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list:
    """Split `text` into lines that each fit within `max_width` pixels (breaks at spaces)."""
    words = text.split()
    lines, current = [], ""
    for word in words:
        trial = word if not current else current + " " + word
        if draw.textlength(trial, font=font) <= max_width or not current:
            current = trial               # still fits (or a single over-long word): keep adding
        else:
            lines.append(current)         # full: start a new line with this word
            current = word
    if current:
        lines.append(current)
    return lines


def add_banner(img: Image.Image, text_lines, level: str) -> Image.Image:
    """Return a NEW picture: a colored text strip stacked on top of `img`.

    text_lines  list of sentences; each is word-wrapped to the picture width.
    level       "ok" | "clash" | "info" picks the strip color.
    """
    font_size = max(14, img.width // 70)            # text scales with the picture so it stays readable
    font = ImageFont.load_default(size=font_size)   # built-in font; size= needs Pillow 10.1 or newer
    pad = int(font_size * 0.7)                      # empty margin around the text
    line_h = int(font_size * 1.35)                  # vertical space per text line

    # First pass on a throw-away canvas just to measure text widths for wrapping.
    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    wrapped = []
    for t in text_lines:
        wrapped.extend(_wrap_text(probe, t, font, img.width - 2 * pad))

    bar_h = 2 * pad + line_h * len(wrapped)
    out = Image.new("RGB", (img.width, img.height + bar_h), _BANNER_COLORS.get(level, _BANNER_COLORS["info"]))
    out.paste(img, (0, bar_h))                      # the mosaic goes under the strip
    d = ImageDraw.Draw(out)
    y = pad
    for line in wrapped:
        d.text((pad, y), line, fill=(255, 255, 255), font=font)
        y += line_h
    return out
