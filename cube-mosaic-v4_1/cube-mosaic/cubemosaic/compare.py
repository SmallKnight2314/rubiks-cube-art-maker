"""compare.py  (v3)  --  see what a setting changed, side by side.

Used for the blue/green mix-up work: run with your new settings and this file shows
  1. the plain v2 result next to your result,
  2. a map of exactly which stickers changed (only those are drawn in bright color),
  3. a count of what turned into what, e.g. "G -> B: 212 stickers".
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .color import linear_to_srgb
from .palette import Palette


def transition_counts(old: np.ndarray, new: np.ndarray) -> np.ndarray:
    """6 x 6 table: entry [i, j] = how many stickers were color i in `old` and color j in `new`.

    The diagonal (i == j) is "unchanged". np.add.at adds 1 at each (old, new) pair, correctly
    even when the same pair occurs many times.
    """
    counts = np.zeros((6, 6), dtype=np.int64)
    np.add.at(counts, (old.ravel(), new.ravel()), 1)
    return counts


def describe_transitions(counts: np.ndarray, names, top: int = 8) -> list:
    """Turn the table into short text lines, biggest changes first (diagonal ignored)."""
    total = int(counts.sum())
    changed = total - int(np.trace(counts))               # trace = sum of the diagonal = unchanged cells
    lines = [f"{changed} of {total} stickers changed ({100.0 * changed / total:.1f}%)"]
    pairs = [(int(counts[i, j]), i, j) for i in range(6) for j in range(6) if i != j and counts[i, j] > 0]
    pairs.sort(reverse=True)                              # largest counts first
    for n, i, j in pairs[:top]:
        lines.append(f"  {names[i]} -> {names[j]}: {n}")
    return lines


def render_changed_cells(old: np.ndarray, new: np.ndarray, palette: Palette, cell: int = 8) -> Image.Image:
    """Picture of the NEW result in which only the changed stickers are bright.

    Unchanged stickers are drawn dark and muted so the eye goes straight to the changes.
    Changed stickers are drawn in the real sticker color they changed TO.
    """
    raw = np.round(linear_to_srgb(palette.raw_linear) * 255).astype(np.float64)       # (6, 3) bright colors
    muted = raw * 0.28 + 20.0                                                           # dark, washed-out versions
    changed = old != new                                                                # (H, W) True where different
    img = np.where(changed[..., None], raw[new], muted[new]).astype(np.uint8)          # pick per cell
    img = np.repeat(np.repeat(img, cell, axis=0), cell, axis=1)                         # enlarge to be viewable
    return Image.fromarray(img, mode="RGB")


def side_by_side(panels) -> Image.Image:
    """Put titled pictures next to each other. panels = [(title, PIL image), ...], all the same size."""
    w, h = panels[0][1].size
    font_size = max(14, w // 28)
    font = ImageFont.load_default(size=font_size)
    gap = max(10, w // 60)                                # space between panels
    title_h = int(font_size * 2.0)                        # strip above each picture for its title
    out = Image.new("RGB", (len(panels) * w + (len(panels) + 1) * gap, h + title_h + gap), (28, 28, 32))
    d = ImageDraw.Draw(out)
    for i, (title, img) in enumerate(panels):
        x = gap + i * (w + gap)
        d.text((x, gap // 2 + font_size // 3), title, fill=(255, 255, 255), font=font)
        out.paste(img, (x, title_h))
    return out
