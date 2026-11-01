"""flip.py  (v4)  --  the exact color flip: "every sticker of color X becomes color Y".

Why it exists
-------------
A big --bias only says "never use this color"; each cell then gets whatever is next best, which
is not necessarily the color you wanted (in v3 testing, forcing green out gave blue AND yellow,
white and orange). A flip is exact: it runs AFTER matching/dithering and simply rewrites stickers.

Rules
-----
* A rule is  source color -> target color, for the whole mosaic or for one rectangle.
* The source can also be "any color" (written * on the command line): that PAINTS every sticker of the
  rectangle with the target color, whatever it was. Select one sticker and it paints just that sticker.
* ALL rules read the matched result, never each other's output. So "B->G" together with "G->B"
  swaps the two colors, and "B->G" with "G->R" does NOT turn blue into red.
* If two rules claim the same sticker, the LATER rule in the list wins.
* Colors can only become one of the six, so no sticker can ever go missing.
* Rectangles are in STICKER coordinates: x0, y0 (inclusive) to x1, y1 (exclusive), counted from the
  top-left corner of the mosaic.
"""
from dataclasses import dataclass
import numpy as np

from .palette import COLOR_NAMES


@dataclass
class FlipRule:
    """One flip. Not frozen on purpose: the GUI ticks `enabled` on and off."""
    src: int                     # index of the color to replace (0..5 in COLOR_NAMES order W Y R O B G), or None = any color
    dst: int                     # index of the color to use instead
    region: tuple = None         # None = whole mosaic, else (x0, y0, x1, y1) in stickers
    enabled: bool = True         # unticked rules are kept in the list but ignored

    def label(self) -> str:
        """Short human text, e.g. 'B -> G, whole mosaic' or 'G -> B, x 10..40, y 5..30'."""
        where = "whole mosaic" if self.region is None else \
            f"x {self.region[0]}..{self.region[2]}, y {self.region[1]}..{self.region[3]}"
        src = "any" if self.src is None else COLOR_NAMES[self.src]
        return f"{src} -> {COLOR_NAMES[self.dst]}, {where}"

    def to_text(self) -> str:
        """The command-line form of this rule (the exact reverse of parse_flip)."""
        text = f"{'*' if self.src is None else COLOR_NAMES[self.src]}={COLOR_NAMES[self.dst]}"
        if self.region is not None:
            text += "@" + ",".join(str(v) for v in self.region)
        return text


def parse_flip(text: str) -> FlipRule:
    """'B=G' or 'G=B@x0,y0,x1,y1'  ->  FlipRule. Raises ValueError with a readable message."""
    rule_part, _, region_part = text.partition("@")               # split off the optional region
    src_name, _, dst_name = rule_part.partition("=")
    src_name, dst_name = src_name.strip().upper(), dst_name.strip().upper()
    any_source = src_name in ("*", "ANY")                           # "*=G" paints with G whatever was there
    if (src_name not in COLOR_NAMES and not any_source) or dst_name not in COLOR_NAMES:
        raise ValueError(f"flip must look like B=G (or *=G) with colors from {', '.join(COLOR_NAMES)}; got {text!r}")
    region = None
    if region_part:
        try:
            region = tuple(int(v) for v in region_part.split(","))
        except ValueError:
            raise ValueError(f"flip region must be four whole numbers x0,y0,x1,y1; got {region_part!r}") from None
        if len(region) != 4 or region[2] <= region[0] or region[3] <= region[1]:
            raise ValueError(f"flip region must be x0,y0,x1,y1 with x1>x0 and y1>y0; got {region_part!r}")
    return FlipRule(None if any_source else COLOR_NAMES.index(src_name), COLOR_NAMES.index(dst_name), region)


def apply_flips(indices: np.ndarray, rules) -> np.ndarray:
    """Return a NEW (H, W) array with the enabled rules applied. The input is never modified.

    Every rule looks at `indices` (the untouched matched result), which is what makes swaps work.
    A rectangle that sticks out of the mosaic is clipped to it; one entirely outside does nothing.
    """
    out = indices.copy()
    h, w = indices.shape
    for r in rules:
        if not r.enabled or r.src == r.dst:
            continue
        # every sticker currently of the source color (or every sticker, for an "any color" paint rule)
        mask = np.ones((h, w), dtype=bool) if r.src is None else (indices == r.src)
        if r.region is not None:
            x0, y0, x1, y1 = r.region
            box = np.zeros((h, w), dtype=bool)
            box[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = True   # numpy slicing clips at the far edge itself
            mask &= box
        out[mask] = r.dst
    return out
