"""process.py  (v1, matching controls added in v3)  --  from a photo to "which of the 6 colors goes in each cell".

The steps, in the order cli.py calls them
-----------------------------------------
  load_image        open the file, fix rotation, flatten transparency
  crop_to_aspect    trim the photo to the exact shape of the cube grid
  target_linear     shrink it to ONE color per sticker position (averaging, in linear light)
  Matcher           for each of those colors pick one of the 6 perceived palette colors
                    (dither.py uses the same Matcher, so both methods obey the same rules)

Why blue and green get mixed up (found in v3 by sweeping hues, see diagnose.py)
------------------------------------------------------------------------------
With a black frame the six PERCEIVED colors are fairly dark, and blue is the darkest chromatic
one (lightness about 0.37) while green sits in the middle (about 0.52). Plain nearest-color
matching mostly compares lightness, because the lightness gaps between palette colors are big.
Three things follow:
  1. Medium-light blues (sky, jeans, steel) are closer in lightness to GREEN than to the dark blue,
     so they turn green.
  2. Gray/desaturated mid-tones have no gray sticker to land on, and green happens to be the
     palette color closest to mid lightness, so gray areas turn greenish.
  3. Cyan/teal is not in the palette at all; it sits between blue and green and goes to whichever
     is closer in lightness.
The controls below (MatchSettings) are tools to push against that. They all default to "off", so
a run without options gives exactly the v2 result.
"""
from dataclasses import dataclass
import numpy as np
from PIL import Image, ImageOps

from .color import srgb_to_linear, linear_to_oklab, oklab_to_lch
from .palette import Palette


def load_image(path: str) -> Image.Image:
    """Open any common image file and return a plain RGB picture.

    Two details that would otherwise cause surprises:
      * Phone photos store rotation as a tag instead of rotating the pixels;
        exif_transpose applies that tag so the picture is the right way up.
      * PNGs with transparency would turn black (or garbage) when converted to
        RGB, so transparent areas are painted onto white first.
    """
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))   # opaque white canvas
        img = Image.alpha_composite(bg, rgba)                     # lay the picture over it
    return img.convert("RGB")


def crop_to_aspect(img: Image.Image, aspect: float, off_x: float = 0.5, off_y: float = 0.5) -> Image.Image:
    """Cut the picture down to a given width/height ratio, removing as little as possible.

    aspect  target width / height (the cube grid's shape, e.g. 105/78).
    off_x   0..1 : where the crop window sits horizontally when the picture is
            too wide (0 = flush left, 0.5 = centered, 1 = flush right).
    off_y   same, vertically, when the picture is too tall.
    """
    w, h = img.size
    if w / h > aspect:
        # Picture is wider than the target: keep full height, trim the sides.
        new_w = int(round(h * aspect))
        left = int(round((w - new_w) * off_x))    # leftover width, shifted by the slider
        return img.crop((left, 0, left + new_w, h))
    # Picture is taller than (or equal to) the target: keep full width, trim top/bottom.
    new_h = int(round(w / aspect))
    top = int(round((h - new_h) * off_y))
    return img.crop((0, top, w, top + new_h))


def target_linear(img: Image.Image, px_x: int, px_y: int) -> np.ndarray:
    """Shrink the picture to px_x by px_y cells and return linear-light colors.

    Result shape is (px_y, px_x, 3): row, column, (R, G, B). Each entry is the
    AVERAGE light of the part of the photo that cell covers.

    Why it is done this way:
      * Convert to linear light FIRST, then average. Averaging sRGB numbers
        directly makes mixed regions too dark.
      * Image.BOX is plain area averaging (every source pixel under a cell
        counts equally), which is what "unit area color" means. Nicer-looking
        resize filters would sharpen/ring and distort the average.
      * PIL can only resize float data one channel at a time (mode "F"), hence
        the loop over R, G, B.
    """
    arr = np.asarray(img, dtype=np.float64) / 255.0              # 0..255 -> 0..1
    lin = srgb_to_linear(arr).astype(np.float32)                 # linear light; float32 is what mode "F" holds
    chans = []
    for c in range(3):                                           # 0 = R, 1 = G, 2 = B
        ch = Image.fromarray(lin[:, :, c], mode="F")             # one channel as a float image
        ch = ch.resize((px_x, px_y), resample=Image.BOX)         # area average down to the grid
        chans.append(np.asarray(ch, dtype=np.float64))
    return np.stack(chans, axis=-1)                              # glue back into (H, W, 3)


@dataclass(frozen=True)
class MatchSettings:
    """The knobs that decide which palette color "wins" for a target color.

    With every default left alone the matcher is exactly the v2 nearest-color rule.
    """
    # How much a LIGHTNESS difference counts compared with a COLOR (hue/saturation) difference.
    # 1.0 = equal (v2). Below 1 = lightness matters less, so a target gets the palette color of
    # the right color family even when it is lighter or darker than that sticker.
    # Try 0.4 to 0.7. Too low and dark things start turning white or yellow.
    lightness_weight: float = 1.0

    # Hue guard. A target with real color in it may only receive a chromatic sticker whose hue is
    # within this many degrees of the target's hue (white is always allowed). This stops a
    # blue-ish target from ever receiving green, a green-ish one from receiving blue, and so on.
    # 0 = off (v2). Try 60 to 75. Smaller = stricter.
    hue_guard_deg: float = 0.0

    # The guard only applies to targets at least this colorful (OKLab chroma). Below it the target
    # is basically gray, its hue is meaningless, and the guard stays out of the way.
    guard_min_chroma: float = 0.04

    # One number per color in the order W, Y, R, O, B, G. It is ADDED to that color's squared
    # distance: a positive number makes the color less likely to be picked, a negative number
    # more likely. The scale is OKLab squared distance: 0.002 is gentle, 0.01 is strong.
    # This is the simple "custom flip rule": e.g. G = +0.004 nudges borderline cells from green to
    # their next best choice.
    bias: tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # Same idea as `bias`, but applied ONLY to near-gray targets (chroma below guard_min_chroma).
    # Gray mid-tones have no matching sticker, and plain matching gives them green. The hue guard
    # cannot help because a gray has no hue. Penalising green for grays just moves them to the next
    # best color (orange or red), so which cast you prefer is a taste decision: this lets you make it.
    # Example: --neutral-bias G=0.02 gives mid-gray to red/orange instead of green.
    neutral_bias: tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class Matcher:
    """Picks palette colors for target colors, following a MatchSettings.

    One Matcher is built per run (it precomputes everything about the palette) and then used
    either on the whole picture at once (match_array, plain nearest color) or one cell at a time
    (match_one, used by the dithering loop in dither.py).
    """

    def __init__(self, palette: Palette, settings: MatchSettings = MatchSettings()):
        self.palette = palette
        self.settings = settings
        self.pal_lab = palette.perceived_lab                       # (6, 3) the six colors the eye will see
        _, pal_chroma, pal_hue = oklab_to_lch(self.pal_lab)        # their colorfulness and hue angle
        self.pal_hue = pal_hue                                     # (6,)
        # White (and any gray) has no hue, so it must never be blocked by the hue guard.
        # "Has a hue" = chroma above a small threshold.
        self.pal_has_hue = pal_chroma > 0.02                       # (6,) True for Y R O B G, False for W
        self.bias = np.asarray(settings.bias, dtype=np.float64)    # (6,)
        self.neutral_bias = np.asarray(settings.neutral_bias, dtype=np.float64)   # (6,)
        # Per-axis weights for (L, a, b): only lightness is adjustable.
        self.axis_weights = np.array([settings.lightness_weight, 1.0, 1.0])

    def distances(self, lab: np.ndarray, guide_lab: np.ndarray = None) -> np.ndarray:
        """Cost of giving each of the 6 colors to each target. Lower = better.

        lab       : targets in OKLab, shape (..., 3): one color (3,), or a whole picture (H, W, 3).
        guide_lab : optional. The colors whose HUE and colorfulness the hue guard and the gray-only
                    bias should judge. Plain matching leaves it out (the target judges itself).
                    Dithering passes the ORIGINAL photo color here: the color being matched there is
                    "photo + accumulated error", and that error can push its hue into a forbidden
                    family (tan sand picking up green speckles), so the guard must look at what the
                    photo really contained.
        returns shape (..., 6). A color the hue guard forbids gets infinity.
        """
        if guide_lab is None:
            guide_lab = lab
        # Difference between every target and every palette color: (..., 6, 3)
        d = lab[..., None, :] - self.pal_lab
        # Weighted squared distance: wL*dL^2 + da^2 + db^2, summed over the last axis.
        dist = np.einsum("...kc,c,...kc->...k", d, self.axis_weights, d)
        dist = dist + self.bias                                    # the flip-rule nudges

        guard = self.settings.hue_guard_deg
        use_neutral = bool(self.neutral_bias.any())                # is a gray-only nudge set?
        if guard > 0 or use_neutral:
            _, chroma, hue = oklab_to_lch(guide_lab)               # colorfulness and hue of the color to judge, shape (...)
        if use_neutral:
            # Near-gray targets (chroma under the threshold) get the extra gray-only nudge.
            is_gray = chroma < self.settings.guard_min_chroma      # (...)
            dist = dist + np.where(is_gray[..., None], self.neutral_bias, 0.0)
        if guard > 0:
            # Angle between target hue and each palette hue, as a circle: 350 and 10 are 20 apart.
            dh = np.abs(hue[..., None] - self.pal_hue)             # (..., 6)
            dh = np.minimum(dh, 360.0 - dh)
            # Forbidden = the palette color has a hue AND is too far in hue AND the target is colorful.
            blocked = self.pal_has_hue & (dh > guard) & (chroma[..., None] >= self.settings.guard_min_chroma)
            guarded = np.where(blocked, np.inf, dist)
            # Safety: if the guard blocked every color for a target (only possible with an
            # unusual custom palette that has no white/gray), fall back to the unguarded cost.
            nothing_left = ~np.isfinite(guarded).any(axis=-1, keepdims=True)
            dist = np.where(nothing_left, dist, guarded)
        return dist

    def match_array(self, target_lin: np.ndarray) -> np.ndarray:
        """Plain nearest color for a whole picture of targets. Returns (H, W) of 0..5.

        Index meaning: 0 = W, 1 = Y, 2 = R, 3 = O, 4 = B, 5 = G (palette.COLOR_NAMES order).
        Every cell always gets one of the six; there is no "none" option, by design.
        """
        lab = linear_to_oklab(target_lin)
        return np.argmin(self.distances(lab), axis=-1).astype(np.uint8)

    def match_one(self, lin_rgb: np.ndarray, guide_lin: np.ndarray = None) -> int:
        """Best palette index (0..5) for ONE linear-light color of shape (3,).

        guide_lin: optional original photo color for the hue guard to judge (see distances()).
        """
        lab = linear_to_oklab(lin_rgb)
        guide = None if guide_lin is None else linear_to_oklab(guide_lin)
        return int(np.argmin(self.distances(lab, guide)))


def match_nearest(target_lin: np.ndarray, palette: Palette,
                  settings: MatchSettings = MatchSettings()) -> np.ndarray:
    """Convenience wrapper: plain nearest-color matching of a whole picture.

    This is "level A" from the plan: every cell is decided alone, so smooth gradients become flat
    bands. dither.py is "level B" and improves on that.
    """
    return Matcher(palette, settings).match_array(target_lin)
