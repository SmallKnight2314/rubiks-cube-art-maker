"""color.py  (v1)  --  color conversions and the "how different do two colors look" space.

Why this file exists
--------------------
The numbers in a PNG/JPG (sRGB) are NOT proportional to light. Doubling an sRGB
value does not double the brightness. Two things in this project need real
light-proportional math:

  * averaging  (shrinking the photo to one value per sticker, and blending
                sticker color with frame color) must be done in LINEAR light,
                otherwise mixes come out too dark;
  * comparing  (which of my 6 colors is closest to this target?) should be done
                in a space where equal distances look like equal differences to
                a human eye. We use OKLab for that.

So every color travels like this:   hex/sRGB  ->  linear light  ->  OKLab

All functions work on whole numpy arrays at once ("vectorised"), so converting
a 100,000-pixel image is one call, not a Python loop.
"""
import numpy as np


def hex_to_rgb01(h: str) -> np.ndarray:
    """'#FFD500' (or 'FFD500', or short '#FD0') -> array([1.0, 0.835, 0.0]).

    Result values are sRGB in the range 0..1 (0 = none, 1 = full).
    """
    h = h.strip().lstrip("#")
    if len(h) == 3:
        # short form: 'abc' means 'aabbcc'
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        raise ValueError(f"bad hex color: {h!r}")
    # Slice the string into 3 two-character pieces (RR, GG, BB), read each as a
    # base-16 number (0..255), then scale to 0..1.
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float64) / 255.0


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """sRGB 0..1  ->  linear light 0..1 (the standard sRGB "decoding" curve).

    Linear light is what physically adds up: mixing 50% of a white and 50% of
    a black patch gives linear value 0.5, which is a LOT brighter than the
    sRGB value 0.5 would suggest.
    """
    c = np.asarray(c, dtype=np.float64)
    # The sRGB curve has two parts: a short straight section near black and a
    # power curve (gamma ~2.4) for everything else. np.where picks per element.
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    """Linear light 0..1  ->  sRGB 0..1 (the inverse of srgb_to_linear).

    Used right before writing pixels into a PNG so they display correctly.
    Values are clipped to 0..1 first because blends can drift a hair outside.
    """
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


# Two fixed 3x3 matrices that define OKLab (published by Bjorn Ottosson).
# Do not edit these numbers: they ARE the definition of the color space.
#   _M1 turns linear RGB into a rough cone-response ("LMS") signal;
#   _M2 turns the cube-rooted cone signal into L (lightness), a (green-red), b (blue-yellow).
_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])


def linear_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """Linear sRGB array of shape (..., 3)  ->  OKLab array of shape (..., 3).

    "(...)" means any number of leading dimensions: a single color (3,), a list
    of colors (6, 3) or a whole image (H, W, 3) all work.

    In OKLab, the plain straight-line distance between two colors is a good
    estimate of how different they look. That is exactly what the matching
    step in process.py relies on.
    """
    lms = rgb @ _M1.T            # '@' = matrix multiply, applied to every color at once
    return np.cbrt(lms) @ _M2.T  # cube root = the perceptual compression step


def hex_to_linear(h: str) -> np.ndarray:
    """Shortcut: hex string -> linear light RGB (the form palette.py works in)."""
    return srgb_to_linear(hex_to_rgb01(h))


# --- added in v3 -------------------------------------------------------------
# The inverse of the two OKLab matrices, calculated once when the file loads.
_M1_INV = np.linalg.inv(_M1)
_M2_INV = np.linalg.inv(_M2)


def oklab_to_linear(lab: np.ndarray) -> np.ndarray:
    """OKLab (..., 3) -> linear sRGB (..., 3): the exact reverse of linear_to_oklab.

    Used by the diagnose tool to turn "a color of this lightness, saturation and
    hue" back into a real RGB color, and to test whether a photo could contain it.
    Values outside 0..1 mean the color cannot appear in an ordinary photo.
    """
    lms = (lab @ _M2_INV.T) ** 3       # undo the cube root, i.e. cube it
    return lms @ _M1_INV.T


def oklab_to_lch(lab: np.ndarray):
    """OKLab (..., 3) -> (L, C, h): lightness, chroma (colorfulness) and hue angle.

    L  0 (black) .. 1 (white)
    C  distance from the gray axis: 0 = gray, bigger = more saturated.
    h  hue as an angle in degrees, 0..360. Rough guide in OKLab:
         ~20 red, ~40 orange, ~95 yellow, ~150 green, ~195 cyan, ~260 blue, ~330 magenta.

    "Hue" is what makes blue look like blue and green look like green,
    regardless of how light or dark it is. The matching controls in process.py
    use it to stop a blue-ish target from being given a green sticker.
    """
    a = lab[..., 1]
    b = lab[..., 2]
    chroma = np.hypot(a, b)                           # sqrt(a^2 + b^2)
    hue = np.degrees(np.arctan2(b, a)) % 360.0        # angle of the (a, b) point, wrapped to 0..360
    return lab[..., 0], chroma, hue
