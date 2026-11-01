"""palette.py  (v1, frame modes added in v1/v2)  --  what the 6 colors REALLY look like.

The key idea of the whole project lives here
--------------------------------------------
A pixel of the finished mosaic is not a pure sticker color. It is a sticker
(say, red) sitting in a cell that also contains frame plastic (usually black).
From more than a short distance the eye merges the two, so the viewer sees:

    perceived = f * sticker_color  +  (1 - f) * frame_color        (in linear light)

where f = the fraction of the cell covered by sticker (geometry.fill_ratio).
With a black frame this makes every color darker; with a translucent white
frame it makes them lighter/grayer. If we matched the photo against the raw hex
codes we would pick the wrong stickers, so process.py matches against these
PERCEIVED colors instead.

This file builds those six perceived colors ("the palette").
"""
from dataclasses import dataclass
import numpy as np

# linear_to_srgb was previously imported inside a method; it lives at the top now.
from .color import hex_to_linear, linear_to_oklab, linear_to_srgb
from .geometry import CubeSpec

# The fixed order of the six colors. EVERYWHERE in the project (the index array,
# counts, blueprint letters later) position 0 = W, 1 = Y, 2 = R, 3 = O, 4 = B, 5 = G.
COLOR_NAMES = ("W", "Y", "R", "O", "B", "G")

# Standard Rubik's brand colors. Manufacturers differ slightly, so the command
# line / GUI can override each one with its own hex code.
DEFAULT_COLORS = {
    "W": "#FFFFFF",   # white
    "Y": "#FFD500",   # yellow
    "R": "#B71234",   # red
    "O": "#FF5800",   # orange
    "B": "#0046AD",   # blue
    "G": "#009B48",   # green
}

# The three ways the plastic around a sticker can behave.
FRAME_OPAQUE = "opaque"            # normal stickered cube: solid frame color (usually black)
FRAME_TRANSLUCENT = "translucent"  # e.g. translucent white plastic: you also see what is behind it
FRAME_FRAMELESS = "frameless"      # stickerless cube: the colored plastic IS the pixel, only a thin seam shows


@dataclass
class FrameSpec:
    """Everything about the frame (the plastic between/around the stickers)."""
    mode: str = FRAME_OPAQUE       # one of the three FRAME_* constants above
    hex: str = "#000000"           # base plastic color (opaque and translucent modes)
    opacity: float = 1.0           # translucent mode only: 1 = solid, 0 = fully see-through
    backdrop_hex: str = "#000000"  # translucent mode only: what shows THROUGH the plastic (the cube's black core)
    seam_mm: float = 0.6           # frameless mode only: visible gap between neighbouring pieces
    seam_hex: str = "#000000"      # frameless mode only: color seen inside that gap

    def effective_linear(self) -> np.ndarray:
        """The color of the non-sticker part of a cell, as a viewer sees it (linear light RGB).

        opaque       -> simply the frame color.
        translucent  -> a mix of frame color and whatever is behind it:
                        opacity * frame + (1 - opacity) * backdrop.
                        A translucent WHITE frame over a black core therefore
                        looks gray, not white.
        frameless    -> the seam color (the thin gap between colored pieces).
        """
        if self.mode == FRAME_OPAQUE:
            return hex_to_linear(self.hex)
        if self.mode == FRAME_TRANSLUCENT:
            a = float(np.clip(self.opacity, 0.0, 1.0))   # keep the slider value within 0..1
            return a * hex_to_linear(self.hex) + (1 - a) * hex_to_linear(self.backdrop_hex)
        if self.mode == FRAME_FRAMELESS:
            return hex_to_linear(self.seam_hex)
        raise ValueError(f"unknown frame mode {self.mode!r}")

    def effective_srgb_hex_array(self) -> np.ndarray:
        """Same color as effective_linear() but as 0..255 sRGB numbers.

        Not used by the command line yet; it is here for the GUI (v4), which
        will want to show the frame color as a swatch.
        """
        return np.round(linear_to_srgb(self.effective_linear()) * 255).astype(np.uint8)


def effective_fill_ratio(cube: CubeSpec, frame: FrameSpec) -> float:
    """f in the formula above: share of each cell's area that is the colored part.

    Stickered cubes (opaque/translucent): f = (sticker / pitch)^2, from geometry.
    Frameless cubes: the colored plastic piece fills the whole pitch except a
    thin seam, so f = ((pitch - seam) / pitch)^2, which is close to 1.
    """
    if frame.mode == FRAME_FRAMELESS:
        visible = max(cube.pitch_mm - frame.seam_mm, 0.0)   # max(...) guards against a seam wider than the pitch
        return (visible / cube.pitch_mm) ** 2
    return cube.fill_ratio


@dataclass
class Palette:
    """The six colors in every form the rest of the program needs.

    Arrays have 6 rows, in COLOR_NAMES order (W, Y, R, O, B, G).
    """
    names: tuple
    raw_hex: tuple                 # the 6 sticker hex codes exactly as entered
    raw_linear: np.ndarray         # (6, 3) sticker colors, linear light (what the preview paints)
    perceived_linear: np.ndarray   # (6, 3) after blending with the frame (what the eye sees from afar)
    perceived_lab: np.ndarray      # (6, 3) the same perceived colors in OKLab (what matching compares against)
    fill_ratio: float              # f used for the blend, kept for display/debugging
    frame_linear: np.ndarray       # (3,) effective frame color, linear light


def build_palette(colors: dict, cube: CubeSpec, frame: FrameSpec) -> Palette:
    """Turn 6 hex codes + cube geometry + frame settings into a Palette.

    `colors` must be a dict with exactly the keys W, Y, R, O, B, G.
    """
    names = COLOR_NAMES
    if set(colors) != set(names):
        raise ValueError("exactly six colors W, Y, R, O, B, G are required")
    raw_hex = tuple(colors[n] for n in names)                       # keep the fixed order
    raw_lin = np.stack([hex_to_linear(h) for h in raw_hex])        # shape (6, 3)
    f = effective_fill_ratio(cube, frame)
    frame_lin = frame.effective_linear()                            # shape (3,)
    # THE blend. Broadcasting: frame_lin (3,) is combined with each of the 6 rows of raw_lin.
    perceived = f * raw_lin + (1.0 - f) * frame_lin
    return Palette(names, raw_hex, raw_lin, perceived, linear_to_oklab(perceived), f, frame_lin)
