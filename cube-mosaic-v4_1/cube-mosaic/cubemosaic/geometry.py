"""geometry.py  (v0)  --  the physical size of things.

This file answers: "given this cube type, these size limits and this image,
how many cubes wide and tall will the mosaic be, and how big is it in mm?"

It knows nothing about colors or images' pixels; it only does measurements.

Vocabulary used everywhere in the project
-----------------------------------------
cube      one whole puzzle cube. In the mosaic each cube shows ONE face.
n         stickers per edge of that face (3 for a 3x3, 7 for a 7x7).
pitch     distance from the center of one sticker to the center of the next,
          = cube edge / n. Each sticker "owns" a pitch x pitch square of the wall.
sticker   the colored square itself (smaller than the pitch).
padding   the plastic frame gap between two stickers = pitch - sticker.
pixel     one sticker position in the final mosaic. The mosaic is a grid of
          (cubes_x * n) x (cubes_y * n) pixels.
"""
from dataclasses import dataclass
import math

# Typical manufactured edge lengths in millimetres, per cube type.
# These are only DEFAULTS: the command line (and later the GUI) lets you override
# the cube size, because brands differ by a millimetre or two.
CUBE_PRESETS_MM = {2: 50.0, 3: 57.0, 4: 62.0, 5: 63.0, 6: 68.0, 7: 70.0}


@dataclass(frozen=True)  # frozen = the values cannot be changed after creation (safer)
class CubeSpec:
    """Describes ONE kind of cube: its type and its two sizes."""
    n: int               # stickers per face edge (2..7)
    cube_mm: float       # cube edge length in mm
    sticker_mm: float    # sticker edge length in mm

    def __post_init__(self):
        # Runs automatically right after the object is created and rejects
        # impossible input early, with a readable message, instead of letting
        # nonsense numbers travel through the program.
        if self.n not in CUBE_PRESETS_MM:
            raise ValueError("cube type must be 2x2 to 7x7")
        if self.cube_mm <= 0 or self.sticker_mm <= 0:
            raise ValueError("sizes must be positive")
        # A sticker bigger than its own cell would overlap its neighbours.
        # The 1e-9 tolerates floating point noise (e.g. 19.000000001).
        if self.sticker_mm > self.pitch_mm + 1e-9:
            raise ValueError(
                f"sticker ({self.sticker_mm} mm) is larger than the pitch ({self.pitch_mm:.2f} mm)")

    # The three @property values below are computed on demand from the three
    # numbers above, so they can never get out of sync with them.

    @property
    def pitch_mm(self) -> float:
        """Centre-to-centre distance of neighbouring stickers (cube edge / n)."""
        return self.cube_mm / self.n

    @property
    def padding_mm(self) -> float:
        """Width of the frame gap between neighbouring stickers."""
        return self.pitch_mm - self.sticker_mm

    @property
    def fill_ratio(self) -> float:
        """Fraction of each pixel's AREA that is sticker (the rest is frame).

        Example: 3x3 with 15 mm stickers on a 19 mm pitch -> (15/19)^2 = 0.62.
        This number is what palette.py uses to blend sticker color with frame
        color, because from a distance the eye averages the two.
        """
        return (self.sticker_mm / self.pitch_mm) ** 2


def default_sticker_mm(n: int, cube_mm: float) -> float:
    """Sticker size to use when the user does not give one.

    A typical 3x3 has ~15 mm stickers on a 19 mm pitch, i.e. the sticker is
    15/19 (about 79%) of the pitch. We keep that same proportion for every
    cube type, so a 7x7 gets proportionally smaller stickers.
    """
    return cube_mm / n * (15.0 / 19.0)


@dataclass(frozen=True)
class MosaicGrid:
    """The result of the layout calculation: how the cubes are arranged."""
    cubes_x: int         # cubes across (width)
    cubes_y: int         # cubes down (height)
    n: int               # stickers per cube edge (copied from the CubeSpec)
    width_mm: float      # real width of the finished mosaic
    height_mm: float     # real height of the finished mosaic
    crop_loss: float     # fraction of the source image that gets cropped away (0 = none)

    @property
    def px_x(self) -> int:
        """Mosaic width in stickers (pixels)."""
        return self.cubes_x * self.n

    @property
    def px_y(self) -> int:
        """Mosaic height in stickers (pixels)."""
        return self.cubes_y * self.n

    @property
    def cube_count(self) -> int:
        """How many physical cubes you need to buy."""
        return self.cubes_x * self.cubes_y


def compute_grid(img_w: int, img_h: int, cube: CubeSpec,
                 max_w_m: float, max_h_m: float) -> MosaicGrid:
    """Pick the biggest cube grid that fits the space and matches the image shape.

    Inputs: image size in pixels, the cube, and the maximum mosaic size in METERS.

    How it decides:
      1. How many cubes fit at most across and down (max_x, max_y)?
      2. For every possible width cx (1..max_x), the "perfect" height for the
         image's aspect ratio is cx / aspect, which is rarely a whole number,
         so we try both neighbours (floor and ceil).
      3. A candidate is only allowed if it fits the height limit AND would crop
         at most ~7% of the image (the image is later center-cropped to the
         grid's exact shape, so a slightly different shape loses a sliver).
      4. Among allowed candidates the one with the MOST cubes wins; ties go to
         the one that crops least.
    """
    # Whole cubes that fit. The tiny +1e-9 stops floating point error from
    # turning an exact fit like 2.000 m / 0.05 m = 39.99999 into 39.
    max_x = int(math.floor(max_w_m * 1000.0 / cube.cube_mm + 1e-9))
    max_y = int(math.floor(max_h_m * 1000.0 / cube.cube_mm + 1e-9))
    if max_x < 1 or max_y < 1:
        raise ValueError("the size limit is smaller than a single cube")
    aspect = img_w / img_h          # >1 landscape, <1 portrait, 1 square

    best = None                     # will hold (score, cx, cy, kept) of the winner
    for cx in range(1, max_x + 1):
        for cy in (math.floor(cx / aspect), math.ceil(cx / aspect)):
            if cy < 1 or cy > max_y:
                continue            # does not fit the height limit (or is zero)
            g_aspect = cx / cy      # shape of this candidate grid
            # "kept" = share of the image that survives the center crop.
            # Dividing the smaller aspect by the larger gives a number <= 1
            # that is exactly 1.0 when the shapes match perfectly.
            kept = min(aspect, g_aspect) / max(aspect, g_aspect)
            if kept < 0.93:         # would crop more than ~7%: not allowed
                continue
            # Python compares tuples left to right: more cubes first, then less crop.
            score = (cx * cy, kept)
            if best is None or score > best[0]:
                best = (score, cx, cy, kept)

    if best is None:
        # Fallback for extreme cases (e.g. a very wide image with a tall limit
        # box) where nothing passed the 7% rule: accept any shape that fits and
        # simply report the crop loss to the user.
        for cx in range(1, max_x + 1):
            cy = min(max_y, max(1, round(cx / aspect)))
            g_aspect = cx / cy
            kept = min(aspect, g_aspect) / max(aspect, g_aspect)
            score = (cx * cy, kept)
            if best is None or score > best[0]:
                best = (score, cx, cy, kept)

    _, cx, cy, kept = best
    return MosaicGrid(cx, cy, cube.n, cx * cube.cube_mm, cy * cube.cube_mm, 1.0 - kept)
