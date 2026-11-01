"""viewing.py  (v2)  --  "from how far away should this mosaic be looked at?"

The user can type a viewing distance. This file works out its OWN best estimate
and compares the two, so the preview can warn when they disagree.

Two separate questions are answered
-----------------------------------
 1. WHOLE-IMAGE question  -> sets the recommended distance range.
    A mosaic is only recognisable if the eye can take in all of it at once.
      * too close  -> the image is bigger than your field of view; you cannot see it all.
      * too far    -> the image is so small in your view that fine detail is lost.
    We measure this with the VISUAL ANGLE the mosaic's diagonal covers:
        near limit : diagonal covers  near_angle_deg  (default 60 deg)
        best       : diagonal covers  best_angle_deg  (default 40 deg)
        far limit  : diagonal covers  far_angle_deg   (default 20 deg)
    Distance for a given angle:   d = size / (2 * tan(angle / 2))

 2. BLENDING question  -> an advisory only, never a "clash".
    Dithering (v3) fools the eye by putting different colors next to each other
    and counting on the eye to merge them. That only works when one sticker
    cell is tiny in your field of view. We measure the angle ONE CELL covers:
        cell angle = atan(pitch / distance)         (reported in arcminutes; 60 arcmin = 1 degree)
    and compare it with blend_arcmin. The eye resolves about 1 arcmin, so the
    default of 4 arcmin is a "mostly merged" threshold, not a perfect one.
    A 19 mm sticker pitch only drops to 4 arcmin at roughly 16 m, so in a normal
    room dithered mosaics look grainy. That is a physical fact, not a bug, and
    it is why v3 will tie dithering strength to this number.

ALL THE CONSTANTS BELOW ARE STARTING ESTIMATES. Tune them after looking at a
real printed/built mosaic; they live in ONE place (ViewingConstants) so you can
edit them without touching any logic.
"""
from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class ViewingConstants:
    """Tunable numbers behind the estimate. Edit the defaults here to change behaviour."""
    near_angle_deg: float = 60.0   # mosaic diagonal covers this much of your view at the CLOSEST comfortable distance
    best_angle_deg: float = 40.0   # ...at the best distance (similar to a comfortable TV view)
    far_angle_deg: float = 20.0    # ...at the FARTHEST useful distance (beyond it detail is lost)
    blend_arcmin: float = 4.0      # one cell must cover about this or less for neighbouring colors to merge


@dataclass(frozen=True)
class ViewingEstimate:
    """Our own recommendation, all distances in METERS."""
    near_m: float    # closest comfortable distance
    best_m: float    # recommended distance
    far_m: float     # farthest useful distance
    blend_m: float   # distance beyond which neighbouring stickers visually merge (dithering works)


@dataclass(frozen=True)
class ViewingReport:
    """The verdict on a user-supplied distance, ready to print and to draw on the preview."""
    level: str                 # "info" (no distance given), "ok" (inside the range) or "clash" (outside it)
    user_m: float              # the distance being judged (the recommended one when none was given)
    recommended_m: float       # our best estimate
    cell_arcmin: float         # how large ONE sticker cell looks from user_m, in arcminutes
    blend_state: str           # "blended", "partial" or "grainy"
    lines: tuple               # short human-readable sentences for the preview banner / terminal
    estimate: ViewingEstimate = field(default=None)   # the full numbers, for anyone who wants them


def distance_for_angle(size_m: float, angle_deg: float) -> float:
    """Distance at which something `size_m` long covers `angle_deg` of your view.

    Geometry: an object of length s seen from distance d spans an angle A where
    tan(A / 2) = (s / 2) / d, so d = s / (2 * tan(A / 2)).
    """
    return size_m / (2.0 * math.tan(math.radians(angle_deg) / 2.0))


def cell_angle_arcmin(pitch_mm: float, distance_m: float) -> float:
    """Angle one sticker cell covers when seen from `distance_m`, in arcminutes.

    degrees(atan(size / distance)) gives degrees; times 60 gives arcminutes.
    """
    return math.degrees(math.atan((pitch_mm / 1000.0) / distance_m)) * 60.0


def estimate_viewing(width_mm: float, height_mm: float, pitch_mm: float,
                     consts: ViewingConstants = ViewingConstants()) -> ViewingEstimate:
    """Compute near / best / far / blend distances for a mosaic of this size."""
    # The DIAGONAL is used as the mosaic's "size" so wide and tall mosaics are
    # treated the same way (the eye has to take in the whole rectangle).
    diag_m = math.hypot(width_mm, height_mm) / 1000.0
    near = distance_for_angle(diag_m, consts.near_angle_deg)
    best = distance_for_angle(diag_m, consts.best_angle_deg)
    far = distance_for_angle(diag_m, consts.far_angle_deg)
    # Blend distance: solve cell_angle_arcmin(pitch, d) = blend_arcmin for d.
    blend = (pitch_mm / 1000.0) / math.tan(math.radians(consts.blend_arcmin / 60.0))
    return ViewingEstimate(near, best, far, blend)


def _blend_state(cell_arcmin: float, consts: ViewingConstants) -> str:
    """Classify how well neighbouring cells merge at the user's distance."""
    ratio = cell_arcmin / consts.blend_arcmin    # 1.0 = exactly at the blend threshold
    if ratio <= 1.0:
        return "blended"     # cells are small enough that dithering merges into smooth tones
    if ratio <= 3.0:
        return "partial"     # merges somewhat; light dithering is fine
    return "grainy"          # cells are clearly visible; heavy dithering will look like noise


def check_viewing(est: ViewingEstimate, pitch_mm: float, user_m,
                  consts: ViewingConstants = ViewingConstants()) -> ViewingReport:
    """Judge the user's distance against our estimate.

    user_m may be None (the user gave no distance): the recommended distance is
    then used for everything downstream and the level is "info".
    """
    if user_m is not None and user_m <= 0:
        raise ValueError("viewing distance must be positive")

    used_m = est.best_m if user_m is None else user_m
    arcmin = cell_angle_arcmin(pitch_mm, used_m)
    blend = _blend_state(arcmin, consts)
    lines = []

    # --- line 1: the whole-image verdict ---------------------------------
    if user_m is None:
        level = "info"
        lines.append(f"No viewing distance given. Using the recommended {est.best_m:.1f} m "
                     f"(comfortable range {est.near_m:.1f} to {est.far_m:.1f} m).")
    elif user_m < est.near_m:
        level = "clash"
        lines.append(f"Viewing distance clash: your {user_m:.1f} m is closer than the comfortable range "
                     f"{est.near_m:.1f} to {est.far_m:.1f} m, so the whole mosaic will not fit your view. "
                     f"Recommended: {est.best_m:.1f} m.")
    elif user_m > est.far_m:
        level = "clash"
        lines.append(f"Viewing distance clash: your {user_m:.1f} m is farther than the comfortable range "
                     f"{est.near_m:.1f} to {est.far_m:.1f} m, so fine detail will be lost. "
                     f"Recommended: {est.best_m:.1f} m.")
    else:
        level = "ok"
        lines.append(f"Your viewing distance {user_m:.1f} m is inside the comfortable range "
                     f"{est.near_m:.1f} to {est.far_m:.1f} m (best about {est.best_m:.1f} m).")

    # --- line 2: the blending advisory (informational, never a clash) ----
    blend_text = {
        "blended": "neighbouring stickers merge, so dithering will look smooth",
        "partial": "neighbouring stickers only partly merge, so keep dithering light",
        "grainy": "stickers stay individually visible, so heavy dithering will look grainy",
    }[blend]
    lines.append(f"From {used_m:.1f} m one sticker spans about {arcmin:.0f} arcmin: {blend_text} "
                 f"(merging starts near {est.blend_m:.0f} m).")

    return ViewingReport(level, used_m, est.best_m, arcmin, blend, tuple(lines), est)


def dither_strength(report: ViewingReport, consts: ViewingConstants = ViewingConstants()) -> float:
    """How much dithering suits this viewing distance, from 0.0 (none) to 1.0 (full). Added in v3.

    Uses the same ratio as the blending advisory: (angle of one sticker cell) / blend_arcmin.
        ratio <= 1  -> neighbouring stickers merge        -> full strength, 1.0
        ratio >= 3  -> stickers clearly visible, grainy   -> 0.0 (plain nearest color)
        in between  -> falls in a straight line from 1.0 to 0.0
    Example: a 19 mm pitch seen from 4 m is about 16 arcmin, ratio 4, so strength 0. The same
    mosaic from 16 m or more gets strength near 1. Override from the command line with
    --dither on / off / --dither-strength.
    """
    ratio = report.cell_arcmin / consts.blend_arcmin
    return float(min(1.0, max(0.0, (3.0 - ratio) / 2.0)))
