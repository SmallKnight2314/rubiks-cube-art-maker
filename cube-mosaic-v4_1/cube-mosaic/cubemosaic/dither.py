"""dither.py  (v3)  --  error-diffusion dithering ("level B" of the plan).

The idea in one paragraph
-------------------------
Plain nearest-color matching decides every cell alone, so a smooth gradient turns into flat
bands and the small difference between the wanted color and the chosen sticker is simply lost.
Error diffusion keeps that difference: after choosing a sticker for a cell, the leftover error
(wanted minus what the sticker really looks like) is pushed onto the cells that have not been
decided yet, so they get a slightly different target. Over a few cells the errors cancel, and the
AVERAGE color of a neighbourhood ends up close to the photo even though no single sticker is.

Whether this helps depends on the viewer. It only reads as smooth tone if the eye merges
neighbouring stickers, which happens at long viewing distances. Close up it looks like noise.
So the strength is a number from 0 to 1 and the command line sets it from the viewing-distance
check (viewing.dither_strength): 0 means "plain nearest color", 1 means "full dithering".

Method: Floyd-Steinberg with serpentine scanning, done in linear light.
"""
import numpy as np

from .process import Matcher

# Floyd-Steinberg shares each cell's error with 4 neighbours in these proportions:
#            this   7/16 -> next cell in the scan direction
#   3/16 (behind) 5/16 (below) 1/16 (ahead)  -> the row underneath
_W_AHEAD, _W_BELOW_BEHIND, _W_BELOW, _W_BELOW_AHEAD = 7 / 16, 3 / 16, 5 / 16, 1 / 16


def dither_floyd_steinberg(target_lin: np.ndarray, matcher: Matcher,
                           strength: float = 1.0, max_error: float = 0.3) -> np.ndarray:
    """Choose a palette color for every cell, spreading the leftover error to neighbours.

    target_lin  (H, W, 3) wanted colors, linear light (from process.target_linear).
    matcher     decides which sticker is best for a color; the SAME rules (lightness weight,
                hue guard, bias) as plain matching, so blue/green settings carry over.
    strength    0..1. The share of the error that is passed on. 0 = no dithering at all (result
                identical to plain matching); 1 = classic Floyd-Steinberg.
    max_error   largest error (per color channel, linear light) that is ever passed on. Colors the
                palette cannot reach at all (cyan, say) would otherwise pile up huge errors that
                smear streaks across the picture; capping keeps those areas calm.

    Returns an (H, W) array of 0..5, same meaning as Matcher.match_array.
    """
    if strength <= 0:
        return matcher.match_array(target_lin)                 # nothing to diffuse: take the fast path

    h, w, _ = target_lin.shape
    work = target_lin.astype(np.float64).copy()                # target + error received so far; edited as we go
    out = np.empty((h, w), dtype=np.uint8)
    perceived = matcher.palette.perceived_linear               # (6, 3) what each sticker really looks like

    for y in range(h):
        # Serpentine: even rows go left to right, odd rows right to left. Always scanning the same
        # way makes the error drift into diagonal streaks; alternating hides them.
        step = 1 if (y % 2 == 0) else -1
        xs = range(w) if step == 1 else range(w - 1, -1, -1)
        for x in xs:
            wanted = np.clip(work[y, x], 0.0, 1.0)             # target plus incoming error, kept inside 0..1
            # Best sticker for that color. The hue guard is told the ORIGINAL photo color, not the
            # error-adjusted one, so accumulated error can never smuggle in a forbidden color family.
            k = matcher.match_one(wanted, target_lin[y, x])
            out[y, x] = k
            # Error = what we wanted minus what we actually got, scaled and capped.
            err = np.clip((wanted - perceived[k]) * strength, -max_error, max_error)

            # Hand the error to the neighbours that are still undecided (skip anything off the edge).
            xa = x + step                                       # next cell in this row's scan direction
            xb = x - step                                       # the cell behind us
            if 0 <= xa < w:
                work[y, xa] += err * _W_AHEAD
            if y + 1 < h:
                if 0 <= xb < w:
                    work[y + 1, xb] += err * _W_BELOW_BEHIND
                work[y + 1, x] += err * _W_BELOW
                if 0 <= xa < w:
                    work[y + 1, xa] += err * _W_BELOW_AHEAD
    return out
