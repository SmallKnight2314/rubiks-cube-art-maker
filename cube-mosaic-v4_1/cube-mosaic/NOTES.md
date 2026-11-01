# Cube mosaic generator: notes

## Version plan

| Version | Status | What it does |
|---|---|---|
| v0 | done | Cube presets and grid maths (`geometry.py`) |
| v1 | done, tested on real images | Linear-light downsample, frame-aware palette, nearest color, previews |
| v2 | done | Viewing-distance estimate and clash note on the preview (`viewing.py`) |
| v3a | done | Color mix-up controls (`process.py` Matcher), `diagnose.py`, `compare.py` |
| v3b | done | Floyd-Steinberg dithering (`dither.py`), strength set by viewing distance |
| v3c | next | Blur-based optimizer: simulate the viewing distance and improve the result |
| v4 / v4.1 | built; GUI runs on the user's machine | PySide6 GUI (`gui.py`) with approve / adjust pop-up, plus the exact color flip (`flip.py`) and the shared `pipeline.py`. Everything except the Qt window itself is tested |
| v5 | planned | A4 PDF blueprint: 6 neighbouring cubes per page, edge coordinates, mini-map |
| v6 | planned | Presets, save/load project, speed |
| v7 | idea | Interactive guide pages: a fixed demo picture next to a live generated one, so each control's effect can be seen. `GUIDE.txt` stands in until then |

## Fixed decisions

- Exactly the 6 colors on every cell. No missing stickers, no extra colors.
- Cube types 2x2 to 7x7 only.
- Frame: opaque, translucent (opacity + backdrop) or frameless.
- 6 neighbouring cubes per A4 page. No scramble instructions.
- GUI toolkit: PySide6.

## Findings that shape v3

- At realistic viewing distances one sticker cell is far larger than the eye can merge
  (19 mm pitch is about 16 arcmin at 4 m; merging needs about 4 arcmin, around 16 m).
  So heavy dithering will look grainy. v3 should tie dithering strength to
  `ViewingReport.cell_arcmin` / `blend_state` instead of using one fixed strength.

## Blue / green mix-ups: what we found (v3)

Cause, found with `diagnose.py`: with a black frame the perceived palette is dark, and plain matching mostly compares
lightness. So mid-light blues (sky, jeans) get GREEN, which is closer in lightness than the dark blue. Gray mid-tones
get green too. Cyan/teal is not in the palette at all and goes to whichever of blue/green is closer in lightness.

Controls (all off by default = exact v2 result), each testable with `--compare`:
- `--hue-guard-deg 70`: a colorful target may only get a sticker within 70 degrees of its hue (white always allowed).
  Fixes blue-ish targets turning green. Pale blue becomes white instead, which looks right.
- `--lightness-weight 0.4..0.7`: lightness counts less than color family. Mild, use with the guard.
- `--bias G=0.004` : custom flip rule, a nudge for or against one color (positive = less likely).
- `--neutral-bias G=0.02`: same, only for near-gray targets. Grays have no hue, so the guard cannot help; moving
  them off green sends them to orange or red, so which cast you prefer is a taste decision.
- Dithering uses the same rules, and the guard judges the ORIGINAL photo color, not photo plus error.

Still open: teal/cyan areas legitimately mix blue and green (the guard allows both near 195 degrees). Consider a
user choice "teal goes to blue only / green only / mixed" once you see it on real photos.

## Exact color flip (decided: build it in v4, with the GUI)

Why: a big `--bias` only means "never use this color". The replacement is whatever is next best for each cell.
Tested on the sky/hill/sea scene: `--bias B=1.0` sent all blue to green (clean), but `--bias G=1.0` sent green to
blue, yellow, white AND orange. So v3 cannot do a true flip.

Spec for v4:
- A step that runs AFTER matching/dithering and rewrites sticker colors exactly: "every X becomes Y".
- Optional region: the whole mosaic, or a rectangle drawn on the preview (so only the sky can be flipped).
- Several rules allowed, applied in order. Two rules (B to G and G to B) are applied at the same time, so they swap
  instead of chaining.
- Always shown against the unflipped result with the changed-stickers map and counts (reuse `compare.py`), with a
  toggle to accept or undo each rule.
- Colors stay inside the six, so nothing can go missing.
- The sticker counts and the blueprint (v5) must be computed from the FLIPPED result.
- Command-line equivalent: `--flip B=G` and `--flip G=B@x0,y0,x1,y1`, so scripts and the GUI share one code path.

## v4 status and what to check first

- `pipeline.py` is now the only place a run is assembled; the command line and the GUI both call it.
- The GUI could not be run where it was written (no network access to install PySide6 there). The Qt window code is
  therefore untested; its logic (flips, region maths, views, saving) is in `flip.py` / `pipeline.py` and IS tested.
- First run: `pip install PySide6` then `python -m cubemosaic.gui`. If anything errors, send the traceback; the
  likely trouble spots are small Qt details in `gui.py` (enum names, layout calls), not the mosaic logic.

## v4.1 additions

- Preview zoom and pan (`zoom.py` holds the tested maths; `ImageView` in `gui.py` draws it), selection snapped to whole
  stickers, sticker and cube grid when zoomed, hover readout (sticker, color, cube column and row).
- "Any color (paint)" flip source: `*=G@x0,y0,x1,y1` repaints the selected stickers, down to a single sticker.
- Not built (ideas): painting several single stickers in one drag (brush), undo history beyond the rules list,
  selecting whole cubes by clicking them.

## Where to change what

| I want to change... | File |
|---|---|
| Cube sizes, grid choice, crop limit | `geometry.py` |
| Blue/green and other matching rules | `process.py` (class `MatchSettings`, `Matcher`) |
| Dithering | `dither.py` |
| Exact color flips and paint | `flip.py` |
| Zoom and pan maths | `zoom.py` |
| How a run is assembled, summary text, saved files | `pipeline.py` |
| Windows, buttons, layout | `gui.py` |
| What changed between two settings | `compare.py` |
| Which sticker each hue/lightness gets | `diagnose.py` |
| Default six colors, frame behaviour | `palette.py` |
| How the photo is shrunk or colors are matched | `process.py` |
| Viewing-distance angles and blend threshold | `viewing.py` (class `ViewingConstants`) |
| Preview look, banner colors | `render.py` |
| Command-line options | `cli.py` |
