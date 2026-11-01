# Cube mosaic generator (v0 to v4: command line + GUI)

    pip install -r requirements.txt
    python -m cubemosaic.cli photo.jpg --n 3 --max-w-m 2.0 --max-h-m 1.5 --view-m 4 --out-dir out
    python tests/test_geometry.py ; python tests/test_viewing.py ; python tests/test_matching.py ; python tests/test_flip.py ; python tests/test_zoom.py

Outputs in the out folder:
- preview.png        the mosaic with a note on top about the viewing distance (green ok, orange clash, slate none given)
- preview_clean.png  the mosaic alone
- perceived.png      blended colors, roughly the view from far away
- compare.png        (with --compare) plain v2 vs your settings vs a map of changed stickers
- indices.npy        color index 0..5 per sticker, order W Y R O B G

Blue/green and other color controls (all off by default):
    --hue-guard-deg 70   --lightness-weight 0.5   --bias G=0.004   --neutral-bias G=0.02   --compare
See what they do before running a photo:
    python -m cubemosaic.diagnose --hue-guard-deg 70

Dithering: --dither auto|off|on (auto follows the viewing distance), --dither-strength 0..1

Frame: --frame-mode opaque|translucent|frameless, --frame-hex, --frame-opacity, --backdrop-hex, --seam-mm
Colors: --color-w / -y / -r / -o / -b / -g (hex)

See NOTES.md for the version plan, findings and a "which file do I edit" table.

GUI (v4): `pip install PySide6` then `python -m cubemosaic.gui`
  Fill in the parameters, press Generate, and the pop-up shows the result with views, the viewing-distance note and the
  exact color-flip tools. Accept saves the files, Adjust returns to the parameters (flip rules are kept), Cancel discards.

Exact color flip on the command line: --flip B=G   --flip G=B   (together = swap)   --flip G=R@0,0,40,20  (rectangle)

Preview pop-up: wheel = zoom, right-drag = move, left-drag = select whole stickers, click = one sticker.
Paint a selection with any color: choose "Any color (paint)" as the flip source (command line: --flip "*=G@x0,y0,x1,y1").
A plain-text guide to every setting is in GUIDE.txt.
