"""zoom.py  (v4.1)  --  the zoom and pan maths for the preview picture, with no Qt in it.

The preview shows one picture inside a widget (a window area). At zoom 1.0 the picture is scaled to
fit the widget and centered. Zooming in magnifies it; panning moves which part of the picture is at the
center of the widget. Keeping this maths in a plain class means it can be tested without a screen.

Vocabulary
----------
ww, wh   widget width/height in pixels (the area the picture is shown in)
pw, ph   picture width/height in pixels (only its shape matters, not its size)
zoom     1.0 = fit the whole picture in the widget; 2.0 = twice as big; and so on
cx, cy   which point of the picture (as a fraction 0..1 of its width/height) sits at the widget's center
"""


class ViewTransform:
    MIN_ZOOM = 1.0       # cannot zoom out past "fit the whole picture"
    MAX_ZOOM = 60.0      # enough to make one sticker fill most of a window

    def __init__(self):
        self.zoom = 1.0
        self.cx = 0.5
        self.cy = 0.5

    def reset(self):
        """Back to 'fit the whole picture, centered'."""
        self.zoom, self.cx, self.cy = 1.0, 0.5, 0.5

    @staticmethod
    def _fit(ww, wh, pw, ph):
        """Size of the picture when scaled to just fit the widget (zoom 1.0)."""
        scale = min(ww / pw, wh / ph)
        return pw * scale, ph * scale

    def rect(self, ww, wh, pw, ph):
        """Where the picture is drawn: (left, top, width, height) in widget pixels. Can extend past the widget."""
        fw, fh = self._fit(ww, wh, pw, ph)
        w, h = fw * self.zoom, fh * self.zoom
        # The picture point (cx, cy) must land on the widget center (ww/2, wh/2).
        return ww / 2 - self.cx * w, wh / 2 - self.cy * h, w, h

    def fraction_at(self, x, y, ww, wh, pw, ph, clamp=True):
        """Widget pixel (x, y) -> position in the picture as fractions 0..1 (clamped to the picture by default)."""
        left, top, w, h = self.rect(ww, wh, pw, ph)
        fx, fy = (x - left) / w, (y - top) / h
        if clamp:
            fx, fy = min(max(fx, 0.0), 1.0), min(max(fy, 0.0), 1.0)
        return fx, fy

    def sticker_at(self, x, y, ww, wh, pw, ph, grid_w, grid_h):
        """Widget pixel -> (sticker_x, sticker_y) of a grid_w x grid_h mosaic, or None if outside the picture."""
        fx, fy = self.fraction_at(x, y, ww, wh, pw, ph, clamp=False)
        if not (0.0 <= fx < 1.0 and 0.0 <= fy < 1.0):
            return None
        return int(fx * grid_w), int(fy * grid_h)

    def zoom_at(self, factor, ax, ay, ww, wh, pw, ph):
        """Multiply the zoom by `factor`, keeping the picture point under widget pixel (ax, ay) where it is.

        This is what makes wheel-zoom feel natural: the spot under the mouse pointer stays under it.
        """
        fx, fy = self.fraction_at(ax, ay, ww, wh, pw, ph, clamp=False)       # picture point under the pointer
        new_zoom = min(max(self.zoom * factor, self.MIN_ZOOM), self.MAX_ZOOM)
        fw, fh = self._fit(ww, wh, pw, ph)
        w, h = fw * new_zoom, fh * new_zoom
        # Solve  ax = (ww/2 - cx*w) + fx*w  for cx (and the same vertically).
        self.cx = (ww / 2 + fx * w - ax) / w
        self.cy = (wh / 2 + fy * h - ay) / h
        self.zoom = new_zoom
        self._clamp()

    def pan_by(self, dx, dy, ww, wh, pw, ph):
        """Drag the picture by (dx, dy) widget pixels."""
        _, _, w, h = self.rect(ww, wh, pw, ph)
        self.cx -= dx / w
        self.cy -= dy / h
        self._clamp()

    def _clamp(self):
        """Keep the view center on the picture; at zoom 1.0 the picture is always centered."""
        if self.zoom <= 1.0:
            self.cx = self.cy = 0.5
        else:
            self.cx = min(max(self.cx, 0.0), 1.0)
            self.cy = min(max(self.cy, 0.0), 1.0)
