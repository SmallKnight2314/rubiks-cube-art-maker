"""gui.py  (v4)  --  the PySide6 window.   Start it with:   python -m cubemosaic.gui

This file is only "glue": it reads the widgets into a pipeline.Settings, runs pipeline.run_pipeline
(the same code the command line uses) and shows the Result. All the real work (matching, dithering,
flips, pictures, region maths, saving) lives in plain modules that have their own tests, so if you
want to change HOW the mosaic is made, you will not need to touch this file.

Two windows
-----------
MainWindow     all the parameters (cube, sizes, colors, frame, viewing distance, matching, dithering)
               and a Generate button.
PreviewDialog  the pop-up asking "is this good?": several views of the result, the viewing-distance note,
               the exact color-flip tools (drag a rectangle on the picture, pick X -> Y, Add), and the
               buttons Accept (save files), Adjust (back to the parameters) and Cancel.

Flip rules live in MainWindow.flips. They are kept when you press Adjust and Generate again, so you do
not lose them while tuning other settings. Rules with a rectangle are in sticker coordinates; if you
change the cube type or size limits, the grid changes and old rectangles may no longer match the picture.
"""
import os
import sys

from PIL import Image
from PySide6 import QtCore, QtGui, QtWidgets as W

from .flip import FlipRule
from .geometry import CUBE_PRESETS_MM, default_sticker_mm
from .palette import COLOR_NAMES, DEFAULT_COLORS, FRAME_FRAMELESS, FRAME_OPAQUE, FRAME_TRANSLUCENT, FrameSpec
from .color import hex_to_rgb01
from .pipeline import (SELECTABLE_VIEWS, VIEWS, Result, Settings, region_from_fractions, render_view,
                       run_pipeline, save_outputs, summary_lines)
from .process import MatchSettings
from .zoom import ViewTransform

COLOR_LONG_NAMES = {"W": "White", "Y": "Yellow", "R": "Red", "O": "Orange", "B": "Blue", "G": "Green"}
# Preview banner colors by viewing-report level (same idea as render.add_banner).
LEVEL_STYLES = {"ok": "#1c5c34", "clash": "#963714", "info": "#2d3c55"}


# --------------------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------------------

def pil_to_pixmap(img: Image.Image) -> QtGui.QPixmap:
    """PIL picture -> Qt pixmap (Qt cannot show PIL pictures directly)."""
    rgb = img.convert("RGB")
    data = rgb.tobytes("raw", "RGB")
    qimg = QtGui.QImage(data, rgb.width, rgb.height, 3 * rgb.width, QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(qimg.copy())        # copy() so Qt owns the bytes after `data` goes away


def dspin(lo, hi, value, decimals=2, step=0.1, suffix="") -> W.QDoubleSpinBox:
    """A number box with a range, a number of decimals and an optional unit text such as ' mm'."""
    box = W.QDoubleSpinBox()
    box.setRange(lo, hi)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    box.setSuffix(suffix)
    box.setValue(value)
    return box


def valid_hex(text: str) -> bool:
    """True if `text` is a usable hex color such as '#FFD500' (uses the same parser as the pipeline)."""
    try:
        hex_to_rgb01(text)
        return True
    except ValueError:
        return False


class HexField(W.QWidget):
    """A hex color text box with a clickable swatch. Click the swatch to pick with a color dialog."""

    def __init__(self, value: str):
        super().__init__()
        self.edit = W.QLineEdit(value)
        self.edit.setMaxLength(7)
        self.edit.setFixedWidth(80)
        self.swatch = W.QPushButton()
        self.swatch.setFixedSize(34, 22)
        row = W.QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.edit)
        row.addWidget(self.swatch)
        row.addStretch(1)
        self.edit.textChanged.connect(self._refresh)
        self.swatch.clicked.connect(self._pick)
        self._refresh()

    def text(self) -> str:
        return self.edit.text().strip()

    def _refresh(self):
        """Repaint the swatch; a red border means the text is not a valid hex color."""
        ok = valid_hex(self.text())
        color = self.text() if ok else "#888888"
        border = "1px solid #555" if ok else "2px solid #d33"
        self.swatch.setStyleSheet(f"background:{color}; border:{border};")

    def _pick(self):
        chosen = W.QColorDialog.getColor(QtGui.QColor(self.text() if valid_hex(self.text()) else "#ffffff"), self)
        if chosen.isValid():
            self.edit.setText(chosen.name().upper())


# --------------------------------------------------------------------------------------------------
# the picture with drag-to-select
# --------------------------------------------------------------------------------------------------

class ImageView(W.QWidget):
    """Shows one picture and lets you zoom, pan and select a rectangle of whole stickers.

    Mouse:  wheel = zoom (the spot under the pointer stays put)     right or middle drag = move the picture
            left drag = select a rectangle                          left click = select ONE sticker
    The selection always snaps to whole stickers, so what you see highlighted is exactly what a flip
    will change. When zoomed in far enough a grid appears: thin lines between stickers, thick lines
    between cubes. All zoom/pan maths lives in zoom.ViewTransform (tested without Qt).

    The selection is kept as sticker coordinates (x0, y0, x1, y1), x1/y1 exclusive, or None.
    """
    selection_changed = QtCore.Signal()
    hover = QtCore.Signal(object)          # (sticker_x, sticker_y) under the pointer, or None

    def __init__(self):
        super().__init__()
        self.setMinimumSize(480, 340)
        self.setSizePolicy(W.QSizePolicy.Expanding, W.QSizePolicy.Expanding)
        self.setMouseTracking(True)        # report hover even when no button is pressed
        self._pixmap = None
        self._selectable = False
        self._grid = (1, 1, 1)             # stickers across, stickers down, stickers per cube edge
        self._view = ViewTransform()
        self._drag_start = None            # picture fractions where a left-drag started
        self._pan_last = None              # last pointer position during a right/middle drag
        self._region = None

    # --- API used by PreviewDialog ---------------------------------------------------------------
    def set_grid(self, px_x: int, px_y: int, n: int):
        self._grid = (px_x, px_y, n)
        self.update()

    def set_pixmap(self, pm: QtGui.QPixmap, selectable: bool):
        self._pixmap = pm
        self._selectable = selectable
        if not selectable:
            self._view.reset()             # views that are not the whole mosaic cannot be zoomed meaningfully
        self.setCursor(QtCore.Qt.CrossCursor if selectable else QtCore.Qt.ArrowCursor)
        self.update()

    def region(self):
        """The selected rectangle in sticker coordinates, or None."""
        return self._region

    def clear_selection(self):
        self._region = None
        self.update()
        self.selection_changed.emit()

    def zoom_by(self, factor: float):
        """Zoom around the center of the widget (used by the Zoom buttons)."""
        if self._pixmap is not None and self._selectable:
            self._view.zoom_at(factor, self.width() / 2, self.height() / 2, *self._dims())
            self.update()

    def reset_zoom(self):
        self._view.reset()
        self.update()

    # --- geometry helpers -------------------------------------------------------------------------
    def _dims(self):
        """The four numbers every ViewTransform call needs: widget size and picture size."""
        return self.width(), self.height(), self._pixmap.width(), self._pixmap.height()

    def _rect(self) -> QtCore.QRectF:
        left, top, w, h = self._view.rect(*self._dims())
        return QtCore.QRectF(left, top, w, h)

    def _fraction(self, pos: QtCore.QPointF):
        return self._view.fraction_at(pos.x(), pos.y(), *self._dims())          # clamped to the picture

    def _region_between(self, a, b):
        """Two picture fractions -> the snapped rectangle of whole stickers that covers them."""
        return region_from_fractions(a[0], a[1], b[0], b[1], self._grid[0], self._grid[1])

    # --- mouse ------------------------------------------------------------------------------------
    def wheelEvent(self, event):
        if self._pixmap is None or not self._selectable:
            return
        factor = 1.25 ** (event.angleDelta().y() / 120.0)       # one wheel notch = 1.25x
        self._view.zoom_at(factor, event.position().x(), event.position().y(), *self._dims())
        self.update()
        event.accept()

    def mousePressEvent(self, event):
        if self._pixmap is None:
            return
        if event.button() == QtCore.Qt.LeftButton and self._selectable:
            self._drag_start = self._fraction(event.position())
            self._region = self._region_between(self._drag_start, self._drag_start)   # one sticker, shown at once
            self.update()
        elif event.button() in (QtCore.Qt.RightButton, QtCore.Qt.MiddleButton):
            self._pan_last = event.position()
            self.setCursor(QtCore.Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self._pixmap is None:
            return
        pos = event.position()
        if self._drag_start is not None:
            self._region = self._region_between(self._drag_start, self._fraction(pos))
            self.update()
        elif self._pan_last is not None:
            self._view.pan_by(pos.x() - self._pan_last.x(), pos.y() - self._pan_last.y(), *self._dims())
            self._pan_last = pos
            self.update()
        # Tell the dialog which sticker is under the pointer (None when outside the picture).
        under = self._view.sticker_at(pos.x(), pos.y(), *self._dims(), self._grid[0], self._grid[1]) \
            if self._selectable else None
        self.hover.emit(under)

    def mouseReleaseEvent(self, event):
        if self._drag_start is not None:
            self._drag_start = None
            self.selection_changed.emit()
        if self._pan_last is not None:
            self._pan_last = None
            self.setCursor(QtCore.Qt.CrossCursor if self._selectable else QtCore.Qt.ArrowCursor)

    def leaveEvent(self, event):
        self.hover.emit(None)

    # --- drawing ----------------------------------------------------------------------------------
    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor("#222"))
        if self._pixmap is None:
            return
        rect = self._rect()
        visible = rect.intersected(QtCore.QRectF(self.rect()))      # only the part of the picture on screen
        if visible.isEmpty():
            return
        pw, ph = self._pixmap.width(), self._pixmap.height()
        source = QtCore.QRectF((visible.left() - rect.left()) / rect.width() * pw,
                               (visible.top() - rect.top()) / rect.height() * ph,
                               visible.width() / rect.width() * pw, visible.height() / rect.height() * ph)
        # Smooth when zoomed out; crisp square pixels when zoomed in.
        p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, self._view.zoom < 3.0)
        p.drawPixmap(visible, self._pixmap, source)
        if self._selectable:
            self._draw_grid(p, rect, visible)
            self._draw_region(p, rect)

    def _draw_grid(self, p, rect, visible):
        """Sticker lines (thin) and cube lines (thick), only where they are big enough to be useful."""
        gx, gy, n = self._grid
        cell_w, cell_h = rect.width() / gx, rect.height() / gy
        if min(cell_w, cell_h) < 4:                                  # cubes would already be too small to see
            return
        thin = QtGui.QPen(QtGui.QColor(255, 255, 255, 70), 1)
        thick = QtGui.QPen(QtGui.QColor(255, 255, 255, 190), 2)
        show_thin = min(cell_w, cell_h) >= 9                         # sticker lines only when stickers are clearly visible
        first_x = max(int((visible.left() - rect.left()) / cell_w), 0)
        last_x = min(int((visible.right() - rect.left()) / cell_w) + 1, gx)
        for i in range(first_x, last_x + 1):
            if i % n == 0 or show_thin:
                p.setPen(thick if i % n == 0 else thin)
                x = rect.left() + i * cell_w
                p.drawLine(QtCore.QPointF(x, visible.top()), QtCore.QPointF(x, visible.bottom()))
        first_y = max(int((visible.top() - rect.top()) / cell_h), 0)
        last_y = min(int((visible.bottom() - rect.top()) / cell_h) + 1, gy)
        for j in range(first_y, last_y + 1):
            if j % n == 0 or show_thin:
                p.setPen(thick if j % n == 0 else thin)
                y = rect.top() + j * cell_h
                p.drawLine(QtCore.QPointF(visible.left(), y), QtCore.QPointF(visible.right(), y))

    def _draw_region(self, p, rect):
        """The selected stickers: a dashed outline with a light fill."""
        if self._region is None:
            return
        gx, gy, _ = self._grid
        x0, y0, x1, y1 = self._region
        box = QtCore.QRectF(rect.left() + x0 / gx * rect.width(), rect.top() + y0 / gy * rect.height(),
                            (x1 - x0) / gx * rect.width(), (y1 - y0) / gy * rect.height())
        p.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 2, QtCore.Qt.DashLine))
        p.setBrush(QtGui.QColor(255, 255, 255, 40))
        p.drawRect(box)


# --------------------------------------------------------------------------------------------------
# the approve / adjust pop-up
# --------------------------------------------------------------------------------------------------

class PreviewDialog(W.QDialog):
    """Shows the result and asks whether it is good. exec() returns:
        1 -> Accept (save the files)      2 -> Adjust (back to the parameters)      0 -> Cancel
    `flips` is the list of FlipRule objects owned by the main window; this dialog edits it in place.
    """
    ADJUST = 2
    HELP = "Wheel: zoom   Right-drag: move   Left-drag: select stickers   Click: one sticker"

    def __init__(self, result: Result, flips: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Is this good?")
        self.resize(1180, 760)
        self.base = result                  # the result as it came out of the pipeline
        self.flips = flips                  # shared, edited in place
        # NOTE: the current result is called `current`, not `result`, because QDialog already has a result() method.
        self.current = result.with_flips(flips)

        # --- top: the viewing-distance note, colored like the banner in the saved preview ---
        self.note = W.QLabel("\n".join(result.report.lines))
        self.note.setWordWrap(True)
        self.note.setStyleSheet(f"background:{LEVEL_STYLES.get(result.report.level, '#2d3c55')};"
                                "color:white; padding:8px; border-radius:4px;")

        # --- left: view chooser and the picture ---
        self.view_box = W.QComboBox()
        self.view_box.addItems(VIEWS)
        self.image = ImageView()
        self.image.hover.connect(self._on_hover)

        left = W.QVBoxLayout()
        row = W.QHBoxLayout()
        row.addWidget(W.QLabel("View:"))
        row.addWidget(self.view_box)
        row.addStretch(1)
        self.hint = W.QLabel("")
        self.hint.setStyleSheet("color:#888;")
        row.addWidget(self.hint)
        for text, slot in (("Zoom +", lambda: self.image.zoom_by(1.5)), ("Zoom -", lambda: self.image.zoom_by(1 / 1.5)),
                           ("Fit", lambda: self.image.reset_zoom())):
            button = W.QPushButton(text)
            button.clicked.connect(slot)
            row.addWidget(button)
        left.addLayout(row)
        left.addWidget(self.image, 1)
        self.hover_label = W.QLabel(self.HELP)                       # shows the sticker under the pointer
        self.hover_label.setStyleSheet("color:#888;")
        left.addWidget(self.hover_label)

        # --- right: facts, flip tools, buttons ---
        self.info = W.QLabel()
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        self.src_box, self.dst_box = W.QComboBox(), W.QComboBox()
        # The source box can also say "any color": that PAINTS the selection with the target color.
        # Each entry carries its color number as data (None for "any"), read back with currentData().
        self.src_box.addItem("Any color (paint)", None)
        for i, name in enumerate(COLOR_NAMES):
            self.src_box.addItem(f"{name}  {COLOR_LONG_NAMES[name]}", i)
            self.dst_box.addItem(f"{name}  {COLOR_LONG_NAMES[name]}", i)
        self.src_box.setCurrentIndex(1 + COLOR_NAMES.index("B"))      # +1 because "Any color" is the first entry
        self.dst_box.setCurrentIndex(COLOR_NAMES.index("G"))
        self.region_label = W.QLabel("Whole mosaic (drag on the picture to pick a rectangle instead)")
        self.region_label.setWordWrap(True)
        add_btn, clear_btn = W.QPushButton("Add flip"), W.QPushButton("Clear rectangle")
        self.rules = W.QListWidget()
        remove_btn = W.QPushButton("Remove selected rule")

        flip_box = W.QGroupBox("Exact color flip / paint")
        fl = W.QVBoxLayout(flip_box)
        pair = W.QHBoxLayout()
        pair.addWidget(self.src_box)
        pair.addWidget(W.QLabel("becomes"))
        pair.addWidget(self.dst_box)
        fl.addLayout(pair)
        fl.addWidget(self.region_label)
        btns = W.QHBoxLayout()
        btns.addWidget(add_btn)
        btns.addWidget(clear_btn)
        fl.addLayout(btns)
        fl.addWidget(W.QLabel("Rules (untick to switch one off):"))
        fl.addWidget(self.rules, 1)
        fl.addWidget(remove_btn)

        accept_btn, adjust_btn, cancel_btn = W.QPushButton("Accept and save"), W.QPushButton("Adjust"), W.QPushButton("Cancel")
        accept_btn.setDefault(True)
        bottom = W.QHBoxLayout()
        bottom.addWidget(accept_btn)
        bottom.addWidget(adjust_btn)
        bottom.addWidget(cancel_btn)

        right = W.QVBoxLayout()
        right.addWidget(self.info)
        right.addWidget(flip_box, 1)
        right.addLayout(bottom)
        right_w = W.QWidget()
        right_w.setLayout(right)
        right_w.setFixedWidth(330)

        body = W.QHBoxLayout()
        body.addLayout(left, 1)
        body.addWidget(right_w)
        outer = W.QVBoxLayout(self)
        outer.addWidget(self.note)
        outer.addLayout(body, 1)

        # --- wiring ---
        self.view_box.currentIndexChanged.connect(self.refresh_view)
        self.image.selection_changed.connect(self._update_region_label)
        add_btn.clicked.connect(self.add_flip)
        clear_btn.clicked.connect(self.image.clear_selection)
        remove_btn.clicked.connect(self.remove_selected_rules)
        self.rules.itemChanged.connect(self._rule_toggled)
        accept_btn.clicked.connect(self.accept)
        adjust_btn.clicked.connect(lambda: self.done(self.ADJUST))
        cancel_btn.clicked.connect(self.reject)

        self.refresh_rules()
        self.refresh_all()

    # --- refreshing ---
    def refresh_all(self):
        """Re-apply the flips (instant, no new run), then redraw facts and picture."""
        self.current = self.base.with_flips(self.flips)
        r, g = self.current, self.current.grid
        counts = "  ".join(f"{n} {int((r.indices == i).sum())}" for i, n in enumerate(COLOR_NAMES))
        text = (f"{g.cubes_x} x {g.cubes_y} cubes = {g.cube_count} cubes\n"
                f"{g.px_x} x {g.px_y} stickers\n"
                f"{g.width_mm / 1000:.3f} x {g.height_mm / 1000:.3f} m\n\n"
                f"Stickers (after flips):\n{counts}")
        changed = int((r.indices != r.matched).sum())
        if changed:
            text += f"\n\n{changed} stickers changed by flips"
        self.info.setText(text)
        self.refresh_view()

    def refresh_view(self):
        name = self.view_box.currentText()
        selectable = name in SELECTABLE_VIEWS
        self.image.set_grid(self.current.grid.px_x, self.current.grid.px_y, self.current.grid.n)
        self.image.set_pixmap(pil_to_pixmap(render_view(self.current, name)), selectable)
        self.hint.setText("" if selectable else "(rectangles can be drawn in the other views)")

    def refresh_rules(self):
        """Rebuild the checkable list from self.flips."""
        self.rules.blockSignals(True)              # do not fire itemChanged while we fill the list
        self.rules.clear()
        for rule in self.flips:
            item = W.QListWidgetItem(rule.label())
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked if rule.enabled else QtCore.Qt.Unchecked)
            self.rules.addItem(item)
        self.rules.blockSignals(False)

    def _on_hover(self, pos):
        """Show which sticker (and which cube) is under the pointer, and its current color."""
        if pos is None:
            self.hover_label.setText(self.HELP)
            return
        x, y = pos
        n = self.current.grid.n
        color = COLOR_NAMES[int(self.current.indices[y, x])]
        self.hover_label.setText(f"sticker ({x}, {y}) is {color}   |   cube column {x // n + 1}, row {y // n + 1}")

    # --- flip tools ---
    def _current_region(self):
        """The dragged rectangle in sticker coordinates, or None for the whole mosaic."""
        return self.image.region()                                   # already snapped to whole stickers

    def _update_region_label(self):
        region = self._current_region()
        self.region_label.setText("Whole mosaic (drag on the picture to pick a rectangle instead)" if region is None
                                  else f"Selected: x {region[0]}..{region[2]}, y {region[1]}..{region[3]} "
                                       f"= {region[2] - region[0]} x {region[3] - region[1]} stickers")

    def add_flip(self):
        src, dst = self.src_box.currentData(), self.dst_box.currentData()     # src is None for "any color"
        if src == dst:
            W.QMessageBox.information(self, "Flip", "Pick two different colors.")
            return
        self.flips.append(FlipRule(src, dst, self._current_region()))
        self.image.clear_selection()
        self.refresh_rules()
        self.refresh_all()

    def remove_selected_rules(self):
        rows = sorted({self.rules.row(i) for i in self.rules.selectedItems()}, reverse=True)   # last first
        for row in rows:
            del self.flips[row]
        self.refresh_rules()
        self.refresh_all()

    def _rule_toggled(self, item: W.QListWidgetItem):
        self.flips[self.rules.row(item)].enabled = item.checkState() == QtCore.Qt.Checked
        self.refresh_all()


# --------------------------------------------------------------------------------------------------
# the worker thread (so the window does not freeze while the mosaic is computed)
# --------------------------------------------------------------------------------------------------

class Worker(QtCore.QThread):
    done = QtCore.Signal(object)       # emits the Result
    failed = QtCore.Signal(str)        # emits a readable error message

    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings

    def run(self):
        try:
            self.done.emit(run_pipeline(self.settings))
        except Exception as err:       # show every problem in the window instead of crashing the program
            self.failed.emit(str(err))


# --------------------------------------------------------------------------------------------------
# the main window
# --------------------------------------------------------------------------------------------------

class MainWindow(W.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cube mosaic generator")
        self.resize(760, 900)
        self.flips = []                    # FlipRule list, edited in the preview dialog
        self.last_settings = None          # for reusing the last result when only the flips changed
        self.last_result = None
        self.worker = None
        self._syncing = False              # True while sticker and padding boxes update each other

        page = W.QWidget()
        col = W.QVBoxLayout(page)
        for group in (self._image_group(), self._cube_group(), self._colors_group(), self._frame_group(),
                      self._viewing_group(), self._matching_group(), self._output_group()):
            col.addWidget(group)
        col.addStretch(1)
        scroll = W.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)

        self.generate_btn = W.QPushButton("Generate")
        self.generate_btn.setDefault(True)
        self.generate_btn.clicked.connect(self.generate)
        self.status = W.QLabel("Choose a photo, then press Generate.")
        self.status.setWordWrap(True)
        central = W.QWidget()
        outer = W.QVBoxLayout(central)
        outer.addWidget(scroll, 1)
        outer.addWidget(self.status)
        outer.addWidget(self.generate_btn)
        self.setCentralWidget(central)
        self._on_cube_type_changed()
        self._on_frame_mode_changed()

    # --- building the parameter groups -------------------------------------------------------
    def _image_group(self):
        box = W.QGroupBox("Photo")
        row = W.QHBoxLayout(box)
        self.image_path = W.QLineEdit()
        self.image_path.setPlaceholderText("png, jpg, jpeg, ...")
        browse = W.QPushButton("Browse...")
        browse.clicked.connect(self._browse_image)
        row.addWidget(self.image_path, 1)
        row.addWidget(browse)
        return box

    def _cube_group(self):
        box = W.QGroupBox("Cube and size")
        form = W.QFormLayout(box)
        self.cube_type = W.QComboBox()
        for n in sorted(CUBE_PRESETS_MM):
            self.cube_type.addItem(f"{n}x{n}", n)
        self.cube_type.setCurrentIndex(1)                                   # 3x3
        self.cube_mm = dspin(10, 200, 57.0, 1, 0.5, " mm")
        self.sticker_mm = dspin(1, 100, 15.0, 2, 0.1, " mm")
        self.padding_mm = dspin(0, 100, 4.0, 2, 0.1, " mm")
        self.pitch_label = W.QLabel()
        self.max_w = dspin(0.05, 100, 2.0, 2, 0.1, " m")
        self.max_h = dspin(0.05, 100, 1.5, 2, 0.1, " m")
        form.addRow("Cube type", self.cube_type)
        form.addRow("Cube size", self.cube_mm)
        form.addRow("Sticker size", self.sticker_mm)
        form.addRow("Padding (frame gap)", self.padding_mm)
        form.addRow("Pitch (cube / n)", self.pitch_label)
        form.addRow("Maximum width", self.max_w)
        form.addRow("Maximum height", self.max_h)
        self.cube_type.currentIndexChanged.connect(self._on_cube_type_changed)
        self.cube_mm.valueChanged.connect(self._on_cube_mm_changed)
        self.sticker_mm.valueChanged.connect(self._on_sticker_changed)
        self.padding_mm.valueChanged.connect(self._on_padding_changed)
        return box

    def _colors_group(self):
        box = W.QGroupBox("Sticker colors (hex)")
        form = W.QFormLayout(box)
        self.color_fields = {}
        for name in COLOR_NAMES:
            self.color_fields[name] = HexField(DEFAULT_COLORS[name])
            form.addRow(COLOR_LONG_NAMES[name], self.color_fields[name])
        return box

    def _frame_group(self):
        box = W.QGroupBox("Frame (plastic around the stickers)")
        form = W.QFormLayout(box)
        self.frame_mode = W.QComboBox()
        self.frame_mode.addItem("Opaque", FRAME_OPAQUE)
        self.frame_mode.addItem("Translucent", FRAME_TRANSLUCENT)
        self.frame_mode.addItem("Frameless / stickerless", FRAME_FRAMELESS)
        self.frame_hex = HexField("#000000")
        self.frame_opacity = dspin(0, 1, 1.0, 2, 0.05)
        self.backdrop_hex = HexField("#000000")
        self.seam_mm = dspin(0, 5, 0.6, 2, 0.1, " mm")
        self.seam_hex = HexField("#000000")
        self.frame_rows = {}                                               # remembered so rows can be greyed out
        for key, label, widget in (("hex", "Frame color", self.frame_hex), ("opacity", "Opacity (1 = solid)", self.frame_opacity),
                                   ("backdrop", "Color seen through it", self.backdrop_hex),
                                   ("seam", "Seam width", self.seam_mm), ("seam_hex", "Seam color", self.seam_hex)):
            form.addRow(label, widget)
            self.frame_rows[key] = widget
        form.insertRow(0, "Frame type", self.frame_mode)
        self.frame_mode.currentIndexChanged.connect(self._on_frame_mode_changed)
        return box

    def _viewing_group(self):
        box = W.QGroupBox("Viewing distance")
        row = W.QHBoxLayout(box)
        self.view_auto = W.QCheckBox("Use the recommended distance")
        self.view_auto.setChecked(True)
        self.view_m = dspin(0.1, 200, 4.0, 1, 0.5, " m")
        self.view_m.setEnabled(False)
        self.view_auto.toggled.connect(lambda on: self.view_m.setEnabled(not on))
        row.addWidget(self.view_auto)
        row.addWidget(self.view_m)
        return box

    def _matching_group(self):
        box = W.QGroupBox("Color matching and dithering")
        form = W.QFormLayout(box)
        self.lightness_weight = dspin(0.0, 2.0, 1.0, 2, 0.1)
        self.hue_guard = dspin(0, 180, 0, 0, 5, " deg")
        self.guard_min_chroma = dspin(0, 0.5, 0.04, 3, 0.01)
        self.bias_boxes = [dspin(-1, 1, 0.0, 4, 0.002) for _ in COLOR_NAMES]
        self.neutral_boxes = [dspin(-1, 1, 0.0, 4, 0.002) for _ in COLOR_NAMES]
        self.dither_mode = W.QComboBox()
        self.dither_mode.addItem("Automatic (from viewing distance)", "auto")
        self.dither_mode.addItem("Off", "off")
        self.dither_mode.addItem("Full strength", "on")
        self.dither_override = W.QCheckBox("Set the strength myself")
        self.dither_value = dspin(0, 1, 0.5, 2, 0.05)
        self.dither_value.setEnabled(False)
        self.dither_override.toggled.connect(self.dither_value.setEnabled)
        form.addRow("Lightness weight (1 = plain)", self.lightness_weight)
        form.addRow("Hue guard (0 = off, try 70)", self.hue_guard)
        form.addRow("Guard ignores grays below chroma", self.guard_min_chroma)
        form.addRow(W.QLabel("Per-color nudge (positive = used less). Order: " + " ".join(COLOR_NAMES)))
        form.addRow("All targets", self._row_of(self.bias_boxes))
        form.addRow("Near-gray targets only", self._row_of(self.neutral_boxes))
        form.addRow("Dithering", self.dither_mode)
        form.addRow(self.dither_override, self.dither_value)
        return box

    def _row_of(self, boxes):
        holder = W.QWidget()
        row = W.QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        for b in boxes:
            b.setMinimumWidth(78)                                              # was setMaximumWidth(62)
            b.setButtonSymbols(W.QAbstractSpinBox.ButtonSymbols.NoButtons)     # hides the arrows, gives the digits the room
            row.addWidget(b)
        return holder

    def _output_group(self):
        box = W.QGroupBox("Output folder (written when you press Accept)")
        row = W.QHBoxLayout(box)
        self.out_dir = W.QLineEdit(os.path.join(os.getcwd(), "out"))
        browse = W.QPushButton("Browse...")
        browse.clicked.connect(self._browse_out)
        row.addWidget(self.out_dir, 1)
        row.addWidget(browse)
        return box

    # --- keeping linked boxes consistent -------------------------------------------------------
    def _n(self) -> int:
        return self.cube_type.currentData()

    def _pitch(self) -> float:
        return self.cube_mm.value() / self._n()

    def _update_pitch_label(self):
        self.pitch_label.setText(f"{self._pitch():.2f} mm")

    def _on_cube_type_changed(self):
        """Choosing a cube type loads its usual size and a proportional sticker."""
        n = self._n()
        self.cube_mm.blockSignals(True)
        self.cube_mm.setValue(CUBE_PRESETS_MM[n])
        self.cube_mm.blockSignals(False)
        self._set_sticker(default_sticker_mm(n, CUBE_PRESETS_MM[n]))

    def _on_cube_mm_changed(self):
        """A new cube size keeps the same sticker/pitch proportion."""
        ratio = self.sticker_mm.value() / max(self._last_pitch, 1e-9) if hasattr(self, "_last_pitch") else 15 / 19
        self._set_sticker(self._pitch() * min(ratio, 1.0))

    def _set_sticker(self, sticker: float):
        """Set the sticker size, then derive padding = pitch - sticker."""
        self._syncing = True
        self.sticker_mm.setValue(sticker)
        self.padding_mm.setValue(max(self._pitch() - sticker, 0.0))
        self._syncing = False
        self._last_pitch = self._pitch()
        self._update_pitch_label()

    def _on_sticker_changed(self):
        if not self._syncing:
            self._syncing = True
            self.padding_mm.setValue(max(self._pitch() - self.sticker_mm.value(), 0.0))
            self._syncing = False
            self._last_pitch = self._pitch()

    def _on_padding_changed(self):
        if not self._syncing:
            self._syncing = True
            self.sticker_mm.setValue(max(self._pitch() - self.padding_mm.value(), 0.1))
            self._syncing = False
            self._last_pitch = self._pitch()

    def _on_frame_mode_changed(self):
        """Grey out the frame fields that the chosen frame type does not use."""
        mode = self.frame_mode.currentData()
        used = {FRAME_OPAQUE: {"hex"}, FRAME_TRANSLUCENT: {"hex", "opacity", "backdrop"},
                FRAME_FRAMELESS: {"seam", "seam_hex"}}[mode]
        for key, widget in self.frame_rows.items():
            widget.setEnabled(key in used)

    # --- file pickers ---------------------------------------------------------------------------
    def _browse_image(self):
        path, _ = W.QFileDialog.getOpenFileName(self, "Choose a photo", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)")
        if path:
            self.image_path.setText(path)

    def _browse_out(self):
        path = W.QFileDialog.getExistingDirectory(self, "Choose the output folder", self.out_dir.text())
        if path:
            self.out_dir.setText(path)

    # --- reading the widgets into Settings ---------------------------------------------------------
    def collect_settings(self) -> Settings:
        """Widgets -> Settings. Raises ValueError with a readable message for bad input."""
        path = self.image_path.text().strip()
        if not path or not os.path.isfile(path):
            raise ValueError("Choose a photo first (the file was not found).")
        colors = {}
        for name, field in self.color_fields.items():
            if not valid_hex(field.text()):
                raise ValueError(f"The {COLOR_LONG_NAMES[name]} color is not a valid hex code like #FFD500.")
            colors[name] = field.text()
        for label, field in (("Frame color", self.frame_hex), ("Color seen through the frame", self.backdrop_hex),
                             ("Seam color", self.seam_hex)):
            if not valid_hex(field.text()):
                raise ValueError(f"{label} is not a valid hex code.")
        frame = FrameSpec(self.frame_mode.currentData(), self.frame_hex.text(), self.frame_opacity.value(),
                          self.backdrop_hex.text(), self.seam_mm.value(), self.seam_hex.text())
        match = MatchSettings(self.lightness_weight.value(), self.hue_guard.value(), self.guard_min_chroma.value(),
                              tuple(b.value() for b in self.bias_boxes), tuple(b.value() for b in self.neutral_boxes))
        return Settings(
            image_path=path, n=self._n(), cube_mm=self.cube_mm.value(), sticker_mm=self.sticker_mm.value(),
            max_w_m=self.max_w.value(), max_h_m=self.max_h.value(), colors=colors, frame=frame, match=match,
            view_m=None if self.view_auto.isChecked() else self.view_m.value(),
            dither=self.dither_mode.currentData(),
            dither_strength=self.dither_value.value() if self.dither_override.isChecked() else None,
            flips=self.flips)

    # --- running and showing -------------------------------------------------------------------------
    def generate(self):
        try:
            settings = self.collect_settings()
        except ValueError as err:
            W.QMessageBox.warning(self, "Check the settings", str(err))
            return
        # Only the flips changed since the last run? Then no new image processing is needed.
        if self.last_result is not None and self.last_settings == settings.without_flips():
            self._show(self.last_result)
            return
        self.last_settings = settings.without_flips()
        self.generate_btn.setEnabled(False)
        self.status.setText("Working...")
        W.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        self.worker = Worker(settings)
        self.worker.done.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _finish_run(self):
        W.QApplication.restoreOverrideCursor()
        self.generate_btn.setEnabled(True)

    def _on_failed(self, message: str):
        self._finish_run()
        self.last_result = self.last_settings = None
        self.status.setText("Failed.")
        W.QMessageBox.critical(self, "Could not generate", message)

    def _on_done(self, result: Result):
        self._finish_run()
        self.last_result = result
        self._show(result)

    def _show(self, result: Result):
        """Open the approve/adjust pop-up and act on the answer."""
        dialog = PreviewDialog(result, self.flips, self)
        answer = int(dialog.exec())           # 1 = Accept, 2 = Adjust, 0 = Cancel (plain numbers on purpose)
        if answer == 1:
            names, _ = save_outputs(dialog.current, self.out_dir.text().strip() or "out")
            self.status.setText(f"Saved to {self.out_dir.text()}: " + ", ".join(names)
                                + "\nReady for the PDF blueprint (v5).")
        elif answer == PreviewDialog.ADJUST:
            self.status.setText("Adjust the settings and press Generate again. Your flip rules are kept.")
        else:
            self.status.setText("Cancelled.")


def main():
    app = W.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
