"""
The instruction box at the top of every camera window: it always says what to
do next, so you do not need to remember the keys.

Remember: a window only receives your key presses after you click on it.
"""
import cv2
import numpy as np

GREEN, RED, BLUE, ORANGE, GREY = (0, 160, 0), (0, 0, 220), (200, 80, 0), (0, 140, 255), (90, 90, 90)


def instruction_box(image, title, lines=(), colour=BLUE):
    """Return the image with a white instruction panel ABOVE it (nothing on the picture is covered):
    a coloured title and grey lines. The picture starts at row panel_height(...) of the result."""
    shown = image.copy()
    if shown.ndim == 2:
        shown = cv2.cvtColor(shown, cv2.COLOR_GRAY2BGR)
    h, w = shown.shape[:2]
    size, title_h, line_h, box_h = _panel_sizes(w, len(lines))
    panel = np.full((box_h, w, 3), 255, np.uint8)
    cv2.rectangle(panel, (0, 0), (w - 1, box_h - 1), colour, 4)
    cv2.putText(panel, title, (12, int(36 * size)), cv2.FONT_HERSHEY_SIMPLEX, 1.05 * size, colour,
                max(2, int(3 * size)), cv2.LINE_AA)
    for n, line in enumerate(lines):
        cv2.putText(panel, line, (12, title_h + line_h * n + int(22 * size)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7 * size, (40, 40, 40), 2, cv2.LINE_AA)
    return np.vstack([panel, shown])


def _panel_sizes(width, n_lines):
    size = max(0.55, min(1.0, width / 1280))          # smaller text on small images
    title_h, line_h = int(48 * size), int(32 * size)
    return size, title_h, line_h, title_h + line_h * n_lines + int(14 * size)


def panel_height(width, n_lines):
    """Height of the panel instruction_box adds above an image of this width."""
    return _panel_sizes(width, n_lines)[3]


def key_legend(image, items, counts=None, target=None, highlight=None):
    """A row of coloured chips below the picture: one per key, e.g. ("1", "red", (40, 40, 230)).
    counts/target add "12/100" under each name; highlight draws a thick frame round one label."""
    h, w = image.shape[:2]
    n = len(items)
    chip_w = w // n
    chip_h = 64 if counts is not None else 46
    shown = np.vstack([image, np.full((chip_h, w, 3), 255, np.uint8)])    # chips go BELOW the picture
    h = shown.shape[0]
    y0 = h - chip_h
    for i, (key, name, bgr) in enumerate(items):
        x0 = i * chip_w
        cv2.rectangle(shown, (x0 + 2, y0), (x0 + chip_w - 2, h - 2), bgr, -1)
        dark = sum(bgr) / 3 > 140                      # dark text on light chips
        ink = (30, 30, 30) if dark else (255, 255, 255)
        cv2.putText(shown, f"{key} = {name}", (x0 + 8, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, ink, 2,
                    cv2.LINE_AA)
        if counts is not None:
            done = counts.get(name, 0)
            text = f"{done}/{target}" if target else str(done)
            cv2.putText(shown, text, (x0 + 8, y0 + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ink, 1, cv2.LINE_AA)
            if target and done >= target:
                cv2.putText(shown, "done", (x0 + chip_w - 58, y0 + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ink, 1,
                            cv2.LINE_AA)
        if name == highlight:
            cv2.rectangle(shown, (x0 + 2, y0), (x0 + chip_w - 2, h - 2), (0, 0, 0), 4)
            cv2.rectangle(shown, (x0 + 6, y0 + 4), (x0 + chip_w - 6, h - 6), (255, 255, 255), 2)
    return shown
