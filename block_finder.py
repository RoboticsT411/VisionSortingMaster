"""
Finds cube-sized objects in the top-down image of the pick zone.

This part is classical image processing, not machine learning. It answers
"where is something that could be a cube?" The trained model (step 3)
answers "what colour is it?", including "other" for things that are not
one of your cubes.

Two ways to separate objects from the platform:
  1. Background difference (default): compare with a picture of the EMPTY
     zone (calibration/background.png). Works for any cube colour.
  2. Colour rule (no background picture): keep strongly coloured pixels
     that are not the blue of the platform.
"""
from dataclasses import dataclass

import cv2
import numpy as np

import config as cfg


@dataclass
class Candidate:
    u: float              # centre, top-down pixels
    v: float
    angle_deg: float      # cube rotation in the image, -45 to +45 degrees
    size_mm: float        # estimated side length
    box: np.ndarray       # 4 corners, for drawing
    crop: np.ndarray      # CROP_SIZE x CROP_SIZE colour image of the cube


def load_background():
    if cfg.BACKGROUND_FILE.exists():
        return cv2.imread(str(cfg.BACKGROUND_FILE))
    return None


def foreground_mask(topdown, background=None):
    """White where something is on the platform."""
    blurred = cv2.GaussianBlur(topdown, (5, 5), 0)
    if background is not None and background.shape == topdown.shape:
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB).astype(np.int16)
        lab_bg = cv2.cvtColor(cv2.GaussianBlur(background, (5, 5), 0),
                              cv2.COLOR_BGR2LAB).astype(np.int16)
        diff = np.abs(lab - lab_bg)
        # Brightness (L) counts half, so shadows and light changes matter less than colour.
        distance = 0.5 * diff[..., 0] + diff[..., 1] + diff[..., 2]
        mask = distance > cfg.FOREGROUND_THRESHOLD
    else:
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        platform_blue = (h > 90) & (h < 130)
        mask = (s > 90) & (v > 60) & ~platform_blue
    mask = mask.astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return mask


def crop_around(image, u, v, half):
    """Square crop centred on (u, v), padded if it runs off the image."""
    padded = cv2.copyMakeBorder(image, half, half, half, half, cv2.BORDER_REPLICATE)
    ui, vi = int(round(u)) + half, int(round(v)) + half
    crop = padded[vi - half:vi + half, ui - half:ui + half]
    return cv2.resize(crop, (cfg.CROP_SIZE, cfg.CROP_SIZE), interpolation=cv2.INTER_AREA)


def edge_angle(topdown, contour):
    """The cube's rotation (-45 to +45 degrees) from the direction of its straight edges.

    The outline of a cube seen from the side is not a square: the camera also sees one or two
    side faces, so a box fitted round the outline can be turned the wrong amount. The top
    and bottom edges of a cube all run at the cube's angle or at 90 degrees to it, so each
    edge pixel votes for its direction "modulo 90 degrees" (angle x 4), weighted by how
    strong the edge is. Returns (angle, agreement 0..1), or (None, 0) if there are no edges."""
    x, y, w, h = cv2.boundingRect(contour)
    pad = 6
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + w + pad, topdown.shape[1]), min(y + h + pad, topdown.shape[0])
    gray = cv2.cvtColor(topdown[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    region = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(region, [contour - [x0, y0]], -1, 255, -1)
    region = cv2.dilate(region, np.ones((5, 5), np.uint8))      # include the outline itself
    mag = np.hypot(gx, gy) * (region > 0)
    theta = np.arctan2(gy, gx)
    c, s = float(np.sum(mag * np.cos(4 * theta))), float(np.sum(mag * np.sin(4 * theta)))
    total = float(np.sum(mag))
    if total < 1e-6:
        return None, 0.0
    angle = np.degrees(np.arctan2(s, c) / 4.0)                   # -45 to +45
    return float(angle), float(np.hypot(c, s) / total)


def find_cubes(topdown, view, background=None):
    """Return (list of Candidate, foreground mask)."""
    mask = foreground_mask(topdown, background)
    side_px = cfg.CUBE_SIZE_MM * cfg.PX_PER_MM
    expected_area = side_px ** 2
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        # Too small = noise. Too big = two touching cubes or a hand: skip it.
        if area < 0.4 * expected_area or area > 3.0 * expected_area:
            continue
        (u, v), (w, h), angle = cv2.minAreaRect(contour)
        if min(w, h) == 0 or max(w, h) / min(w, h) > 1.6:
            continue                       # not square enough to be a cube
        if not view.in_zone(u, v):
            continue                       # outside the safe pick zone
        angle = angle % 90.0               # a cube looks the same every 90 degrees
        if angle >= 45.0:
            angle -= 90.0
        edges, agreement = edge_angle(topdown, contour)
        if edges is not None and agreement > 0.15:
            angle = edges                  # straight edges are more reliable than the outline
        # Drawn box: a square the size of a cube, turned by the measured angle.
        box = cv2.boxPoints(((u, v), (side_px, side_px), angle))
        crop = crop_around(topdown, u, v, int(0.75 * side_px))
        size_mm = float(np.sqrt(area)) / cfg.PX_PER_MM
        candidates.append(Candidate(u, v, angle, size_mm, box, crop))
    return candidates, mask
