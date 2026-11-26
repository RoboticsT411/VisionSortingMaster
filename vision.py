"""
The complete vision pipeline: camera frame in, list of classified cubes out.

    frame -> find ArUco markers -> top-down view -> find cube candidates
          -> trained model predicts colour + confidence -> robot x, y
"""
import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

import config as cfg


def parallax_corrected(x, y):
    from mapping import check_parallax, correct_parallax
    if cfg.CAMERA_POSITION_M is None:
        raise SystemExit("USE_PARALLAX_CORRECTION is True but CAMERA_POSITION_M in config.py is not measured yet")
    check_parallax()
    return correct_parallax(x, y, cfg.PARALLAX_HEIGHT_MM / 1000.0, cfg.CAMERA_POSITION_M)
from aruco_view import MarkerDetector, TopDownView, load_marker_positions
from block_finder import find_cubes, load_background
from robot import load_lab5_poses, square_centres

COLOURS_BGR = dict(cfg.CLASS_DRAW_BGR, **{"?": (255, 255, 255)})


@dataclass
class Detection:
    label: str
    confidence: float
    x: float              # robot coordinates, metres
    y: float
    angle_deg: float      # cube turn in the ROBOT frame, -45 to +45 (+ = anticlockwise from above)
    u: float              # top-down pixel
    v: float
    box: np.ndarray
    crop: np.ndarray


@dataclass
class VisionResult:
    detections: list = field(default_factory=list)
    topdown: np.ndarray = None
    mask: np.ndarray = None
    markers_seen: int = 0
    fresh: bool = False     # all 4 markers seen in THIS frame
    inference_ms: float = 0.0


class VisionSystem:
    def __init__(self, model=None, use_background=True):
        self.markers = MarkerDetector()
        self.view = TopDownView(load_marker_positions())
        self._exclude_target_squares()
        self.background = load_background() if use_background else None
        self.model = model
        self._last_marker_draw = None

    def _exclude_target_squares(self):
        """Cubes stacked on A, B and C must never be picked up again."""
        try:
            poses = load_lab5_poses()
        except (FileNotFoundError, ValueError) as e:
            print(f"Warning: squares A, B, C not excluded from the pick zone ({e})")
            return
        self.view.keep_out(list(square_centres(poses).values()))

    def process(self, frame):
        result = VisionResult()
        centres, corners, ids = self.markers.detect(frame)
        self._last_marker_draw = (corners, ids)
        result.markers_seen = sum(1 for i in cfg.MARKER_IDS if i in centres)
        result.fresh = self.view.update(centres)
        if not self.view.ready():
            return result

        result.topdown = self.view.warp(frame)
        candidates, result.mask = find_cubes(result.topdown, self.view, self.background)

        start = time.perf_counter()
        if self.model is not None:
            labels, confs = self.model.predict([c.crop for c in candidates])
        else:
            labels, confs = ["?"] * len(candidates), [0.0] * len(candidates)
        result.inference_ms = (time.perf_counter() - start) * 1000.0

        for c, label, conf in zip(candidates, labels, confs):
            x, y = self.view.topdown_to_robot(c.u, c.v)
            if cfg.USE_PARALLAX_CORRECTION:                   # step 4: KEY PART in mapping.py
                x, y = parallax_corrected(x, y)
            result.detections.append(Detection(label, conf, x, y, self.robot_angle(c),
                                               c.u, c.v, c.box, c.crop))
        return result

    def robot_angle(self, c):
        """The cube's turn in the ROBOT frame (degrees, -45 to +45, + = anticlockwise seen from above).
        The top-down picture is flipped compared with the robot (right = -y, down = -x), so an
        angle measured in the picture cannot be used directly: convert two points along an edge."""
        a = math.radians(c.angle_deg)
        x0, y0 = self.view.topdown_to_robot(c.u, c.v)
        x1, y1 = self.view.topdown_to_robot(c.u + 10 * math.cos(a), c.v + 10 * math.sin(a))
        angle = math.degrees(math.atan2(y1 - y0, x1 - x0))
        return (angle + 45.0) % 90.0 - 45.0                 # a cube looks the same every 90 degrees

    # ------------------------------------------------------------ drawing
    def draw(self, frame, result):
        """Return (camera image, top-down image) with markers, zone and detections drawn."""
        cam = frame.copy()
        corners, ids = self._last_marker_draw
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(cam, corners, ids)
        status = f"markers {result.markers_seen}/4" + ("" if result.fresh else "  (NOT all visible)")
        cv2.putText(cam, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0) if result.fresh else (0, 0, 255), 2)
        if result.topdown is None:
            return cam, None

        top = result.topdown.copy()
        # RETR_LIST: the outer edge and the holes (the circles round A, B and C inside the zone)
        contours, _ = cv2.findContours(self.view.zone_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(top, contours, -1, (255, 255, 255), 1)
        # The same pick-zone outline on the camera picture: put cubes inside it.
        for c in contours:
            pts = [self.view.robot_to_camera(*self.view.topdown_to_robot(float(u), float(v))) for u, v in c[:, 0]]
            cv2.polylines(cam, [np.int32(pts)], True, (255, 255, 255), 2, cv2.LINE_AA)
        for d in result.detections:
            colour = COLOURS_BGR.get(d.label, (255, 255, 255))
            cv2.drawContours(top, [np.int32(d.box)], 0, colour, 2)
            # No model yet (step 2): every found object is just "object", ready to be labelled.
            text = "object" if self.model is None else f"{d.label} {d.confidence:.2f}"
            where = f"x {d.x * 1000:.0f}  y {d.y * 1000:.0f} mm"
            label(top, [text], (int(d.u) - 30, int(d.v) - 24), colour, 0.45)
            box = [self.view.robot_to_camera(*self.view.topdown_to_robot(float(u), float(v))) for u, v in d.box]
            cv2.polylines(cam, [np.int32(box)], True, colour, 3, cv2.LINE_AA)
            px, py = np.min(np.array(box), axis=0)
            label(cam, [text, where], (px, py - 6), colour, 0.45)
        return cam, top


def label(image, lines, org, colour, size=0.6):
    """Black text on a white tag, with a border in the predicted colour, so it is readable on any
    background. org = bottom-left of the tag; the tag is moved down if it would leave the image."""
    font, thick, pad = cv2.FONT_HERSHEY_SIMPLEX, 1, 4
    sizes = [cv2.getTextSize(t, font, size, thick)[0] for t in lines]
    w = max(sz[0] for sz in sizes) + 2 * pad
    line_h = max(sz[1] for sz in sizes) + 6
    h = line_h * len(lines) + pad
    x, y = int(org[0]), int(org[1])
    x = max(0, min(x, image.shape[1] - w - 1))
    y = max(h + 1, y)
    cv2.rectangle(image, (x, y - h), (x + w, y), (255, 255, 255), -1)
    cv2.rectangle(image, (x, y - h), (x + w, y), colour, 2)
    for i, t in enumerate(lines):
        cv2.putText(image, t, (x + pad, y - h + line_h * (i + 1) - 2), font, size, (0, 0, 0), thick + 1,
                    cv2.LINE_AA)


def show(name, image, max_width=960):
    """imshow that shrinks big images to fit the screen."""
    if image is None:
        return
    h, w = image.shape[:2]
    if w > max_width:
        image = cv2.resize(image, (max_width, int(h * max_width / w)))
    cv2.imshow(name, image)
