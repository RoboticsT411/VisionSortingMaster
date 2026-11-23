"""
ArUco markers and the top-down view of the pick zone.

Four ArUco markers are stuck at the corners of the pick zone. You measure
the robot x, y of each marker centre once (step1_measure_markers.py).
In every camera frame the markers are found again, so the code always knows
how camera pixels map to robot coordinates, even if the camera is moved.

TopDownView uses that mapping (a homography) to:
  - warp the camera image into a flat top-down image of the zone,
    where 1 mm = PX_PER_MM pixels;
  - convert a pixel in that image into robot x, y in metres.
"""
import json

import cv2
import numpy as np

import config as cfg


# ---------------------------------------------------------------- markers
def get_dictionary():
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.ARUCO_DICT))


def marker_image(marker_id, size_px):
    """Black and white image of one marker, for printing."""
    dictionary = get_dictionary()
    if hasattr(cv2.aruco, "generateImageMarker"):        # OpenCV 4.7 and newer
        return cv2.aruco.generateImageMarker(dictionary, marker_id, size_px)
    return cv2.aruco.drawMarker(dictionary, marker_id, size_px)


DETECTION_DEFAULTS = {"gamma": 1.0, "clahe": False, "threshold_constant": 7,
                      "window_max": 23, "min_perimeter": 0.03}


def load_detection_settings():
    """Image adjustments for marker detection saved by step1_tune_camera.py."""
    settings = dict(DETECTION_DEFAULTS)
    if cfg.CAMERA_SETTINGS_FILE.exists():
        settings.update(json.loads(cfg.CAMERA_SETTINGS_FILE.read_text()).get("detection", {}))
    return settings


class MarkerDetector:
    """Finds the ArUco markers. Only the image used for finding markers is
    adjusted (gamma, contrast); the colours the model sees are not changed."""

    def __init__(self, settings=None):
        self.configure(settings or load_detection_settings())

    def configure(self, settings):
        self.settings = dict(DETECTION_DEFAULTS, **settings)
        dictionary = get_dictionary()
        if hasattr(cv2.aruco, "ArucoDetector"):             # OpenCV 4.7 and newer
            params = cv2.aruco.DetectorParameters()
            params.adaptiveThreshConstant = float(self.settings["threshold_constant"])
            params.adaptiveThreshWinSizeMax = int(self.settings["window_max"])
            params.minMarkerPerimeterRate = float(self.settings["min_perimeter"])
            detector = cv2.aruco.ArucoDetector(dictionary, params)
            self._detect = detector.detectMarkers
        else:                                                # older OpenCV
            params = cv2.aruco.DetectorParameters_create()
            params.adaptiveThreshConstant = float(self.settings["threshold_constant"])
            params.adaptiveThreshWinSizeMax = int(self.settings["window_max"])
            params.minMarkerPerimeterRate = float(self.settings["min_perimeter"])
            self._detect = lambda img: cv2.aruco.detectMarkers(img, dictionary, parameters=params)
        gamma = float(self.settings["gamma"])
        self._lut = None
        if abs(gamma - 1.0) > 1e-3:
            x = np.arange(256) / 255.0
            self._lut = np.clip(255.0 * x ** (1.0 / gamma), 0, 255).astype(np.uint8)
        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)) if self.settings["clahe"] else None

    def prepare(self, frame):
        """The grey image the detector actually looks at."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._lut is not None:
            gray = cv2.LUT(gray, self._lut)
        if self._clahe is not None:
            gray = self._clahe.apply(gray)
        return gray

    def detect(self, frame):
        """Return {marker id: (u, v) pixel centre} and the raw corners for drawing."""
        corners, ids, _ = self._detect(self.prepare(frame))
        centres = {}
        if ids is not None:
            for c, marker_id in zip(corners, ids.flatten()):
                centres[int(marker_id)] = c.reshape(4, 2).mean(axis=0)
        return centres, corners, ids


def load_marker_positions():
    """Robot x, y (metres) of each marker centre, from calibration/markers.json."""
    if not cfg.MARKERS_FILE.exists():
        raise FileNotFoundError(f"{cfg.MARKERS_FILE} not found. Run step1_measure_markers.py first.")
    data = json.loads(cfg.MARKERS_FILE.read_text())
    positions = {int(k): np.array(v[:2], dtype=float) for k, v in data["markers"].items()}
    missing = [i for i in cfg.MARKER_IDS if i not in positions]
    if missing:
        raise ValueError(f"markers.json has no position for marker(s) {missing}")
    return {i: positions[i] for i in cfg.MARKER_IDS}


# ---------------------------------------------------------------- top-down view
class TopDownView:
    def __init__(self, marker_xy_m):
        self.marker_xy_m = marker_xy_m
        s = cfg.PX_PER_MM
        pts_mm = np.array(list(marker_xy_m.values())) * 1000.0
        self.x_min, self.y_min = pts_mm.min(axis=0)
        self.x_max, self.y_max = pts_mm.max(axis=0)
        self.width = int(round((self.y_max - self.y_min) * s)) + 1
        self.height = int(round((self.x_max - self.x_min) * s)) + 1

        # Robot (x, y) in mm  ->  top-down pixel (u, v).
        # u grows towards the robot's -y side, v grows towards the robot base.
        self.A = np.array([[0.0, -s, self.y_max * s],
                           [-s, 0.0, self.x_max * s],
                           [0.0, 0.0, 1.0]])

        self.H = None        # camera pixel -> top-down pixel
        self.fresh = False   # True only if all 4 markers were seen in this frame
        self.zone_mask = self._make_zone_mask()

    # ------------------------------------------------------------ calibration
    def update(self, marker_centres):
        """Recompute the mapping from the markers seen in this frame."""
        ids = [i for i in cfg.MARKER_IDS if i in marker_centres]
        if len(ids) < 4:
            self.fresh = False
            return False
        from mapping import check_homography, compute_homography
        check_homography()                     # your functions must pass before they are used
        src = [tuple(map(float, marker_centres[i])) for i in ids]
        dst_mm = [tuple(map(float, self.marker_xy_m[i] * 1000.0)) for i in ids]
        H_cam_to_mm = compute_homography(src, dst_mm)          # KEY PART (mapping.py)
        if H_cam_to_mm is None:
            self.fresh = False
            return False
        self.H_mm = np.asarray(H_cam_to_mm, float)             # camera pixel -> robot mm
        self.H = self.A @ self.H_mm                           # camera pixel -> top-down pixel
        self.fresh = True
        return True

    def keep_out(self, points_xy_m, radius_mm=cfg.KEEP_OUT_RADIUS_MM):
        """Remove circles around these robot x, y points (squares A, B, C) from the pick zone."""
        r = int(radius_mm * cfg.PX_PER_MM)
        for x, y in points_xy_m:
            u, v = self.robot_to_topdown(x, y)
            cv2.circle(self.zone_mask, (int(round(u)), int(round(v))), r, 0, -1)

    def ready(self):
        return self.H is not None

    # ------------------------------------------------------------ images
    def warp(self, frame):
        """Top-down image of the pick zone."""
        return cv2.warpPerspective(frame, self.H, (self.width, self.height))

    def _make_zone_mask(self):
        """White where a cube may be picked: inside the markers, away from the edges."""
        pts = np.array([self.robot_to_topdown(*xy) for xy in self.marker_xy_m.values()])
        centre = pts.mean(axis=0)
        order = np.argsort(np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0]))
        mask = np.zeros((self.height, self.width), np.uint8)
        cv2.fillPoly(mask, [np.int32(pts[order])], 255)
        shrink_mm = cfg.MARKER_SIZE_MM / 2 + cfg.EDGE_MARGIN_MM
        r = max(1, int(shrink_mm * cfg.PX_PER_MM))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        # Pixels outside the image count as "not zone", so the edges shrink too.
        return cv2.erode(mask, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)

    # ------------------------------------------------------------ coordinates
    def robot_to_topdown(self, x_m, y_m):
        u, v, w = self.A @ np.array([x_m * 1000.0, y_m * 1000.0, 1.0])
        return u / w, v / w

    def topdown_to_robot(self, u, v):
        s = cfg.PX_PER_MM
        x_mm = self.x_max - v / s
        y_mm = self.y_max - u / s
        return x_mm / 1000.0, y_mm / 1000.0

    def camera_to_robot(self, px, py):
        from mapping import apply_homography
        x_mm, y_mm = apply_homography(self.H_mm, px, py)        # KEY PART (mapping.py)
        return x_mm / 1000.0, y_mm / 1000.0

    def robot_to_camera(self, x_m, y_m):
        u, v = self.robot_to_topdown(x_m, y_m)
        p = np.linalg.inv(self.H) @ np.array([u, v, 1.0])
        return p[0] / p[2], p[1] / p[2]

    def in_zone(self, u, v):
        ui, vi = int(round(u)), int(round(v))
        if 0 <= vi < self.height and 0 <= ui < self.width:
            return self.zone_mask[vi, ui] > 0
        return False
