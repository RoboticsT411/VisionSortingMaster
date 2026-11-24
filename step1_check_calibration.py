"""
STEP 1.5 - Calibrate the camera to the robot from squares A, B and C.

    python step1_check_calibration.py
    python step1_check_calibration.py --source dataset_raw     (saved images)

The box at the top of the picture always says what to do next.
FIRST CLICK ON THE PICTURE so it receives your key presses.

HOW IT WORKS
    The four markers are fixed to the platform, in the same layout on every station
    (MARKER_LAYOUT_MM in config.py), so the camera can measure anything on the platform in
    millimetres. What it does not know is where the platform is in ROBOT coordinates. Your Lab 5
    positions (step 1.1) say exactly that: the robot x, y of the centres of squares A, B and C.

1. Calibrate. Click the centre of A, then B, then C. The program finds the position and turn of
   the marker layout that puts your three clicks on your Lab 5 positions, and saves the robot
   x, y of each marker to calibration/markers.json. If the squares do not fit, it says why and
   what to fix (usually one Lab 5 position, in step 1.1) and saves nothing.
2. Confirm. The red circles now show your Lab 5 positions for A, B and C: each should sit in the
   centre of its square. Click A, B and C again: all within 5 mm and the step is done.
3. Ruler checks (optional). Lines join the marker centres, labelled with their length in mm.
   Measure the same distance with a ruler, click the line, type your value and press Enter:
   green within 5 mm. A wrong length means MARKER_LAYOUT_MM does not match this platform.
   Or click two points away from the lines and compare their distance with a ruler.

All positions are robot coordinates (origin at the robot base, x forward, y to the robot's left).

Keys:  0-9 and .  type a ruler value     Enter = compare     Backspace = delete / redo last square
       Esc = cancel     R = start the calibration again     C = clear ruler checks     Q = quit
Every comparison is saved to results/calibration_checks.csv.
"""
import argparse
import csv
import json
import math
import shutil
from datetime import datetime
from itertools import combinations

import cv2
import numpy as np

import config as cfg
from aruco_view import MarkerDetector, TopDownView
from camera import Camera
from robot import load_lab5_poses, square_centres
from progress import project_status, with_progress
from ui import instruction_box

WINDOW = "calibration"
TOLERANCE_MM = 5.0          # confirmation: each square within this of its Lab 5 position
FIT_MM = 2.5                # calibration: after the best placement, each square within this
MAX_TURN_DEG = 5.0          # the platform is square to the robot: a good calibration turns it less than this
SWAPPED_TURN_DEG = 20.0     # a turn this big means markers that do not match MARKER_LAYOUT_MM (e.g. swapped)
LINE_PICK_PX = 12           # a click this close to a marker line selects it
GREEN, RED, BLUE, ORANGE, GREY = (0, 170, 0), (0, 0, 230), (200, 90, 0), (0, 140, 255), (60, 60, 60)
LIGHT_GREY = (150, 150, 150)
HOLE_MM = 25.0              # squares A, B, C are 25 x 25 mm holes: a click is moved to the centre of the hole
SNAP_MM = 12.0              # ...if the hole's edges are found within this distance of the click (inside the hole)

clicks = []                 # clicks waiting to be handled, in camera-image pixels
display_scale = [1.0]
panel_offset = [0]          # height of the instruction panel above the picture
picture_width = [10 ** 6]   # clicks right of the picture (on the progress panel) are ignored


def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        y_image = y / display_scale[0] - panel_offset[0]
        if y_image >= 0 and x / display_scale[0] < picture_width[0]:   # not on the panels
            clicks.append((x / display_scale[0], y_image))


def put(img, text, org, colour, size=0.65, thick=2):
    org = (int(org[0]), int(org[1]))
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), thick + 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, size, colour, thick, cv2.LINE_AA)


def distance_to_segment(p, a, b):
    p, a, b = np.asarray(p, float), np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1)
    return float(np.linalg.norm(p - (a + t * ab)))


def segments_cross(a, b, c, d):
    """True if segment a-b crosses segment c-d (used to find the two diagonals)."""
    def side(p, q, r):
        return np.sign((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))
    return side(a, b, c) != side(a, b, d) and side(c, d, a) != side(c, d, b)


def save_row(what, measured, ruler):
    diff = measured - ruler
    ok = abs(diff) <= TOLERANCE_MM
    cfg.RESULTS_DIR.mkdir(exist_ok=True)
    path = cfg.RESULTS_DIR / "calibration_checks.csv"
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "what", "measured_mm", "ruler_mm", "difference_mm", "verdict"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), what, round(measured, 1), ruler,
                    round(diff, 1), "OK" if ok else "NOT OK"])
    return diff, ok


# ======================================================================================
# Click snapping: find the edges of the 25 mm hole near a click, and use its centre
# ======================================================================================
def snap_to_hole(frame, view, xy_m, px_per_mm=4.0):
    """xy_m: the click on the marker layout (metres). Returns the centre of the square hole found
    around it (metres) and a score, or (None, score) if no clear 25 mm square outline is there."""
    s = px_per_mm
    half = HOLE_MM / 2 + SNAP_MM + 4                                   # mm around the click
    x0, y0 = xy_m[0] * 1000 - half, xy_m[1] * 1000 - half
    size = int(2 * half * s)
    # patch pixel (u, v) <- layout mm: u along y, v along x, so the hole edges are horizontal/vertical
    P = np.array([[0.0, s, -y0 * s], [s, 0.0, -x0 * s], [0.0, 0.0, 1.0]])
    patch = cv2.warpPerspective(frame, P @ view.H_mm, (size, size))
    grey = cv2.GaussianBlur(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    edges = sum(np.abs(cv2.Sobel(c, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(c, cv2.CV_32F, 0, 1))
                for c in (grey, lab[:, :, 1], lab[:, :, 2]))           # brightness and colour edges
    side, ring = int(round(HOLE_MM * s)), max(2, int(round(1.5 * s)))
    tpl = np.zeros((side + 2 * ring, side + 2 * ring), np.float32)
    cv2.rectangle(tpl, (ring, ring), (ring + side, ring + side), 1.0, ring)
    tpl -= tpl.mean()
    score = cv2.matchTemplate(edges, tpl, cv2.TM_CCOEFF_NORMED)
    # Only hole centres within SNAP_MM of the click count (not a label or a cube further away)
    vv, uu = np.mgrid[0:score.shape[0], 0:score.shape[1]]
    near = np.hypot(uu + tpl.shape[1] / 2 - half * s, vv + tpl.shape[0] / 2 - half * s) <= SNAP_MM * s
    score = np.where(near, score, -1.0).astype(np.float32)
    _, best, _, (u, v) = cv2.minMaxLoc(score)
    cu, cv_ = u + tpl.shape[1] / 2, v + tpl.shape[0] / 2
    centre = np.array([(x0 + cv_ / s) / 1000, (y0 + cu / s) / 1000])
    moved = float(np.hypot(*(centre - np.asarray(xy_m)))) * 1000
    if best < 0.2 or moved > SNAP_MM:
        return None, best
    return centre, best


# ======================================================================================
# Calibration: place the fixed marker layout on your Lab 5 positions
# ======================================================================================
def layout_m():
    """The fixed marker layout from config.py, in metres (x towards the robot's front, y to its left)."""
    return {i: np.array(cfg.MARKER_LAYOUT_MM[i], float) / 1000.0 for i in cfg.MARKER_IDS}


def turn_deg(R):
    return math.degrees(math.atan2(R[1, 0], R[0, 0]))


def rigid_fit(src, dst):
    """The turn R and shift t (no stretching) that best move the points src onto dst.
    Returns R (2 x 2), t (metres) and the distance left at each point, in mm."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    sc, dc = src.mean(axis=0), dst.mean(axis=0)
    u, _, vt = np.linalg.svd((src - sc).T @ (dst - dc))
    d = np.sign(np.linalg.det(vt.T @ u.T))              # a turn only, never a mirror image
    R = vt.T @ np.diag([1.0, d]) @ u.T
    t = dc - R @ sc
    left = np.linalg.norm(src @ R.T + t - dst, axis=1) * 1000
    return R, t, left


def place_layout(lab5, clicked):
    """lab5: {square: robot (x, y)} from step 1.1.  clicked: {square: (x, y) on the marker layout}.
    Returns a dict with "case": good / one_square / inconsistent / turned, and the numbers."""
    names = sorted(n for n in lab5 if n in clicked)
    R, t, left = rigid_fit([clicked[n] for n in names], [lab5[n] for n in names])
    out = {"names": names, "R": R, "t": t, "left": dict(zip(names, left)), "turn": turn_deg(R)}
    if abs(out["turn"]) > SWAPPED_TURN_DEG:
        out["case"] = "turned"
    elif max(left) <= FIT_MM and abs(out["turn"]) <= MAX_TURN_DEG:
        out["case"] = "good"
    else:
        # One square on its own? Place the layout with the other two and see if only that one misses.
        best = None
        for odd in names:
            rest = [n for n in names if n != odd]
            if len(rest) < 2:
                continue
            R2, t2, left2 = rigid_fit([clicked[n] for n in rest], [lab5[n] for n in rest])
            miss = float(np.linalg.norm(R2 @ np.asarray(clicked[odd]) + t2 - np.asarray(lab5[odd]))) * 1000
            # Two squares can always be matched by turning far enough, so only a square-on turn counts
            if (max(left2) <= FIT_MM and abs(turn_deg(R2)) <= MAX_TURN_DEG and miss > TOLERANCE_MM
                    and (best is None or miss > best[1])):
                best = (odd, miss)
        if best:
            out.update(case="one_square", odd=best[0], odd_mm=best[1])
        else:
            out["case"] = "inconsistent"
    return out


def save_markers(placement, clicked):
    """Write the robot x, y of every marker (the layout moved and turned by the placement)."""
    R, t = placement["R"], placement["t"]
    cfg.CALIB_DIR.mkdir(exist_ok=True)
    if cfg.MARKERS_FILE.exists():
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        shutil.copy(cfg.MARKERS_FILE, cfg.CALIB_DIR / f"markers_backup_{stamp}.json")
    data = {"saved": datetime.now().isoformat(timespec="seconds"), "units": "metres",
            "method": "marker layout (config.py) placed on the Lab 5 centres of A, B, C (step1_check_calibration.py)",
            "markers": {str(i): [round(float(v), 4) for v in R @ xy + t] for i, xy in layout_m().items()},
            "placement": {"turn_deg": round(placement["turn"], 2), "shift_mm": [round(float(v) * 1000, 1) for v in t]},
            "fit_mm": {n: round(float(v), 1) for n, v in placement["left"].items()},
            "squares_clicked_on_layout_mm": {n: [round(float(v) * 1000, 1) for v in xy] for n, xy in clicked.items()},
            "check": None}
    cfg.MARKERS_FILE.write_text(json.dumps(data, indent=2))
    project_status(max_age=0)
    return {i: np.array(v, float) for i, v in ((int(k), v) for k, v in data["markers"].items())}


def save_check(off_mm):
    """Record the confirmation in markers.json (progress.py reads it)."""
    data = json.loads(cfg.MARKERS_FILE.read_text())
    data["check"] = {"time": datetime.now().isoformat(timespec="seconds"),
                     "off_mm": {n: round(v, 1) for n, v in off_mm.items()},
                     "passed": max(off_mm.values()) <= TOLERANCE_MM}
    cfg.MARKERS_FILE.write_text(json.dumps(data, indent=2))
    project_status(max_age=0)


def load_saved_markers():
    """Marker robot x, y from an earlier calibration, or None."""
    if not cfg.MARKERS_FILE.exists():
        return None
    try:
        data = json.loads(cfg.MARKERS_FILE.read_text())
        return {i: np.array(data["markers"][str(i)][:2], float) for i in cfg.MARKER_IDS}
    except (ValueError, KeyError, TypeError):
        return None


# ======================================================================================
# What the instruction box says
# ======================================================================================
def placement_box(p):
    """Title, colour and lines when the calibration clicks could not be used."""
    fits = ", ".join(f"{n} {p['left'][n]:.1f} mm" for n in p["names"])
    keys = "R = click A, B, C again    Q = quit"
    if p["case"] == "one_square":
        n = p["odd"]
        return (f"Not calibrated: square {n} does not fit with the others ({p['odd_mm']:.1f} mm)", RED,
                [f"The other squares fit the marker layout; {n} does not. So {n}'s Lab 5 position (step 1.1) is not",
                 f"in the centre of {n}: the gripper was off-centre when it was recorded, or a number was mistyped.",
                 f"(Or your click on {n} was off: press R and click again.)",
                 f"FIX: python step1_lab5_positions.py -> select PICK {n} level 1 -> Test. If the gripper is off-centre,",
                 f"move it by hand over the centre of {n}, press Read robot position, Save. Then run this step again.",
                 keys])
    if p["case"] == "turned":
        return (f"Not calibrated: the layout would have to turn {p['turn']:.0f} deg to fit", RED,
                ["The platform is square to the robot, so this means the markers do not match MARKER_LAYOUT_MM in",
                 "config.py (for example two markers swapped, or a marker in the wrong corner). Tell the technician.",
                 keys])
    return ("Not calibrated: A, B and C do not fit the marker layout", RED,
            [f"Distance left after the best fit: {fits} (needs {FIT_MM:g} mm or less).",
             "1. R = click A, B, C again, carefully on the centre of each hole.",
             "2. Test PICK A, B and C level 1 in step1_lab5_positions.py: is the gripper centred over each square?",
             "3. Ruler checks: click each line between markers and compare with a ruler. A wrong length means",
             "   MARKER_LAYOUT_MM in config.py does not match this platform: tell the technician.",
             keys])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None, help="camera index, image, folder or video")
    args = parser.parse_args()
    print(__doc__)

    cam = Camera(args.source)
    detector = MarkerDetector()
    layout = layout_m()
    layout_view = TopDownView(layout)           # camera pixel -> position on the marker layout
    try:
        squares = {s: np.array(xy) for s, xy in square_centres(load_lab5_poses()).items()}
    except (FileNotFoundError, ValueError) as e:
        print(f"Lab 5 positions needed first: {e}")
        squares = {}
    markers = load_saved_markers()             # robot x, y of the markers, once calibrated
    robot_view = TopDownView(markers) if markers else None

    pairs = list(combinations(cfg.MARKER_IDS, 2))
    pair_mm = {p: float(np.hypot(*(layout[p[0]] - layout[p[1]])) * 1000) for p in pairs}
    pair_result = {}                 # pair -> (ruler, diff, ok)

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_click)
    point_px = []                    # two-point measurement: image pixels
    free_result = None               # (ruler, diff, ok)
    selected, typed = None, ""       # a pair, "free", or None; the ruler value being typed
    message, message_colour = None, BLUE
    stage = "calibrate"              # calibrate -> confirm -> done; or "failed" (placement not saved)
    square_queue = list(squares)     # the squares still to click in this round
    square_px = {}                   # name -> pixel used in this round: the hole centre, or the click
    clicked_px = {}                  # name -> where you actually clicked, when it was moved to the hole
    missed = {}                      # name -> 1 after a click where the hole was not found
    placement = None                 # result of the calibration round
    off_mm = {}                      # confirmation: distance of each click from its Lab 5 position
    last_shown = None

    while True:
        frame = cam.read()
        centres, corners, ids = detector.detect(frame)
        layout_view.update(centres)
        if robot_view is not None:
            robot_view.update(centres)
        shown = frame.copy()
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(shown, corners, ids)
        seen = sum(1 for i in cfg.MARKER_IDS if i in centres)
        ready = layout_view.ready()

        marker_px = {}               # pixel centre of each marker: detected now, or from the last good mapping
        if ready:
            for i in cfg.MARKER_IDS:
                marker_px[i] = np.asarray(centres[i] if i in centres else layout_view.robot_to_camera(*layout[i]),
                                          float)

        # ---------------------------------------------------------- clicks
        while clicks:
            pt = clicks.pop(0)
            if not ready:
                print(f"Click ignored: only {seen}/4 markers found.")
                continue
            if square_queue and stage in ("calibrate", "confirm"):
                name = square_queue[0]
                hole, score = snap_to_hole(frame, layout_view, layout_view.camera_to_robot(*pt))
                if hole is None and missed.get(name, 0) == 0:    # ask once more before using a bare click
                    missed[name] = 1
                    message, message_colour = (f"Hole {name} not found around your click: click again, nearer the "
                                               f"centre of {name}.", RED)
                    print(f"square {name}: hole edges not found (score {score:.2f}), asked to click again")
                    continue
                square_queue.pop(0)
                missed.pop(name, None)
                message = None
                if hole is not None:                         # use the centre of the hole, not the click
                    square_px[name] = tuple(layout_view.robot_to_camera(*hole))
                    clicked_px[name] = pt
                    print(f"square {name}: hole found (score {score:.2f}), click moved to its centre")
                else:
                    square_px[name] = pt
                    clicked_px.pop(name, None)
                    print(f"square {name}: hole edges not found (score {score:.2f}), your click is used")
                if square_queue:
                    continue
                if stage == "calibrate":                     # all three clicked: place the layout
                    clicked = {n: np.array(layout_view.camera_to_robot(*square_px[n])) for n in squares}
                    placement = place_layout(squares, clicked)
                    print(f"Calibration: {placement['case']}, fit "
                          + ", ".join(f"{n} {v:.1f} mm" for n, v in placement["left"].items())
                          + f", turn {placement['turn']:+.2f} deg")
                    if placement["case"] == "good":
                        markers = save_markers(placement, clicked)
                        robot_view = TopDownView(markers)
                        robot_view.update(centres)
                        print(f"Saved the marker positions to {cfg.MARKERS_FILE}")
                        stage, square_queue, square_px = "confirm", list(squares), {}
                        clicked_px.clear()
                    else:
                        stage = "failed"
                else:                                        # confirmation round
                    off_mm = {n: float(np.hypot(*(squares[n] - robot_view.camera_to_robot(*square_px[n])))) * 1000
                              for n in squares}
                    for n, d in off_mm.items():
                        save_row(f"square {n} centre vs Lab 5 position", d, 0)
                    save_check(off_mm)
                    stage = "done"
                    print("Confirmation: " + ", ".join(f"{n} {d:.1f} mm" for n, d in off_mm.items()))
                continue
            dist, pair = min((distance_to_segment(pt, marker_px[a], marker_px[b]), (a, b)) for a, b in pairs)
            if dist < LINE_PICK_PX / display_scale[0]:
                selected, typed = pair, ""
                continue
            if len(point_px) == 2:
                point_px, free_result = [], None
            point_px.append(pt)
            if len(point_px) == 2:
                selected, typed = "free", ""
            else:
                selected = None
                message, message_colour = "First point set: click the second point.", BLUE

        # ---------------------------------------------------------- marker lines (lengths from the fixed layout)
        diagonals = [p for p in pairs if marker_px and
                     segments_cross(marker_px[p[0]], marker_px[p[1]],
                                    *[marker_px[i] for i in cfg.MARKER_IDS if i not in p])]
        for a, b in pairs:
            if not marker_px:
                break
            pa, pb = marker_px[a], marker_px[b]
            if (a, b) in pair_result:
                ruler, diff, ok = pair_result[(a, b)]
                colour = GREEN if ok else RED
                label = f"{pair_mm[(a, b)]:.1f} mm / ruler {ruler:g} ({diff:+.1f})"
            else:
                colour = ORANGE if selected == (a, b) else GREY
                label = f"{pair_mm[(a, b)]:.1f} mm"
            cv2.line(shown, tuple(int(v) for v in pa), tuple(int(v) for v in pb), colour,
                     3 if selected == (a, b) else 2, cv2.LINE_AA)
            t = 0.5 if (a, b) not in diagonals else (0.3 if diagonals.index((a, b)) == 0 else 0.7)
            mid = pa + t * (pb - pa)
            put(shown, label, (mid[0] - 45, mid[1] - 8), colour, 0.6)

        # ---------------------------------------------------------- two-point measurement (distances need no calibration)
        free_mm = None
        for p in point_px:
            cv2.circle(shown, (int(p[0]), int(p[1])), 7, BLUE, -1)
        if len(point_px) == 2 and ready:
            p0, p1 = (np.array(layout_view.camera_to_robot(*p)) for p in point_px)
            free_mm = float(np.hypot(*(p0 - p1))) * 1000
            colour = BLUE if free_result is None else (GREEN if free_result[2] else RED)
            cv2.line(shown, tuple(int(v) for v in point_px[0]), tuple(int(v) for v in point_px[1]), colour, 3,
                     cv2.LINE_AA)
            label = f"{free_mm:.1f} mm" if free_result is None else \
                f"{free_mm:.1f} mm / ruler {free_result[0]:g} ({free_result[1]:+.1f})"
            mid = (np.asarray(point_px[0]) + np.asarray(point_px[1])) / 2
            put(shown, label, (mid[0] + 10, mid[1] - 12), colour, 0.7)

        # ---------------------------------------------------------- A, B, C
        if ready:
            if robot_view is not None and robot_view.ready():
                # Lab 5 positions on the picture. During a new calibration round they are the OLD ones (grey).
                old = stage in ("calibrate", "failed")
                for name, xy in squares.items():
                    px, py = robot_view.robot_to_camera(*xy)
                    colour = LIGHT_GREY if old else RED
                    cv2.circle(shown, (int(px), int(py)), 9, colour, 2)
                    put(shown, name + (" (before)" if old else ""), (px + 10, py - 10), colour, 0.6)
            for name, (cx, cy) in square_px.items():
                if name in clicked_px:                       # your click, and the hole centre it moved to
                    ux, uy = clicked_px[name]
                    cv2.circle(shown, (int(ux), int(uy)), 3, BLUE, -1)
                    cv2.line(shown, (int(ux), int(uy)), (int(cx), int(cy)), BLUE, 1, cv2.LINE_AA)
                    xy = layout_view.camera_to_robot(cx, cy)
                    corners_px = [layout_view.robot_to_camera(xy[0] + dx * HOLE_MM / 2000, xy[1] + dy * HOLE_MM / 2000)
                                  for dx, dy in ((-1, -1), (-1, 1), (1, 1), (1, -1))]
                    cv2.polylines(shown, [np.int32(corners_px)], True, GREEN, 1, cv2.LINE_AA)
                cv2.drawMarker(shown, (int(cx), int(cy)), BLUE, cv2.MARKER_CROSS, 18, 2)
            if stage == "done" and robot_view is not None:
                for name, d in off_mm.items():
                    xy = robot_view.robot_to_camera(*squares[name])
                    ok = d <= TOLERANCE_MM
                    put(shown, f"{d:.1f} mm", (xy[0] + 12, xy[1] + 24), GREEN if ok else RED, 0.6)

        # ---------------------------------------------------------- instruction box
        click_line = "Click the picture first.   Backspace = redo the previous square   R = start again   Q = quit"
        if not squares:
            title, colour = "Lab 5 positions needed first", RED
            lines = ["Run python step1_lab5_positions.py: enter, test and save your Lab 5 positions.", "Q = quit"]
        elif not ready:
            title, colour = f"Markers not found ({seen}/4) - clicks are ignored", RED
            lines = ["Check the focus and that nothing hides a marker, or run step1_tune_camera.py again.",
                     "Q = quit"]
        elif selected is not None:
            what = "the two points" if selected == "free" else f"marker {selected[0]} to {selected[1]}"
            title, colour = f"Type your ruler value for {what}, then press Enter", ORANGE
            lines = [f"ruler value: {typed}_ mm",
                     message if message_colour == RED else "Backspace = delete     Esc = cancel"]
        elif stage == "calibrate":
            name = square_queue[0]
            done = len(squares) - len(square_queue)
            title, colour = f"Calibrate {done + 1}/{len(squares)}: click the centre of square {name}", BLUE
            lines = [f"Click the centre of the hole {name}. The program finds the hole's edges (green outline) and uses",
                     "its centre; if it cannot, it uses your click, so click as close to the centre as you can.",
                     "The program then places the marker layout so that A, B and C land on your Lab 5 positions.",
                     click_line]
            if markers is not None:
                lines.insert(2, "Grey circles = the calibration you have now. This round replaces it.")
            if message and message_colour == RED:
                lines.insert(0, message)
        elif stage == "confirm":
            name = square_queue[0]
            done = len(squares) - len(square_queue)
            fit = max(placement["left"].values()) if placement else 0
            title, colour = f"Calibrated - now confirm {done + 1}/{len(squares)}: click the centre of square {name}", GREEN
            lines = [f"Marker positions saved (your squares fit within {fit:.1f} mm, platform turned "
                     f"{placement['turn']:+.1f} deg).",
                     "Red circle = your Lab 5 position, drawn with the new calibration: each should sit in the centre",
                     f"of its square. Click the centre of {name} again; within {TOLERANCE_MM:g} mm for every square passes.",
                     click_line]
            if message and message_colour == RED:
                lines.insert(0, message)
        elif stage == "failed":
            title, colour, lines = placement_box(placement)
        else:                                                            # done
            offs = ", ".join(f"{n} {d:.1f} mm" for n, d in off_mm.items())
            if max(off_mm.values()) <= TOLERANCE_MM:
                title, colour = f"Camera calibrated: {offs}", GREEN
                lines = ["Saved. Next step: python step2_collect_data.py",
                         "Optional ruler checks: click a line between two markers (or two points), measure it with",
                         "a ruler, type the value, Enter.    R = calibrate again    Q = quit"]
            else:
                title, colour = f"Not confirmed: {offs}", RED
                lines = ["Your second clicks do not agree with the first ones within 5 mm.",
                         "R = start again, and click the centre of each hole carefully both times.",
                         "Q = quit"]
            if message and message_colour in (GREEN, RED):
                lines = [message] + lines
        shown = instruction_box(with_progress(shown, "1.5"), title, lines, colour)

        panel_offset[0] = shown.shape[0] - frame.shape[0]
        picture_width[0] = frame.shape[1]
        display_scale[0] = min(1.0, 1280 / shown.shape[1], 1000 / shown.shape[0])   # fits a 1080p screen
        last_shown = shown
        cv2.imshow(WINDOW, cv2.resize(shown, None, fx=display_scale[0], fy=display_scale[0]))

        # ---------------------------------------------------------- keys
        key = cv2.waitKey(30) & 0xFF
        if key == 255:
            continue
        char = chr(key)
        if selected is not None and (char.isdigit() or char == "."):
            typed += char
        elif selected is not None and key in (8, 127):
            typed = typed[:-1]
        elif selected is not None and key == 27:
            selected, typed = None, ""
            message, message_colour = None, BLUE
        elif selected is not None and key in (13, 10):
            try:
                ruler = float(typed)
            except ValueError:
                message, message_colour = "Type a number first, for example 226.5", RED
                continue
            if selected == "free":
                what, d = "two points", free_mm
                diff, ok = save_row(what, d, ruler)
                free_result = (ruler, diff, ok)
            else:
                what = f"marker {selected[0]} to {selected[1]}"
                d = pair_mm[selected]
                diff, ok = save_row(what, d, ruler)
                pair_result[selected] = (ruler, diff, ok)
            message = (f"{what}: measured {d:.1f} mm, ruler {ruler:g} mm, difference {diff:+.1f} mm "
                       f"-> {'OK' if ok else 'NOT OK'}")
            message_colour = GREEN if ok else RED
            print(message + "   (saved to results/calibration_checks.csv)")
            selected, typed = None, ""
        elif key in (8, 127) and square_queue and square_px:            # redo the previous square
            last = [n for n in squares if n in square_px][-1]
            square_px.pop(last)
            clicked_px.pop(last, None)
            square_queue.insert(0, last)
        elif char in "rR" and squares:
            stage, square_queue, square_px, off_mm, placement = "calibrate", list(squares), {}, {}, None
            clicked_px.clear()
            message = None
        elif char in "cC":
            point_px, free_result, selected, typed = [], None, None, ""
            pair_result.clear()
            message, message_colour = None, BLUE
        elif char in "qQ":
            break
    cam.release()
    cv2.destroyAllWindows()
    return last_shown


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as e:
        raise SystemExit(f"\nMissing: {e}\nCheck the function named in the message.")
