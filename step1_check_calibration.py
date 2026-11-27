"""
STEP 1.5 - Calibrate the camera to the robot from squares A, B and C.

    python step1_check_calibration.py
    python step1_check_calibration.py --source dataset_raw     (saved images, no robot)

The box at the top of the picture always says what to do next, and the checklist under it shows
what is done for each square. FIRST CLICK ON THE PICTURE so it receives your key presses.

HOW IT WORKS
    The four markers are fixed to the platform, in the same layout on every station
    (MARKER_LAYOUT_MM in config.py), so the camera can measure anything on the platform in
    millimetres. What it does not know is where the platform is in ROBOT coordinates. Your Lab 5
    positions (step 1.1) say exactly that: the robot x, y of the centres of squares A, B and C.

THE CHECKLIST: four ticks for each of A, B and C. The steps come one after another and cannot be
skipped: any other key or click only shows what to do now.
  1 Clicked        Click the centre of A, then B, then C. The program finds the edges of each hole
                   and uses its centre.
  2 Fits camera    The program places the marker layout so that A, B and C land on your Lab 5
                   positions. If they fit, it saves the marker positions (calibration/markers.json).
                   An orange dot is not a failure: it means 'not yet', and the robot checks (3) correct it.
  3 Robot centred  Press T for A, then T for B, then T for C. Each time: Y starts the move and the
                   robot goes down to the square's PICK level 1 position. Look from the front and the
                   side, from at least 30 cm (risk assessment RA001650/2). If the gripper is not centred over the hole, move it as seen standing in front
                   of the robot:
                     D / A = to your right / left     W / S = away from you / towards you
                     R / F = up / down to look closer (never saved)     1 / 5 = step 1 or 5 mm
                   Y = centred: saved for the square (the same x, y at every level, pick-up and drop;
                   heights unchanged). The circles are then refitted from your clicks: no new clicks.
                   N = leave it (the square is not ticked: press T again for it).
                   After each check the robot goes to its sleep pose in learning mode.
  4 Confirmed      Click A, B and C again: each within 5 mm of its Lab 5 position. Step 1.5 is then
                   ticked in the progress list.
  5 Markers        Optional, more accurate: press I. The robot also visits the 4 markers (T, Y, centre the
                   gripper over the middle of the marker, Y; N skips one). The markers are far apart, so
                   the fit with A, B, C and the markers is more accurate than with A, B, C alone (they are
                   only about 94 mm apart). With 3 or more markers it also includes the robot's own
                   scale: its distances can be a few % off the ruler-measured layout (up to 5 % is
                   accepted; more means MARKER_LAYOUT_MM or a marker point is wrong). It is saved only if
                   it is better; step 1.5 stays ticked.

When step 1.5 is already complete, the program opens on the finished checklist: I = step 5,
R = calibrate again from step 1, Q = quit.

L, or the ROBOT: LEARNING MODE button, any time: robot to its sleep pose with learning mode on.
Saved images (--source, no robot): S skips the robot checks; U then corrects one square from the camera.
Ruler checks (optional, when step 1.5 is complete): click a line between two markers, type your ruler
value and press Enter (green within 5 mm), or click two points and compare.

All positions are robot coordinates (origin at the robot base, x forward, y to the robot's left).

Keys:  Backspace = redo the previous square   R = start again   T = robot check of the next square
       I = step 5 (optional, when step 1.5 is complete)
       L = robot to sleep pose + learning mode   C = clear ruler checks   Q = quit
Every confirmation and ruler check is saved to results/calibration_checks.csv.
"""
import argparse
import csv
import json
import math
import queue
import shutil
import sys
import threading
from datetime import datetime
from itertools import combinations

import cv2
import numpy as np

import config as cfg
from aruco_view import MarkerDetector, TopDownView
from camera import Camera
from robot import (LEVELS, check_pose, load_lab5_poses, read_lab5_file, save_lab5_file,
                   square_centres)
from progress import project_status, with_progress
from ui import instruction_box

WINDOW = "calibration"
VERSION = "26 Nov 2026 16:10 (also runs in the online simulator)"
# The online simulator (Python in a web browser) has no real threads: there each robot command runs
# straight away and the camera picture pauses during the move. On the lab PC nothing changes.
NO_THREADS = sys.platform == "emscripten"
TOLERANCE_MM = 5.0          # confirmation: each square within this of its Lab 5 position
FIT_MM = 2.5                # calibration: after the best placement, each square within this
FIT_ROBOT_MM = 6.0          # ...or within this when all three squares were centred with the robot
MAX_TURN_DEG = 2.5          # the robot stands on the same board as the holes: a real turn is small
MAX_TURN_ROBOT_DEG = 6.0    # ...but up to this when all three squares were centred with the robot (the platform
                            # may sit a little turned to the robot, and joint 1's zero can be off by a degree or two)
SWAPPED_TURN_DEG = 20.0     # a turn this big means markers that do not match MARKER_LAYOUT_MM (e.g. swapped)
MAX_ADJUST_MM = 15.0        # centring with the robot: never further than this from the Lab 5 position
MARKER_HOVER_M = 0.02       # step 5: the gripper hovers this far above your highest level-1 pick height
MIN_MARKERS_STRETCH = 3     # step 5: with this many markers or more, the fit may also stretch (the robot's own scale)
MAX_STRETCH_PCT = 5.0       # step 5: a bigger stretch is not believed: the layout or a marker point must be wrong
HOLE_MM = 25.0              # squares A, B, C are 25 x 25 mm holes: a click is moved to the centre of the hole
SNAP_MM = 12.0              # ...if the hole's edges are found within this distance of the click (inside the hole)
LINE_PICK_PX = 12           # a click this close to a marker line selects it
GREEN, RED, BLUE, ORANGE, GREY = (0, 170, 0), (0, 0, 230), (200, 90, 0), (0, 140, 255), (60, 60, 60)
LIGHT_GREY = (150, 150, 150)

clicks = []                 # clicks waiting to be handled, in camera-image pixels
display_scale = [1.0]
panel_offset = [0]          # height of the panels above the picture
button_rect = [None]        # (x0, y0, x1, y1) of the "learning mode" button, in window pixels before scaling
button_clicks = []          # clicks on that button
picture_width = [10 ** 6]   # clicks right of the picture (on the progress panel) are ignored


def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xs, ys = x / display_scale[0], y / display_scale[0]
        b = button_rect[0]
        if b and b[0] <= xs <= b[2] and b[1] <= ys <= b[3]:
            button_clicks.append(1)                                  # the learning-mode button
            return
        y_image = ys - panel_offset[0]
        if y_image >= 0 and xs < picture_width[0]:                   # not on the panels
            clicks.append((xs, y_image))


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


def rigid_fit(src, dst, stretch=False):
    """The turn R and shift t (no stretching) that best move the points src onto dst.
    stretch=True: also the same stretch in every direction (R then holds turn x stretch).
    Returns R (2 x 2), t (metres) and the distance left at each point, in mm."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    sc, dc = src.mean(axis=0), dst.mean(axis=0)
    u, sv, vt = np.linalg.svd((src - sc).T @ (dst - dc))
    d = np.sign(np.linalg.det(vt.T @ u.T))              # a turn only, never a mirror image
    R = vt.T @ np.diag([1.0, d]) @ u.T
    if stretch:
        R = R * (sv[0] + d * sv[1]) / np.sum((src - sc) ** 2)
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
                best = (odd, miss, R2, t2)
        if best:
            odd, miss, R2, t2 = best
            # Where the camera sees the centre of the odd square, in robot coordinates
            seen = R2 @ np.asarray(clicked[odd]) + t2
            out.update(case="one_square", odd=odd, odd_mm=miss, R_rest=R2, t_rest=t2, odd_seen=seen,
                       odd_shift_mm=(seen - np.asarray(lab5[odd])) * 1000)
        else:
            out["case"] = "inconsistent"
    return out


def set_square_xy(name, xy_m):
    """Give one square the same x, y (metres) at every level, pick-up and drop: a stack is vertical, so
    only the heights differ, and the heights are not changed. Keeps a backup of the old file and clears
    that square's test results. Returns the backup name."""
    data = read_lab5_file()
    for kind in ("pick", "drop"):
        for lv in LEVELS:
            p = data[kind][name][lv]
            new = [round(float(xy_m[0]), 4), round(float(xy_m[1]), 4), p[2]]
            check_pose(new + list(data["orientation"]), {"pick": data["pick"], "drop": data["drop"]})
            data[kind][name][lv] = new
    for lv in LEVELS:
        for kind in ("pick", "drop"):
            data["tested"].pop(f"{kind} {name} {lv}", None)        # changed, so not tested any more
    backup = f"lab5_positions_backup_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    shutil.copy(cfg.LAB5_FILE, cfg.CALIB_DIR / backup)
    save_lab5_file(data)
    return backup


def record_test(name, ok):
    """Store a robot test result for PICK name level 1, as step1_lab5_positions.py does."""
    data = read_lab5_file()
    data["tested"][f"pick {name} 1"] = "ok" if ok else "wrong"
    save_lab5_file(data)


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
            "lab5_level1": {n: [round(float(v), 4) for v in xy] for n, xy in
                            square_centres(load_lab5_poses()).items()},       # progress.py: changed since?
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
# The robot, in its own thread so the picture keeps moving
# ======================================================================================
class RobotTest(threading.Thread):
    """Connect once. On request: go above a square at the safe height and down to its level-1 pick-up
    position, take small correction moves, then go back up and rest (sleep pose, learning mode on)."""

    def __init__(self):
        super().__init__(daemon=True)
        self.commands = queue.Queue()
        self.state = "connecting"      # connecting / ready / moving / arrived / failed / closed
        self.status = "Connecting and calibrating the robot (about 30 s)..."
        self.robot = None

    def run(self):
        if not self.connect():
            return
        if NO_THREADS:                 # browser simulator: send() runs each command itself
            return
        while True:
            if not self.do(self.commands.get()):
                return

    def connect(self):
        try:
            from robot import NiryoRobot
            self.robot = NiryoRobot()
            self.robot.start()
            self.state, self.status = "ready", f"Robot {self.robot.ip} ready"
            return True
        except BaseException as e:                       # NiryoRobot.start() raises SystemExit on failure
            self.state = "failed"
            self.status = "Could not connect: " + str(e).strip().splitlines()[0][:90]
            return False

    def do(self, cmd):
        """Carry out one command. Returns False after "quit"."""
        try:
            if cmd[0] == "test":
                _, name, pose, z_safe = cmd
                self.state, self.status = "moving", f"Moving to {name}..."
                self.robot.learning_mode(False)                     # motors on to move
                self.robot.move_pose([pose[0], pose[1], z_safe] + pose[3:])
                self.robot.move_pose(pose)
                self.state, self.status = "arrived", f"Gripper at {name}"
            elif cmd[0] == "jog":                                   # a small correction while centring
                self.state = "moving"
                self.robot.move_pose(cmd[1])
                self.state, self.status = "arrived", "Adjusted"
            elif cmd[0] == "up":                                    # back up, then rest
                _, pose, z_safe = cmd
                self.state, self.status = "moving", "Moving back up..."
                self.robot.move_pose([pose[0], pose[1], z_safe] + pose[3:])
                self.robot.move_joints(cfg.SLEEP_JOINTS)
                self.robot.learning_mode(True)
                self.state, self.status = "ready", "Robot resting (sleep pose, learning mode on)"
            elif cmd[0] == "rest":                                  # L key: sleep pose, learning mode
                self.state, self.status = "moving", "Moving to the sleep pose..."
                self.robot.move_joints(cfg.SLEEP_JOINTS)
                self.robot.learning_mode(True)
                self.state, self.status = "ready", "Robot resting (sleep pose, learning mode on)"
            elif cmd[0] == "quit":
                self.robot.finish()
                self.state = "closed"
                return False
        except Exception as e:
            self.state, self.status = "ready", f"Move failed: {e}"
        return True

    def send(self, *cmd):
        if NO_THREADS:                 # browser simulator: run the command now (the picture pauses meanwhile)
            if self.state not in ("connecting", "failed", "closed"):
                self.do(cmd)
            return
        if cmd[0] in ("test", "jog", "up", "rest") and self.state not in ("connecting", "failed"):
            self.state = "moving"
        self.commands.put(cmd)

    def close(self):
        if self.state in ("failed", "closed", "connecting"):
            return
        if NO_THREADS:
            self.do(("quit",))
        elif self.is_alive():
            self.send("quit")
            self.join(timeout=30)


# ======================================================================================
# The checklist under the instruction box: four ticks per square
# ======================================================================================
CHECK_COLUMNS = ["1 Clicked", "2 Fits camera", "3 Robot centred", "4 Confirmed", "5 Markers (optional)"]


def draw_mark(img, x, y, size, state):
    """state: "ok" (green tick), "bad" (red cross), "next" (orange outline), "todo" (empty), "skip",
    "later" (amber: not yet, the next step corrects it)."""
    colour = {"ok": GREEN, "bad": RED, "next": ORANGE, "todo": LIGHT_GREY, "skip": LIGHT_GREY,
              "later": ORANGE}[state]
    cv2.rectangle(img, (x, y), (x + size, y + size), colour, 3 if state == "next" else 2)
    if state == "ok":
        cv2.line(img, (x + 4, y + size // 2), (x + size // 2 - 1, y + size - 5), GREEN, 3, cv2.LINE_AA)
        cv2.line(img, (x + size // 2 - 1, y + size - 5), (x + size - 3, y + 4), GREEN, 3, cv2.LINE_AA)
    elif state == "bad":
        cv2.line(img, (x + 5, y + 5), (x + size - 5, y + size - 5), RED, 3, cv2.LINE_AA)
        cv2.line(img, (x + size - 5, y + 5), (x + 5, y + size - 5), RED, 3, cv2.LINE_AA)
    elif state == "skip":
        cv2.line(img, (x + 5, y + size // 2), (x + size - 5, y + size // 2), LIGHT_GREY, 2, cv2.LINE_AA)
    elif state == "later":
        cv2.circle(img, (x + size // 2, y + size // 2), size // 4, ORANGE, -1, cv2.LINE_AA)


def checklist_panel(width, rows):
    """rows: {square: [state for each of the four columns]}."""
    row_h, head_h, left = 34, 30, 120
    h = head_h + row_h * len(rows) + 8
    img = np.full((h, width, 3), 255, np.uint8)
    cv2.rectangle(img, (0, 0), (width - 1, h - 1), (200, 200, 200), 1)
    col_w = (width - left - 10) // len(CHECK_COLUMNS)
    cv2.putText(img, "Checklist", (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 2, cv2.LINE_AA)
    for c, name in enumerate(CHECK_COLUMNS):
        cv2.putText(img, name, (left + c * col_w, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
    for r, (square, states) in enumerate(rows.items()):
        y = head_h + r * row_h + 4
        done = all(s.partition("|")[0] in ("ok", "skip") for s in states[:4])
        cv2.putText(img, f"Square {square}", (10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    GREEN if done else (40, 40, 40), 2, cv2.LINE_AA)
        for c, state in enumerate(states):
            state, _, note = state.partition("|")                   # "state|note": a few words beside the box
            draw_mark(img, left + c * col_w + 4, y + 2, 24, state)
            note = note or {"skip": "skipped", "later": "fixed in step 3"}.get(state, "")
            if note:
                cv2.putText(img, note, (left + c * col_w + 34, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            ORANGE if state in ("later", "next") else (110, 110, 110), 1, cv2.LINE_AA)
    return img


def robot_button(width, height, status):
    """The button beside the checklist: robot to its sleep pose with learning mode on (same as key L)."""
    img = np.full((height, width, 3), 255, np.uint8)
    cv2.rectangle(img, (0, 0), (width - 1, height - 1), (200, 200, 200), 1)
    x0, y0, x1, y1 = 10, 10, width - 10, height - 34
    cv2.rectangle(img, (x0, y0), (x1, y1), (230, 240, 255), -1)
    cv2.rectangle(img, (x0, y0), (x1, y1), BLUE, 2)
    for i, text in enumerate(["ROBOT: LEARNING MODE", "sleep pose, motors", "relaxed   (key L)"]):
        size = 0.6 if i == 0 else 0.5
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, 2 if i == 0 else 1)
        cv2.putText(img, text, ((width - tw) // 2, y0 + 26 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, size, BLUE,
                    2 if i == 0 else 1, cv2.LINE_AA)
    status = status.replace("Robot resting (sleep pose, learning mode on)", "Resting: learning mode ON")
    cv2.putText(img, status[:40], (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 90), 1, cv2.LINE_AA)
    return img, (x0, y0, x1, y1)


def placement_lines(p, robot_checked):
    """Title, colour and lines when the clicks and the Lab 5 positions do not fit together."""
    keys = "- R = click A, B, C again    - Q = quit"
    if p["case"] == "one_square":
        n = p["odd"]
        dx, dy = p["odd_shift_mm"]
        others = [s for s in p["names"] if s != n and s not in robot_checked]
        if n in robot_checked and others:
            # The square that "does not fit" is the one checked with the robot: the others are suspect
            o = " and ".join(others)
            return (f"{n} was centred with the robot, but {o} not yet: check {o} with the robot", ORANGE,
                    [f"- The camera compares A, B and C with each other. {n} now differs from {o} by {p['odd_mm']:.1f} mm.",
                     f"- The robot is the reference for picking, so {o} probably need the same correction.",
                     f"- Press T, then {others[0]}, then Y, and centre the gripper over {others[0]}.", keys])
        return (f"Square {n} does not fit with the others: its Lab 5 position is {p['odd_mm']:.1f} mm off", RED,
                [f"- Green circle = where the camera sees the centre of {n}. Red circle = your Lab 5 position for {n}.",
                 f"- Fix with the robot (best): press T, then {n}, then Y, and centre the gripper over {n}.",
                 f"- Or fix from the camera: press U (moves {n} by x {dx:+.1f} mm, y {dy:+.1f} mm; heights unchanged).",
                 f"- If your click on {n} was off instead, press R and click again.", keys])
    if p["case"] == "turned":
        return (f"Not calibrated: the layout would have to turn {p['turn']:.0f} deg to fit", RED,
                ["- The markers do not match MARKER_LAYOUT_MM in config.py (for example two markers swapped).",
                 "- Or the clicks were not on A, B and C in that order. Press R and click again.",
                 "- If it repeats, tell the technician.", keys])
    fits = ", ".join(f"{n} {p['left'][n]:.1f} mm" for n in p["names"])
    return ("Not calibrated: A, B and C do not fit the marker layout", RED,
            [f"- Distance left after the best fit: {fits} (needs {FIT_MM:g} mm or less, turn {p['turn']:+.1f} deg).",
             "- Fix with the robot (best): press T, then A, then Y, centre the gripper, Y. Then B, then C.",
             "- Circles refit after each square. When all three are centred with the robot, the fit is accepted.",
             "- If your clicks were off: press R and click A, B, C again on the centre of each hole.",
             "- Ruler checks: a wrong marker-to-marker length means MARKER_LAYOUT_MM does not match: tell the technician.",
             keys])


def saved_images(source):
    """True when the pictures come from saved images or a video, not from the camera (then no robot)."""
    return source is not None and not str(source).isdigit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None, help="camera index, image, folder or video")
    args = parser.parse_args()
    print(__doc__)

    cam = Camera(args.source)
    detector = MarkerDetector()
    layout = layout_m()
    view = TopDownView(layout)                  # camera pixel <-> position on the marker layout (metres)
    try:
        squares = {s: np.array(xy) for s, xy in square_centres(load_lab5_poses()).items()}
    except (FileNotFoundError, ValueError) as e:
        print(f"Lab 5 positions needed first: {e}")
        squares = {}
    names = list(squares)

    def robot_checked():
        """Squares whose PICK level 1 position was confirmed with the robot (here or in step 1.1)."""
        try:
            tested = read_lab5_file()["tested"]
        except (TypeError, KeyError):
            return set()
        return {n for n in names if tested.get(f"pick {n} 1") == "ok"}

    pairs = list(combinations(cfg.MARKER_IDS, 2))
    pair_mm = {p: float(np.hypot(*(layout[p[0]] - layout[p[1]])) * 1000) for p in pairs}
    pair_result = {}                 # pair -> (ruler, diff, ok)

    cv2.namedWindow(WINDOW)
    cv2.setWindowTitle(WINDOW, f"Step 1.5 calibration - version {VERSION}")
    print(f"step1_check_calibration.py version {VERSION}")
    cv2.setMouseCallback(WINDOW, on_click)
    point_px, free_result = [], None # two-point measurement
    selected, typed = None, ""       # a pair, "free", or None; the ruler value being typed
    message, message_colour = None, BLUE
    offline = saved_images(args.source)                                   # saved images: no robot here
    stage = "click"                  # click -> robot -> confirm -> done, one after another; or "fix" / "failed"
    accepted = False                 # the fit from the clicks was accepted and saved
    queue_ = list(names)             # squares still to click in this round
    click_px = {}                    # round 1: pixel used for each square (the hole centre, or the click)
    clicked = {}                     # round 1: the same, on the marker layout (metres): kept for refits
    raw_px = {}                      # where you actually clicked, when it was moved to the hole centre
    confirm_px = {}                  # round 2 pixels
    missed = {}                      # square -> 1 after a click where the hole was not found
    placement = None                 # last fit of the layout to the Lab 5 positions
    place = None                     # (R, t) of the saved calibration: layout -> robot coordinates
    off_mm = {}                      # confirmation: distance of each square from its Lab 5 position
    robot_ok = set()                 # squares centred with the robot in THIS run (3rd tick)
    skip_robot = False
    tester = None                    # RobotTest, once T has been used
    test = None                      # robot check in progress
    last_shown = None
    loaded = None                    # time of a calibration loaded as already complete
    improved = None                  # step 5 result saved in markers.json
    marker_queue = []                # step 5: markers still to visit with the robot
    marker_robot = {}                # step 5: marker id -> robot x, y where the gripper was centred
    marker_skipped = set()
    report = []                      # step 5: lines shown after it

    # Already complete (step 1.5 ticked, Lab 5 unchanged)? Start on the finished checklist.
    status_15 = {row[0]: row for row in project_status(max_age=0)}.get("1.5")
    saved = load_saved_markers()
    if squares and status_15 and status_15[2] and saved is not None:
        data = json.loads(cfg.MARKERS_FILE.read_text())
        R0, t0, _ = rigid_fit([layout[i] for i in cfg.MARKER_IDS], [saved[i] for i in cfg.MARKER_IDS],
                              stretch=True)                     # step 5 may have saved a stretch
        place = (R0, t0)
        on_layout = data.get("squares_clicked_on_layout_mm", {})
        if all(n in on_layout for n in names):
            clicked.update({n: np.array(on_layout[n], float) / 1000 for n in names})
            placement = place_layout(squares, clicked)
        off_mm.update({n: float(v) for n, v in data["check"]["off_mm"].items()})
        robot_ok |= robot_checked()
        accepted, stage, loaded = True, "done", data.get("saved", "")[:16].replace("T", " ")
        improved = data.get("improved")
        print(f"Step 1.5 already complete (calibration saved {loaded}).")

    def to_robot(pt):
        x, y = view.camera_to_robot(*pt)
        return place[0] @ np.array([x, y]) + place[1]

    def to_camera(xy):
        lx, ly = np.linalg.solve(place[0], np.asarray(xy, float) - place[1])
        return view.robot_to_camera(lx, ly)

    def fit_and_save(p):
        """Accept a fit: save the marker positions and use it from now on."""
        nonlocal place
        save_markers(p, clicked)
        place = (p["R"], p["t"])
        print(f"Saved the marker positions to {cfg.MARKERS_FILE}")

    def refit():
        """Place the layout again from the round-1 clicks (after a Lab 5 position changed). No new clicks."""
        nonlocal squares, placement, accepted
        squares = {s: np.array(xy) for s, xy in square_centres(load_lab5_poses()).items()}
        placement = place_layout(squares, clicked)
        all_robot = robot_ok >= set(names)
        ok = placement["case"] == "good" or (all_robot and abs(placement["turn"]) <= MAX_TURN_ROBOT_DEG
                                             and max(placement["left"].values()) <= FIT_ROBOT_MM)
        print(f"Refit: {placement['case']}, fit " + ", ".join(f"{n} {v:.1f} mm" for n, v in placement["left"].items())
              + f", turn {placement['turn']:+.2f} deg" + (" (all squares centred with the robot)" if all_robot else ""))
        accepted = ok
        if ok:
            fit_and_save(placement)
        return ok

    def checklist():
        rows = {}
        fit_left = placement["left"] if placement else {}
        for n in names:
            c1 = "ok" if n in clicked or (loaded and n in off_mm) else ("next" if stage == "click" and queue_ and queue_[0] == n else "todo")
            if loaded and accepted:
                c2 = "ok"
            elif placement is None or n not in fit_left:
                c2 = "todo"
            elif stage == "failed":
                c2 = "bad"
            elif accepted or n in robot_ok:
                c2 = "ok"                    # fits, or its Lab 5 position was centred with the robot
            elif stage == "robot":
                c2 = "later"                 # not a failure: the robot check of this square corrects it
            else:
                c2 = "bad" if placement["case"] != "one_square" or placement["odd"] == n else "ok"
            if skip_robot:
                c3 = "skip"
            elif n in robot_ok:
                c3 = "ok"
            else:
                c3 = "next" if stage == "robot" and next_robot() == n else "todo"
            if n in off_mm:
                c4 = "ok" if off_mm[n] <= TOLERANCE_MM else "bad"
            else:
                c4 = "next" if stage == "confirm" and queue_ and queue_[0] == n else "todo"
            if improved:
                c5 = "ok"
            elif stage == "improve":
                c5 = "next"
            else:
                c5 = "todo"
            if n == names[0]:                                    # a few words beside the box, on the first row
                if stage == "improve":
                    c5 += f"|marker {len(marker_robot) + len(marker_skipped) + 1}/{len(cfg.MARKER_IDS)}"
                elif improved:
                    c5 += f"|{improved.get('after_mm', 0):.1f} mm"
                elif stage == "done":
                    c5 += "|press I"
            rows[n] = [c1, c2, c3, c4, c5]
        return rows

    def next_robot():
        left_ = [n for n in names if n not in robot_ok]
        return left_[0] if left_ else None

    def start_confirm():
        nonlocal stage, queue_
        stage, queue_ = "confirm", list(names)
        confirm_px.clear()
        raw_px.clear()
        off_mm.clear()

    def restart():
        nonlocal stage, queue_, placement, message, accepted, loaded, improved
        stage, queue_, placement, message, accepted = "click", list(names), None, None, False
        loaded = improved = None
        marker_queue.clear()
        marker_robot.clear()
        marker_skipped.clear()
        report.clear()
        for d in (click_px, clicked, raw_px, confirm_px, off_mm, missed):
            d.clear()

    def hint():
        """What to do now, for a key or click that is not part of the current step."""
        if test is not None and test["step"] == "confirm":
            return f"Press Y to move the robot to {test['what']}, or N to cancel."
        if test is not None and test.get("arrived"):
            return "Use D A W S to centre the gripper, then Y = centred (save) or N = leave it."
        if test is not None:
            return "Wait: the robot is moving."
        if stage == "click":
            return f"Step 1: click the centre of square {queue_[0]}."
        if stage == "robot":
            return f"Step 3: press T to check square {next_robot()} with the robot."
        if stage == "confirm":
            return f"Step 4: click the centre of square {queue_[0]} again."
        if stage == "fix":
            return "Press U to correct the square, or R to start again."
        if stage == "failed":
            return "Press R to start again."
        if stage == "improve":
            return f"Step 5: press T to send the robot to marker {marker_queue[0]}."
        return "Step 1.5 is complete. Q = quit, R = calibrate again."

    def marker_test(i):
        """Step 5: the robot hovers over marker i where the current calibration puts it."""
        poses = load_lab5_poses()
        xy = place[0] @ layout[i] + place[1]
        z = max(poses["pick"][n][1][2] for n in names) + MARKER_HOVER_M
        pose = [float(xy[0]), float(xy[1]), z] + list(poses["rpy"])
        for p_ in ([pose[0], pose[1], poses["z_safe"]] + pose[3:], pose):
            check_pose(p_, poses)
        return {"step": "confirm", "name": i, "what": f"marker {i}", "marker": i, "pose": pose,
                "z_safe": poses["z_safe"]}

    def finish_markers():
        """Step 5: fit the layout with A, B, C and the markers the robot visited; keep it only if better."""
        nonlocal place, improved, stage, message, message_colour
        stage = "done"
        ids = [i for i in cfg.MARKER_IDS if i in marker_robot]
        report.clear()
        if not ids:
            message, message_colour = "No marker was measured: calibration kept as it was.", ORANGE
            return
        labels = list(names) + [f"marker {i}" for i in ids]
        src = np.array([clicked[n] for n in names] + [layout[i] for i in ids], float)
        dst = np.array([squares[n] for n in names] + [marker_robot[i] for i in ids], float)
        before = np.linalg.norm(src @ place[0].T + place[1] - dst, axis=1) * 1000
        # Markers far apart: the robot's own scale (its distances vs the ruler-measured layout) can be fitted too
        Rs, ts, after_s = rigid_fit(src, dst, stretch=True)
        scale_pct = (math.hypot(Rs[0, 0], Rs[1, 0]) - 1) * 100
        use_stretch = len(ids) >= MIN_MARKERS_STRETCH and abs(scale_pct) <= MAX_STRETCH_PCT
        R, t, after = (Rs, ts, after_s) if use_stretch else rigid_fit(src, dst)
        print("Step 5, distance left per point (mm), before -> after:")
        for lab, b, a in zip(labels, before, after):
            print(f"  {lab:9s} {b:5.1f} -> {a:5.1f}")
        print(f"  turn {turn_deg(R):+.2f} deg, robot distances / layout distances {scale_pct:+.2f} % "
              f"({'included' if use_stretch else 'not included'})")
        better = after.max() < before.max() and after[:len(names)].max() <= TOLERANCE_MM
        report.extend([f"- Worst distance over A, B, C and {len(ids)} marker(s): {before.max():.1f} mm before, "
                       f"{after.max():.1f} mm after.",
                       "- Per point after: " + ", ".join(f"{lab} {a:.1f}" for lab, a in zip(labels, after)) + " mm"])
        if use_stretch:
            report.append(f"- The robot's distances are {scale_pct:+.1f} % of the marker layout: included in the "
                          "calibration (each robot has its own small scale error).")
        elif abs(scale_pct) > MAX_STRETCH_PCT:
            report.append(f"- Robot and layout distances differ by {scale_pct:+.1f} %: too much to be the robot. Measure "
                          "the markers with a ruler (MARKER_LAYOUT_MM in config.py) and tell the technician.")
        else:
            report.append(f"- Only {len(ids)} marker(s) measured: the robot's scale ({scale_pct:+.1f} %) needs "
                          f"{MIN_MARKERS_STRETCH} or more, so it is not included.")
        if not better:
            message, message_colour = "Step 5 did not improve the fit: calibration kept as it was.", ORANGE
            return
        p_ = {"R": R, "t": t, "turn": turn_deg(R), "left": dict(zip(names, after[:len(names)]))}
        check = json.loads(cfg.MARKERS_FILE.read_text()).get("check")
        save_markers(p_, clicked)
        data = json.loads(cfg.MARKERS_FILE.read_text())
        off = {n: round(float(a), 1) for n, a in zip(names, after)}
        data["check"] = {"time": datetime.now().isoformat(timespec="seconds"), "off_mm": off, "passed": True,
                         "method": "A, B, C and markers centred with the robot (step 5)",
                         "previous": check}
        improved = {"time": datetime.now().isoformat(timespec="seconds"), "markers": ids,
                    "before_mm": round(float(before.max()), 1), "after_mm": round(float(after.max()), 1),
                    "scale_pct": round(scale_pct, 2), "scale_included": bool(use_stretch),
                    "markers_robot_mm": {str(i): [round(float(v) * 1000, 1) for v in marker_robot[i]] for i in ids}}
        data["improved"] = improved
        cfg.MARKERS_FILE.write_text(json.dumps(data, indent=2))
        project_status(max_age=0)
        place = (R, t)
        off_mm.clear()
        off_mm.update(off)
        message, message_colour = f"Step 5 saved: worst {before.max():.1f} -> {after.max():.1f} mm.", GREEN

    while True:
        frame = cam.read()
        centres, corners, ids = detector.detect(frame)
        view.update(centres)
        shown = frame.copy()
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(shown, corners, ids)
        seen = sum(1 for i in cfg.MARKER_IDS if i in centres)
        ready = view.ready()
        if stage == "improve" and test is None and not marker_queue:
            finish_markers()
        if stage == "robot" and test is None and (skip_robot or robot_ok >= set(names)):
            if not accepted and not skip_robot:
                refit()                                        # all three centred with the robot: looser limits
            if accepted:
                start_confirm()                                # all robot checks done: confirm
            elif skip_robot and placement["case"] == "one_square":
                stage = "fix"                                  # no robot: U corrects the square from the camera
            else:
                stage = "failed"

        marker_px = {}
        if ready:
            for i in cfg.MARKER_IDS:
                marker_px[i] = np.asarray(centres[i] if i in centres else view.robot_to_camera(*layout[i]), float)

        # ---------------------------------------------------------- clicks
        while clicks:
            pt = clicks.pop(0)
            if not ready:
                print(f"Click ignored: only {seen}/4 markers found.")
                continue
            if stage in ("click", "confirm") and queue_ and test is None:
                name = queue_[0]
                hole, score = snap_to_hole(frame, view, view.camera_to_robot(*pt))
                if hole is None and missed.get(name, 0) == 0:   # ask once more before using a bare click
                    missed[name] = 1
                    message, message_colour = (f"Hole {name} not found around your click: click again, nearer the "
                                               f"centre of {name}.", RED)
                    print(f"square {name}: hole edges not found (score {score:.2f}), asked to click again")
                    continue
                queue_.pop(0)
                missed.pop(name, None)
                message = None
                used = tuple(view.robot_to_camera(*hole)) if hole is not None else pt
                if hole is not None:
                    raw_px[name] = pt
                print(f"square {name}: " + (f"hole found (score {score:.2f}), click moved to its centre"
                                            if hole is not None else "hole not found, your click is used"))
                if stage == "click":
                    click_px[name] = used
                    clicked[name] = np.array(view.camera_to_robot(*used))
                    if not queue_:                                     # all three clicked: place the layout
                        placement = place_layout(squares, clicked)
                        print(f"Calibration: {placement['case']}, fit "
                              + ", ".join(f"{n} {v:.1f} mm" for n, v in placement["left"].items())
                              + f", turn {placement['turn']:+.2f} deg")
                        accepted = placement["case"] == "good"
                        if accepted:
                            fit_and_save(placement)
                        # markers that do not match the layout cannot be fixed with the robot
                        stage = "failed" if placement["case"] == "turned" else "robot"
                        raw_px.clear()
                else:
                    confirm_px[name] = used
                    if not queue_:                                     # confirmation round complete
                        off_mm.update({n: float(np.hypot(*(squares[n] - to_robot(confirm_px[n])))) * 1000
                                       for n in names})
                        for n, d in off_mm.items():
                            save_row(f"square {n} centre vs Lab 5 position", d, 0)
                        save_check(off_mm)
                        stage = "done"
                        print("Confirmation: " + ", ".join(f"{n} {d:.1f} mm" for n, d in off_mm.items()))
                continue
            if stage != "done":
                message, message_colour = hint(), ORANGE           # no ruler checks before the steps are done
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

        # ---------------------------------------------------------- two-point measurement
        free_mm = None
        for p_ in point_px:
            cv2.circle(shown, (int(p_[0]), int(p_[1])), 7, BLUE, -1)
        if len(point_px) == 2 and ready:
            p0, p1 = (np.array(view.camera_to_robot(*p_)) for p_ in point_px)
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
            fitted = placement is not None and not accepted and stage in ("robot", "fix", "failed")
            if fitted and placement["case"] == "one_square":
                Rr, tr = placement["R_rest"], placement["t_rest"]   # Lab 5 positions placed with the other squares
                for name, xy in squares.items():
                    lx, ly = Rr.T @ (np.asarray(xy) - tr)
                    px, py = view.robot_to_camera(lx, ly)
                    cv2.circle(shown, (int(px), int(py)), 9, RED, 2)
                    put(shown, name, (px + 10, py - 10), RED, 0.6)
                odd = placement["odd"]
                gx, gy = click_px[odd]
                cv2.circle(shown, (int(gx), int(gy)), 13, GREEN, 3, cv2.LINE_AA)
                put(shown, f"{odd} here", (gx + 16, gy + 26), GREEN, 0.6)
            elif fitted:                                             # best fit, shown even though it failed
                for name, xy in squares.items():
                    lx, ly = placement["R"].T @ (np.asarray(xy) - placement["t"])
                    px, py = view.robot_to_camera(lx, ly)
                    cv2.circle(shown, (int(px), int(py)), 9, RED, 2)
                    put(shown, f"{name} {placement['left'][name]:.1f} mm", (px + 10, py - 10), RED, 0.6)
            if place is not None and stage in ("improve", "done"):
                for i in cfg.MARKER_IDS:                             # step 5: robot points on the markers
                    if i in marker_robot:
                        px, py = to_camera(marker_robot[i])
                        cv2.circle(shown, (int(px), int(py)), 7, RED, 2)
                    elif stage == "improve" and marker_queue and marker_queue[0] == i:
                        px, py = to_camera(place[0] @ layout[i] + place[1])
                        cv2.circle(shown, (int(px), int(py)), 16, ORANGE, 3)
                        put(shown, "next", (px + 18, py - 14), ORANGE, 0.6)
            if place is not None and stage != "click" and not fitted:
                for name, xy in squares.items():                     # Lab 5 positions on the picture
                    px, py = to_camera(xy)
                    cv2.circle(shown, (int(px), int(py)), 9, RED, 2)
                    put(shown, name, (px + 10, py - 10), RED, 0.7)
                    if name in off_mm:
                        ok = off_mm[name] <= TOLERANCE_MM
                        put(shown, f"{off_mm[name]:.1f} mm", (px + 12, py + 24), GREEN if ok else RED, 0.6)
            for name, (cx, cy) in (confirm_px if stage in ("confirm", "done") else click_px).items():
                if name in raw_px:
                    ux, uy = raw_px[name]
                    cv2.circle(shown, (int(ux), int(uy)), 3, BLUE, -1)
                    cv2.line(shown, (int(ux), int(uy)), (int(cx), int(cy)), BLUE, 1, cv2.LINE_AA)
                xy = view.camera_to_robot(cx, cy)
                sq = [view.robot_to_camera(xy[0] + dx * HOLE_MM / 2000, xy[1] + dy * HOLE_MM / 2000)
                      for dx, dy in ((-1, -1), (-1, 1), (1, 1), (1, -1))]
                cv2.polylines(shown, [np.int32(sq)], True, GREEN, 1, cv2.LINE_AA)
                cv2.drawMarker(shown, (int(cx), int(cy)), BLUE, cv2.MARKER_CROSS, 18, 2)

        # ---------------------------------------------------------- robot state
        if test is not None and test.get("step") == "sent" and tester is not None:
            if tester.state == "arrived":
                test["arrived"] = True                                   # down at the square: adjusting allowed
            elif tester.state == "failed" or (tester.state == "ready" and tester.status.startswith("Move failed")):
                message, message_colour, test = tester.status, RED, None # could not connect, or the move failed

        # ---------------------------------------------------------- instruction box
        keys = "- Backspace = redo the previous square    - R = start again    - Q = quit"
        if not squares:
            title, colour = "Lab 5 positions needed first", RED
            lines = ["- Run python step1_lab5_positions.py: enter, test and save your Lab 5 positions.", "- Q = quit"]
        elif not ready:
            title, colour = f"Markers not found ({seen}/4) - clicks are ignored", RED
            lines = ["- Check that nothing hides a marker: the robot arm, a hand, a cube.",
                     "- Robot arm in the way? Click ROBOT: LEARNING MODE (or press L): it goes to its sleep pose.",
                     "- Check the focus, or run step1_tune_camera.py again.", "- Q = quit"]
        elif test is not None:
            colour = ORANGE
            if test["step"] == "confirm":
                n, pose = test["what"], test["pose"]
                title = f"Robot check of {n}: press Y to move, N to cancel"
                lines = [f"- The robot moves above {n} (z = {test['z_safe']:.3f} m), then down to x = {pose[0]:.3f}, "
                         f"y = {pose[1]:.3f}, z = {pose[2]:.3f}"
                         + (f" ({MARKER_HOVER_M * 1000:.0f} mm above your level-1 pick height)." if "marker" in test
                            else "."),
                         "- KEEP CLEAR OF THE ROBOT.",
                         "- Close every other program connected to the robot first."]
            elif tester is not None and test.get("arrived"):
                n = test["name"]
                dx, dy = (test["now"][0] - test["pose"][0]) * 1000, (test["now"][1] - test["pose"][1]) * 1000
                title, colour = (f"Marker {n}: is the gripper centred over the middle of the marker?" if "marker" in test
                                 else f"Square {n}: is the gripper centred over the hole?"), BLUE
                lines = ["- Stand in front of the robot, at least 30 cm away. Look from the front, then from the side.",
                         "- Not centred? Move it one step per key:  D = your right   A = your left   "
                         "W = away from you   S = towards you",
                         f"- R / F = up / down to look closer (height never saved)   1 / 5 = step 1 or 5 mm "
                         f"(now {test['step_mm']:g} mm)",
                         f"- Moved so far: x {dx:+.1f} mm, y {dy:+.1f} mm" + ("  (moving...)" if tester.state == "moving" else ""),
                         ("- Y = centred: use this point (your Lab 5 positions are not changed)" if "marker" in test else
                          f"- Y = centred: save it for {n} (same x, y at every level, pick-up and drop; heights unchanged)"),
                         ("- N = skip this marker" if "marker" in test else "- N = leave it as it was")
                         + "    - L = leave it, robot to sleep pose + learning mode"]
            else:
                title = f"Robot check of {test['what']}"
                lines = ["- " + (tester.status if tester else ""), "- Please wait."]
        elif selected is not None:
            what = "the two points" if selected == "free" else f"marker {selected[0]} to {selected[1]}"
            title, colour = f"Type your ruler value for {what}, then press Enter", ORANGE
            lines = [f"- Ruler value: {typed}_ mm",
                     "- " + (message if message_colour == RED else "Backspace = delete     Esc = cancel")]
        elif stage == "click":
            name = queue_[0]
            title, colour = f"1 Click the centre of square {name}  ({len(names) - len(queue_) + 1}/{len(names)})", BLUE
            lines = [f"- Click the centre of hole {name}. The program finds the hole's edges (green outline) and uses its centre.",
                     "- After C, it places the marker layout so that A, B and C land on your Lab 5 positions.",
                     "- Then red circles show where your Lab 5 positions are on the picture.",
                     "- Robot arm over the platform? Click ROBOT: LEARNING MODE (right) or press L.",
                     "- Click the picture first.", keys]
        elif stage == "failed" and robot_ok >= set(names) and placement and placement["case"] != "turned":
            fits = ", ".join(f"{n} {placement['left'][n]:.1f} mm" for n in placement["names"])
            title, colour = "Not calibrated: A, B and C still do not fit after the robot checks", RED
            lines = [f"- Distance left after the best fit: {fits} (needs {FIT_ROBOT_MM:g} mm or less, turn "
                     f"{placement['turn']:+.1f} deg, needs {MAX_TURN_ROBOT_DEG:g} or less).",
                     "- Your clicks were probably off, or a marker has moved. Press R and click A, B, C again.",
                     "- If it repeats, do the ruler checks and tell the technician.", "- R = start again    - Q = quit"]
        elif stage in ("fix", "failed"):
            title, colour, lines = placement_lines(placement, robot_ok)
        elif stage == "robot":
            n = next_robot()
            k = names.index(n) + 1
            title, colour = f"3 Press T to check square {n} with the robot  ({k}/{len(names)})", BLUE
            lines = [("- The camera fits your Lab 5 positions (ticks 2). Now the robot checks each square, A, then B, then C."
                      if accepted else
                      "- The camera fit is not finished yet (orange dots): that is normal, the robot checks correct it."),
                     f"- Press T: the robot will go down to square {n}. You then press Y to start the move.",
                     "- At the square you centre the gripper with the keys shown, and press Y.",
                     "- Keep clear of the robot. Close every other program connected to the robot first.",
                     "- L = robot to sleep pose with learning mode on, any time."]
            if offline:
                lines.append("- No robot here (saved images): S = skip the robot checks.")
            lines.append("- R = start again    - Q = quit")
        elif stage == "improve":
            i = marker_queue[0]
            k = len(marker_robot) + len(marker_skipped) + 1
            title, colour = f"5 Press T to send the robot to marker {i}  ({k}/{len(cfg.MARKER_IDS)}, optional)", BLUE
            lines = ["- The robot visits the 4 markers too. They are far apart, so the fit gets more accurate than with "
                     "A, B, C alone.",
                     f"- Press T, then Y: the robot hovers {MARKER_HOVER_M * 1000:.0f} mm above marker {i} (orange circle).",
                     "- Centre the gripper over the middle of the marker with D A W S, then Y. N = skip this marker.",
                     "- At the end the calibration is saved only if it is better. Step 1.5 stays ticked.",
                     "- Keep clear of the robot.    - L = robot to sleep pose + learning mode    - Q = quit"]
        elif stage == "confirm":
            name = queue_[0]
            title, colour = (f"4 Confirm: click the centre of square {name} again  "
                             f"({len(names) - len(queue_) + 1}/{len(names)})"), GREEN
            lines = ["- Each red circle (your Lab 5 position) should sit in the centre of its square.",
                     f"- Click the centre of {name}. Every square within {TOLERANCE_MM:g} mm passes.", keys]
        else:                                                            # done
            offs = ", ".join(f"{n} {d:.1f} mm" for n, d in off_mm.items())
            if max(off_mm.values()) <= TOLERANCE_MM:
                title, colour = (f"Step 1.5 already complete (saved {loaded}): {offs}" if loaded
                                 else f"Step 1.5 complete: {offs}"), GREEN
                lines = list(report) + ["- Next step: python step2_collect_data.py"]
                if not improved:
                    lines.append("- Optional, more accurate: press I. The robot also visits the 4 markers (about 5 min).")
                else:
                    lines.append(f"- Step 5 done ({improved['before_mm']:.1f} -> {improved['after_mm']:.1f} mm"
                                 + (f", robot scale {improved['scale_pct']:+.1f} % included"
                                    if improved.get("scale_included") else "") + "). I = do it again.")
                lines += ["- Optional ruler checks: click a line between two markers, type your ruler value, Enter.",
                          "- R = calibrate again from step 1    - Q = quit"]
            else:
                title, colour = f"Not confirmed: {offs}", RED
                lines = ["- The second clicks do not agree with the first within 5 mm.",
                         "- Press R and do it again, clicking the centre of each hole.", "- Q = quit"]
        if message and message_colour in (RED, GREEN, ORANGE) and selected is None:
            lines = ["- " + message] + lines
        if tester is not None and test is None:
            lines = lines + [f"- {tester.status}"]

        pic = with_progress(shown, "1.5")
        strip = checklist_panel(frame.shape[1], checklist()) if squares else np.full((150, frame.shape[1], 3), 255, np.uint8)
        button, rect = robot_button(pic.shape[1] - frame.shape[1], strip.shape[0],
                                    tester.status if tester is not None else "Robot not connected yet")
        pic = np.vstack([np.hstack([strip, button]), pic])
        shown = instruction_box(pic, title, lines, colour)
        top = shown.shape[0] - pic.shape[0]                              # the button, in window pixels
        button_rect[0] = (frame.shape[1] + rect[0], top + rect[1], frame.shape[1] + rect[2], top + rect[3])

        panel_offset[0] = shown.shape[0] - frame.shape[0]
        picture_width[0] = frame.shape[1]
        display_scale[0] = min(1.0, 1280 / shown.shape[1], 1000 / shown.shape[0])   # fits a 1080p screen
        last_shown = shown
        cv2.imshow(WINDOW, cv2.resize(shown, None, fx=display_scale[0], fy=display_scale[0]))

        # ---------------------------------------------------------- keys
        key = cv2.waitKey(30) & 0xFF
        if button_clicks:                                                # the button = key L
            button_clicks.clear()
            key = ord("l")
        if key == 255:
            continue
        char = chr(key)
        if char in "lL":                                                 # robot to rest, learning mode on
            if test is not None and test["step"] == "sent" and not test.get("arrived"):
                message, message_colour = "Wait until the robot has stopped, then press L.", ORANGE
            elif test is not None and test.get("arrived"):               # at a square: leave it, go up and rest
                tester.send("up", list(test["now"]), test["z_safe"])
                message, message_colour = f"{test['what'].capitalize()}: left as it was. Robot going to rest.", ORANGE
                test = None
            else:
                test = None
                if tester is None or tester.state == "failed":
                    tester = RobotTest()
                    tester.start()
                tester.send("rest")
                message, message_colour = "Robot going to its sleep pose, then learning mode on.", ORANGE
            continue
        if test is not None:                                             # robot keys come first
            if test["step"] == "confirm" and (key == 27 or char in "nN"):
                test = None
            elif test["step"] == "confirm" and char in "yY":
                if tester is None or tester.state == "failed":
                    tester = RobotTest()
                    tester.start()
                tester.send("test", test["what"], test["pose"], test["z_safe"])
                test.update(step="sent", now=list(test["pose"]), step_mm=1.0)
                message = None
            elif test["step"] == "sent" and test.get("arrived") and char in "wWsSaAdDrRfF":
                # As seen standing in front of the robot: towards you = robot +x, your right = robot +y
                step = test["step_mm"] / 1000
                move = {"s": (step, 0, 0), "w": (-step, 0, 0), "d": (0, step, 0), "a": (0, -step, 0),
                        "r": (0, 0, step), "f": (0, 0, -step)}[char.lower()]
                new = [test["now"][i] + move[i] for i in range(3)] + test["pose"][3:]
                new[2] = min(max(new[2], test["pose"][2]), test["z_safe"])   # never below the pick height
                far = max(abs(new[0] - test["pose"][0]), abs(new[1] - test["pose"][1])) * 1000
                try:
                    if far > MAX_ADJUST_MM:
                        raise ValueError(f"more than {MAX_ADJUST_MM:g} mm from your Lab 5 position: check the square")
                    check_pose(new, load_lab5_poses())
                except ValueError as e:
                    message, message_colour = f"Not moved: {e}", RED
                    continue
                test["now"] = new
                tester.send("jog", new)
            elif test["step"] == "sent" and test.get("arrived") and char in "15":
                test["step_mm"] = float(char)
            elif test["step"] == "sent" and tester.state == "arrived" and char in "yYnN" and "marker" in test:
                i, now = test["marker"], list(test["now"])
                tester.send("up", now, test["z_safe"])
                test = None
                marker_queue.pop(0)
                if char in "yY":
                    marker_robot[i] = np.array(now[:2])
                    message, message_colour = f"Marker {i}: point saved.", GREEN
                    print(f"marker {i}: robot x {now[0]:.4f}, y {now[1]:.4f}")
                else:
                    marker_skipped.add(i)
                    message, message_colour = f"Marker {i} skipped.", ORANGE
            elif test["step"] == "sent" and tester.state == "arrived" and char in "yYnN":
                name, now = test["name"], list(test["now"])
                dx, dy = (now[0] - test["pose"][0]) * 1000, (now[1] - test["pose"][1]) * 1000
                tester.send("up", now, test["z_safe"])                  # back up and rest in learning mode
                test = None
                if char in "nN":
                    message, message_colour = f"Square {name}: left as it was.", ORANGE
                elif abs(dx) < 0.05 and abs(dy) < 0.05:
                    record_test(name, True)
                    robot_ok.add(name)
                    message, message_colour = f"Square {name}: centred with the robot - OK.", GREEN
                    print(f"Square {name}: centred with the robot, no change")
                else:
                    try:
                        backup = set_square_xy(name, now[:2])
                    except ValueError as e:
                        message, message_colour = f"Not saved: {e}", RED
                        continue
                    record_test(name, True)
                    robot_ok.add(name)
                    print(f"Square {name}: Lab 5 x, y moved by x {dx:+.1f} mm, y {dy:+.1f} mm with the robot "
                          f"(old file kept as calibration/{backup})")
                    if clicked and len(clicked) == len(names):
                        ok = refit()                                    # circles placed again, no new clicks
                        message, message_colour = (
                            f"Square {name} saved (x {dx:+.1f}, y {dy:+.1f} mm). Circles refitted automatically"
                            + (" - the camera fits now." if ok else ": not finished yet, carry on."), GREEN)
                    else:
                        message, message_colour = f"Square {name} saved (x {dx:+.1f}, y {dy:+.1f} mm).", GREEN
            else:
                message, message_colour = hint(), ORANGE                  # not part of this step
            continue
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
        elif key in (8, 127) and stage in ("click", "confirm"):           # redo the previous square
            done_ = [n for n in names if n not in queue_]
            if done_:
                last = done_[-1]
                (click_px if stage == "click" else confirm_px).pop(last, None)
                clicked.pop(last, None) if stage == "click" else None
                raw_px.pop(last, None)
                queue_.insert(0, last)
        elif char in "tT" and stage == "robot":                           # robot check of the next square
            name = next_robot()
            poses = load_lab5_poses()
            pose = list(poses["pick"][name][1]) + list(poses["rpy"])
            try:
                for p_ in ([pose[0], pose[1], poses["z_safe"]] + pose[3:], pose):
                    check_pose(p_, poses)
                test = {"step": "confirm", "name": name, "what": f"square {name}", "pose": pose, "z_safe": poses["z_safe"]}
                message = None
            except ValueError as e:
                message, message_colour = str(e), RED
        elif char in "tT" and stage == "improve":                         # step 5: next marker
            i = marker_queue[0]
            try:
                test = marker_test(i)
                message = None
            except ValueError as e:
                marker_queue.pop(0)
                marker_skipped.add(i)
                message, message_colour = f"Marker {i} skipped: {e}", ORANGE
        elif char in "iI" and stage == "done" and max(off_mm.values()) <= TOLERANCE_MM:
            if len(clicked) < len(names):
                message, message_colour = "Step 5 needs your clicks on A, B, C: press R and calibrate again first.", RED
            elif offline:
                message, message_colour = "Step 5 needs the robot: not possible with saved images.", RED
            else:
                stage = "improve"
                marker_queue[:] = list(cfg.MARKER_IDS)
                marker_robot.clear()
                marker_skipped.clear()
                report.clear()
                message = None
        elif char in "sS" and stage == "robot" and offline:
            skip_robot = True
            print("Robot checks skipped (saved images)")
        elif char in "uU" and stage == "fix" and placement["odd"] not in robot_ok:
            odd = placement["odd"]
            dx, dy = placement["odd_shift_mm"]
            try:
                backup = set_square_xy(odd, placement["odd_seen"])
            except ValueError as e:
                message, message_colour = f"Not changed: {e}", RED
                continue
            print(f"Square {odd}: Lab 5 position moved by x {dx:+.1f} mm, y {dy:+.1f} mm from the camera "
                  f"(old file kept as calibration/{backup})")
            if refit():
                start_confirm()
            message, message_colour = (f"Square {odd} moved by x {dx:+.1f}, y {dy:+.1f} mm from the camera.", GREEN)
        elif char in "rR" and squares:
            restart()
        elif char in "cC" and stage == "done":
            point_px, free_result, selected, typed = [], None, None, ""
            pair_result.clear()
            message, message_colour = None, BLUE
        elif char in "qQ":
            break
        else:
            message, message_colour = hint(), ORANGE                      # not part of this step
    if tester is not None:
        print("Closing the robot connection (sleep pose)...")
        tester.close()
    cam.release()
    cv2.destroyAllWindows()
    return last_shown


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as e:
        raise SystemExit(f"\nMissing: {e}\nCheck the function named in the message.")
