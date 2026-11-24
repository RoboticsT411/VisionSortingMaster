"""
STEP 1.5 - Check the camera-to-robot calibration, and correct it until it is right.

    python step1_check_calibration.py
    python step1_check_calibration.py --source dataset_raw     (saved images)

The box at the top of the picture always says what to do next.
FIRST CLICK ON THE PICTURE so it receives your key presses.

WHERE THE RED CIRCLES A, B, C COME FROM
    Each circle is your Lab 5 pick-up position for that square at level 1 (step 1.1,
    step1_lab5_positions.py), drawn on the picture with the camera-to-robot mapping. That
    mapping comes from the four markers: where the camera sees them (now, in every frame) and
    where the robot says they are (step 1.4, step1_measure_markers.py). So a circle that is not
    in the centre of its square means step 1.1 and step 1.4 disagree with each other.

1. Square check. Click the centre of A, then B, then C. The program then says which step is
   wrong and what to do:
     - all three off the same way  -> the marker positions (step 1.4) are shifted. Fix them here:
           F  = correct the marker positions from your clicks (no robot needed), or
           K  = connect the robot and read the markers again by hand (see 3).
     - one square off on its own   -> that square's Lab 5 position (step 1.1) is wrong.
     - different amounts each      -> click again carefully, then check the ruler distances.
   The distances update live while the markers are corrected, so you see the circles move.
2. Ruler checks. Lines join the marker centres, labelled with their length in mm. Measure the
   same distance with a ruler, click the line, type your value and press Enter: green within
   5 mm, red if not. Or click two points away from the lines and compare their distance.
3. Robot (optional). K connects to the robot chosen in step 1.1 (it calibrates, ~30 s). A blue
   dot shows where the robot says the gripper is. L switches learning mode on, so you can move
   the gripper by hand: point it straight down, centre the closed fingers about 5 mm above a
   marker and press that marker's number (0-3). The marker fit is redone and saved at once.
   Close every other program connected to the robot first.

All positions are robot coordinates (origin at the robot base, x forward, y to the robot's left).

Keys:  0-9 and .  type a ruler value     Enter = compare     Backspace = delete / redo last square
       Esc = cancel   F = fix markers   R = repeat the square check   S = skip it
       K = connect robot   L = learning mode   0-3 = read that marker (robot connected)
       C = clear     Q = quit
Every comparison is saved to results/calibration_checks.csv.
"""
import argparse
import csv
import json
import math
import queue
import shutil
import threading
import time
from datetime import datetime
from itertools import combinations

import cv2
import numpy as np

import config as cfg
from aruco_view import MarkerDetector, TopDownView, load_marker_positions
from camera import Camera
from robot import load_lab5_poses, square_centres
from progress import with_progress
from ui import instruction_box

WINDOW = "calibration check"
TOLERANCE_MM = 5.0
SAME_SHIFT_MM = 2.5         # after removing one common shift + turn, each square is within this: "all moved together"
BIG_SHIFT_MM, BIG_TURN_DEG = 25.0, 5.0   # a correction this large usually means a marker in the wrong corner
ONE_SQUARE_TURN_DEG = 2.0  # "one square off" only if the other squares agree without a bigger turn than this
MAX_TILT_DEG = 3.0          # same limit as step 1.4
LINE_PICK_PX = 12           # a click this close to a marker line selects it
GREEN, RED, BLUE, ORANGE, GREY = (0, 170, 0), (0, 0, 230), (200, 90, 0), (0, 140, 255), (60, 60, 60)
CYAN = (200, 160, 0)

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
# Diagnosis: which step is wrong, from where you clicked A, B, C
# ======================================================================================
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


def diagnose(lab5, seen):
    """lab5: {square: (x, y)} from step 1.1.  seen: {square: (x, y)} where the camera puts your click.
    Returns a dict with "case": ok / shift / one_square / inconsistent / few, and the numbers."""
    names = [n for n in sorted(lab5) if n in seen]
    off = {n: float(np.hypot(*(np.subtract(lab5[n], seen[n])))) * 1000 for n in names}
    out = {"names": names, "off": off, "case": "few"}
    if len(names) < 2:
        out["case"] = "ok" if names and max(off.values()) <= TOLERANCE_MM else "few"
        return out
    R, t, left = rigid_fit([seen[n] for n in names], [lab5[n] for n in names])
    shift = (np.mean([lab5[n] for n in names], axis=0) - np.mean([seen[n] for n in names], axis=0)) * 1000
    out.update(R=R, t=t, left=dict(zip(names, left)), turn_deg=math.degrees(math.atan2(R[1, 0], R[0, 0])),
               shift_mm=shift)
    out["big"] = float(np.hypot(*shift)) > BIG_SHIFT_MM or abs(out["turn_deg"]) > BIG_TURN_DEG
    if max(off.values()) <= TOLERANCE_MM:
        out["case"] = "ok"
    elif max(left) <= SAME_SHIFT_MM:
        out["case"] = "shift"
    else:
        # Is it one square on its own? Fit the others, and see whether only that one is left over.
        best = None
        if len(names) >= 3:
            for odd in names:
                rest = [n for n in names if n != odd]
                R2, t2, left2 = rigid_fit([seen[n] for n in rest], [lab5[n] for n in rest])
                miss = float(np.linalg.norm(R2 @ np.asarray(seen[odd]) + t2 - np.asarray(lab5[odd]))) * 1000
                turn2 = abs(math.degrees(math.atan2(R2[1, 0], R2[0, 0])))
                # Two squares can always be matched by turning far enough, so only a small turn counts
                if (max(left2) <= SAME_SHIFT_MM and turn2 <= ONE_SQUARE_TURN_DEG and miss > TOLERANCE_MM
                        and (best is None or miss > best[1])):
                    best = (odd, miss, max(off[n] for n in rest))
        if best:
            out.update(case="one_square", odd=best[0], odd_mm=best[1], others_off=best[2])
        else:
            out["case"] = "inconsistent"
    return out


def gripper_heading_and_tilt(rpy):
    """Heading of the gripper about the vertical and its tilt from straight down, degrees."""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rm = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                   [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                   [-sp, cp * sr, cp * cr]])
    down = Rm[:, 0]                                   # the gripper points along its x axis
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, -down[2]))))
    side = Rm[:, 2]                                   # its z axis is horizontal when pointing down
    return math.degrees(math.atan2(side[1], side[0])), tilt


def orientation_note():
    """A hint if the gripper was held differently in step 1.4 and in step 1.1."""
    try:
        m = json.loads(cfg.MARKERS_FILE.read_text()).get("orientation")
        l5 = json.loads(cfg.LAB5_FILE.read_text()).get("orientation")
        if not m or not l5:
            return None
        hm, _ = gripper_heading_and_tilt(m)
        hl, tl = gripper_heading_and_tilt(l5)
        turn = abs((hm - hl + 180) % 360 - 180)
        notes = []
        if turn > 15:
            notes.append(f"the gripper was turned {turn:.0f} deg differently in step 1.4 than in your Lab 5 positions")
        if tl > MAX_TILT_DEG:
            notes.append(f"your Lab 5 orientation tilts the gripper {tl:.0f} deg from straight down")
        return "; ".join(notes) or None
    except (OSError, ValueError, TypeError):
        return None


def save_correction(R, t, diagnosis):
    """Move and turn the four marker positions by R, t, keep a backup, and record what was done."""
    data = json.loads(cfg.MARKERS_FILE.read_text())
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    shutil.copy(cfg.MARKERS_FILE, cfg.CALIB_DIR / f"markers_backup_{stamp}.json")
    data.setdefault("markers_measured", data["markers"])        # the step 1.4 values, kept once
    data["markers"] = {k: [round(float(v), 4) for v in (R @ np.asarray(xy[:2], float) + t)]
                       for k, xy in data["markers"].items()}
    data.setdefault("corrections", []).append({
        "time": datetime.now().isoformat(timespec="seconds"), "by": "step1_check_calibration.py (F)",
        "turn_deg": round(diagnosis["turn_deg"], 2), "shift_mm": [round(float(v), 1) for v in diagnosis["shift_mm"]],
        "squares_off_before_mm": {n: round(v, 1) for n, v in diagnosis["off"].items()}})
    cfg.MARKERS_FILE.write_text(json.dumps(data, indent=2))
    return stamp


def read_marker_with_robot(marker_id, pose, fresh):
    """Store a new robot reading for one marker and save the refitted layout (as step 1.4 does).
    fresh = the markers already read again in this window (this one is added).
    Returns (ok, message)."""
    from step1_measure_markers import fit_layout, load_saved, save_markers, tilt_from_vertical
    tilt = tilt_from_vertical(pose)
    if tilt > MAX_TILT_DEG:
        return False, (f"Marker {marker_id} NOT read: the gripper is tilted {tilt:.1f} deg (max {MAX_TILT_DEG:g}). "
                       f"Point it straight down and press {marker_id} again.")
    readings, tilts, _ = load_saved()
    readings[marker_id] = [round(v, 4) for v in pose[:3]]
    tilts[marker_id] = round(tilt, 1)
    missing = [i for i in cfg.MARKER_IDS if i not in readings]
    if missing:
        return False, f"markers.json has no reading for marker(s) {missing}: run step1_measure_markers.py first."
    save_markers(readings, tilts, [round(v, 4) for v in pose[3:6]])
    fresh.add(marker_id)
    todo = [i for i in cfg.MARKER_IDS if i not in fresh]
    msg = f"Marker {marker_id} read and saved (tilt {tilt:.1f} deg)."
    if todo:            # the fit mixes new and old readings until all four are read again
        return True, msg + f" Now read marker{'s' if len(todo) > 1 else ''} {', '.join(map(str, todo))} the same way."
    fit_error = fit_layout(readings)[1]
    bad = [i for i, e in fit_error.items() if e > cfg.MAX_FIT_ERROR_MM]
    if bad:
        return False, msg + (f" All four read: marker(s) {', '.join(map(str, bad))} more than "
                             f"{cfg.MAX_FIT_ERROR_MM} mm from the layout fit - read only those again.")
    return True, msg + " All four read and they fit the layout: press R and click A, B, C to confirm."


# ======================================================================================
# The robot runs in its own thread, so the picture keeps moving while it calibrates
# ======================================================================================
class RobotLink(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.commands, self.results = queue.Queue(), queue.Queue()
        self.state = "connecting"          # connecting / ready / failed / closed
        self.status = "Connecting and calibrating (about 30 s)..."
        self.pose = None
        self.learning = False
        self.robot = None

    def run(self):
        try:
            from robot import NiryoRobot
            self.robot = NiryoRobot()
            self.robot.start()
            self.state, self.status = "ready", f"Connected to {self.robot.ip}"
        except BaseException as e:                       # NiryoRobot.start() raises SystemExit on failure
            self.state, self.status = "failed", f"Could not connect: {str(e).strip().splitlines()[0][:90]}"
            return
        while self.state == "ready":
            try:
                cmd = self.commands.get(timeout=0.15)
            except queue.Empty:
                cmd = None
            try:
                if cmd == "learning":
                    self.learning = not self.learning
                    self.robot.learning_mode(self.learning)
                elif isinstance(cmd, tuple) and cmd[0] == "read":
                    self.results.put(("marker", cmd[1], self.robot.get_pose()))
                elif cmd == "quit":
                    self.status = "Moving to the sleep pose..."
                    self.robot.finish()
                    self.state = "closed"
                    return
                self.pose = self.robot.get_pose()
            except Exception as e:
                self.status = f"Robot error: {e}"

    def send(self, cmd):
        if self.state == "ready":
            self.commands.put(cmd)

    def close(self):
        if self.state == "ready":
            self.commands.put("quit")
            self.join(timeout=20)


# ======================================================================================
# What the instruction box says after the square check
# ======================================================================================
def diagnosis_box(d, robot_ready):
    """Title, lines and colour for the result of the square check."""
    offs = ", ".join(f"{n} {d['off'][n]:.1f} mm" for n in d["names"])
    keys = "R = click A, B, C again    Q = quit"
    robot_way = ("K = connect the robot and read the markers again by hand: the circles move live"
                 if not robot_ready else "Robot connected: L = learning mode, then centre the gripper over a "
                                         "marker and press its number 0-3")
    if d["case"] == "ok":
        return (f"Square check passed: {offs}", GREEN,
                ["The camera and your Lab 5 positions agree (all within 5 mm).",
                 "Next: ruler checks. Click a line between two markers, measure it with a ruler, type it, Enter.",
                 keys])
    if d["case"] == "shift":
        sx, sy = d["shift_mm"]
        size = float(np.hypot(sx, sy))
        lines = [f"Off by {offs}. All three are off the same way (about {size:.0f} mm, turned "
                 f"{d['turn_deg']:+.1f} deg),",
                 "so your Lab 5 positions agree with each other: the MARKER POSITIONS (step 1.4) are shifted.",
                 "Usual causes in step 1.4: gripper tilted or not centred over a marker; gripper turned another way",
                 "than in Lab 5; a marker or the platform moved after step 1.4. Green circles = A, B, C after F."]
        note = orientation_note()
        if note:
            lines.append("Found: " + note + ".")
        if d.get("big"):
            lines.append("This is a LARGE correction: first check every marker ID is in its corner as in step 1.4.")
        lines += ["FIX:  F = correct the marker positions from your clicks (no robot needed)",
                  "  or  " + robot_way, keys]
        return f"A, B and C are all off the same way: the markers (step 1.4) need correcting", RED, lines
    if d["case"] == "one_square":
        n = d["odd"]
        return (f"Square {n} is off on its own ({d['off'][n]:.1f} mm): check its Lab 5 position (step 1.1)", RED,
                [f"Off by {offs}. The other squares agree with the camera, so the markers are fine and the",
                 f"problem is {n}'s own position: in step 1.1 the gripper was not centred over {n} when its",
                 "level-1 pick-up was recorded, or a number was typed wrongly. (Or your click was off: R.)",
                 f"FIX: run python step1_lab5_positions.py, select PICK {n} level 1, press Test. If the gripper is",
                 f"off-centre, move it by hand over the centre of {n}, Read robot position, Save. Then run this again.",
                 keys])
    if d["case"] == "inconsistent":
        return ("A, B and C are off by different amounts in different directions", RED,
                [f"Off by {offs}. Neither one shift of the markers nor one bad square explains it.",
                 "1. R = click A, B, C again, carefully on the centre of each hole.",
                 "2. If it stays: ruler checks - click each line between markers. A wrong distance means a marker",
                 "   reading is wrong or a marker ID is in the wrong corner: " + robot_way.split(':')[0].lower() +
                 " or redo step 1.4.",
                 keys])
    return ("Square check incomplete", ORANGE, ["Click the squares, or R to start again.", keys])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None, help="camera index, image, folder or video")
    args = parser.parse_args()

    cam = Camera(args.source)
    detector = MarkerDetector()
    markers = load_marker_positions()
    view = TopDownView(markers)
    print(__doc__)
    try:
        squares = {s: np.array(xy) for s, xy in square_centres(load_lab5_poses()).items()}
    except (FileNotFoundError, ValueError) as e:
        print(f"(A, B, C not drawn: {e})")
        squares = {}

    pairs = list(combinations(cfg.MARKER_IDS, 2))
    pair_mm = {p: float(np.hypot(*(markers[p[0]] - markers[p[1]])) * 1000) for p in pairs}
    pair_result = {}                 # pair -> (ruler, diff, ok)

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_click)
    points, point_px = [], []        # two-point measurement: image pixels (robot x, y is worked out live)
    free_result = None               # (ruler, diff, ok)
    selected, typed = None, ""       # a pair, "free", or None; the ruler value being typed
    message, message_colour = None, BLUE
    square_queue = list(squares)     # guided check first: click the centre of A, then B, then C
    square_px = {}                   # name -> click pixel (the robot x, y is worked out live from it)
    confirming = False               # True after a correction: the square check is being repeated
    robot = None                     # RobotLink when connected
    fresh_markers = set()            # markers read again with the robot in this window
    last_shown = None

    def reload_markers(note):
        nonlocal markers, view
        markers = load_marker_positions()
        view = TopDownView(markers)
        for p in pairs:
            pair_mm[p] = float(np.hypot(*(markers[p[0]] - markers[p[1]])) * 1000)
        print(note)

    while True:
        frame = cam.read()
        centres, corners, ids = detector.detect(frame)
        view.update(centres)
        shown = frame.copy()
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(shown, corners, ids)
        seen = sum(1 for i in cfg.MARKER_IDS if i in centres)
        robot_ready = robot is not None and robot.state == "ready"

        # Pixel position of each marker centre: as detected now, or from the last good mapping.
        marker_px = {}
        if view.ready():
            for i in cfg.MARKER_IDS:
                marker_px[i] = np.asarray(centres[i] if i in centres else view.robot_to_camera(*markers[i]), float)

        # Square check, worked out live from the click pixels: changes as soon as the markers change.
        seen_xy = {n: view.camera_to_robot(*square_px[n]) for n in square_px} if view.ready() else {}
        diag = diagnose(squares, seen_xy) if seen_xy and not square_queue else None

        # ---------------------------------------------------------- robot results
        while robot is not None and not robot.results.empty():
            kind, marker_id, pose = robot.results.get()
            ok, text = read_marker_with_robot(marker_id, pose, fresh_markers)
            if "saved" in text:
                reload_markers(text)
            else:
                print(text)
            message, message_colour = text, (GREEN if ok else RED)

        # ---------------------------------------------------------- clicks
        while clicks:
            pt = clicks.pop(0)
            if not view.ready():
                print(f"Click ignored: only {seen}/4 markers found.")
                continue
            if square_queue:                                   # guided A, B, C check
                name = square_queue.pop(0)
                square_px[name] = pt
                x, y = view.camera_to_robot(*pt)
                off_mm = float(np.hypot(*(squares[name] - (x, y))) * 1000)
                save_row(f"square {name} centre vs Lab 5 position", off_mm, 0)
                print(f"square {name}: centre clicked at x = {x * 1000:.1f} mm, y = {y * 1000:.1f} mm, "
                      f"{off_mm:.1f} mm from its Lab 5 position -> {'OK' if off_mm <= TOLERANCE_MM else 'OVER 5 mm'}")
                if not square_queue:
                    confirming = False
                    d = diagnose(squares, {n: view.camera_to_robot(*square_px[n]) for n in square_px})
                    title, _, lines = diagnosis_box(d, robot_ready)
                    print("\n" + title + "\n  " + "\n  ".join(lines))
                    message = None
                continue
            dist, pair = min((distance_to_segment(pt, marker_px[a], marker_px[b]), (a, b)) for a, b in pairs)
            if dist < LINE_PICK_PX / display_scale[0]:
                selected, typed = pair, ""
                continue
            if len(point_px) == 2:
                points, point_px, free_result = [], [], None
            x, y = view.camera_to_robot(*pt)
            points.append((x, y))
            point_px.append(pt)
            print(f"point: x = {x * 1000:.1f} mm, y = {y * 1000:.1f} mm")
            if len(point_px) == 2:
                selected, typed = "free", ""
                d = np.hypot(points[0][0] - points[1][0], points[0][1] - points[1][1]) * 1000
                print(f"distance between the two points: {d:.1f} mm")
            else:
                selected = None
                message, message_colour = "First point set: click the second point.", BLUE

        # ---------------------------------------------------------- marker lines
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
            # The two diagonals cross in the middle, so their labels go a third of the way along.
            t = 0.5 if (a, b) not in diagonals else (0.3 if diagonals.index((a, b)) == 0 else 0.7)
            mid = pa + t * (pb - pa)
            put(shown, label, (mid[0] - 45, mid[1] - 8), colour, 0.6)

        # ---------------------------------------------------------- two-point measurement
        for p in point_px:
            cv2.circle(shown, (int(p[0]), int(p[1])), 7, BLUE, -1)
        if len(point_px) == 2 and view.ready():
            points = [view.camera_to_robot(*p) for p in point_px]
            d = np.hypot(points[0][0] - points[1][0], points[0][1] - points[1][1]) * 1000
            colour = BLUE if free_result is None else (GREEN if free_result[2] else RED)
            cv2.line(shown, tuple(int(v) for v in point_px[0]), tuple(int(v) for v in point_px[1]), colour, 3,
                     cv2.LINE_AA)
            label = f"{d:.1f} mm" if free_result is None else \
                f"{d:.1f} mm / ruler {free_result[0]:g} ({free_result[1]:+.1f})"
            mid = (np.asarray(point_px[0]) + np.asarray(point_px[1])) / 2
            put(shown, label, (mid[0] + 10, mid[1] - 12), colour, 0.7)

        # ---------------------------------------------------------- A, B, C
        if view.ready():
            for name, xy in squares.items():
                px, py = view.robot_to_camera(*xy)
                cv2.circle(shown, (int(px), int(py)), 9, RED, 2)
                put(shown, name, (px + 10, py - 10), RED, 0.7)
                if name in square_px and name in seen_xy:
                    cx, cy = square_px[name]
                    d = float(np.hypot(*(xy - seen_xy[name]))) * 1000
                    ok = d <= TOLERANCE_MM
                    cv2.drawMarker(shown, (int(cx), int(cy)), GREEN if ok else RED, cv2.MARKER_CROSS, 18, 2)
                    cv2.line(shown, (int(cx), int(cy)), (int(px), int(py)), GREEN if ok else RED, 1, cv2.LINE_AA)
                    put(shown, f"{d:.1f} mm", (cx + 12, cy + 22), GREEN if ok else RED, 0.6)
            # Preview: where the circles go if F is pressed (they land on your clicks, give or take).
            if diag is not None and diag["case"] == "shift":
                Rinv, t = diag["R"].T, diag["t"]
                for name, xy in squares.items():
                    gx, gy = view.robot_to_camera(*(Rinv @ (xy - t)))
                    cv2.circle(shown, (int(gx), int(gy)), 11, GREEN, 2, cv2.LINE_AA)

        # ---------------------------------------------------------- live gripper position
        if robot_ready and robot.pose is not None and view.ready():
            gx, gy = view.robot_to_camera(*robot.pose[:2])
            cv2.circle(shown, (int(gx), int(gy)), 7, CYAN, -1)
            cv2.circle(shown, (int(gx), int(gy)), 13, CYAN, 2)
            nearest, d_mm = min(((i, float(np.hypot(*(markers[i] - robot.pose[:2]))) * 1000)
                                 for i in cfg.MARKER_IDS), key=lambda v: v[1])
            label = f"gripper (robot reading)  {d_mm:.1f} mm from marker {nearest}" if d_mm < 40 else \
                "gripper (robot reading)"
            put(shown, label, (gx + 16, gy + 5), CYAN, 0.55)

        # ---------------------------------------------------------- instruction box
        robot_line = None
        if robot is not None:
            if robot.state == "ready":
                read = "  ".join(f"{i}:{'done' if i in fresh_markers else '-'}" for i in cfg.MARKER_IDS)
                robot_line = (f"Robot: learning mode {'ON' if robot.learning else 'OFF'} (L).  0-3 = read that "
                              f"marker, gripper straight down 5 mm above it.  Read here: {read}")
            else:
                robot_line = f"Robot: {robot.status}"
        if not view.ready():
            title, colour = f"Markers not found ({seen}/4) - clicks are ignored", RED
            lines = ["Check the focus and that nothing hides a marker, or run step1_tune_camera.py again.",
                     "Q = quit"]
        elif square_queue:
            name = square_queue[0]
            done = len(squares) - len(square_queue)
            title = (f"Corrected - now confirm: click the centre of square {name} ({done + 1}/{len(squares)})"
                     if confirming else f"Square check {done + 1}/{len(squares)}: click the centre of square {name}")
            colour = GREEN if confirming else BLUE
            lines = [f"Click where square {name} really is: the centre of the hole.",
                     f"Red circle {name} = your Lab 5 pick-up position for {name}, level 1 (step 1.1), drawn on the",
                     "picture with the marker positions from step 1.4. Each should sit in the centre of its square.",
                     "Click the picture first.   Backspace = redo the previous square   S = skip   Q = quit"]
        elif selected is not None:
            what = "the two points" if selected == "free" else f"marker {selected[0]} to {selected[1]}"
            title, colour = f"Type your ruler value for {what}, then press Enter", ORANGE
            lines = [f"ruler value: {typed}_ mm",
                     message if message_colour == RED else "Backspace = delete     Esc = cancel"]
        elif diag is not None:
            title, colour, lines = diagnosis_box(diag, robot_ready)
            if message and message_colour in (GREEN, RED):
                lines = [message] + lines
        else:
            title, colour = "Click a line between two markers, or click two points", BLUE
            lines = [message or "Then type the distance you measure with a ruler and press Enter.",
                     ("Red circles A, B, C should sit in the centre of each square.  R = square check" if squares else
                      "To see A, B, C: enter your Lab 5 positions with step1_lab5_positions.py."),
                     "Click the picture first.    K = connect robot    C = clear    Q = quit when done"]
        if robot_line and selected is None:
            lines = lines + [robot_line]
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
                what = "two points"
                d = float(np.hypot(points[0][0] - points[1][0], points[0][1] - points[1][1]) * 1000)
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
        elif key in (8, 127) and square_queue and square_px:          # redo the previous square
            last = [n for n in squares if n in square_px and n not in square_queue][-1]
            square_px.pop(last)
            square_queue.insert(0, last)
        elif char in "0123" and robot_ready and selected is None:
            robot.send(("read", int(char)))
            message, message_colour = f"Reading marker {char}...", BLUE
        elif char in "lL" and robot_ready:
            robot.send("learning")
        elif char in "kK" and robot is None:
            robot = RobotLink()
            robot.start()
        elif char in "kK" and robot is not None and robot.state == "failed":
            robot = RobotLink()                                    # try again
            robot.start()
        elif char in "fF" and diag is not None and diag["case"] == "shift":
            stamp = save_correction(diag["R"], diag["t"], diag)
            reload_markers(f"Marker positions corrected by a turn of {diag['turn_deg']:+.2f} deg and a shift of "
                           f"{diag['shift_mm'][0]:+.1f}, {diag['shift_mm'][1]:+.1f} mm (x, y). "
                           f"Old file kept as calibration/markers_backup_{stamp}.json")
            square_queue[:] = list(squares)
            square_px.clear()
            confirming = True
            message = None
        elif char in "rR" and squares:
            square_queue[:] = list(squares)
            square_px.clear()
            confirming, message = False, None
        elif char in "sS" and selected is None and squares:
            if square_queue:                                   # skip the square check
                square_queue.clear()
            else:                                              # repeat it
                square_queue[:] = list(squares)
                square_px.clear()
            message, message_colour = None, BLUE
        elif char in "cC":
            points, point_px, free_result, selected, typed = [], [], None, None, ""
            pair_result.clear()
            message, message_colour = None, BLUE
        elif char in "qQ":
            break
    if robot is not None:
        print("Closing the robot connection (sleep pose)...")
        robot.close()
    cam.release()
    cv2.destroyAllWindows()
    return last_shown


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as e:
        raise SystemExit(f"\nMissing: {e}\nCheck the function named in the message.")