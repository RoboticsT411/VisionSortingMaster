"""
STEP 1c - Check the camera-to-robot calibration against a ruler.

    python step1_check_calibration.py
    python step1_check_calibration.py --source dataset_raw     (saved images)

The box at the top of the picture always says what to do next.
FIRST CLICK ON THE PICTURE so it receives your key presses.

1. Marker distances. Lines join the marker centres, each labelled with its
   length in mm (from calibration/markers.json). Measure the same distance on
   the rig with a ruler, marker centre to marker centre. Click the line, type
   your ruler value and press Enter. The label turns green (within 5 mm) or red.
2. Any two points. Click two points away from the lines, for example the left
   and right platform edges (260 mm). The distance appears on the picture.
   Type your ruler value and press Enter to compare, or click again to start
   a new measurement.
3. A, B and C. Click the centre of each square: the box says how far it is from your
   Lab 5 position, and what to redo if it is more than 5 mm. Red circles show squares A, B and C from your Lab 5 positions
   (entered with step1_lab5_positions.py, pick level 1): each should sit in
   the centre of its square.

All positions are robot coordinates (origin at the robot base, x forward,
y to the robot's left). A distance between two points does not depend on the
origin, so it can be compared directly with a ruler.

Square check first: the box asks you to click the centre of A, then B, then C, and
then says whether anything must be redone (S skips or repeats it).

Keys:  0-9 and .  type a ruler value     Enter = compare     Backspace = delete
       Esc = cancel     c = clear     q = quit
Every comparison is saved to results/calibration_checks.csv.
Target: within 5 mm. Bigger errors usually mean a marker position was measured
wrongly (repeat step1_measure_markers.py) or a marker is not flat.
"""
import argparse
import csv
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
LINE_PICK_PX = 12           # a click this close to a marker line selects it
GREEN, RED, BLUE, ORANGE, GREY = (0, 170, 0), (0, 0, 230), (200, 90, 0), (0, 140, 255), (60, 60, 60)

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


def square_summary(done):
    """One line per result and what to do next, separated by ' | '."""
    parts = [f"{n} {d:.1f} mm {'OK' if d <= TOLERANCE_MM else 'OFF'}" for n, (d, _) in sorted(done.items())]
    off = [n for n, (d, _) in sorted(done.items()) if d > TOLERANCE_MM]
    if not off:
        return "Squares: " + ", ".join(parts) + " | All within 5 mm: the camera and your Lab 5 positions agree."
    names = ", ".join(off)
    return ("Squares: " + ", ".join(parts) + f" | Square{'s' if len(off) > 1 else ''} {names} more than 5 mm off. "
            f"In step1_lab5_positions.py test pick {names} at level 1:"
            f" | gripper off-centre on the square -> redo step 1.1 for {names} only (read the true centre)"
            f" | gripper centred on the square -> redo step 1.4 for the markers nearest {names}")


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
    points, point_px = [], []        # two-point measurement: robot (x, y) and image pixels
    free_result = None               # (ruler, diff, ok)
    selected, typed = None, ""       # a pair, "free", or None; the ruler value being typed
    message, message_colour = None, BLUE
    square_queue = list(squares)     # guided check first: click the centre of A, then B, then C
    square_done = {}                 # name -> (distance mm, click pixel)

    while True:
        frame = cam.read()
        centres, corners, ids = detector.detect(frame)
        view.update(centres)
        shown = frame.copy()
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(shown, corners, ids)
        seen = sum(1 for i in cfg.MARKER_IDS if i in centres)

        # Pixel position of each marker centre: as detected now, or from the last good mapping.
        marker_px = {}
        if view.ready():
            for i in cfg.MARKER_IDS:
                marker_px[i] = np.asarray(centres[i] if i in centres else view.robot_to_camera(*markers[i]), float)

        # ---------------------------------------------------------- clicks
        while clicks:
            pt = clicks.pop(0)
            if not view.ready():
                print(f"Click ignored: only {seen}/4 markers found.")
                continue
            if square_queue:                                   # guided A, B, C check
                name = square_queue.pop(0)
                x, y = view.camera_to_robot(*pt)
                off_mm = float(np.hypot(*(squares[name] - (x, y))) * 1000)
                square_done[name] = (off_mm, pt)
                save_row(f"square {name} centre vs Lab 5 position", off_mm, 0)
                print(f"square {name}: centre clicked at x = {x * 1000:.1f} mm, y = {y * 1000:.1f} mm, "
                      f"{off_mm:.1f} mm from its Lab 5 position -> {'OK' if off_mm <= TOLERANCE_MM else 'OVER 5 mm'}")
                if not square_queue:
                    message, message_colour = square_summary(square_done), None
                    print("\n" + message.replace(" | ", "\n"))
                continue
            dist, pair = min((distance_to_segment(pt, marker_px[a], marker_px[b]), (a, b)) for a, b in pairs)
            if dist < LINE_PICK_PX / display_scale[0]:
                selected, typed = pair, ""
                continue
            if len(points) == 2:
                points, point_px, free_result = [], [], None
            x, y = view.camera_to_robot(*pt)
            points.append((x, y))
            point_px.append(pt)
            text = f"point: x = {x * 1000:.1f} mm, y = {y * 1000:.1f} mm"
            if squares:
                name, xy = min(squares.items(), key=lambda kv: np.hypot(*(kv[1] - (x, y))))
                text += f"   ({np.hypot(*(xy - (x, y))) * 1000:.1f} mm from the Lab 5 centre of square {name})"
            print(text)
            if len(points) == 2:
                d = np.hypot(points[0][0] - points[1][0], points[0][1] - points[1][1]) * 1000
                selected, typed = "free", ""
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
        if len(point_px) == 2:
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
            abc = RED
            for name, xy in squares.items():
                px, py = view.robot_to_camera(*xy)
                cv2.circle(shown, (int(px), int(py)), 9, abc, 2)
                put(shown, name, (px + 10, py - 10), abc, 0.7)
                if name in square_done:
                    d, (cx, cy) = square_done[name]
                    ok = d <= TOLERANCE_MM
                    cv2.drawMarker(shown, (int(cx), int(cy)), GREEN if ok else RED, cv2.MARKER_CROSS, 18, 2)
                    cv2.line(shown, (int(cx), int(cy)), (int(px), int(py)), GREEN if ok else RED, 1, cv2.LINE_AA)
                    put(shown, f"{d:.1f} mm", (cx + 12, cy + 22), GREEN if ok else RED, 0.6)

        # ---------------------------------------------------------- instruction box
        if not view.ready():
            title, colour = f"Markers not found ({seen}/4) - clicks are ignored", RED
            lines = ["Check the focus and that nothing hides a marker, or run step1_tune_camera.py again.",
                     "Q = quit"]
        elif square_queue:
            name = square_queue[0]
            done = len(squares) - len(square_queue)
            title, colour = f"Square check {done + 1}/{len(squares)}: click the centre of square {name}", BLUE
            lines = [f"Click where square {name} really is. The red circle {name} is where your Lab 5 position "
                     f"puts it.",
                     "Each square is compared with your Lab 5 position: within 5 mm is OK.",
                     "Click the picture first.    S = skip this check    Q = quit"]
        elif message and message_colour is None:                      # the square summary
            parts = message.split(" | ")
            bad = "OFF" in parts[0]
            title, colour = (("Some squares are off - what to do next" if bad else "Square check passed"),
                             RED if bad else GREEN)
            lines = parts + ["Click a line between two markers, or two points, to continue.   "
                             "S = repeat the square check   Q = quit"]
        elif selected is not None:
            what = "the two points" if selected == "free" else f"marker {selected[0]} to {selected[1]}"
            title, colour = f"Type your ruler value for {what}, then press Enter", ORANGE
            lines = [f"ruler value: {typed}_ mm",
                     message if message_colour == RED else "Backspace = delete     Esc = cancel"]
        else:
            title, colour = "Click a line between two markers, or click two points", BLUE
            lines = [message or "Then type the distance you measure with a ruler and press Enter.",
                     ("Red circles A, B, C should sit in the centre of each square." if squares else
                      "To see A, B, C: enter your Lab 5 positions with step1_lab5_positions.py."),
                     "Click the picture first.    C = clear    Q = quit when done"]
        use_message_colour = (message and selected is None and view.ready() and not square_queue
                              and message_colour not in (None, BLUE))
        shown = instruction_box(with_progress(shown, "1.5"), title, lines,
                                message_colour if use_message_colour else colour)

        panel_offset[0] = shown.shape[0] - frame.shape[0]
        picture_width[0] = frame.shape[1]
        display_scale[0] = min(1.0, 1280 / shown.shape[1])
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
        elif key in (ord("s"), ord("S")) and selected is None and squares:
            if square_queue:                                   # skip the square check
                square_queue.clear()
            else:                                              # repeat it
                square_queue[:] = list(squares)
                square_done.clear()
            message, message_colour = None, BLUE
        elif key in (ord("c"), ord("C")):
            points, point_px, free_result, selected, typed = [], [], None, None, ""
            pair_result.clear()
            message, message_colour = None, BLUE
        elif key in (ord("q"), ord("Q")):
            break
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as e:
        raise SystemExit(f"\nMissing: {e}\nCheck the function named in the message.")
