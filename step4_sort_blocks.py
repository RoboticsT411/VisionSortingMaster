"""
STEP 4 - Sort the cubes with the robot.

KEY PARTS: two functions in this file (marked "KEY PART"):
  choose()         - which cube to sort next
  pick_and_drop()  - the robot moves for one cube, based on the Lab 5 program
Around them: the camera, the loop, the safety checks and the log.

Run it (with the robot, confirming every move):
    python step4_sort_blocks.py

Options:
    --stacks 0,0,0   cubes already on A,B,C at the start (default none)
    --auto           do not ask before each move (only after supervised runs work)
    --measure-offset after each pick, type how far (mm) the gripper was from the cube centre:
                     compare with and without USE_PARALLAX_CORRECTION (config.py)
    --voice          OPTIONAL EXTRA: voice commands (see voice_commands.py)

Each cycle: arm to the observe pose (sleep pose, learning mode on, out of the camera's way) ->
fresh picture -> choose() -> your pick_and_drop() is first run on a RECORDER that checks every
pose and the order of the moves -> only if everything passes does the real robot repeat it ->
back to the sleep pose, learning mode on.
Every attempt is logged in results/trials.csv for your success rate.

GUIDED PICKS (the first GUIDED_PICKS of every run, config.py): the robot stops 20 mm above the
cube. Centre the gripper, one thing at a time, ENTER after each: 1 move it over the middle of the cube (D/A = your
right/left, W/S = away from you/towards you), 2 turn it to line up (LEFT/RIGHT arrows), 3 it goes
down and you check the height (UP/DOWN arrows); ENTER closes the gripper. Esc = stop. 4 it stops
5 mm above the drop position: W/A/S/D, ENTER = let go; a change is saved to the Lab 5 DROP position
of that square and level (a backup of the old file is kept; pick positions are not changed).
The pick correction is saved after every guided pick (running average). The average of these corrections is saved in
calibration/pick_correction.json and applied to every later pick (and to the next runs). Every
correction is logged in results/pick_corrections.csv.

The box at the top of the camera window says what the program is doing and
what to answer. You answer in the PyCharm Run window (click in it first).

Stopping: answer q at a prompt, or press Ctrl+C. The program then sends the
robot to its sleep pose. If the robot is about to hit something, switch off
its power supply.
"""
import argparse
import csv
import json
import math
import shutil
import threading
import time
from datetime import datetime

import cv2

import config as cfg
from camera import Camera
from colour_model import ColourModel
from robot import (DryRunRobot, NiryoRobot, PlanRecorder, _rotation, check_pose, load_lab5_poses, read_lab5_file,
                   save_lab5_file, turned_rpy)
from progress import print_progress, with_progress
from ui import BLUE, GREEN, ORANGE, RED, instruction_box
from vision import VisionSystem, show


# =====================================================================================
# KEY PART 1 of 2: which cube to sort next
# =====================================================================================
def choose(detections, stacks, wanted=None):
    """Choose the cube the robot should sort next.

    detections : cubes found by the camera; each d has d.label, d.confidence (0 to 1),
                 d.x, d.y (robot coordinates, metres) and d.angle_deg
    stacks     : cubes already on each square, e.g. {"A": 1, "B": 0, "C": 2}
    wanted     : voice extra only: a colour to sort, or None for all

    Return (cube, reasons): the detection to sort or None, and a list of short texts saying
    why the other cubes were skipped (shown on screen).
    Requirements: never choose a colour that is not in cfg.COLOUR_TO_SQUARE, a prediction below
    cfg.CONFIDENCE_THRESHOLD, or a cube whose square already holds cfg.MAX_STACK cubes.
    Explain in your report how you choose among the cubes that are left.
    """
    reasons, usable = [], []
    for d in detections:
        if d.label not in cfg.COLOUR_TO_SQUARE:
            reasons.append(f"{d.label} not sorted")
        elif wanted is not None and d.label != wanted:
            reasons.append(f"{d.label} not asked for")
        elif d.confidence < cfg.CONFIDENCE_THRESHOLD:
            reasons.append(f"{d.label} {d.confidence:.2f} below threshold")
        elif stacks[cfg.COLOUR_TO_SQUARE[d.label]] >= cfg.MAX_STACK:
            reasons.append(f"{d.label}: square {cfg.COLOUR_TO_SQUARE[d.label]} full")
        else:
            usable.append(d)
    usable.sort(key=lambda d: d.confidence, reverse=True)
    return (usable[0] if usable else None), reasons


# =====================================================================================
# KEY PART 2 of 2: the robot moves for one cube, based on the Lab 5 program
# =====================================================================================
def pick_and_drop(robot, cube, poses, square, level):
    """Pick the cube up from the zone and drop it on its square at the given level.

    Write it like your Lab 5 program. robot has the same commands:
        robot.move_pose(x, y, z, roll, pitch, yaw), robot.open_gripper(), robot.close_gripper()
    cube.x, cube.y    where the camera found the cube (robot coordinates, metres)
    poses             your Lab 5 values: poses["rpy"], poses["z_safe"], poses["pick"][square][level],
                      poses["drop"][square][level] (each [x, y, z])
    square, level     where this cube goes, e.g. "A" and 2 (level 1 = first cube on the square)

    Requirements (checked on a recorder BEFORE the real robot moves): the gripper is open before
    it goes down, closes at the cube and opens at the drop position for this square and level;
    every sideways move is at the safe height; the sequence ends at the safe height.
    Decide and justify the pick height for a cube in the zone (cfg.ZONE_PICK_OFFSET_MM may help).
    """
    roll, pitch, yaw = poses["rpy"]
    z_safe = poses["z_safe"]
    p_roll, p_pitch, p_yaw = roll, pitch, yaw                 # gripper orientation at the cube
    if cfg.USE_ORIENTATION:                                   # Level 2
        angle = max(-cfg.MAX_YAW_CORRECTION_DEG, min(cfg.MAX_YAW_CORRECTION_DEG, cube.angle_deg))
        p_roll, p_pitch, p_yaw = turned_rpy(poses["rpy"], angle)
    z_pick = poses["pick"]["A"][1][2] + cfg.ZONE_PICK_OFFSET_MM / 1000
    dx, dy, dz = poses["drop"][square][level]

    robot.open_gripper()
    robot.move_pose(cube.x, cube.y, z_safe, p_roll, p_pitch, p_yaw)    # above the cube, turned to it
    robot.move_pose(cube.x, cube.y, z_pick, p_roll, p_pitch, p_yaw)    # down
    robot.close_gripper()
    robot.move_pose(cube.x, cube.y, z_safe, p_roll, p_pitch, p_yaw)    # up
    robot.move_pose(dx, dy, z_safe, roll, pitch, yaw)                  # above the square
    robot.move_pose(dx, dy, dz, roll, pitch, yaw)                      # down to the drop level
    robot.open_gripper()
    robot.move_pose(dx, dy, z_safe, roll, pitch, yaw)                  # up


# =====================================================================================
# Given: checks on the recorded moves, so mistakes are found before the robot moves
# =====================================================================================
def review_moves(steps, cube, poses, square, level, tol=0.005):
    """Problems with the order of the moves, as clues. An empty list means the sequence is fine."""
    problems = []
    moves = [s[1] for s in steps if s[0] == "move"]
    if not moves:
        return ["pick_and_drop() did not call robot.move_pose() at all"]
    z_safe = poses["z_safe"]
    drop = poses["drop"][square][level]
    near = lambda p, x, y: math.hypot(p[0] - x, p[1] - y) <= tol

    if moves[0][2] < z_safe - tol:
        problems.append(f"the first move goes to z={moves[0][2]:.3f}: go to the safe height z={z_safe:.3f} first")
    for a, b in zip(moves, moves[1:]):
        sideways = math.hypot(a[0] - b[0], a[1] - b[1]) > tol
        if sideways and min(a[2], b[2]) < z_safe - tol:
            problems.append(f"a sideways move at z={min(a[2], b[2]):.3f} from ({a[0]:.3f}, {a[1]:.3f}) to "
                            f"({b[0]:.3f}, {b[1]:.3f}): travel at the safe height, then go straight down")
            break
    if moves[-1][2] < z_safe - tol:
        problems.append("the last move does not go back up to the safe height")

    kinds = [s[0] for s in steps]
    if "close" not in kinds:
        problems.append("the gripper never closes (robot.close_gripper())")
        return problems
    first_close = kinds.index("close")
    if "open" not in kinds[:first_close]:
        problems.append("open the gripper before going down to the cube")
    if "open" not in kinds[first_close:]:
        problems.append("the gripper never opens after picking the cube (robot.open_gripper())")

    last_pose = None
    for i, step in enumerate(steps):
        if step[0] == "move":
            last_pose = step[1]
        elif step[0] == "close" and last_pose is not None:
            if not near(last_pose, cube.x, cube.y):
                problems.append(f"the gripper closes at ({last_pose[0]:.3f}, {last_pose[1]:.3f}), but the cube is "
                                f"at ({cube.x:.3f}, {cube.y:.3f})")
            elif last_pose[2] > z_safe - tol:
                problems.append("the gripper closes at the safe height: go down to the cube first")
        elif step[0] == "open" and i > first_close and last_pose is not None:
            if not near(last_pose, drop[0], drop[1]) or abs(last_pose[2] - drop[2]) > tol:
                problems.append(f"the gripper opens at ({last_pose[0]:.3f}, {last_pose[1]:.3f}, "
                                f"{last_pose[2]:.3f}), not at the drop position for {square} level {level}: "
                                f"({drop[0]:.3f}, {drop[1]:.3f}, {drop[2]:.3f})")
    return problems


# =====================================================================================
# Given: the sorting loop
# =====================================================================================
CURRENT_STEP = ["4:Robot trials"]
GUIDED_MAX_DEG = getattr(cfg, "GUIDED_MAX_DEG", 30)    # in case config.py is older than this file
GUIDED_DROP_HOVER_MM = getattr(cfg, "GUIDED_DROP_HOVER_MM", 5)
VERSION = "26 Nov 2026 17:45 (robot stops handled: back to step 1, recorded in the trials)"


def show_status(cam_img, title, lines, colour, refresh=True):
    """cam_img is a picture, or a function that returns the live camera picture.
    refresh=False when the caller reads the keys itself (so no key press is lost here)."""
    if callable(cam_img):
        cam_img = cam_img()
    show("camera", instruction_box(with_progress(cam_img, CURRENT_STEP[0]), title, lines, colour))
    if refresh:
        cv2.waitKey(1)


KEY_NAMES = {"\r": "Enter", " ": "Space"}


def ask(prompt, auto, cam_img=None, title=None, lines=None, colour=BLUE, keys="\rsq", keys_line=None):
    """Wait for a key pressed IN THE CAMERA WINDOW, keeping the window alive (no "Not Responding").
    Returns "" for Enter, otherwise the letter. keys_line replaces the list of keys at the bottom."""
    if auto:
        return ""
    print(prompt + " (press the key in the camera window)")
    while True:
        if cam_img is not None:
            show_status(cam_img, title or prompt, (lines or []) + [
                keys_line or "(Click on the camera window first, then press the key.)"], colour, refresh=False)
        key = cv2.waitKey(50) & 0xFF
        if key == 255:
            continue
        ch = "\r" if key in (10, 13) else chr(key).lower()
        if ch in keys:
            print("  ->", KEY_NAMES.get(ch, ch))
            return "" if ch == "\r" else ch


def ask_number(prompt, cam_img, title):
    """Type a number in the camera window (digits, minus, point; Backspace; Enter = done, empty = skip)."""
    typed = ""
    print(prompt + " (type it in the camera window)")
    while True:
        show_status(cam_img, title, [prompt, f"Typed: {typed or '(nothing)'}_",
                                     "Click on the camera window. Backspace = delete, Enter = done (empty = skip)"],
                    BLUE, refresh=False)
        key = cv2.waitKey(50) & 0xFF
        if key in (10, 13):
            print("  ->", typed or "skipped")
            return typed
        if key == 8:
            typed = typed[:-1]
        elif key != 255 and chr(key) in "0123456789.-":
            typed += chr(key)


class _Stoppable:
    """Passes commands to the robot, but refuses the next one once stop is set (Ctrl+C)."""

    def __init__(self, robot, stop):
        self._robot, self._stop = robot, stop

    def __getattr__(self, name):
        command = getattr(self._robot, name)

        def call(*args, **kwargs):
            if self._stop.is_set():
                raise KeyboardInterrupt
            return command(*args, **kwargs)
        return call


def while_busy(action, robot, cam_img, title, lines):
    """Run action(robot) in the background (robot moves take seconds) while the camera window stays
    alive. Ctrl+C lets the current move finish, then stops before the next one."""
    box, stop = {}, threading.Event()

    def run():
        try:
            action(_Stoppable(robot, stop))
        except BaseException as e:            # passed back to the main program below
            box["error"] = e

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    try:
        while worker.is_alive():
            show_status(cam_img, title, lines, ORANGE, refresh=False)
            cv2.waitKey(50)
    except KeyboardInterrupt:
        stop.set()
        print("\nCtrl+C: stopping after the current move...")
        worker.join()
        raise
    if "error" in box:
        raise box["error"]


# =====================================================================================
# Given: resting the arm, guided picks and the learned pick correction
# =====================================================================================
def rest(r):
    """Sleep pose, then learning mode on: the arm is out of the camera's way."""
    r.move_joints(cfg.SLEEP_JOINTS)
    r.learning_mode(True)


def back_off(r, poses):
    """After the robot stopped itself: open the gripper, go straight up to the safe height, then rest."""
    r.learning_mode(False)
    r.open_gripper()
    p = list(r.get_pose())
    r.move_pose(p[:2] + [max(p[2], poses["z_safe"])] + p[3:])
    rest(r)


def run_steps(steps, poses, then_rest=False):
    """An action for while_busy(): send these recorded steps to the robot (motors on first)."""
    plan = PlanRecorder(poses)
    plan.steps = list(steps)

    def action(r):
        r.learning_mode(False)
        plan.replay(r)
        if then_rest:
            rest(r)
    return action


def load_pick_correction():
    """{"dx_mm", "dy_mm", "dz_mm", ...} saved by earlier guided picks, or zeros."""
    zero = {"dx_mm": 0.0, "dy_mm": 0.0, "dz_mm": 0.0, "turn_deg": 0.0, "picks": 0}
    try:
        data = json.loads(cfg.PICK_CORRECTION_FILE.read_text())
    except (FileNotFoundError, ValueError):
        return zero
    if data.get("parallax_corrected") != cfg.USE_PARALLAX_CORRECTION:
        print("Pick correction not used: it was measured with USE_PARALLAX_CORRECTION = "
              f"{data.get('parallax_corrected')}. The guided picks measure it again.")
        return zero
    return {**zero, **data}


def save_pick_correction(old, corrections, angles):
    """Save the average TOTAL correction (what was applied + what you changed) of this run's guided
    picks, combined with the one saved by earlier runs (counted as up to 5 picks).
    corrections: total (dx, dy, dz mm, turn deg) per guided pick; angles: the cube angle the camera saw."""
    n = len(corrections)
    w = min(old.get("picks", 0), 5)
    prev = [old["dx_mm"], old["dy_mm"], old["dz_mm"], old["turn_deg"]]
    mean = [(w * prev[i] + sum(c[i] for c in corrections)) / (w + n) for i in range(4)]
    # Does the turn correction depend on the cube angle? A slope near -2 means the gripper turns the
    # wrong way (YAW_SIGN); near 0 means a constant offset, which the average fixes.
    slope = None
    if n >= 3 and max(angles) - min(angles) >= 20:
        ma = sum(angles) / n
        slope = (sum((a - ma) * (c[3] - mean[3]) for a, c in zip(angles, corrections))
                 / sum((a - ma) ** 2 for a in angles))
    spread = max(math.hypot(c[0] - mean[0], c[1] - mean[1]) for c in corrections)
    data = {"time": datetime.now().isoformat(timespec="seconds"),
            "dx_mm": round(mean[0], 1), "dy_mm": round(mean[1], 1),
            "dz_mm": round(mean[2], 1), "turn_deg": round(mean[3], 1),
            "picks": w + n, "turn_vs_angle_slope": None if slope is None else round(slope, 2),
            "spread_mm": round(spread, 1), "parallax_corrected": cfg.USE_PARALLAX_CORRECTION}
    cfg.CALIB_DIR.mkdir(exist_ok=True)
    cfg.PICK_CORRECTION_FILE.write_text(json.dumps(data, indent=2))
    return data, spread, slope


ARROWS = {2424832: "left", 2555904: "right", 2490368: "up", 2621440: "down",      # Windows
          65361: "left", 65363: "right", 65362: "up", 65364: "down",              # Linux
          63234: "left", 63235: "right", 63232: "up", 63233: "down"}              # macOS


def ask_step(picture, title, lines, allowed, hint):
    """One guided step: only the keys in allowed work ("left", "right", "up", "down", "enter", "esc"
    or letters). Any other key shows the hint. Returns the key's name."""
    note = ""
    print(title)
    while True:
        show_status(picture, title, lines + (note.split("\n") if note else []), ORANGE if note else BLUE,
                    refresh=False)
        key = cv2.waitKeyEx(50)
        if key == -1:
            continue
        if key in ARROWS:
            name = ARROWS[key]
        elif key & 0xFF in (10, 13):
            name = "enter"
        elif key & 0xFF == 27:
            name = "esc"
        else:
            name = chr(key & 0xFF).lower()
        if name in allowed:
            return name
        note = "- That key does nothing here. " + hint


def guided_pick(recorder, robot, cube, poses, picture, target, which, square, level):
    """Replay the recorded moves, but stop above the cube and at the grip height for corrections,
    one thing at a time: 1 move sideways, 2 turn, 3 height; then 4 above the drop position (sideways).
    which: e.g. "Guided pick 2 of 5". A drop correction is saved to the Lab 5 positions of that
    square and level (drop only; a backup of the old file is kept).
    Returns (dx, dy, dz in mm, turn in degrees), "stop" if the student pressed Esc, or None if the
    moves have no descent to the cube followed by close_gripper() (then the caller replays normally)."""
    steps = with_drop_turn(recorder.steps, square, level)
    k = next((i for i, s_ in enumerate(steps) if s_[0] == "move" and s_[1][2] < poses["z_safe"] - 0.001
              and math.hypot(s_[1][0] - cube.x, s_[1][1] - cube.y) < 0.002), None)
    if k is None or k + 1 >= len(steps) or steps[k + 1][0] != "close":
        return None
    down = list(steps[k][1])
    xyz = [down[0], down[1], down[2] + cfg.GUIDED_HOVER_MM / 1000]
    turn = 0.0                                    # degrees, + = anticlockwise seen from above (as cube.angle_deg)
    busy = ["- Keep clear of the robot.", "- Ctrl+C in the Run window = stop"]

    def pose(p_xyz=None, p_turn=None):
        return list(p_xyz or xyz) + list(turned_rpy(down[3:], turn if p_turn is None else p_turn))

    def go(new_xyz, new_turn, what):
        """Move there if it is allowed; returns "" or the reason it did not move."""
        nonlocal xyz, turn
        gap_z = abs(new_xyz[2] - down[2]) if what == "height" else 0.0   # the hover height is not a correction
        if max(abs(new_xyz[0] - down[0]), abs(new_xyz[1] - down[1]), gap_z) * 1000 > cfg.GUIDED_MAX_MM:
            return f"- Not moved: more than {cfg.GUIDED_MAX_MM} mm from where the camera saw the cube."
        if abs(new_turn) > GUIDED_MAX_DEG:
            return f"- Not turned: more than {GUIDED_MAX_DEG} degrees."
        try:
            check_pose(pose(new_xyz, new_turn), poses)
        except ValueError as e:
            return f"- Not moved: {e}"
        try:
            while_busy(run_steps([("move", pose(new_xyz, new_turn))], poses), robot, picture, f"{which}: moving",
                       busy)
        except RuntimeError as e:                 # e.g. the gripper pressed on the cube or the platform
            print(f"  Move refused by the robot: {e}")
            return MOVE_REFUSED
        xyz, turn = list(new_xyz), new_turn
        return ""

    while_busy(run_steps(steps[:k] + [("move", pose())], poses), robot, picture,
               f"{which}: moving above the cube", busy)
    stop_line = "- Esc = stop: nothing is picked, the robot goes back to rest."
    stages = ["move", "turn", "height"]
    stage, note = 0, ""
    while True:
        dx, dy, dz = ((xyz[i] - down[i]) * 1000 for i in range(3))
        names = ["1 Keys: position", "2 Arrows: turn", "3 Arrows: height", "4 Keys: drop position"]
        route = "- Steps:  " + "  ->  ".join(f"[{n}]" if i == stage else n for i, n in enumerate(names)) + \
                "      ENTER = next step"
        if stages[stage] == "move":
            title = f"{which} - step 1 of 4: move the gripper over the MIDDLE of the cube"
            lines = ["- Stand in front of the robot, at least 30 cm away. Look from the front, then from the side.",
                     "- Press D to move it to your RIGHT, A to your LEFT (1 mm each press).",
                     "- Press W to move it AWAY from you, S TOWARDS you (1 mm each press).",
                     f"- Moved so far: {dx:+.0f} mm towards/away, {dy:+.0f} mm right/left.",
                     "- Over the middle? Press ENTER to go to step 2 (turning), also if it was right already.",
                     "- Guided picks teach the robot its corrections; they are NOT counted in your success rate.",
                     stop_line]
            allowed, hint = "wasd", "Use W, A, S, D to move, then ENTER."
        elif stages[stage] == "turn":
            title = f"{which} - step 2 of 4: turn the gripper to line up with the cube"
            lines = ["- Look from at least 30 cm: the fingers should be parallel to two sides of the cube.",
                     "- Press the LEFT arrow to turn anticlockwise, the RIGHT arrow to turn clockwise (2 degrees each).",
                     f"- Turned so far: {turn:+.0f} degrees.",
                     "- Lined up? Press ENTER: the gripper goes down to the grip height (it does not close yet).",
                     stop_line]
            allowed, hint = ("left", "right"), "Use the LEFT and RIGHT arrows to turn, then ENTER."
        else:
            title = f"{which} - step 3 of 4: check the height before it grips"
            lines = ["- The fingers should be around the middle of the cube: not pressing on its top, not touching "
                     "the platform.",
                     "- Press the UP arrow to go up, the DOWN arrow to go down (1 mm each press).",
                     f"- Height changed so far: {dz:+.0f} mm.",
                     "- Right height? Press ENTER: the gripper closes and carries the cube to the square, where it "
                     "stops again (step 4).", stop_line]
            allowed, hint = ("up", "down"), "Use the UP and DOWN arrows, then ENTER."
        key = ask_step(picture, title, [route] + lines + (note.split("\n") if note else []), set(allowed) | {"enter", "esc"},
                       hint)
        note = ""
        if key == "esc":
            while_busy(run_steps([("move", xyz[:2] + [poses["z_safe"]] + pose()[3:])], poses, then_rest=True), robot,
                       picture, "Stopped: moving up, then to the sleep pose", busy)
            return "stop"
        if key == "enter":
            if stages[stage] == "height":
                break
            stage += 1
            if stages[stage] == "height":                       # down to the grip height
                note = go([xyz[0], xyz[1], down[2]], turn, "height")
                if note == MOVE_REFUSED:                         # a finger landed on the cube: back up, step 1 again
                    try:
                        while_busy(run_steps([("move", pose())], poses), robot, picture, f"{which}: moving back up",
                                   busy)
                    except RuntimeError as e:
                        print(f"  Move refused by the robot: {e}")
                    stage = 0
                    note = ("- The robot STOPPED on the way down: a finger landed on the cube. It went back up.\n"
                            "  Move it over the MIDDLE of the cube again (W/A/S/D), then ENTER.")
            continue
        mm, deg = 0.001, 2.0                  # standing in front of the robot: towards you = +x, your right = +y
        new = list(xyz)
        if key == "s":
            new[0] += mm
        elif key == "w":
            new[0] -= mm
        elif key == "d":
            new[1] += mm
        elif key == "a":
            new[1] -= mm
        elif key == "up":
            new[2] += mm
        elif key == "down":
            new[2] -= mm
        new_turn = turn + (deg if key == "left" else -deg if key == "right" else 0.0)
        note = go(new, new_turn, stages[stage])
    corr = [xyz[i] - down[i] for i in range(3)]
    rest_steps = [("close",)]
    for s_ in steps[k + 2:]:                   # the moves at the cube (e.g. back up) get the same correction
        if s_[0] == "move" and math.hypot(s_[1][0] - cube.x, s_[1][1] - cube.y) < 0.002:
            s_ = ("move", [s_[1][0] + corr[0], s_[1][1] + corr[1], s_[1][2]] + pose()[3:])
        rest_steps.append(s_)
    moving = ["- " + target, "- Ctrl+C in the Run window = stop (power switch for an emergency)"]
    result = (corr[0] * 1000, corr[1] * 1000, corr[2] * 1000, turn)

    # ---- step 4: stop just above the drop position (the move before open_gripper())
    d = next((i for i, s_ in enumerate(rest_steps) if s_[0] == "open"), None)
    if d is None or d == 0 or rest_steps[d - 1][0] != "move":
        while_busy(run_steps(rest_steps, poses, then_rest=True), robot, picture, "Robot moving - keep clear", moving)
        return result
    drop = list(rest_steps[d - 1][1])
    dxy = [drop[0], drop[1]]
    dturn = 0.0                                   # extra turn at the drop, degrees (+ = anticlockwise from above)
    hover = drop[2] + GUIDED_DROP_HOVER_MM / 1000

    def drop_pose(xy, z, t):
        return list(xy) + [z] + list(turned_rpy(drop[3:], t))
    while_busy(run_steps(rest_steps[:d - 1] + [("move", drop_pose(dxy, hover, dturn))], poses), robot, picture,
               "Robot moving to the square - keep clear", moving)
    note = ""
    while True:
        ddx, ddy = (dxy[0] - drop[0]) * 1000, (dxy[1] - drop[1]) * 1000
        route = "- Steps:  " + "  ->  ".join(n if i < 3 else f"[{n}]" for i, n in enumerate(names)) + \
                "      ENTER = drop"
        title = f"{which} - step 4 of 4: is the cube right above square {square}?"
        lines = [route,
                 f"- The cube is {GUIDED_DROP_HOVER_MM} mm above where it will be let go (square {square}, level {level}).",
                 "- From at least 30 cm, look from the front, then from the side: is it centred over the square?",
                 "- Press D to move it to your RIGHT, A to your LEFT (1 mm each press).",
                 "- Press W to move it AWAY from you, S TOWARDS you (1 mm each press).",
                 "- Press the LEFT arrow to turn it anticlockwise, the RIGHT arrow clockwise (2 degrees each), so it "
                 "lines up with the square.",
                 f"- Changed so far: {ddx:+.0f} mm towards/away, {ddy:+.0f} mm right/left, turn {dturn:+.0f} degrees.",
                 f"- Lined up? Press ENTER: it goes down and lets go. Changes are saved for square {square}, every "
                 "level.",
                 "- Esc = no change: drop at the saved position."]
        key = ask_step(picture, title, lines + (note.split("\n") if note else []),
                       {"w", "a", "s", "d", "left", "right", "enter", "esc"},
                       "Use W, A, S, D to move, LEFT / RIGHT arrows to turn, then ENTER.")
        note = ""
        if key == "esc":
            dxy, dturn = [drop[0], drop[1]], 0.0
            break
        if key == "enter":
            break
        st = 0.001
        new = list(dxy)
        new[0] += {"s": st, "w": -st}.get(key, 0.0)
        new[1] += {"d": st, "a": -st}.get(key, 0.0)
        new_turn = dturn + {"left": 2.0, "right": -2.0}.get(key, 0.0)
        if max(abs(new[0] - drop[0]), abs(new[1] - drop[1])) * 1000 > cfg.GUIDED_MAX_MM:
            note = f"- Not moved: more than {cfg.GUIDED_MAX_MM} mm from the saved drop position."
            continue
        if abs(new_turn) > GUIDED_MAX_DEG:
            note = f"- Not turned: more than {GUIDED_MAX_DEG} degrees."
            continue
        try:
            check_pose(drop_pose(new, hover, new_turn), poses)
        except ValueError as e:
            note = f"- Not moved: {e}"
            continue
        try:
            while_busy(run_steps([("move", drop_pose(new, hover, new_turn))], poses), robot, picture,
                       f"{which}: moving", busy)
        except RuntimeError as e:
            print(f"  Move refused by the robot: {e}")
            note = MOVE_REFUSED
            continue
        dxy, dturn = new, new_turn
    shift = [dxy[0] - drop[0], dxy[1] - drop[1]]
    tail = [("move", drop_pose(dxy, drop[2], dturn))]
    for s_ in rest_steps[d:]:                  # the moves at the square (e.g. back up) get the same shift and turn
        if s_[0] == "move" and math.hypot(s_[1][0] - drop[0], s_[1][1] - drop[1]) < 0.002:
            s_ = ("move", [s_[1][0] + shift[0], s_[1][1] + shift[1], s_[1][2]] + list(turned_rpy(s_[1][3:], dturn)))
        tail.append(s_)
    while_busy(run_steps(tail, poses, then_rest=True), robot, picture, "Robot moving - keep clear", moving)
    if max(abs(shift[0]), abs(shift[1])) >= 0.0005:
        save_drop_xy(square, level, dxy, poses)
        print(f"  Drop position of square {square} (every level) moved by x {shift[0] * 1000:+.0f} mm, "
              f"y {shift[1] * 1000:+.0f} mm (saved in {cfg.LAB5_FILE.name}, old file kept as a backup)")
    if abs(dturn) >= 1:
        turns = load_drop_turns()
        total = turns.get(square, turns.get(f"{square} {level}", 0.0)) + dturn
        save_drop_turn(square, total)
        print(f"  Drop turn of square {square} (every level): {dturn:+.0f} deg, now {total:+.0f} deg in total "
              f"(saved in {DROP_TURN_FILE.name})")
    return result


DROP_TURN_FILE = cfg.CALIB_DIR / "drop_turn.json"        # guided picks, step 4: gripper turn at each drop


def load_drop_turns():
    try:
        return json.loads(DROP_TURN_FILE.read_text())
    except (FileNotFoundError, ValueError):
        return {}


def with_drop_turn(steps, square, level):
    """The recorded steps with the saved turn for this drop (square, level) added at the drop: the move
    before open_gripper() and the moves at the same place after it."""
    turns = load_drop_turns()
    turn = turns.get(square, turns.get(f"{square} {level}", 0.0))       # one turn per square (all levels)
    c = next((i for i, s_ in enumerate(steps) if s_[0] == "close"), len(steps))
    d = next((i for i, s_ in enumerate(steps) if s_[0] == "open" and i > c), None)   # the open after the grip
    if not turn or d is None or d == 0 or steps[d - 1][0] != "move":
        return list(steps)
    drop = steps[d - 1][1]
    out = list(steps)
    for i in range(c, len(out)):              # every move at the drop place, from above it to back up
        s_ = out[i]
        if s_[0] == "move" and math.hypot(s_[1][0] - drop[0], s_[1][1] - drop[1]) < 0.002:
            out[i] = ("move", list(s_[1][:3]) + list(turned_rpy(s_[1][3:], turn)))
    return out


def save_drop_turn(square, turn):
    turns = {k: v for k, v in load_drop_turns().items() if not k.startswith(square)}   # old per-level entries
    turns[square] = round(float(turn), 1)
    cfg.CALIB_DIR.mkdir(exist_ok=True)
    DROP_TURN_FILE.write_text(json.dumps(turns, indent=2))


MOVE_REFUSED = ("- The robot STOPPED: the gripper pressed on something (the cube or the platform),\n"
                "  or the wrist could not turn there. Move back (e.g. the UP arrow) and try smaller steps.")


def save_drop_xy(square, level, xy, poses):
    """Save a new x, y for the DROP positions of one square at EVERY level (a stack is vertical: only
    the heights differ, and they are not changed), with a backup of the old file. Pick positions are
    not changed, so step 1.5 stays ticked. level: the level where it was corrected (marked tested)."""
    data = read_lab5_file()
    for lv in data["drop"][square]:
        old = data["drop"][square][lv]
        data["drop"][square][lv] = [round(float(xy[0]), 4), round(float(xy[1]), 4), old[2]]
        poses["drop"][square][lv] = list(data["drop"][square][lv])        # used from the next pick on
    data["tested"][f"drop {square} {level}"] = "ok"
    backup = f"lab5_positions_backup_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    shutil.copy(cfg.LAB5_FILE, cfg.CALIB_DIR / backup)
    save_lab5_file(data)


VOICE_WORDS = ("Voice words:  red / yellow / green = sort only that colour,  all = every colour,  "
               "go = pick,  skip = leave it,  stop = end")


def ask_voice(voice, cam_img, title, colour, expected, say, sorting, timeout=None):
    """Voice extra: wait for one of the expected words, keeping the window alive.
    The screen shows what to say now, the last word heard and all the voice words.
    Words said before this question (e.g. during a robot move) are ignored. Returns the word, or
    None after timeout seconds."""
    start = time.time()
    ignored = None
    while timeout is None or time.time() - start < timeout:
        last = voice.last_heard[0] if voice.last_heard else None
        note = f'  - "{ignored}" is not used here' if ignored and last == ignored else ""
        show_status(cam_img, title, [f"SAY: {say}      (sorting now: {sorting or 'all colours'})",
                                     voice.heard_text() + note, VOICE_WORDS], colour, refresh=False)
        word = voice.get(since=start)
        if word in expected:
            print(f"  Voice: {word}")
            return word
        if word:
            ignored = word
            print(f'  Voice: "{word}" ignored here (expected: {", ".join(expected)})')
        cv2.waitKey(50)
    return None


def gripper_turns(steps, poses):
    """From the recorded moves: how far (degrees) the gripper is turned, relative to your Lab 5 yaw,
    where it closes on the cube and where it opens over the square. None if not found."""
    base = _rotation(poses["rpy"])

    def turn_before(index):
        for kind, *rest in reversed(steps[:index]):
            if kind == "move":
                r = _rotation(rest[0][3:]) @ base.T            # the turn from the Lab 5 orientation
                return math.degrees(math.atan2(r[1, 0], r[0, 0]))
        return None
    kinds = [st[0] for st in steps]
    if "close" not in kinds or "open" not in kinds:
        return None, None
    close_i = kinds.index("close")
    open_i = len(kinds) - 1 - kinds[::-1].index("open")
    return turn_before(close_i), turn_before(open_i)


def turn_text(cube, at_cube, at_square):
    """One line for the window and the Run window: cube angle and what the gripper does about it."""
    if at_cube is None:
        return f"cube turned {cube.angle_deg:+.0f} deg"
    text = (f"cube turned {cube.angle_deg:+.0f} deg; gripper turns {at_cube:+.0f} deg at the cube, "
            f"{at_square:+.0f} deg at the square")
    if abs(cube.angle_deg) > 5 and abs(at_cube) < 1:
        text += ("  (no turn: USE_ORIENTATION is False in config.py)" if not cfg.USE_ORIENTATION
                 else "  (no turn: pick_and_drop() does not use cube.angle_deg)")
    return text


def main():
    parser = argparse.ArgumentParser()
    # Technician's test without a robot (commands only printed, e.g. with --source): not part of the lab.
    parser.add_argument("--dry-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--stacks", default="0,0,0")
    parser.add_argument("--source", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--voice", action="store_true", help="optional extra: voice commands")
    parser.add_argument("--measure-offset", action="store_true",
                        help="ask for the gripper offset from the cube centre (mm) after each pick")
    args = parser.parse_args()
    if args.source is not None and not args.dry_run:
        raise SystemExit("--source (saved images) is only allowed with --dry-run")
    CURRENT_STEP[0] = "4:Robot trials"
    print_progress(CURRENT_STEP[0])

    poses = load_lab5_poses()
    model = ColourModel.load()
    vision = VisionSystem(model)
    cam = Camera(args.source)
    stacks = dict(zip("ABC", (int(n) for n in args.stacks.split(","))))
    robot = DryRunRobot() if args.dry_run else NiryoRobot()
    voice, wanted = None, None
    if args.voice or cfg.USE_VOICE:             # --voice, or USE_VOICE = True in config.py
        from voice_commands import VoiceListener
        voice = VoiceListener()
        print('Voice commands on: say a colour or "all" to choose what to sort; "go", "skip", "stop".')

    cfg.RESULTS_DIR.mkdir(exist_ok=True)
    log_path = cfg.RESULTS_DIR / "trials.csv"
    columns = ["time", "label", "confidence", "x_mm", "y_mm", "angle_deg", "square", "level", "outcome", "dry_run",
               "parallax_corrected", "offset_mm", "failure_cause", "guided"]
    if log_path.exists():                         # an older file without the "guided" column: add it (empty)
        with open(log_path, newline="") as f:
            old_rows = list(csv.DictReader(f))
        if old_rows and "guided" not in old_rows[0]:
            with open(log_path, "w", newline="") as f:
                w = csv.DictWriter(f, columns)
                w.writeheader()
                w.writerows({c: r.get(c, "") for c in columns} for r in old_rows)
    new_log = not log_path.exists()
    log_file = open(log_path, "a", newline="")
    log = csv.writer(log_file)
    if new_log:
        log.writerow(columns)

    print(f"step4_sort_blocks.py version {VERSION}")
    correction = load_pick_correction()
    start_correction = correction
    base_z_offset = cfg.ZONE_PICK_OFFSET_MM
    cfg.ZONE_PICK_OFFSET_MM = base_z_offset + correction["dz_mm"]        # used by pick_and_drop()
    guided_left = 0 if (args.dry_run or args.auto) else cfg.GUIDED_PICKS
    guided = []                                   # this run's guided picks: total correction (mm, deg)
    changes = []                                  # this run's guided picks: what you changed (mm, deg)
    corr_path = cfg.RESULTS_DIR / "pick_corrections.csv"
    header = ("time,label,camera_x_mm,camera_y_mm,camera_angle_deg,applied_dx_mm,applied_dy_mm,applied_dz_mm,"
              "applied_turn_deg,dx_mm,dy_mm,dz_mm,turn_deg,parallax_corrected\n")
    if corr_path.exists() and corr_path.open().readline() != header:      # older file without the turn
        corr_path.rename(corr_path.with_name("pick_corrections_old.csv"))
    if not corr_path.exists():
        corr_path.write_text(header)
    guided_angles = []                            # cube angle (as seen) of each guided pick
    if correction["picks"]:
        print(f"Pick correction applied (from {cfg.PICK_CORRECTION_FILE.name}): x {correction['dx_mm']:+.1f} mm, "
              f"y {correction['dy_mm']:+.1f} mm, height {correction['dz_mm']:+.1f} mm, turn "
              f"{correction['turn_deg']:+.1f} deg")
    if guided_left:
        print(f"Guided picks: the first {guided_left} picks stop above the cube for your correction.\n"
              "  Put those cubes at DIFFERENT places and angles in the zone. Guided picks are not counted in the "
              "success rate:\n  the picks after them are the real trials.")

    print(f"Model {model.name}. Stacks at start: {stacks}")
    robot.start()
    empty_looks = 0

    def live(tag="LIVE"):
        """The camera picture now, with the predictions drawn. tag = "LIVE", or "STILL" for a picture
        that is kept on screen while the robot moves."""
        frame = cam.read()
        img, _ = vision.draw(frame, vision.process(frame))
        cv2.putText(img, tag, (img.shape[1] - 95, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255) if tag == "LIVE" else (90, 90, 90), 2, cv2.LINE_AA)
        return img

    def live_target(cube, tag="LIVE"):
        """Live picture with a cross where the robot will go (from the picture the cube was chosen in).
        If the cube is moved, the cross no longer sits on it: press S to skip and look again."""
        img = live(tag)
        try:
            px, py = (int(round(c)) for c in vision.view.robot_to_camera(cube.x, cube.y))
        except Exception:                         # markers not seen yet in this frame
            return img
        cv2.drawMarker(img, (px, py), (255, 255, 255), cv2.MARKER_CROSS, 34, 5, cv2.LINE_AA)
        cv2.drawMarker(img, (px, py), (0, 0, 0), cv2.MARKER_CROSS, 30, 2, cv2.LINE_AA)
        cv2.putText(img, "robot goes here", (px + 20, py + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 4,
                    cv2.LINE_AA)
        cv2.putText(img, "robot goes here", (px + 20, py + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                    cv2.LINE_AA)
        return img
    try:
        while True:
            while_busy(lambda r: (r.learning_mode(False), r.move_joints(cfg.OBSERVE_JOINTS), r.learning_mode(True)),
                       robot, live("STILL"), "Robot moving to the observe position (sleep pose, learning mode on)",
                       ["Keep clear. Ctrl+C in the Run window = stop"])
            time.sleep(0.5)                        # let the arm settle before the picture
            frame = cam.read_fresh()
            result = vision.process(frame)
            cam_img, _ = vision.draw(frame, result)
            show_status(cam_img, "Looking for cubes...",
                        [f"Stacks: {stacks}",
                         "Voice commands ON: listening" if voice is not None else
                         "Voice commands OFF (to use them: USE_VOICE = True in config.py, or run with --voice)"],
                        BLUE)

            if not result.fresh:
                print("Not all 4 markers visible: no move. Clear the view and try again.")
                if ask("Enter = look again, q = quit", False, live,
                       f"Markers not found ({result.markers_seen}/4): the robot will not move",
                       ["Clear the view of the markers.", "Enter = look again, Q = quit"], RED, "\rq") == "q":
                    break
                continue

            # VOICE EXTRA: a colour word or "all" changes what is sorted; "stop" ends the program
            if voice is not None:
                word = voice.get()
                if word == "stop":
                    print("Voice: stop")
                    break
                if word in cfg.COLOUR_TO_SQUARE or word == "all":
                    wanted = None if word == "all" else word
                    print(f"Voice: sorting {word}")

            cube, reasons = choose(result.detections, stacks, wanted)
            if reasons:
                print("  skipped:", "; ".join(reasons))
            if cube is None:
                empty_looks += 1
                print("No cube to sort.")
                if args.auto and empty_looks >= 2:
                    break
                if voice is not None:
                    word = ask_voice(voice, live, "No cube to sort: " + ("; ".join(reasons)[:70] or
                                                                         "put cubes in the zone"), ORANGE,
                                     list(cfg.COLOUR_TO_SQUARE) + ["all", "stop"],
                                     'a colour or "all" to change what is sorted, "stop" to end '
                                     '(looks again after 5 s)', wanted, timeout=5)
                    if word == "stop":
                        break
                    if word is not None:
                        wanted = None if word == "all" else word
                    continue
                if ask("Enter = look again, q = quit", args.auto, live, "No cube to sort",
                       ["; ".join(reasons)[:110] or "Put cubes in the zone.",
                        "Enter = look again, Q = quit when done"], ORANGE, "\rq") == "q":
                    break
                continue
            empty_looks = 0
            seen_x, seen_y, seen_angle = cube.x, cube.y, cube.angle_deg   # what the camera saw
            cube.x += correction["dx_mm"] / 1000             # learned pick correction (guided picks)
            cube.y += correction["dy_mm"] / 1000
            cube.angle_deg += correction["turn_deg"]         # used by pick_and_drop() when USE_ORIENTATION

            square = cfg.COLOUR_TO_SQUARE[cube.label]
            level = stacks[square] + 1                    # level 1 = first cube on the square

            # Run YOUR pick_and_drop() on the recorder first: nothing moves yet.
            recorder = PlanRecorder(poses)
            try:
                pick_and_drop(recorder, cube, poses, square, level)
                problems = review_moves(recorder.steps, cube, poses, square, level)
            except ValueError as e:                       # a pose outside the safe limits
                problems = [str(e)]
            if problems:
                print("The robot will NOT move. Fix pick_and_drop():")
                for p in problems:
                    print("  -", p)
                if ask("Enter = look again, q = quit", args.auto, cam_img,
                       "Your pick_and_drop() was refused: the robot will not move",
                       [p[:120] for p in problems[:3]] + ["Details in the Run window. Enter = look again, Q = quit"],
                       RED, "\rq") == "q":
                    break
                continue

            target = (f"{cube.label} ({cube.confidence:.2f}) at x={cube.x * 1000:.0f} mm y={cube.y * 1000:.0f} mm "
                      f"-> square {square}, level {level}")
            turns = turn_text(cube, *gripper_turns(recorder.steps, poses))
            print("\n" + target + "\n  " + turns)
            if voice is not None:
                word = ask_voice(voice, lambda: live_target(cube), "Ready: " + target, GREEN,
                                 ["go", "skip", "stop", "all"] + list(cfg.COLOUR_TO_SQUARE),
                                 '"go" to pick this cube, "skip", or "stop"   (a colour or "all" changes what '
                                 'is sorted)', wanted)
                if word in cfg.COLOUR_TO_SQUARE or word == "all":
                    wanted = None if word == "all" else word
                    print(f"  Voice: sorting {word}; looking again")
                answer = {"go": "", "skip": "s", "stop": "q"}.get(word, "s")
            else:
                answer = ask("Enter = go, s = skip, q = quit", args.auto, lambda: live_target(cube),
                             "Ready to move: " + target,
                             [turns, "Check the cross is on the cube and the target square is right. Moved the "
                              "cube? S = skip and look again.",
                              "Enter = go, S = skip, Q = quit      (voice commands OFF: USE_VOICE = True in config.py)"],
                             GREEN, "\rsq")
            if answer == "q":
                break
            if answer == "s":
                continue

            still = live_target(cube, "STILL")                    # the picture stays still while the robot moves
            result_ = None
            was_guided = False                    # corrected by hand: not counted in the success rate
            if guided_left > 0:
                which = f"Guided pick {cfg.GUIDED_PICKS - guided_left + 1} of {cfg.GUIDED_PICKS}"
                result_ = guided_pick(recorder, robot, cube, poses, lambda: live_target(cube), target, which,
                                      square, level)
                if result_ == "stop":
                    print("Guided pick stopped: nothing picked")
                    continue
                if result_ is None:
                    print("Guided pick not possible (no move down to the cube followed by close_gripper()): "
                          "normal pick")
            if result_ is None:
                try:
                    while_busy(run_steps(with_drop_turn(recorder.steps, square, level), poses, then_rest=True),
                               robot, still, "Robot moving - keep clear",
                               [target + "   " + turns, "Ctrl+C in the Run window = stop (power switch for an emergency)"]
                               + (["Voice: words said while the robot moves are ignored"] if voice is not None else []))
                except RuntimeError as e:             # the robot stopped itself (e.g. the gripper pressed on the cube)
                    print(f"The robot STOPPED during the pick: {e}")
                    while_busy(lambda r: back_off(r, poses), robot, still, "Robot stopped - moving up",
                               ["Opening the gripper, moving up and resting"])
                    log.writerow([datetime.now().isoformat(timespec="seconds"), cube.label, round(cube.confidence, 3),
                                  round(cube.x * 1000, 1), round(cube.y * 1000, 1), round(cube.angle_deg, 1),
                                  square, level, "n", args.dry_run, cfg.USE_PARALLAX_CORRECTION, "",
                                  "robot stopped", was_guided])
                    log_file.flush()
                    ask("Enter = carry on", False, live, "The robot STOPPED",
                        ["- The gripper pressed on something (the cube or the platform), so the robot stopped.",
                         "- Recorded as a failed pick (robot stopped). Check the cube, then press Enter."], RED,
                        "\r", "Press Enter to carry on (click on the camera window first).")
                    continue
            else:
                guided_left -= 1
                was_guided = True
                applied = [correction["dx_mm"], correction["dy_mm"], correction["dz_mm"],
                           correction["turn_deg"] if cfg.USE_ORIENTATION else 0.0]
                guided.append([applied[i] + result_[i] for i in range(4)])       # total correction at this pick
                changes.append(result_)
                guided_angles.append(seen_angle)
                print(f"  Guided correction: x {result_[0]:+.0f} mm, y {result_[1]:+.0f} mm, "
                      f"height {result_[2]:+.0f} mm, turn {result_[3]:+.0f} deg  ({guided_left} guided pick(s) left)")
                with open(corr_path, "a") as f:
                    f.write(f"{datetime.now().isoformat(timespec='seconds')},{cube.label},{seen_x * 1000:.1f},"
                            f"{seen_y * 1000:.1f},{seen_angle:.1f},{correction['dx_mm']},{correction['dy_mm']},"
                            f"{correction['dz_mm']},{correction['turn_deg']},{result_[0]:.1f},{result_[1]:.1f},"
                            f"{result_[2]:.1f},{result_[3]:.0f},{cfg.USE_PARALLAX_CORRECTION}\n")
                # Saved after EVERY guided pick and used from the NEXT pick on (running average), so the next
                # guided pick already starts closer and nothing is lost if the run stops early.
                correction, spread, slope = save_pick_correction(start_correction, guided, guided_angles)
                cfg.ZONE_PICK_OFFSET_MM = base_z_offset + correction["dz_mm"]
                print(f"  Saved and used from the next pick on: x {correction['dx_mm']:+.1f} mm, "
                      f"y {correction['dy_mm']:+.1f} mm, height {correction['dz_mm']:+.1f} mm, "
                      f"turn {correction['turn_deg']:+.1f} deg")
                if guided_left == 0:
                    mean = [sum(c[i] for c in changes) / len(changes) for i in range(4)]
                    lines = [f"- On average you changed: x {mean[0]:+.1f} mm, y {mean[1]:+.1f} mm, "
                             f"height {mean[2]:+.1f} mm, turn {mean[3]:+.1f} deg per guided pick.",
                             f"- Total correction now: x {correction['dx_mm']:+.1f}, y {correction['dy_mm']:+.1f}, "
                             f"height {correction['dz_mm']:+.1f} mm, turn {correction['turn_deg']:+.1f} deg  "
                             f"({cfg.PICK_CORRECTION_FILE.name})"]
                    if not cfg.USE_PARALLAX_CORRECTION and spread <= 3:
                        lines += ["- Notice: did the gripper often need shifting the same way, more for cubes further "
                                  "from the camera? That is parallax: the camera sees the cube tops from an angle.",
                                  "- Optional, to fine-tune the picking: correct it with USE_PARALLAX_CORRECTION "
                                  "(config.py), or learn it from your corrections in results/pick_corrections.csv "
                                  "(guide: 'Optional: fine-tuning the picking')."]
                    if not cfg.USE_ORIENTATION and abs(mean[3]) >= 1:
                        lines.append("- The turn correction is only used when USE_ORIENTATION = True (config.py).")
                    if slope is not None and slope < -1.2:
                        lines.append("- Your turn corrections grow with the cube's angle, the other way: the gripper "
                                     "turns the WRONG WAY. Run test_wrist_turn.py and check YAW_SIGN in config.py.")
                    elif slope is not None and abs(slope) > 0.5:
                        lines.append(f"- Your turn corrections depend on the cube's angle (slope {slope:+.1f}): one "
                                     "average cannot fix that. Check how the cube angle is measured.")
                    if spread > 3:
                        lines.append(f"- Your corrections differ by up to {spread:.0f} mm from place to place: one "
                                     "offset cannot fix that. It is parallax. Recommended: measure CAMERA_POSITION_M and switch on "
                                     "USE_PARALLAX_CORRECTION (config.py).")
                    print("\n".join(lines))
                    ask("Enter = carry on", False, live, "Guided picks done", lines, GREEN, "\r",
                        "Press Enter to carry on (click on the camera window first).")
            stacks[square] += 1

            outcome = ""
            if not args.dry_run and not args.auto:
                outcome = ask("Placed correctly on the stack? y/n", False, live, "Was the cube placed correctly?",
                              ["Y = yes, N = no"], BLUE, "yn")
            cause = ""
            if outcome == "n":
                cause = {"w": "wrong colour", "g": "gripper missed", "f": "fell off stack",
                         "p": "wrong position"}.get(
                    ask("What went wrong?", False, live, "What went wrong?",
                        [f"W = wrong colour (it was not {cube.label}),  G = gripper missed",
                         "F = fell off the stack,  P = wrong position,  Enter = other"], RED, "wgfp\r"), "other")
            offset = ""
            if args.measure_offset and not args.dry_run:
                offset = ask_number("Gripper offset from the cube centre when it closed, mm", live,
                                    "Measure the grip offset")
            log.writerow([datetime.now().isoformat(timespec="seconds"), cube.label, round(cube.confidence, 3),
                          round(cube.x * 1000, 1), round(cube.y * 1000, 1), round(cube.angle_deg, 1),
                          square, level, outcome, args.dry_run, cfg.USE_PARALLAX_CORRECTION, offset,
                          cause, was_guided])
            log_file.flush()
            print(f"Stacks now: {stacks}")
    except KeyboardInterrupt:
        print("\nStopped by Ctrl+C")
    finally:
        robot.finish()
        log_file.close()
        cam.release()
        if voice is not None:
            voice.stop()
        cv2.destroyAllWindows()
        print_progress(CURRENT_STEP[0])


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as e:
        raise SystemExit(f"\nMissing: {e}\nCheck the function named in the message.")
