"""
STEP 4 - Sort the cubes with the robot.

KEY PARTS: two functions in this file (marked "KEY PART"):
  choose()         - which cube to sort next
  pick_and_drop()  - the robot moves for one cube, based on the Lab 5 program
Around them: the camera, the loop, the safety checks and the log.

ALWAYS start with a dry run (the robot does not move, commands are printed):
    python step4_sort_blocks.py --dry-run

Then with the robot, confirming every move:
    python step4_sort_blocks.py

Options:
    --stacks 0,0,0   cubes already on A,B,C at the start (default none)
    --auto           do not ask before each move (only after supervised runs work)
    --source PATH    use saved images instead of the camera (dry run only)
    --measure-offset after each pick, type how far (mm) the gripper was from the cube centre:
                     compare with and without USE_PARALLAX_CORRECTION (config.py)
    --voice          OPTIONAL EXTRA: voice commands (see voice_commands.py)

Each cycle: arm to the observe pose -> fresh picture -> choose() -> your
pick_and_drop() is first run on a RECORDER that checks every pose and the order
of the moves -> only if everything passes does the real robot repeat it.
Every attempt is logged in results/trials.csv for your success rate.

The box at the top of the camera window says what the program is doing and
what to answer. You answer in the PyCharm Run window (click in it first).

Stopping: answer q at a prompt, or press Ctrl+C. The program then sends the
robot to its sleep pose. If the robot is about to hit something, switch off
its power supply.
"""
import argparse
import csv
import math
import threading
import time
from datetime import datetime

import cv2

import config as cfg
from camera import Camera
from colour_model import ColourModel
from robot import DryRunRobot, NiryoRobot, PlanRecorder, _rotation, load_lab5_poses, turned_rpy
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
CURRENT_STEP = ["4:Dry run"]


def show_status(cam_img, title, lines, colour, refresh=True):
    """cam_img is a picture, or a function that returns the live camera picture.
    refresh=False when the caller reads the keys itself (so no key press is lost here)."""
    if callable(cam_img):
        cam_img = cam_img()
    show("camera", instruction_box(with_progress(cam_img, CURRENT_STEP[0]), title, lines, colour))
    if refresh:
        cv2.waitKey(1)


KEY_NAMES = {"\r": "Enter", " ": "Space"}


def ask(prompt, auto, cam_img=None, title=None, lines=None, colour=BLUE, keys="\rsq"):
    """Wait for a key pressed IN THE CAMERA WINDOW, keeping the window alive (no "Not Responding").
    Returns "" for Enter, otherwise the letter."""
    if auto:
        return ""
    print(prompt + " (press the key in the camera window)")
    while True:
        if cam_img is not None:
            show_status(cam_img, title or prompt, (lines or []) + ["Click on the camera window, then press: " +
                        ", ".join(KEY_NAMES.get(k, k.upper()) for k in keys)], colour, refresh=False)
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
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--stacks", default="0,0,0")
    parser.add_argument("--source", default=None)
    parser.add_argument("--voice", action="store_true", help="optional extra: voice commands")
    parser.add_argument("--measure-offset", action="store_true",
                        help="ask for the gripper offset from the cube centre (mm) after each pick")
    args = parser.parse_args()
    if args.source is not None and not args.dry_run:
        raise SystemExit("--source (saved images) is only allowed with --dry-run")
    CURRENT_STEP[0] = "4:Dry run" if args.dry_run else "4:Robot trials"
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
    new_log = not log_path.exists()
    log_file = open(log_path, "a", newline="")
    log = csv.writer(log_file)
    if new_log:
        log.writerow(["time", "label", "confidence", "x_mm", "y_mm", "angle_deg",
                      "square", "level", "outcome", "dry_run", "parallax_corrected", "offset_mm",
                      "failure_cause"])

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
            while_busy(lambda r: r.move_joints(cfg.OBSERVE_JOINTS), robot, live("STILL"),
                       "Robot moving to the observe position", ["Keep clear. Ctrl+C in the Run window = stop"])
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
            while_busy(recorder.replay, robot, still, "Robot moving - keep clear",
                       [target + "   " + turns, "Ctrl+C in the Run window = stop (power switch for an emergency)"]
                       + (["Voice: words said while the robot moves are ignored"] if voice is not None else []))
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
                          cause])
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
