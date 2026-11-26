"""
Robot connection, your Lab 5 poses, and the safety checks.

NiryoRobot  - talks to the real Niryo One (niryo_one_tcp_client, Section 3).
DryRunRobot - prints every command instead of moving. Use it first, always.

Your Lab 5 positions come from step1_lab5_positions.py.
Every move goes through check_pose(): a pose outside the limits in
config.py, or lower than your lowest Lab 5 pick or drop height, is refused.
"""
import json
import math
import sys
import time
from datetime import datetime

import config as cfg

# Find the Niryo One TCP client cloned in Section 3 of the lab manual, whether it
# sits inside this folder or next to it, so the scripts also work from a plain terminal.
for _base in (cfg.BASE_DIR, cfg.BASE_DIR.parent):
    for _folder in ("ApiTCP_Python_NiryoOne", "ApiTCP_Pyhton_NiryoOne"):
        _path = _base / _folder
        if _path.exists() and str(_path) not in sys.path:
            sys.path.insert(0, str(_path))


# ---------------------------------------------------------------- Lab 5 positions
SQUARES = ("A", "B", "C")
LEVELS = (1, 2, 3)


def safe_height(data):
    """Travel height: the highest pick or drop position plus SAFE_CLEARANCE_M."""
    zs = [p[2] for kind in ("pick", "drop") for sq in data[kind].values() for p in sq.values()]
    return max(zs) + cfg.SAFE_CLEARANCE_M


def check_lab5(data):
    """Check Lab 5 positions for typing errors.

    data = {"orientation": [roll, pitch, yaw],
            "pick": {"A": {1: [x, y, z], 2: ..., 3: ...}, "B": ..., "C": ...}, "drop": {same}}
    Returns (errors, warnings). Each item is (where, message); where is ("pick", "A", 1, "z"),
    ("orientation", i) or None. Errors stop saving and moving; warnings are worth checking."""
    errors, warnings = [], []
    orientation = data.get("orientation", [])
    for i, name in enumerate(("roll", "pitch", "yaw")):
        v = orientation[i] if i < len(orientation) else None
        if not isinstance(v, (int, float)):
            errors.append((("orientation", i), f"orientation {name}: type a number (radians)"))
        elif abs(v) > 3.2:
            errors.append((("orientation", i), f"orientation {name} = {v}: must be between -3.14 and 3.14 radians"))
    if not errors:
        tilt = abs(90 - abs(math.degrees(orientation[1])))
        if tilt > 10:
            warnings.append((("orientation", 1), f"pitch {orientation[1]:.3f}: the gripper is {tilt:.0f} degrees "
                                                 f"from pointing straight down"))
    names = ("x", "y", "z")
    complete = {}
    for kind in ("pick", "drop"):
        for sq in SQUARES:
            for lv in LEVELS:
                p = data.get(kind, {}).get(sq, {}).get(lv)
                ok = True
                for i, axis in enumerate(names):
                    v = p[i] if p and i < len(p) else None
                    where = (kind, sq, lv, axis)
                    label = f"{kind.upper()} {sq} level {lv} {axis}"
                    if not isinstance(v, (int, float)):
                        errors.append((where, f"{label}: type a number (metres)"))
                        ok = False
                        continue
                    if axis == "x" and not cfg.X_LIMITS[0] <= v <= cfg.X_LIMITS[1]:
                        errors.append((where, f"{label} = {v}: must be between {cfg.X_LIMITS[0]} and {cfg.X_LIMITS[1]}"))
                        ok = False
                    if axis == "y" and not cfg.Y_LIMITS[0] <= v <= cfg.Y_LIMITS[1]:
                        errors.append((where, f"{label} = {v}: must be between {cfg.Y_LIMITS[0]} and {cfg.Y_LIMITS[1]}"))
                        ok = False
                    if axis == "z" and v < cfg.Z_MIN:
                        errors.append((where, f"{label} = {v}: lower than the minimum height Z_MIN = {cfg.Z_MIN} m "
                                              f"(typing error?)"))
                        ok = False
                    if axis == "z" and v > cfg.Z_MAX:
                        errors.append((where, f"{label} = {v}: higher than Z_MAX = {cfg.Z_MAX} m"))
                        ok = False
                if ok:
                    complete[(kind, sq, lv)] = p
    for kind in ("pick", "drop"):
        for sq in SQUARES:
            if not all((kind, sq, lv) in complete for lv in LEVELS):
                continue
            pts = [complete[(kind, sq, lv)] for lv in LEVELS]
            for lv in (2, 3):
                step = (pts[lv - 1][2] - pts[lv - 2][2]) * 1000
                where = (kind, sq, lv, "z")
                if step <= 0:
                    errors.append((where, f"{kind.upper()} {sq}: level {lv} z must be higher than level {lv - 1} z"))
                elif step > 40:
                    warnings.append((where, f"{kind.upper()} {sq}: level {lv} is {step:.0f} mm above level {lv - 1} "
                                            f"(a cube is {cfg.CUBE_SIZE_MM} mm)"))
            for lv in (2, 3):
                d = math.hypot(pts[lv - 1][0] - pts[0][0], pts[lv - 1][1] - pts[0][1]) * 1000
                if d > 10:
                    warnings.append(((kind, sq, lv, "x"), f"{kind.upper()} {sq}: level {lv} x, y is {d:.0f} mm away "
                                                          f"from level 1 (same square?)"))
    for sq in SQUARES:
        for lv in LEVELS:
            if ("pick", sq, lv) in complete and ("drop", sq, lv) in complete:
                a, b = complete[("pick", sq, lv)], complete[("drop", sq, lv)]
                d = math.hypot(a[0] - b[0], a[1] - b[1]) * 1000
                if d > 10:
                    warnings.append((("drop", sq, lv, "x"), f"square {sq} level {lv}: drop x, y is {d:.0f} mm away "
                                                            f"from pick x, y"))
    centres = {sq: complete[("pick", sq, 1)] for sq in SQUARES if ("pick", sq, 1) in complete}
    for s1, s2 in (("A", "B"), ("B", "C"), ("A", "C")):
        if s1 in centres and s2 in centres:
            d = math.hypot(centres[s1][0] - centres[s2][0], centres[s1][1] - centres[s2][1]) * 1000
            if d < 30:
                warnings.append((("pick", s2, 1, "y"), f"squares {s1} and {s2} are only {d:.0f} mm apart"))
    return errors, warnings


def read_lab5_file(path=None):
    """The saved positions as typed (may be incomplete), or None if not saved yet."""
    path = path or cfg.LAB5_FILE
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    data = {"orientation": raw.get("orientation", []), "tested": raw.get("tested", {}),
            "robot": raw.get("robot"), "robot_ip": raw.get("robot_ip")}
    for kind in ("pick", "drop"):
        data[kind] = {sq: {int(lv): p for lv, p in raw.get(kind, {}).get(sq, {}).items()} for sq in SQUARES}
    return data


def robot_ip():
    """The IP of the robot chosen in step1_lab5_positions.py, else ROBOT_IP in config.py."""
    if cfg.LAB5_FILE.exists():
        ip = json.loads(cfg.LAB5_FILE.read_text()).get("robot_ip")
        if ip:
            return ip
    return cfg.ROBOT_IP


def save_lab5_file(data, path=None):
    out = {"saved": datetime.now().isoformat(timespec="minutes"),
           "robot": data.get("robot"), "robot_ip": data.get("robot_ip"),
           "orientation": data["orientation"],
           "pick": {sq: {str(lv): data["pick"][sq][lv] for lv in LEVELS} for sq in SQUARES},
           "drop": {sq: {str(lv): data["drop"][sq][lv] for lv in LEVELS} for sq in SQUARES},
           "tested": data.get("tested", {})}
    path = path or cfg.LAB5_FILE
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=2))


def load_lab5_poses():
    """Your Lab 5 positions (entered with step1_lab5_positions.py), checked.

    Returns {"rpy": [roll, pitch, yaw], "z_safe": z,
             "pick": {"A": {1: [x, y, z], 2: ..., 3: ...}, "B": ..., "C": ...}, "drop": {same}}"""
    data = read_lab5_file()
    if data is None:
        raise FileNotFoundError("Lab 5 positions not entered yet: run step1_lab5_positions.py")
    errors, _ = check_lab5(data)
    if errors:
        raise ValueError(f"Lab 5 positions have {len(errors)} problem(s), e.g. {errors[0][1]}. "
                         f"Fix them with step1_lab5_positions.py")
    return {"rpy": [float(v) for v in data["orientation"]], "z_safe": safe_height(data),
            "pick": data["pick"], "drop": data["drop"]}


def square_centres(poses):
    """x, y of squares A, B and C (their level-1 pick position)."""
    return {s: poses["pick"][s][1][:2] for s in SQUARES}


def lowest_allowed_z(poses):
    """Nothing may go lower than Z_MIN, or than the lowest Lab 5 pick or drop height."""
    zs = [p[2] for t in ("pick", "drop") for sq in poses[t].values() for p in sq.values()]
    return max(cfg.Z_MIN, min(zs) - 0.002)


def check_pose(pose, poses):
    x, y, z = pose[:3]
    problems = []
    if not cfg.X_LIMITS[0] <= x <= cfg.X_LIMITS[1]:
        problems.append(f"x={x:.3f} outside {cfg.X_LIMITS}")
    if not cfg.Y_LIMITS[0] <= y <= cfg.Y_LIMITS[1]:
        problems.append(f"y={y:.3f} outside {cfg.Y_LIMITS}")
    if z < lowest_allowed_z(poses):
        problems.append(f"z={z:.3f} below the lowest Lab 5 pick or drop height")
    if problems:
        raise ValueError("Unsafe pose refused: " + "; ".join(problems))


def pose_from_args(args):
    """move_pose accepts move_pose(x, y, z, roll, pitch, yaw), as in Lab 5, or move_pose([x, y, z, r, p, y])."""
    pose = list(args[0]) if len(args) == 1 else list(args)
    if len(pose) != 6:
        raise ValueError(f"move_pose needs 6 numbers (x, y, z, roll, pitch, yaw), got {len(pose)}: {pose}")
    pose = [float(v) for v in pose]
    # The robot only accepts angles from -pi to +pi: -3.41 rad is the same direction as +2.87 rad.
    return pose[:3] + [wrap_angle(a) for a in pose[3:]]


def wrap_angle(a):
    """The same angle, written between -pi and +pi (radians)."""
    return (a + math.pi) % (2 * math.pi) - math.pi


# ---------------------------------------------------------------- robots
def _rotation(rpy):
    """Rotation matrix for roll, pitch, yaw (radians), the convention the Niryo One uses."""
    import numpy as np
    r, p, y = rpy
    rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


def orientation_gap_deg(rpy_a, rpy_b):
    """The angle (degrees) between two gripper orientations, whatever roll, pitch, yaw they are written in."""
    import numpy as np
    r = _rotation(rpy_a).T @ _rotation(rpy_b)
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(r) - 1) / 2))))


def turned_rpy(rpy, angle_deg):
    """Your Lab 5 orientation turned by angle_deg about the vertical (+ = anticlockwise seen from
    above, the same as cube.angle_deg), to line the gripper up with a cube.
    Adding to yaw turns about the vertical for any pitch. If test_wrist_turn.py showed that only
    roll works on your robot, set cfg.TURN_AXIS = "roll"."""
    roll, pitch, yaw = rpy
    a = cfg.YAW_SIGN * math.radians(angle_deg)
    if cfg.TURN_AXIS == "roll":            # gripper pointing down: + roll turns it clockwise from above
        return [wrap_angle(roll - math.copysign(1.0, pitch) * a), pitch, yaw]
    return [roll, pitch, wrap_angle(yaw + a)]


class PlanRecorder:
    """Runs your pick-and-drop code WITHOUT moving anything: it records every command and checks
    every pose. The real robot only moves after the whole sequence has passed."""

    def __init__(self, poses):
        self.poses = poses
        self.steps = []                       # ("move", pose), ("open",), ("close",), ("joints", j)
        self.pose = None

    def move_pose(self, *args):
        pose = pose_from_args(args)
        check_pose(pose, self.poses)          # raises ValueError if unsafe
        self.steps.append(("move", pose))
        self.pose = pose

    def move_joints(self, *args):
        self.steps.append(("joints", list(args[0]) if len(args) == 1 else list(args)))
        self.pose = None

    def open_gripper(self):
        self.steps.append(("open",))

    def close_gripper(self):
        self.steps.append(("close",))

    def get_pose(self):
        return self.pose

    def replay(self, robot, pause=0.4):
        """Send the recorded commands to a real (or dry-run) robot."""
        import time
        for step in self.steps:
            if step[0] == "move":
                robot.move_pose(step[1])
            elif step[0] == "joints":
                robot.move_joints(step[1])
            elif step[0] == "close":
                robot.close_gripper()
                time.sleep(pause)             # let the gripper close, as in Lab 5
            else:
                robot.open_gripper()
                time.sleep(pause)


class NiryoRobot:
    def __init__(self, ip=None):
        from niryo_one_tcp_client import NiryoOneClient, RobotTool, CalibrateMode
        self._tool = RobotTool.GRIPPER_1
        self._calibrate_mode = CalibrateMode.AUTO
        self.client = NiryoOneClient()
        self.ip = ip or robot_ip()

    @staticmethod
    def _ok(result, what):
        status, data = result
        if not status:
            raise RuntimeError(f"{what} failed: {data}")
        return data

    def start(self, attempts=3):
        for attempt in range(1, attempts + 1):
            if self.client.connect(self.ip):
                break
            print(f"Cannot connect to the robot at {self.ip} (try {attempt} of {attempts}).")
            if attempt < attempts:
                print("  Trying again in 5 s...")
                time.sleep(5)
        else:
            raise SystemExit(
                f"\nCould not connect to the robot at {self.ip}. The robot accepts ONE program at a time:\n"
                "  - close every other program connected to it (Niryo One Studio, the Lab 5 positions or markers\n"
                "    window, another Run tab in PyCharm, a Lab 5 program);\n"
                "  - if the last run was stopped with PyCharm's red Stop button, the robot may still hold that\n"
                "    connection: wait about 30 s and try again, or restart the robot;\n"
                "  - check the robot IP in config.py / your Lab 5 positions and that this PC is on the robot's network.\n"
                "Quit with Q in the window next time, so the connection is closed properly.")
        self._ok(self.client.calibrate(self._calibrate_mode), "calibrate")
        self._ok(self.client.change_tool(self._tool), "change_tool")
        self._ok(self.client.open_gripper(self._tool, cfg.GRIPPER_SPEED), "open_gripper")
        print(f"Robot {self.ip} connected and calibrated")

    def move_pose(self, *args):
        pose = pose_from_args(args)
        self._ok(self.client.move_pose(*pose), f"move_pose {pose}")
        # Check the robot really reached the orientation it was given (e.g. the turn to match a cube).
        try:
            actual = self.get_pose()
            gap = orientation_gap_deg(pose[3:], actual[3:])
            if gap > 5:
                print(f"  WARNING: gripper orientation is {gap:.0f} deg away from the one asked for "
                      f"(asked roll, pitch, yaw {[round(v, 2) for v in pose[3:]]}, "
                      f"robot reports {[round(v, 2) for v in actual[3:]]}). Run python test_wrist_turn.py")
        except Exception as e:                  # the check must never stop the robot program
            print("  (orientation check skipped:", e, ")")

    def move_joints(self, joints):
        self._ok(self.client.move_joints(*joints), f"move_joints {joints}")

    def open_gripper(self):
        self._ok(self.client.open_gripper(self._tool, cfg.GRIPPER_SPEED), "open_gripper")

    def close_gripper(self):
        self._ok(self.client.close_gripper(self._tool, cfg.GRIPPER_SPEED), "close_gripper")

    def get_pose(self):
        return self._ok(self.client.get_pose(), "get_pose").to_list()

    def get_joints(self):
        return list(self._ok(self.client.get_joints(), "get_joints"))

    def learning_mode(self, enabled):
        self._ok(self.client.set_learning_mode(enabled), "set_learning_mode")

    def finish(self):
        """Go to the sleep pose and release the motors."""
        try:
            self.client.set_learning_mode(False)          # it may be resting in learning mode
            self.client.move_joints(*cfg.SLEEP_JOINTS)
            self.client.set_learning_mode(True)
        finally:
            self.client.quit()


class DryRunRobot:
    """Same commands as NiryoRobot, but only prints them."""

    def __init__(self, *_):
        self.pose = [0.0] * 6

    def start(self):
        print("[dry run] robot not connected - commands are only printed")

    def move_pose(self, *args):
        pose = pose_from_args(args)
        self.pose = pose
        print("[dry run] move_pose  " + "  ".join(f"{v:+.3f}" for v in pose))

    def move_joints(self, joints):
        print("[dry run] move_joints " + "  ".join(f"{v:+.3f}" for v in joints))

    def open_gripper(self):
        print("[dry run] open_gripper")

    def close_gripper(self):
        print("[dry run] close_gripper")

    def get_pose(self):
        return self.pose

    def learning_mode(self, enabled):
        print(f"[dry run] learning_mode {enabled}")

    def finish(self):
        print("[dry run] finish: sleep pose, learning mode on")
