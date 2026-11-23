"""
STEP 1.4 - Measure where the four ArUco markers are, in robot coordinates.

    python step1_measure_markers.py

A window opens; the line at the top always says what to do next.

  1. Connect to the robot (it calibrates, about 30 seconds).
  2. For each marker: select it, switch Learning mode ON, move the gripper by hand
     so it points straight down with the closed fingers centred over the marker,
     about 5 mm above it, and press "Read robot position". The tilt is checked:
     more than 3 degrees from vertical is shown in red, so straighten and read again.
     Without the robot you can type x, y, z (metres) instead.
  3. Check the fit. The marker spacing is the same on every station (MARKER_LAYOUT_MM
     in config.py), so the program fits that exact rectangle to your four readings.
     Each reading's distance from the fit is shown: over 5 mm (red) means that reading
     is poor - read ONLY that marker again. "Test" moves the gripper above a marker so
     you can see whether the reading is right.
  4. Save. Your earlier readings are loaded next time, so you only redo the markers
     that need it.

Saved to calibration/markers.json: the fitted positions (used by every step) and your
raw readings.
"""
import json
import math
import tkinter as tk
from datetime import datetime
from itertools import combinations
from tkinter import messagebox

import numpy as np

import config as cfg
from progress import is_current, project_status
from robot import NiryoRobot, pose_from_args, robot_ip

MAX_TILT_DEG = 3.0
HOVER_M = 0.03                 # "Test" stops this far above the reading
GREEN, RED, ORANGE, GREY, BLUE = "#006000", "#b00000", "#a06000", "#707070", "#204080"


def tilt_from_vertical(pose):
    """Degrees between the gripper and straight down (pitch = 90 degrees when pointing down)."""
    return abs(90.0 - abs(math.degrees(pose[4])))


def fit_layout(measured):
    """Move and rotate the known layout (no stretching) to best match the robot readings.
    Returns fitted {id: [x, y]} in metres and the distance of each reading from the fit in mm."""
    ids = cfg.MARKER_IDS
    p = np.array([cfg.MARKER_LAYOUT_MM[i] for i in ids], dtype=float) / 1000.0   # layout, metres
    q = np.array([measured[i][:2] for i in ids], dtype=float)                    # robot readings
    pc, qc = p - p.mean(axis=0), q - q.mean(axis=0)
    u, _, vt = np.linalg.svd(pc.T @ qc)
    d = np.sign(np.linalg.det(vt.T @ u.T))            # rotation only, never a mirror image
    rot = vt.T @ np.diag([1.0, d]) @ u.T
    fitted = pc @ rot.T + q.mean(axis=0)
    errors = np.linalg.norm(fitted - q, axis=1) * 1000
    return ({i: [round(float(x), 4), round(float(y), 4)] for i, (x, y) in zip(ids, fitted)},
            {i: float(e) for i, e in zip(ids, errors)})


def load_saved():
    """Earlier raw readings {id: [x, y, z]}, tilts {id: deg} and orientation, if any."""
    if not cfg.MARKERS_FILE.exists():
        return {}, {}, None
    data = json.loads(cfg.MARKERS_FILE.read_text())
    raw = data.get("readings") or data.get("measured") or data.get("markers") or {}
    if not isinstance(raw, dict):
        raw = {}
    return ({int(k): list(v) for k, v in raw.items()},
            {int(k): v for k, v in data.get("tilt_deg", {}).items()}, data.get("orientation"))


def save_markers(readings, tilts, orientation):
    layout = getattr(cfg, "MARKER_LAYOUT_MM", None)
    fitted = fit_layout(readings)[0] if layout else {i: readings[i][:2] for i in cfg.MARKER_IDS}
    data = {"saved": datetime.now().isoformat(timespec="minutes"), "robot_ip": robot_ip(), "units": "metres",
            "markers": {str(i): fitted[i] for i in cfg.MARKER_IDS},
            "readings": {str(i): readings[i] for i in cfg.MARKER_IDS},
            "tilt_deg": {str(i): tilts[i] for i in tilts},
            "orientation": orientation, "layout_fitted": bool(layout)}
    cfg.CALIB_DIR.mkdir(exist_ok=True)
    cfg.MARKERS_FILE.write_text(json.dumps(data, indent=2))


class MarkersWindow:
    def __init__(self, root):
        self.root = root
        self.robot = None
        self.learning = False
        root.title("Step 1.4 - Measure the markers")
        saved, self.tilts, self.orientation = load_saved()
        self.selected = tk.IntVar(value=cfg.MARKER_IDS[0])
        self.tested = {}

        # ---------------------------------------------------------- project progress (right side)
        progress = tk.LabelFrame(root, text="Project progress", padx=8, pady=6)
        progress.pack(side="right", fill="y", padx=(0, 10), pady=10)
        self.progress_labels = [tk.Label(progress, anchor="w", justify="left", font=("Arial", 10))
                                for _ in project_status(max_age=0)]
        for lbl in self.progress_labels:
            lbl.pack(fill="x", pady=2)

        # ---------------------------------------------------------- instruction
        self.step_label = tk.Label(root, font=("Arial", 14, "bold"), anchor="w", padx=10, pady=6)
        self.step_label.pack(fill="x")
        self.hint_label = tk.Label(root, font=("Arial", 10), anchor="w", padx=10, justify="left")
        self.hint_label.pack(fill="x")

        # ---------------------------------------------------------- robot
        bar = tk.LabelFrame(root, text="Robot", padx=6, pady=4)
        bar.pack(fill="x", padx=10, pady=4)
        self.connect_btn = tk.Button(bar, width=24, command=self.connect)
        self.connect_btn.grid(row=0, column=0, padx=3)
        self.learn_btn = tk.Button(bar, text="Learning mode: OFF", width=18, command=self.toggle_learning)
        self.learn_btn.grid(row=0, column=1, padx=3)
        self.read_btn = tk.Button(bar, text="Read robot position", width=18, command=self.read_position)
        self.read_btn.grid(row=0, column=2, padx=3)
        self.test_btn = tk.Button(bar, text="Test: go above marker", width=20, command=self.test_marker)
        self.test_btn.grid(row=0, column=3, padx=3)
        self.status = tk.Label(bar, text="Not connected: you can still type the values.", fg=GREY)
        self.status.grid(row=1, column=0, columnspan=4, sticky="w")

        # ---------------------------------------------------------- table
        table = tk.LabelFrame(root, text="Marker readings (metres): gripper centred over each marker, "
                                         "about 5 mm above it", padx=6, pady=4)
        table.pack(fill="x", padx=10, pady=4)
        for c, head in enumerate(["", "Marker", "x", "y", "z", "Tilt", "Distance from fit", "Test"]):
            tk.Label(table, text=head, font=("Arial", 10, "bold")).grid(row=0, column=c, padx=6)
        self.entries, self.tilt_labels, self.fit_labels, self.test_labels = {}, {}, {}, {}
        for r, mid in enumerate(cfg.MARKER_IDS, start=1):
            tk.Radiobutton(table, variable=self.selected, value=mid, command=self.refresh).grid(row=r, column=0)
            tk.Label(table, text=f"ID {mid}", font=("Arial", 11, "bold")).grid(row=r, column=1)
            value = saved.get(mid, [])
            for c, axis in enumerate("xyz"):
                e = tk.Entry(table, width=9, justify="right")
                if c < len(value) and value[c] is not None:
                    e.insert(0, f"{value[c]:g}")
                e.grid(row=r, column=2 + c, padx=2)
                e.bind("<KeyRelease>", lambda _e, m=mid: self.typed(m))
                e.bind("<FocusIn>", lambda _e, m=mid: (self.selected.set(m), self.refresh()))
                self.entries[(mid, axis)] = e
            self.tilt_labels[mid] = tk.Label(table, width=10)
            self.tilt_labels[mid].grid(row=r, column=5)
            self.fit_labels[mid] = tk.Label(table, width=18)
            self.fit_labels[mid].grid(row=r, column=6)
            self.test_labels[mid] = tk.Label(table, width=10)
            self.test_labels[mid].grid(row=r, column=7)

        # ---------------------------------------------------------- distances and buttons
        bottom = tk.Frame(root)
        bottom.pack(fill="both", expand=True, padx=10, pady=4)
        self.distances = tk.Label(bottom, font=("Courier", 10), justify="left", anchor="nw")
        self.distances.pack(side="left", fill="both", expand=True)
        side = tk.Frame(bottom)
        side.pack(side="left", padx=8)
        self.save_btn = tk.Button(side, text="Save", width=14, font=("Arial", 11, "bold"), command=self.save)
        self.save_btn.pack(pady=3)
        tk.Button(side, text="Quit", width=14, command=self.quit).pack(pady=3)
        root.protocol("WM_DELETE_WINDOW", self.quit)
        self.refresh()

    # ------------------------------------------------------------------ data
    def readings(self):
        """{id: [x, y, z]} for every marker whose x and y are numbers (z may be None)."""
        out = {}
        for mid in cfg.MARKER_IDS:
            vals = []
            for axis in "xyz":
                try:
                    vals.append(float(self.entries[(mid, axis)].get()))
                except ValueError:
                    vals.append(None)
            if vals[0] is not None and vals[1] is not None:
                out[mid] = vals
        return out

    def typed(self, mid):
        self.tilts.pop(mid, None)            # a typed value has no measured tilt
        self.tested.pop(mid, None)
        self.refresh()

    # ------------------------------------------------------------------ checks and display
    def refresh(self):
        readings = self.readings()
        complete = len(readings) == len(cfg.MARKER_IDS)
        layout = getattr(cfg, "MARKER_LAYOUT_MM", None)
        errors = fit_layout(readings)[1] if complete and layout else {}
        bad_fit = [m for m, e in errors.items() if e > cfg.MAX_FIT_ERROR_MM]
        bad_tilt = [m for m, t in self.tilts.items() if t > MAX_TILT_DEG]
        for mid in cfg.MARKER_IDS:
            t = self.tilts.get(mid)
            if t is None:
                self.tilt_labels[mid].config(text="typed" if mid in readings else "", fg=GREY)
            else:
                self.tilt_labels[mid].config(text=f"{t:.1f} deg", fg=RED if t > MAX_TILT_DEG else GREEN)
            e = errors.get(mid)
            if e is None:
                self.fit_labels[mid].config(text="")
            else:
                bad = e > cfg.MAX_FIT_ERROR_MM
                self.fit_labels[mid].config(text=f"{e:.1f} mm " + ("- read again" if bad else "OK"),
                                            fg=RED if bad else GREEN)
            result = self.tested.get(mid)
            self.test_labels[mid].config(text={"ok": "centred", "off": "off-centre"}.get(result, ""),
                                         fg=GREEN if result == "ok" else RED)
        lines = []
        if complete:
            fitted = fit_layout(readings)[0] if layout else {m: readings[m][:2] for m in readings}
            lines.append("Distances between marker centres (check them with a ruler in step 1.5):")
            for a, b in combinations(cfg.MARKER_IDS, 2):
                d = math.hypot(fitted[a][0] - fitted[b][0], fitted[a][1] - fitted[b][1]) * 1000
                lines.append(f"  marker {a} to {b}: {d:6.1f} mm")
            if errors and max(errors.values()) > 3 * cfg.MAX_FIT_ERROR_MM:
                lines.append("Very large differences: check that each reading is for the right marker ID,")
                lines.append("and that MARKER_LAYOUT_MM in config.py matches the platform.")
        self.distances.config(text="\n".join(lines))

        connected = self.robot is not None
        missing = [m for m in cfg.MARKER_IDS if m not in readings]
        if missing and not connected and not readings:
            step, hint, colour = ("STEP 1: connect to the robot",
                                  "Press Connect (or type the values for every marker if you measured them "
                                  "another way).", BLUE)
        elif missing:
            step, hint, colour = (f"STEP 2: read marker {missing[0]} ({len(missing)} still to read)",
                                  f"Select marker {missing[0]}, switch Learning mode ON, move the gripper so it "
                                  f"points straight down, centred over the marker, about 5 mm above it,\nthen "
                                  f"press 'Read robot position'.", BLUE)
        elif bad_tilt or bad_fit:
            redo = sorted(set(bad_tilt + bad_fit))
            step, hint, colour = (f"Read marker{'s' if len(redo) > 1 else ''} {', '.join(map(str, redo))} again",
                                  "Red = the gripper was tilted, or the reading is far from the fitted layout. "
                                  "Select only that marker and read it again;\nthe others are fine. 'Test' moves "
                                  "the gripper above a marker so you can see the reading.", RED)
        else:
            step, hint, colour = ("STEP 3: all markers OK - press Save, then Quit",
                                  "Then run step1_check_calibration.py to check the result with a ruler.", GREEN)
        self.step_label.config(text=step, fg=colour)
        self.hint_label.config(text=hint)
        self.save_btn.config(state="normal" if complete else "disabled")
        for b in (self.learn_btn, self.read_btn):
            b.config(state="normal" if connected else "disabled")
        sel = readings.get(self.selected.get())
        can_test = connected and sel is not None and sel[2] is not None and self.orientation
        self.test_btn.config(state="normal" if can_test else "disabled")
        self.connect_btn.config(state="disabled" if connected else "normal",
                                text=f"Connected to {self.robot.ip}" if connected else f"Connect to {robot_ip()}")
        for lbl, (step_id, title, done, detail, started) in zip(self.progress_labels, project_status()):
            here = is_current(step_id, title, "1.4")
            mark = "✔" if done else ("▪" if started else "☐")
            lbl.config(text=f"{mark}  {step_id}  {title}" + (f"\n        {detail}" if detail else ""),
                       fg=GREEN if done else (BLUE if here else GREY),
                       bg="#ffebc8" if here else lbl.master.cget("bg"),
                       font=("Arial", 10, "bold" if here else "normal"))

    # ------------------------------------------------------------------ robot
    def busy(self, text):
        self.status.config(text=text, fg=ORANGE)
        self.root.config(cursor="watch")
        self.root.update()

    def idle(self, text, fg=GREEN):
        self.status.config(text=text, fg=fg)
        self.root.config(cursor="")
        self.refresh()

    def connect(self):
        self.busy("Connecting and calibrating the robot (about 30 seconds)...")
        try:
            robot = NiryoRobot()
            robot.start()
        except Exception as e:                   # no robot, wrong IP, client not found
            self.idle(f"Could not connect: {e}", RED)
            return
        self.robot = robot
        self.idle(f"Connected to {robot.ip}. Switch Learning mode ON to move the arm by hand.")

    def toggle_learning(self):
        self.learning = not self.learning
        self.robot.learning_mode(self.learning)
        self.learn_btn.config(text=f"Learning mode: {'ON' if self.learning else 'OFF'}")
        self.idle("Learning mode ON: move the arm by hand." if self.learning else "Learning mode OFF.")

    def read_position(self):
        mid = self.selected.get()
        pose = list(self.robot.get_pose())
        for i, axis in enumerate("xyz"):
            e = self.entries[(mid, axis)]
            e.delete(0, "end")
            e.insert(0, f"{pose[i]:.4f}")
        self.tilts[mid] = round(tilt_from_vertical(pose), 1)
        self.orientation = [round(v, 4) for v in pose[3:6]]
        self.tested.pop(mid, None)
        tilt = self.tilts[mid]
        self.idle(f"Marker {mid}: x={pose[0]:.4f} y={pose[1]:.4f} z={pose[2]:.4f}, tilt {tilt:.1f} deg" +
                  ("  - too tilted: straighten the gripper and read again" if tilt > MAX_TILT_DEG else ""),
                  RED if tilt > MAX_TILT_DEG else GREEN)

    def test_marker(self):
        mid = self.selected.get()
        x, y, z = self.readings()[mid]
        z_hover = max(z + HOVER_M, cfg.Z_MIN + HOVER_M)
        z_travel = max(z_hover, 0.20)
        for value, limits, name in ((x, cfg.X_LIMITS, "x"), (y, cfg.Y_LIMITS, "y")):
            if not limits[0] <= value <= limits[1]:
                messagebox.showerror("Unsafe position", f"{name} = {value:.3f} is outside {limits}")
                return
        if not messagebox.askokcancel("Test", f"The robot moves straight up, then above marker {mid}, and down to "
                                      f"{HOVER_M * 1000:.0f} mm above your reading.\nKeep clear. Continue?"):
            return
        try:
            self.robot.learning_mode(False)
            self.learning = False
            self.learn_btn.config(text="Learning mode: OFF")
            self.busy(f"Moving above marker {mid}...")
            current = list(self.robot.get_pose())
            rpy = list(self.orientation)
            self.robot.move_pose(pose_from_args([current[0], current[1], z_travel] + rpy))   # straight up
            self.robot.move_pose(pose_from_args([x, y, z_travel] + rpy))
            self.robot.move_pose(pose_from_args([x, y, z_hover] + rpy))
            ok = messagebox.askyesno("Check", f"Is the gripper centred over marker {mid}?")
            self.tested[mid] = "ok" if ok else "off"
            self.robot.move_pose(pose_from_args([x, y, z_travel] + rpy))
        except Exception as e:
            self.idle(f"Move failed: {e}", RED)
            return
        self.idle(f"Marker {mid}: " + ("centred" if ok else "off-centre: read it again with the gripper centred"),
                  GREEN if ok else RED)

    # ------------------------------------------------------------------ save and quit
    def save(self):
        readings = self.readings()
        for mid in cfg.MARKER_IDS:
            if readings[mid][2] is None:
                readings[mid][2] = 0.0
        errors = fit_layout(readings)[1] if getattr(cfg, "MARKER_LAYOUT_MM", None) else {}
        if errors and max(errors.values()) > cfg.MAX_FIT_ERROR_MM and not messagebox.askyesno(
                "Save", "Some readings are more than 5 mm from the fitted layout. Save anyway?"):
            return
        save_markers(readings, self.tilts, self.orientation)
        project_status(max_age=0)
        self.refresh()
        print(f"Saved {cfg.MARKERS_FILE}")
        messagebox.showinfo("Saved", f"Saved to calibration/{cfg.MARKERS_FILE.name}.\n"
                                     f"Next: python step1_check_calibration.py")

    def quit(self):
        if self.robot is not None:
            try:
                self.busy("Moving to the sleep pose...")
                self.robot.finish()
            except Exception as e:
                print("Shutdown warning:", e)
        self.root.destroy()


def main():
    root = tk.Tk()
    MarkersWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
