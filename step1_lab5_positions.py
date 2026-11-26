"""
STEP 1.1 - Enter and test your Lab 5 pick-up and drop positions.

    python step1_lab5_positions.py

This project builds on your Lab 5 program: the robot picks up and drops cubes
on squares A, B and C at three levels, at the positions you found in Lab 5.
In this window you:

  0. Choose your robot from the list. Every program then uses its IP address.
     If staff have saved positions for that robot, press "Load saved positions"
     to fill them in, then check and test them (steps 2 and 3).
  1. Type the gripper orientation (roll, pitch, yaw) and every pick-up and drop
     position (x, y, z in metres) from your Lab 5 program. Your program may use
     other names: only the numbers matter.
     Or (technician only, risk assessment RA001650/2: students keep 30 cm from a
     powered robot) connect, select a position, switch learning mode on, move the arm
     there by hand and press "Read robot position".
  2. Fix anything shown in red. Every value is checked as you type: x and y
     inside the safe area, z never below Z_MIN (config.py) so a typing error
     such as 0.009 instead of 0.093 cannot drive the gripper into the platform,
     and each level higher than the one below.
  3. Test: select a position and press "Test". The robot moves above it at the
     safe height, goes down, and asks you whether the gripper is in the right
     place, then goes back up. Tested positions turn green.
  4. Press "Save". The positions are saved to calibration/lab5_positions.json
     and used by step1_check_calibration.py and step4_sort_blocks.py.

Staff: to save a robot's positions for students to load, run
    python step1_lab5_positions.py --preset
enter and test the positions, and press "Save as preset for this robot"
(written to stations/<robot name>.json).

The safe travel height is not typed: it is your highest pick or drop position
plus SAFE_CLEARANCE_M (config.py), so the gripper always clears the stacks.
"""
import argparse
import math
import tkinter as tk
from tkinter import messagebox

import config as cfg
from progress import is_current, project_status
from robot import (LEVELS, SQUARES, NiryoRobot, check_lab5, check_pose, read_lab5_file, safe_height,
                   save_lab5_file)

RED_BG, ORANGE_BG, GREEN_BG, WHITE = "#f8c8c8", "#fbe0b0", "#cdeccd", "white"
KINDS = (("pick", "PICK-UP positions (gripper closes here)"), ("drop", "DROP positions (gripper opens here)"))


NO_ROBOT = "Choose your robot..."


def preset_path(name):
    return cfg.STATIONS_DIR / (name.replace(" ", "_").lower() + ".json")


class Lab5Window:
    def __init__(self, root, preset_mode=False):
        self.root = root
        self.preset_mode = preset_mode
        self.robot = None
        self.learning = False
        self.selected = tk.StringVar(value="pick A 1")
        root.title("Step 1.1 - Lab 5 pick-up and drop positions")

        # ---------------------------------------------------------- project progress (right side)
        progress = tk.LabelFrame(root, text="Project progress", padx=8, pady=6)
        progress.pack(side="right", fill="y", padx=(0, 10), pady=10)
        self.progress_labels = []
        for _ in project_status(max_age=0):
            lbl = tk.Label(progress, anchor="w", justify="left", font=("Arial", 10))
            lbl.pack(fill="x", pady=2)
            self.progress_labels.append(lbl)

        saved = read_lab5_file() or {}
        self.tested = dict(saved.get("tested", {}))
        self.robot_name = tk.StringVar(value=saved.get("robot") if saved.get("robot") in cfg.ROBOTS else NO_ROBOT)

        # ---------------------------------------------------------- instruction box
        self.step_label = tk.Label(root, font=("Arial", 14, "bold"), anchor="w", padx=10, pady=6)
        self.step_label.pack(fill="x")
        self.hint_label = tk.Label(root, font=("Arial", 10), anchor="w", padx=10, justify="left")
        self.hint_label.pack(fill="x")

        # ---------------------------------------------------------- robot
        bar = tk.LabelFrame(root, text="Robot", padx=6, pady=4)
        bar.pack(fill="x", padx=10, pady=4)
        choose = tk.Frame(bar)
        choose.grid(row=0, column=0, columnspan=6, sticky="w", pady=(0, 4))
        tk.Label(choose, text="Your robot:", font=("Arial", 10, "bold")).pack(side="left")
        self.robot_menu = tk.OptionMenu(choose, self.robot_name, *cfg.ROBOTS, command=lambda _v: self.refresh())
        self.robot_menu.config(width=12)
        self.robot_menu.pack(side="left", padx=4)
        self.load_btn = tk.Button(choose, text="Load saved positions", command=self.load_preset)
        self.load_btn.pack(side="left", padx=4)
        self.preset_label = tk.Label(choose, text="", fg="grey")
        self.preset_label.pack(side="left", padx=4)
        self.connect_btn = tk.Button(bar, text="Connect", width=22, command=self.connect)
        self.connect_btn.grid(row=1, column=0, padx=3)
        self.learn_btn = tk.Button(bar, text="Learning mode: OFF", width=18, command=self.toggle_learning,
                                   state="disabled")
        self.learn_btn.grid(row=1, column=1, padx=3)
        self.read_btn = tk.Button(bar, text="Read robot position", width=18, command=self.read_position,
                                  state="disabled")
        self.read_btn.grid(row=1, column=2, padx=3)
        self.test_btn = tk.Button(bar, text="Test selected position", width=20, command=self.test_position,
                                  state="disabled")
        self.test_btn.grid(row=1, column=3, padx=3)
        self.open_btn = tk.Button(bar, text="Open gripper", command=lambda: self.gripper(True), state="disabled")
        self.open_btn.grid(row=1, column=4, padx=3)
        self.close_btn = tk.Button(bar, text="Close gripper", command=lambda: self.gripper(False), state="disabled")
        self.close_btn.grid(row=1, column=5, padx=3)
        self.robot_status = tk.Label(bar, text="Not connected: you can still type and save the values.", fg="grey")
        self.robot_status.grid(row=2, column=0, columnspan=6, sticky="w")

        # ---------------------------------------------------------- orientation and safe height
        orient = tk.LabelFrame(root, text="Gripper orientation used for every move (radians)", padx=6, pady=4)
        orient.pack(fill="x", padx=10, pady=4)
        self.orientation = []
        start = saved.get("orientation", [])
        for i, name in enumerate(("roll", "pitch", "yaw")):
            tk.Label(orient, text=name).grid(row=0, column=2 * i, padx=(8, 2))
            e = tk.Entry(orient, width=9, justify="right")
            if i < len(start):
                e.insert(0, f"{start[i]:g}")
            e.grid(row=0, column=2 * i + 1)
            e.bind("<KeyRelease>", lambda _e: self.refresh())
            self.orientation.append(e)
        self.safe_label = tk.Label(orient, text="", fg="#204080")
        self.safe_label.grid(row=0, column=6, padx=20)

        # ---------------------------------------------------------- pick and drop tables
        tables = tk.Frame(root)
        tables.pack(padx=10, pady=4)
        self.entries, self.cells = {}, {}
        for col, (kind, title) in enumerate(KINDS):
            frame = tk.LabelFrame(tables, text=title, padx=6, pady=4)
            frame.grid(row=0, column=col, padx=5, sticky="n")
            for c, sq in enumerate(SQUARES):
                tk.Label(frame, text=f"Square {sq}", font=("Arial", 10, "bold")).grid(row=0, column=c + 1)
            for r, lv in enumerate(reversed(LEVELS)):                 # level 3 on top, like a stack
                tk.Label(frame, text=f"Level {lv}").grid(row=r + 1, column=0, padx=4)
                for c, sq in enumerate(SQUARES):
                    cell = tk.Frame(frame, bd=2, relief="groove", padx=3, pady=2)
                    cell.grid(row=r + 1, column=c + 1, padx=3, pady=3)
                    key = f"{kind} {sq} {lv}"
                    tk.Radiobutton(cell, text=f"{sq}{lv}", variable=self.selected, value=key,
                                   command=self.refresh).grid(row=0, column=0, columnspan=2, sticky="w")
                    p = saved.get(kind, {}).get(sq, {}).get(lv, [])
                    for i, axis in enumerate(("x", "y", "z")):
                        tk.Label(cell, text=axis).grid(row=i + 1, column=0)
                        e = tk.Entry(cell, width=8, justify="right")
                        if i < len(p) and p[i] is not None:
                            e.insert(0, f"{p[i]:g}")
                        e.grid(row=i + 1, column=1)
                        e.bind("<KeyRelease>", lambda _e: self.refresh())
                        e.bind("<FocusIn>", lambda _e, k=key: self.select(k))
                        self.entries[(kind, sq, lv, axis)] = e
                    self.cells[key] = cell

        # ---------------------------------------------------------- problems and save
        bottom = tk.Frame(root)
        bottom.pack(fill="both", expand=True, padx=10, pady=4)
        self.problems = tk.Text(bottom, height=7, width=110, font=("Courier", 9), wrap="word")
        self.problems.pack(side="left", fill="both", expand=True)
        self.problems.tag_config("error", foreground="#b00000")
        self.problems.tag_config("warning", foreground="#a06000")
        self.problems.tag_config("ok", foreground="#006000")
        side = tk.Frame(bottom)
        side.pack(side="left", padx=8)
        self.save_btn = tk.Button(side, text="Save", width=14, font=("Arial", 11, "bold"), command=self.save)
        self.save_btn.pack(pady=3)
        if preset_mode:
            self.preset_btn = tk.Button(side, text="Save as preset\nfor this robot", width=14,
                                        command=self.save_preset)
            self.preset_btn.pack(pady=3)
        tk.Button(side, text="Quit", width=14, command=self.quit).pack(pady=3)
        root.protocol("WM_DELETE_WINDOW", self.quit)
        self.refresh()

    # ------------------------------------------------------------------ data
    def values(self):
        """Everything typed, with numbers where the text is a number and None elsewhere."""
        def num(entry):
            try:
                v = float(entry.get().strip())
                return v if math.isfinite(v) else None
            except ValueError:
                return None
        name = self.robot_name.get()
        data = {"orientation": [num(e) for e in self.orientation], "tested": self.tested,
                "robot": name if name in cfg.ROBOTS else None, "robot_ip": cfg.ROBOTS.get(name)}
        for kind, _ in KINDS:
            data[kind] = {sq: {lv: [num(self.entries[(kind, sq, lv, a)]) for a in ("x", "y", "z")]
                               for lv in LEVELS} for sq in SQUARES}
        return data

    def selected_parts(self):
        kind, sq, lv = self.selected.get().split()
        return kind, sq, int(lv)

    def select(self, key):
        self.selected.set(key)
        self.refresh()

    # ------------------------------------------------------------------ checking and colours
    def refresh(self):
        data = self.values()
        errors, warnings = check_lab5(data)
        bad = {w for w, _ in errors if w}
        warn = {w for w, _ in warnings if w}
        for where, e in self.entries.items():
            e.config(bg=RED_BG if where in bad and e.get().strip() else ORANGE_BG if where in warn else WHITE)
        for i, e in enumerate(self.orientation):
            w = ("orientation", i)
            e.config(bg=RED_BG if w in bad and e.get().strip() else ORANGE_BG if w in warn else WHITE)
        for key, cell in self.cells.items():
            selected = key == self.selected.get()
            colour = {"ok": GREEN_BG, "wrong": RED_BG}.get(self.tested.get(key), "#f0f0f0")
            cell.config(bg=colour, relief="solid" if selected else "groove", bd=3 if selected else 2)
            for child in cell.winfo_children():
                if not isinstance(child, tk.Entry):
                    child.config(bg=colour)

        empty = sum(1 for e in list(self.entries.values()) + self.orientation if not e.get().strip())
        typed_errors = [m for w, m in errors if not (w and not self.field_text(w))]
        self.problems.config(state="normal")
        self.problems.delete("1.0", "end")
        for m in typed_errors:
            self.problems.insert("end", "ERROR    " + m + "\n", "error")
        for _, m in warnings:
            self.problems.insert("end", "CHECK    " + m + "\n", "warning")
        if not errors and not warnings:
            self.problems.insert("end", "All values pass the checks.\n", "ok")
        self.problems.config(state="disabled")

        if not errors:
            z = safe_height(data)
            self.safe_label.config(text=f"Safe travel height (automatic): {z:.3f} m = highest position "
                                        f"+ {cfg.SAFE_CLEARANCE_M:.3f} m")
        else:
            self.safe_label.config(text="Safe travel height: shown when every value is valid")

        n_tested = sum(1 for v in self.tested.values() if v == "ok")
        connected = self.robot is not None
        chosen = self.robot_name.get() in cfg.ROBOTS
        if not chosen:
            step, hint, colour = ("STEP 1: choose your robot",
                                  "Pick it from the 'Your robot' list: the number is on the label on your "
                                  "platform.", "#204080")
        elif empty:
            step, hint, colour = (f"STEP 2: enter your Lab 5 values ({empty} boxes still empty)",
                                  "Type each value from your Lab 5 program (metres and radians), or connect, "
                                  "select a position, and ask the technician to switch learning mode ON,\nmove the arm there by hand and "
                                  "press 'Read robot position' (keep 30 cm from the robot yourself).\nIf staff saved positions for your robot, press 'Load saved positions'.",
                                  "#204080")
        elif errors:
            step, hint, colour = (f"Fix the {len(errors)} value(s) shown in red",
                                  "The list at the bottom says what is wrong with each one.", "#b00000")
        elif n_tested < len(self.cells):
            step, hint, colour = (f"STEP 3: test each position ({n_tested} of {len(self.cells)} tested OK)",
                                  ("Select a position and press 'Test selected position'. Keep clear of the robot."
                                   if connected else "Connect to the robot to test, or press Save to test later."),
                                  "#a06000")
        else:
            step, hint, colour = ("STEP 4: all positions tested - press Save, then Quit",
                                  "The positions are saved to calibration/lab5_positions.json.", "#006000")
        self.step_label.config(text=step, fg=colour)
        self.hint_label.config(text=hint)
        self.save_btn.config(state="disabled" if errors or not chosen else "normal")
        if self.preset_mode:
            self.preset_btn.config(state="disabled" if errors or not chosen else "normal")
        has_preset = chosen and preset_path(self.robot_name.get()).exists()
        self.load_btn.config(state="normal" if has_preset else "disabled")
        self.preset_label.config(text="" if not chosen else
                                 f"IP {cfg.ROBOTS[self.robot_name.get()]}" +
                                 ("" if has_preset else "   (no saved positions for this robot)"))
        self.connect_btn.config(state="disabled" if connected or not chosen else "normal",
                                text=f"Connected to {self.robot.ip}" if connected else "Connect")
        self.robot_menu.config(state="disabled" if connected else "normal")
        state = "normal" if connected else "disabled"
        for b in (self.learn_btn, self.read_btn, self.open_btn, self.close_btn):
            b.config(state=state)
        self.test_btn.config(state="normal" if connected and not errors else "disabled")
        self.show_progress()
        return errors

    def show_progress(self):
        for lbl, (step, title, done, detail, started) in zip(self.progress_labels, project_status()):
            here = is_current(step, title, "1.1")
            mark = "\u2714" if done else ("\u25aa" if started else "\u2610")
            lbl.config(text=f"{mark}  {step}  {title}" + (f"\n        {detail}" if detail else ""),
                       fg="#006000" if done else ("#204080" if here else "#707070"),
                       bg="#ffebc8" if here else lbl.master.cget("bg"),
                       font=("Arial", 10, "bold" if here else "normal"))

    def field_text(self, where):
        if where[0] == "orientation":
            return self.orientation[where[1]].get().strip()
        return self.entries[where].get().strip()

    # ------------------------------------------------------------------ robot
    def busy(self, text):
        self.robot_status.config(text=text, fg="#a06000")
        self.root.config(cursor="watch")
        self.root.update()

    def idle(self, text, fg="#006000"):
        self.robot_status.config(text=text, fg=fg)
        self.root.config(cursor="")
        self.refresh()

    def connect(self):
        if self.robot is not None:
            return
        self.busy("Connecting and calibrating the robot (about 30 seconds)...")
        try:
            robot = NiryoRobot(cfg.ROBOTS[self.robot_name.get()])
            robot.start()
        except Exception as e:                   # no robot, wrong IP, client not found
            self.idle(f"Could not connect: {e}", "#b00000")
            return
        self.robot = robot
        self.robot.learning_mode(False)
        self.learning = False
        self.idle(f"Connected to {self.robot_name.get()} ({robot.ip}).")

    def toggle_learning(self):
        self.learning = not self.learning
        self.robot.learning_mode(self.learning)
        self.learn_btn.config(text=f"Learning mode: {'ON' if self.learning else 'OFF'}")
        self.idle("Learning mode ON: the technician moves the arm by hand, then press 'Read robot position'." if self.learning
                  else "Learning mode OFF: the motors hold the arm.")

    def read_position(self):
        kind, sq, lv = self.selected_parts()
        pose = self.robot.get_pose()
        for i, axis in enumerate(("x", "y", "z")):
            e = self.entries[(kind, sq, lv, axis)]
            e.delete(0, "end")
            e.insert(0, f"{pose[i]:.3f}")
        if not any(e.get().strip() for e in self.orientation):
            for i, e in enumerate(self.orientation):
                e.insert(0, f"{pose[3 + i]:.3f}")
        self.tested.pop(self.selected.get(), None)
        self.idle(f"Read {kind} {sq}{lv}: x={pose[0]:.3f} y={pose[1]:.3f} z={pose[2]:.3f}")

    def gripper(self, open_it):
        (self.robot.open_gripper if open_it else self.robot.close_gripper)()

    def test_position(self):
        if self.refresh():
            return
        data = self.values()
        kind, sq, lv = self.selected_parts()
        x, y, z = data[kind][sq][lv]
        rpy = data["orientation"]
        z_safe = safe_height(data)
        poses = {"pick": data["pick"], "drop": data["drop"]}
        try:
            for pose in ([x, y, z_safe] + rpy, [x, y, z] + rpy):
                check_pose(pose, poses)
        except ValueError as e:
            messagebox.showerror("Unsafe position", str(e))
            return
        if not messagebox.askokcancel("Test move", f"The robot will move above {kind} {sq}{lv} at "
                                      f"z={z_safe:.3f} m, then down to\nx={x:.3f}  y={y:.3f}  z={z:.3f}\n\n"
                                      f"Keep clear of the robot. Continue?"):
            return
        key = self.selected.get()
        try:
            self.robot.learning_mode(False)          # the motors must hold the arm to move it
            self.learning = False
            self.learn_btn.config(text="Learning mode: OFF")
            self.busy(f"Moving to {kind} {sq}{lv}...")
            self.robot.move_pose([x, y, z_safe] + rpy)
            self.robot.move_pose([x, y, z] + rpy)
            ok = messagebox.askyesno("Check the position",
                                     f"Is the gripper where it should be for {kind} {sq} level {lv}?\n"
                                     "(Yes = position OK, No = it needs changing)")
            self.tested[key] = "ok" if ok else "wrong"
            self.busy("Moving back up...")
            self.robot.move_pose([x, y, z_safe] + rpy)
        except Exception as e:
            self.idle(f"Move failed: {e}", "#b00000")
            return
        self.idle(f"{kind} {sq}{lv}: " + ("OK" if ok else "marked wrong: change the value, or read it again "
                                                          "with the arm in the right place"),
                  "#006000" if ok else "#b00000")

    # ------------------------------------------------------------------ presets
    def load_preset(self):
        preset = read_lab5_file(preset_path(self.robot_name.get()))
        if preset is None:
            return
        typed = any(e.get().strip() for e in self.entries.values())
        if typed and not messagebox.askyesno("Load saved positions",
                                             "Replace the values in the window with the saved positions "
                                             f"for {self.robot_name.get()}?"):
            return
        for i, e in enumerate(self.orientation):
            e.delete(0, "end")
            if i < len(preset["orientation"]):
                e.insert(0, f"{preset['orientation'][i]:g}")
        for (kind, sq, lv, axis), e in self.entries.items():
            e.delete(0, "end")
            p = preset[kind][sq].get(lv, [])
            i = "xyz".index(axis)
            if i < len(p) and p[i] is not None:
                e.insert(0, f"{p[i]:g}")
        self.tested.clear()                      # loaded values must be tested on YOUR robot
        self.idle(f"Loaded the saved positions for {self.robot_name.get()}. Check them and test each one.",
                  "#204080")

    def save_preset(self):
        if self.refresh():
            return
        path = preset_path(self.robot_name.get())
        if path.exists() and not messagebox.askyesno("Save preset", f"Replace {path.name}?"):
            return
        data = self.values()
        data["tested"] = {}
        save_lab5_file(data, path)
        print(f"Saved preset {path}")
        messagebox.showinfo("Preset saved", f"Saved stations/{path.name}.\nStudents choosing "
                                            f"{self.robot_name.get()} can now load it.")
        self.refresh()

    # ------------------------------------------------------------------ save and quit
    def save(self):
        if self.refresh():
            return
        data = self.values()
        untested = sum(1 for k in self.cells if self.tested.get(k) != "ok")
        if untested and not messagebox.askyesno("Save", f"{untested} position(s) have not been tested OK "
                                                        f"on the robot.\nSave anyway?"):
            return
        save_lab5_file(data)
        print(f"Saved {cfg.LAB5_FILE}")
        project_status(max_age=0)                 # tick 1.1 now
        self.show_progress()
        messagebox.showinfo("Saved", f"Saved to calibration/{cfg.LAB5_FILE.name}.\n"
                                     f"Safe travel height: {safe_height(data):.3f} m")

    def quit(self):
        if self.robot is not None:
            try:
                self.busy("Moving to the sleep pose...")
                self.robot.finish()
            except Exception as e:
                print("Shutdown warning:", e)
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", action="store_true", help="staff: show 'Save as preset for this robot'")
    args = parser.parse_args()
    root = tk.Tk()
    Lab5Window(root, preset_mode=args.preset)
    root.mainloop()


if __name__ == "__main__":
    main()
