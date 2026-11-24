"""
Project progress: which steps are done, worked out from the files each step saves.

    python progress.py          shows the checklist in a window and in the terminal

Every camera window shows the same checklist down its right side, and the
terminal-only programs print it when they start and finish. A step is ticked when
its result exists; the step you are running now is highlighted.
"""
import csv
import json
import time

import cv2
import numpy as np

import config as cfg

TRIALS_TARGET = 20

_cache = {"time": 0.0, "status": None}


def _trials():
    path = cfg.RESULTS_DIR / "trials.csv"
    if not path.exists():
        return 0, 0
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    dry = sum(1 for r in rows if r.get("dry_run") == "True")
    return dry, len(rows) - dry


def _calibration():
    """Done when step1_check_calibration.py confirmed the calibration (A, B, C all within 5 mm)
    and the Lab 5 positions have not been changed since."""
    if not cfg.MARKERS_FILE.exists():
        return False, "", False
    try:
        check = json.loads(cfg.MARKERS_FILE.read_text()).get("check")
    except ValueError:
        return False, "", False
    if not check:
        return False, "not confirmed yet", True
    if cfg.LAB5_FILE.exists() and cfg.LAB5_FILE.stat().st_mtime > cfg.MARKERS_FILE.stat().st_mtime:
        return False, "Lab 5 changed: calibrate again", True
    worst = max(check.get("off_mm", {0: 0}).values())
    return bool(check.get("passed")), f"worst {worst:.1f} mm", True


def _dataset():
    """Ticked when the sorted colours and "other" reach the MINIMUM (the one training needs).
    The detail shows the lowest of those against the TARGET."""
    from colour_model import below_minimum, dataset_counts, required_classes
    counts = dataset_counts()
    started = any(n for n, _ in counts.values())
    lowest = min(required_classes(), key=lambda c: counts[c][0])
    detail = f"lowest: {lowest} {counts[lowest][0]}/{cfg.IMAGES_PER_CLASS}"
    return not below_minimum(counts), detail, started


def project_status(max_age=2.0):
    """[(step id, title, done, detail, started)], re-read from the files at most every max_age seconds."""
    if _cache["status"] is not None and time.time() - _cache["time"] < max_age:
        return _cache["status"]
    status = []
    status.append(("1.1", "Lab 5 positions", cfg.LAB5_FILE.exists(), "", cfg.LAB5_FILE.exists()))
    status.append(("1.3", "Camera tuned", cfg.CAMERA_SETTINGS_FILE.exists(), "", False))
    done, detail, started = _calibration()
    status.append(("1.5", "Camera calibrated", done, detail, started))
    done, detail, started = _dataset()
    status.append(("2", "Dataset collected", done, detail if started else "", started))
    status.append(("3a", "Model trained", cfg.MODEL_FILE.exists(), "", False))
    status.append(("3b", "Baseline rule", (cfg.RESULTS_DIR / "baseline_report.txt").exists(), "", False))
    live = list(cfg.RESULTS_DIR.glob("live_*_camera.png")) if cfg.RESULTS_DIR.exists() else []
    status.append(("3c", "Live test", bool(live), "screenshot saved" if live else "", False))
    status.append(("3d", "Experiments", (cfg.RESULTS_DIR / "experiments.txt").exists(), "", False))
    dry, real = _trials()
    status.append(("4", "Dry run", dry > 0, "", dry > 0))
    status.append(("4", "Robot trials", real >= TRIALS_TARGET, f"{real}/{TRIALS_TARGET} cubes" if real else "",
                   real > 0))
    _cache.update(time=time.time(), status=status)
    return status


def is_current(step, title, current):
    """current is a step id ("2") or "id:title" for the two step-4 lines ("4:Robot trials")."""
    return current is not None and (current == step or current == f"{step}:{title}")


def print_progress(current=None):
    print("\nProject progress:")
    for step, title, done, detail, _ in project_status(max_age=0):
        mark = "[x]" if done else "[ ]"
        here = "  <- this step" if is_current(step, title, current) else ""
        print(f"  {mark} {step:4s} {title:20s} {detail}{here}")
    print()


def _tick(img, x, y, size, colour):
    pts = np.array([[x + size * 0.15, y + size * 0.55], [x + size * 0.42, y + size * 0.82],
                    [x + size * 0.9, y + size * 0.2]], np.int32)
    cv2.polylines(img, [pts], False, colour, 3, cv2.LINE_AA)


def progress_panel(height, current=None, width=270):
    """A white column listing the steps: ticked when done, highlighted for the current step."""
    panel = np.full((height, width, 3), 255, np.uint8)
    cv2.line(panel, (0, 0), (0, height), (200, 200, 200), 2)
    cv2.putText(panel, "PROJECT PROGRESS", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 2, cv2.LINE_AA)
    status = project_status()
    row_h = max(36, min(60, (height - 50) // len(status)))
    for i, (step, title, done, detail, started) in enumerate(status):
        y = 46 + i * row_h
        if is_current(step, title, current):
            cv2.rectangle(panel, (4, y - 2), (width - 4, y + row_h - 4), (255, 235, 200), -1)
            cv2.rectangle(panel, (4, y - 2), (width - 4, y + row_h - 4), (200, 80, 0), 2)
        box = (12, y + 4, 20)
        cv2.rectangle(panel, (box[0], box[1]), (box[0] + box[2], box[1] + box[2]),
                      (0, 150, 0) if done else (150, 150, 150), 2)
        if done:
            _tick(panel, box[0], box[1], box[2], (0, 150, 0))
        elif started:
            cv2.rectangle(panel, (box[0] + 6, box[1] + 6), (box[0] + 14, box[1] + 14), (0, 140, 255), -1)
        ink = (40, 40, 40) if done or is_current(step, title, current) else (120, 120, 120)
        cv2.putText(panel, f"{step}  {title}", (42, y + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.62, ink, 2, cv2.LINE_AA)
        if detail and row_h >= 40:
            cv2.putText(panel, detail, (42, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (110, 110, 110), 1,
                        cv2.LINE_AA)
    return panel


def with_progress(image, current=None):
    """The image with the progress column on its right."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    width = max(280, image.shape[1] // 4)
    return np.hstack([image, progress_panel(image.shape[0], current, width)])


def main():
    print_progress()
    img = progress_panel(560, None, 340)
    cv2.imshow("project progress", img)
    print("Press any key in the window to close it.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
