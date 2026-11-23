"""
STEP 1a - Set up the camera so all four markers are found.

    python step1_tune_camera.py
    python step1_tune_camera.py --source dataset_raw     (saved images: detection only)

One window, no sliders: the program does the tuning. Follow the instruction
in the box at the top of the picture:

  1. Click once on the camera picture, then press  A  (auto-tune, about a minute).
     It locks the sharpest focus, locks the white balance so colours do not
     drift, and finds the best exposure and marker-detection settings.
  2. When all four markers are found, press  S  to save.
     The terminal prints "Saved ...camera_settings.json".
  3. Press  Q  to quit.

If a marker is never found, the box says which one and what to check (glare,
a light surround on all four sides, fully in view). Fix it and press A again.

Every later step uses the saved settings. Set them up BEFORE you collect your
dataset and keep them: if you change them later, save a new background (step 2,
key b) and collect new training images.
"""
import argparse
import json
from collections import deque
from datetime import datetime

import cv2
import numpy as np

import config as cfg
from aruco_view import DETECTION_DEFAULTS, MarkerDetector, load_detection_settings
from camera import WINDOWS, Camera, load_settings
from progress import with_progress
from ui import BLUE, GREEN, ORANGE, RED, instruction_box

WINDOW = "camera setup"
FOCUS_STEP = 5                     # most USB cameras only accept focus values in steps of 5
AUTO_EXPOSURES = list(range(-11, -2)) if WINDOWS else [50, 100, 150, 250, 400, 600]
AUTO_GRID = [{"gamma": g, "clahe": c, "threshold_constant": t, "window_max": w, "min_perimeter": 0.02}
             for g in (0.5, 0.7, 1.0, 1.5, 2.2) for c in (False, True) for t in (3, 7, 12) for w in (23, 53)]
GOOD_RATE = 27                     # a marker counts as found if seen in 27 of the last 30 frames



# ---------------------------------------------------------------- drawing
def show(frame, title, lines=(), colour=BLUE, centres=None, corners=None, ids=None):
    """The camera picture with markers labelled and an instruction box at the top."""
    shown = frame.copy()
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(shown, corners)
    for marker_id, (u, v) in (centres or {}).items():
        c = GREEN if marker_id in cfg.MARKER_IDS else RED
        org = (int(u) + 20, int(v) - 20)
        cv2.putText(shown, f"ID {marker_id}", org, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 7, cv2.LINE_AA)
        cv2.putText(shown, f"ID {marker_id}", org, cv2.FONT_HERSHEY_SIMPLEX, 1.2, c, 3, cv2.LINE_AA)
        cv2.circle(shown, (int(u), int(v)), 6, c, -1)
    shown = instruction_box(with_progress(shown, "1.3"), title, lines, colour)
    w = shown.shape[1]
    scale = min(1.0, 1280 / w)
    cv2.imshow(WINDOW, cv2.resize(shown, None, fx=scale, fy=scale))


def stopped():
    return cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"))


# ---------------------------------------------------------------- auto-tune parts
def sharpness(frame, centres=None):
    """How sharp the picture is (variance of the Laplacian; higher = sharper), measured
    between the markers if they are found, else in the middle of the picture."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if centres and len(centres) >= 2:
        pts = np.array(list(centres.values()))
        x0, y0 = np.maximum(pts.min(axis=0).astype(int) - 40, 0)
        x1, y1 = np.minimum(pts.max(axis=0).astype(int) + 40, (w, h))
        gray = gray[y0:y1, x0:x1]
    else:
        gray = gray[h // 4:3 * h // 4, w // 4:3 * w // 4]
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def auto_focus(cam, detector):
    """Autofocus off, try every focus value, lock the sharpest. None if focus cannot be changed."""
    centres = detector.detect(cam.read())[0]
    results = []
    for focus in range(0, 256, FOCUS_STEP):
        cam.apply({"autofocus": 0, "focus": focus})
        for _ in range(4):                       # let the lens move
            cam.read()
        frame = cam.read()
        results.append((sharpness(frame, centres) + sharpness(cam.read(), centres), focus))
        show(frame, "Auto-tune 1/3: finding the sharpest focus...",
             [f"focus {focus} of 255", "Please wait. Do not move the camera."], ORANGE)
        if stopped():
            return None
    values = [r[0] for r in results]
    if max(values) < 1.05 * min(values):
        return None                              # picture does not change: fixed-focus camera
    best = max(results)[1]
    cam.apply({"autofocus": 0, "focus": best})
    return best


def lock_white_balance(cam):
    """Let auto white balance settle, then lock it at that value. Returns the value or None."""
    cam.apply({"auto_wb": 1})
    for _ in range(30):
        cam.read()
    value = cam.get("wb_temperature")
    if value is None or not 2000 <= value <= 10000:
        return None
    value = int(round(value / 100.0) * 100)
    cam.apply({"auto_wb": 0, "wb_temperature": value})
    return value


def search_exposure_and_detection(cam, frames_per_test=3):
    """Try each exposure with each detection setting.
    Returns (best, best rate per marker over every setting) or (None, None) if stopped."""
    tester = MarkerDetector(DETECTION_DEFAULTS)
    best, best_ever = None, {i: 0.0 for i in cfg.MARKER_IDS}
    exposures = [None] + AUTO_EXPOSURES if cam.live else ["as is"]
    for k, exposure in enumerate(exposures):
        if cam.live:
            cam.apply({"auto_exposure": 1} if exposure is None else {"auto_exposure": 0, "exposure": exposure})
            for _ in range(10):                  # let the camera settle on the new exposure
                cam.read()
        frames = [cam.read() for _ in range(frames_per_test)]
        for n, det in enumerate(AUTO_GRID):
            tester.configure(det)
            per = {i: 0.0 for i in cfg.MARKER_IDS}
            for f in frames:
                found = tester.detect(f)[0]
                for i in cfg.MARKER_IDS:
                    per[i] += (i in found) / len(frames)
            for i in cfg.MARKER_IDS:
                best_ever[i] = max(best_ever[i], per[i])
            score = sum(per.values())
            simpler = -((det["gamma"] != 1.0) + det["clahe"] + (det["threshold_constant"] != 7))
            if best is None or (score, simpler) > (best["score"], best["simpler"]):
                best = {"score": score, "simpler": simpler, "exposure": None if exposure == "as is" else exposure,
                        "det": dict(det), "per": per}
            if n % 15 == 0:
                show(frames[-1], "Auto-tune 3/3: exposure and marker detection...",
                     [f"test {k + 1} of {len(exposures)}: best so far {best['score']:.1f} of 4 markers",
                      "Please wait about a minute. Press Q to stop."], ORANGE,
                     *MarkerDetector(best["det"]).detect(frames[-1]))
                if stopped():
                    return None, None
    return best, best_ever


def auto_tune(cam):
    """Run every part of the auto-tune. Returns (camera settings, detection settings, never-found markers)."""
    camera_settings = {}
    detector = MarkerDetector(DETECTION_DEFAULTS)
    if cam.live:
        focus = auto_focus(cam, detector)
        if focus is None:
            cam.apply({"autofocus": 1})
            print("Focus: the picture does not change with focus (fixed-focus camera). "
                  "If the lens has a ring, turn it by hand until the markers are sharp.")
        else:
            camera_settings.update({"autofocus": 0, "focus": focus})
            print(f"Focus locked at {focus}.")
        show(cam.read(), "Auto-tune 2/3: locking the colours...", ["Please wait."], ORANGE)
        cv2.waitKey(1)
        wb = lock_white_balance(cam)
        if wb is not None:
            camera_settings.update({"auto_wb": 0, "wb_temperature": wb})
            print(f"White balance locked at {wb} K.")
    best, best_ever = search_exposure_and_detection(cam)
    if best is None:
        return None
    if cam.live:
        if best["exposure"] is None:
            camera_settings["auto_exposure"] = 1
        else:
            camera_settings.update({"auto_exposure": 0, "exposure": best["exposure"]})
        cam.apply(camera_settings)
    detection = {k: best["det"][k] for k in DETECTION_DEFAULTS}
    never = [i for i in cfg.MARKER_IDS if best_ever[i] == 0]
    print(f"Exposure: {'automatic' if best['exposure'] is None else best['exposure']}. Detection: {detection}")
    return camera_settings, detection, never


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None, help="camera index, image, folder or video")
    args = parser.parse_args()

    cam = Camera(args.source)
    saved = load_settings()
    detector = MarkerDetector(load_detection_settings())
    cv2.namedWindow(WINDOW)
    print(__doc__)

    state = "start"            # start -> tuned -> saved
    camera_settings, detection, never = saved.get("camera", {}), None, []
    history = deque(maxlen=30)

    while True:
        frame = cam.read()
        centres, corners, ids = detector.detect(frame)
        history.append(set(centres))
        rate = {i: sum(i in h for h in history) for i in cfg.MARKER_IDS}
        all_good = len(history) == history.maxlen and all(r >= GOOD_RATE for r in rate.values())
        counts = "   ".join(f"ID {i}: {rate[i]}/{len(history)}" for i in cfg.MARKER_IDS)

        if state == "start":
            show(frame, "STEP 1: click this picture, then press A", [
                "A = auto-tune: focus, colours, exposure and marker detection (about a minute).",
                f"Markers found now:  {counts}"], BLUE, centres, corners, ids)
        elif state == "tuned" and all_good:
            show(frame, "STEP 2: all 4 markers found - press S to save", [
                f"Markers found:  {counts}",
                "S = save these settings      A = auto-tune again"], GREEN, centres, corners, ids)
        elif state == "tuned":
            missing = [i for i in cfg.MARKER_IDS if rate[i] < GOOD_RATE]
            advice = (f"Marker(s) {never} were never found with any setting." if never
                      else f"Marker(s) {missing} are not found reliably yet.")
            show(frame, "Not all markers found - fix, then press A again", [
                advice,
                "Check: glare on the marker, light surround on all four sides, fully in view.",
                f"Markers found:  {counts}      A = auto-tune again    S = save anyway"],
                 RED, centres, corners, ids)
        else:
            show(frame, "STEP 3: saved - press Q to quit", [
                f"Settings saved to calibration/{cfg.CAMERA_SETTINGS_FILE.name}. Every step now uses them.",
                "Next: python step1_measure_markers.py      A = auto-tune again"], GREEN, centres, corners, ids)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), ord("Q")):
            if state == "tuned":
                print("Quit without saving: the new settings are lost.")
            break
        elif key in (ord("a"), ord("A")):
            print("Auto-tune started...")
            result = auto_tune(cam)
            if result is None:
                print("Auto-tune stopped.")
                continue
            camera_settings, detection, never = result
            detector.configure(detection)
            history.clear()
            state = "tuned"
        elif key in (ord("s"), ord("S")) and state == "tuned":
            data = {"saved": datetime.now().isoformat(timespec="minutes"),
                    "camera": camera_settings if cam.live else saved.get("camera", {}),
                    "detection": detection}
            cfg.CALIB_DIR.mkdir(exist_ok=True)
            cfg.CAMERA_SETTINGS_FILE.write_text(json.dumps(data, indent=2))
            print(f"Saved {cfg.CAMERA_SETTINGS_FILE}")
            state = "saved"
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
