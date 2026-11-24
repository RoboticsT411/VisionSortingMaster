"""
Camera input.

Camera(source) reads frames from:
  - a USB camera:          Camera(0)
  - a folder of images:    Camera("dataset_raw")      (loops over the images)
  - one image or a video:  Camera("frame.png") / Camera("clip.mp4")

Using saved images lets you test your code at home without the robot.

A USB camera is set up with the values saved by step1_tune_camera.py
(calibration/camera_settings.json), so every step sees the same image.
"""
import json
import platform
from pathlib import Path

import cv2

import config as cfg

IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".bmp"}
WINDOWS = platform.system() == "Windows"

# Camera controls, in the order they must be applied: each "auto" switch comes
# before the value it controls, or the camera ignores the value.
CONTROLS = [
    ("auto_exposure", cv2.CAP_PROP_AUTO_EXPOSURE),
    ("exposure", cv2.CAP_PROP_EXPOSURE),
    ("auto_wb", cv2.CAP_PROP_AUTO_WB),
    ("wb_temperature", cv2.CAP_PROP_WB_TEMPERATURE),
    ("autofocus", cv2.CAP_PROP_AUTOFOCUS),
    ("focus", cv2.CAP_PROP_FOCUS),
    ("brightness", cv2.CAP_PROP_BRIGHTNESS),
    ("contrast", cv2.CAP_PROP_CONTRAST),
    ("saturation", cv2.CAP_PROP_SATURATION),
    ("gain", cv2.CAP_PROP_GAIN),
]
SWITCHES = {"auto_exposure", "auto_wb", "autofocus"}
PAIRED_SWITCH = {"exposure": "auto_exposure", "wb_temperature": "auto_wb", "focus": "autofocus"}


def load_settings():
    """Saved camera and marker-detection settings, or {} if none saved yet."""
    if cfg.CAMERA_SETTINGS_FILE.exists():
        return json.loads(cfg.CAMERA_SETTINGS_FILE.read_text())
    return {}


def switch_value(name, on):
    """The number a camera driver expects for an auto switch."""
    if name == "auto_exposure" and not WINDOWS:
        return 3 if on else 1          # Linux (V4L2) uses 3 = auto, 1 = manual
    return 1 if on else 0              # Windows (DirectShow) uses 1 = auto, 0 = manual


def camera_not_found_message(index, backend):
    """Which camera numbers can be opened, so the message says what to set."""
    working = []
    quiet = getattr(getattr(cv2, "utils", None), "logging", None)     # hide OpenCV's own warnings
    level = quiet.getLogLevel() if quiet else None
    if quiet:
        quiet.setLogLevel(quiet.LOG_LEVEL_SILENT)
    try:
        for i in range(5):
            if i == index:
                continue
            cap = cv2.VideoCapture(i, backend)
            if cap.isOpened() and cap.read()[0]:
                working.append(i)
            cap.release()
    finally:
        if quiet:
            quiet.setLogLevel(level)
    lines = [f"\nCannot open camera {index}.",
             "  1. Stop every other program that uses the camera: other PyCharm runs (red square),",
             "     Teams, the Windows Camera app. Only one program can use the camera at a time.",
             "  2. Check the USB cable, unplug and plug it in again, wait 5 seconds."]
    if working:
        lines.append(f"  3. Cameras that work on this PC: {working}. Set CAMERA_INDEX = {working[0]} in config.py "
                     f"(on a laptop, one of them is the built-in camera: try each).")
    else:
        lines.append("  3. No other camera works either: the camera is not connected, or another program has it.")
    return "\n".join(lines)


class Camera:
    def __init__(self, source=None, apply_saved=True):
        if source is None:
            source = cfg.CAMERA_INDEX
        self.images = None
        self.index = 0
        self.cap = None
        self.live = False

        if isinstance(source, str) and source.isdigit():
            source = int(source)

        if isinstance(source, int):
            # On Windows DirectShow opens USB cameras faster and more reliably.
            backend = cv2.CAP_DSHOW if WINDOWS else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(source, backend)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.FRAME_HEIGHT)
            if not self.cap.isOpened():
                self.cap.release()
                raise SystemExit(camera_not_found_message(source, backend))
            self.live = True
            if apply_saved:
                saved = load_settings().get("camera", {})
                if saved:
                    self.apply(saved)
                    # Some drivers ignore focus and exposure sent while the camera is
                    # starting: send everything again once the first frames have arrived.
                    for _ in range(5):
                        self.cap.read()
                    self.apply(saved)
                    focus = ("auto" if saved.get("autofocus", 1) else f"locked at {saved.get('focus')}")
                    print(f"Camera settings loaded from {cfg.CAMERA_SETTINGS_FILE.name} (focus {focus})")
        else:
            path = Path(source)
            if path.is_dir():
                self.images = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_TYPES)
                if not self.images:
                    raise RuntimeError(f"No images found in {path}")
            elif path.suffix.lower() in IMAGE_TYPES:
                self.images = [path]
            else:
                self.cap = cv2.VideoCapture(str(path))
                if not self.cap.isOpened():
                    raise RuntimeError(f"Cannot open video {path}")

    # ------------------------------------------------------------ camera controls
    def apply(self, values):
        """Send camera controls, e.g. {"auto_exposure": 0, "exposure": -6}.

        Exposure, white balance and focus are sent only when their auto switch is
        explicitly off: on many cameras, setting one of these values quietly
        switches auto mode off (a fixed, possibly wrong focus makes the image blur)."""
        if not self.live:
            return
        if not hasattr(self, "_auto"):
            self._auto = {}
        for name, prop in CONTROLS:
            if name not in values:
                continue
            value = values[name]
            if name in SWITCHES:
                self._auto[name] = bool(value)
                value = switch_value(name, bool(value))
            elif name in PAIRED_SWITCH and self._auto.get(PAIRED_SWITCH[name]) is not False:
                # Exposure, white balance and focus are only sent when their auto mode
                # has been switched OFF on purpose; otherwise the camera keeps its auto mode.
                continue
            self.cap.set(prop, float(value))

    def get(self, name):
        """Current value of a camera control as the driver reports it (None if unknown)."""
        if not self.live:
            return None
        prop = dict(CONTROLS)[name]
        value = self.cap.get(prop)
        return None if value == -1 else value

    def open_driver_dialog(self):
        """Windows only: the camera driver's own settings window."""
        if self.live and WINDOWS:
            self.cap.set(cv2.CAP_PROP_SETTINGS, 1)
            return True
        return False

    # ------------------------------------------------------------ frames
    def read(self):
        """Return the next frame (BGR image)."""
        if self.images is not None:
            frame = cv2.imread(str(self.images[self.index % len(self.images)]))
            self.index += 1
            return frame
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Camera did not return a frame.")
        return frame

    def read_fresh(self, skip=5):
        """Throw away buffered frames so the image shows the scene as it is now."""
        if self.images is not None:
            return self.read()
        for _ in range(skip):
            self.cap.grab()
        return self.read()

    def release(self):
        if self.cap is not None:
            self.cap.release()
