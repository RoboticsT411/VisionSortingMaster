"""
Settings for the Project 1 vision sorting code.

Change the values in this file to match your robot station.
Values that you MEASURE are not typed here: the marker positions are written
to the calibration/ folder by step1_measure_markers.py, and your Lab 5 pick and
drop positions are entered with step1_lab5_positions.py.
"""
from pathlib import Path

# ---------------------------------------------------------------- folders
BASE_DIR = Path(__file__).resolve().parent
CALIB_DIR = BASE_DIR / "calibration"
DATA_DIR = BASE_DIR / "dataset"          # dataset/<class name>/*.png
RAW_DIR = BASE_DIR / "dataset_raw"       # full camera frames (optional)
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MARKERS_FILE = CALIB_DIR / "markers.json"        # written by step1_measure_markers.py
BACKGROUND_FILE = CALIB_DIR / "background.png"   # empty pick zone, top-down view
CAMERA_SETTINGS_FILE = CALIB_DIR / "camera_settings.json"  # written by step1_tune_camera.py
MODEL_FILE = MODEL_DIR / "colour_model.joblib"   # written by step3_train.py
SPLIT_FILE = MODEL_DIR / "split.json"            # which images are train/val/test
LAB5_FILE = CALIB_DIR / "lab5_positions.json"    # written by step1_lab5_positions.py

# ---------------------------------------------------------------- robot
# The lab robots. Choose yours in step1_lab5_positions.py: every program then
# uses that robot's IP address. The address is also on the label on your platform.
ROBOTS = {
    "Robot 102": "192.168.1.102",
    "Robot 103": "192.168.1.103",
    "Robot 104": "192.168.1.104",
    "Robot 105": "192.168.1.105",
    "Robot 106": "192.168.1.106",
    "Robot 109": "192.168.1.109",
}
# Used only until you have chosen your robot in step1_lab5_positions.py.
# Use "10.10.10.10" if you connect to the robot's own Wi-Fi hotspot.
ROBOT_IP = "192.168.1.109"
# Saved positions for each robot, prepared by staff: stations/<robot name>.json
STATIONS_DIR = BASE_DIR / "stations"
GRIPPER_SPEED = 1000
SLEEP_JOINTS = [0.0, 0.55, -1.2, 0.0, 0.0, 0.0]
# Arm pose used while the camera takes a picture. It must keep the arm out of
# the camera's view of the pick zone. The sleep pose works on most stations;
# check it in the live view (step 3) and change it if the arm hides the zone.
OBSERVE_JOINTS = SLEEP_JOINTS

# ---------------------------------------------------------------- camera
CAMERA_INDEX = 0          # 0 = first USB camera; try 1 if you get the laptop camera
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# ---------------------------------------------------------------- ArUco markers
ARUCO_DICT = "DICT_4X4_50"
MARKER_IDS = [0, 1, 2, 3]  # one marker at each corner of the platform area the camera watches
MARKER_SIZE_MM = 24        # side of the black marker square, mm

# Known marker layout of the platform (mm, measured with a ruler, centre to centre),
# the same on every station: x = towards the robot's front, y = to the robot's left.
# step1_measure_markers.py fits this exact shape to the robot readings (position and
# rotation only), which averages out small errors in each reading.
# Set to None to use the robot readings exactly as measured.
MARKER_LAYOUT_MM = {
    3: (0, 0),        # near the robot, on its right
    2: (0, 230),      # near the robot, on its left
    0: (140, 0),      # far from the robot, on its right
    1: (140, 230),    # far from the robot, on its left
}
MAX_FIT_ERROR_MM = 5       # a reading further than this from the fitted layout is reported

# ---------------------------------------------------------------- camera position (parallax)
# Where the camera lens is: robot x, y (metres) and its height above the PLATFORM (metres).
# Measure it with a tape measure, e.g. (0.42, 0.0, 0.55). Needed for the parallax correction.
CAMERA_POSITION_M = None
USE_PARALLAX_CORRECTION = False   # True once correct_parallax() in mapping.py is written
PARALLAX_HEIGHT_MM = 25           # height of the point whose position is corrected (top of a cube)

# ---------------------------------------------------------------- top-down view
PX_PER_MM = 2.0            # resolution of the top-down image (2 px = 1 mm)
EDGE_MARGIN_MM = 10        # cubes closer than this to the zone edge are ignored
KEEP_OUT_RADIUS_MM = 30    # no picking within this distance of squares A, B and C

# ---------------------------------------------------------------- cubes
CUBE_SIZE_MM = 25          # measure one of your cubes with a ruler and update
# Every cube colour you have gets its own class, plus "other" for things that are
# not cubes (paper, tape, a pen lid...). In step 2 each class has a number key.
CLASS_KEYS = {
    "1": "red",
    "2": "orange",
    "3": "yellow",
    "4": "green",
    "5": "blue",
    "6": "purple",
    "7": "pink",
    "8": "white",
    "0": "other",
}
CLASSES = list(CLASS_KEYS.values())
# Colour used to draw each class on screen (blue, green, red).
CLASS_DRAW_BGR = {"red": (40, 40, 230), "orange": (0, 140, 255), "yellow": (0, 220, 255),
                  "green": (40, 180, 40), "blue": (220, 120, 30), "purple": (160, 50, 130),
                  "pink": (180, 130, 255), "white": (245, 245, 245), "other": (150, 150, 150)}
IMAGES_PER_CLASS = 100     # step 2 TARGET per class: recommended. More, more varied images = a better model
# MINIMUM before training (enforced by step3_train.py, ticks step 2 in the progress panel):
# the colours the robot sorts and "other" need at least this many images and scenes, so that
# whole scenes are left for validation and testing. Other colours are optional.
MIN_IMAGES_PER_CLASS = 30
MIN_SCENES_PER_CLASS = 5
# Which colours the robot sorts, and onto which square. Cubes of the other colours
# are recognised but left in the zone.
COLOUR_TO_SQUARE = {"red": "A", "yellow": "B", "green": "C"}
MAX_STACK = 3              # pick and drop positions for 3 stack levels

# Block finder: how a pixel is judged "not background".
FOREGROUND_THRESHOLD = 30  # colour difference from the empty-zone image (0-255)

# ---------------------------------------------------------------- model
CROP_SIZE = 64             # crops are saved and classified at 64 x 64 pixels
CONFIDENCE_THRESHOLD = 0.80

# ---------------------------------------------------------------- safety limits
# The robot is never sent to an x, y outside these limits, or lower than Z_MIN (metres).
X_LIMITS = (0.10, 0.40)
Y_LIMITS = (-0.30, 0.30)
Z_MIN = 0.085              # lowest gripper height allowed: catches typing errors such as 0.009 for 0.093
Z_MAX = 0.30
# Safe travel height = highest pick or drop position + this. The robot moves between
# positions at this height, then goes straight down, as in Lab 5.
SAFE_CLEARANCE_M = 0.05
# A cube in the pick zone is gripped at the height of your Lab 5 pick at square A,
# level 1. If cubes in the zone sit higher or lower than cubes on the squares,
# change this (mm, + = higher).
ZONE_PICK_OFFSET_MM = 0
# Optional extra: voice commands in step 4 (needs vosk, sounddevice and a model folder,
# see voice_commands.py). True = on without typing --voice.
USE_VOICE = False

# Level 2 (cube orientation): turn the gripper to line up with the cube's sides.
# cube.angle_deg is in the robot frame (+ = anticlockwise seen from above), so adding it to yaw
# turns the gripper towards the cube's sides. Check once with python test_wrist_turn.py.
USE_ORIENTATION = True
YAW_SIGN = 1               # leave at 1: the cube angle is already in the robot frame
TURN_AXIS = "yaw"          # "yaw" (normal); "roll" only if python test_wrist_turn.py says so
MAX_YAW_CORRECTION_DEG = 45
