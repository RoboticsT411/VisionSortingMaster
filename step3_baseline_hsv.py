"""
STEP 3b - Rule-based baseline: classify by colour thresholds, NO learning.

KEY PART: the colour rule in classify(), marked "KEY PART".

    python step3_baseline_hsv.py --train     tune your rule on the TRAINING images
    python step3_baseline_hsv.py             final result on the TEST images (do this once)

Uses the same split as step3_train.py (models/split.json), so the rule and your
learned model are compared on exactly the same test pictures. Tune only with
--train: tuning on the test images would make the comparison unfair.

The result shows where a hand-written rule is enough and where learning helps,
for example in shadow, under yellow sunlight, or for pink against red.
"""
import argparse
import json

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import config as cfg
from progress import print_progress
from colour_model import centre_region


def hsv_of(crop):
    """Median hue (0-179 in OpenCV), saturation (0-255) and value (brightness, 0-255) of the
    centre of the crop. Hue: red ~0 and ~179, orange ~12, yellow ~27, green ~60, blue ~110,
    purple ~140, pink ~165. Low saturation = greyish/white."""
    hsv = cv2.cvtColor(centre_region(crop), cv2.COLOR_BGR2HSV)
    return float(np.median(hsv[..., 0])), float(np.median(hsv[..., 1])), float(np.median(hsv[..., 2]))


# =====================================================================================
# KEY PART: the rule
# =====================================================================================
def classify(crop):
    """Return the colour name of one crop using thresholds only (no learning).

    hue, sat, val = hsv_of(crop) gives the median hue (0-179), saturation and brightness (0-255).
    Requirements: return one of cfg.CLASSES ("other" for anything you cannot place); choose the
    thresholds from your TRAINING images (--train), never from the test images.
    """
    hue, sat, val = hsv_of(crop)
    if sat < 70:
        return "white" if val >= 150 and "white" in cfg.CLASSES else "other"
    ranges = {"red": [(0, 7), (172, 179)], "orange": [(8, 17)], "yellow": [(18, 35)], "green": [(36, 85)],
              "blue": [(86, 128)], "purple": [(129, 150)], "pink": [(151, 171)]}
    for label, spans in ranges.items():
        if label in cfg.CLASSES and any(lo <= hue <= hi for lo, hi in spans):
            if label == "red" and sat < 150 and "pink" in cfg.CLASSES:
                return "pink"
            return label
    return "other"


# =====================================================================================
# Given
# =====================================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="measure on the TRAINING images (for tuning)")
    args = parser.parse_args()
    if not cfg.SPLIT_FILE.exists():
        raise SystemExit("Run step3_train.py first: it creates the train/test split.")
    part = "train" if args.train else "test"
    data = json.loads(cfg.SPLIT_FILE.read_text())[part]
    y_true = [label for _, label in data]
    y_pred = [classify(cv2.imread(path)) for path, _ in data]
    bad = sorted({p for p in y_pred if p not in cfg.CLASSES})
    if bad:
        raise SystemExit(f"classify() returned {bad}: it must return one of {cfg.CLASSES}")
    text = (f"Colour rule baseline\n{part.upper()} accuracy: {accuracy_score(y_true, y_pred):.3f}\n\n"
            f"{classification_report(y_true, y_pred, labels=cfg.CLASSES, zero_division=0)}\n"
            f"Confusion matrix (rows = true, columns = predicted) {cfg.CLASSES}\n"
            f"{confusion_matrix(y_true, y_pred, labels=cfg.CLASSES)}\n")
    print(text)
    if args.train:
        print("Tuning on the training images. When you are happy, run without --train for the test result.")
    else:
        cfg.RESULTS_DIR.mkdir(exist_ok=True)
        (cfg.RESULTS_DIR / "baseline_report.txt").write_text(text)
        print("Saved results/baseline_report.txt")


if __name__ == "__main__":
    try:
        print_progress('3b')
        main()
        print_progress('3b')
    except NotImplementedError as e:
        raise SystemExit(f"\nMissing: {e}\nCheck the function named in the message.")
