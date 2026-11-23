"""
STEP 2 - Collect and label your dataset.

    python step2_collect_data.py

The box at the top of the camera window always says what to do next.
Click on the camera window first, so it receives your key presses.

Before you start: move the robot to its sleep pose, clear the pick zone
and press  b  to save the empty-zone picture (background).

Then, for each "scene" (the box at the top suggests the colour that has the fewest images):
  1. Put 1-6 cubes OF ONE COLOUR on the blue area inside the white outline:
     not on squares A, B, C (they are only for dropping), not in the round holes,
     not touching each other.
  2. Check that every cube has a box on the camera picture.
  3. Press the colour's NUMBER key, shown on the coloured chips at the bottom of
     the camera window (CLASS_KEYS in config.py):
       1 red  2 orange  3 yellow  4 green  5 blue  6 purple  7 pink  8 white
       0 other (things that are not cubes: paper, tape, a pen lid...)
     EVERY boxed object is saved with that label, which is why only one colour
     may be in the zone at a time.
  4. Move the cubes, turn them, change the lighting, and repeat.
     Target: at least 20 scenes and IMAGES_PER_CLASS (100) images per class.

For the lighting experiment (step 3d), collect a smaller second set in clearly
different lighting into its own folder:
    python step2_collect_data.py --dataset dataset_lighting2

Other keys:  b = save the background (EMPTY zone only)
             f = save the full camera frame to dataset_raw/     q = quit

Only the camera window opens. The white outline on it is the pick zone, and
every object found gets a box. With --debug two more windows open: "top-down"
(the zone seen from straight above, where the program actually looks for
objects) and "foreground" (white = what differs from the empty-zone picture).
Use them only if objects are not being found.
"""
import argparse
import time

import cv2

import config as cfg
from camera import Camera
from colour_model import below_minimum, dataset_counts, required_classes
from progress import with_progress
from ui import BLUE, GREEN, ORANGE, RED, instruction_box, key_legend
from vision import VisionSystem, show


def what(label):
    return "objects that are not your cubes" if label == "other" else "cubes"


def counts():
    return {c: len(list((cfg.DATA_DIR / c).glob("*.png"))) if (cfg.DATA_DIR / c).exists() else 0
            for c in cfg.CLASSES}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None, help="camera index, image, folder or video")
    parser.add_argument("--debug", action="store_true", help="also show the top-down and foreground windows")
    parser.add_argument("--dataset", default=None,
                        help="save into another folder, e.g. dataset_lighting2 for the lighting experiment")
    args = parser.parse_args()
    if args.dataset:
        cfg.DATA_DIR = cfg.BASE_DIR / args.dataset
        print(f"Saving into {cfg.DATA_DIR}")

    keys = {ord(k): c for k, c in cfg.CLASS_KEYS.items()}
    key_of = {c: k for k, c in cfg.CLASS_KEYS.items()}
    legend = [(k, c, cfg.CLASS_DRAW_BGR.get(c, (150, 150, 150))) for k, c in cfg.CLASS_KEYS.items()]
    cam = Camera(args.source)
    vision = VisionSystem(model=None)
    print(__doc__)
    print("Background mode:", "empty-zone picture" if vision.background is not None else
          "colour rule (press b with the zone empty to switch)")

    message = None
    confirm_background = False
    scene_counts, scene_total = None, [-1]
    while True:
        frame = cam.read()
        result = vision.process(frame)
        cam_img, top = vision.draw(frame, result)
        n = counts()
        if scene_counts is None or sum(n.values()) != scene_total[0]:   # re-read only after a save
            scene_counts = dataset_counts(cfg.DATA_DIR)
            scene_total[0] = sum(n.values())
        short = below_minimum(scene_counts)
        # Suggest first the colours training needs (sorted colours and "other") still below the
        # minimum, then the class with the fewest images.
        needed = [c for c in required_classes() if any(t.startswith(c + ":") for t in short)]
        next_class = min(needed or cfg.CLASSES, key=lambda c: n[c])
        progress_line = (f"Minimum for training reached (step 3 can run). Keep going towards "
                         f"{cfg.IMAGES_PER_CLASS} per colour for a better model." if not short else
                         f"Before training: {', '.join(t.split(':')[0] for t in short)} need at least "
                         f"{cfg.MIN_IMAGES_PER_CLASS} images in {cfg.MIN_SCENES_PER_CLASS} scenes.")
        if not result.fresh:
            title, colour = f"Markers not found ({result.markers_seen}/4) - nothing can be saved", RED
            lines = ["Move the robot to its sleep pose and clear the view of the markers.", "Q = quit"]
        elif vision.background is None:
            title, colour = "STEP 1: clear the pick zone, then press B", BLUE
            lines = ["B = save the empty zone as the background (nothing in the zone!)", "Q = quit"]
        elif not result.detections:
            title, colour = (f"STEP 2: put 1-6 {next_class.upper()} {what(next_class)} in the zone "
                             f"(key {key_of[next_class]})", BLUE)
            lines = [message or ("Inside the white outline, on the blue area: NOT on squares A, B, C, NOT in the "
                                 "round holes, not touching."),
                     "Only ONE colour at a time: one key press labels every boxed object.",
                     progress_line,
                     "F = save a frame    B = save the background again (EMPTY zone only)    Q = quit when done"]
        else:
            n_found = len(result.detections)
            title, colour = (f"STEP 3: {n_found} object(s) found - press the NUMBER of their colour", GREEN)
            lines = [message or "Every object must have a box. If one is missing, move it away from the edge.",
                     f"All of them get that label (for example {key_of[next_class]} = {next_class}). "
                     f"The chips below show the keys and how many images each colour has.",
                     "Then move the cubes or change the light and press again.   Q = quit when done"]
        shown = instruction_box(with_progress(cam_img, "2"), title, lines, colour)
        if vision.background is not None and result.fresh:
            shown = key_legend(shown, legend, n, cfg.IMAGES_PER_CLASS, highlight=next_class)
        show("camera", shown)
        if args.debug:
            show("top-down", top)
            if result.mask is not None:
                show("foreground", result.mask, 480)

        key = cv2.waitKey(30) & 0xFF
        if key != 255 and chr(key).isupper():
            key = ord(chr(key).lower())
        if key == ord("q"):
            break
        if key == ord("f"):
            cfg.RAW_DIR.mkdir(exist_ok=True)
            path = cfg.RAW_DIR / f"frame_{int(time.time() * 1000)}.png"
            cv2.imwrite(str(path), frame)
            print("saved", path)
            message = "Frame saved to dataset_raw/."
            continue
        if key == ord("b"):
            if not result.fresh:
                print("Cannot save background: all 4 markers must be visible.")
                continue
            if result.detections and vision.background is not None and not confirm_background:
                # B is not "blue": it saves the EMPTY zone. Saving with cubes in view would hide them.
                message = (f"B saves the EMPTY zone. {len(result.detections)} object(s) in the zone: "
                           f"remove them first, or press B again to save anyway.")
                confirm_background = True
                continue
            confirm_background = False
            cv2.imwrite(str(cfg.BACKGROUND_FILE), result.topdown)
            vision.background = result.topdown.copy()
            print("Background saved. Objects are now found by difference from the empty zone.")
            message = None
            continue
        if key != 255 and key != ord("b"):
            confirm_background = False
        if key in keys:
            if not result.fresh:
                print("Not saved: all 4 markers must be visible.")
                continue
            if not result.detections:
                print("Not saved: no objects found in the zone.")
                continue
            label = keys[key]
            folder = cfg.DATA_DIR / label
            folder.mkdir(parents=True, exist_ok=True)
            scene = str(int(time.time() * 1000))
            for k, d in enumerate(result.detections):
                cv2.imwrite(str(folder / f"{scene}_{k}.png"), d.crop)
            print(f"saved {len(result.detections)} x '{label}'  (scene {scene})")
            message = (f"Saved {len(result.detections)} x {label} ({n[label] + len(result.detections)}/"
                       f"{cfg.IMAGES_PER_CLASS}). Move the cubes or change the light, then press again.")

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as e:
        raise SystemExit(f"\nMissing: {e}\nCheck the function named in the message.")
