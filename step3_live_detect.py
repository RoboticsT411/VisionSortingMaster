"""
STEP 3c - Run the trained model on the live camera. The robot does not move.

    python step3_live_detect.py
    python step3_live_detect.py --source dataset_raw    (saved frames)

Each cube is shown with its predicted colour, confidence and robot x, y (mm).
The title bar shows the frame rate and the model's time per frame.

The box at the top of the camera window says what to do. Click on the camera
window first, so it receives your key presses.

Keys:  s = save a screenshot to results/    q = quit
"""
import argparse
import time

import cv2

import config as cfg
from camera import Camera
from colour_model import ColourModel
import progress
from progress import with_progress
from ui import BLUE, GREEN, RED, instruction_box
from vision import VisionSystem, show


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None)
    parser.add_argument("--debug", action="store_true", help="also show the top-down window")
    args = parser.parse_args()

    model = ColourModel.load()
    print("Model:", model.name)
    cam = Camera(args.source)
    vision = VisionSystem(model)
    last = time.perf_counter()
    saved_until = 0.0
    while True:
        frame = cam.read()
        result = vision.process(frame)
        cam_img, top = vision.draw(frame, result)
        now = time.perf_counter()
        fps, last = 1.0 / max(now - last, 1e-6), now
        speed = f"{fps:.1f} frames per second, model {result.inference_ms:.1f} ms per frame"
        if not result.fresh:
            box = (f"Markers not found ({result.markers_seen}/4)", ["Clear the view of the markers.", "Q = quit"], RED)
        else:
            box = ("Put cubes in the zone and check each prediction",
                   [f"Box = predicted colour. Number = confidence: how sure the model is (0 to 1), not proof it is "
                    f"right.   {speed}",
                    "Check: every colour correct? A high number on a WRONG colour = that colour is missing or "
                    "mislabelled in your data.",
                    "S = save a screenshot for your logbook (this ticks 3c in Project progress)    Q = quit when done"],
                   BLUE)
            if time.time() < saved_until:
                box = ("Screenshot saved in results/ - step 3c ticked",
                       ["Save more screenshots for different cube colours, or Q = quit."], GREEN)
        show("camera", instruction_box(with_progress(cam_img, "3c"), *box))
        if args.debug:
            show("top-down", top)
        key = cv2.waitKey(30) & 0xFF
        if key != 255 and chr(key).isupper():
            key = ord(chr(key).lower())
        if key == ord("q"):
            break
        if key == ord("s"):
            cfg.RESULTS_DIR.mkdir(exist_ok=True)
            stamp = int(time.time())
            cv2.imwrite(str(cfg.RESULTS_DIR / f"live_{stamp}_camera.png"), cam_img)
            if top is not None:
                cv2.imwrite(str(cfg.RESULTS_DIR / f"live_{stamp}_topdown.png"), top)
            progress._cache["status"] = None          # tick 3c straight away
            saved_until = time.time() + 2.5
            print("screenshot saved")
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as e:
        raise SystemExit(f"\nMissing: {e}\nCheck the function named in the message.")
