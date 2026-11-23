"""
Camera-to-robot mapping and parallax correction.

KEY PARTS: the four functions marked "KEY PART" are the heart of this file and are
used by every step from 1.5 onwards. Study them: the guide asks you to explain each one.

1. The platform is flat, so a single 3 x 3 matrix, a HOMOGRAPHY, maps a camera
   pixel (u, v) to a point (x, y) on the platform in robot coordinates. It is
   found from the four markers: their pixel positions (from the camera) and their
   robot positions (from step1_measure_markers.py).
2. The markers lie ON the platform, but the camera sees the TOP of a cube, which
   is higher. Seen at an angle, the top appears shifted away from the camera.
   correct_parallax() moves the position back, using where the camera is.

Check your work:
    python mapping.py           runs the self-checks on your functions
    python step1_check_calibration.py    measures your mapping against a ruler
"""
import numpy as np

import config as cfg


# =====================================================================================
# KEY PART 1 of 4: the homography from four (or more) point pairs
# =====================================================================================
def compute_homography(pixel_points, robot_points_mm):
    """Return the 3 x 3 homography H that maps camera pixels to robot coordinates (mm).

    pixel_points     list of (u, v): the marker centres in the camera image
    robot_points_mm  list of (x, y): the same markers in robot coordinates, mm
    Return a 3 x 3 numpy array, or None if it cannot be computed.
    Requirement: apply_homography(H, u, v) gives back each marker's (x, y) within 1 mm.
    """
    import cv2
    H, _ = cv2.findHomography(np.float32(pixel_points), np.float32(robot_points_mm))
    return H


# =====================================================================================
# KEY PART 2 of 4: use the homography
# =====================================================================================
def apply_homography(H, u, v):
    """Map one camera pixel (u, v) to robot coordinates with H. Return (x_mm, y_mm).

    Requirement: use homogeneous coordinates, so a point (u, v) becomes (u, v, 1).
    """
    x, y, w = H @ np.array([u, v, 1.0])
    return x / w, y / w


# =====================================================================================
# KEY PART 3 of 4: mapping error, to compare with your ruler checks
# =====================================================================================
def mapping_errors_mm(H, pixel_points, robot_points_mm):
    """For each point pair, the distance in mm between where H maps the pixel and where the
    point really is. Return a list of distances, one per point.
    """
    errors = []
    for (u, v), (x, y) in zip(pixel_points, robot_points_mm):
        mx, my = apply_homography(H, u, v)
        errors.append(float(np.hypot(mx - x, my - y)))
    return errors


# =====================================================================================
# KEY PART 4 of 4: parallax correction
# =====================================================================================
def correct_parallax(x_m, y_m, height_m, camera_m):
    """Correct the position of a point seen at a height above the platform.

    x_m, y_m   where the homography puts the point (robot coordinates, metres): this is
               where the camera's line of sight through the point meets the PLATFORM
    height_m   how high the point really is above the platform (e.g. the top of a cube)
    camera_m   (cx, cy, cz): the camera lens in robot x, y (metres) and its height above
               the PLATFORM (metres) - CAMERA_POSITION_M in config.py, which you measure
    Return the corrected (x, y) in metres: where the point really is.
    Requirement: a point straight below the camera does not move; a point at height 0 does not move.
    """
    cx, cy, cz = camera_m
    k = 1.0 - height_m / cz            # similar triangles along the line of sight
    return cx + (x_m - cx) * k, cy + (y_m - cy) * k


# =====================================================================================
# Given: self-checks
# =====================================================================================
_checked = set()


def check_homography():
    """Test compute_homography, apply_homography and mapping_errors_mm on a known mapping."""
    if "homography" in _checked:
        return
    true_H = np.array([[0.02, -0.55, 330.0], [-0.6, 0.01, 180.0], [0.00002, 0.0004, 1.0]])
    pixels = [(200, 120), (1080, 140), (1150, 650), (150, 620), (640, 380)]
    robot = []
    for u, v in pixels:
        x, y, w = true_H @ np.array([u, v, 1.0])
        robot.append((x / w, y / w))
    H = compute_homography(pixels[:4], robot[:4])
    if H is None or np.shape(H) != (3, 3):
        raise SystemExit("compute_homography() must return a 3 x 3 array")
    errors = mapping_errors_mm(H, pixels, robot)
    if len(errors) != len(pixels):
        raise SystemExit("mapping_errors_mm() must return one distance per point pair")
    if max(errors) > 1.0:
        raise SystemExit(f"Your mapping is off by up to {max(errors):.1f} mm on a test where the answer is known "
                         f"(the 5th point was not used to compute H). Check both functions.")
    _checked.add("homography")


def check_parallax():
    if "parallax" in _checked:
        return
    cam = (0.30, 0.0, 0.50)
    tests = [((0.30, 0.0, 0.025), (0.30, 0.0)),          # straight below the camera: no change
             ((0.20, 0.10, 0.0), (0.20, 0.10)),           # on the platform: no change
             ((0.20, 0.10, 0.025), (0.205, 0.095))]       # 5% of the way back towards the camera
    for (x, y, h), (ex, ey) in tests:
        rx, ry = correct_parallax(x, y, h, cam)
        if abs(rx - ex) > 1e-4 or abs(ry - ey) > 1e-4:
            raise SystemExit(f"correct_parallax({x}, {y}, {h}, camera {cam}) gave ({rx:.4f}, {ry:.4f}), "
                             f"expected ({ex:.4f}, {ey:.4f})")
    _checked.add("parallax")


def main():
    check_homography()
    print("Homography functions: OK (within 1 mm on a known test)")
    check_parallax()
    print("Parallax correction: OK")
    if cfg.CAMERA_POSITION_M is None:
        print("CAMERA_POSITION_M in config.py is not set yet: measure the camera lens position.")


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as e:
        raise SystemExit(f"\nMissing: {e}\nCheck the function named in the message.")
