"""
Test how to turn the gripper so it lines up with a turned cube (step 4, Level 2).

    python test_wrist_turn.py

The robot goes above square A at the safe height with your Lab 5 orientation, then
tries three ways of turning the gripper by +30 degrees, going back to straight after each:

    1. yaw + 30 deg       2. roll + 30 deg       3. the wrist joint (joint 6) + 30 deg

After each one, answer in this window: did the gripper turn about the vertical, and which way?
At the end it tells you what to set in config.py (TURN_AXIS).
Keep clear of the robot; the power switch is the emergency stop.
"""
import math

from robot import NiryoRobot, load_lab5_poses, orientation_gap_deg

TURN_DEG = 30


def ask(question, answers):
    while True:
        a = input(question).strip().lower()
        if a in answers:
            return a
        print("  Please type one of:", ", ".join(answers))


def report(robot, asked_rpy=None):
    pose, joints = robot.get_pose(), robot.get_joints()
    print(f"  robot reports roll, pitch, yaw = {[round(v, 3) for v in pose[3:]]}"
          f"   joint 6 = {math.degrees(joints[5]):.1f} deg")
    if asked_rpy is not None:
        gap = orientation_gap_deg(asked_rpy, pose[3:])
        print(f"  difference from the orientation asked for: {gap:.1f} deg"
              + ("  <- the robot did NOT go to it" if gap > 5 else ""))
    return pose, joints


def main():
    poses = load_lab5_poses()
    roll, pitch, yaw = poses["rpy"]
    x, y, _ = poses["pick"]["A"][1]
    z = poses["z_safe"]
    a = math.radians(TURN_DEG)
    print(f"Lab 5 orientation: roll {roll:.3f}, pitch {pitch:.3f}, yaw {yaw:.3f} rad")
    if abs(abs(pitch) - math.pi / 2) > 0.15:
        print("  Note: pitch is not close to 1.57 rad, so the gripper does not point straight down.\n"
              "  Turning yaw or roll then also tilts the gripper. Consider re-teaching Lab 5 with pitch about 1.57.")

    robot = NiryoRobot()
    robot.start()
    results = {}
    try:
        straight = [x, y, z, roll, pitch, yaw]
        tests = [("yaw", [x, y, z, roll, pitch, yaw + a]),
                 ("roll", [x, y, z, roll + a, pitch, yaw])]
        print("\nGoing above square A, straight (your Lab 5 orientation)...")
        robot.move_pose(*straight)
        _, j_straight = report(robot, straight[3:])
        input("Look at the gripper from above. Press Enter to try the first turn...")

        for name, pose in tests:
            print(f"\nTest: {name} + {TURN_DEG} deg")
            robot.move_pose(*pose)
            report(robot, pose[3:])
            results[name] = ask("Did the gripper turn about the vertical? c = clockwise (seen from above), "
                                "a = anticlockwise, n = no turn, t = it tilted instead: ", ["c", "a", "n", "t"])
            robot.move_pose(*straight)

        print(f"\nTest: wrist joint (joint 6) + {TURN_DEG} deg")
        j = list(j_straight)
        j[5] += a
        robot.move_joints(j)
        pose6, _ = report(robot)
        results["joint6"] = ask("Did the gripper turn about the vertical? c = clockwise, a = anticlockwise, "
                                "n = no turn: ", ["c", "a", "n"])
        robot.move_pose(*straight)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        robot.finish()

    print("\nResults:", results)
    for axis in ("yaw", "roll"):
        if results.get(axis) in ("c", "a"):
            print(f"\nUse {axis}. In config.py set:\n    TURN_AXIS = \"{axis}\"\n"
                  "Keep YAW_SIGN = 1. Then watch one turned cube in step 4: the gripper should line up with its "
                  "sides.")
            return
    if results.get("joint6") in ("c", "a"):
        print("\nOnly the wrist joint turned the gripper. Send these lines to your lab technician:"
              f"\n  Lab 5 rpy {[round(v, 3) for v in (roll, pitch, yaw)]}; after joint 6 + {TURN_DEG} deg "
              f"the robot reports rpy {[round(v, 3) for v in pose6[3:]]}")
    else:
        print("\nNo test turned the gripper: check the robot and its Lab 5 orientation with your lab technician.")


if __name__ == "__main__":
    main()
