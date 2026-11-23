# Project 1 specification: vision-based block sorting

This is the specification your system must meet, as an engineer would receive it from a
client. Read it before you start. Your report shows, requirement by requirement, whether your
system meets it, with the evidence and its uncertainty.

## 1. Purpose and scope

The system sorts coloured cubes from the pick zone onto squares A, B and C with a camera and
the Niryo One arm:

| Colour | Goes to |
|---|---|
| red | square A |
| yellow | square B |
| green | square C |
| any other colour, or anything that is not a cube | stays in the pick zone |

A person places up to 9 cubes, of any colours, anywhere inside the pick zone (the white outline
on the camera picture), not touching each other. The robot sorts every red, yellow and green
cube onto its square, stacking up to 3 per square, and leaves the rest.

**Out of scope:** cubes touching each other, cubes outside the pick zone, stacks higher than 3,
and lighting darker than the lab with the blinds closed.

## 2. Requirements

Every requirement must meet its **minimum**. The **stretch** targets are for the higher grade
bands. "How it is measured" names the program that gives you the number.

| ID | Requirement | Minimum | Stretch | How it is measured |
|---|---|---|---|---|
| R1 | Colour accuracy on red, yellow and green | ≥ 90 % | ≥ 97 % | `step3_train.py` test report ("Sorted colours only") |
| R2 | Colour accuracy on all classes | ≥ 80 % | ≥ 90 % | `step3_train.py` test report |
| R3 | Better than a hand-written rule | model beats your colour rule | by ≥ 10 points, explained colour by colour | `step3_baseline_hsv.py` against `step3_train.py` |
| R4 | Wrong-colour drops | 0 in 20 picks | 0 in 50 picks | `trials_summary.py` |
| R5 | Pick success (cube placed correctly) | ≥ 80 % of 20 picks | ≥ 95 % | `trials_summary.py` |
| R6 | Grip offset from the cube centre | ≤ 6 mm on average | ≤ 3 mm, with the gain from parallax correction shown | `step4_sort_blocks.py --measure-offset`, then `trials_summary.py` |
| R7 | Calibration accuracy | every ruler check within 5 mm | within 3 mm | `step1_check_calibration.py` |
| R8 | Time per cube | ≤ 30 s | ≤ 15 s | `trials_summary.py` |
| R9 | Skipped cubes of red, yellow or green | ≤ 15 % | ≤ 5 % | threshold curve (`step3_experiments.py`), checked by counting cubes left after a run |
| R10 | Other lighting | accuracy drops ≤ 15 points | ≤ 5 points, and how you achieved it | `step3_experiments.py` lighting shift |
| R11 | Safety | no collision; every move rehearsed and confirmed | — | observation; the step 4 review |

## 3. Cost of errors and the confidence threshold

Not all mistakes are equally bad. This table ranks them by what it takes to put each one right:

| Error | What has to happen to fix it | Relative cost |
|---|---|---|
| Collision | Stop the session, inspect the robot, possibly repair it | never accepted: prevented by hard rules (R11) |
| Wrong colour dropped | Notice it, lift off up to 2 cubes above it, restack (about 60 s) | **10** |
| Cube falls off the stack | Put it back (about 30 s) | **5** |
| Gripper misses the cube | The robot returns empty; the cube is picked next cycle (about 15 s) | **2** |
| Cube skipped (confidence too low) | It stays in the zone; pick it later or by hand (about 5 s) | **1** |

The confidence threshold (`CONFIDENCE_THRESHOLD` in `config.py`) decides between the two errors
it controls: a low threshold skips few cubes but drops more wrong; a high threshold drops
almost none wrong but skips more. Because a wrong drop costs 10 skips, choose the threshold
with the **lowest expected cost per cube**:

    expected cost = 10 × (fraction dropped wrongly) + 1 × (fraction skipped)
    fraction dropped wrongly = coverage × (1 − accuracy of kept)
    fraction skipped         = 1 − coverage

Coverage and accuracy of kept, for each threshold, come from your threshold curve
(`step3_experiments.py`). In your report, fill in this table from it and give your choice:

| Threshold | Coverage | Accuracy of kept | Dropped wrongly | Skipped | Expected cost |
|---|---|---|---|---|---|
| 0.50 | | | | | |
| … | | | | | |
| 0.95 | | | | | |

Then check the chosen threshold against R4 and R9. If no threshold meets both, say so and
explain what would fix it (more data, better features, a different model).

## 4. System architecture

| Block | File | Input → output | Units and frame | Status |
|---|---|---|---|---|
| Camera | `camera.py` | — → colour image | pixels | given |
| Marker detection | `aruco_view.py` | image → marker centres | pixels | given |
| Mapping | `mapping.py` | marker pixels and robot positions → homography; pixel → robot x, y | pixels → mm, robot base frame | given: **key part** |
| Object finder | `block_finder.py` | top-down image → cube candidates | top-down pixels | given |
| Classifier | `step3_train.py` | crop → colour and confidence | confidence 0 to 1 | given: **key part** (models and choice) |
| Parallax correction | `mapping.py` | x, y on the platform, height, camera position → corrected x, y | metres, robot base frame | given: **key part** |
| Decision | `step4_sort_blocks.py` `choose()` | detections and stacks → cube or none | — | given: **key part** |
| Motion | `step4_sort_blocks.py` `pick_and_drop()` | cube and Lab 5 positions → moves | metres, radians | given: **key part** |
| Safety | `robot.py` | every pose → allowed or refused | limits in `config.py` | given |

**Robot base frame:** origin at the base of the robot, x forward, y to the robot's left,
z up. All positions in the code are in metres in this frame.

## 5. Data plan

- **Classes:** red, orange, yellow, green, blue, purple, pink, white, and other (not a cube).
- **Amount:** a target of about 100 images per class, from at least 20 scenes per class (one key
  press each). The minimum before training is 30 images in 5 scenes for red, yellow, green and
  other (training refuses to run below it, and the progress panel ticks step 2 when it is met).
  The other colours are optional: a colour with fewer than 3 scenes is left out of training.
- **Conditions:** at least two lighting conditions in the main dataset, for example blinds open
  and blinds closed. A separate small set in clearly different lighting, about 10 images per
  class, kept only for the lighting test (R10).
- **Labels:** one colour per scene. A second person checks about 10 % of each folder; record
  how many wrong labels were found.
- **Split:** by scene, 70 / 15 / 15 (given). Images of one scene are near-copies, so a split
  image by image would let the test set leak into training.

## 6. Evaluation plan

| Requirement | Evidence | Sample size | Uncertainty |
|---|---|---|---|
| R1, R2 | test report, used once | your test set (about 15 % of your images) | 95 % interval printed by `step3_train.py` |
| R3 | baseline and model on the same test set | same | compare the intervals: do they overlap? |
| R4, R5, R8 | robot trials | at least 20 picks | 95 % interval printed by `trials_summary.py` |
| R6 | offsets with and without parallax correction | at least 10 picks each | average and standard deviation |
| R7 | ruler checks | the 6 marker distances and at least 2 two-point checks | the largest difference |
| R9 | threshold curve, and cubes left after a run | validation set; one run of 9 cubes | — |
| R10 | lighting shift | the separate lighting set | 95 % interval |

**Small samples mislead.** 19 out of 20 is 95 %, but the 95 % interval is about 76 % to 99 %.
0 wrong out of 20 does not prove the wrong rate is 0: it can still be up to about 16 %.
`trials_summary.py` prints these intervals for you; use them when you say pass or fail.

## 7. Risks and mitigations

| Risk | Likelihood | Effect | Mitigation |
|---|---|---|---|
| Camera knocked during a session | medium | every position wrong | ruler check at the start of each session; markers are re-detected in every frame |
| Sunlight changes the colours | high | accuracy drops | two lightings in the dataset; blinds closed for trials; R10 |
| Yellow confused with orange | medium | an orange cube dropped on B (wrong drop) | enough images of both; threshold from section 3 |
| Glare on a cube or marker | medium | missed cube or marker | camera auto-tune; matte markers |
| Wrong colour dropped | low | restack | threshold from section 3; R4 |
| Arm hits the camera arm or a stack | low | damage | pose limits, rehearsal of every move, a person confirms each move, power switch within reach |

## 8. Your report

Structure your report around this specification:

1. **Verification table**, first: for every requirement R1 to R11, the target, your result with
   its uncertainty, and pass or fail.

   | ID | Target (minimum) | Result | 95 % interval | Pass? | Evidence |
   |---|---|---|---|---|---|
   | R1 | ≥ 90 % | | | | test report |

2. **Design and implementation:** how each key part in section 4 works, why it is designed that way,
   and any change you made, with its effect measured.
3. **Experiments:** the four experiments from step 3.5 and the threshold table from section 3.
4. **Failure analysis:** every failed pick by cause (from `trials_summary.py`), with examples.
5. **Limitations and maintenance:** what your system cannot handle, and what someone would have
   to do to keep it working (for example after the camera is moved or the lighting changes).
