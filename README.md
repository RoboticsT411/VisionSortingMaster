# AI Vision Colour Sorting with the Niryo One (complete version)

**Lab guide (PDF):** `docs/AI_Vision_Colour_Sorting_Lab_Guide.pdf` (setup, safety, every step with questions, the report).

A camera finds coloured cubes in the pick zone, a trained classifier names their colour, and
the Niryo One sorts red, yellow and green cubes onto squares A, B and C. **All the code is
given.** Tasks: set it up, run every step on your station, measure the results against
`SPECIFICATION.md`, and explain how each part works (the parts marked `KEY PART`).
Before you start, read the lab risk assessment RA001650/2 (summarised in the lab guide).

## A. Setup (once, on the lab PC)

Open your PyCharm project from the labs (the one with `ApiTCP_Python_NiryoOne`), then open the
**Terminal** tab at the bottom of PyCharm. The prompt must start with `(.venv)` (the project's
virtual environment); if it does not, run `.venv\Scripts\activate`.

1. Robot client (skip if the folder `ApiTCP_Python_NiryoOne` is already in the project):

       git clone https://github.com/LSBU-Electronics-Lab/ApiTCP_Python_NiryoOne.git

   In PyCharm, right-click that folder > Mark Directory as > Sources Root.

2. This project:

       git clone https://github.com/RoboticsT411/VisionSortingMaster.git
       cd VisionSortingMaster

   (Without git: unzip this folder into the project instead and `cd` into it.)

3. Packages:

       pip install -r requirements.txt

4. Settings: open `config.py` and set `CUBE_SIZE_MM` to your cubes' side length.

5. Check:

       python progress.py

   It prints the progress list with nothing ticked.

Optional, voice commands: `pip install vosk sounddevice`, unzip a Vosk English model
(e.g. `vosk-model-small-en-us-0.15`) next to `voice_commands.py`, test with `python voice_commands.py`.

## B. Order to run

Run these one after another from the project folder. A window shows what to do next and which
key to press: click the window once before pressing a key; `Q` quits. `python progress.py`
shows what is done at any time. Results go to `results/`.

| # | Command | What you do |
| --- | --- | --- |
| 1 | `python step1_lab5_positions.py` | Choose your robot; type the gripper orientation and the Lab 5 pick-up and drop positions for A, B, C at levels 1, 2, 3; test each; save |
| 2 | `python step1_tune_camera.py` | Click the picture, `A` auto-tune, `S` save, `Q` quit |
| 3 | `python mapping.py` | Self-test of the camera-to-robot mapping (prints OK) |
| 4 | `python step1_check_calibration.py` | Follow the checklist (4 ticks for A, B, C): click the centres of A, B, C; check each square with the robot (T, centre the gripper, Y: circles refit automatically); click A, B, C again to confirm, all within 5 mm. Then the ruler checks (R7) |
| 5 | `python step2_collect_data.py` | `B` saves the empty background; then one colour per scene, its number key (`1` red ... `8` white, `0` other). Minimum 30 images in 5 scenes for red, yellow, green, other |
| 6 | `python step2_collect_data.py --dataset dataset_lighting2` | A small second set in clearly different lighting (about 10 images per class) |
| 7 | `python step3_train.py` | Compares the models, chooses one, tests it once |
| 8 | `python step3_baseline_hsv.py --train` | Tunes the colour rule on the training images |
| 9 | `python step3_baseline_hsv.py` | Tests the colour rule (once) |
| 10 | `python step3_live_detect.py` | Live colours and confidence; `S` saves a screenshot. The robot does not move |
| 11 | `python step3_experiments.py` | Learning curve, leakage test, threshold curve, lighting shift |
| 12 | `python step4_sort_blocks.py --measure-offset` | Sorting with the robot; confirm each move with a key. First 5 picks guided, then at least 20 picks |
| 13 | `python trials_summary.py` | Your trials against the specification, with 95 % intervals |

Optional: `python explore_kmeans.py` (unsupervised k-means on your dataset),
`python test_wrist_turn.py` (checks the gripper turns the right way), step 4 with `--voice`.

**Safety (risk assessment RA001650/2):** keep at least 30 cm from the robot while it is powered
(also when checking the gripper), only the technician moves the arm by hand or connects power,
one PC per robot, and confirm a move only when nothing is in the way. To stop: `Q` (or `Esc`
in a guided pick), `Ctrl+C` in the terminal, or switch off the robot's power.

## C. Key parts to study and explain

| Step | File | Key part | What it does |
| --- | --- | --- | --- |
| 3 | `mapping.py` | `compute_homography()`, `apply_homography()`, `mapping_errors_mm()` | Camera pixel to robot x, y from the four markers |
| opt. | `mapping.py` | `correct_parallax()` | Corrects the cube-top position for the camera angle |
| 7 | `step3_train.py` | `candidate_models()`, `compare_and_choose()` | Models tried; train, measure, choose on validation |
| 8-9 | `step3_baseline_hsv.py` | `classify()` | A colour rule with no learning, to compare with the model |
| 11 | `step3_experiments.py` | the four experiments | Evidence about the model |
| 12 | `step4_sort_blocks.py` | `choose()`, `pick_and_drop()` | Which cube next; the robot moves (from the Lab 5 program) |

Other files: `config.py` (all settings; the only file you need to edit), `camera.py`,
`aruco_view.py`, `block_finder.py`, `colour_model.py`, `vision.py`, `robot.py`, `ui.py`,
`progress.py`, `voice_commands.py`.

## D. Updates

Staff may update the project during the term. Inside the project folder run:

    git pull
