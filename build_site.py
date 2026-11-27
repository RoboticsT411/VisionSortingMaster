"""Builds project/student.json (the files students start with) from the VisionSortingMaster project.

    python build_site.py ../VisionSortingMaster

Labs 1 to 4 are kept as they are in the current student.json. The AI vision sorting files are taken from the
project folder given; config.py gets the simulator block below appended, and stations/simulator.json holds the
simulated station's Lab 5 positions (press "Load saved positions" in step 1.1).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "project" / "student.json"
FOLDER = "AI-VISION-SORTING"
SKIP_DIRS = {"docs", "calibration", "dataset", "dataset_lighting2", "models", "results", "__pycache__", ".git",
             ".venv", ".idea"}
TEXT = {".py", ".md", ".txt", ".json", ".csv"}

SIM_CONFIG = '''

# ---------------------------------------------------------------- online simulator
# These lines adapt the project to the simulated station in the browser.
# They replace the lab values above; leave them as they are in the simulator.
ROBOTS = {"Simulator": "simulator"}
ROBOT_IP = "simulator"
CUBE_SIZE_MM = 20                       # the simulator's cubes are 20 mm
ZONE_PICK_OFFSET_MM = 6                 # a cube in the zone stands 6 mm higher than one in a hole
CAMERA_POSITION_M = (0.40, -0.02, 0.47) # the simulated camera (x, y, height); measure it yourself as a check
PARALLAX_HEIGHT_MM = 10                 # the box found round a cube includes its sides: its centre is about half-way up
'''


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else HERE.parent / "VisionSortingMaster").resolve()
    old = json.loads(OUT.read_text())
    files = {k: v for k, v in old["files"].items() if not k.startswith(FOLDER + "/")}
    sim_station = old["files"].get(f"{FOLDER}/stations/simulator.json")
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        if p.is_dir() or rel.parts[0] in SKIP_DIRS or p.suffix not in TEXT or p.name.startswith("."):
            continue
        text = p.read_text(encoding="utf-8")
        if rel.as_posix() == "config.py":
            text = text.rstrip("\n") + "\n" + SIM_CONFIG
        files[f"{FOLDER}/{rel.as_posix()}"] = text
    if sim_station:
        files[f"{FOLDER}/stations/simulator.json"] = sim_station
    OUT.write_text(json.dumps({"package": "student", "files": files}, indent=0, sort_keys=True))
    print(f"{OUT.name}: {len(files)} files ({sum(k.startswith(FOLDER) for k in files)} in {FOLDER})")


if __name__ == "__main__":
    main()
