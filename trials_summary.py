"""
STEP 4.3 - Summarise your robot trials against the specification.

    python trials_summary.py

Reads results/trials.csv (written by step4_sort_blocks.py) and prints, for the real
robot runs (dry runs are left out):
  - pick success rate (cubes placed correctly), with a 95 % confidence interval
  - wrong-colour drops and the other failure causes (you give the cause when you answer
    n after a pick), with the upper limit of the true rate when none happened
  - time per cube
  - grip offsets with and without the parallax correction (from --measure-offset)
It also saves the summary to results/trials_summary.txt for your report.
"""
import csv
from datetime import datetime
from statistics import mean, stdev

import config as cfg
from progress import print_progress


def wilson(successes, n, z=1.96):
    """95 % confidence interval of a proportion (Wilson score interval): (low, high)."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def rate_text(successes, n):
    if n == 0:
        return "no data"
    low, high = wilson(successes, n)
    return f"{successes}/{n} = {successes / n:.0%}  (95 % interval {low:.0%} to {high:.0%})"


def main():
    path = cfg.RESULTS_DIR / "trials.csv"
    if not path.exists():
        raise SystemExit("No results/trials.csv yet: run step4_sort_blocks.py with the robot first.")
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("dry_run") != "True"]
    if not rows:
        raise SystemExit("trials.csv only holds dry runs: run step4_sort_blocks.py without --dry-run.")

    answered = [r for r in rows if r.get("outcome") in ("y", "n")]
    placed = sum(r["outcome"] == "y" for r in answered)
    wrong = sum(r.get("failure_cause") == "wrong colour" for r in answered)
    lines = [f"Robot trials: {len(rows)} picks ({len(answered)} with an answer to 'placed correctly?')", "",
             f"Pick success (R5):        {rate_text(placed, len(answered))}"]

    lines.append(f"Wrong-colour drops (R4):  {rate_text(wrong, len(answered))}")
    if wrong == 0 and answered:
        lines.append(f"  (0 wrong does not prove the rate is 0: with {len(answered)} picks it can still be up to "
                     f"{wilson(0, len(answered))[1]:.0%})")
    causes = {}
    for r in answered:
        if r["outcome"] == "n":
            causes[r.get("failure_cause") or "not recorded"] = causes.get(r.get("failure_cause") or "not recorded", 0) + 1
    if causes:
        lines.append("Failures by cause:        " + ", ".join(f"{c} {n}" for c, n in sorted(causes.items())))

    times = sorted(datetime.fromisoformat(r["time"]) for r in rows)
    gaps = [(b - a).total_seconds() for a, b in zip(times, times[1:]) if (b - a).total_seconds() < 120]
    if gaps:
        lines.append(f"Time per cube (R8):       {mean(gaps):.1f} s on average ({len(gaps)} intervals; "
                     f"gaps over 2 minutes are left out as breaks)")

    for corrected in ("False", "True"):
        offsets = []
        for r in rows:
            try:
                if r.get("parallax_corrected") == corrected and r.get("offset_mm", "").strip():
                    offsets.append(float(r["offset_mm"]))
            except ValueError:
                pass
        label = "with" if corrected == "True" else "without"
        if len(offsets) >= 2:
            lines.append(f"Grip offset {label:7s} parallax correction (R6): {mean(offsets):.1f} mm average, "
                         f"standard deviation {stdev(offsets):.1f} mm, {len(offsets)} picks")
        elif offsets:
            lines.append(f"Grip offset {label:7s} parallax correction (R6): {offsets[0]:.1f} mm (1 pick)")

    colours = {}
    for r in answered:
        colours.setdefault(r["label"], []).append(r["outcome"] == "y")
    lines += ["", "Success by colour:"]
    lines += [f"  {c:7s} {rate_text(sum(v), len(v))}" for c, v in sorted(colours.items())]

    text = "\n".join(lines)
    print(text)
    cfg.RESULTS_DIR.mkdir(exist_ok=True)
    (cfg.RESULTS_DIR / "trials_summary.txt").write_text(text)
    print("\nSaved results/trials_summary.txt")


if __name__ == "__main__":
    print_progress("4:Robot trials")
    main()
