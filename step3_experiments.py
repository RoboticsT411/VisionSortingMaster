"""
STEP 3d - Experiments: evidence about your model, for your report.

    python step3_experiments.py

Run step3_train.py first: these experiments use its split (models/split.json) and
the kind of model you chose (models/colour_model.joblib).

KEY PARTS: the four experiments marked "KEY PART". Also in this file: loading the data,
training a fresh copy of your chosen model, the plots and the report
(results/experiments.txt and results/experiment_*.png).

  1. learning_curve   - does more data help? Validation accuracy against training-set size.
  2. leakage_test     - why split by scene? The same model with a split by scene and by image.
  3. threshold_curve  - which CONFIDENCE_THRESHOLD? Accuracy against how many cubes are accepted.
  4. lighting_shift   - does it work in other lighting? Needs dataset_lighting2:
                            python step2_collect_data.py --dataset dataset_lighting2
"""
import json
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config as cfg
from colour_model import ColourModel, extract, list_dataset, load_images, scene_of
from progress import print_progress

SEED = 0


# =====================================================================================
# Given: data and your chosen model
# =====================================================================================
class Data:
    """The split from step3_train.py, loaded once.

    data.images[part], data.labels[part], data.scenes[part] for part in "train", "val", "test"
    data.feature            "colour" or "pixels": the features your chosen model uses
    data.fresh_model()      an UNTRAINED copy of your chosen model (scaling included)
    data.features(images)   the feature numbers for a list of images
    """

    def __init__(self):
        if not cfg.SPLIT_FILE.exists() or not cfg.MODEL_FILE.exists():
            raise SystemExit("Run step3_train.py first: the experiments use its split and chosen model.")
        split = json.loads(cfg.SPLIT_FILE.read_text())
        self.images, self.labels, self.scenes = {}, {}, {}
        for part in ("train", "val", "test"):
            paths = [p for p, _ in split[part]]
            self.images[part] = load_images(paths)
            self.labels[part] = [label for _, label in split[part]]
            self.scenes[part] = [scene_of(p) for p in paths]
        self.chosen = ColourModel.load()
        self.feature = self.chosen.feature_name

    def fresh_model(self):
        return clone(self.chosen.pipeline)

    def features(self, images):
        return extract(images, self.feature)

    def train(self, images, labels):
        """A fresh copy of your chosen model, trained on these images."""
        model = self.fresh_model()
        model.fit(self.features(images), labels)
        return model


# =====================================================================================
# KEY PART 1 of 4: learning curve
# =====================================================================================
def learning_curve(data):
    """Train a fresh copy of the chosen model on 25 %, 50 %, 75 % and 100 % of the training
    SCENES (keep whole scenes together) and measure validation accuracy each time.
    Return a list of (number_of_training_images, validation_accuracy).
    """
    rng = random.Random(SEED)
    scenes = sorted(set(data.scenes["train"]))
    rng.shuffle(scenes)
    X_val = data.features(data.images["val"])
    points = []
    for fraction in (0.25, 0.5, 0.75, 1.0):
        keep = set(scenes[:max(1, round(len(scenes) * fraction))])
        idx = [i for i, s in enumerate(data.scenes["train"]) if s in keep]
        if len({data.labels["train"][i] for i in idx}) < 2:
            continue                                   # too few scenes for two colours
        model = data.train([data.images["train"][i] for i in idx], [data.labels["train"][i] for i in idx])
        points.append((len(idx), accuracy_score(data.labels["val"], model.predict(X_val))))
    return points


# =====================================================================================
# KEY PART 2 of 4: why split by scene?
# =====================================================================================
def leakage_test(data):
    """Pool the training and validation images. Split them 80/20 in two ways, train a fresh
    copy of the chosen model on each and measure accuracy on its 20 %:
      a) by SCENE: all images of one scene on the same side
      b) by IMAGE: images shuffled one by one, so one scene can be on both sides
    Return (accuracy_by_scene, accuracy_by_image).
    """
    images = data.images["train"] + data.images["val"]
    labels = data.labels["train"] + data.labels["val"]
    scenes = data.scenes["train"] + data.scenes["val"]
    rng = random.Random(SEED)

    def score(test_idx):
        train_idx = [i for i in range(len(images)) if i not in test_idx]
        model = data.train([images[i] for i in train_idx], [labels[i] for i in train_idx])
        X = data.features([images[i] for i in sorted(test_idx)])
        return accuracy_score([labels[i] for i in sorted(test_idx)], model.predict(X))

    unique = sorted(set(scenes))
    rng.shuffle(unique)
    held = set(unique[:max(1, len(unique) // 5)])
    by_scene = score({i for i, s in enumerate(scenes) if s in held})
    order = list(range(len(images)))
    rng.shuffle(order)
    by_image = score(set(order[:max(1, len(order) // 5)]))
    return by_scene, by_image


# =====================================================================================
# KEY PART 3 of 4: choosing the confidence threshold
# =====================================================================================
def threshold_curve(data):
    """With the chosen model (data.chosen, already trained), predict the VALIDATION images.
    For each threshold 0.50, 0.55, ... 0.95, keep only predictions at or above it and work out
      coverage: the fraction of crops kept (cubes the robot would act on)
      accuracy: the accuracy of the kept predictions
    Return a list of (threshold, coverage, accuracy). Use accuracy = 1.0 if nothing is kept.
    """
    predicted, confidence = data.chosen.predict(data.images["val"])
    truth = data.labels["val"]
    rows = []
    for threshold in np.arange(0.50, 0.951, 0.05):
        kept = [i for i, c in enumerate(confidence) if c >= threshold]
        coverage = len(kept) / len(truth)
        accuracy = np.mean([predicted[i] == truth[i] for i in kept]) if kept else 1.0
        rows.append((round(float(threshold), 2), float(coverage), float(accuracy)))
    return rows


# =====================================================================================
# KEY PART 4 of 4: other lighting
# =====================================================================================
def lighting_shift(data, other_images, other_labels):
    """The chosen model (data.chosen, already trained) on two sets it has never seen:
    the normal TEST images, and the images collected in different lighting.
    Return (accuracy_on_test, accuracy_in_other_lighting).
    """
    test_pred, _ = data.chosen.predict(data.images["test"])
    other_pred, _ = data.chosen.predict(other_images)
    return (accuracy_score(data.labels["test"], test_pred), accuracy_score(other_labels, other_pred))


# =====================================================================================
# Given: run the experiments, plot, report
# =====================================================================================
def plot(xs, series, xlabel, ylabel, title, name):
    plt.figure(figsize=(5.5, 4))
    for label, ys in series:
        plt.plot(xs, ys, "o-", label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(0, 1.05)
    plt.title(title)
    plt.grid(alpha=0.3)
    if len(series) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(cfg.RESULTS_DIR / f"experiment_{name}.png", dpi=130)
    plt.close()


def main():
    cfg.RESULTS_DIR.mkdir(exist_ok=True)
    data = Data()
    report = [f"Experiments with the chosen model: {data.chosen.name}", ""]
    missing = []

    def run(title, fn):
        try:
            return fn()
        except NotImplementedError as e:
            missing.append(str(e))
            report.extend([title, "  missing", ""])
            return None

    points = run("1. Learning curve", lambda: learning_curve(data))
    if points:
        n, acc = zip(*points)
        plot(n, [("validation", acc)], "training images", "validation accuracy", "Learning curve", "learning_curve")
        report += ["1. Learning curve (training images -> validation accuracy)"]
        report += [f"  {a:5d} images -> {b:.3f}" for a, b in points] + [""]

    result = run("2. Leakage test", lambda: leakage_test(data))
    if result:
        report += ["2. Leakage test (same model, two ways of splitting)",
                   f"  split by scene: {result[0]:.3f}", f"  split by image: {result[1]:.3f}",
                   "  Is splitting by image more optimistic? Explain why in your report.", ""]

    rows = run("3. Threshold curve", lambda: threshold_curve(data))
    if rows:
        t, cov, acc = zip(*rows)
        plot(t, [("accuracy of kept", acc), ("coverage (kept)", cov)], "confidence threshold", "fraction",
             "Confidence threshold trade-off", "threshold")
        report += ["3. Threshold curve (threshold: coverage, accuracy of kept predictions)"]
        report += [f"  {a:.2f}: coverage {b:.2f}, accuracy {c:.3f}" for a, b, c in rows]
        report += [f"  Your CONFIDENCE_THRESHOLD in config.py is {cfg.CONFIDENCE_THRESHOLD}: justify it from "
                   f"this table.", ""]

    other_dir = cfg.BASE_DIR / "dataset_lighting2"
    if other_dir.exists():
        paths, labels = list_dataset(other_dir)
        if paths:
            result = run("4. Lighting shift", lambda: lighting_shift(data, load_images(paths), labels))
            if result:
                report += ["4. Lighting shift", f"  test set (normal lighting): {result[0]:.3f}",
                           f"  other lighting ({len(paths)} images): {result[1]:.3f}", ""]
    else:
        report += ["4. Lighting shift", "  no dataset_lighting2 folder yet: collect it with",
                   "  python step2_collect_data.py --dataset dataset_lighting2", ""]

    text = "\n".join(report)
    print(text)
    if not missing:
        (cfg.RESULTS_DIR / "experiments.txt").write_text(text)
        print("Saved results/experiments.txt and results/experiment_*.png")
    else:
        print("Missing:\n  " + "\n  ".join(missing))
        print("results/experiments.txt is saved when all four experiments are written.")


if __name__ == "__main__":
    print_progress("3d")
    main()
    print_progress("3d")
