"""
STEP 3a - Train, compare and choose the colour classifier.

    python step3_train.py
    python step3_train.py --feature colour        (only one kind of features)

KEY PARTS: two functions in this file (marked "KEY PART"):
  candidate_models()    - the models, and the settings of each, you want to try
  compare_and_choose()  - train each one, measure it, choose the best on VALIDATION
Around them:
  1. The dataset is split into train / validation / test BY SCENE: all crops from
     one key press go into the same set, so the test set really holds pictures the
     model has never seen. The split is saved to models/split.json.
  2. Features: each crop becomes numbers (colour_model.py): "colour" (hue histogram
     and average Lab colour) or "pixels" (the raw small picture). The report says how
     many numbers each gives.
  3. After the comparison: checks on the choice, then a report on EVERY candidate
     (results/candidates_report.txt): what the model is, its settings, training and
     validation accuracy, training time, weakest colour, and an outcome (chosen,
     overfits, underfits, good fit), with a validation confusion matrix for each in
     results/candidates/. Then ONE test of the chosen model on the test set.
     With --test-all the report also shows every candidate's test accuracy, after
     the choice, for discussion only.
  4. The chosen model is saved to models/colour_model.joblib and used by steps 3c and 4.
"""
import argparse
import csv
import json
import random
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config as cfg
from progress import print_progress
from colour_model import ColourModel, extract, list_dataset, load_images, scene_of, below_minimum

SEED = 42
FRACTIONS = (0.70, 0.15, 0.15)   # train, validation, test


# =====================================================================================
# KEY PART 1 of 2: the models to try
# =====================================================================================
def candidate_models():
    """Return a dictionary {name: model} of the scikit-learn classifiers you want to compare.

    Requirements (checked when the program runs):
      - at least two DIFFERENT kinds of classifier, each with settings you chose to tune;
      - the chosen one must give a confidence (predict_proba), because step 4 uses it;
      - each name says what the model is and its settings, e.g. "knn k=3".
    Justify your choice of models and settings in your report.
    """
    return {
        "knn k=3": KNeighborsClassifier(n_neighbors=3),
        "knn k=9": KNeighborsClassifier(n_neighbors=9),
        "svm C=1": CalibratedClassifierCV(SVC(kernel="rbf", C=1, gamma="scale"), ensemble=False),
        "svm C=10": CalibratedClassifierCV(SVC(kernel="rbf", C=10, gamma="scale"), ensemble=False),
        "mlp (64,32)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=SEED),
    }


# =====================================================================================
# KEY PART 2 of 2: train, measure and choose
# =====================================================================================
def compare_and_choose(X, y, feature, rows):
    """Train every model from candidate_models(), measure it, and choose one.

    X["train"], X["val"]  feature arrays of the training and validation crops
    y["train"], y["val"]  their labels
    feature               "colour" or "pixels"
    rows                  append one row per model: [name, feature, train_accuracy, validation_accuracy]

    Return (validation_accuracy, name, trained_model) for the model you choose.
    Requirements: the test set is never used here; the choice is made on validation accuracy;
    features are scaled before the model. save_loss_curve(mlp, feature, name) is available.
    """
    best = None
    for name, model in candidate_models().items():
        pipe = make_pipeline(StandardScaler(), model)
        pipe.fit(X["train"], y["train"])
        train_acc = accuracy_score(y["train"], pipe.predict(X["train"]))
        val_acc = accuracy_score(y["val"], pipe.predict(X["val"]))
        rows.append([name, feature, round(train_acc, 4), round(val_acc, 4)])
        print(f"{name:14s} + {feature:6s}  train {train_acc:.3f}   validation {val_acc:.3f}")
        if isinstance(model, MLPClassifier):
            save_loss_curve(pipe[-1], feature, name)
        if best is None or val_acc > best[0]:
            best = (val_acc, name, pipe)
    return best


# =====================================================================================
# Given
# =====================================================================================
def save_loss_curve(mlp, feature, name="mlp"):
    """Training loss per epoch of the neural network (MLP)."""
    plt.plot(mlp.loss_curve_)
    plt.xlabel("epoch")
    plt.ylabel("training loss (cross-entropy)")
    plt.title(f"Training curve - {name}, {feature} features")
    plt.tight_layout()
    safe = "".join(c if c.isalnum() else "_" for c in name)
    plt.savefig(cfg.RESULTS_DIR / f"loss_curve_{safe}_{feature}.png", dpi=150)
    plt.close()


def split_by_scene(paths, labels):
    """For each class, shuffle its scenes and share them out 70 / 15 / 15."""
    rng = random.Random(SEED)
    scenes = defaultdict(set)
    for p, l in zip(paths, labels):
        scenes[l].add(scene_of(p))
    assignment = {}
    for label, ids in scenes.items():
        ids = sorted(ids)
        if len(ids) < 3:
            # An optional colour with too few scenes cannot be split: leave it out for now.
            # (The colours the robot sorts and "other" are checked before this, in main.)
            print(f"  note: '{label}' has only {len(ids)} scene(s), so it is left out of training "
                  f"(at least 3 scenes are needed to split it)")
            continue
        rng.shuffle(ids)
        n_val = max(1, round(len(ids) * FRACTIONS[1]))
        n_test = max(1, round(len(ids) * FRACTIONS[2]))
        for i, s in enumerate(ids):
            assignment[(label, s)] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    split = {"train": [], "val": [], "test": []}
    for p, l in zip(paths, labels):
        part = assignment.get((l, scene_of(p)))
        if part:
            split[part].append([p, l])
    return split


def describe(model):
    """A plain description of a model and its settings, for the report."""
    inner = getattr(model, "estimator", None) or model
    kind = type(inner).__name__
    if kind == "KNeighborsClassifier":
        return (f"k-nearest neighbours, k={inner.n_neighbors}: labels a crop like the majority of the "
                f"{inner.n_neighbors} most similar training crops. It stores the examples instead of learning weights.")
    if kind == "SVC":
        return (f"support vector machine, {inner.kernel} kernel, C={inner.C}: draws boundaries between the colours; "
                f"a higher C fits the training data more tightly. Calibrated so it can give a confidence.")
    if kind == "MLPClassifier":
        return (f"neural network (multi-layer perceptron), hidden layers {inner.hidden_layer_sizes}: learns "
                f"weighted combinations of the features by gradient descent.")
    if kind == "RandomForestClassifier":
        return f"random forest, {inner.n_estimators} trees: many decision trees vote on the colour."
    if kind == "LogisticRegression":
        return "logistic regression: one weighted sum of the features per colour (a linear model)."
    return kind


def report_all_candidates(images, y, feature_names, chosen, test_all=False):
    """GIVEN: train every candidate again and describe how it did, on VALIDATION.
    Writes results/candidates_report.txt and a validation confusion matrix per candidate."""
    import time
    lines = ["Every candidate model, measured on the VALIDATION set (the test set is not used to choose).", ""]
    for feature in feature_names:
        X = {part: extract(images[part], feature) for part in images}
        lines.append(f"=== {feature} features: {X['train'].shape[1]} numbers per crop ===")
        for name, model in candidate_models().items():
            pipe = make_pipeline(StandardScaler(), model)
            t0 = time.perf_counter()
            pipe.fit(X["train"], y["train"])
            seconds = time.perf_counter() - t0
            train_acc = accuracy_score(y["train"], pipe.predict(X["train"]))
            val_pred = pipe.predict(X["val"])
            val_acc = accuracy_score(y["val"], val_pred)
            per_class = {c: np.mean([p == c for p, t in zip(val_pred, y["val"]) if t == c])
                         for c in cfg.CLASSES if c in y["val"]}
            weakest = min(per_class, key=per_class.get) if per_class else None
            if (name, feature) == chosen:
                outcome = "CHOSEN: best validation accuracy."
            elif train_acc - val_acc > 0.10:
                outcome = "Overfits: much better on training than on validation (it memorised the training crops)."
            elif train_acc < 0.90:
                outcome = "Underfits: weak even on training data (model too simple, or features not informative)."
            else:
                outcome = "Good fit: training and validation accuracy are close."
            lines += [f"{name}  +  {feature}",
                      f"  model:      {describe(model)}",
                      f"  accuracy:   training {train_acc:.3f}   validation {val_acc:.3f}   "
                      f"(trained in {seconds:.2f} s)"]
            if weakest is not None and per_class[weakest] < 1:
                lines.append(f"  weakest colour on validation: {weakest} ({per_class[weakest]:.0%} correct)")
            elif weakest is not None:
                lines.append("  every colour correct on validation")
            if test_all:
                test_acc = accuracy_score(y["test"], pipe.predict(extract(images["test"], feature)))
                lines.append(f"  test accuracy (for discussion only, NOT for choosing): {test_acc:.3f}")
            lines += [f"  outcome:    {outcome}", ""]
            labels = [c for c in cfg.CLASSES if c in y["val"] or c in val_pred]
            ConfusionMatrixDisplay(confusion_matrix(y["val"], val_pred, labels=labels),
                                   display_labels=labels).plot(cmap="Blues", colorbar=False)
            plt.title(f"Validation - {name} + {feature}")
            plt.tight_layout()
            safe = "".join(c if c.isalnum() else "_" for c in f"{name}_{feature}")
            plt.savefig(cfg.RESULTS_DIR / "candidates" / f"val_confusion_{safe}.png", dpi=110)
            plt.close()
    text = "\n".join(lines)
    (cfg.RESULTS_DIR / "candidates_report.txt").write_text(text)
    print("\n" + text)
    print(f"Saved results/candidates_report.txt and a confusion matrix per candidate in results/candidates/")


def interval(truth, predicted):
    """95 % Wilson interval of the accuracy, as text."""
    from trials_summary import wilson
    n = len(truth)
    low, high = wilson(sum(t == p for t, p in zip(truth, predicted)), n)
    return f"{low:.3f} to {high:.3f}"


def sorted_colour_accuracy(truth, predicted):
    """Accuracy on the colours the robot sorts (requirement R1), with its interval."""
    pairs = [(t, p) for t, p in zip(truth, predicted) if t in cfg.COLOUR_TO_SQUARE]
    if not pairs:
        return "no test images of these colours"
    t, p = zip(*pairs)
    return f"{accuracy_score(t, p):.3f} (95 % interval {interval(t, p)}, {len(t)} images)"


def model_kind(pipe):
    """'KNeighborsClassifier', 'SVC', 'MLPClassifier'... of a trained pipeline."""
    m = pipe[-1] if hasattr(pipe, "steps") else pipe
    inner = getattr(m, "estimator", None)
    return type(inner if inner is not None else m).__name__


def check_choice(rows, chosen_name, chosen_val, pipe):
    """Checks on the model code. Problems stop the program; notes are worth a look."""
    problems, notes = [], []
    if len({r[0] for r in rows}) < 2:
        problems.append("compare at least two models (candidate_models() must return two or more)")
    kinds = {model_kind(m) for m in candidate_models().values()}
    if len(kinds) < 2:
        problems.append(f"try at least two DIFFERENT kinds of model, not only {kinds.pop()}")
    if any(len(r) != 4 for r in rows):
        problems.append("each row must be [name, feature, train_accuracy, validation_accuracy]")
    elif round(chosen_val, 4) < max(r[3] for r in rows) - 1e-6:      # rows hold accuracies rounded to 4 places
        problems.append("the chosen model is not the one with the best VALIDATION accuracy")
    if not hasattr(pipe, "predict_proba"):
        problems.append("the chosen model cannot give a confidence (predict_proba): wrap an SVM in "
                        "CalibratedClassifierCV(...) as in the clue")
    for r in rows:
        if len(r) == 4 and r[2] - r[3] > 0.10:
            notes.append(f"{r[0]} + {r[1]}: training {r[2]:.2f} vs validation {r[3]:.2f}: overfitting?")
    return problems, notes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", choices=["colour", "pixels", "all"], default="all")
    parser.add_argument("--test-all", action="store_true",
                        help="after choosing, also show every candidate's TEST accuracy (for discussion only)")
    args = parser.parse_args()

    paths, labels = list_dataset()
    if not paths:
        raise SystemExit("No images in dataset/. Run step2_collect_data.py first.")
    print("Images per class:", dict(Counter(labels)))
    short = below_minimum()
    if short:
        raise SystemExit("Not enough data to train yet. Collect more with step2_collect_data.py:\n  " +
                         "\n  ".join(short))
    for c in cfg.CLASSES:
        if Counter(labels)[c] == 0:
            print(f"  note: no images of '{c}': the model will not know this class")
        elif Counter(labels)[c] < cfg.IMAGES_PER_CLASS:
            print(f"  note: '{c}' has {Counter(labels)[c]} images, below the target of {cfg.IMAGES_PER_CLASS}: "
                  f"more images usually give a better model")

    split = split_by_scene(paths, labels)
    cfg.MODEL_DIR.mkdir(exist_ok=True)
    cfg.RESULTS_DIR.mkdir(exist_ok=True)
    cfg.SPLIT_FILE.write_text(json.dumps(split, indent=1))
    for part in ("train", "val", "test"):
        print(f"{part:5s}: {len(split[part])} images", dict(Counter(l for _, l in split[part])))

    images = {part: load_images([p for p, _ in split[part]]) for part in split}
    y = {part: [l for _, l in split[part]] for part in split}

    feature_names = ["colour", "pixels"] if args.feature == "all" else [args.feature]

    rows, best = [], None
    for feature in feature_names:
        X = {part: extract(images[part], feature) for part in ("train", "val")}
        print(f"\n--- {feature} features ---")
        result = compare_and_choose(X, y, feature, rows)            # KEY PART
        if result is None:
            raise SystemExit("compare_and_choose() returned nothing: return (val_acc, name, pipe)")
        val_acc, name, pipe = result
        if best is None or val_acc > best[0]:
            best = (val_acc, name, feature, pipe)

    with open(cfg.RESULTS_DIR / "model_comparison.csv", "w", newline="") as f:
        csv.writer(f).writerows([["model", "feature", "train_accuracy", "val_accuracy"]] + rows)

    val_acc, name, feature, pipe = best
    problems, notes = check_choice(rows, name, val_acc, pipe)
    for n in notes:
        print("  note:", n)
    if problems:
        print("\nThe model code needs a change before the test:")
        for p in problems:
            print("  -", p)
        raise SystemExit(1)
    (cfg.RESULTS_DIR / "candidates").mkdir(exist_ok=True)
    report_all_candidates(images, y, feature_names, (name, feature), args.test_all)
    print(f"\nChosen on validation: {name} + {feature} ({val_acc:.3f}). Testing it ONCE on the test set...")
    model = ColourModel(pipe, feature, f"{name}+{feature}")
    predicted, _ = model.predict(images["test"])
    report = classification_report(y["test"], predicted, labels=cfg.CLASSES, zero_division=0)
    cm = confusion_matrix(y["test"], predicted, labels=cfg.CLASSES)
    text = (f"Best model on validation: {model.name}\n"
            f"TEST accuracy: {accuracy_score(y['test'], predicted):.3f}   "
            f"(95 % interval {interval(y['test'], predicted)} on {len(y['test'])} test images)\n"
            f"Sorted colours only {sorted(cfg.COLOUR_TO_SQUARE)}: {sorted_colour_accuracy(y['test'], predicted)}\n\n"
            f"{report}\n"
            f"Confusion matrix (rows = true, columns = predicted) {cfg.CLASSES}\n{cm}\n")
    print("\n" + text)
    (cfg.RESULTS_DIR / "test_report.txt").write_text(text)

    ConfusionMatrixDisplay(cm, display_labels=cfg.CLASSES).plot(cmap="Blues", colorbar=False)
    plt.title(f"Test set - {model.name}")
    plt.tight_layout()
    plt.savefig(cfg.RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    model.save()
    print(f"Saved {cfg.MODEL_FILE} and reports in {cfg.RESULTS_DIR}")
    print(HOW_TO_READ)


HOW_TO_READ = """
How to read these numbers
  training accuracy    share of TRAINING crops predicted right. 1.000 alone means little: the model
                       may have memorised them.
  validation accuracy  share of crops from scenes the model never saw. This is what you choose on.
                       Close to training (within about 0.05) = good fit. Much lower = overfitting.
                       Both low = underfitting: model too simple, poor features, or wrong labels.
  test accuracy        measured ONCE on unseen scenes; the 95 % interval shows how sure that number
                       is. A wide interval = too few test images to be precise.
  confusion matrix     rows = true colour, columns = predicted. Off the diagonal = mistakes: which
                       colour is taken for which (e.g. yellow taken for orange).
  confidence (live)    the model's estimate that its label is right, 0 to 1. With 9 classes it is
                       never below about 0.11. Best: high AND correct. Worst: high and WRONG, usually
                       a colour missing or mislabelled in the dataset. Low = unsure: the robot skips it.
  Images per class     a colour with no images cannot be predicted: such cubes are given the nearest
                       colour the model knows, often with high confidence.
"""


if __name__ == "__main__":
    try:
        print_progress('3a')
        main()
        print_progress('3a')
    except NotImplementedError as e:
        raise SystemExit(f"\nMissing: {e}\nCheck the function named in the message.")
