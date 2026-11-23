"""
Features, dataset loading and the trained colour model.

A crop (64 x 64 image of one candidate cube) is turned into a feature
vector, and a scikit-learn model predicts its class and a confidence.

Two feature types, so you can compare them:
  "colour" - hand-designed: hue/saturation histogram + mean and spread of
             Lab colour, from the centre of the crop (78 numbers).
  "pixels" - the raw centre pixels, shrunk to 16 x 16 (768 numbers).
             The model has to learn its own features from these.
"""
from pathlib import Path

import cv2
import joblib
import numpy as np

import config as cfg


# ---------------------------------------------------------------- features
def centre_region(crop):
    """Middle half of the crop: mostly cube, little background."""
    h, w = crop.shape[:2]
    return crop[h // 4: 3 * h // 4, w // 4: 3 * w // 4]


def colour_features(crop):
    centre = centre_region(crop)
    hsv = cv2.cvtColor(centre, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [18, 4], [0, 180, 0, 256]).flatten()
    hist /= hist.sum() + 1e-6
    lab = cv2.cvtColor(centre, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32) / 255.0
    return np.concatenate([hist, lab.mean(axis=0), lab.std(axis=0)]).astype(np.float32)


def pixel_features(crop):
    small = cv2.resize(centre_region(crop), (16, 16), interpolation=cv2.INTER_AREA)
    return (small.astype(np.float32) / 255.0).flatten()


FEATURES = {"colour": colour_features, "pixels": pixel_features}


def extract(crops, feature_name):
    fn = FEATURES[feature_name]
    return np.array([fn(c) for c in crops])


# ---------------------------------------------------------------- dataset
def scene_of(path):
    """Images saved by the same key press share a scene id (file name before '_')."""
    return Path(path).stem.split("_")[0]


def dataset_counts(root=None):
    """{class: (images, scenes)} for every class in the dataset folder."""
    paths, labels = list_dataset(root)
    out = {c: [0, set()] for c in cfg.CLASSES}
    for p, l in zip(paths, labels):
        out[l][0] += 1
        out[l][1].add(scene_of(p))
    return {c: (n, len(s)) for c, (n, s) in out.items()}


def required_classes():
    """The classes that must reach the minimum: the sorted colours and "other"."""
    return [c for c in list(cfg.COLOUR_TO_SQUARE) + ["other"] if c in cfg.CLASSES]


def below_minimum(counts=None):
    """Texts for each required class that is below the minimum, e.g. "yellow: 12 images, 3 scenes"."""
    counts = counts or dataset_counts()
    out = []
    for c in required_classes():
        n, s = counts.get(c, (0, 0))
        if n < cfg.MIN_IMAGES_PER_CLASS or s < cfg.MIN_SCENES_PER_CLASS:
            out.append(f"{c}: {n} images in {s} scenes (minimum {cfg.MIN_IMAGES_PER_CLASS} images in "
                       f"{cfg.MIN_SCENES_PER_CLASS} scenes)")
    return out


def list_dataset(root=None):
    """Return (paths, labels) for every image in dataset/<class>/ (or another folder, same layout)."""
    root = Path(root) if root else cfg.DATA_DIR
    paths, labels = [], []
    for label in cfg.CLASSES:
        folder = root / label
        if not folder.exists():
            continue
        for p in sorted(folder.glob("*.png")):
            paths.append(str(p))
            labels.append(label)
    return paths, labels


def load_images(paths):
    return [cv2.imread(p) for p in paths]


# ---------------------------------------------------------------- trained model
class ColourModel:
    def __init__(self, pipeline, feature_name, name):
        self.pipeline = pipeline
        self.feature_name = feature_name
        self.name = name

    @property
    def classes(self):
        return list(self.pipeline.classes_)

    def predict(self, crops):
        """Return (labels, confidences) for a list of crops."""
        if len(crops) == 0:
            return [], []
        X = extract(crops, self.feature_name)
        proba = self.pipeline.predict_proba(X)
        best = proba.argmax(axis=1)
        labels = [self.pipeline.classes_[i] for i in best]
        return labels, proba.max(axis=1).tolist()

    def save(self, path=cfg.MODEL_FILE):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "feature": self.feature_name, "name": self.name}, path)

    @staticmethod
    def load(path=cfg.MODEL_FILE):
        if not Path(path).exists():
            raise FileNotFoundError(f"{path} not found. Run step3_train.py first.")
        d = joblib.load(path)
        return ColourModel(d["pipeline"], d["feature"], d["name"])
