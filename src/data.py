"""Local dataset loading, group-disjoint splits and published test files.

The frozen release consumed four *exported* files (``*.arrow`` / ``source.zip``)
whose SHA-256 hashes are hard-coded in the notebook.  Two of those exports are
byte-identical to the public files shipped in ``dataset/`` (NSL-KDD
``KDDTrain+.txt`` and UNSW-NB15 ``UNSW_NB15_training-set.csv``); ToN-IoT and
RT-IoT are read here from their original public CSV files instead of the
``.arrow`` / ``.zip`` re-packing.  Consequently the ToN-IoT / RT-IoT split
hashes cannot match the release and are not asserted; the *split rule*,
feature front-end, encoders and decision rule are identical.

Held-out evaluation uses the published external test files where they exist
(NSL-KDD ``KDDTest+.txt``, UNSW-NB15 ``UNSW_NB15_testing-set.csv``) and the
20% group-disjoint test partition for ToN-IoT and RT-IoT.
"""

import hashlib
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .pipeline import DROP

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"

FILES = {
    "ToN-IoT": DATASET_DIR / "ton_iot" / "train_test_network.csv",
    "UNSW-NB15": DATASET_DIR / "unsw" / "UNSW_NB15_training-set.csv",
    "NSL-KDD": DATASET_DIR / "nsl_kdd" / "KDDTrain+.txt",
    "RT-IoT": DATASET_DIR / "rt_iot" / "RT_IOT2022.csv",
}
EXTERNAL_TESTS = {
    "NSL-KDD": DATASET_DIR / "nsl_kdd" / "KDDTest+.txt",
    "UNSW-NB15": DATASET_DIR / "unsw" / "UNSW_NB15_testing-set.csv",
}
DATASETS = ["ToN-IoT", "UNSW-NB15", "NSL-KDD", "RT-IoT"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _finalise(f, name):
    predictors = [c for c in f if c not in DROP]
    # String vs uint64 changes the ordering of unique groups in sklearn.
    f["_fingerprint"] = pd.util.hash_pandas_object(f[predictors], index=False).astype(str)
    if not f.label.isin([0, 1]).all():
        raise ValueError(f"{name}: label must be binary")
    if not f._row_id.is_unique:
        raise ValueError(f"{name}: duplicate row id")
    return f.set_index("_row_id", drop=False)


def _load_nsl_kdd(path):
    raw = pd.read_csv(path, header=None)
    if raw.shape[1] != 43:
        raise ValueError(f"{path}: expected 43 columns, got {raw.shape[1]}")
    names = [f"feature_{i:02d}" for i in range(41)]
    names[1:4] = ["protocol_type", "service", "flag"]
    f = raw.iloc[:, :41].copy()
    f.columns = names
    f["type"] = raw.iloc[:, 41].astype(str).str.strip().str.rstrip(".")
    f["label"] = (f.type != "normal").astype(int)
    f["_row_id"] = np.arange(len(f))
    return f.drop_duplicates([c for c in f if c != "_row_id"]).copy()


def _load_unsw(path):
    f = pd.read_csv(path, encoding="utf-8-sig").rename(columns={"attack_cat": "type"})
    if "id" in f.columns:
        f = f.drop(columns="id")
    f["_row_id"] = np.arange(len(f))
    return f.drop_duplicates([c for c in f if c != "_row_id"]).copy()


def _load_ton(path):
    f = pd.read_csv(path)
    f["_row_id"] = np.arange(len(f))
    f = f.drop_duplicates([c for c in f if c != "_row_id"]).copy()
    f.columns = [c.strip().lower() for c in f]
    f = f.replace(["-", "?", "", "NA", "N/A", "null", "None"], np.nan)
    return f


def _load_rt(path):
    f = pd.read_csv(path)
    for col in ("Unnamed: 0", "no"):
        if col in f.columns:
            f = f.drop(columns=col)
    f = f.rename(columns={"Attack_type": "type"})
    f["label"] = (~f.type.isin(["MQTT_Publish", "Thing_Speak", "Wipro_bulb"])).astype(int)
    f["_row_id"] = np.arange(len(f))
    predictors = [c for c in f if c not in DROP]
    return f.drop_duplicates(predictors + ["type"]).copy()


def load_dataset(name):
    path = FILES[name]
    if not path.is_file():
        raise FileNotFoundError(path)
    if name == "NSL-KDD":
        f = _load_nsl_kdd(path)
    elif name == "UNSW-NB15":
        f = _load_unsw(path)
    elif name == "ToN-IoT":
        f = _load_ton(path)
    elif name == "RT-IoT":
        f = _load_rt(path)
    else:
        raise ValueError(name)
    return _finalise(f, name)


def load_external_test(name):
    """Published test file for NSL-KDD / UNSW-NB15, or ``None``."""
    path = EXTERNAL_TESTS.get(name)
    if path is None or not path.is_file():
        return None
    if name == "NSL-KDD":
        f = _load_nsl_kdd(path)
    elif name == "UNSW-NB15":
        f = _load_unsw(path)
    else:
        return None
    return _finalise(f, name)


def grouped_split(f, fraction, seed):
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
                .split(f, groups=f._fingerprint))
    return f.iloc[a].copy(), f.iloc[b].copy()


def id_hash(values):
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def partitions(f, dataset, seed):
    """Group-disjoint train/tune/calibration/test partitions.

    The rule is identical to the frozen notebook.  For ToN-IoT and RT-IoT the
    reserved 20% is returned as the held-out ``test`` partition; for NSL-KDD
    and UNSW-NB15 the internal ``test`` is empty and the published external
    file is used instead (see :func:`load_external_test`).
    """
    if dataset == "ToN-IoT":
        dev, test = grouped_split(f, .2, seed)
        train, val = grouped_split(dev, .25, seed + 1)
        tune, cal = grouped_split(val, .5, seed + 2)
    elif dataset == "RT-IoT":
        dev, test = grouped_split(f, .2, 202709)
        train, val = grouped_split(dev, .25, seed)
        tune, cal = grouped_split(val, .5, seed + 1)
    else:
        train, val = grouped_split(f, .2, seed)
        tune, cal = grouped_split(val, .5, seed + 1)
        test = f.iloc[:0].copy()

    parts = dict(train=train, tune=tune, calibration=cal, test=test)
    seen = set()
    for key, part in parts.items():
        if seen & set(part._fingerprint):
            raise AssertionError(f"Group leakage: {dataset}/{seed}/{key}")
        seen |= set(part._fingerprint)
    return parts


def stratified_indices(frame, n, seed):
    """Indices of a class-stratified subsample of ``n`` rows (sorted)."""
    idx, _ = train_test_split(np.arange(len(frame)), train_size=min(n, len(frame)),
                              stratify=frame.label, random_state=seed + 10)
    return np.sort(idx)


def fingerprint_digest(part):
    return id_hash(part._row_id.to_numpy())

