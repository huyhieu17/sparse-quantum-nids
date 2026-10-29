"""Utilities for a controlled reimplementation of the Cirillo IDS benchmark.

Reference: Cirillo et al., Quantum Machine Intelligence 8:27 (2026).
The split rule, feature-map families and bundled Nairobi/Lagos/Perth snapshots
support matched comparisons; exact reproduction is not implied. Preprocessing,
routing, classifier choice and dataset caps must be disclosed per experiment.

V9 corrects Pegasos callable-kernel offset and repeated-support handling.
Historical cirillo-* baseline outputs are superseded, not silently recomputed.
See baselines/BASELINE_CORRECTION.md and baselines/REPORT_VI.md.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import load_dataset
from .experiment import SPARSE_CONFIG
from .pipeline import DROP

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = ROOT / "device_snapshots"

#: The three snapshots [9] used.  Same coupling map, different calibrations.
DEVICES = ("nairobi", "lagos", "perth")

#: ``[9]`` fixes both; only ``C`` and ``tau`` are exposed for the ablation.
PEGASOS_C = 1000.0
PEGASOS_STEPS = 100
SHOTS = 1024


# ---------------------------------------------------------------------------
# Frozen device calibrations
# ---------------------------------------------------------------------------
class SnapshotBackend:
    """Minimal ``BackendV1`` view of a vendored ``conf_*.json`` / ``props_*.json``.

    ``qiskit.providers.fake_provider`` in Qiskit 1.2 only ships the pulse (V1)
    backends, so ``FakeNairobiV2`` & friends must be reconstructed.  Aer's
    ``NoiseModel.from_backend`` only needs ``configuration()`` and
    ``properties()``, which is exactly what the snapshot files contain.
    """

    def __init__(self, name):
        import json

        from qiskit.providers.models import BackendConfiguration, BackendProperties

        directory = SNAPSHOT_DIR / name
        self.name = name
        conf = json.loads((directory / f"conf_{name}.json").read_text())
        props = json.loads((directory / f"props_{name}.json").read_text())
        self._configuration = BackendConfiguration.from_dict(conf)
        self._properties = BackendProperties.from_dict(props)
        self.num_qubits = int(self._configuration.n_qubits)

    def configuration(self):
        return self._configuration

    def properties(self):
        return self._properties

    @property
    def coupling_map(self):
        return [tuple(e) for e in self._configuration.coupling_map]

    def __repr__(self):
        return f"SnapshotBackend({self.name!r}, {self.num_qubits}q)"


@lru_cache(maxsize=None)
def load_device(name):
    """Load (and cache) one of the three frozen calibrations."""
    if name not in DEVICES:
        raise ValueError(f"unknown device {name!r}; expected one of {DEVICES}")
    return SnapshotBackend(name)


@lru_cache(maxsize=None)
def device_rates(name):
    """Reportable summary of a calibration (median CX / 1q / readout)."""
    backend = load_device(name)
    props = backend.properties()
    cx = [p.value for g in props.gates if g.gate == "cx" for p in g.parameters
          if "error" in str(p.name).lower()]
    sq = [p.value for g in props.gates if g.gate in ("sx", "x") for p in g.parameters
          if "error" in str(p.name).lower()]
    readout = [props.readout_error(q) for q in range(backend.num_qubits)]
    return dict(device=name, qubits=backend.num_qubits,
                cx_median=float(np.median(cx)), cx_max=float(np.max(cx)),
                oneq_median=float(np.median(sq)) if sq else 0.0,
                readout_median=float(np.median(readout)),
                readout_max=float(np.max(readout)))


@lru_cache(maxsize=None)
def device_noise_model(name):
    """``NoiseModel.from_backend`` on the frozen calibration (gate + thermal)."""
    from qiskit_aer.noise import NoiseModel

    return NoiseModel.from_backend(load_device(name))


# ---------------------------------------------------------------------------
# Data protocol: stratified 80/20, per-class 15 % subsample, normalise + PCA
# ---------------------------------------------------------------------------
def _per_class_subsample(y, frac, seed):
    """Positions of a ``frac`` subsample drawn independently within each class.

    [9]: "a balanced 15 % sample per class was used, resulting in 11,270
    training and 2,818 testing samples" -- with 125,973 NSL-KDD rows this is a
    *per-class 15 %* draw (15,114 / 3,780), i.e. the class ratio is preserved,
    not equalised.
    """
    rng = np.random.default_rng(seed)
    keep = []
    for label in np.unique(y):
        pos = np.flatnonzero(y == label)
        take = max(1, int(round(frac * len(pos))))
        keep.append(rng.choice(pos, size=min(take, len(pos)), replace=False))
    return np.sort(np.concatenate(keep))


def _columns(frame):
    """Numeric / categorical split, using the frozen front-end's own rule."""
    numeric, categorical = [], []
    for c in frame.columns:
        if c in DROP:
            continue
        s = frame[c]
        if s.notna().sum() == 0:
            continue
        ratio = pd.to_numeric(s, errors="coerce").notna().sum() / max(1, s.notna().sum())
        if ratio >= .95:
            numeric.append(c)
        elif s.nunique() <= 100:
            categorical.append(c)
    return numeric, categorical


def protocol_split(dataset, seed, test_size=0.2, frac=0.15):
    """``(frame, train_positions, test_positions)`` for the [9] split rule."""
    frame = load_dataset(dataset)
    y = frame.label.to_numpy()
    tr, te = train_test_split(np.arange(len(frame)), test_size=test_size,
                              stratify=y, random_state=seed)
    tr = np.sort(tr[_per_class_subsample(y[tr], frac, seed + 1)])
    te = np.sort(te[_per_class_subsample(y[te], frac, seed + 2)])
    return frame, tr, te




def protocol_features(dataset, q, seed, test_size=0.2, frac=0.15, angle_scale=np.pi):
    """Run the [9] pre-processing and return PCA scores plus angle vectors.

    Returns a dict with

    ``Z``      PCA scores ``(train, test)`` -- what our amplitude encoders and
               the classical baselines consume (one classical feature per
               qubit is *not* imposed on them, see ``REPLICATION_C9.md``);
    ``A``      the same components min-max scaled to ``[0, angle_scale]`` --
               what the angle feature maps of [9] rotate by;
    ``y``      labels, ``index`` the original row positions, ``explained`` the
               PCA variance ratio.
    """
    from sklearn.compose import ColumnTransformer

    frame, tr, te = protocol_split(dataset, seed, test_size=test_size, frac=frac)
    numeric, categorical = _columns(frame)
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("encode", OneHotEncoder(handle_unknown="ignore",
                                                   min_frequency=10,
                                                   sparse_output=True))]), categorical),
    ], sparse_threshold=1.)
    X = frame[numeric + categorical]
    Xtr = pre.fit_transform(X.iloc[tr])
    Xte = pre.transform(X.iloc[te])
    if hasattr(Xtr, "toarray"):
        Xtr, Xte = Xtr.toarray(), Xte.toarray()
    Xtr = np.asarray(Xtr, float)
    Xte = np.asarray(Xte, float)

    centre = StandardScaler().fit(Xtr)
    Xtr, Xte = centre.transform(Xtr), centre.transform(Xte)
    pca = PCA(n_components=q, random_state=seed).fit(Xtr)
    Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)

    lo, hi = Ztr.min(axis=0), Ztr.max(axis=0)
    span = np.where(hi - lo < 1e-12, 1., hi - lo)
    Atr = np.clip(angle_scale * (Ztr - lo) / span, 0., angle_scale)
    Ate = np.clip(angle_scale * (Zte - lo) / span, 0., angle_scale)

    return dict(dataset=dataset, q=q, seed=seed,
                Z=(Ztr, Zte), A=(Atr, Ate),
                y=(frame.label.to_numpy()[tr], frame.label.to_numpy()[te]),
                index=(tr, te), explained=pca.explained_variance_ratio_,
                n_features=int(Xtr.shape[1]))


def stratified_cap(y, n, seed):
    """Positions of a class-stratified subsample with at most ``n`` rows.

    The device conditions cannot afford the full 2,818 / 3,780-row test
    partition, so they use a stratified ``n``-row subsample (class *ratios*
    preserved, so accuracy stays comparable with the noiseless full-partition
    row).  ``n <= 0`` keeps every row.
    """
    if n <= 0 or n >= len(y):
        return np.arange(len(y))
    idx, _ = train_test_split(np.arange(len(y)), train_size=n, stratify=y,
                              random_state=seed)
    return np.sort(idx)



# ---------------------------------------------------------------------------
# Circuits and kernels
# ---------------------------------------------------------------------------
def angle_circuit(kind, q, reps=1, entanglement="linear"):
    """The angle feature maps benchmarked in [9]."""
    from qiskit.circuit.library import PauliFeatureMap, ZFeatureMap, ZZFeatureMap

    if kind == "zz":
        return ZZFeatureMap(q, reps=reps, entanglement=entanglement)
    if kind == "z":
        return ZFeatureMap(q, reps=reps)
    if kind == "pauli":
        return PauliFeatureMap(q, reps=reps, paulis=["Z", "YY"])
    raise ValueError(kind)


def feature_map_circuits(A, kind, reps=1, entanglement="linear"):
    """Bind every row of the angle matrix into the requested feature map."""
    template = angle_circuit(kind, A.shape[1], reps=reps, entanglement=entanglement)
    params = template.parameters
    return [template.assign_parameters(row) for row in np.asarray(A, float)]


def amplitude_circuits(Z_train, Z_test, support=32, loss=0.):
    """This paper's encoders applied to the PCA front-end.

    ``support=32`` is the dense amplitude encoder (each of the 32 PCA
    components becomes an amplitude); ``support=2**k < 32`` is the structured
    sub-cube encoder of the paper.  ``support``/``loss`` must be the paper's own
    per-dataset choice (``src.experiment.SPARSE_CONFIG``: ToN-IoT ``8/1.0``,
    NSL-KDD ``16/0.3``), otherwise the row is not the encoder the paper reports.
    The importance vector is uniform because the PCA components replace the
    ANOVA ranking of the released front-end.  The encoder is fitted on the
    training rows only.
    """
    from .pipeline import DenseEncoder, StructuredSparseEncoding, synthesize_structured

    def pad(X):
        X = np.asarray(X, float)[:, :32]
        return np.pad(X, ((0, 0), (0, 32 - X.shape[1]))) if X.shape[1] < 32 else X

    Z_train, Z_test = pad(Z_train), pad(Z_test)
    dense = DenseEncoder().fit(Z_train, np.ones(32))
    enc = dense if support == 32 else StructuredSparseEncoding(dense, support=support,
                                                               loss=loss)
    out = [enc.transform(Z_train), enc.transform(Z_test)]
    circuits, cxs = [], []
    for block in out:
        for v in np.atleast_2d(block):
            qc, info = synthesize_structured(v)
            circuits.append(qc)
            cxs.append(info["cx"])
    return circuits[:len(out[0])], circuits[len(out[0]):], float(np.mean(cxs))


class StatevectorKernel:
    """Exact ``|<psi_i|psi_j>|^2`` from statevectors (noiseless, no shots)."""

    def __init__(self, circuits, seed=11):
        from qiskit.quantum_info import Statevector

        self.states = np.asarray([Statevector.from_instruction(c).data
                                  for c in circuits], dtype=complex)

    def rows(self, i, js):
        return np.clip(np.abs(self.states[i] @ self.states[js].conj().T) ** 2, 0., 1.)

    def block(self, ii, jj):
        return np.clip(np.abs(self.states[ii] @ self.states[jj].conj().T) ** 2, 0., 1.)


class DeviceKernel:
    """1024-shot fidelity kernel on one frozen calibration.

    Each logical overlap is composed before routing. Physical calibration
    channels are restricted and remapped to the same local qubit coordinates.
    """

    def __init__(self, circuits, device, layout, shots=SHOTS, seed=11, chunk=256,
                 noise_model=None):
        from qiskit_aer import AerSimulator

        from .noise import overlap_circuit
        from .topology import remap_noise_model, device_overlap_circuit

        backend = load_device(device)
        self.device, self.layout, self.shots, self.seed = device, list(layout), shots, seed
        self.chunk = chunk
        self.circuits = list(circuits)
        self.backend = backend
        self._device_overlap = device_overlap_circuit
        self._overlap = overlap_circuit
        nm = (remap_noise_model(device_noise_model(device), layout)
              if noise_model is None else noise_model)
        self.sim = AerSimulator(noise_model=nm, method="density_matrix",
                                seed_simulator=seed, max_parallel_threads=1)
        self.n_qubits = self.circuits[0].num_qubits
        self.calls = 0
        self.executed_circuits = self.executed_cx = self.self_circuits = self.self_cx = 0

    def overlap(self, i, j):
        # Compose logically before routing: different preparations can have
        # different final permutations after independent routing.
        return self._device_overlap(self.circuits[i], self.circuits[j],
                                    self.backend, self.layout, seed=self.seed,
                                    optimise=False)

    def record_cost(self, circuit, self_overlap=False):
        cx = circuit.count_ops().get("cx", 0)
        self.executed_circuits += 1
        self.executed_cx += cx
        if self_overlap:
            self.self_circuits += 1
            self.self_cx += cx

    def rows(self, i, js):
        js = np.asarray(js, int)
        out = np.empty(len(js))
        for start in range(0, len(js), self.chunk):
            part = js[start:start + self.chunk]
            circuits = []
            for j in part:
                c = self.overlap(int(i), int(j))
                self.record_cost(c, int(i) == int(j))
                c.measure_all()
                circuits.append(c)
            result = self.sim.run(circuits, shots=self.shots,
                                  seed_simulator=self.seed + self.calls).result()
            self.calls += 1
            zero = "0" * self.n_qubits
            out[start:start + len(part)] = [
                result.get_counts(k).get(zero, 0) / self.shots for k in range(len(part))]
        return out


class PairCache:
    """Memoise single kernel entries so no circuit is sampled twice."""

    def __init__(self, kernel):
        self.kernel, self.entries, self.evaluations = kernel, {}, 0

    def row(self, i, js):
        js = np.asarray(js, int)
        missing = sorted({int(j) for j in js if (int(i), int(j)) not in self.entries})
        if missing:
            values = self.kernel.rows(int(i), np.array(missing))
            self.evaluations += len(missing)
            for j, v in zip(missing, values):
                self.entries[(int(i), j)] = float(v)
        return np.array([self.entries[(int(i), int(j))] for j in js])


# ---------------------------------------------------------------------------
# Pegasos-QSVC ([9]: C = 1000, tau = 100) and our kernel SVC
# ---------------------------------------------------------------------------
def pegasos_qsvc(cache, n_train, y, C=PEGASOS_C, steps=PEGASOS_STEPS, seed=11,
                 balance=False, kernel_offset=1.0):
    """Qiskit callable-kernel Pegasos update, expressed with attack-positive scores.

    Qiskit 0.7 adds an offset of one to every callable-kernel entry to represent
    an implicit bias. Each active support index must occur exactly once.
    ``kernel_offset=0`` is an explicit diagnostic, not the callable default.
    """
    rng = np.random.default_rng(seed)
    y = np.where(np.asarray(y) > 0, 1., -1.)
    pools = [np.flatnonzero(y == c) for c in (-1., 1.)] if balance else None
    alpha, updates = {}, 0
    for step in range(1, steps + 1):
        i = int(rng.choice(pools[step % 2])) if balance else int(rng.integers(n_train))
        js = np.array(list(alpha), dtype=int)
        coef = np.array([alpha[int(j)] * y[int(j)] for j in js])
        value = float(coef @ (cache.row(i, js) + kernel_offset)) if len(js) else 0.
        if (y[i] * C / step) * value < 1.:
            alpha[i] = alpha.get(i, 0) + 1
            updates += 1
    sv = np.array(list(alpha), dtype=int)
    coef = np.array([alpha[int(j)] * y[int(j)] for j in sv])
    return dict(sv=sv, alpha=np.array([alpha[int(j)] for j in sv]),
                coef=coef, bias=float(kernel_offset * coef.sum()), updates=updates,
                kernel_offset=kernel_offset)


def pegasos_score(cache, model, idx):
    return np.array([float(model["coef"] @ cache.row(int(i), model["sv"]))
                     + model.get("bias", 0.) for i in idx])


def svc_on_kernel(K_train, y_train, C=1.0):
    from sklearn.svm import SVC

    return SVC(C=C, kernel="precomputed", class_weight="balanced").fit(K_train, y_train)


def svc_refs(svc, max_refs=0):
    """Support vectors and their dual coefficients (optionally capped)."""
    refs = np.asarray(svc.support_, int)
    coef = np.asarray(svc.dual_coef_[0], float)
    if max_refs and len(refs) > max_refs:
        order = np.argsort(-np.abs(coef), kind="stable")[:max_refs]
        refs, coef = refs[order], coef[order]
    order = np.argsort(refs)
    return refs[order], coef[order]


def svc_score(cache, svc, idx, max_refs=0):
    """Kernel-SVC decision for query positions ``idx`` (analytic train kernel)."""
    refs, coef = svc_refs(svc, max_refs)
    scores = np.array([float(cache.row(int(i), refs) @ coef) for i in idx])
    return scores + svc.intercept_[0], len(refs)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metrics(y, score):
    """Accuracy / F1 / precision / recall, plus both single-class recalls.

    ``[9]`` report *one* precision/recall/F1 per configuration (their Tables 11
    and 12), and their NSL-KDD row ``P = 96.68``, ``R = 93.82``, ``F1 = 95.23``
    is exactly the harmonic mean of the two, i.e. a single-class figure.  This
    paper's ``recall`` column is attack recall (``src.pipeline.decision``), so
    both are returned: ``recall`` is the macro average over the two classes,
    ``recall_attack`` is ``TP / (TP + FN)`` on the attack class and
    ``f1_attack`` its binary F1.  The four confusion counts travel with the row
    so a reader can recompute any of them.
    """
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 confusion_matrix, f1_score, precision_score,
                                 recall_score)

    y = np.asarray(y)
    pred = np.asarray(score) >= 0
    tn, fp, fn, tp = confusion_matrix(y, pred.astype(int), labels=[0, 1]).ravel()
    return dict(accuracy=100 * accuracy_score(y, pred),
                macro_f1=100 * f1_score(y, pred, average="macro"),
                weighted_f1=100 * f1_score(y, pred, average="weighted"),
                balanced_accuracy=100 * balanced_accuracy_score(y, pred),
                precision=100 * precision_score(y, pred, average="macro", zero_division=0),
                recall=100 * recall_score(y, pred, average="macro", zero_division=0),
                recall_attack=100 * tp / max(1, int(tp) + int(fn)),
                recall_normal=100 * tn / max(1, int(tn) + int(fp)),
                f1_attack=100 * f1_score(y, pred, pos_label=1, zero_division=0),
                tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))


# ---------------------------------------------------------------------------
# What is run
# ---------------------------------------------------------------------------
#: ``[9]``'s reference rows: the headline configuration (q=7 ZFeatureMap r=2,
#: the one that reaches 94.60 % / 94.13 % on NSL-KDD under FakeNairobi) and its
#: best *entangled* configuration.
MODEL_SPECS = {
    "cirillo-z7r2": dict(kind="z", q=7, reps=2, entanglement="linear"),
    "cirillo-zz6r1": dict(kind="zz", q=6, reps=1, entanglement="linear"),
    # this paper: amplitude encoding of the same PCA front-end, five qubits
    "ours-dense32": dict(kind="amp", q=32, support=32),
    "ours-sparse16": dict(kind="amp", q=32, sparse=True),
    "ours-dense-pca5": dict(kind="amp", q=5, support=32),
}

CLASSICAL_SPECS = {
    "svm-rbf-pca7": dict(q=7, kind="svm"),
    "rf-pca7": dict(q=7, kind="rf"),
    "svm-rbf-pca32": dict(q=32, kind="svm"),
}

MODELS = tuple(MODEL_SPECS) + tuple(CLASSICAL_SPECS)


@lru_cache(maxsize=None)
def _features(dataset, q, seed, angle_scale):
    return protocol_features(dataset, q, seed, angle_scale=angle_scale)


def _layout(device, qubits):
    from .topology import select_layout

    return select_layout(load_device(device), qubits)["layout"]


def run_condition(dataset, seed, model, device, *, train_cap=600, test_cap=200,
                  shots=SHOTS, angle_scale=np.pi, max_refs=100, balance_train=False):
    """One row of the replication table.

    ``device="ideal"`` evaluates the *full* protocol test partition with exact
    statevector overlaps (comparable with [9]'s noiseless table); a device name
    evaluates a stratified ``test_cap``-row subsample with 1024 shots.

    The wall clock is split the way ``[9]`` report it (fit vs. prediction,
    Figs. 6-7): ``feat_seconds`` is the classical front-end, ``encode_seconds``
    builds the circuits, ``prep_seconds`` simulates/maps them, ``fit_seconds``
    is the training loop (including the kernel entries it triggers) and
    ``predict_seconds`` scores the test rows.  ``seconds`` stays the total.
    """
    import time

    start = t_mark = time.perf_counter()

    def _lap():
        nonlocal t_mark
        now = time.perf_counter()
        out = now - t_mark
        t_mark = now
        return out

    if model in CLASSICAL_SPECS:
        spec = CLASSICAL_SPECS[model]
        feats = _features(dataset, spec["q"], seed, angle_scale)
        t_feat = _lap()
        Ztr, Zte = feats["Z"]
        ytr, yte = feats["y"]
        tr = stratified_cap(ytr, train_cap, seed + 3)
        te = np.arange(len(yte)) if device == "ideal" else \
            stratified_cap(yte, test_cap, seed + 5)
        if spec["kind"] == "svm":
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import SVC
            clf = make_pipeline(StandardScaler(),
                                SVC(C=1, kernel="rbf", gamma="scale",
                                    class_weight="balanced"))
        else:
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                         random_state=seed, n_jobs=1)
        clf.fit(Ztr[tr], ytr[tr])
        t_fit = _lap()
        proba = clf.decision_function(Zte[te]) if hasattr(clf, "decision_function") \
            else clf.predict_proba(Zte[te])[:, 1] - .5
        t_pred = _lap()
        row = dict(dataset=dataset, seed=seed, model=model, device="ideal",
                   q=spec["q"], qubits=0, n_train=len(tr), n_test=len(te),
                   n_refs=0, kernel_evals=0, cx_mean=0.0, shots=0,
                   angle_scale=angle_scale,
                   feat_seconds=t_feat, encode_seconds=0.0, prep_seconds=0.0,
                   fit_seconds=t_fit, predict_seconds=t_pred,
                   seconds=time.perf_counter() - start, **metrics(yte[te], proba))
        return row

    spec = MODEL_SPECS[model]
    feats = _features(dataset, spec["q"], seed, angle_scale)
    t_feat = _lap()
    ytr, yte = feats["y"]
    tr = stratified_cap(ytr, train_cap, seed + 3)
    te = stratified_cap(yte, test_cap, seed + 5)

    if spec["kind"] == "amp":
        Ztr, Zte = feats["Z"]
        if spec.get("sparse"):
            support, loss = SPARSE_CONFIG.get(dataset, (16, .3))
        else:
            support, loss = spec["support"], 0.
        train_circuits, test_circuits, cx_mean = amplitude_circuits(
            Ztr[tr], Zte[te], support=support, loss=loss)
        qubits = 5
    else:
        Atr, Ate = feats["A"]
        rows = np.vstack([Atr[tr], Ate[te]])
        all_circuits = feature_map_circuits(rows, spec["kind"], spec["reps"],
                                            spec["entanglement"])
        train_circuits = all_circuits[:len(tr)]
        test_circuits = all_circuits[len(tr):]
        cx_mean = float(np.mean([c.count_ops().get("cx", 0) for c in all_circuits]))
        qubits = spec["q"]
    t_enc = _lap()

    circuits = list(train_circuits) + list(test_circuits)
    n_tr, n_te = len(train_circuits), len(test_circuits)
    if device == "ideal":
        kernel = StatevectorKernel(circuits, seed=seed)
    else:
        kernel = DeviceKernel(circuits, device, _layout(device, qubits),
                              shots=shots, seed=seed)
    t_prep = _lap()

    if model.startswith("cirillo-"):
        cache = PairCache(kernel)
        fit = pegasos_qsvc(cache, n_tr, ytr[tr], seed=seed, balance=balance_train)
        t_fit = _lap()
        score = pegasos_score(cache, fit, np.arange(n_tr, n_tr + n_te))
        t_pred = _lap()
        n_refs = len(fit["sv"])
        evaluations = cache.evaluations
    else:
        train_kernel = StatevectorKernel(train_circuits, seed=seed)
        K = train_kernel.block(np.arange(n_tr), np.arange(n_tr))
        svc = svc_on_kernel(K, ytr[tr])
        refs, coef = svc_refs(svc, max_refs)
        n_refs = len(refs)
        t_fit = _lap()
        if device == "ideal":
            queries = np.arange(n_tr, n_tr + n_te)
            block = kernel.block(queries, refs)
            score = block @ coef + svc.intercept_[0]
            evaluations = int(block.size)
        else:
            cache = PairCache(kernel)
            score = np.array([float(cache.row(int(i), refs) @ coef)
                              for i in np.arange(n_tr, n_tr + n_te)]) \
                + svc.intercept_[0]
            evaluations = cache.evaluations
        t_pred = _lap()

    return dict(dataset=dataset, seed=seed, model=model, device=device,
                q=spec["q"], qubits=qubits, n_train=n_tr, n_test=n_te,
                n_refs=n_refs, kernel_evals=evaluations, cx_mean=cx_mean,
                executed_circuits=getattr(kernel,"executed_circuits",0),
                executed_cx=getattr(kernel,"executed_cx",0),
                shots=shots if device != "ideal" else 0, angle_scale=angle_scale,
                feat_seconds=t_feat, encode_seconds=t_enc, prep_seconds=t_prep,
                fit_seconds=t_fit, predict_seconds=t_pred,
                seconds=time.perf_counter() - start, **metrics(yte[te], score))



def overlap_cx(model, dataset="NSL-KDD", seed=11, angle_scale=np.pi, n=6):
    """Two-qubit gates inside the *overlap* circuit of ``model`` (unrouted).

    ``run_condition`` reports the CX count of the preparation circuit, which is
    not comparable across families (a ZFeatureMap preparation has none, an
    amplitude preparation has ~26).  An overlap is ``U(x) U(y)^dagger``, so the
    number that decides the noise sensitivity is the count after composition --
    the same quantity Section 6 reports (52 for dense, 7.6-25.6 for sparse).
    This recomputes it from scratch for the comparison table.
    """
    from .noise import overlap_circuit
    from .pipeline import BASIS, transpile

    spec = MODEL_SPECS[model]
    feats = _features(dataset, spec["q"], seed, angle_scale)
    if spec["kind"] == "amp":
        if spec.get("sparse"):
            support, loss = SPARSE_CONFIG.get(dataset, (16, .3))
        else:
            support, loss = spec["support"], 0.
        circuits, _, _ = amplitude_circuits(feats["Z"][0][:n], feats["Z"][1][:n],
                                            support=support, loss=loss)
    else:
        circuits = feature_map_circuits(feats["A"][0][:n], spec["kind"], spec["reps"],
                                        spec["entanglement"])
    counts = []
    for i in range(len(circuits) - 1):
        c = overlap_circuit(circuits[i], circuits[i + 1], seed=seed, optimise=True)
        c = transpile(c, basis_gates=BASIS, optimization_level=1,
                      seed_transpiler=seed, num_processes=1)
        counts.append(sum(1 for inst in c.data if len(inst.qubits) == 2))
    return float(np.mean(counts))


def overlap_shape(model, dataset="NSL-KDD", seed=11, angle_scale=np.pi, n=8,
                  device=None):
    """Circuit cost of one overlap for ``model``: qubits, two-qubit gates, depth.

    An overlap is ``U(x) U(y)^dagger`` -- the circuit that is sampled per kernel
    entry -- so both the gate count and the depth scale with ``queries x
    references``.  ``device=None`` transpiles to the frozen basis
    (:data:`src.pipeline.BASIS`) exactly as :func:`overlap_cx` does; a device
    name maps every preparation onto that snapshot's coupling map first
    (:func:`src.topology.map_circuit`), which is the circuit ``DeviceKernel``
    samples and therefore the depth that carries the device noise.

    ``[9]`` report neither gate counts nor depths (they describe complexity
    only qualitatively, e.g. "FakeLagos is more sensitive to feature maps with
    a higher number of CX gates"), so this measures both families instead of
    quoting theirs.  ``n`` pairs are averaged; ``max`` is kept because a single
    deeper overlap is what dominates the worst sampled entry.
    """
    from .noise import overlap_circuit
    from .pipeline import BASIS, transpile
    from .topology import map_circuit

    spec = MODEL_SPECS[model]
    feats = _features(dataset, spec["q"], seed, angle_scale)
    if spec["kind"] == "amp":
        if spec.get("sparse"):
            support, loss = SPARSE_CONFIG.get(dataset, (16, .3))
        else:
            support, loss = spec["support"], 0.
        circuits, _, _ = amplitude_circuits(feats["Z"][0][:n], feats["Z"][1][:n],
                                            support=support, loss=loss)
        qubits = 5
    else:
        circuits = feature_map_circuits(feats["A"][0][:n], spec["kind"], spec["reps"],
                                        spec["entanglement"])
        qubits = spec["q"]

    def _twoq(c):
        return sum(1 for inst in c.data if len(inst.qubits) == 2)

    logical = dict(prep_cx=float(np.mean([_twoq(c) for c in circuits])),
                   prep_depth=float(np.mean([c.depth() for c in circuits])))

    if device:
        backend = load_device(device)
        layout = _layout(device, qubits)
        mapped = [map_circuit(c, backend, layout, seed=seed) for c in circuits]
    else:
        mapped = [transpile(c, basis_gates=BASIS, optimization_level=1,
                            seed_transpiler=seed, num_processes=1) for c in circuits]

    overlaps = []
    for i in range(len(circuits) - 1):
        if device:
            from .topology import device_overlap_circuit
            c = device_overlap_circuit(circuits[i], circuits[i + 1], backend, layout,
                                       seed=seed, optimise=False)
        else:
            c = overlap_circuit(circuits[i], circuits[i + 1], seed=seed, optimise=True)
        overlaps.append(c)

    counts = [_twoq(c) for c in overlaps]
    depths = [c.depth() for c in overlaps]
    return dict(model=model, dataset=dataset, device=device or "ideal",
                qubits=qubits, n_pairs=len(overlaps),
                prep_cx=float(np.mean([_twoq(c) for c in mapped])),
                prep_depth=float(np.mean([c.depth() for c in mapped])),
                prep_cx_logical=logical["prep_cx"],
                prep_depth_logical=logical["prep_depth"],
                overlap_cx=float(np.mean(counts)),
                overlap_cx_max=float(np.max(counts)),
                overlap_depth=float(np.mean(depths)),
                overlap_depth_max=float(np.max(depths)),
                overlap_size=float(np.mean([c.size() for c in overlaps])))

