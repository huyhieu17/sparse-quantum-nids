"""Frozen pipeline primitives.

Everything in this module is copied verbatim from
``ICC2027_Minimal_Training_Comparison.ipynb`` (cells 11, 12, 14, 16, 18) so
that the noise / held-out extension below evaluates *exactly* the same
preprocessing, encoders, fidelity kernel and SVC decision rule as the frozen
release.  Do not "improve" these functions: their round-off behaviour is part
of the released protocol.
"""

import time
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import Statevector, state_fidelity

BASIS = ["rx", "ry", "rz", "h", "cx"]
DROP = {"label", "type", "src_ip", "dst_ip", "dns_query", "http_uri",
        "_row_id", "_fingerprint", "_host_pair"}


# ---------------------------------------------------------------------------
# Feature front-end (cell 11)
# ---------------------------------------------------------------------------
class Features:
    def __init__(self, dimension=64):
        self.dimension = dimension

    def clean(self, df):
        X = df[self.columns].copy()
        for c in self.numeric:
            X[c] = pd.to_numeric(X[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        return X

    def fit(self, train):
        self.numeric, categorical = [], []
        self.dropped = []
        for c in train:
            if c in DROP:
                continue
            s = train[c]
            if s.notna().sum() == 0:
                self.dropped.append(c)
                continue
            ratio = pd.to_numeric(s, errors="coerce").notna().sum() / max(1, s.notna().sum())
            if ratio >= .95:
                self.numeric.append(c)
            elif s.nunique() <= 100:
                categorical.append(c)
            else:
                self.dropped.append(c)
        self.columns = self.numeric + categorical
        self.pipeline = Pipeline([
            ("preprocess", ColumnTransformer([
                ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                                  ("scale", StandardScaler())]), self.numeric),
                ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                  ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=10,
                                                           sparse_output=True))]), categorical),
            ], sparse_threshold=1.)),
            ("variance", VarianceThreshold(1e-12)),
            ("select", SelectKBest(f_classif, k=self.dimension)),
        ])
        base = self.pipeline[:-1].fit_transform(self.clean(train), train.label)
        self.pipeline.named_steps["select"].k = min(self.dimension, base.shape[1])
        self.pipeline.named_steps["select"].fit(base, train.label)
        select = self.pipeline.named_steps["select"]
        scores = np.nan_to_num(select.scores_[select.get_support()], nan=0., posinf=0., neginf=0.)
        scores = np.log1p(np.maximum(scores, 0.))
        self.importance = scores / scores.max() if scores.max() else np.ones(self.dimension)
        names = self.pipeline.named_steps["preprocess"].get_feature_names_out()
        names = names[self.pipeline.named_steps["variance"].get_support()][select.get_support()]
        self.names = names.tolist()
        self.real_dimensions = len(self.names)
        if self.real_dimensions < self.dimension:
            padding = self.dimension - self.real_dimensions
            self.importance = np.pad(self.importance, (0, padding))
            self.names.extend(f"zero_padding_{i}" for i in range(padding))
        return self
    def transform(self, df):
        X = self.pipeline.transform(self.clean(df))
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, float)
        if X.shape[1] < self.dimension:
            X = np.pad(X, ((0, 0), (0, self.dimension - X.shape[1])))
        if not np.isfinite(X).all():
            raise ValueError("Nonfinite preprocessed values")
        return X


# ---------------------------------------------------------------------------
# Encoders and analytic fidelity kernel (cell 12)
# ---------------------------------------------------------------------------
def normalize(X):
    X = np.asarray(X, dtype=float)
    if not np.isfinite(X).all():
        raise ValueError("State input must be finite")
    single = X.ndim == 1
    A = np.atleast_2d(X).copy()
    norms = np.linalg.norm(A, axis=1)
    zero = norms < 1e-12
    A /= np.where(zero, 1., norms)[:, None]
    A[zero] = 0.
    A[zero, 0] = 1.
    return A[0] if single else A


class RobustSparseEncoding:
    def __init__(self, support=16, anchor=True, task=False, robust=True):
        self.support, self.anchor, self.task, self.robust = support, anchor, task, robust

    def fit(self, X, y, importance):
        X = np.asarray(X, float)
        self.indices = np.sort(np.argsort(-np.asarray(importance), kind="stable")[:31 if self.anchor else 32])
        Z = X[:, self.indices]
        self.scale = np.maximum(np.quantile(np.abs(Z), .75, axis=0), .1)
        imp = np.asarray(importance)[self.indices]
        self.weights = (.05 + imp) if self.task else np.ones(len(imp))
        if self.anchor:
            self.weights = np.r_[self.weights, 1.]
        if not 1 <= self.support <= 32:
            raise ValueError("support out of range")
        return self

    def transform(self, X):
        Z = np.asarray(X, float)[..., self.indices].copy()
        if not np.isfinite(Z).all():
            raise ValueError("Nonfinite input")
        if self.robust:
            Z = np.sign(Z) * np.log1p(np.abs(Z) / self.scale)
        if self.anchor:
            Z = np.concatenate([Z, np.ones((*Z.shape[:-1], 1))], axis=-1)
        # Anchor competes for the support budget; it is not a free extra amplitude.
        if self.support < 32:
            idx = np.argsort(-(Z * Z * self.weights), axis=-1, kind="stable")[..., :self.support]
            out = np.zeros_like(Z)
            np.put_along_axis(out, idx, np.take_along_axis(Z, idx, axis=-1), axis=-1)
            Z = out
        return normalize(Z)


class DenseEncoder(RobustSparseEncoding):
    qubits = 5

    def __init__(self):
        super().__init__(support=32, anchor=True, task=False, robust=True)

    def fit(self, X, importance):
        return super().fit(X, None, importance)


def kernel(a, b):
    # Preserve frozen clipping and array operations (including roundoff).
    return np.clip(np.square(np.asarray(a) @ np.asarray(b).T), 0., 1.)



# ---------------------------------------------------------------------------
# Structured sparse (sub-cube) encoding (cell 14)
# ---------------------------------------------------------------------------
class StructuredSparseEncoding:
    def __init__(self, projection, support, loss=0., structured=True):
        self.projection, self.support, self.loss, self.structured = projection, support, loss, structured
        q = projection.qubits
        k = int(np.log2(support))
        if support != 2 ** k or not 1 <= support <= 2 ** q or not 0 <= loss <= 1:
            raise ValueError("Invalid support/loss")
        cubes = []
        for active in combinations(range(q), k):
            fixed = [b for b in range(q) if b not in active]
            for pattern in range(2 ** len(fixed)):
                base = sum(((pattern >> i) & 1) << b for i, b in enumerate(fixed))
                cubes.append([base + sum(((j >> i) & 1) << b for i, b in enumerate(active))
                              for j in range(support)])
        self.cubes = np.asarray(cubes)

    def transform(self, X):
        z = np.asarray(self.projection.transform(X))
        single = z.ndim == 1
        z = np.atleast_2d(z)
        out = np.zeros_like(z)
        # Chunk cube scoring to avoid an n_samples x cubes x support allocation.
        for start in range(0, len(z), 512):
            a = z[start:start + 512]
            top = np.argsort(-(a * a), axis=1, kind="stable")[:, :self.support]
            chosen = top.copy()
            if self.structured:
                energies = np.sum(a[:, self.cubes] ** 2, axis=2)
                best = np.argmax(energies, axis=1)
                topenergy = np.sum(np.take_along_axis(a, top, axis=1) ** 2, axis=1)
                use = energies[np.arange(len(a)), best] >= (1 - self.loss) * topenergy
                chosen[use] = self.cubes[best[use]]
            np.put_along_axis(out[start:start + len(a)], chosen,
                              np.take_along_axis(a, chosen, axis=1), axis=1)
        v = normalize(out)
        return v[0] if single else v


# ---------------------------------------------------------------------------
# Circuit synthesis and overlap (cell 16)
# ---------------------------------------------------------------------------
def synthesize_structured(v, seed=11):
    start = time.perf_counter()
    v = np.asarray(v, float)
    q = int(np.log2(len(v)))
    if len(v) != 2 ** q or not np.isfinite(v).all() or not np.isclose(np.linalg.norm(v), 1):
        raise ValueError("Invalid state")
    support = np.flatnonzero(v != 0)
    fixed = {}
    active = []
    for b in range(q):
        bits = (support >> b) & 1
        if np.all(bits == bits[0]):
            fixed[b] = int(bits[0])
        else:
            active.append(b)
    qc = QuantumCircuit(q)
    for b, bit in fixed.items():
        if bit:
            qc.x(b)
    if active:
        small = np.zeros(2 ** len(active))
        for j in support:
            small[sum(((int(j) >> b) & 1) << i for i, b in enumerate(active))] = v[j]
        qc.append(StatePreparation(small), active)
    elif v[support[0]] < 0:
        qc.global_phase = np.pi
    qc = transpile(qc, basis_gates=BASIS, optimization_level=1, seed_transpiler=seed, num_processes=1)
    return qc, dict(cx=qc.count_ops().get("cx", 0), depth=qc.depth(),
                    active_qubits=len(active), artifact_ms=1000 * (time.perf_counter() - start))


def remove_leading_cx_gates(qc):
    """Remove leading CX gates whose controls are always false."""
    dag = circuit_to_dag(qc)
    for w in dag.wires:
        for node in list(dag.nodes_on_wire(w, only_ops=True)):
            if node.name == "barrier":
                continue
            elif node.name != "cx":
                break
            elif node.qargs[0] == w:
                dag.remove_op_node(node)
            else:
                break
    return dag_to_circuit(dag)


def overlap(a, b, seed):
    c = transpile(a.compose(b.inverse()), basis_gates=BASIS, optimization_level=1,
                  approximation_degree=1., seed_transpiler=seed, num_processes=1)
    c = remove_leading_cx_gates(c)
    value = float(abs(Statevector.from_instruction(c).data[0]) ** 2)
    return value, int(c.count_ops().get("cx", 0)), c.depth()


# ---------------------------------------------------------------------------
# Training / evaluation (cell 18)
# ---------------------------------------------------------------------------
def train_model(encoder, X, y):
    start = time.perf_counter()
    refs = encoder.transform(X)
    model = SVC(C=1, kernel="precomputed", class_weight="balanced").fit(kernel(refs, refs), y)
    return {"encoder": encoder, "model": model, "refs": refs[model.support_],
            "train_seconds": time.perf_counter() - start}


def decision(a, X):
    z = a["encoder"].transform(X)
    m = a["model"]
    return np.concatenate([kernel(z[i:i + 512], a["refs"]) @ m.dual_coef_[0] + m.intercept_[0]
                           for i in range(0, len(z), 512)])


def quality(y, score):
    pred = np.asarray(score) >= 0
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(accuracy=accuracy_score(y, pred) * 100,
                macro_f1=f1_score(y, pred, average="macro") * 100,
                recall=100 * tp / (tp + fn), fpr=100 * fp / (fp + tn))

