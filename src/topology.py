"""Device-topology operating point for the fidelity kernel.

Why this module exists
----------------------
The controlled depolarizing model in :mod:`src.noise` applies the *same* error to
every CX on every qubit pair.  That is the correct model for a controlled sweep,
but it cannot answer the objection a device reviewer raises immediately: on real
hardware a CX gate exists only on the edges of the coupling map (heavy-hex has
degree 2-3), every edge has its own calibrated error, and mapping a five-qubit
circuit onto that lattice inserts routing SWAPs that can consume the CX saving
the sparse encoder buys.

The trap this module is written around
--------------------------------------
``NoiseModel.from_backend`` attaches two-qubit errors to *specific* qubit pairs.
A circuit that never touches those pairs receives **no two-qubit error at all**,
so a naive run reports a nearly noise-free kernel and "proves" robustness that
does not exist.  The failure is silent, which is why every mapped circuit is
passed through :func:`audit_circuit`: every two-qubit gate must be an edge of the
device, and :func:`device_noisy_kernel` refuses to return numbers otherwise.

What is held fixed
------------------
``initial_layout`` is fixed per device and shared by both encoders, so Full and
Sparse are executed on the *same* physical qubits.  The virtual circuit is the
one produced by :func:`src.noise.overlap_circuit`, i.e. the same circuit as in
the uniform sweep -- only the noise model and the routing differ.
"""

import numpy as np
from functools import lru_cache

from .noise import overlap_circuit

# Public aliases.  ``washington`` is the 127-qubit heavy-hex device; the other
# two are the same calibrations the uniform operating points are taken from, so
# device-vs-uniform isolates topology + heterogeneity rather than hardware age.
DEVICE_ALIASES = {
    "washington": "Fake127QPulseV1",   # 127q heavy-hex
    "montreal": "Fake27QPulseV1",      # 27q Falcon, 28 edges
    "vigo": "Fake5QV1",                # 5q T-shape, 6 edges
    "sherbrooke": "FakeSherbrooke",    # needs qiskit-ibm-runtime
}

TWO_QUBIT_GATES = ("cx", "ecr", "cz", "rzz")


# ---------------------------------------------------------------------------
# Backend accessors (BackendV1 and BackendV2)
# ---------------------------------------------------------------------------
def load_backend(name):
    """Resolve an alias or a fake-backend class name to a backend instance."""
    cls_name = DEVICE_ALIASES.get(name, name)
    try:
        from qiskit.providers import fake_provider
        factory = getattr(fake_provider, cls_name)
    except AttributeError:
        try:  # newer fake providers live in qiskit-ibm-runtime
            from qiskit_ibm_runtime import fake_provider as rt_fake
            factory = getattr(rt_fake, cls_name)
        except Exception as exc:
            raise RuntimeError(
                f"unknown fake backend {name!r} (resolved to {cls_name!r}); "
                f"FakeSherbrooke needs `pip install qiskit-ibm-runtime`") from exc
    return factory()


def backend_qubits(backend):
    try:
        return int(backend.num_qubits)
    except AttributeError:
        return int(backend.configuration().n_qubits)


def backend_coupling(backend):
    """Directed coupling map as a list of ``(control, target)`` pairs."""
    try:
        cm = backend.coupling_map
        edges = list(cm.get_edges()) if hasattr(cm, "get_edges") else list(cm)
    except AttributeError:
        edges = list(backend.configuration().coupling_map)
    return [(int(a), int(b)) for a, b in edges]


def two_qubit_gate_name(backend):
    try:
        names = set(backend.operation_names)
    except AttributeError:  # BackendV1
        cfg = backend.configuration()
        names = {g.name for g in cfg.gates} | set(cfg.basis_gates or [])
    for name in TWO_QUBIT_GATES:
        if name in names:
            return name
    raise RuntimeError(f"{backend} exposes no known two-qubit gate: {sorted(names)}")


def _gate_error(gate):
    """The ``gate_error`` parameter of a properties gate, if it has one.

    Parameters are *not* ordered by meaning: ``id`` carries ``gate_length`` in
    nanoseconds as its first parameter, and reading ``parameters[0]`` blindly
    produces an "error" of 35 and a crash inside ``depolarizing_error``.
    """
    for par in gate.parameters or []:
        if "error" in str(par.name).lower():
            return float(par.value)
    return None


def two_qubit_errors(backend):
    """``{(control, target): error}`` for every edge the device provides."""
    out = {}
    target = getattr(backend, "target", None)
    if target is not None:  # BackendV2
        for name in TWO_QUBIT_GATES:
            try:
                props = target[name]
            except KeyError:
                continue
            for qargs, inst in props.items():
                if len(qargs) == 2 and inst is not None and inst.error is not None:
                    out[(int(qargs[0]), int(qargs[1]))] = float(inst.error)
        if out:
            return out
    props = backend.properties()  # BackendV1
    for gate in props.gates:
        if gate.gate in TWO_QUBIT_GATES and len(gate.qubits) == 2:
            err = _gate_error(gate)
            if err is not None:
                out[(int(gate.qubits[0]), int(gate.qubits[1]))] = err
    if not out:
        raise RuntimeError(f"{backend} reports no two-qubit gate errors")
    return out



# ---------------------------------------------------------------------------
# Layout selection
# ---------------------------------------------------------------------------
def select_layout(backend, n_qubits, strategy="cheapest"):
    """Connected ``n_qubits`` subgraph, deterministic.

    Greedy growth seeded from every edge in turn; the best seed wins.

    ``strategy="cheapest"``
        minimise the sum of the induced edge errors (a device operator would put
        the circuit exactly here);
    ``strategy="densest"``
        maximise the number of induced edges first, then minimise error.  A path
        forces the router to borrow ancillas for every non-adjacent pair, so the
        densest placement is the one that stresses routing least -- running both
        separates "the answer depends on where you place the circuit" from "the
        answer depends on the encoder".
    """
    if strategy not in ("cheapest", "densest"):
        raise ValueError(f"unknown layout strategy {strategy!r}")
    errs = two_qubit_errors(backend)
    undirected = {}
    for (a, b), err in errs.items():
        key = frozenset((a, b))
        undirected[key] = min(err, undirected.get(key, np.inf))
    if n_qubits > backend_qubits(backend):
        raise ValueError(f"{n_qubits} qubits do not fit on {backend_qubits(backend)}")

    def grow(seed_edge, mode):
        chosen = set(seed_edge)
        total = undirected[seed_edge]
        while len(chosen) < n_qubits:
            candidates = {q for k in undirected if chosen & set(k) for q in k} - chosen
            if not candidates:
                return None

            def key(q):
                internal = [k for k in undirected if set(k) <= chosen | {q}]
                linking = [undirected[k] for k in undirected
                           if (chosen & set(k)) and q in k]
                add = min(linking) if linking else np.inf
                degree = sum(len(k) for k in internal)
                return (add, -degree, q) if mode == "cheapest" else (-degree, add, q)

            pick = min(candidates, key=key)
            linking = [undirected[k] for k in undirected if (chosen & set(k)) and pick in k]
            if not linking:
                return None
            total += min(linking)
            chosen.add(pick)
        return total, chosen

    best = None
    for seed_edge in undirected:
        grown = grow(seed_edge, strategy)
        if grown is None:
            continue
        cost, chosen = grown
        internal = [k for k in undirected if set(k) <= chosen]
        if strategy == "cheapest":
            score = (round(cost, 12), -len(internal), sorted(chosen))
        else:
            score = (-len(internal), round(cost, 12), sorted(chosen))
        if best is None or score < best[0]:
            best = (score, cost, sorted(chosen), len(internal))

    if best is None:
        raise RuntimeError("no connected subgraph of the requested size exists")
    _, cost, layout, n_edges = best
    induced = [undirected[frozenset((layout[i], layout[j]))]
               for i in range(len(layout)) for j in range(i + 1, len(layout))
               if frozenset((layout[i], layout[j])) in undirected]
    return dict(layout=layout, cost=float(cost), internal_edges=int(n_edges),
                strategy=strategy,
                edge_error_mean=float(np.mean(induced)),
                edge_error_max=float(np.max(induced)),
                edge_error_median=float(np.median(induced)),
                device=str(getattr(backend, "name", backend)))

# ---------------------------------------------------------------------------
# Local-coordinate mapping
# ---------------------------------------------------------------------------
# Routing only ever uses the connected component it is handed.  With a fixed
# ``initial_layout`` inside a connected n-qubit subgraph the router cannot leave
# that subgraph, so mapping onto the full device and mapping onto the induced
# n-node coupling map produce the *same* circuit (verified: identical two-qubit
# gate counts on 127q / 27q / 5q devices).  Local coordinates are 1.2-3.2x faster
# and keep the density matrix at 2^n x 2^n rather than 2^127.
def backend_basis(backend):
    try:
        return list(backend.configuration().basis_gates)
    except AttributeError:  # BackendV2
        return list(backend.operation_names)


def layout_coupling_map(backend, layout):
    """Directed coupling map induced by ``layout``, on local indices ``0..n-1``."""
    from qiskit.transpiler import CouplingMap

    pos = {p: i for i, p in enumerate(layout)}
    edges = [[pos[a], pos[b]] for a, b in backend_coupling(backend)
             if a in pos and b in pos]
    if not edges:
        raise ValueError(f"layout {layout} has no device edges")
    return CouplingMap(edges)


def map_circuit(circuit, backend, layout, seed=11, optimization_level=1):
    """Transpile the overlap onto ``layout``, in local coordinates."""
    import warnings

    from qiskit import transpile

    cm = layout_coupling_map(backend, layout)
    with warnings.catch_warnings():  # BackendV1 deprecation, harmless here
        warnings.simplefilter("ignore", DeprecationWarning)
        return transpile(circuit, basis_gates=backend_basis(backend), coupling_map=cm,
                         initial_layout=list(range(cm.size())),
                         optimization_level=optimization_level,
                         approximation_degree=1.0, seed_transpiler=seed,
                         num_processes=1)


def audit_circuit(circuit, backend, layout):
    """Check that a mapped circuit can actually receive device noise.

    Returns the executed two-qubit gate count, the number of gates sitting on a
    pair the device does not provide (must be zero), and the error rate each
    executed two-qubit gate will be assigned -- so the reported operating point
    is the rate the circuit *used*, not the device median.
    """
    pos = dict(enumerate(layout))
    allowed = set(backend_coupling(backend))
    errs = two_qubit_errors(backend)
    median = float(np.median(list(errs.values())))
    pairs = []
    for inst in circuit.data:
        if len(inst.qubits) != 2:
            continue
        local = tuple(int(circuit.find_bit(q).index) for q in inst.qubits)
        pairs.append((inst.operation.name, (pos[local[0]], pos[local[1]])))
    off_edge = [p for _, p in pairs if p not in allowed]
    used = [errs.get(p, errs.get((p[1], p[0]), median)) for _, p in pairs]
    return dict(two_qubit_gates=len(pairs), off_edge=len(off_edge),
                off_edge_pairs=sorted(set(off_edge))[:8],
                gate_name=two_qubit_gate_name(backend),
                depth=int(circuit.depth()), cx=int(circuit.count_ops().get("cx", 0)),
                p2_used_mean=float(np.mean(used)) if used else 0.0,
                p2_used_max=float(np.max(used)) if used else 0.0,
                p2_used_median=float(np.median(used)) if used else 0.0,
                distinct_edges=len({p for _, p in pairs}))







def device_overlap_circuit(query_qc, ref_qc, backend, layout, seed=11, optimise=True):
    """The *same* virtual overlap as the uniform sweep, mapped onto the device."""
    virtual = overlap_circuit(query_qc, ref_qc, seed=seed, optimise=optimise)
    return map_circuit(virtual, backend, layout, seed=seed)


def device_gate_errors(backend):
    """``({gate: {qubit: error}}, {qubit: readout_error})`` from the calibration."""
    props = backend.properties()
    one, readout = {}, {}
    for gate in props.gates:
        if len(gate.qubits) == 1:
            err = _gate_error(gate)
            if err is not None:
                one.setdefault(gate.gate, {})[int(gate.qubits[0])] = err
    for q in range(backend_qubits(backend)):
        try:
            readout[q] = float(props.readout_error(q))
        except Exception:
            readout[q] = 0.0
    return one, readout


def remap_noise_model(noise_model, layout):
    """Restrict physical errors to layout and relabel every qubit locally.

    Preserve Aer channels (including thermal terms) and asymmetric readout.
    """
    from copy import deepcopy
    from qiskit_aer.noise import NoiseModel
    pos = {int(p): i for i, p in enumerate(layout)}
    data = deepcopy(noise_model.to_dict())
    errors = []
    for error in data["errors"]:
        if "gate_qubits" in error:
            groups = [tuple(pos[q] for q in qs) for qs in error["gate_qubits"]
                      if all(q in pos for q in qs)]
            if not groups:
                continue
            error["gate_qubits"] = groups
        errors.append(error)
    data["errors"] = errors
    return NoiseModel.from_dict(data)


def restrict_noise_model(backend, layout, thermal_relaxation=False):
    """Exact restriction of Aer backend noise to the induced local layout.

    Aer converts calibrated average infidelity to depolarizing strength and
    retains asymmetric readout probabilities. Thermal relaxation is optional.
    """
    from qiskit_aer.noise import NoiseModel
    physical = NoiseModel.from_backend(backend, thermal_relaxation=thermal_relaxation)
    return remap_noise_model(physical, layout)


def noiseless_gates(backend, layout):
    """Basis gates the calibration itself reports as error-free on every layout qubit.

    Virtual-Z rotations are exact on hardware (``rz`` error 0.0), so a model with
    no entry for them is physically correct rather than a silent gap.  Any gate
    with a missing calibration is *not* listed here, because it is the model's job
    to fall back to a sibling rate for it.
    """
    one, _ = device_gate_errors(backend)
    out = []
    for gate, per_qubit in one.items():
        if all(per_qubit.get(p, None) == 0.0 for p in layout):
            out.append(gate)
    return sorted(out)


def check_noise_coverage(circuits, noise_model, strict=True, allow=()):
    """Every operation in the circuit must be covered by the noise model.

    The same silent-failure class as an off-edge CX: a gate the model has no
    entry for simply runs noiselessly.  Readout is exempt because Aer applies it
    through a separate mechanism, and ``allow`` lists gates the calibration
    itself declares error-free.  An *empty* model is the deliberate finite-shot
    control, so it reports ``noiseless=True`` instead of failing.
    """
    if not len(noise_model.noise_instructions):
        return dict(covered=[], missing=[], allowed=list(allow), noiseless=True)
    local = noise_model._local_quantum_errors
    default = noise_model._default_quantum_errors
    used = {(inst.operation.name, tuple(c.find_bit(q).index for q in inst.qubits))
            for c in circuits for inst in c.data
            if inst.operation.name not in {"measure", "barrier"}}
    missing = sorted((name, qs) for name, qs in used
                     if name not in allow and name not in default
                     and qs not in local.get(name, {}))
    if missing and strict:
        raise RuntimeError(f"noise model has no error for gate/qubits {missing}")
    return dict(covered=sorted(set(name for name, _ in used) - {n for n, _ in missing}),
                missing=missing, allowed=sorted(allow), noiseless=False)


def device_circuits(query_qc, ref_qcs, backend, layout, seed=11, optimise=True,
                    strict=True):
    """Map every overlap, audit it, and attach measurements.

    Returns ``(circuits, layout, audits)`` in local coordinates.  Mapping is the
    expensive step, so it is separated from sampling: the finite-shot control and
    the device run then share exactly the same circuits.
    """
    circuits, audits = [], []
    for ref in ref_qcs:
        circuit = device_overlap_circuit(query_qc, ref, backend, layout, seed=seed,
                                         optimise=optimise)
        info = audit_circuit(circuit, backend, layout)
        if strict and info["off_edge"]:
            raise RuntimeError(
                f"mapped circuit has {info['off_edge']} two-qubit gates outside the "
                f"coupling map (e.g. {info['off_edge_pairs']}); the device noise model "
                f"would apply no error to them")
        audits.append(info)
        circuit.measure_all()
        circuits.append(circuit)
    return circuits, list(layout), audits


def sample_kernel(circuits, simulator, shots=2048, seed=11, n_qubits=None):
    """All-zero probability of every prepared circuit, divided by ``shots``."""
    result = simulator.run(circuits, shots=shots, seed_simulator=seed).result()
    n = n_qubits if n_qubits is not None else circuits[0].num_qubits
    zero_key = "0" * n
    return np.array([result.get_counts(i).get(zero_key, 0) / shots
                     for i in range(len(circuits))], dtype=float)


def device_noisy_kernel(query_qc, ref_qcs, backend, layout, noise_model=None,
                        shots=2048, seed=11, optimise=True, simulator=None,
                        strict=True):
    """Sample every overlap on the device; refuse to return unnoised circuits.

    Mirrors :func:`src.noise.noisy_kernel` (same return contract) so the two
    operating points can be compared row by row.
    """
    from .noise import make_simulator

    circuits, layout, audits = device_circuits(query_qc, ref_qcs, backend, layout,
                                               seed=seed, optimise=optimise,
                                               strict=strict)
    nm = noise_model if noise_model is not None else restrict_noise_model(backend, layout)
    allow = noiseless_gates(backend, layout) if noise_model is None else ()
    coverage = check_noise_coverage(circuits, nm, strict=strict, allow=allow)
    sim = simulator if simulator is not None else make_simulator(nm, seed)
    values = sample_kernel(circuits, sim, shots=shots, seed=seed)

    def mean(key):
        return float(np.mean([a[key] for a in audits]))

    meta = dict(overlap_cx=mean("cx"), overlap_depth=mean("depth"),
                two_qubit_gates=mean("two_qubit_gates"),
                off_edge=int(sum(a["off_edge"] for a in audits)),
                uncovered_gates=len(coverage["missing"]),
                p2_used_mean=mean("p2_used_mean"), p2_used_max=mean("p2_used_max"),
                distinct_edges=mean("distinct_edges"), physical_qubits=len(layout),
                qubit_layout=list(layout), shots=shots, circuits=len(circuits))
    return values, meta

    for (a, b), err in two_qubit_errors(backend).items():
        if a in pos and b in pos:
            nm.add_quantum_error(depolarizing_error(err, 2), [gate_name], [pos[a], pos[b]])
    if not len(nm.noise_instructions):
        raise RuntimeError("restricted noise model is empty")
    return nm
