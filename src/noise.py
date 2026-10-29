"""Noisy fidelity-kernel evaluation with Qiskit Aer.

The frozen paper measures *ideal* statevector overlaps, i.e. it assumes that a
CX gate is free and noiseless.  Its own motivation, however, is that CX gates
are the dominant error source.  This module closes that gap: it executes the
same overlap circuits on Aer under a controlled depolarizing model and reports
how the fidelity kernel (and therefore the SVC decision) degrades as a
function of the two-qubit error rate.

Controlled model
----------------
* single-qubit depolarizing error ``p1 = ratio * p2`` on Rx/Ry/Rz/H
* two-qubit depolarizing error ``p2`` on CX  (the swept variable)
* optional symmetric readout error ``p_ro``
Noise is applied to *both* encoders identically, so Full-vs-Sparse differences
come only from the circuit each encoder produces (CX count / depth).
"""

import numpy as np
from functools import lru_cache

from .pipeline import BASIS, remove_leading_cx_gates, transpile


def build_noise_model(p2, p1_ratio=0.1, p_ro=0.0):
    """Return an Aer ``NoiseModel`` for a two-qubit depolarizing rate ``p2``."""
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

    nm = NoiseModel()
    if p2 <= 0 and p_ro <= 0:
        return nm
    p1 = max(0.0, p1_ratio * p2)
    if p1 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["rx", "ry", "rz", "h"])
    if p2 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    if p_ro > 0:
        nm.add_all_qubit_readout_error(ReadoutError([[1 - p_ro, p_ro], [p_ro, 1 - p_ro]]))
    return nm


def build_backend_noise_model(backend_name="Fake5QV1"):
    """Best-effort noise model from a bundled fake backend (optional)."""
    from qiskit_aer.noise import NoiseModel

    try:
        from qiskit.providers.fake_provider import Fake5QV1, Fake7QPulseV1
        factory = {"Fake5QV1": Fake5QV1, "Fake7QPulseV1": Fake7QPulseV1}[backend_name]
        return NoiseModel.from_backend(factory())
    except Exception as exc:  # pragma: no cover - depends on qiskit build
        raise RuntimeError(f"backend noise model unavailable: {exc}") from exc


# ---------------------------------------------------------------------------
# Calibrated operating points
# ---------------------------------------------------------------------------
# The circuits are prepared on five qubits in a device-independent basis, so we
# deliberately do *not* map them onto a device topology: on a heavy-hex lattice
# most 5-qubit pairs are not edges, and ``NoiseModel.from_backend`` would then
# silently drop the two-qubit error on exactly those pairs.  Instead we
# calibrate the *rates* of the controlled depolarizing model to a published
# device calibration, plus the readout error the abstract model leaves at zero.
CALIBRATED_BACKENDS = {
    "montreal": "Fake27QPulseV1",   # median CX 0.7 %, readout 1.3 %
    "vigo": "Fake5QV1",             # older 5-qubit device: CX 2.1 %, readout 6.3 %
}


@lru_cache(maxsize=None)
def backend_error_rates(name="montreal"):
    """Median 1q / 2q / readout error rates of a bundled fake-backend calibration."""
    from qiskit.providers import fake_provider

    cls_name = CALIBRATED_BACKENDS.get(name, name)
    try:
        backend = getattr(fake_provider, cls_name)()
    except AttributeError as exc:
        raise RuntimeError(f"unknown calibrated backend {name!r}") from exc

    props = backend.properties()
    if props is None:
        raise RuntimeError(f"{cls_name} exposes no calibration properties")
    config = backend.configuration() if hasattr(backend, "configuration") else backend
    n = int(getattr(config, "n_qubits", getattr(backend, "num_qubits", 0)))
    coupling = getattr(config, "coupling_map", None)
    if coupling is None:
        coupling = getattr(backend, "coupling_map", None) or []

    cx = []
    for edge in coupling:
        try:
            value = props.gate_error("cx", list(edge))
        except Exception:
            value = None
        if value is not None:
            cx.append(float(value))
    sq = [float(g.parameters[0].value) for g in props.gates
          if g.gate in ("sx", "x") and len(g.qubits) == 1 and g.parameters]
    ro = [float(props.readout_error(q)) for q in range(n)]
    if not cx:
        raise RuntimeError(f"{cls_name} reports no two-qubit gate errors")

    return dict(backend=cls_name, qubits=n,
                p2=float(4 * np.median(cx) / 3), p2_p90=float(4 * np.percentile(cx, 90) / 3),
                p1=float(2 * np.median(sq)) if sq else 0.0,
                gate_infidelity_2q=float(np.median(cx)),
                gate_infidelity_1q=float(np.median(sq)) if sq else 0.0,
                p_ro=float(np.median(ro)) if ro else 0.0,
                n_cx_edges=len(cx))


def build_calibrated_noise_model(name="montreal"):
    """Depolarizing model with rates taken from a device calibration.

    Returns ``(noise_model, rates)`` where ``rates`` carries the numbers that
    must be reported: median CX error (the operating point on the ``p2`` axis),
    median 1q error and median readout error.
    """
    rates = backend_error_rates(name)
    p1_ratio = rates["p1"] / rates["p2"] if rates["p2"] > 0 else 0.1
    nm = build_noise_model(rates["p2"], p1_ratio=p1_ratio, p_ro=rates["p_ro"])
    return nm, rates


def overlap_circuit(query_qc, ref_qc, seed=11, optimise=True):
    """U(z)^dagger U(x) overlap circuit without measurement."""
    if optimise:
        c = transpile(query_qc.compose(ref_qc.inverse()), basis_gates=BASIS,
                      optimization_level=1, approximation_degree=1.,
                      seed_transpiler=seed, num_processes=1)
    else:
        # Both operands are already transpiled to BASIS; composition keeps it.
        c = query_qc.compose(ref_qc.inverse())
    return remove_leading_cx_gates(c)


def make_simulator(noise_model=None, seed=11, max_parallel_threads=0):
    from qiskit_aer import AerSimulator

    return AerSimulator(noise_model=noise_model, method="density_matrix",
                        seed_simulator=seed, max_parallel_threads=max_parallel_threads)


def noisy_kernel(query_qc, ref_qcs, noise_model, shots=2048, seed=11,
                 optimise=True, simulator=None):
    """Sample the all-zero probability of every overlap circuit.

    Returns ``(kernel, meta)`` where ``kernel[i]`` estimates
    ``|<0|U(z)^dagger U(x)|0>|^2`` for ``ref_qcs[i]`` and ``meta`` holds mean
    CX count / depth of the *executed* overlap circuits.
    """
    sim = simulator if simulator is not None else make_simulator(noise_model, seed)
    n = query_qc.num_qubits
    circuits, cxs, depths = [], [], []
    for rc in ref_qcs:
        c = overlap_circuit(query_qc, rc, seed=seed, optimise=optimise)
        cxs.append(c.count_ops().get("cx", 0))
        depths.append(c.depth())
        c.measure_all()
        circuits.append(c)
    result = sim.run(circuits, shots=shots, seed_simulator=seed).result()
    zero_key = "0" * n
    values = np.array([result.get_counts(i).get(zero_key, 0) / shots
                       for i in range(len(circuits))], dtype=float)
    meta = dict(overlap_cx=float(np.mean(cxs)), overlap_depth=float(np.mean(depths)),
                shots=shots, circuits=len(circuits))
    return values, meta


def kernel_error(ideal, noisy):
    """Summary statistics of ``noisy`` against the exact ``ideal`` kernel."""
    ideal = np.asarray(ideal, float)
    noisy = np.asarray(noisy, float)
    diff = noisy - ideal
    denom = np.linalg.norm(ideal) * np.linalg.norm(noisy)
    return dict(mae=float(np.mean(np.abs(diff))),
                rmse=float(np.sqrt(np.mean(diff ** 2))),
                max_abs=float(np.max(np.abs(diff))),
                cosine=float(ideal @ noisy / denom) if denom else float("nan"))


def exact_probabilities(circuits, noise_model=None, simulator=None):
    """Exact noisy all-zero probabilities, with calibrated readout included.

    Density-matrix evolution removes Monte Carlo gate/shot variance. Subsequent
    binomial draws reproduce the marginal distribution of zero-bin counts.
    """
    sim = simulator if simulator is not None else make_simulator(noise_model, max_parallel_threads=1)
    saved=[]
    for c in circuits:
        c=c.remove_final_measurements(inplace=False)
        c.save_probabilities(label='probabilities')
        saved.append(c)
    result=sim.run(saved,shots=1).result()
    values=[]
    for i,c in enumerate(circuits):
        probs=np.asarray(result.data(i)['probabilities'],float)
        weights=np.ones(len(probs))
        if noise_model is not None:
            local=noise_model._local_readout_errors
            default=noise_model._default_readout_error
            for q in range(c.num_qubits):
                error=local.get((q,),default)
                if error is not None:
                    weights *= np.asarray(error.probabilities)[(np.arange(len(probs))>>q)&1,0]
                else:
                    weights *= ((np.arange(len(probs))>>q)&1)==0
            if any(len(qs)!=1 for qs in local):
                raise ValueError('Exact readout currently supports independent single-qubit channels only')
            p=float(probs@weights)
        else:p=float(probs[0])
        values.append(np.clip(p,0.,1.))
    return np.asarray(values)
