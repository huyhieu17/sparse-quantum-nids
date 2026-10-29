"""Theory: kernel perturbation bound for sub-cube (structured sparse) encoding.

The "retained energy" rule used to pick a sub-cube is currently a heuristic.
It admits a short bound.  Write ``E(x) = ||P_x psi_x||^2`` for the fraction of
the dense state's squared norm that lies inside the selected sub-cube and
``psi~ = P psi / ||P psi||``.  Then

    ||psi - psi~||^2 = 2 (1 - ||P psi||) = 2 (1 - sqrt(E(x)))  ==:  d(x)^2.

For two samples x, z with fidelity ``K = |<psi_x|psi_z>|^2`` and sparse
fidelity ``K~``,

    |sqrt(K) - sqrt(K~)| <= ||psi_x - psi~_x|| + ||psi_z - psi~_z||
                          = d(x) + d(z),

hence

    |K - K~| <= (sqrt(K) + sqrt(K~)) (d(x) + d(z)) <= 2 (d(x) + d(z)).

The bound is worst-case; :func:`bound_report` measures how tight it is on the
actual encodings.
"""

import numpy as np

from .pipeline import kernel


def retained_energy(sparse_encoder, X):
    """Per-sample retained energy ``E`` and perturbation norm ``d``."""
    z = np.atleast_2d(sparse_encoder.projection.transform(X))
    energies = np.sum(z[:, sparse_encoder.cubes] ** 2, axis=2)
    best = np.argmax(energies, axis=1)
    top = np.argsort(-(z * z), axis=1, kind="stable")[:, :sparse_encoder.support]
    topenergy = np.sum(np.take_along_axis(z, top, axis=1) ** 2, axis=1)
    use = energies[np.arange(len(z)), best] >= (1 - sparse_encoder.loss) * topenergy
    chosen = top.copy()
    chosen[use] = sparse_encoder.cubes[best[use]]
    e = np.sum(np.take_along_axis(z, chosen, axis=1) ** 2, axis=1)
    e = np.clip(e, 1e-12, 1.0)
    d = np.sqrt(2 * (1 - np.sqrt(e)))
    return e, d


def bound_report(sparse_encoder, X_a, X_b=None, max_pairs=4000, seed=11):
    """Compare the analytic bound with the realised kernel perturbation."""
    X_b = X_a if X_b is None else X_b
    z_a = np.atleast_2d(sparse_encoder.projection.transform(X_a))
    z_b = np.atleast_2d(sparse_encoder.projection.transform(X_b))
    v_a = np.atleast_2d(sparse_encoder.transform(X_a))
    v_b = np.atleast_2d(sparse_encoder.transform(X_b))
    e_a, d_a = retained_energy(sparse_encoder, X_a)
    e_b, d_b = retained_energy(sparse_encoder, X_b)

    rng = np.random.default_rng(seed)
    n_a, n_b = len(z_a), len(z_b)
    ia = rng.integers(0, n_a, max_pairs)
    ib = rng.integers(0, n_b, max_pairs)
    k_dense = kernel(z_a[ia], z_b[ib])[np.arange(max_pairs), np.arange(max_pairs)]
    k_sparse = kernel(v_a[ia], v_b[ib])[np.arange(max_pairs), np.arange(max_pairs)]
    actual = np.abs(k_dense - k_sparse)
    bound = 2 * (d_a[ia] + d_b[ib])
    holds = actual <= bound + 1e-9
    return dict(
        retained_energy_mean=float(np.mean(e_a)),
        retained_energy_min=float(np.min(e_a)),
        perturbation_mean=float(np.mean(d_a)),
        kernel_error_mean=float(np.mean(actual)),
        kernel_error_p95=float(np.percentile(actual, 95)),
        bound_mean=float(np.mean(bound)),
        bound_clipped_mean=float(np.mean(np.minimum(bound, 1.))),
        informative_fraction=float(np.mean(bound < 1.)),
        tightness_mean=float(np.mean(actual / np.maximum(bound, 1e-12))),
        pairs=int(max_pairs),
        bound_holds_fraction=float(np.mean(holds)),
    )
