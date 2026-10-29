"""Orchestration: ideal quality, held-out test, noise sweep, theory, stats."""

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .classical import BASELINES, classical_score, train_classical
from .data import DATASETS, load_dataset, load_external_test, partitions, stratified_indices
from .noise import (backend_error_rates, build_calibrated_noise_model, build_noise_model,
                    kernel_error, make_simulator, noisy_kernel)
from .pipeline import (DenseEncoder, Features, StructuredSparseEncoding, decision,
                       kernel, quality, synthesize_structured, train_model)
from .stats import (bootstrap_ci_mean, mcnemar, paired_bootstrap_diff,
                    paired_ratio_bootstrap, noninferiority, wilson_ci)
from .theory import bound_report

SEEDS = [11, 23, 37]
TRAIN_SAMPLES = 2000
SPARSE_CONFIG = {  # (support, loss): matches the frozen paper choices
    "ToN-IoT": (8, 1.0),
    "UNSW-NB15": (16, .3),
    "NSL-KDD": (16, .3),
    "RT-IoT": (16, .3),
}
QUALITY_COLUMNS = ["accuracy", "macro_f1", "recall", "fpr"]


def prepare(dataset, seed, train_samples=TRAIN_SAMPLES):
    """Load, split, fit preprocessing and train Full + Sparse on one seed."""
    frame = load_dataset(dataset)
    parts = partitions(frame, dataset, seed)
    feat = Features(64).fit(parts["train"])
    X_train = feat.transform(parts["train"])
    ri = stratified_indices(parts["train"], train_samples, seed)
    y_train = parts["train"].label.to_numpy()[ri]
    dense = DenseEncoder().fit(X_train, feat.importance)
    support, loss = SPARSE_CONFIG[dataset]
    sparse = StructuredSparseEncoding(dense, support, loss, True)
    models = {
        "Full": train_model(dense, X_train[ri], y_train),
        "Sparse": train_model(sparse, X_train[ri], y_train),
    }
    return dict(frame=frame, parts=parts, feat=feat, models=models)


def heldout_frame(prepared, dataset):
    """External published test when available, else the reserved 20% split."""
    ext = load_external_test(dataset)
    if ext is not None:
        return ext, "external"
    return prepared["parts"]["test"], "internal"


def evaluate(models, feat, frame):
    y = frame.label.to_numpy()
    X = feat.transform(frame)
    out = {}
    for name, art in models.items():
        score = decision(art, X)
        out[name] = dict(quality=quality(y, score), score=score, y=y)
    return out


def evaluate_classical(prepared, frame, seed, train_samples=TRAIN_SAMPLES):
    y = frame.label.to_numpy()
    X_train = prepared["feat"].transform(prepared["parts"]["train"])
    ri = stratified_indices(prepared["parts"]["train"], train_samples, seed)
    models = train_classical(X_train[ri], prepared["parts"]["train"].label.to_numpy()[ri], seed)
    X = prepared["feat"].transform(frame)
    return {name: dict(quality=quality(y, classical_score(m, X)),
                       score=classical_score(m, X), y=y)
            for name, m in models.items()}


def _query_indices(y, n_queries, seed):
    rng = np.random.default_rng(seed + 3000)
    per_class = max(1, n_queries // 2)
    idx = [rng.choice(np.flatnonzero(y == c), min(per_class, int((y == c).sum())), replace=False)
           for c in (0, 1)]
    return np.sort(np.concatenate(idx))


def select_queries(models, feat, frame, n_margin, n_control, seed, model="Full"):
    """Margin-stratified query set plus a random control group.

    The frozen protocol drew queries uniformly at random *within each class*.
    Measured on the held-out flows, those queries sit at the 99-100th percentile
    of ``|decision score|``, i.e. as far from the decision boundary as the data
    allows.  A noise sweep over that set cannot observe a decision flip, so the
    resulting "1/192 flips" is a property of the query selection, not evidence
    of noise tolerance.

    This selector fixes the design:

    * ``n_margin`` queries with the *smallest* ``|score|`` -- the flows whose
      classification a modest kernel perturbation can actually overturn;
    * ``n_control`` queries drawn uniformly from the remainder -- the frozen
      protocol's regime, kept as an internal control.

    Selection uses ``model`` (the dense reference) and therefore is *identical*
    for Full and Sparse: the comparison is on a fixed query set, paired at the
    query level.  Selection is deterministic given ``seed``.

    Caveat, learned the hard way: sharing a sample does not make an absolute
    flip rate comparable across encoders.  The set is near-boundary *for the
    model that selected it*, so at 2048 shots its sampling error exceeds its
    margin for that model (p2 = 0 flip rate 34 % dense vs 3.4 % sparse) and the
    gap is a property of the selection, not of the encoder; selecting on the
    sparse margin equalises the two.  Cross-encoder claims belong on the
    ``control`` group, whose p2 = 0 floor is 0.3 % for both; the ``margin``
    group is a stress set and is reported as ``floor + excess``.
    """
    y = np.asarray(frame.label)
    X = feat.transform(frame)
    score = decision(models[model], X)
    margin = np.abs(score)

    order = np.argsort(margin, kind="stable")
    n_margin = int(min(n_margin, len(order) - 1))
    margin_idx = np.sort(order[:n_margin])
    pool = order[n_margin:]
    rng = np.random.default_rng(seed + 3200)
    n_control = int(min(n_control, len(pool)))
    control_idx = np.sort(rng.choice(pool, n_control, replace=False))

    idx = np.concatenate([margin_idx, control_idx])
    groups = np.array(["margin"] * len(margin_idx) + ["control"] * len(control_idx))
    return idx.astype(int), groups, margin.copy()



def noise_sweep(models, feat, frame, seed, levels, n_queries=8, max_refs=128,
                shots=2048, optimise=False, verbose=True, queries=None,
                query_groups=None, noise_model="depolarizing", simulators=None,
                extra=None):
    """Run Full and Sparse overlap circuits under a depolarizing sweep.

    For every (model, query, level) the exact analytic kernel is compared with
    the Aer-sampled kernel, and the SVC decision is recomputed from the noisy
    kernel.  ``max_refs`` caps the support vectors per query (deterministic
    subsample); ``max_refs = 0`` uses the full support set and is the protocol the
    reported numbers use.  A finite cap is *not* a neutral speed-up: the score is
    ``sum_i alpha_i k_i + b`` over the retained vectors, so truncation shrinks the
    decision function's sensitivity to kernel error and makes every flip rate a
    property of the cap.  ``verify_v3.py`` detects that case by comparing the
    control-group accuracy with the held-out accuracy of the same model.

    ``queries`` / ``query_groups`` override the default random query draw (see
    :func:`select_queries`); ``simulators`` overrides the per-level Aer
    simulator (used by the calibrated operating points); ``extra`` adds
    constant columns to every row (e.g. the calibrated readout error).
    """
    y = frame.label.to_numpy()
    if queries is None:
        idx = _query_indices(y, n_queries, seed)
        groups = np.array(["random"] * len(idx))
    else:
        idx = np.asarray(queries, int)
        groups = (np.asarray(query_groups, object) if query_groups is not None
                  else np.array(["random"] * len(idx)))
    extra = dict(extra or {})
    X_all = feat.transform(frame)
    score_ref = decision(models["Full"], X_all)
    Xq = X_all[idx]
    rng = np.random.default_rng(seed + 3100)
    if simulators is None:
        simulators = {float(level): make_simulator(build_noise_model(level), seed)
                      for level in levels}
    rows = []
    for name, art in models.items():
        enc, model = art["encoder"], art["model"]
        refs = art["refs"]
        dual = np.asarray(model.dual_coef_[0])
        if max_refs and len(refs) > max_refs:
            sel = np.sort(rng.choice(len(refs), max_refs, replace=False))
            refs = refs[sel]
            dual = dual[sel]
        ref_circs, ref_cx = [], []
        for v in refs:
            qc, cost = synthesize_structured(v, seed)
            ref_circs.append(qc)
            ref_cx.append(cost["cx"])
        for qi, row_i in enumerate(idx):
            v = enc.transform(Xq[qi:qi + 1])[0]
            qc, qcost = synthesize_structured(v, seed)
            ideal = kernel(v[None, :], refs)[0]
            ideal_score = float(ideal @ dual + model.intercept_[0])
            for level in levels:
                sim = simulators[float(level)]
                start = time.perf_counter()
                k, meta = noisy_kernel(qc, ref_circs, None, shots=shots, seed=seed,
                                       optimise=optimise, simulator=sim)
                seconds = time.perf_counter() - start
                noisy_score = float(k @ dual + model.intercept_[0])
                err = kernel_error(ideal, k)
                rows.append(dict(model=name, level=float(level), query=int(row_i),
                                 query_group=str(groups[qi]),
                                 ref_margin=float(abs(score_ref[row_i])),
                                 label=int(y[row_i]), prep_cx=qcost["cx"],
                                 prep_depth=qcost["depth"], ref_cx=float(np.mean(ref_cx)),
                                 overlap_cx=meta["overlap_cx"], overlap_depth=meta["overlap_depth"],
                                 sv=len(refs), shots=shots, seconds=seconds,
                                 ideal_score=ideal_score, noisy_score=noisy_score,
                                 ideal_pred=int(ideal_score >= 0), noisy_pred=int(noisy_score >= 0),
                                 flipped=int((ideal_score >= 0) != (noisy_score >= 0)),
                                 correct=int((noisy_score >= 0) == y[row_i]),
                                 noise_model=str(noise_model),
                                 **({"p1": .1 * float(level)} | extra), **err))
                if verbose:
                    print(f"[noise:{noise_model}] {name} q={row_i} p2={level:.3f} "
                          f"mae={err['mae']:.4f} flip={rows[-1]['flipped']} "
                          f"cx={meta['overlap_cx']:.1f}", flush=True)
    return pd.DataFrame(rows)


def summarise_noise(df):
    """Aggregate the noise sweep per (dataset, setting, model, level[, query group])."""
    keys = (["dataset"] if "dataset" in df.columns else []) \
        + (["noise_model"] if "noise_model" in df.columns else []) \
        + ["model", "level"] \
        + (["query_group"] if "query_group" in df.columns else [])
    return (df.groupby(keys)
              .agg(mae=("mae", "mean"), rmse=("rmse", "mean"),
                   max_abs=("max_abs", "mean"), cosine=("cosine", "mean"),
                   flip_rate=("flipped", "mean"), accuracy=("correct", "mean"),
                   mean_abs_dscore=("noisy_score", "size"),
                   overlap_cx=("overlap_cx", "mean"), overlap_depth=("overlap_depth", "mean"),
                   prep_cx=("prep_cx", "mean"), sv=("sv", "mean"),
                   queries=("query", "nunique"), circuits=("query", "size"))
              .reset_index()
              .drop(columns=["mean_abs_dscore"]))


def noise_stats(df, n_boot=4000, seed=11, alpha=0.05):
    """Query-level statistics of the noise sweep, with confidence intervals.

    Two quantities matter for the robustness claim and both were previously
    point estimates over a single query set and a single noise seed:

    ``excess MAE``
        ``mae(level) - mae(level=0)`` per query, averaged over queries.  The
        ``p2 = 0`` finite-shot row isolates sampling noise, so the difference is
        the error attributable to gate noise.  Averaging *per query first* also
        removes the composition bias of averaging over a changing denominator.

    ``Full/Sparse ratio``
        Ratio of the two excess-MAE means, bootstrapped by resampling the paired
        queries jointly, which yields the CI the paper's "2.06-13.94x" lacked.

    Returns one row per (dataset, seed, noise_model, level).
    """
    if not {"noise_model", "level"} <= set(df.columns):
        return pd.DataFrame()
    control = df[(df.noise_model == "depolarizing") & (df.level == 0)]
    if not len(control):
        return pd.DataFrame()
    base = (control.groupby(["dataset", "seed", "query", "model"])["mae"].mean()
            .unstack("model").rename(columns=lambda c: f"base_{c}"))
    rows = []
    for (dataset, s, nm, level), part in df.groupby(["dataset", "seed", "noise_model", "level"]):
        if nm == "depolarizing" and float(level) == 0:
            continue  # the control itself
        wide = (part.groupby(["query", "model"])["mae"].mean().unstack("model")
                .join(base.loc[(dataset, s)], how="inner"))
        if wide.empty or not {"Full", "Sparse"} <= set(wide.columns):
            continue
        exc = {m: wide[m].to_numpy() - wide[f"base_{m}"].to_numpy() for m in ("Full", "Sparse")}
        ratio = paired_ratio_bootstrap(exc["Full"], exc["Sparse"],
                                       n_boot=n_boot, seed=seed + 97, alpha=alpha)
        row = dict(dataset=dataset, seed=int(s), noise_model=nm, level=float(level),
                   n_queries=int(len(wide)))
        for m in ("Full", "Sparse"):
            mean_ci = bootstrap_ci_mean(exc[m], n_boot=n_boot, seed=seed + 11, alpha=alpha)
            row.update({f"excess_mae_{m.lower()}": mean_ci["mean"],
                        f"excess_mae_{m.lower()}_low": mean_ci["low"],
                        f"excess_mae_{m.lower()}_high": mean_ci["high"]})
            row[f"mae_{m.lower()}"] = float(wide[m].mean())
        row.update(ratio=ratio["mean"], ratio_low=ratio["low"], ratio_high=ratio["high"],
                   ratio_p_gt_one=ratio.get("p_gt_one", float("nan")),
                   ratio_undefined=ratio.get("undefined", 0.0))
        for m in ("Full", "Sparse"):
            sub = part[part.model == m]
            row[f"mean_abs_dscore_{m.lower()}"] = float(
                (sub.noisy_score - sub.ideal_score).abs().mean())
        rows.append(row)
    return pd.DataFrame(rows)


def flip_stats(df, alpha=0.05):
    """Decision-flip rate per (dataset, seed, setting, level, model, query group).

    The Wilson interval is the right one here: at ``p2 <= 2 %`` the observed
    counts are single digit, where the normal approximation collapses.
    """
    if "query_group" not in df.columns:
        return pd.DataFrame()
    keys = (["dataset"] if "dataset" in df.columns else []) \
        + (["noise_model"] if "noise_model" in df.columns else []) \
        + ["seed", "level", "model", "query_group"]
    rows = []
    for key, part in df.groupby(keys):
        ci = wilson_ci(int(part.flipped.sum()), int(len(part)), alpha=alpha)
        rows.append(dict(zip(keys, key)) if isinstance(key, tuple)
                    else {keys[0]: key})
        rows[-1].update(flips=int(part.flipped.sum()), n=int(len(part)),
                        flip_rate=ci["rate"], flip_low=ci["low"], flip_high=ci["high"])
    return pd.DataFrame(rows)



def paired_stats(y, score_sparse, score_full, seed, margin=1.0):
    out = {}
    for metric in QUALITY_COLUMNS:
        out[metric] = noninferiority(y, score_sparse, score_full, metric,
                                          margin=margin, seed=seed)
    out["mcnemar"] = mcnemar(y, score_sparse, score_full)
    return out



def run_dataset(dataset, seeds=None, noise_datasets=None, noise_seeds=(11,),
                levels=(0.0, 0.01, 0.02, 0.05), n_queries=8, max_refs=128,
                shots=2048, margin=1.0, with_classical=True, with_noise=True,
                train_samples=TRAIN_SAMPLES, query_select="random", n_margin=100,
                n_control=50, noise_models=("depolarizing",), n_boot=4000,
                verbose=True):
    """Run one dataset over ``seeds`` and return the per-dataset frames.

    Splitting this out of :func:`run_all` is what makes ``--workers`` possible:
    datasets are independent, so they are the natural unit of parallelism, and
    the parent process merges the returned frames into one set of artefacts.
    """
    seeds = seeds or SEEDS
    noise_datasets = noise_datasets if noise_datasets is not None else [dataset]
    quality_rows, test_rows, classical_rows, stats_rows, theory_rows = [], [], [], [], []
    noise_frames = []

    for seed in seeds:
        tag = f"{dataset}/seed{seed}"
        if verbose:
            print(f"\n=== {tag} ===", flush=True)
        prepared = prepare(dataset, seed, train_samples)
        cal = prepared["parts"]["calibration"]
        q_cal = evaluate(prepared["models"], prepared["feat"], cal)
        for name, res in q_cal.items():
            quality_rows.append(dict(dataset=dataset, seed=seed, model=name,
                                     subset="calibration", n=len(res["y"]), **res["quality"]))
            if verbose:
                print(f"[calib] {name}: " + " ".join(
                    f"{k}={res['quality'][k]:.3f}" for k in QUALITY_COLUMNS), flush=True)

        frame, source = heldout_frame(prepared, dataset)
        if len(frame):
            q_test = evaluate(prepared["models"], prepared["feat"], frame)
            for name, res in q_test.items():
                test_rows.append(dict(dataset=dataset, seed=seed, model=name,
                                      subset=f"test_{source}", n=len(res["y"]), **res["quality"]))
                if verbose:
                    print(f"[test:{source}] {name}: " + " ".join(
                        f"{k}={res['quality'][k]:.3f}" for k in QUALITY_COLUMNS), flush=True)

            if with_classical:
                q_cls = evaluate_classical(prepared, frame, seed, train_samples)
                for name, res in q_cls.items():
                    classical_rows.append(dict(dataset=dataset, seed=seed, model=name,
                                               subset=f"test_{source}", n=len(res["y"]),
                                               **res["quality"]))

            stats_rows.append(dict(dataset=dataset, seed=seed, subset=f"test_{source}",
                                   stats=paired_stats(q_test["Sparse"]["y"],
                                                      q_test["Sparse"]["score"],
                                                      q_test["Full"]["score"], seed, margin)))

        if dataset in noise_datasets and with_noise and seed in tuple(noise_seeds):
            noise_frame, _ = heldout_frame(prepared, dataset)
            if not len(noise_frame):
                noise_frame = prepared["parts"]["calibration"]
            q_idx, q_grp = None, None
            if query_select == "margin":
                q_idx, q_grp, _ = select_queries(prepared["models"], prepared["feat"],
                                                 noise_frame, n_margin, n_control, seed)
            sims = {float(l): make_simulator(build_noise_model(l), seed) for l in levels}
            noise_frames.append(
                noise_sweep(prepared["models"], prepared["feat"], noise_frame,
                            seed, levels, n_queries=n_queries, max_refs=max_refs,
                            shots=shots, verbose=verbose, queries=q_idx,
                            query_groups=q_grp, noise_model="depolarizing",
                            simulators=sims, extra=dict(p_ro=0.0))
                .assign(dataset=dataset, seed=seed))
            for nm in noise_models:
                if nm == "depolarizing":
                    continue
                model_nm, rates = build_calibrated_noise_model(nm)
                level = float(rates["p2"])
                noise_frames.append(
                    noise_sweep(prepared["models"], prepared["feat"], noise_frame,
                                seed, [level], max_refs=max_refs, shots=shots,
                                verbose=verbose, queries=q_idx, query_groups=q_grp,
                                noise_model=nm,
                                simulators={level: make_simulator(model_nm, seed)},
                                extra=dict(p_ro=rates["p_ro"], p1=rates["p1"]))
                    .assign(dataset=dataset, seed=seed))
            sparse = prepared["models"]["Sparse"]["encoder"]
            Xt = prepared["feat"].transform(noise_frame)
            if len(Xt) > 2000:
                Xt = Xt[np.random.default_rng(seed).choice(len(Xt), 2000, replace=False)]
            theory_rows.append(dict(dataset=dataset, seed=seed,
                                    **bound_report(sparse, Xt, max_pairs=4000, seed=seed)))

    noise = pd.concat(noise_frames, ignore_index=True) if noise_frames else pd.DataFrame()
    return dict(
        quality_calibration=pd.DataFrame(quality_rows),
        quality_test=pd.DataFrame(test_rows),
        classical_test=pd.DataFrame(classical_rows),
        theory_bound=pd.DataFrame(theory_rows),
        paired_stats=_flatten_stats(stats_rows),
        noise_sweep=noise,
        noise_summary=summarise_noise(noise) if len(noise) else pd.DataFrame(),
        noise_stats=noise_stats(noise, n_boot=n_boot, seed=seeds[0]) if len(noise) else pd.DataFrame(),
        flip_stats=flip_stats(noise) if len(noise) else pd.DataFrame(),
    )


def _flatten_stats(stats_rows):
    flat = []
    for r in stats_rows:
        for metric, ci in r["stats"].items():
            row = dict(dataset=r["dataset"], seed=r["seed"], subset=r["subset"], metric=metric)
            row.update({k: v for k, v in ci.items() if not isinstance(v, (dict, list))})
            flat.append(row)
    return pd.DataFrame(flat)


def _run_dataset_job(job):
    return run_dataset(**job)


ARTEFACTS = ["quality_calibration", "quality_test", "classical_test", "paired_stats",
             "theory_bound", "noise_sweep", "noise_summary", "noise_stats", "flip_stats"]


def run_all(out_dir="results", datasets=None, seeds=None, noise_datasets=None,
            noise_seeds=(11,), levels=(0.0, 0.01, 0.02, 0.05), n_queries=8,
            max_refs=128, shots=2048, margin=1.0, with_classical=True,
            with_noise=True, train_samples=TRAIN_SAMPLES, query_select="random",
            n_margin=100, n_control=50, noise_models=("depolarizing",), workers=1,
            n_boot=4000, verbose=True):
    """Execute the study and write CSV/JSON artefacts under ``out_dir``.

    ``workers > 1`` distributes the datasets over processes (they are fully
    independent) and merges the per-dataset frames afterwards, so the artefacts
    are identical to a serial run.
    """
    datasets = list(datasets or DATASETS)
    seeds = list(seeds or SEEDS)
    noise_datasets = list(noise_datasets) if noise_datasets is not None else datasets
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    jobs = [dict(dataset=d, seeds=[s], noise_datasets=noise_datasets,
                 noise_seeds=tuple(noise_seeds), levels=tuple(levels),
                 n_queries=n_queries, max_refs=max_refs, shots=shots, margin=margin,
                 with_classical=with_classical, with_noise=with_noise,
                 train_samples=train_samples, query_select=query_select,
                 n_margin=n_margin, n_control=n_control,
                 noise_models=tuple(noise_models), n_boot=n_boot, verbose=verbose)
            for d in datasets for s in seeds]
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(int(workers), len(jobs))) as pool:
            parts = list(pool.map(_run_dataset_job, jobs))
    else:
        parts = [_run_dataset_job(j) for j in jobs]

    merged = {}
    for key in ARTEFACTS:
        frames = [p[key] for p in parts if key in p and len(p[key])]
        if frames:
            merged[key] = pd.concat(frames, ignore_index=True)
    for key, frame in merged.items():
        frame.to_csv(out / f"{key}.csv", index=False)
    if verbose and "noise_summary" in merged:
        print("\nNoise summary:\n",
              merged["noise_summary"].round(4).to_string(index=False), flush=True)

    rates = {nm: backend_error_rates(nm) for nm in noise_models if nm != "depolarizing"}
    (out / "run_config.json").write_text(json.dumps(dict(
        datasets=datasets, seeds=seeds, noise_datasets=noise_datasets,
        noise_seeds=list(noise_seeds), levels=list(levels), n_queries=n_queries,
        max_refs=max_refs, shots=shots, margin=margin, train_samples=train_samples,
        query_select=query_select, n_margin=n_margin, n_control=n_control,
        noise_models=list(noise_models), calibrated_rates=rates, workers=workers,
        sparse_config=SPARSE_CONFIG,
    ), indent=2))
    return out

    return out

