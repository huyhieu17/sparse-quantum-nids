"""Statistical tests for the paired Full-vs-Sparse comparison.

The frozen paper reports point estimates over three seeds and explicitly says
it "does not replace a noninferiority test".  These helpers supply one:
paired bootstrap confidence intervals over the evaluation samples and a
one-sided non-inferiority decision for a specified margin.

The bootstrap resamples the *confusion matrix* rather than the raw rows.  For
accuracy / recall / FPR / macro-F1 the metric is a function of the four
confusion cells only, so drawing multinomial counts over the joint
``(y, pred_new, pred_ref)`` categories is an exact nonparametric bootstrap of
the same statistic and runs in ``O(n_boot)`` instead of ``O(n_boot * n)``.
This matters on the 56k-row UNSW-NB15 test set.
"""

import numpy as np


def _cells(y, pred):
    """Return ``(tp, fp, tn, fn)`` for binary labels and predictions."""
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    tp = int(np.sum((y == 1) & (pred == 1)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    return tp, fp, tn, fn


def _metric_from_cells(tp, fp, tn, fn, name):
    total = tp + fp + tn + fn
    if total <= 0:
        return 0.0
    if name == "accuracy":
        return 100.0 * (tp + tn) / total
    if name == "recall":
        return 100.0 * tp / (tp + fn) if (tp + fn) else 0.0
    if name == "fpr":
        return 100.0 * fp / (fp + tn) if (fp + tn) else 0.0
    if name == "macro_f1":
        def f1(t, false_pos, false_neg):
            prec = t / (t + false_pos) if (t + false_pos) else 0.0
            rec = t / (t + false_neg) if (t + false_neg) else 0.0
            return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return 100.0 * 0.5 * (f1(tp, fp, fn) + f1(tn, fn, fp))
    raise ValueError(name)


def _point(y, pred, name):
    return _metric_from_cells(*_cells(y, pred), name)


def bootstrap_ci(y, score, metric, n_boot=2000, seed=11, alpha=0.05):
    """Percentile bootstrap CI of one metric for one model."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    pred = (np.asarray(score) >= 0).astype(int)
    counts = np.bincount(y * 2 + pred, minlength=4).astype(float)
    probs = counts / counts.sum()
    draws = rng.multinomial(int(counts.sum()), probs, size=n_boot)
    tp, fp, tn, fn = draws[:, 3], draws[:, 1], draws[:, 0], draws[:, 2]
    values = np.array([_metric_from_cells(tp[b], fp[b], tn[b], fn[b], metric)
                       for b in range(n_boot)])
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return dict(mean=float(_point(y, pred, metric)), low=float(lo), high=float(hi),
                boot_mean=float(np.mean(values)))


def paired_bootstrap_diff(y, score_new, score_ref, metric, n_boot=2000, seed=11, alpha=0.05):
    """Paired bootstrap CI of ``metric(new) - metric(ref)`` on the same samples."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    pred_new = (np.asarray(score_new) >= 0).astype(int)
    pred_ref = (np.asarray(score_ref) >= 0).astype(int)
    # code = y*4 + pred_new*2 + pred_ref  (8 joint categories)
    counts = np.bincount(y * 4 + pred_new * 2 + pred_ref, minlength=8).astype(float)
    probs = counts / counts.sum()
    draws = rng.multinomial(int(counts.sum()), probs, size=n_boot)
    tp_n, fp_n = draws[:, 6] + draws[:, 7], draws[:, 2] + draws[:, 3]
    tn_n, fn_n = draws[:, 0] + draws[:, 1], draws[:, 4] + draws[:, 5]
    tp_r, fp_r = draws[:, 5] + draws[:, 7], draws[:, 1] + draws[:, 3]
    tn_r, fn_r = draws[:, 0] + draws[:, 2], draws[:, 4] + draws[:, 6]
    diffs = np.array([
        _metric_from_cells(tp_n[b], fp_n[b], tn_n[b], fn_n[b], metric)
        - _metric_from_cells(tp_r[b], fp_r[b], tn_r[b], fn_r[b], metric)
        for b in range(n_boot)
    ])
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    point = _point(y, pred_new, metric) - _point(y, pred_ref, metric)
    return dict(mean=float(point), low=float(lo), high=float(hi),
                boot_mean=float(np.mean(diffs)))



def noninferiority(y, score_new, score_ref, metric, margin, n_boot=2000, seed=11):
    """One-sided non-inferiority check.

    ``new`` is declared non-inferior to ``ref`` when the lower bound of the
    paired bootstrap CI of ``metric(new) - metric(ref)`` lies above ``-margin``.
    Uses a one-sided 95% bound (the matching endpoint of a two-sided 90%
    interval). FPR is lower-is-better; pass the original scores unchanged.
    """
    ci = paired_bootstrap_diff(y, score_new, score_ref, metric, n_boot=n_boot, seed=seed, alpha=0.10)
    if metric == "fpr":
        # FPR increases are harmful: require upper bound < +margin.
        noninferior = ci["high"] < margin
    else:
        noninferior = ci["low"] > -margin
    ci.update(metric=metric, margin=margin, noninferior=bool(noninferior))
    return ci


def bootstrap_ci_mean(values, n_boot=4000, seed=11, alpha=0.05):
    """Percentile bootstrap CI of the mean of ``values`` (non-paired use)."""
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return dict(mean=float("nan"), low=float("nan"), high=float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return dict(mean=float(values.mean()), low=float(lo), high=float(hi))


def wilson_ci(k, n, alpha=0.05):
    """Wilson score interval for a binomial proportion ``k / n``.

    Used for the decision-flip rate, where ``n`` is small (a few hundred) and the
    normal approximation is unusable.
    """
    k = float(k)
    n = float(n)
    if n <= 0:
        return dict(rate=float("nan"), low=float("nan"), high=float("nan"))
    from scipy.stats import norm

    z = float(norm.ppf(1 - alpha / 2))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return dict(rate=float(p), low=float(max(0.0, centre - half)),
                high=float(min(1.0, centre + half)))


def paired_ratio_bootstrap(numerator, denominator, n_boot=4000, seed=11, alpha=0.05):
    """Bootstrap CI for ``mean(numerator) / mean(denominator)``.

    The two arrays are paired (same query index), so the resampling draws the
    same indices for both — this is the quantity behind the paper's
    "``2.06-13.94x`` excess-MAE ratio", which was previously a point estimate
    over a single query set and a single noise seed.

    ``p_gt_one`` is the bootstrap probability that Full's excess error exceeds
    Sparse's, a descriptive bootstrap fraction, NOT a frequentist p-value.
    """
    num = np.asarray(numerator, float)
    den = np.asarray(denominator, float)
    mask = np.isfinite(num) & np.isfinite(den)
    num, den = num[mask], den[mask]
    if num.size == 0:
        return dict(mean=float("nan"), low=float("nan"), high=float("nan"),
                    p_gt_one=float("nan"), n=0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, num.size, size=(n_boot, num.size))
    a = num[idx].mean(axis=1)
    b = den[idx].mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = a / b
    finite = ratios[np.isfinite(ratios)]
    point = float(num.mean() / den.mean()) if den.mean() != 0 else float("inf")
    if finite.size < max(10, 0.5 * n_boot):
        return dict(mean=point, low=float("inf"), high=float("inf"),
                    p_gt_one=float(np.mean(ratios > 1)) if ratios.size else float("nan"),
                    n=int(num.size), undefined=float(1 - finite.size / max(1, ratios.size)))
    lo, hi = np.percentile(finite, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return dict(mean=point, low=float(lo), high=float(hi),
                p_gt_one=float(np.mean(finite > 1)), n=int(num.size),
                undefined=float(1 - finite.size / max(1, ratios.size)))


def mcnemar(y, score_a, score_b):
    """Exact binomial McNemar test on paired correctness (a vs b)."""
    y = np.asarray(y)
    correct_a = (np.asarray(score_a) >= 0) == y
    correct_b = (np.asarray(score_b) >= 0) == y
    b = int(np.sum(correct_a & ~correct_b))   # a right, b wrong
    c = int(np.sum(~correct_a & correct_b))   # a wrong, b right
    n = b + c
    if n == 0:
        p = 1.0
    else:
        from scipy.stats import binomtest
        p = float(binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)
    return dict(a_only_correct=b, b_only_correct=c, discordant=n, p_value=p)


# Compatibility for historical callers; this is not a TOST equivalence test.
tost_noninferiority = noninferiority
