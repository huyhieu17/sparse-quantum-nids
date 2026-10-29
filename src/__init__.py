"""Noise-robustness extension for the ICC 2027 sparse quantum-state IDS paper.

Modules
-------
pipeline   : frozen encoders / kernel / SVC decision rule (verbatim from the
             release notebook so the extension is directly comparable).
data       : local dataset loaders (NSL-KDD, UNSW-NB15, ToN-IoT, RT-IoT),
             group-disjoint splits and published external test files.
noise      : parametric / backend Aer noise models and noisy fidelity-kernel
             evaluation.
stats      : paired bootstrap, one-sided non-inferiority and McNemar tests.
classical  : classical SVM / RandomForest baselines.
experiment : orchestration of the ideal, held-out and noisy experiments.
"""
