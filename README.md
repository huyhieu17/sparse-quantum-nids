# Noise-Aware Sparse Quantum Intrusion Detection

A research implementation of noise-aware projected quantum kernels for binary intrusion detection on **ToN-IoT, RT-IoT, NSL-KDD, and UNSW-NB15**. The experiments compare dense amplitude encoding, adaptive sparse encoding, and matched classical and quantum baselines. A separate benchmark measures CPU simulator training and prediction time.

Experiments use calibrated noisy simulation, not QPU execution. The Z7/Pegasos baseline is a reimplementation evaluated on the matched cohort, not the original authors' code or an exact reproduction of their published scores.

## Repository layout

```text
Intrusion Detection/
├── src/                         # Shared preprocessing, encoders, circuits and noise
├── experiments/                 # Main training, selection, evaluation and controls
├── baselines/                   # Shared Pegasos implementation
├── benchmarks/                  # Controlled CPU simulator timing
├── analysis/quantum_cost/       # Resource accounting and summary tables
├── figures/quantum_circuits/    # Qiskit drawing code and combined circuit PDF
├── device_snapshots/            # Frozen device configuration and calibration
├── dataset/                     # Local input datasets and their SHA-256 manifest
├── results/
│   ├── detection/               # Frozen metrics, selections and protocol records
│   └── timing/                  # Timing measurements and environment records
├── tests/                       # Numerical noise and Pegasos tests
├── requirements.txt
├── verify_package.py
└── PACKAGE_MANIFEST.json        # Checksums for the distributable repository files
```

## Installation

Use Python 3.11 and the pinned dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run all commands from the repository root. The scripts use relative output paths.

## Input datasets

Obtain the following four input files separately and place them at these paths. Neither the local repository nor the ZIP includes raw dataset contents. Exact hashes and sizes are recorded in `dataset/INPUT_MANIFEST.json`.

| Dataset | Required file |
|---|---|
| ToN-IoT | `dataset/ton_iot/train_test_network.csv` |
| RT-IoT | `dataset/rt_iot/RT_IOT2022.csv` |
| NSL-KDD | `dataset/nsl_kdd/KDDTrain+.txt` |
| UNSW-NB15 | `dataset/unsw/UNSW_NB15_training-set.csv` |

Only the input files used by the main internal-holdout experiment are listed. Published external test files are not required by this pipeline.

## Recorded results

Redundant derived summary tables and packaging QA notes are omitted. The per-run metrics, main tables and original experiment validation records are retained. Reporting scripts may regenerate additional summaries when run.

- `results/detection/headline.csv`: validation-selected detection results.
- `results/detection/metrics.csv`: candidate and control measurements.
- `results/detection/frozen_selection.json`: frozen model choices.
- `results/detection/fidelity_metrics.csv` and `z7_metrics.csv`: quantum controls.
- `analysis/quantum_cost/summary.csv`: dense/sparse circuit resource comparisons.
- `results/timing/dataset_timing.csv`: aggregated training and prediction times.

The main experiment uses 2,000 training, 500 validation and 1,000 test rows per dataset, with four measurement/training seeds on one fixed split. The timing benchmark uses a separate 200-training/20-prediction workload. Its times must not be presented as runtime measurements for the full accuracy experiment.

See [the experiment protocol](experiments/PROTOCOL.md) and [the timing protocol](benchmarks/PROTOCOL.md) for split rules, shots, noise assumptions and model selection.

## Verification

```bash
python verify_package.py
python -W ignore -m unittest discover -s tests -p 'test_*.py'
```

The checksum checker verifies the distributed code and result tables. Dataset contents can be checked independently against `dataset/INPUT_MANIFEST.json`; they are not required for the numerical unit tests. Historical validation JSON files document the original completed experiments, not a fresh verification of omitted models or predictions.

## Reproduce the experiments

Use a working copy. Move the recorded result directories outside the repository before running so that old frozen selections are not mixed with new artifacts. Choose an empty destination for the backup:

```bash
mkdir ../intrusion-recorded-results
mv results/detection results/timing ../intrusion-recorded-results/
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python experiments/run.py
python experiments/fidelity.py --workers 20
python experiments/z7.py
python experiments/calibration_control.py
python experiments/cost_profile.py
python experiments/provenance.py
python experiments/finalize.py
python analysis/quantum_cost/compare.py
python figures/quantum_circuits/draw_circuits.py
python benchmarks/run.py
python benchmarks/report.py
```

The main runner uses up to 20 simulation workers and 12 training workers. Adjust worker counts in the runner for smaller machines. Run timing sequentially without competing workloads. Full model/prediction validation requires the regenerated artifacts, and may take substantially longer than the unit tests. The scripts support resuming interrupted runs through their existing artifact checks.

## GitHub use

`.gitignore` excludes local datasets, environments, model files, arrays and simulation caches. Review the staged files before committing. Raw dataset copies have been removed from this repository. The lightweight repository ZIP excludes raw datasets as well. No remote repository has been created or pushed automatically.
