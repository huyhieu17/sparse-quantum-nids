# V11: noise-aware development, frozen before test scoring

This is a new internal experiment, not a rerun of the v10 test cohort and not
external validation. Historical work has already used these source datasets;
the new holdout is untouched by **v11 selection**, not claimed never seen in
any historical experiment. No promise of 95–97% accuracy is made.

## Fixed design

- Four local datasets; deterministic fingerprint-group-disjoint train,
  validation and test. Remove conflicting-label fingerprint groups. Split seed
  20261026. Fit all preprocessing using 2,000 training rows only; validation
  500 and test 1,000, stratified with natural class proportions.
- Fixed split across four independent training/measurement seeds 11,23,37,53.
  Their SD is conditional on this split, not split-to-split uncertainty.
- Same vendored Nairobi/Lagos/Perth snapshots, thermal relaxation, gate errors
  and asymmetric readout as v10. No idle scheduling or crosstalk. Simulation,
  not QPU execution. No test-based backend/layout/configuration selection.
- Shared PCA32 frontend for all encoders; additionally RBF-SVC on PCA32 and
  ExtraTrees on the same processed raw inputs. No supervised MLP advantage.
- Candidate state preparations: original dense amplitude, adaptive16 and
  adaptive8 (loss .3); shallow angle encoding (3q/2 layers, 5q/2 layers,
  5q/4 layers) and 5q/4-layer product-state control. Amplitude anchor fixed 1.
- New projected kernel: single- and two-qubit Z and X correlators, measured
  in two settings. RBF-SVC on standardized measured features is PSD by
  construction. This **changes the fidelity kernel**, and is reported as a
  separate method, not mislabeled as a refit of the old fidelity SVC.
- 1,024 TOTAL shots/sample (512 per basis), supplemental 4,096 total.
  Joint multinomial draws preserve correlations among observables. Calibration
  mitigation uses independent 4,096-shot |0>,|1> readout calibration per qubit;
  it corrects readout, not preparation or gate errors. Include its cost.
- SVC grid C={.1,1,10,100}, gamma={.03,.1,1}/feature_count and
  class_weight={None,balanced}. Threshold selected on validation only.
  Ideal-trained/noisy-inference and noisy-trained/noisy-inference are separate.
- Choose shared encoder/hyperparameters by mean validation macro-F1 across
  all three noisy backends (per seed), not best backend on test. Include all
  candidate results and an unchanged zero-threshold ablation.
- Freeze selection JSON and its SHA256 before reading test labels for scoring.
  Report mean/SD, paired bootstrap conditional on fitted models, confusion
  matrices, recall/FPR, CX, depth, circuits and shots. Preserve v10 unchanged.

## Separate fidelity-kernel experiment

- Dense and Adaptive16 retain fidelity overlap measurements, with Nyström
  rank selected from {8,16}; this is not the full 2,000x2,000 Gram matrix SVC.
- Sixteen train-only landmarks (8/class), interleaved so rank 8 has 4/class.
  No validation or test points become landmarks. Same 2,000 training labels.
- Overlap blocks are separately compiled; terminal routing permutations are
  restored with physical-edge SWAPs. No optimization across the block boundary.
  This differs from v10 compose-before-route compilation and is disclosed.
- The exact adjoint noisy reference channel is contracted with each noisy
  query density matrix, including inverse-preparation errors and asymmetric
  readout. Equality to full Aer overlap simulation is tested at 1e-10.
- Each pair uses 1,024 shots: 8,192 or 16,384 total inference shots/sample,
  versus 1,024 total for projected kernels. Count these costs explicitly.
- Symmetrize the train-only landmark block, clip negative eigenvalues and add
  relative ridge {0.01,0.1}; linear SVC C={0.1,1,10}, weights={None,balanced}.
  Rank/ridge/C/weight/encoder selected by mean validation F1 over three devices;
  thresholds selected on validation per device. Four independent sampled fits.
- The initial compose-before-route runtime pilot was stopped before selection,
  archived in `fidelity_compose_route_pilot`, and excluded from all final tables.
  The block compiler was chosen for simulation reuse, without test outcomes.

Projected-kernel gains cannot establish that the original Adaptive16 fidelity
classifier improved. The Nyström experiment isolates a second approach, and
its approximation/compiler changes must remain explicit in the paper.

## Additional controls and audits

- Z7–Pegasos is rerun on the v11 cohort, with both ideal/noisy training,
  validation-selected angle scale {1, pi}, and zero/tuned thresholds. Keep
  C=1000, 100 steps and offset=1. First seven shared PCA components, not the
  full 32-dimensional amplitude input. Product-state channel factorization is
  verified against full seven-qubit Aer overlap simulation. The reference is
  a pipeline comparator, not an identical classifier/feature-budget ablation.
- Ideal-training/noisy-threshold control uses the selected projected
  hyperparameters, isolating benefits beyond threshold calibration.
- `gate_duration_profile.csv` profiles the first eight test inputs per
  configuration/device in both bases. Durations are gate-only ASAP estimates
  from calibration, excluding readout, reset, pulse alignment and queues;
  they are not measured QPU latency or a scheduled-idle noise simulation.
- Test sensitivity after preprocessing removes exactly identical transformed
  inputs found in train/validation, without refitting or changing selection.
  Headline cohort remains the originally frozen 1,000 rows.
