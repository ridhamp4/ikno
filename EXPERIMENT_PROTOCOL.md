# Corrected experiment protocol

The files under `results/` predate the protocol fixes and must not be used in
a submission. In particular, the stored Maxwell split has overlapping seeds
and is intentionally rejected by the loader.

## 1. Generate an isolated dataset

Never generate into a directory containing trajectories from another run.
The generator now refuses this by default.

Quick end-to-end data:

```bash
python3 dataset_generation/generate_all.py \
  --quick --out_root dataset_generation/data_corrected
```

Paper-scale data:

```bash
python3 dataset_generation/generate_all.py \
  --out_root dataset_generation/data_paper
```

Each PDE directory receives `dataset_manifest.json`. Passing `--overwrite` is
required to replace an existing generated dataset.

## 2. Run a protocol smoke experiment

Use a new output directory. Checkpoint selection uses a deterministic
validation holdout; the test split is evaluated once after selection.

```bash
python3 run_experiments.py \
  --experiment main --pde maxwell \
  --data_dir dataset_generation/data_corrected \
  --output_dir results_corrected_smoke \
  --epochs 2 --seeds 0
```

## 3. Run the main workshop experiment

```bash
python3 run_experiments.py \
  --experiment main --pde all \
  --data_dir dataset_generation/data_paper \
  --output_dir results_paper \
  --epochs 200 --seeds 0,1,2,3,4
```

Every trained run writes:

- `run_metadata.json`: configuration, software versions, seed, and exact files
- `training_log.csv`: training and validation metrics only
- `validation_history.json`: checkpoint-selection history
- `test_metrics.json`: the single final test evaluation
- `best_model.pt` and `final_model.pt`

The combined table reports standard deviation across independent training
seeds for learned models. Persistence and two-frame linear baselines are always
included. `rotation_rel_l2` measures predictive accuracy after unseen SO(3)
rotations; `eq_error` is a normalized operator-level equivariance diagnostic.

## Scientific scope

- Maxwell uses the complete `(E, B)` state.
- Elasticity uses the complete `(u, v)` state.
- Navier--Stokes uses `u`.
- `dt` is read from saved timestamps and conflicting overrides fail.
- The verified production architecture currently supports scalar and vector
  (`l <= 1`) SO(3) representations.
- PDE recovery is disabled by default. The previous learned feature names are
  not sufficient evidence of symbolic differential-operator recovery. Do not
  report PRR until an explicit cross-field operator library is implemented and
  validated against analytic fields.

## Required release checks

```bash
python3 -m pytest -q
python3 test_smoke.py
```

Do not report the old `results/` metrics or enable `strict_integrity: false` in
submission experiments.
