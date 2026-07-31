# Experiments

`experiments.csv` is the experimental log: one row per run, keyed by a unique `uid`.

Columns: `uid, timestamp, hypothesis, model, dataset, init, labeled_fraction, seed, hyperparameters, dice, miou, detection_f1, gap, tag`. `timestamp` is the run's ISO-8601 datetime from its log (or file mtime for the notebook-based runs).

Each row's full configuration is `configs/config_<uid>.yaml`, linked by `uid`. The config files are generated from the CSV:

```bash
uv run python scripts/build_experiment_configs.py
```

`init` distinguishes the arms compared in the report: `ssl_pretrained` (same-species SSL), `crossspecies_ssl` (H2 source), `from_scratch`, `supervised`, `untrained`, and `zero_shot` (Cellpose). Per-clip `dice`/`miou` are the held-out training-protocol scores; `detection_f1` and the field-level `dice` are the native-resolution instance evaluation; `gap` is the H3 within-minus-between similarity.
