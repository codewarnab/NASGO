# Configuration

Precedence is defaults, YAML, environment, then CLI flags. Supported environment overrides:

- `NAS_SEARCH_STRATEGY`
- `NAS_MAX_EVALUATIONS`
- `NAS_NUM_WORKERS`
- `NAS_EVALUATOR_TYPE`
- `NAS_STORAGE_PATH`
- `NAS_USE_GPU`

For example: `NAS_MAX_EVALUATIONS=50 NAS_NUM_WORKERS=4 nas search --config configs/default.yaml`. CLI flags such as `--evaluations` and `--workers` still win.
