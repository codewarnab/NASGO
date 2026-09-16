# Configuration

Precedence is defaults, YAML, environment, then CLI flags. Supported environment overrides:

- `NAS_CONFIG` - YAML path used when `--config` is not supplied
- `NAS_SEARCH_STRATEGY`
- `NAS_MAX_EVALUATIONS`
- `NAS_NUM_WORKERS`
- `NAS_EVALUATOR_TYPE`
- `NAS_STORAGE_PATH`
- `NAS_USE_GPU`

For example:

```sh
NAS_CONFIG=configs/default.yaml NAS_MAX_EVALUATIONS=50 NAS_NUM_WORKERS=4 nas search
```

CLI flags such as `--config`, `--evaluations`, and `--workers` still win. Empty values and malformed integer/boolean values fail clearly; `NAS_NUM_WORKERS` must be at least 1.

The container sets `NAS_CONFIG=/app/configs/default.yaml`, so replacing its default command does not silently discard the default configuration. All other `NAS_*` overrides still apply after that YAML file is loaded.
