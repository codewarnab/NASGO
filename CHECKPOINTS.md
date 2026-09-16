# Checkpoints and resume

SQLite searches save a versioned checkpoint every `storage.checkpoint_interval` completed evaluations. Resume random or evolutionary search with:

```sh
nas search --config configs/default.yaml --strategy evolutionary --resume EXPERIMENT_ID
```

The evaluation budget is the total across both runs, so completed candidates are not re-evaluated. Resume rejects missing, corrupt, version-incompatible, or strategy-incompatible checkpoints. Current snapshots preserve evaluated history and best fitness; exact RNG continuation is not guaranteed yet.
