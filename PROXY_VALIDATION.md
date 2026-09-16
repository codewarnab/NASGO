# Proxy-vs-real validation (CIFAR-10, bounded CPU)

Date: 17 September 2026. Base commit: `1b0313b` (main after PRs #13/#15).
Question: does NASGO's zero-cost proxy fitness predict real trained validation
accuracy, and does search find models that beat a simple hand-designed baseline?

**Short answer: no predictive signal was detected at this budget.** Spearman
rank correlation between proxy fitness and trained validation accuracy across
14 architectures is -0.024 (bootstrap 95% CI [-0.60, 0.53]). A plain 3-block
CNN baseline trained under the identical budget outscored every one of the 14
sampled architectures. Treat proxy fitness as a cheap heuristic for exploring
the space, not as a ranking of trained quality. See "Limitations" before
generalizing.

## Protocol

- Candidate pools: three proxy searches on current main (random, evolutionary,
  regularized; seed 42; 500 evaluations each; 3-node search space; config in
  `experiments/proxy_validation/` reproduction steps below).
- Sample: 12 architectures stratified evenly across the random pool's fitness
  ranks (proxy 0.385-0.892) plus the best architecture from the evolutionary
  (0.648) and regularized (0.913) pools. 14 total, listed in
  `experiments/proxy_validation/selected_architectures.json`.
- Training: deterministic 4,096-image train subset of CIFAR-10 and 1,024-image
  validation subset of the official test set (subset seeds fixed); batch 64;
  3 epochs; SGD lr 0.025, momentum 0.9, weight decay 3e-4, cosine schedule;
  standard crop/flip augmentation; channels 16, layers 8 (the repo defaults).
  Two training seeds (7, 8) per architecture. The hand-designed baseline
  (`BaselineCNN`, 3 conv blocks, 66,410 params) was trained under the
  identical budget.
- Model construction reuses `scripts/train.py` (`NASNetwork`), so measured
  architectures are exactly what NASGO trains.
- Hardware: 2-core x86-64 CPU sandbox, 2 GB RAM, PyTorch 2.6.0+cpu. Total
  measured training time: 1,672 s across all 30 runs (~56 s average).
- Raw per-run records: `experiments/proxy_validation/results.jsonl`.

## Results

| arch | proxy fitness | best val acc (2 seeds) | mean | params |
| --- | ---: | --- | ---: | ---: |
| arch-08 | 0.570 | 0.414 / 0.394 | 0.404 | 33,914 |
| arch-03 | 0.513 | 0.402 / 0.387 | 0.395 | 157,690 |
| arch-01 | 0.477 | 0.399 / 0.369 | 0.384 | 77,946 |
| arch-11 (random best) | 0.892 | 0.394 / 0.358 | 0.376 | 27,834 |
| arch-02 | 0.493 | 0.363 / 0.377 | 0.370 | 50,874 |
| arch-09 | 0.583 | 0.367 / 0.370 | 0.369 | 76,218 |
| arch-06 | 0.546 | 0.336 / 0.389 | 0.362 | 109,818 |
| arch-07 | 0.561 | 0.339 / 0.357 | 0.348 | 45,306 |
| arch-12 (evolutionary best) | 0.648 | 0.357 / 0.331 | 0.344 | 60,794 |
| arch-13 (regularized best) | 0.913 | 0.328 / 0.359 | 0.344 | 26,810 |
| arch-10 | 0.607 | 0.351 / 0.335 | 0.343 | 42,170 |
| arch-00 (random worst) | 0.385 | 0.336 / 0.334 | 0.335 | 29,114 |
| arch-04 | 0.527 | 0.356 / 0.304 | 0.330 | 106,810 |
| arch-05 | 0.533 | 0.308 / 0.302 | 0.305 | 96,442 |
| **baseline CNN** | n/a | 0.410 / 0.413 | **0.412** | 66,410 |

Statistics:

- Spearman(proxy fitness, mean val acc) = **-0.024**, bootstrap 95% CI
  [-0.602, 0.527] (10,000 resamples over architectures).
- Spearman(parameter count, mean val acc) = 0.029 - size alone does not
  explain the ranking either.
- The two highest-proxy architectures (0.892, 0.913) rank 4th and 10th of 14.
- 0 of 14 architectures beat the baseline mean; the best architecture
  (arch-08) sits within one seed's noise of the baseline.

Interpretation, bounded on purpose: at this budget the proxy shows no
ability to rank architectures by trained accuracy, and search guided by it
did not find anything better than a trivial baseline. The proxy rewards
~100 "relative parameters", operation diversity, ~70% density and ~20% skip
ratio; none of these tracked 3-epoch accuracy here.

## Defect found and fixed during validation

`scripts/train.py` crashed for most multi-layer architectures before this
branch: after a reduction cell, the two cell inputs have different spatial
sizes (32x32 and 16x16), and `Cell.preprocess` never aligned them, so any
normal cell summing edges from both inputs failed with a tensor size
mismatch. The earlier FakeData bridge test used `layers: 1` and never hit
this. The fix downsizes the larger input with a `FactorizedReduce` before
preprocessing (DARTS convention). All 14 architectures train successfully with
the fix. Verified against the pre-fix script: 12 of the 14 sampled
architectures crash on a single forward pass at `layers: 8` (any cell that
mixes both inputs after a reduction cell).

## Export tooling

`experiments/export_results.py` exports any NASGO SQLite experiment database
to `architectures.csv`, `experiments.json`, and a Markdown summary:

```sh
python3 experiments/export_results.py --db experiments.db --out export/
```

This covers the "experiment output/export" gap at a basic level: results
leave the SQLite file in diff-able, shareable formats. It is not a
dashboard; richer reporting (plots, run comparison UI) remains open.

## Limitations

- n=14 architectures, 2 seeds: the confidence interval is wide. This run
  rules out a *strong* positive correlation at this budget; it does not
  prove the proxy is useless at every budget.
- 3 epochs on 4,096 images is a small budget. Rankings can change with
  longer training; short-budget correlation is nonetheless the regime the
  proxy is meant to serve (cheap pre-screening).
- CPU-only, single machine. No GPU measurements.
- CIFAR-10 came from the fast.ai S3 mirror (folder layout) because the
  upstream Toronto archive was unreachable from the test environment at
  usable speed; archive contents were verified (10 classes, 5,000 train /
  1,000 test images per class).
- The trainer-bridge defect above means `VALIDATION.md`'s "real trainer
  bridge works" statement only held for single-layer configs before this
  branch.

## Reproduce

```sh
pip install -r requirements-trainer.txt pillow
# candidate pools (proxy search, seconds each)
nas search --config <pool-config> --json   # see experiments/proxy_validation/
# select architectures
python3 experiments/validate_proxy.py select \
  --random-db runs/random.db --evolutionary-db runs/evolutionary.db \
  --regularized-db runs/regularized.db --out selected.json
# train (resumable; safe to interrupt between epochs)
python3 experiments/validate_proxy.py run --repo . --archs selected.json \
  --data /path/to/cifar10-folder --results results.jsonl --budget-seconds 3600
# statistics
python3 experiments/validate_proxy.py analyze --results results.jsonl
```
