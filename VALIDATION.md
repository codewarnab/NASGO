# End-to-end validation

This page records a bounded manual validation of commit
`2ec586c0bc402799dd8b068cab63c9c0097c4caf` on 17 September 2026. These runs
used the CLI outside the Go test fixtures. The numbers are a reproducibility
snapshot, not a general benchmark.

## Proxy baseline

The binary was built with:

```sh
go build -o /tmp/nasgo-e2e/nas ./cmd/nas
```

Each strategy used one worker, 500 evaluations, the same search space and the
same three seeds. The search space had three nodes, two inputs per node, two
edges per node, and the operations `identity`, `conv_3x3`, `sep_conv_3x3`,
`max_pool_3x3`, and `zero`. The evaluator was `proxy`; history and checkpoints
were stored in SQLite.

A run had this shape, repeated for each strategy and seed:

```sh
/tmp/nasgo-e2e/nas search \
  --config /tmp/nasgo-e2e/base.yaml \
  --strategy random \
  --seed 42 \
  --json
```

`base.yaml` set `max_evaluations: 500`, `num_workers: 1`,
`checkpoint_interval: 25`, and the search space described above.

| Strategy | Seed | Best proxy fitness | Wall time (s) | Evaluations/s |
| --- | ---: | ---: | ---: | ---: |
| random | 42 | 0.832417 | 2.298 | 217.6 |
| random | 1337 | 0.798086 | 2.150 | 232.6 |
| random | 2026 | 0.840195 | 2.912 | 171.7 |
| evolutionary | 42 | 0.847658 | 2.278 | 219.4 |
| evolutionary | 1337 | 0.847658 | 2.450 | 204.1 |
| evolutionary | 2026 | 0.847658 | 2.679 | 186.7 |
| regularized | 42 | 0.847658 | 2.284 | 218.9 |
| regularized | 1337 | 0.847658 | 2.558 | 195.5 |
| regularized | 2026 | 0.847658 | 2.483 | 201.4 |

All nine runs completed. Direct SQLite queries found nine completed experiments
and 4,500 architecture rows. Fitness in this table is the score from NASGO's
proxy evaluator. It is not trained-model accuracy.

The equal best fitness for the evolutionary strategies across these seeds does
not establish that either strategy is better than random. The budget, search
space, evaluator and seed count are deliberately small.

## Checkpoint and resume

Each strategy also received `SIGINT` during a seed-42 run and was resumed by
experiment ID:

```sh
/tmp/nasgo-e2e/nas search --config /tmp/nasgo-e2e/resume-random.yaml &
pid=$!
sleep 0.55
kill -INT "$pid"
wait "$pid" || true

experiment_id=$(sqlite3 /tmp/nasgo-e2e/resume-random.db \
  'select id from experiments limit 1;')
/tmp/nasgo-e2e/nas search \
  --config /tmp/nasgo-e2e/resume-random.yaml \
  --resume "$experiment_id"
```

| Strategy | Evaluation at interruption | Interrupted status | Evaluations after resume | History rows | Topology sequence vs. uninterrupted run |
| --- | ---: | --- | ---: | ---: | --- |
| random | 165 | cancelled | 500 | 500 | exact match |
| evolutionary | 130 | cancelled | 500 | 500 | exact match |
| regularized | 149 | cancelled | 500 | 500 | exact match |

The comparison parsed every stored `arch_json`, removed the generated
architecture ID and metadata, and compared the ordered topology objects. IDs
and timestamps are expected to differ. This validates exact continuation of
the candidate topology sequence for these single-worker runs.

The run also exposed a checkpoint-frequency defect: with
`checkpoint_interval: 25`, checkpoint rows were written at evaluations
25, 26, 27, and every evaluation thereafter instead of only at 25, 50, 75,
and so on. Each resumed database contained 477 checkpoint rows. Track and fix
that behavior separately from these validation results.

## Real PyTorch trainer bridge

The smallest meaningful trainer path was run with the pinned CPU dependencies
from `requirements-trainer.txt`: PyTorch 2.6.0+cpu and torchvision
0.21.0+cpu. The config used torchvision `FakeData`, one candidate, one epoch,
a batch size of four, two channels, one layer, and CPU execution:

```sh
/tmp/nasgo-e2e/nas search \
  --config /tmp/nasgo-e2e/trainer-fake.yaml \
  --json
```

The bridge constructed and trained a real model through `scripts/train.py`.
The run completed in 2.539 seconds wall time, with 2.416 seconds reported as
search time. SQLite recorded one architecture with 162 trainable parameters,
a fitness/validation accuracy of 0.25, and a checkpoint.

This proves the Go-to-Python protocol, model construction, training loop,
result parsing and persistence work together for this bounded CPU case.
`FakeData` accuracy does not say anything about architecture quality or CIFAR-10
performance.

## CIFAR-10 ranking: not completed

The planned check selected six proxy candidates spanning scores 0.3200 to
0.8324. Each candidate was to train for two epochs on the same deterministic
2,048-image training subset and 1,024-image test subset on CPU, followed by a
Spearman rank correlation between proxy fitness and validation accuracy.

Training never started. Three bounded attempts to retrieve the upstream
Toronto CIFAR-10 archive, approximately 170 MB, did not complete in this
environment. Therefore there are no trained CIFAR-10 accuracies and no ranking
correlation to report. Do not cite this run as CIFAR-10 validation.

A future run should use a reliable pre-populated dataset cache, keep the same
candidate set and deterministic subsets, and publish the per-candidate proxy
score, validation accuracy, failures, runtime and Spearman correlation.
