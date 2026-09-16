# End-to-end validation

Bounded manual validation of commit `2ec586c0bc402799dd8b068cab63c9c0097c4caf` on 17 September 2026. Runs used the CLI outside Go test fixtures. These are reproducibility results, not a general benchmark.

## Proxy baseline

Built with `go build -o /tmp/nasgo-e2e/nas ./cmd/nas`. Each strategy used one worker, 500 evaluations, seeds 42/1337/2026, and the same three-node search space and proxy evaluator. A run was:

```sh
/tmp/nasgo-e2e/nas search --config /tmp/nasgo-e2e/base.yaml \
  --strategy random --seed 42 --json
```

| Strategy | Seed | Best proxy fitness | Wall s | Eval/s |
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

All nine runs completed; direct SQLite queries found 4,500 architecture rows. Proxy fitness is not trained-model accuracy. This small budget and seed count do not establish strategy superiority.

## Checkpoint and resume

Each seed-42 strategy was interrupted with `SIGINT`, then resumed by experiment ID:

```sh
/tmp/nasgo-e2e/nas search --config resume-random.yaml &
pid=$!; sleep 0.55; kill -INT "$pid"; wait "$pid" || true
/tmp/nasgo-e2e/nas search --config resume-random.yaml --resume EXPERIMENT_ID
```

| Strategy | Interrupted at | Status | Final/history | Topology sequence |
| --- | ---: | --- | ---: | --- |
| random | 165 | cancelled | 500/500 | exact match |
| evolutionary | 130 | cancelled | 500/500 | exact match |
| regularized | 149 | cancelled | 500/500 | exact match |

The comparison removed generated IDs and metadata from stored `arch_json`, then compared ordered topology objects. IDs and timestamps differ as expected.

A separate defect was found: `checkpoint_interval: 25` wrote at 25, 26, 27...500, not 25, 50, 75...500. Each resumed DB had 477 checkpoint rows.

## Real trainer bridge

PyTorch 2.6.0+cpu and torchvision 0.21.0+cpu ran one `FakeData` candidate for one epoch, batch size 4, channels 2, layer 1:

```sh
/tmp/nasgo-e2e/nas search --config trainer-fake.yaml --json
```

The real model completed in 2.539s wall/2.416s search time. SQLite recorded 162 trainable parameters, fitness/validation accuracy 0.25, and a checkpoint. This validates the Go-Python protocol, construction, training, parsing and persistence for this CPU case. `FakeData` accuracy says nothing about CIFAR-10 quality.

## CIFAR-10 ranking: not completed

The plan was six candidates spanning proxy scores 0.3200-0.8324, two epochs each on deterministic 2,048-train/1,024-test subsets, then Spearman correlation. Training never started: three bounded attempts to retrieve the upstream Toronto CIFAR-10 archive (about 170 MB) did not complete. There are no trained CIFAR-10 accuracies or ranking correlation. Do not cite this run as CIFAR-10 validation.
