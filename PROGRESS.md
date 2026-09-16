# Project status

NASGO is an experimental neural architecture search toolkit. It is suitable for local experiments, test fixtures, and continued development. It is not presented as production-ready.

## Available today

- CLI search and search-space inspection (`go run ./cmd/nas help`)
- Configurable cell search space and eight operation types
- Random, evolutionary, and regularized-evolution strategies
- Proxy, Python trainer, and combined evaluators
- SQLite experiment history and versioned checkpoints
- Bounded random-search workers through `num_workers`
- Pinned trainer-capable container dependencies
- Go unit, integration, subprocess-protocol, race, lint, and cross-build CI

Run the fast validation suite with `go test ./...` or `go test -race ./...`. See [TESTING.md](TESTING.md).

## Known limits and follow-up

- The trainer image defaults to CPU dependencies; GPU training needs a CUDA-specific image and runtime. See [issue #2](https://github.com/codewarnab/NASGO/issues/2).
- Environment configuration supports an explicit, documented subset of `NAS_` variables. See [issue #3](https://github.com/codewarnab/NASGO/issues/3) and [CONFIGURATION.md](CONFIGURATION.md).
- Parallel workers currently apply to independent random-search batches; evolutionary strategies remain sequential. See [issue #4](https://github.com/codewarnab/NASGO/issues/4).
- Resume supports random and evolutionary histories, but exact RNG continuation and regularized-evolution state are not yet preserved. See [issue #5](https://github.com/codewarnab/NASGO/issues/5) and [CHECKPOINTS.md](CHECKPOINTS.md).
- Integration coverage focuses on fast fixtures rather than CIFAR or GPU training. See [issue #1](https://github.com/codewarnab/NASGO/issues/1).

The older implementation walkthrough in [NAS_GO_Implementation.md](NAS_GO_Implementation.md) explains architecture and original design intent. This page is the source of truth for current capability status.

## Validation note

The Go suite, race detector, vet, and Python syntax compile are validated in CI-compatible
commands. The trainer image includes the pinned PyTorch dependencies and training script,
but a real image build/smoke test still requires a host with Docker or another OCI runtime.
