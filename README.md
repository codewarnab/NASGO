# NASGO

NASGO is an experimental neural architecture search toolkit written in Go, with an optional Python/PyTorch trainer bridge.

## Quick start

```sh
go test ./...
go run ./cmd/nas info --config configs/default.yaml
go run ./cmd/nas search --config examples/fast.yaml
```

## Capabilities

- Configurable cell search space
- Random, evolutionary, and regularized-evolution search
- Proxy, trainer, and combined evaluation
- SQLite experiment history and checkpoints
- YAML, documented environment overrides, and CLI precedence
- Cross-platform Go builds and a trainer-capable container

See [PROGRESS.md](PROGRESS.md) for current status and limitations, [TESTING.md](TESTING.md) for validation, [CONFIGURATION.md](CONFIGURATION.md) for precedence, and [CHECKPOINTS.md](CHECKPOINTS.md) for resume behavior.

## Container

```sh
docker build -t nasgo .
docker run --rm nasgo search --config /app/configs/default.yaml
```

The image includes pinned CPU PyTorch dependencies. Mount `/app/data` to persist downloaded datasets. GPU use needs a CUDA-compatible derivative and host runtime. Proxy searches do not use PyTorch.

## Maturity

This repository is for experiments and continued development. It is not a claim of production readiness. The status page links the remaining work tracked in issues #1 through #5.
