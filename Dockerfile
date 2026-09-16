# Build the static Go CLI separately from the Python trainer runtime.
FROM golang:1.25-bookworm AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build \
    -ldflags="-s -w -X main.version=$(git describe --tags --always 2>/dev/null || echo docker) -X main.buildDate=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    -o /app/build/nas ./cmd/nas/

# PyTorch's supported Linux wheels target glibc, not Alpine/musl.
FROM python:3.12-slim-bookworm
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-trainer.txt /app/requirements-trainer.txt
RUN python -m pip install --no-cache-dir -r /app/requirements-trainer.txt
COPY --from=builder /app/build/nas /app/nas
COPY configs/ /app/configs/
COPY examples/ /app/examples/
COPY scripts/ /app/scripts/
RUN mkdir -p /app/data
ENV NAS_CONFIG=/app/configs/default.yaml
ENTRYPOINT ["/app/nas"]
CMD ["search", "--config", "/app/configs/default.yaml"]
