# Build stage
FROM golang:1.25-alpine AS builder

WORKDIR /app

# Install build dependencies
RUN apk add --no-cache git make

# Copy go mod files first for caching
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build binary with optimizations
RUN CGO_ENABLED=0 GOOS=linux go build \
    -ldflags="-s -w -X main.version=$(git describe --tags --always 2>/dev/null || echo docker) -X main.buildDate=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    -o /app/build/nas ./cmd/nas/

# Runtime stage
FROM alpine:3.21

WORKDIR /app

# Install runtime dependencies
RUN apk add --no-cache ca-certificates python3 py3-pip

# Copy binary from builder
COPY --from=builder /app/build/nas /app/nas

# Copy configs and scripts
COPY configs/ /app/configs/
COPY examples/ /app/examples/
COPY scripts/ /app/scripts/

# Create data directory
RUN mkdir -p /app/data

# Set default environment
ENV NAS_CONFIG=/app/configs/default.yaml

ENTRYPOINT ["/app/nas"]
CMD ["search", "--config", "/app/configs/default.yaml"]
