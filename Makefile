# NAS - Neural Architecture Search in Go
# Build automation

# Variables
BINARY_NAME := nas
BUILD_DIR := build
VERSION := $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
COMMIT := $(shell git rev-parse --short HEAD 2>/dev/null || echo "none")
BUILD_DATE := $(shell date -u +"%Y-%m-%dT%H:%M:%SZ")
LDFLAGS := -ldflags "-s -w -X main.version=$(VERSION) -X main.commit=$(COMMIT) -X main.buildDate=$(BUILD_DATE)"

# Go settings
GO := go
GOTEST := $(GO) test
GOVET := $(GO) vet
GOBUILD := $(GO) build

.PHONY: all build clean test test-verbose test-cover lint vet fmt tidy run help

## help: Show this help message
help:
	@echo "NAS - Neural Architecture Search in Go"
	@echo ""
	@echo "Usage:"
	@echo "  make <target>"
	@echo ""
	@echo "Targets:"
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/## /  /'

## all: Build, test, and lint
all: tidy vet lint test build

## build: Build the binary
build:
	@echo "Building $(BINARY_NAME)..."
	@mkdir -p $(BUILD_DIR)
	$(GOBUILD) $(LDFLAGS) -o $(BUILD_DIR)/$(BINARY_NAME) ./cmd/nas/

## clean: Remove build artifacts
clean:
	@echo "Cleaning..."
	@rm -rf $(BUILD_DIR)
	@rm -f experiments*.db
	@rm -f *.log

## test: Run all tests
test:
	@echo "Running tests..."
	$(GOTEST) ./... -count=1

## test-verbose: Run tests with verbose output
test-verbose:
	@echo "Running tests (verbose)..."
	$(GOTEST) -v ./... -count=1

## test-cover: Run tests with coverage
test-cover:
	@echo "Running tests with coverage..."
	@mkdir -p $(BUILD_DIR)
	$(GOTEST) -coverprofile=$(BUILD_DIR)/coverage.out ./...
	$(GO) tool cover -html=$(BUILD_DIR)/coverage.out -o $(BUILD_DIR)/coverage.html
	@echo "Coverage report: $(BUILD_DIR)/coverage.html"

## lint: Run golangci-lint
lint:
	@echo "Running linter..."
	@golangci-lint run ./... || echo "golangci-lint not installed, skipping"

## vet: Run go vet
vet:
	@echo "Running go vet..."
	$(GOVET) ./...

## fmt: Format code
fmt:
	@echo "Formatting code..."
	$(GO) fmt ./...

## tidy: Tidy and verify go modules
tidy:
	@echo "Tidying modules..."
	$(GO) mod tidy

## run: Build and run with default config
run: build
	./$(BUILD_DIR)/$(BINARY_NAME) search --config configs/default.yaml

## run-fast: Build and run with fast config
run-fast: build
	./$(BUILD_DIR)/$(BINARY_NAME) search --config examples/fast.yaml

## run-info: Show search space info
run-info: build
	./$(BUILD_DIR)/$(BINARY_NAME) info --config configs/default.yaml

## docker-build: Build Docker image
docker-build:
	docker build -t nas-go:$(VERSION) .

## docker-run: Run in Docker
docker-run:
	docker run --rm nas-go:$(VERSION) search --config /app/configs/default.yaml

## install: Install binary to GOPATH/bin
install:
	@echo "Installing $(BINARY_NAME)..."
	$(GO) install $(LDFLAGS) ./cmd/nas/

## version: Print version info
version: build
	./$(BUILD_DIR)/$(BINARY_NAME) version
